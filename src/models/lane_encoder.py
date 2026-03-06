"""Lane encoder for contrastive representation learning.

Fuses annotation geometry, trajectory behavior, and aggregate statistics
into a single lane embedding, then projects to a contrastive space.

Architecture:
    annotation waypoints  ->  PolylineEncoder  --+
                                                  +-> fusion MLP -> embedding -> projection head
    assigned trajectories ->  PolylineEncoder  --+
                                                  |
    traj_stats            ->  stats MLP        --+

Optional cross-lane attention (gated by use_cross_lane_attention):
    Per-lane embeddings packed by group_id
        -> MultiheadSelfAttention with pairwise relative feature bias
        -> Unpacked back to flat batch

Roles (lateral_rank, edge flags, group_size) are concatenated with traj_stats
as encoder input (stats_dim=9). The regression heads also predict roles as
targets, providing direct supervision signal for learning.
"""

import torch
import torch.nn as nn

from src.models.cross_lane_attention import CrossLaneAttention
from src.models.polyline_encoder import PolylineEncoder


# ---------------------------------------------------------------------------
# Group packing helpers (module-level for reuse)
# ---------------------------------------------------------------------------

def _pack_groups(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    traj_stats: torch.Tensor,
) -> tuple:
    """Pack flat batch into (G, max_L, D) grouped tensors.

    Args:
        embeddings: (B, D) per-lane embeddings.
        group_ids: (B,) integer group id per lane.
        traj_stats: (B, S) trajectory statistics per lane.

    Returns:
        (grouped_emb, grouped_stats, group_mask, unique_gids, lane_indices_per_group)
        - grouped_emb: (G, max_L, D)
        - grouped_stats: (G, max_L, S)
        - group_mask: (G, max_L) boolean
        - unique_gids: (G,) unique group ids
        - lane_indices_per_group: list of lists, original batch indices per group
    """
    unique_gids = group_ids.unique()
    G = len(unique_gids)
    D = embeddings.shape[1]
    S = traj_stats.shape[1]
    device = embeddings.device

    # Find max lanes per group
    lane_indices_per_group = []
    for gid in unique_gids:
        indices = (group_ids == gid).nonzero(as_tuple=True)[0].tolist()
        lane_indices_per_group.append(indices)
    max_L = max(len(idx) for idx in lane_indices_per_group)

    grouped_emb = torch.zeros(G, max_L, D, device=device)
    grouped_stats = torch.zeros(G, max_L, S, device=device)
    group_mask = torch.zeros(G, max_L, dtype=torch.bool, device=device)

    for g, indices in enumerate(lane_indices_per_group):
        n = len(indices)
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        grouped_emb[g, :n] = embeddings[idx_tensor]
        grouped_stats[g, :n] = traj_stats[idx_tensor]
        group_mask[g, :n] = True

    return grouped_emb, grouped_stats, group_mask, unique_gids, lane_indices_per_group


def _unpack_groups(
    grouped_emb: torch.Tensor,
    lane_indices_per_group: list,
    B: int,
) -> torch.Tensor:
    """Unpack (G, max_L, D) back to (B, D).

    Args:
        grouped_emb: (G, max_L, D) attended embeddings.
        lane_indices_per_group: list of lists, original batch indices per group.
        B: original batch size.

    Returns:
        (B, D) unpacked embeddings.
    """
    D = grouped_emb.shape[2]
    device = grouped_emb.device
    output = torch.zeros(B, D, device=device)

    for g, indices in enumerate(lane_indices_per_group):
        for pos, batch_idx in enumerate(indices):
            output[batch_idx] = grouped_emb[g, pos]

    return output


def _compute_relative_features(
    grouped_stats: torch.Tensor,
    group_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise relative features between lanes in each group.

    Features for pair (i, j):
    - lateral_offset_diff: stats[i,2] - stats[j,2]
    - speed_diff: stats[i,0] - stats[j,0]
    - density_ratio: stats[i,3] / (stats[j,3] + eps)

    Args:
        grouped_stats: (G, L, S) trajectory stats per lane.
        group_mask: (G, L) boolean mask.

    Returns:
        (G, L, L, 3) pairwise relative features.
    """
    G, L, _ = grouped_stats.shape
    device = grouped_stats.device
    eps = 1e-6

    # Extract individual stat channels
    speed = grouped_stats[:, :, 0]          # (G, L)
    lateral = grouped_stats[:, :, 2]        # (G, L)
    density = grouped_stats[:, :, 3]        # (G, L)

    # Pairwise differences: (G, L, 1) - (G, 1, L) -> (G, L, L)
    lateral_diff = lateral.unsqueeze(2) - lateral.unsqueeze(1)
    speed_diff = speed.unsqueeze(2) - speed.unsqueeze(1)
    density_ratio = density.unsqueeze(2) / (density.unsqueeze(1) + eps)
    # Clamp ratio to prevent extreme values from padded positions (density=0)
    density_ratio = density_ratio.clamp(-10.0, 10.0)

    rel_features = torch.stack([lateral_diff, speed_diff, density_ratio], dim=-1)
    return rel_features  # (G, L, L, 3)


class LaneEncoder(nn.Module):
    """Encode a lane (geometry + trajectories + stats) into a contrastive embedding.

    Args:
        polyline_k: Number of points per polyline.
        d_model: Polyline encoder hidden dimension.
        embed_dim: Final lane embedding dimension.
        proj_dim: Projection head output dimension (for contrastive loss).
        polyline_mode: "transformer" or "mlp".
        polyline_layers: Number of transformer layers.
        polyline_heads: Number of attention heads.
        stats_dim: Dimension of stats input (default 9 = 4 traj_stats + 5 roles).
        geometry_dropout: Probability of dropping geometry input during training
            to force trajectory-only learning for zero-shot.
        dropout: General dropout rate.
        use_cross_lane_attention: Enable cross-lane attention within groups.
        cross_lane_heads: Number of attention heads for cross-lane attention.
        rel_feat_dim: Dimension of pairwise relative features.
    """

    def __init__(
        self,
        polyline_k: int = 16,
        d_model: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 64,
        polyline_mode: str = "transformer",
        polyline_layers: int = 2,
        polyline_heads: int = 4,
        stats_dim: int = 9,
        geometry_dropout: float = 0.2,
        dropout: float = 0.1,
        use_cross_lane_attention: bool = False,
        cross_lane_heads: int = 4,
        rel_feat_dim: int = 3,
    ):
        super().__init__()

        self.geometry_dropout = geometry_dropout
        self.use_cross_lane_attention = use_cross_lane_attention

        # Geometry encoder (annotation waypoints)
        self.geometry_enc = PolylineEncoder(
            k=polyline_k,
            d_model=d_model,
            mode=polyline_mode,
            num_layers=polyline_layers,
            num_heads=polyline_heads,
            dropout=dropout,
        )

        # Trajectory encoder (shared weights for all trajectories per lane)
        self.traj_enc = PolylineEncoder(
            k=polyline_k,
            d_model=d_model,
            mode=polyline_mode,
            num_layers=polyline_layers,
            num_heads=polyline_heads,
            dropout=dropout,
        )

        # Stats encoder (trajectory statistics + role descriptor)
        self.stats_enc = nn.Sequential(
            nn.Linear(stats_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Input normalization: prevent collapse from similar raw feature ranges
        self.geom_norm = nn.BatchNorm1d(d_model)
        self.traj_norm = nn.BatchNorm1d(d_model)
        self.stats_norm = nn.BatchNorm1d(d_model)

        # Fusion: geometry_emb + traj_emb + stats_emb -> lane embedding
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Cross-lane attention (optional)
        if use_cross_lane_attention:
            self.cross_lane_attn = CrossLaneAttention(
                embed_dim=embed_dim,
                num_heads=cross_lane_heads,
                rel_feat_dim=rel_feat_dim,
                dropout=dropout,
            )

        # Projection head for contrastive loss (discarded at inference)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

        # Role regression heads — direct supervision to prevent collapse
        self.rank_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.edge_head = nn.Linear(embed_dim, 2)
        self.size_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
        )

        self.d_model = d_model
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization to spread embeddings apart from the start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_trajectories(
        self,
        traj_polylines: torch.Tensor,
        traj_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode variable-count trajectories per lane via mean pooling.

        Args:
            traj_polylines: (B, T, K, 2) padded trajectory polylines.
            traj_mask: (B, T) boolean mask (True = valid trajectory).

        Returns:
            (B, d_model) mean-pooled trajectory embedding.
        """
        B, T, K, _ = traj_polylines.shape

        flat = traj_polylines.reshape(B * T, K, 2)
        flat_emb = self.traj_enc(flat)
        emb = flat_emb.reshape(B, T, self.d_model)

        mask_expanded = traj_mask.unsqueeze(-1).float()
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        pooled = (emb * mask_expanded).sum(dim=1) / counts

        return pooled

    def _encode_per_lane(
        self,
        geometry: torch.Tensor,
        traj_polylines: torch.Tensor,
        traj_mask: torch.Tensor,
        traj_stats: torch.Tensor,
        drop_geometry: bool = None,
    ) -> torch.Tensor:
        """Per-lane encoding: geometry + trajectory + stats -> embedding.

        Extracts the shared encoding logic used by both forward() and
        forward_grouped(). Returns the fused embedding before projection/heads.

        Returns:
            (B, embed_dim) lane embedding.
        """
        # Geometry encoding
        geom_emb = self.geometry_enc(geometry)

        # Geometry dropout
        if drop_geometry is True:
            geom_emb = torch.zeros_like(geom_emb)
        elif drop_geometry is None and self.training and self.geometry_dropout > 0:
            mask = torch.bernoulli(
                torch.full((geom_emb.shape[0], 1), 1.0 - self.geometry_dropout,
                           device=geom_emb.device)
            )
            geom_emb = geom_emb * mask / (1.0 - self.geometry_dropout)

        # Trajectory encoding
        traj_emb = self.encode_trajectories(traj_polylines, traj_mask)

        # Stats encoding
        stats_emb = self.stats_enc(traj_stats)

        # Normalize each stream
        geom_emb = self.geom_norm(geom_emb)
        traj_emb = self.traj_norm(traj_emb)
        stats_emb = self.stats_norm(stats_emb)

        # Fusion
        fused = torch.cat([geom_emb, traj_emb, stats_emb], dim=-1)
        embedding = self.fusion(fused)

        return embedding

    def _apply_heads(
        self,
        embedding: torch.Tensor,
        proj_embedding: torch.Tensor = None,
    ) -> dict:
        """Apply projection and regression heads to embedding.

        Args:
            embedding: (B, embed_dim) per-lane embedding for regression heads.
            proj_embedding: (B, embed_dim) group-aware embedding for projection
                head. If None, uses embedding (non-grouped path).

        Returns:
            Dict with keys: embedding, projection, pred_rank, pred_edge, pred_size.
        """
        if proj_embedding is None:
            proj_embedding = embedding

        projection = self.proj_head(proj_embedding)
        projection = nn.functional.normalize(projection, dim=-1)

        # Regression heads use per-lane embedding (camera-invariant)
        pred_rank = self.rank_head(embedding).squeeze(-1)
        pred_edge = self.edge_head(embedding)
        pred_size = self.size_head(embedding).squeeze(-1)

        return {
            "embedding": proj_embedding,
            "projection": projection,
            "pred_rank": pred_rank,
            "pred_edge": pred_edge,
            "pred_size": pred_size,
        }

    def forward(
        self,
        geometry: torch.Tensor,
        traj_polylines: torch.Tensor,
        traj_mask: torch.Tensor,
        traj_stats: torch.Tensor,
        drop_geometry: bool = None,
    ) -> dict:
        """Forward pass (no cross-lane attention).

        Args:
            geometry: (B, K, 2) lane annotation waypoints.
            traj_polylines: (B, T, K, 2) assigned trajectory polylines.
            traj_mask: (B, T) boolean mask for valid trajectories.
            traj_stats: (B, 4) aggregate trajectory statistics.
            drop_geometry: If True, zero out geometry embedding. If None,
                uses stochastic dropout during training.

        Returns:
            Dict with keys: embedding, projection, pred_rank, pred_edge, pred_size.
        """
        embedding = self._encode_per_lane(
            geometry, traj_polylines, traj_mask, traj_stats, drop_geometry
        )
        return self._apply_heads(embedding)

    def forward_grouped(
        self,
        geometry: torch.Tensor,
        traj_polylines: torch.Tensor,
        traj_mask: torch.Tensor,
        traj_stats: torch.Tensor,
        group_ids: torch.Tensor,
        drop_geometry: bool = None,
    ) -> dict:
        """Forward pass with cross-lane attention within groups.

        Falls back to regular forward() if cross-lane attention is disabled.

        Args:
            geometry: (B, K, 2) lane annotation waypoints.
            traj_polylines: (B, T, K, 2) assigned trajectory polylines.
            traj_mask: (B, T) boolean mask for valid trajectories.
            traj_stats: (B, 4) aggregate trajectory statistics.
            group_ids: (B,) integer group id per lane.
            drop_geometry: If True, zero out geometry embedding.

        Returns:
            Dict with keys: embedding, projection, pred_rank, pred_edge, pred_size.
        """
        if not self.use_cross_lane_attention:
            return self.forward(
                geometry, traj_polylines, traj_mask, traj_stats, drop_geometry
            )

        B = geometry.shape[0]

        # Step 1: Per-lane encoding (camera-invariant, no group context)
        per_lane_emb = self._encode_per_lane(
            geometry, traj_polylines, traj_mask, traj_stats, drop_geometry
        )

        # Step 2: Pack into groups
        grouped_emb, grouped_stats, group_mask, unique_gids, lane_indices = \
            _pack_groups(per_lane_emb, group_ids, traj_stats)

        # Step 3: Compute pairwise relative features
        rel_features = _compute_relative_features(grouped_stats, group_mask)

        # Step 4: Cross-lane attention
        attended = self.cross_lane_attn(grouped_emb, group_mask, rel_features)

        # Step 5: Unpack back to flat batch (group-aware embedding)
        group_emb = _unpack_groups(attended, lane_indices, B)

        # Step 6: Apply heads — regression heads use per-lane (pre-attention)
        # embedding to avoid group-composition domain gap at zero-shot;
        # projection head uses group-aware embedding for contrastive learning.
        return self._apply_heads(per_lane_emb, proj_embedding=group_emb)
