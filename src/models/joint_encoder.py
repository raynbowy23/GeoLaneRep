"""Joint lane encoder for simultaneous contrastive + temporal training.

Combines a trainable LaneEncoder with a GRU temporal head and anomaly
detector. Unlike LaneTemporalEncoder (which freezes the encoder), this
model backpropagates through the encoder from both losses:

  - Contrastive path: mean-pool per-window embeddings -> projection head -> InfoNCE
  - Temporal path: GRU over per-window embeddings -> anomaly head -> BCE

Architecture:
    geometry (static)          <- shared across all windows
    traj_polylines(w)          <- per window
    traj_stats(w) + roles      <- per window
            |
      LaneEncoder (trainable)
            |
      e_i(w)  <-  per-window embedding (B, W, 128)
            |                    |
      GRU over windows      mean-pool over windows
            |                    |
      anomaly_head           contrastive projection head
            |                    |
      BCE loss               InfoNCE + role regression loss
"""

import torch
import torch.nn as nn

from src.models.lane_encoder import LaneEncoder
from src.models.temporal_encoder import LaneTemporalEncoder


class JointLaneEncoder(nn.Module):
    """Unified model: trainable LaneEncoder + GRU + anomaly head + contrastive heads.

    Key difference from LaneTemporalEncoder: encoder is NOT frozen, and
    the model produces both temporal and contrastive outputs in one forward pass.

    Args:
        lane_encoder: LaneEncoder instance (will remain trainable).
        embed_dim: Embedding dimension (must match lane_encoder.embed_dim).
        gru_layers: Number of GRU layers.
        dropout: Dropout rate for anomaly head.
    """

    def __init__(
        self,
        lane_encoder: LaneEncoder,
        embed_dim: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lane_encoder = lane_encoder  # TRAINABLE
        self.embed_dim = embed_dim

        # Temporal path: GRU + anomaly head (same as LaneTemporalEncoder)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.anomaly_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        # Contrastive path reuses lane_encoder.proj_head and regression heads

    def forward(
        self,
        geometry: torch.Tensor,
        window_traj_polylines: torch.Tensor,
        window_traj_mask: torch.Tensor,
        window_traj_stats: torch.Tensor,
        window_valid: torch.Tensor,
        roles: torch.Tensor = None,
    ) -> dict:
        """Forward pass producing both temporal and contrastive outputs.

        Args:
            geometry: (B, K, 2) static lane waypoints.
            window_traj_polylines: (B, W, T, K, 2) per-window trajectory polylines.
            window_traj_mask: (B, W, T) boolean mask for valid trajectories.
            window_traj_stats: (B, W, 4) per-window trajectory statistics.
            window_valid: (B, W) boolean mask for valid windows.
            roles: (B, 5) role descriptors.

        Returns:
            Dict with keys:
                anomaly_scores: (B, W) per-window anomaly logits.
                h_seq: (B, W, embed_dim) GRU hidden states.
                window_embeddings: (B, W, embed_dim) per-window lane embeddings.
                projection: (B, proj_dim) L2-normalized for InfoNCE.
                embedding: (B, embed_dim) mean-pooled across valid windows.
                pred_rank: (B,) predicted lateral rank.
                pred_edge: (B, 2) predicted edge logits.
                pred_size: (B,) predicted group size.
        """
        B, W, T, K, _ = window_traj_polylines.shape

        # --- Encode each window through the trainable encoder ---

        # Expand geometry across windows: (B, K, 2) -> (B*W, K, 2)
        geom_rep = geometry.unsqueeze(1).expand(B, W, K, 2).reshape(B * W, K, 2)

        # Flatten windows: (B*W, T, K, 2) and (B*W, T)
        traj_flat = window_traj_polylines.reshape(B * W, T, K, 2)
        mask_flat = window_traj_mask.reshape(B * W, T)

        # Build stats input: concatenate roles if provided
        if roles is not None:
            roles_exp = roles.unsqueeze(1).expand(B, W, 5)
            stats_input = torch.cat([window_traj_stats, roles_exp], dim=-1)  # (B, W, 9)
        else:
            stats_input = window_traj_stats  # (B, W, 4)

        stats_flat = stats_input.reshape(B * W, -1)

        # Encoder is trainable — no torch.no_grad()
        window_emb = self.lane_encoder._encode_per_lane(
            geom_rep, traj_flat, mask_flat, stats_flat
        )  # (B*W, embed_dim)

        window_emb = window_emb.reshape(B, W, self.embed_dim)  # (B, W, embed_dim)

        # Forward-fill empty windows
        window_emb = LaneTemporalEncoder._forward_fill(window_emb, window_valid)

        # --- Temporal path: GRU -> anomaly scores ---
        h_seq, _ = self.gru(window_emb)  # (B, W, embed_dim)
        anomaly_logits = self.anomaly_head(h_seq).squeeze(-1)  # (B, W)

        # --- Contrastive path: mean-pool valid windows -> projection + regression ---
        valid_mask = window_valid.unsqueeze(-1).float()  # (B, W, 1)
        valid_count = valid_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        mean_emb = (window_emb * valid_mask).sum(dim=1) / valid_count  # (B, embed_dim)

        # Apply projection and regression heads via LaneEncoder
        head_output = self.lane_encoder._apply_heads(mean_emb)

        return {
            "anomaly_scores": anomaly_logits,       # (B, W)
            "h_seq": h_seq,                          # (B, W, embed_dim)
            "window_embeddings": window_emb,         # (B, W, embed_dim)
            "projection": head_output["projection"], # (B, proj_dim)
            "embedding": mean_emb,                   # (B, embed_dim)
            "pred_rank": head_output["pred_rank"],   # (B,)
            "pred_edge": head_output["pred_edge"],   # (B, 2)
            "pred_size": head_output["pred_size"],   # (B,)
        }
