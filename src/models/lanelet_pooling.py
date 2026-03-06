"""Lanelet Pooling: differentiable graph coarsening for lanelet discovery.

Produces a small lanelet graph (M nodes + edges) from backbone tracklet
embeddings via soft assignment, feature pooling, and attribute prediction.

Architecture:
    h: (N, D) backbone tracklet embeddings
            |
    Assignment GNN (2-layer GATConv):
        h -> S (N, M) soft assignment matrix
            |
    Feature Pool:   X = S^T @ h         (M, D)
    Adjacency:      A = S^T @ A_in @ S  (M, M)
            |
    Attribute Heads (from X):
        pos     = soft_centroid + offset_head(X)  (M, 2)
        heading = normalize(heading_head(X))      (M, 2)
        conf    = sigmoid(conf_head(X))           (M,)
            |
    Binary Successor Head (geometry-filtered candidates):
        [X_i * X_j || geo_features] -> sigmoid logit
        Geometry filters: dist < 50m, downstream > 1m, heading_cos > 0.7
            |
    Regularization:
        entropy_loss = -(S log S).mean()
        link_loss    = ||S^T A S - A_coarse||_F
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# Legacy edge type labels (kept for backward compat with dataset GT)
EDGE_TYPE_NAMES = ["no_edge", "successor", "merge", "diverge", "adjacent"]
NUM_EDGE_TYPES = len(EDGE_TYPE_NAMES)

# Geometry feature indices for successor head input
GEO_FEAT_DIM = 5  # delta_along, delta_across, heading_cos, dist, downstream_indicator


class LaneletPooling(nn.Module):
    """Differentiable graph coarsening for lanelet node/edge discovery.

    Args:
        hidden_dim: Dimension of backbone node embeddings.
        num_lanelet_nodes: Number of coarsened lanelet nodes M.
        num_edge_types: Number of edge type classes (default 5).
        assign_layers: Number of GATConv layers in the assignment GNN.
        edge_dim: Dimension of input edge features.
        adj_threshold: Minimum coarsened adjacency to consider an edge candidate.
        num_heads: Number of attention heads in GATConv.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_lanelet_nodes: int = 16,
        num_edge_types: int = NUM_EDGE_TYPES,
        assign_layers: int = 2,
        edge_dim: int = 9,
        adj_threshold: float = 0.1,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_classifier: bool = True,
        successor_max_distance: float = 50.0,
        successor_min_heading_cos: float = 0.7,
        successor_min_downstream: float = 1.0,
        successor_max_lateral: float = 8.0,
        slot_across_weight: float = 3.0,
        density_radius: float = 15.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.M = num_lanelet_nodes
        self.num_edge_types = num_edge_types
        self.adj_threshold = adj_threshold
        self.use_edge_classifier = use_edge_classifier
        self.successor_max_distance = successor_max_distance
        self.successor_min_heading_cos = successor_min_heading_cos
        self.successor_min_downstream = successor_min_downstream
        self.successor_max_lateral = successor_max_lateral
        self.slot_across_weight = slot_across_weight
        self.density_radius = density_radius

        # --- Assignment GNN: produces soft assignment S (N, M) ---
        self.assign_convs = nn.ModuleList()
        self.assign_norms = nn.ModuleList()
        if assign_layers > 0:
            for i in range(assign_layers):
                in_dim = hidden_dim
                self.assign_convs.append(
                    GATConv(
                        in_dim,
                        hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        add_self_loops=False,
                    )
                )
                self.assign_norms.append(nn.LayerNorm(hidden_dim))
        # When assign_layers == 0, assign_convs is empty — assignment
        # comes directly from assign_head (Linear projection of backbone features)
        # +2 for normalized spatial coordinates (x, y) so assignment is position-aware
        self.assign_head = nn.Linear(hidden_dim + 2, num_lanelet_nodes)
        self.dropout = dropout

        # Learnable slot prototypes in normalized [-1, 1] space — provide initial
        # spatial diversity so each slot prefers a different region at init
        self.slot_positions = nn.Parameter(torch.empty(num_lanelet_nodes, 2))
        self._init_slot_positions()

        # --- Attribute heads (from pooled features X, shape (M, hidden_dim)) ---
        # Position: residual offset from soft centroid
        # Input: pooled features (hidden_dim) + 4 spatial conditioning features
        # (t_along, t_across, cos_heading, sin_heading)
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Heading: predict unit tangent
        self.heading_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        # Confidence: is this lanelet node active?
        # +1 for density feature (local tracklet density around each slot)
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Drivable probability head ---
        self.drivable_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Binary successor head (replaces 5-class edge classifier) ---
        # Input: [X_i * X_j || geo_features] = hidden_dim + GEO_FEAT_DIM
        if use_edge_classifier:
            successor_in = hidden_dim + GEO_FEAT_DIM
            self.successor_head = nn.Sequential(
                nn.Linear(successor_in, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.successor_head = None

        # Zero-init output heads for stable start
        self._zero_init_outputs()

    def _init_slot_positions(self):
        """Initialize slot prototypes for (s,d) frame layout.

        Wide spread along s-axis (traffic flow, x) [-1, 1].
        Narrow spread along d-axis (cross-lane, y) [-0.3, 0.3].
        """
        M = self.M
        n_rows = max(int(round(math.sqrt(M / 2))), 2)  # d-axis (across-lane)
        n_cols = int(math.ceil(M / n_rows))               # s-axis (along-lane)
        with torch.no_grad():
            idx = 0
            for r in range(n_rows):
                for c in range(n_cols):
                    if idx >= M:
                        break
                    s = 2.0 * c / max(n_cols - 1, 1) - 1.0       # s: [-1, 1] wide
                    d = 0.6 * r / max(n_rows - 1, 1) - 0.3       # d: [-0.3, 0.3] narrow
                    self.slot_positions.data[idx] = torch.tensor([s, d])
                    idx += 1

    def _zero_init_outputs(self):
        """Zero-init output layer weights for residual-safe initialization."""
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.heading_head[-1].weight)
        nn.init.zeros_(self.heading_head[-1].bias)
        nn.init.zeros_(self.conf_head[-1].weight)
        nn.init.zeros_(self.conf_head[-1].bias)
        # Small random init for assign_head (default) — slot prototypes
        # provide spatial diversity, assign_head refines with features

    @torch.no_grad()
    def _input_conditioned_slots(self, c_norm: torch.Tensor) -> torch.Tensor:
        """Compute per-sample slot positions via FPS + k-means on normalized centroids.

        Replaces the fixed grid slot_positions with input-adapted positions.
        Falls back to learned slot_positions when N < M.

        Args:
            c_norm: (N, 2) centroids normalized to [-1, 1].

        Returns:
            (M, 2) slot center positions in normalized space.
        """
        N = c_norm.shape[0]
        M = self.M
        if N < M:
            return self.slot_positions.data

        # Farthest-point sampling for diverse initialization
        centroid = c_norm.mean(dim=0)
        dists_to_centroid = (c_norm - centroid).norm(dim=1)
        selected = [dists_to_centroid.argmin().item()]
        min_dists = (c_norm - c_norm[selected[0]]).norm(dim=1)
        for _ in range(M - 1):
            farthest = min_dists.argmax().item()
            selected.append(farthest)
            new_dists = (c_norm - c_norm[farthest]).norm(dim=1)
            min_dists = torch.min(min_dists, new_dists)

        centers = c_norm[selected].clone()

        # K-means refinement (5 iterations)
        for _ in range(5):
            dists = torch.cdist(c_norm.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
            assigns = dists.argmin(dim=1)
            for c in range(M):
                mask = assigns == c
                if mask.any():
                    centers[c] = c_norm[mask].mean(dim=0)

        return centers

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        centroids: torch.Tensor,
        tangents: torch.Tensor,
        lane_group_heading: float = None,
    ) -> dict:
        """Forward pass: coarsen tracklet graph into lanelet graph.

        Args:
            h: (N, D) backbone node embeddings.
            edge_index: (2, E) tracklet graph edges.
            edge_attr: (E, edge_dim) edge features.
            centroids: (N, 2) tracklet positions in meters.
            tangents: (N, 2) tracklet unit tangent vectors.
            lane_group_heading: scalar heading (radians) of this lane group,
                or None for backward compatibility (spatial_cond = zeros).

        Returns:
            dict with:
                lanelet_positions: (M, 2) predicted waypoint positions.
                lanelet_headings: (M, 2) unit tangent at each waypoint.
                lanelet_confidence: (M,) active node probability.
                assignment_matrix: (N, M) soft assignment.
                coarsened_adj: (M, M) pooled adjacency.
                successor_logits: (E_c,) binary successor logits.
                edge_type_index: (2, E_c) lanelet edge indices.
                pooling_entropy_loss: scalar regularization.
                pooling_link_loss: scalar regularization.
                node_embeddings: (N, D) backbone output passthrough.
        """
        # Run entire pooling in float32 — small module, AMP float16 causes NaN
        # in GATConv attention scores
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            return self._forward_fp32(
                h.float(), edge_index, edge_attr.float(),
                centroids.float(), tangents.float(),
                lane_group_heading=lane_group_heading,
            )

    def _forward_fp32(self, h, edge_index, edge_attr, centroids, tangents,
                       lane_group_heading=None):
        """Float32 forward implementation."""
        N, D = h.shape
        device = h.device

        # --- Pre-compute heading axes (used for assignment + attribute heads) ---
        if lane_group_heading is not None:
            heading_val = float(lane_group_heading)
            cos_val = math.cos(heading_val)
            sin_val = math.sin(heading_val)
            axis_along = torch.tensor([cos_val, sin_val], device=device, dtype=h.dtype)  # (2,)
            axis_across = torch.tensor([-sin_val, cos_val], device=device, dtype=h.dtype)  # (2,)
        else:
            heading_val = cos_val = sin_val = None
            axis_along = axis_across = None

        # --- 1. Assignment (position-aware) ---
        # Normalize centroids to [-1, 1] for this graph so assignment
        # can partition by spatial location, not just feature similarity
        c_min = centroids.min(dim=0, keepdim=True)[0]
        c_max = centroids.max(dim=0, keepdim=True)[0]
        c_range = (c_max - c_min).clamp(min=1e-4)
        c_norm = 2.0 * (centroids - c_min) / c_range - 1.0  # (N, 2)

        if len(self.assign_convs) > 0:
            # GATConv assignment GNN
            h_assign = h
            for conv, norm in zip(self.assign_convs, self.assign_norms):
                h_new = conv(h_assign, edge_index, edge_attr=edge_attr)
                h_new = norm(h_new)
                h_new = F.gelu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                h_assign = h_assign + h_new  # residual
            assign_logits = self.assign_head(torch.cat([h_assign, c_norm], dim=-1))  # (N, M)
        else:
            # No GATConv — direct linear assignment from backbone features + position
            assign_logits = self.assign_head(torch.cat([h, c_norm], dim=-1))  # (N, M)

        # Per-sample slot positions from k-means (replaces fixed grid)
        effective_slots = self._input_conditioned_slots(c_norm)  # (M, 2)

        # Spatial affinity: anisotropic when heading is available
        # Weights across-lane distance more heavily so slots distinguish lanes
        delta = c_norm.unsqueeze(1) - effective_slots.unsqueeze(0)  # (N, M, 2)
        if axis_along is not None:
            # Transform heading axes into normalized coordinate space
            scale = (2.0 / c_range).squeeze(0)  # (2,) = [2/range_x, 2/range_y]
            along_n = axis_along * scale
            along_n = along_n / along_n.norm().clamp(min=1e-8)
            across_n = axis_across * scale
            across_n = across_n / across_n.norm().clamp(min=1e-8)
            d_along = (delta * along_n).sum(-1)   # (N, M)
            d_across = (delta * across_n).sum(-1)  # (N, M)
            dist_sq = d_along ** 2 + (self.slot_across_weight * d_across) ** 2
        else:
            dist_sq = (delta ** 2).sum(-1)  # (N, M)
        assign_logits = assign_logits + (-dist_sq) * 3.0  # scale controls initial sharpness

        # Soft assignment: (N, M) with softmax over M lanelet nodes
        S = F.softmax(assign_logits, dim=1)  # (N, M)

        # --- 2. Feature pooling: X = S^T @ h ---
        S_t = S.t()  # (M, N)
        mass = S_t.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (M, 1)
        X = (S_t @ h) / mass  # (M, D) — mass-normalized pooled features

        # --- 3. Build coarsened adjacency: A_coarse = S^T @ A_in @ S ---
        A_in = torch.zeros(N, N, device=device)
        A_in[edge_index[0], edge_index[1]] = 1.0
        A_coarse_raw = S_t @ A_in @ S  # (M, M)
        mass_outer = mass @ mass.t()  # (M, M)
        A_coarse = A_coarse_raw / mass_outer.clamp(min=1e-8)  # (M, M)

        # --- 4. Attribute heads ---
        soft_centroids = (S_t @ centroids) / mass  # (M, 2)

        # Spatial conditioning: project soft_centroids onto lane group axes
        # (axis_along, axis_across, cos_val, sin_val pre-computed at top of forward)
        if axis_along is not None:
            # Project all tracklet centroids to find lane group extent
            proj_along = centroids @ axis_along  # (N,)
            proj_across = centroids @ axis_across  # (N,)
            along_min, along_max = proj_along.min(), proj_along.max()
            across_min, across_max = proj_across.min(), proj_across.max()
            along_range = (along_max - along_min).clamp(min=1e-4)
            across_range = (across_max - across_min).clamp(min=1e-4)

            # Project soft centroids
            sc_along = soft_centroids @ axis_along  # (M,)
            sc_across = soft_centroids @ axis_across  # (M,)

            # Fractional position in [-1, 1]
            t_along = 2.0 * (sc_along - along_min) / along_range - 1.0  # (M,)
            t_across = 2.0 * (sc_across - across_min) / across_range - 1.0  # (M,)

            cos_h = torch.full((self.M,), cos_val, device=device, dtype=h.dtype)
            sin_h = torch.full((self.M,), sin_val, device=device, dtype=h.dtype)
            spatial_cond = torch.stack([
                t_along, t_across, cos_h, sin_h,
            ], dim=-1)  # (M, 4)
        else:
            spatial_cond = torch.zeros(self.M, 4, device=device, dtype=h.dtype)

        offset_sd = self.offset_head(torch.cat([X, spatial_cond], dim=-1))  # (M, 2) = (Δs, Δd) in road frame
        if lane_group_heading is not None:
            # Clamp offsets to observed extent: Δs within along-range, Δd within across-range
            # This prevents lanelet nodes from escaping the lane group area
            margin_along = 0.0  # no margin — stay within tracklet extent
            margin_across = 0.0  # no margin — stay within tracklet extent
            offset_sd = torch.stack([
                offset_sd[:, 0].clamp(-along_range * 0.5 - margin_along, along_range * 0.5 + margin_along),
                offset_sd[:, 1].clamp(-across_range * 0.5 - margin_across, across_range * 0.5 + margin_across),
            ], dim=-1)
            # Rotate road-aligned offsets (along, across) back to world XY
            offset = offset_sd[:, 0:1] * axis_along + offset_sd[:, 1:2] * axis_across
        else:
            # No heading available — fall back to raw XY offsets
            offset = offset_sd
        lanelet_positions = soft_centroids + offset  # (M, 2)

        # Hard clamp: keep positions strictly within tracklet centroid extent
        # No margin — predictions must stay within the lane group area
        pos_min = centroids.min(dim=0)[0]
        pos_max = centroids.max(dim=0)[0]
        lanelet_positions = lanelet_positions.clamp(min=pos_min, max=pos_max)

        raw_heading = self.heading_head(X)  # (M, 2)
        soft_tangents = (S_t @ tangents) / mass  # (M, 2)
        # Scale correction with tanh to prevent flipping the soft tangent direction
        heading_correction = torch.tanh(raw_heading) * 0.3
        heading_sum = soft_tangents + heading_correction
        lanelet_headings = heading_sum / (heading_sum.norm(dim=-1, keepdim=True) + 1e-8)

        # Density-conditioned confidence: append local tracklet density as feature
        with torch.no_grad():
            density_dist = torch.cdist(
                centroids.unsqueeze(0), soft_centroids.unsqueeze(0)
            ).squeeze(0)  # (N, M)
            density = torch.exp(
                -density_dist ** 2 / (2 * self.density_radius ** 2)
            ).sum(dim=0)  # (M,)
            density_norm = density / density.max().clamp(min=1e-8)  # [0, 1]
        conf_input = torch.cat([X, density_norm.unsqueeze(-1)], dim=-1)  # (M, D+1)
        confidence_logits = self.conf_head(conf_input).squeeze(-1)  # (M,)
        lanelet_confidence = torch.sigmoid(confidence_logits)  # (M,)

        # --- Drivable probability (trajectory-supported lanelets) ---
        traj_support = S.sum(dim=0)  # (M,) soft count of tracklets per lanelet
        traj_support_norm = traj_support / (traj_support.max() + 1e-8)  # [0, 1]
        drivable_logits = self.drivable_head(X).squeeze(-1)  # (M,)
        drivable_prob = torch.sigmoid(drivable_logits)  # (M,)

        # --- 5. Binary successor classification (geometry-based candidates) ---
        if self.use_edge_classifier and self.successor_head is not None:
            # Geometry-based candidate selection (replaces A_coarse > threshold)
            pos = lanelet_positions  # (M, 2)
            head = lanelet_headings  # (M, 2)

            # Pairwise vectors: delta[i,j] = pos[j] - pos[i]
            delta = pos.unsqueeze(1) - pos.unsqueeze(0)  # (M, M, 2)
            dist = delta.norm(dim=-1)  # (M, M)

            # Project delta onto heading_i to get along/across components
            # delta_along = delta . heading_i (positive = downstream)
            delta_along = (delta * head.unsqueeze(1)).sum(-1)  # (M, M)
            # delta_across = |delta - delta_along * heading_i|
            along_vec = delta_along.unsqueeze(-1) * head.unsqueeze(1)  # (M, M, 2)
            across_vec = delta - along_vec  # (M, M, 2)
            delta_across = across_vec.norm(dim=-1)  # (M, M)

            # Heading cosine similarity between i and j
            heading_cos = (head.unsqueeze(1) * head.unsqueeze(0)).sum(-1)  # (M, M)

            # Candidate mask: geometry filters
            not_self = ~torch.eye(self.M, dtype=torch.bool, device=device)
            candidate_mask = (
                (dist < self.successor_max_distance)
                & (delta_along > self.successor_min_downstream)
                & (delta_across < self.successor_max_lateral)
                & (heading_cos > self.successor_min_heading_cos)
                & not_self
            )

            edge_i, edge_j = torch.where(candidate_mask)
            E_c = edge_i.shape[0]

            if E_c > 0:
                # Build geometry features (5-dim, normalized)
                max_d = self.successor_max_distance
                geo_feats = torch.stack([
                    delta_along[edge_i, edge_j] / max_d,
                    delta_across[edge_i, edge_j] / max_d,
                    heading_cos[edge_i, edge_j],
                    dist[edge_i, edge_j] / max_d,
                    (delta_along[edge_i, edge_j] > 0).float(),  # downstream indicator
                ], dim=-1)  # (E_c, 5)

                # Feature interaction: element-wise product of node embeddings
                xi_xj = X[edge_i] * X[edge_j]  # (E_c, D)
                edge_features = torch.cat([xi_xj, geo_feats], dim=-1)  # (E_c, D+5)
                successor_logits = self.successor_head(edge_features).squeeze(-1)  # (E_c,)
                edge_type_index = torch.stack([edge_i, edge_j], dim=0)
            else:
                successor_logits = torch.zeros(0, device=device)
                edge_type_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        else:
            successor_logits = torch.zeros(0, device=device)
            edge_type_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        # --- 6. Regularization losses (float32) ---
        # Entropy: encourage sharp assignment
        S_log = (S + 1e-8).log()
        entropy_loss = -(S * S_log).mean()

        # Balancing: prevent collapse — encourage uniform slot utilization
        # L_bal = sum_k (s_bar_k - 1/M)^2  where s_bar_k = mean_i S[i,k]
        s_bar = S.mean(dim=0)  # (M,) mean assignment per lanelet
        balancing_loss = ((s_bar - 1.0 / self.M) ** 2).sum()

        # Link loss: penalize off-diagonal mass for clean clustering
        A_link = A_coarse.clone().fill_diagonal_(0)
        link_loss = A_link.pow(2).mean()

        return {
            "lanelet_positions": lanelet_positions,      # (M, 2)
            "lanelet_headings": lanelet_headings,        # (M, 2)
            "lanelet_confidence": lanelet_confidence,    # (M,) sigmoid
            "confidence_logits": confidence_logits,      # (M,) raw logits
            "drivable_prob": drivable_prob,              # (M,) sigmoid
            "drivable_logits": drivable_logits,          # (M,) raw logits
            "traj_support": traj_support_norm,           # (M,) normalized [0,1]
            "traj_support_raw": traj_support,              # (M,) raw soft counts
            "assign_logits": assign_logits,              # (N, M) raw logits before softmax
            "assignment_matrix": S,                      # (N, M)
            "coarsened_adj": A_coarse,                   # (M, M)
            "successor_logits": successor_logits,          # (E_c,)
            "edge_type_index": edge_type_index,          # (2, E_c)
            "pooling_entropy_loss": entropy_loss,        # scalar
            "pooling_balancing_loss": balancing_loss,    # scalar
            "pooling_link_loss": link_loss,              # scalar
            "node_embeddings": h,                        # (N, D)
        }
