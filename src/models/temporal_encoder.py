"""Temporal lane encoder for time-series anomaly detection.

Wraps a frozen LaneEncoder, encodes each time window independently,
then feeds the sequence of embeddings through a GRU to capture temporal
dynamics. An anomaly head predicts per-window anomaly scores.

Architecture:
    For each window w:
        e_w = frozen_lane_encoder._encode_per_lane(geometry, traj_w, mask_w, stats_w)
    [e_0, ..., e_{W-1}] -> GRU -> h_seq (B, W, embed_dim)
    h_seq -> anomaly_head -> anomaly_scores (B, W)
"""

import torch
import torch.nn as nn

from src.models.lane_encoder import LaneEncoder


class LaneTemporalEncoder(nn.Module):
    """Temporal wrapper around a frozen LaneEncoder.

    Encodes per-window lane embeddings via the frozen encoder, then processes
    the window sequence through a GRU for temporal context. An anomaly head
    produces per-window anomaly scores.

    Args:
        lane_encoder: Pre-trained LaneEncoder instance (will be frozen).
        embed_dim: Embedding dimension (must match lane_encoder.embed_dim).
        freeze_encoder: Whether to freeze the lane encoder weights.
        gru_layers: Number of GRU layers.
        dropout: Dropout rate for anomaly head.
    """

    def __init__(
        self,
        lane_encoder: LaneEncoder,
        embed_dim: int = 128,
        freeze_encoder: bool = True,
        gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lane_encoder = lane_encoder
        self.embed_dim = embed_dim

        if freeze_encoder:
            for param in self.lane_encoder.parameters():
                param.requires_grad = False
            self.lane_encoder.eval()

        self.freeze_encoder = freeze_encoder

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

    def train(self, mode: bool = True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.lane_encoder.eval()
        return self

    def forward(
        self,
        geometry: torch.Tensor,
        window_traj_polylines: torch.Tensor,
        window_traj_mask: torch.Tensor,
        window_traj_stats: torch.Tensor,
        window_valid: torch.Tensor,
        roles: torch.Tensor = None,
    ) -> dict:
        """Forward pass: encode each window, run GRU, predict anomaly scores.

        Args:
            geometry: (B, K, 2) static lane waypoints.
            window_traj_polylines: (B, W, T, K, 2) per-window trajectory polylines.
            window_traj_mask: (B, W, T) boolean mask for valid trajectories.
            window_traj_stats: (B, W, 4) per-window trajectory statistics.
            window_valid: (B, W) boolean mask for valid windows.
            roles: (B, 5) role descriptors (concatenated with stats for encoder).

        Returns:
            Dict with keys:
                h_seq: (B, W, embed_dim) GRU hidden states per window.
                anomaly_scores: (B, W) per-window anomaly logits.
                window_embeddings: (B, W, embed_dim) per-window lane embeddings.
        """
        B, W, T, K, _ = window_traj_polylines.shape

        # Expand geometry across windows: (B, K, 2) -> (B*W, K, 2)
        geom_rep = geometry.unsqueeze(1).expand(B, W, K, 2).reshape(B * W, K, 2)

        # Flatten windows: (B*W, T, K, 2) and (B*W, T)
        traj_flat = window_traj_polylines.reshape(B * W, T, K, 2)
        mask_flat = window_traj_mask.reshape(B * W, T)

        # Build stats input: (B, W, 4) -> (B*W, stats_dim)
        # Concatenate roles if provided (same as contrastive training)
        if roles is not None:
            # Expand roles across windows: (B, 5) -> (B, W, 5)
            roles_exp = roles.unsqueeze(1).expand(B, W, 5)
            stats_input = torch.cat([window_traj_stats, roles_exp], dim=-1)  # (B, W, 9)
        else:
            stats_input = window_traj_stats  # (B, W, 4)

        stats_flat = stats_input.reshape(B * W, -1)

        # Encode each window through the frozen encoder
        with torch.no_grad() if self.freeze_encoder else _nullcontext():
            window_emb = self.lane_encoder._encode_per_lane(
                geom_rep, traj_flat, mask_flat, stats_flat
            )  # (B*W, embed_dim)

        window_emb = window_emb.reshape(B, W, self.embed_dim)  # (B, W, embed_dim)

        # Forward-fill empty windows with previous embedding
        window_emb = self._forward_fill(window_emb, window_valid)

        # GRU over window sequence
        h_seq, _ = self.gru(window_emb)  # (B, W, embed_dim)

        # Anomaly head
        anomaly_logits = self.anomaly_head(h_seq).squeeze(-1)  # (B, W)

        return {
            "h_seq": h_seq,
            "anomaly_scores": anomaly_logits,
            "window_embeddings": window_emb,
        }

    @staticmethod
    def _forward_fill(
        embeddings: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Forward-fill invalid window embeddings with the last valid one.

        Args:
            embeddings: (B, W, D) window embeddings.
            valid: (B, W) boolean mask.

        Returns:
            (B, W, D) with invalid windows replaced by previous valid embedding.
        """
        B, W, D = embeddings.shape
        result = embeddings.clone()

        for b in range(B):
            last_valid = None
            for w in range(W):
                if valid[b, w]:
                    last_valid = result[b, w].clone()
                elif last_valid is not None:
                    result[b, w] = last_valid
                # If no valid window seen yet, leave as-is (zero from encoder)

        return result


class _nullcontext:
    """Minimal no-op context manager for Python <3.7 compat."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
