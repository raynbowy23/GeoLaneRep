"""Predictive decoder: slot -> motion displacement distribution.

Each slot is a motion generator that predicts the expected next-step
displacement for tracklets assigned to it. This makes slots explain
motion — the representation has physical meaning.
"""

import torch
import torch.nn as nn


class PredictiveDecoder(nn.Module):
    """Decode slot embeddings into Gaussian displacement distributions."""

    def __init__(self, slot_dim: int, output_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * 2), # mean + log_var
        )
        self.output_dim = output_dim

    def forward(self, slots: torch.Tensor):
        """
        Args:
            slots: (B, S, dim) or (S, dim) slot embeddings.

        Returns:
            mu: (..., S, output_dim) predicted mean displacement.
            log_var: (..., S, output_dim) predicted log-variance.
        """
        out = self.mlp(slots) # (..., S, 2*output_dim)
        mu, log_var = out.split(self.output_dim, dim=-1)
        # Clamp log_var for numerical stability
        log_var = log_var.clamp(-10.0, 2.0)
        return mu, log_var
