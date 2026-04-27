import math

import torch
import torch.nn as nn


class PolylineEncoder(nn.Module):
    """Encode a K-point local-frame polyline into a fixed-dim embedding.

    Two modes:
      - "transformer": 1D Transformer on point tokens (recommended).
      - "mlp": Flatten + Fourier features + MLP (lightweight).
    """

    def __init__(
        self,
        k: int,
        d_model: int = 64,
        mode: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.mode = mode

        if mode == "transformer":
            self.point_proj = nn.Linear(2, d_model)
            self.pos_enc = nn.Parameter(
                self._sinusoidal_encoding(k, d_model), requires_grad=False
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # Fourier features + MLP
            n_freq = 8
            self.register_buffer(
                "freq_bands",
                2.0 ** torch.linspace(0, n_freq - 1, n_freq) * math.pi,
            )
            fourier_dim = k * 2 * (1 + 2 * n_freq)
            self.mlp = nn.Sequential(
                nn.Linear(fourier_dim, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            )

    @staticmethod
    def _sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, polylines: torch.Tensor) -> torch.Tensor:
        """
        Args:
            polylines: (N, K, 2) local-frame polyline points.

        Returns:
            (N, d_model) polyline embeddings.
        """
        if self.mode == "transformer":
            x = self.point_proj(polylines) + self.pos_enc.unsqueeze(0) # (N, K, d)
            x = self.transformer(x) # (N, K, d)
            return x.mean(dim=1) # (N, d) mean-pool over points
        else:
            N = polylines.shape[0]
            flat = polylines.reshape(N, -1) # (N, K*2)
            # Fourier features
            scaled = flat.unsqueeze(-1) * self.freq_bands # (N, K*2, n_freq)
            fourier = torch.cat([flat.unsqueeze(-1), torch.sin(scaled), torch.cos(scaled)], dim=-1)
            fourier = fourier.reshape(N, -1)
            return self.mlp(fourier)
