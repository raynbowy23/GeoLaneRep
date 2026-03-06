"""Swappable GNN / Transformer / Graph Transformer backbone.

All backbones share the interface:
    forward(x, edge_index, edge_attr) -> h  where h is (N, hidden_dim).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class TrackletGNN(nn.Module):
    """GraphSAGE backbone with edge-feature conditioning."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        h = F.gelu(self.proj(x))
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new
        return h


class TrackletTransformer(nn.Module):
    """Full self-attention Transformer backbone (no explicit edges).

    Uses sinusoidal encoding from centroids for position awareness.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int = 8, dropout: float = 0.1, max_nodes: int = 2048):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos_proj = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None,
                edge_attr: torch.Tensor = None,
                centroids: torch.Tensor = None) -> torch.Tensor:
        h = self.proj(x) # (N, hidden)
        if centroids is not None:
            h = h + self.pos_proj(centroids)
        h = h.unsqueeze(0) # (1, N, hidden) — single-graph batch
        h = self.transformer(h)
        return h.squeeze(0) # (N, hidden)


class TrackletGraphTransformer(nn.Module):
    """Hybrid backbone: local GAT on graph edges + global self-attention.

    Each layer:
      1. Local: sparse GAT over edges with edge-feature bias.
      2. Global: full self-attention over all nodes.
      3. Gated combination of local + global.
      4. LayerNorm + residual + FFN.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int,
                 num_heads: int = 8, dropout: float = 0.1, edge_dim: int = 7):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.num_layers = num_layers

        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(num_layers):
            self.local_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                        dropout=dropout, edge_dim=edge_dim, add_self_loops=False)
            )
            self.global_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.gate_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            self.norms1.append(nn.LayerNorm(hidden_dim))
            self.norms2.append(nn.LayerNorm(hidden_dim))
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            ))

        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        h = F.gelu(self.proj(x)) # (N, hidden)

        for i in range(self.num_layers):
            residual = h

            # Local: sparse GAT
            h_local = self.local_layers[i](h, edge_index, edge_attr=edge_attr)

            # Global: full self-attention (single graph -> batch dim = 1)
            h_seq = h.unsqueeze(0) # (1, N, hidden)
            h_global, _ = self.global_layers[i](h_seq, h_seq, h_seq)
            h_global = h_global.squeeze(0) # (N, hidden)

            # Gated combination
            gate = torch.sigmoid(self.gate_layers[i](torch.cat([h_local, h_global], dim=-1)))
            h_combined = gate * h_local + (1 - gate) * h_global

            h = self.norms1[i](residual + h_combined)

            # FFN
            h = self.norms2[i](h + self.ffns[i](h))

        return h
