import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLaneAttention(nn.Module):
    """Multi-head self-attention across lanes within the same group.

    Args:
        embed_dim: Lane embedding dimension.
        num_heads: Number of attention heads.
        rel_feat_dim: Dimension of pairwise relative features.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        rel_feat_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Project pairwise relative features to per-head attention bias
        self.rel_proj = nn.Linear(rel_feat_dim, num_heads)

    def forward(
        self,
        embeddings: torch.Tensor,
        group_mask: torch.Tensor,
        rel_features: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-lane attention within groups.

        Manual implementation to avoid NaN from nn.MultiheadAttention with
        combined key_padding_mask + attn_mask on padded positions.

        Args:
            embeddings: (G, L, D) lane embeddings packed by group.
            group_mask: (G, L) boolean mask (True = valid lane).
            rel_features: (G, L, L, rel_feat_dim) pairwise relative features.

        Returns:
            (G, L, D) attended embeddings.
        """
        G, L, D = embeddings.shape
        H = self.num_heads
        d_k = self.head_dim

        # Project Q, K, V: (G, L, D) -> (G, H, L, d_k)
        Q = self.q_proj(embeddings).view(G, L, H, d_k).transpose(1, 2)
        K = self.k_proj(embeddings).view(G, L, H, d_k).transpose(1, 2)
        V = self.v_proj(embeddings).view(G, L, H, d_k).transpose(1, 2)

        # Scaled dot-product attention scores: (G, H, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        # Add relative feature bias: (G, L, L, rel_feat_dim) -> (G, H, L, L)
        rel_bias = self.rel_proj(rel_features)  # (G, L, L, H)
        rel_bias = rel_bias.permute(0, 3, 1, 2)  # (G, H, L, L)
        scores = scores + rel_bias

        # Mask invalid positions: use large negative (not -inf) to avoid NaN
        # in softmax backward when a query has all-masked keys.
        pair_mask = group_mask.unsqueeze(2) & group_mask.unsqueeze(1)  # (G, L, L)
        pair_mask = pair_mask.unsqueeze(1)  # (G, 1, L, L) — broadcast over heads
        scores = scores.masked_fill(~pair_mask, -1e4)

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Zero out attention FROM padded query positions
        # (softmax gives uniform weights for all-masked rows; zero them out)
        query_mask = group_mask.unsqueeze(1).unsqueeze(-1)  # (G, 1, L, 1)
        attn_weights = attn_weights * query_mask.float()

        # Weighted sum: (G, H, L, d_k)
        attended = torch.matmul(attn_weights, V)

        # Concatenate heads: (G, L, D)
        attended = attended.transpose(1, 2).contiguous().view(G, L, D)
        attended = self.out_proj(attended)

        # Residual + LayerNorm
        output = self.norm(embeddings + attended)

        # Zero out padded positions
        output = output * group_mask.unsqueeze(-1).float()

        return output
