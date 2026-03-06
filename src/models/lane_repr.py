"""LaneReprModel v3: lanelet discovery via differentiable graph coarsening.

Simplified pipeline (single path, no branching):
    1. PolylineEncoder: per-tracklet shape encoding.
    2. Backbone (GraphTransformer): relational reasoning.
    3. LaneletPooling: differentiable coarsening into lanelet graph.
"""

import torch
import torch.nn as nn

from src.models.polyline_encoder import PolylineEncoder
from src.models.backbone import TrackletGNN, TrackletTransformer, TrackletGraphTransformer
from src.models.lanelet_pooling import LaneletPooling


def build_backbone(backbone_type, in_dim, hidden_dim, num_layers, num_heads, dropout, edge_dim):
    """Factory for backbone selection."""
    if backbone_type == "gnn":
        return TrackletGNN(in_dim, hidden_dim, num_layers, dropout)
    elif backbone_type == "transformer":
        return TrackletTransformer(in_dim, hidden_dim, num_layers, num_heads, dropout)
    else:
        return TrackletGraphTransformer(in_dim, hidden_dim, num_layers, num_heads, dropout, edge_dim)


class LaneReprModel(nn.Module):
    """Lanelet discovery model via differentiable graph coarsening.

    Pipeline:
        PolylineEncoder -> Backbone -> LaneletPooling -> lanelet graph
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", config)
        data_cfg = config.get("data", {})

        hidden_dim = model_cfg.get("hidden_dim", 128)
        num_layers = model_cfg.get("num_layers", 4)
        num_heads = model_cfg.get("num_heads", 8)
        dropout = model_cfg.get("dropout", 0.1)
        polyline_k = model_cfg.get("polyline_k", 10)
        backbone_type = model_cfg.get("backbone", "graph_transformer")

        # Polyline encoder
        poly_enc_mode = model_cfg.get("polyline_encoder", "transformer")
        poly_enc_dim = model_cfg.get("polyline_encoder_dim", 64)
        poly_enc_layers = model_cfg.get("polyline_encoder_layers", 2)
        poly_enc_heads = model_cfg.get("polyline_encoder_heads", 4)
        self.polyline_enc = PolylineEncoder(
            k=polyline_k,
            d_model=poly_enc_dim,
            mode=poly_enc_mode,
            num_layers=poly_enc_layers,
            num_heads=poly_enc_heads,
            dropout=dropout,
        )

        # Node feature dim = polyline_emb + tracklet features (6)
        node_feat_dim = 6
        in_dim = poly_enc_dim + node_feat_dim
        edge_dim = data_cfg.get(
            "edge_feature_dim",
            9 if data_cfg.get("use_lateral_ordering", False) else 7,
        )

        # Backbone
        self.backbone = build_backbone(
            backbone_type, in_dim, hidden_dim, num_layers, num_heads, dropout, edge_dim
        )
        self.backbone_type = backbone_type

        # Lanelet pooling (core v3 module)
        self.lanelet_pool = LaneletPooling(
            hidden_dim=hidden_dim,
            num_lanelet_nodes=model_cfg.get("num_lanelet_nodes", 16),
            num_edge_types=model_cfg.get("num_lanelet_edge_types", 5),
            assign_layers=model_cfg.get("lanelet_assign_layers", 2),
            edge_dim=edge_dim,
            adj_threshold=model_cfg.get("lanelet_adj_threshold", 0.1),
            num_heads=num_heads,
            dropout=dropout,
            use_edge_classifier=model_cfg.get("use_edge_classifier", True),
            successor_max_distance=model_cfg.get("successor_max_distance", 50.0),
            successor_min_heading_cos=model_cfg.get("successor_min_heading_cos", 0.7),
            successor_min_downstream=model_cfg.get("successor_min_downstream", 1.0),
            successor_max_lateral=model_cfg.get("successor_max_lateral", 8.0),
            slot_across_weight=model_cfg.get("slot_across_weight", 3.0),
            density_radius=model_cfg.get("density_radius", 15.0),
        )

        self.apply(self._init_weights)
        # Re-apply zero-init on output heads after global init
        self.lanelet_pool._zero_init_outputs()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GRUCell,)):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, data, return_intermediates: bool = False) -> dict:
        """Forward pass.

        Args:
            data: PyG Data with fields:
                x: (N, 6) tracklet features.
                polylines: (N, K, 2) local-frame polylines.
                edge_index: (2, E) graph edges.
                edge_attr: (E, edge_dim) edge features.
                centroids: (N, 2) positions in meters.
                tangents: (N, 2) unit tangent vectors.

        Returns:
            dict with lanelet graph predictions (see LaneletPooling.forward).
        """
        # 1. Encode polylines
        poly_emb = self.polyline_enc(data.polylines)  # (N, poly_dim)

        # 2. Combine with tracklet features
        x = torch.cat([poly_emb, data.x], dim=-1)  # (N, poly_dim + 6)

        # 3. Backbone
        if self.backbone_type == "transformer":
            h = self.backbone(x, data.edge_index, data.edge_attr,
                              centroids=data.centroids)
        else:
            h = self.backbone(x, data.edge_index, data.edge_attr)  # (N, hidden)

        # 4. Lanelet pooling
        heading = getattr(data, "lane_group_heading", None)
        result = self.lanelet_pool(
            h, data.edge_index, data.edge_attr,
            data.centroids, data.tangents,
            lane_group_heading=heading,
        )

        return result
