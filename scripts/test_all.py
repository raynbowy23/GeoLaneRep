"""Smoke tests for v3 lanelet discovery modules."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np


def test_lanelet_pooling():
    """Test LaneletPooling forward pass with random input."""
    from src.models.lanelet_pooling import LaneletPooling

    N = 50
    hidden_dim = 128
    M = 16
    edge_dim = 7

    pool = LaneletPooling(
        hidden_dim=hidden_dim,
        num_lanelet_nodes=M,
        num_edge_types=5,
        assign_layers=2,
        edge_dim=edge_dim,
        adj_threshold=0.1,
        num_heads=4,
        dropout=0.1,
    )

    # Random inputs
    h = torch.randn(N, hidden_dim)
    # Build random edges (sparse)
    src = torch.randint(0, N, (100,))
    dst = torch.randint(0, N, (100,))
    edge_index = torch.stack([src, dst])
    edge_attr = torch.randn(100, edge_dim)
    centroids = torch.randn(N, 2) * 50  # meters
    tangents = torch.randn(N, 2)
    tangents = tangents / (tangents.norm(dim=-1, keepdim=True) + 1e-8)

    out = pool(h, edge_index, edge_attr, centroids, tangents)

    # Check output shapes
    assert out["lanelet_positions"].shape == (M, 2), f"positions: {out['lanelet_positions'].shape}"
    assert out["lanelet_headings"].shape == (M, 2), f"headings: {out['lanelet_headings'].shape}"
    assert out["lanelet_confidence"].shape == (M,), f"confidence: {out['lanelet_confidence'].shape}"
    assert out["assignment_matrix"].shape == (N, M), f"assignment: {out['assignment_matrix'].shape}"
    assert out["coarsened_adj"].shape == (M, M), f"adj: {out['coarsened_adj'].shape}"
    assert out["node_embeddings"].shape == (N, hidden_dim), f"node_emb: {out['node_embeddings'].shape}"

    # Assignment matrix should sum to ~1 per row
    row_sums = out["assignment_matrix"].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(N), atol=1e-5), f"Row sums: {row_sums}"

    # Confidence should be in [0, 1]
    assert (out["lanelet_confidence"] >= 0).all() and (out["lanelet_confidence"] <= 1).all()

    # Regularization losses should be scalar
    assert out["pooling_entropy_loss"].dim() == 0
    assert out["pooling_link_loss"].dim() == 0

    print("  LaneletPooling: PASSED")
    return True


def test_lane_repr_model():
    """Test full LaneReprModel forward pass."""
    from src.models.lane_repr import LaneReprModel
    from torch_geometric.data import Data

    config = {
        "model": {
            "backbone": "graph_transformer",
            "hidden_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "polyline_k": 10,
            "polyline_encoder": "transformer",
            "polyline_encoder_dim": 32,
            "polyline_encoder_layers": 1,
            "polyline_encoder_heads": 2,
            "num_lanelet_nodes": 8,
            "num_lanelet_edge_types": 5,
            "lanelet_assign_layers": 2,
            "lanelet_adj_threshold": 0.1,
        },
        "data": {},
    }

    model = LaneReprModel(config)
    N = 30
    K = 10

    data = Data(
        x=torch.randn(N, 6),
        polylines=torch.randn(N, K, 2),
        edge_index=torch.stack([
            torch.randint(0, N, (80,)),
            torch.randint(0, N, (80,)),
        ]),
        edge_attr=torch.randn(80, 7),
        centroids=torch.randn(N, 2) * 50,
        tangents=torch.randn(N, 2),
    )
    data.tangents = data.tangents / (data.tangents.norm(dim=-1, keepdim=True) + 1e-8)

    out = model(data)

    M = config["model"]["num_lanelet_nodes"]
    assert out["lanelet_positions"].shape == (M, 2)
    assert out["assignment_matrix"].shape == (N, M)
    assert "pooling_entropy_loss" in out

    print("  LaneReprModel: PASSED")
    return True


def test_losses():
    """Test loss computation with synthetic data."""
    from src.training.losses import compute_total_loss

    M = 8
    N = 30
    G = 6
    E_c = 10

    conf_logits = torch.randn(M)
    drv_logits = torch.randn(M)
    traj_sup = torch.rand(M)
    model_output = {
        "lanelet_positions": torch.randn(M, 2),
        "lanelet_headings": torch.randn(M, 2),
        "lanelet_confidence": torch.sigmoid(conf_logits),
        "confidence_logits": conf_logits,
        "drivable_prob": torch.sigmoid(drv_logits),
        "drivable_logits": drv_logits,
        "traj_support": traj_sup,
        "assignment_matrix": torch.softmax(torch.randn(N, M), dim=1),
        "coarsened_adj": torch.rand(M, M),
        "successor_logits": torch.randn(E_c),
        "edge_type_index": torch.stack([
            torch.randint(0, M, (E_c,)),
            torch.randint(0, M, (E_c,)),
        ]),
        "pooling_entropy_loss": torch.tensor(0.5),
        "pooling_link_loss": torch.tensor(0.3),
        "pooling_balancing_loss": torch.tensor(0.2),
        "assign_logits": torch.randn(N, M),
        "node_embeddings": torch.randn(N, 64),
    }

    class FakeData:
        gt_lanelet_positions = torch.randn(G, 2)
        gt_lanelet_tangents = torch.randn(G, 2)
        gt_lanelet_lane_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        gt_lanelet_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
        gt_lanelet_edge_types = torch.tensor([1, 1, 1, 1])
        centroids = torch.randn(N, 2) * 50
        sumo_adj_types = torch.zeros(G, G, dtype=torch.long)

    data = FakeData()
    gt_labels = torch.tensor([0] * 10 + [1] * 10 + [-1] * 10)

    config = {"training": {
        "lanelet_position_weight": 5.0,
        "lanelet_heading_weight": 2.0,
        "lanelet_edge_weight": 3.0,
        "lanelet_entropy_weight": 1.0,
        "lanelet_link_weight": 1.0,
        "lanelet_assign_consistency_weight": 2.0,
        "lanelet_confidence_weight": 0.5,
    }}

    total_loss, loss_dict = compute_total_loss(model_output, data, config, gt_labels)

    assert not torch.isnan(total_loss), "Total loss is NaN"
    assert total_loss.item() > 0, "Total loss should be positive"
    assert "total" in loss_dict
    assert "lanelet_position" in loss_dict
    assert "lanelet_heading" in loss_dict

    print("  Losses: PASSED")
    return True


def test_evaluator():
    """Test metric computation with synthetic data."""
    from src.training.evaluator import compute_lanelet_metrics

    M = 8
    N = 30
    G = 6

    eval_conf_logits = torch.randn(M)
    model_output = {
        "lanelet_positions": torch.randn(M, 2),
        "lanelet_headings": torch.randn(M, 2),
        "lanelet_confidence": torch.sigmoid(eval_conf_logits),
        "confidence_logits": eval_conf_logits,
        "assignment_matrix": torch.softmax(torch.randn(N, M), dim=1),
        "coarsened_adj": torch.rand(M, M),
        "successor_logits": torch.randn(5),
        "edge_type_index": torch.stack([
            torch.randint(0, M, (5,)),
            torch.randint(0, M, (5,)),
        ]),
        "pooling_entropy_loss": torch.tensor(0.5),
        "pooling_link_loss": torch.tensor(0.3),
        "pooling_balancing_loss": torch.tensor(0.2),
        "assign_logits": torch.randn(N, M),
        "node_embeddings": torch.randn(N, 64),
    }

    class FakeData:
        gt_lanelet_positions = torch.randn(G, 2)
        gt_lanelet_tangents = torch.randn(G, 2)
        gt_lanelet_lane_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        gt_lanelet_edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
        gt_lanelet_edge_types = torch.tensor([1, 1, 1, 1])

    data = FakeData()
    gt_labels = torch.tensor([0] * 10 + [1] * 10 + [-1] * 10)

    metrics = compute_lanelet_metrics(model_output, data, gt_labels)

    assert "node_position_error" in metrics
    assert "heading_angular_error" in metrics
    assert "assignment_ari" in metrics
    assert "n_active_lanelet_nodes" in metrics

    print("  Evaluator: PASSED")
    return True


def test_lane_extraction():
    """Test lane extraction from lanelet output."""
    from src.data.lane_extraction import extract_lanelet_lanes, lanelet_lanes_to_dict

    M = 8
    model_output = {
        "lanelet_positions": torch.randn(M, 2),
        "lanelet_headings": torch.randn(M, 2),
        "lanelet_confidence": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1, 0.05, 0.02, 0.01]),
        "assignment_matrix": torch.softmax(torch.randn(30, M), dim=1),
        "coarsened_adj": torch.rand(M, M),
        # Create successor edges: 0->1->2->3 (chain)
        "successor_logits": torch.zeros(6),
        "edge_type_index": torch.tensor([
            [0, 1, 2, 4, 5, 6],
            [1, 2, 3, 5, 6, 7],
        ]),
        "pooling_entropy_loss": torch.tensor(0.5),
        "pooling_link_loss": torch.tensor(0.3),
        "node_embeddings": torch.randn(30, 64),
    }
    # Set high logit for first 3 edges (successors), low for rest
    model_output["successor_logits"][:3] = 5.0   # sigmoid > 0.5 -> successor
    model_output["successor_logits"][3:] = -5.0   # sigmoid < 0.5 -> no successor

    lanes = extract_lanelet_lanes(model_output, conf_threshold=0.3, min_nodes=2)
    assert len(lanes) >= 1, f"Expected at least 1 lane, got {len(lanes)}"

    dicts = lanelet_lanes_to_dict(lanes)
    assert len(dicts) >= 1
    assert "waypoints" in dicts[0]

    print("  Lane extraction: PASSED")
    return True


def main():
    print("Running v3 smoke tests...")
    print()

    results = []
    tests = [
        ("LaneletPooling forward", test_lanelet_pooling),
        ("LaneReprModel forward", test_lane_repr_model),
        ("Loss computation", test_losses),
        ("Evaluator metrics", test_evaluator),
        ("Lane extraction", test_lane_extraction),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("=" * 50)
    n_pass = sum(1 for _, p in results if p)
    n_total = len(results)
    print(f"Results: {n_pass}/{n_total} passed")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
