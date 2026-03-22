#!/usr/bin/env python3
"""Baseline and ablation evaluation for comparison table.

Methods:
    traj-stats     — 5 hand-crafted trajectory stats (no roles, no encoder)
    stats-oracle   — traj_stats(4) + roles(5) = 9-dim (uses ground-truth roles)
    per-camera-sup — per-camera supervised classifier (not zero-shot)
    no-cross-attn  — encoder checkpoint trained without cross-lane attention
    encoder        — full encoder (contrastive-only or joint checkpoint)

Usage:
    python scripts/eval_baseline.py --method traj-stats
    python scripts/eval_baseline.py --method per-camera-sup
    python scripts/eval_baseline.py --method no-cross-attn --checkpoint results/...
    python scripts/eval_baseline.py --method encoder --checkpoint results/joint_encoder/checkpoints/best.pt
    python scripts/eval_baseline.py --all  # run all methods
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Dataset loading ──────────────────────────────────────────────────


def _load_dataset(config):
    from src.data.lane_dataset import LaneDataset
    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})
    return LaneDataset(
        config=config,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )


# ── Leave-one-camera-out eval (shared) ───────────────────────────────


def _leave_one_out_eval(dataset, features: torch.Tensor) -> dict:
    """Run leave-one-camera-out matching evaluation given pre-computed features.

    Args:
        dataset: LaneDataset.
        features: (N, D) L2-normalized feature vectors.

    Returns:
        Dict with aggregate and per-camera metrics.
    """
    all_results = {}
    for camera in dataset.cameras:
        held_out_indices = dataset.get_camera_indices(camera)
        if not held_out_indices:
            continue

        held_out_set = set(held_out_indices)
        train_indices = [i for i in range(len(dataset)) if i not in held_out_set]

        ref_feats = features[train_indices]
        query_feats = features[held_out_indices]

        sim_matrix = torch.mm(query_feats, ref_feats.t())
        best_match_idx = sim_matrix.argmax(dim=1)
        best_match_sim = sim_matrix.gather(1, best_match_idx.unsqueeze(1)).squeeze(1)

        query_roles = torch.stack([dataset[i]["role"] for i in held_out_indices])
        ref_roles = torch.stack([dataset[train_indices[j]]["role"] for j in best_match_idx])

        lat_rank_diff = (query_roles[:, 0] - ref_roles[:, 0]).abs()
        role_sim = torch.nn.functional.cosine_similarity(query_roles, ref_roles, dim=1)

        # Edge F1: macro-average over leftmost and rightmost
        def _f1(pred, gt):
            tp = ((pred == 1) & (gt == 1)).sum().float()
            fp = ((pred == 1) & (gt == 0)).sum().float()
            fn = ((pred == 0) & (gt == 1)).sum().float()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        left_f1 = _f1(ref_roles[:, 1], query_roles[:, 1])
        right_f1 = _f1(ref_roles[:, 2], query_roles[:, 2])
        edge_f1 = (left_f1 + right_f1) / 2.0

        all_results[camera] = {
            "mean_match_sim": best_match_sim.mean().item(),
            "mean_lat_rank_diff": lat_rank_diff.mean().item(),
            "edge_flag_f1": edge_f1,
            "mean_role_similarity": role_sim.mean().item(),
            "n_query": len(held_out_indices),
        }

    # Aggregate
    agg = defaultdict(list)
    for metrics in all_results.values():
        for k in ["mean_match_sim", "mean_lat_rank_diff", "edge_flag_f1",
                   "mean_role_similarity"]:
            agg[k].append(metrics[k])

    return {
        "per_camera": all_results,
        "aggregate": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                      for k, v in agg.items()},
    }


# ── Method 1: Trajectory stats only (no roles, no encoder) ──────────


def eval_traj_stats_only(dataset) -> dict:
    """5 hand-crafted stats per lane, NO role labels, NO encoder.

    Features: [mean_speed, speed_std, lateral_spread, density, mean_traj_length]
    """
    features = []
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        trajs = sample.trajectories

        if trajs:
            speeds = []
            lengths = []
            all_lat_offsets = []
            for t in trajs:
                if len(t) >= 2:
                    diffs = np.diff(t, axis=0)
                    dists = np.linalg.norm(diffs, axis=1)
                    speeds.append(dists.mean())
                    lengths.append(len(t))
                    # Lateral spread: std of x coords
                    all_lat_offsets.extend(t[:, 0].tolist())

            mean_speed = np.mean(speeds) if speeds else 0.0
            speed_std = np.std(speeds) if len(speeds) > 1 else 0.0
            lateral_spread = np.std(all_lat_offsets) if all_lat_offsets else 0.0
            density = len(trajs) / max(50, len(trajs))  # normalized count
            mean_length = np.mean(lengths) if lengths else 0.0
        else:
            mean_speed = speed_std = lateral_spread = density = mean_length = 0.0

        feat = torch.tensor([mean_speed, speed_std, lateral_spread, density, mean_length],
                            dtype=torch.float32)
        features.append(feat)

    features = torch.stack(features)
    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)

    result = _leave_one_out_eval(dataset, features)
    result["method"] = "traj-stats"
    result["feature_dim"] = 5
    result["supervision"] = "none"
    result["generalizes"] = "partial"
    return result


# ── Method 2: Stats oracle (stats + roles) ───────────────────────────


def eval_stats_oracle(dataset) -> dict:
    """traj_stats(4) + roles(5) = 9-dim. Uses ground-truth role labels."""
    features = []
    for i in range(len(dataset)):
        item = dataset[i]
        feat = torch.cat([item["traj_stats"], item["role"]])
        features.append(feat)

    features = torch.stack(features)
    features = features / (features.norm(dim=1, keepdim=True) + 1e-8)

    result = _leave_one_out_eval(dataset, features)
    result["method"] = "stats-oracle"
    result["feature_dim"] = 9
    result["supervision"] = "role labels required"
    result["generalizes"] = "NO"
    return result


# ── Method 3: Per-camera supervised classifier ───────────────────────


def eval_per_camera_supervised(dataset) -> dict:
    """Train a per-camera MLP to predict lateral_rank + edge flags from traj stats.

    For each camera: train on that camera's lanes, test on same camera (seen)
    and on other cameras (unseen) to show zero-shot failure.
    """
    results_seen = defaultdict(list)
    results_unseen = defaultdict(list)

    for test_cam in dataset.cameras:
        test_indices = dataset.get_camera_indices(test_cam)
        if len(test_indices) < 2:
            continue

        # Train on this camera's data
        X_train = torch.stack([dataset[i]["traj_stats"] for i in test_indices])
        y_rank = torch.tensor([dataset.samples[i].role.lateral_rank for i in test_indices])
        y_left = torch.tensor([float(dataset.samples[i].role.is_leftmost) for i in test_indices])
        y_right = torch.tensor([float(dataset.samples[i].role.is_rightmost) for i in test_indices])

        # Simple linear model (enough for per-camera with few lanes)
        model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(200):
            pred = model(X_train)
            loss = (
                nn.functional.mse_loss(pred[:, 0], y_rank) +
                nn.functional.binary_cross_entropy_with_logits(pred[:, 1], y_left) +
                nn.functional.binary_cross_entropy_with_logits(pred[:, 2], y_right)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            # Evaluate on same camera (seen)
            pred_seen = model(X_train)
            rank_err = (pred_seen[:, 0] - y_rank).abs().mean().item()
            pred_left = (pred_seen[:, 1] > 0).float()
            pred_right = (pred_seen[:, 2] > 0).float()

            def _f1_sup(pred, gt):
                tp = ((pred == 1) & (gt == 1)).sum().float()
                fp = ((pred == 1) & (gt == 0)).sum().float()
                fn = ((pred == 0) & (gt == 1)).sum().float()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            edge_f1 = (_f1_sup(pred_left, y_left) + _f1_sup(pred_right, y_right)) / 2.0

            results_seen["lat_rank_diff"].append(rank_err)
            results_seen["edge_flag_f1"].append(edge_f1)

            # Evaluate on other cameras (unseen)
            for other_cam in dataset.cameras:
                if other_cam == test_cam:
                    continue
                other_indices = dataset.get_camera_indices(other_cam)
                if not other_indices:
                    continue
                X_other = torch.stack([dataset[i]["traj_stats"] for i in other_indices])
                y_rank_o = torch.tensor([dataset.samples[i].role.lateral_rank for i in other_indices])
                y_left_o = torch.tensor([float(dataset.samples[i].role.is_leftmost) for i in other_indices])
                y_right_o = torch.tensor([float(dataset.samples[i].role.is_rightmost) for i in other_indices])

                pred_o = model(X_other)
                rank_err_o = (pred_o[:, 0] - y_rank_o).abs().mean().item()
                pred_left_o = (pred_o[:, 1] > 0).float()
                pred_right_o = (pred_o[:, 2] > 0).float()
                edge_f1_o = (_f1_sup(pred_left_o, y_left_o) + _f1_sup(pred_right_o, y_right_o)) / 2.0

                results_unseen["lat_rank_diff"].append(rank_err_o)
                results_unseen["edge_flag_f1"].append(edge_f1_o)

    return {
        "method": "per-camera-sup",
        "supervision": "per-camera labels",
        "generalizes": "NO",
        "seen": {
            "mean_lat_rank_diff": float(np.mean(results_seen["lat_rank_diff"])),
            "edge_flag_f1": float(np.mean(results_seen["edge_flag_f1"])),
        },
        "unseen": {
            "mean_lat_rank_diff": float(np.mean(results_unseen["lat_rank_diff"])),
            "edge_flag_f1": float(np.mean(results_unseen["edge_flag_f1"])),
        },
    }


# ── Method 4: Encoder evaluation (with or without cross-attn) ────────


def eval_encoder(dataset, checkpoint_path: str, config: dict, label: str = "encoder") -> dict:
    """Evaluate a trained encoder checkpoint via leave-one-camera-out.

    Cross-camera generalization: both references and queries encoded with
    full info (geometry + trajectories + roles). Tests whether the encoder
    produces a cross-camera aligned embedding space — a lane from a held-out
    camera should match same-role lanes from other cameras.
    """
    from torch.utils.data import Subset

    from src.data.lane_dataset import collate_fn
    from src.training.zero_shot_eval import encode_lanes, load_trained_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_config = load_trained_encoder(checkpoint_path, device)

    has_cross_attn = model.use_cross_lane_attention
    logger.info(f"  cross_lane_attention={has_cross_attn}")

    train_cfg = config.get("contrastive_training", {})
    batch_size = train_cfg.get("batch_size", 32)

    # Leave-one-camera-out: encode both ref and query with full info
    # (geometry + traj + roles are all available in deployment from
    # annotation + YOLO tracking + OSM lane ordering)
    all_results = {}
    for camera in dataset.cameras:
        held_out_indices = dataset.get_camera_indices(camera)
        if not held_out_indices:
            continue

        held_out_set = set(held_out_indices)
        train_indices = [i for i in range(len(dataset)) if i not in held_out_set]

        ref_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )
        query_loader = DataLoader(
            Subset(dataset, held_out_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )

        ref_proj, ref_roles, _ = encode_lanes(
            model, ref_loader, device, drop_geometry=False,
        )
        query_proj, query_roles, _ = encode_lanes(
            model, query_loader, device, drop_geometry=False,
        )

        sim_matrix = torch.mm(query_proj, ref_proj.t())
        best_match_idx = sim_matrix.argmax(dim=1)
        best_match_sim = sim_matrix.gather(1, best_match_idx.unsqueeze(1)).squeeze(1)

        matched_ref_roles = ref_roles[best_match_idx]

        lat_rank_diff = (query_roles[:, 0] - matched_ref_roles[:, 0]).abs()
        role_sim = torch.nn.functional.cosine_similarity(query_roles, matched_ref_roles, dim=1)

        def _f1(pred, gt):
            tp = ((pred == 1) & (gt == 1)).sum().float()
            fp = ((pred == 1) & (gt == 0)).sum().float()
            fn = ((pred == 0) & (gt == 1)).sum().float()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        left_f1 = _f1(matched_ref_roles[:, 1], query_roles[:, 1])
        right_f1 = _f1(matched_ref_roles[:, 2], query_roles[:, 2])
        edge_f1 = (left_f1 + right_f1) / 2.0

        all_results[camera] = {
            "mean_match_sim": best_match_sim.mean().item(),
            "mean_lat_rank_diff": lat_rank_diff.mean().item(),
            "edge_flag_f1": edge_f1,
            "mean_role_similarity": role_sim.mean().item(),
            "n_query": len(held_out_indices),
        }

    # Aggregate
    agg = defaultdict(list)
    for metrics in all_results.values():
        for k in ["mean_match_sim", "mean_lat_rank_diff", "edge_flag_f1",
                   "mean_role_similarity"]:
            agg[k].append(metrics[k])

    result = {
        "per_camera": all_results,
        "aggregate": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                      for k, v in agg.items()},
    }
    result["method"] = label
    result["feature_dim"] = int(query_proj.shape[1])
    result["supervision"] = "none"
    result["generalizes"] = "YES"
    result["cross_lane_attention"] = has_cross_attn
    return result


# ── Threshold-based anomaly detection for traj-stats baseline ────────


def eval_anomaly_threshold(dataset) -> float:
    """Simple threshold anomaly detection using trajectory stats deviation.

    For each lane, compute z-score of stats across windows.
    Flag as anomalous if any stat deviates > 2 std.
    Compare to synthetic anomaly labels.
    """
    from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
    from src.training.temporal_trainer import inject_anomalies

    config_path = "configs/lane_contrastive.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    mc = config.get("model", {})
    td = TemporalLaneDataset(config=config, polyline_k=mc.get("polyline_k", 16))

    loader = DataLoader(td, batch_size=16, shuffle=False, collate_fn=temporal_collate_fn)
    rng = np.random.default_rng(42)

    all_correct = 0
    all_total = 0

    for batch in loader:
        poly = batch["window_traj_polylines"]
        mask = batch["window_traj_mask"]
        stats = batch["window_traj_stats"]
        valid = batch["window_valid"]

        # Inject anomalies (same as training)
        _, _, corrupt_stats, labels = inject_anomalies(
            poly, mask, stats, valid, anomaly_ratio=0.3, rng=rng,
        )

        B, W, _ = corrupt_stats.shape
        valid_np = valid.numpy()
        labels_np = labels.numpy()

        for b in range(B):
            valid_windows = np.where(valid_np[b])[0]
            if len(valid_windows) < 3:
                continue
            window_stats = corrupt_stats[b, valid_windows].numpy()  # (V, 4)
            mean_stats = window_stats.mean(axis=0)
            std_stats = window_stats.std(axis=0) + 1e-8
            z_scores = np.abs((window_stats - mean_stats) / std_stats)
            predicted = (z_scores.max(axis=1) > 2.0).astype(float)

            for idx, w in enumerate(valid_windows):
                all_correct += int(predicted[idx] == labels_np[b, w])
                all_total += 1

    accuracy = all_correct / max(all_total, 1)
    return accuracy


# ── SVM anomaly baseline ──────────────────────────────────────────────


def eval_anomaly_svm(config: dict) -> float:
    """One-Class SVM on per-window traj_stats for anomaly detection.

    Trains on clean windows, scores corrupted windows.
    Returns accuracy at threshold=0.5.
    """
    from sklearn.svm import OneClassSVM
    from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
    from src.training.temporal_trainer import inject_anomalies

    mc = config.get("model", {})
    td = TemporalLaneDataset(config=config, polyline_k=mc.get("polyline_k", 16))
    loader = DataLoader(td, batch_size=16, shuffle=False, collate_fn=temporal_collate_fn)
    rng = np.random.default_rng(42)

    # Collect clean stats for fitting
    clean_stats_all = []
    for batch in loader:
        stats = batch["window_traj_stats"]  # (B, W, 4)
        valid = batch["window_valid"].numpy().astype(bool)
        B, W, D = stats.shape
        for b in range(B):
            for w in range(W):
                if valid[b, w]:
                    clean_stats_all.append(stats[b, w].numpy())

    if len(clean_stats_all) < 10:
        logger.warning("Not enough clean windows for SVM, skipping")
        return 0.0

    clean_stats_arr = np.stack(clean_stats_all)

    # Fit One-Class SVM on clean data
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    svm.fit(clean_stats_arr)

    # Evaluate on corrupted data
    all_correct = 0
    all_total = 0

    rng2 = np.random.default_rng(42)
    loader2 = DataLoader(td, batch_size=16, shuffle=False, collate_fn=temporal_collate_fn)

    for batch in loader2:
        poly = batch["window_traj_polylines"]
        mask = batch["window_traj_mask"]
        stats = batch["window_traj_stats"]
        valid = batch["window_valid"]

        _, _, corrupt_stats, labels = inject_anomalies(
            poly, mask, stats, valid, anomaly_ratio=0.3, rng=rng2,
        )

        B, W, _ = corrupt_stats.shape
        valid_np = valid.numpy().astype(bool)
        labels_np = labels.numpy()

        for b in range(B):
            for w in range(W):
                if not valid_np[b, w]:
                    continue
                x = corrupt_stats[b, w].numpy().reshape(1, -1)
                pred = svm.predict(x)[0]  # +1 = normal, -1 = anomaly
                predicted_label = 1.0 if pred == -1 else 0.0
                all_correct += int(predicted_label == labels_np[b, w])
                all_total += 1

    accuracy = all_correct / max(all_total, 1)
    logger.info(f"  SVM anomaly accuracy: {accuracy:.3f} (n={all_total})")
    return accuracy


# ── LSTM anomaly baseline ────────────────────────────────────────────


def eval_anomaly_lstm(config: dict) -> float:
    """LSTM baseline for anomaly detection on per-window traj_stats.

    Trains a small LSTM + linear head on the same inject_anomalies protocol
    as the GRU temporal encoder, but without any lane encoder embeddings —
    just raw 4-dim traj_stats per window.
    """
    from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
    from src.training.temporal_trainer import inject_anomalies

    mc = config.get("model", {})
    td = TemporalLaneDataset(config=config, polyline_k=mc.get("polyline_k", 16))
    loader = DataLoader(td, batch_size=16, shuffle=False, collate_fn=temporal_collate_fn)

    # Build LSTM model
    class LSTMAnomaly(nn.Module):
        def __init__(self, input_dim=4, hidden_dim=32, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                                batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x, valid):
            # x: (B, W, 4), valid: (B, W)
            h_seq, _ = self.lstm(x)  # (B, W, hidden)
            logits = self.head(h_seq).squeeze(-1)  # (B, W)
            return logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAnomaly().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for 50 epochs
    model.train()
    for epoch in range(50):
        epoch_loss = 0.0
        n_batches = 0
        rng_train = np.random.default_rng(epoch)

        for batch in loader:
            stats = batch["window_traj_stats"].to(device)
            valid = batch["window_valid"].to(device)

            poly = batch["window_traj_polylines"]
            mask_t = batch["window_traj_mask"]

            _, _, corrupt_stats, labels = inject_anomalies(
                poly, mask_t, batch["window_traj_stats"], batch["window_valid"],
                anomaly_ratio=0.3, rng=rng_train,
            )
            corrupt_stats = corrupt_stats.to(device)
            labels = labels.to(device)

            logits = model(corrupt_stats, valid)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[valid.bool()], labels[valid.bool()],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

    # Evaluate
    model.eval()
    rng_eval = np.random.default_rng(42)
    all_correct = 0
    all_total = 0

    with torch.no_grad():
        for batch in loader:
            poly = batch["window_traj_polylines"]
            mask_t = batch["window_traj_mask"]
            stats = batch["window_traj_stats"]
            valid = batch["window_valid"]

            _, _, corrupt_stats, labels = inject_anomalies(
                poly, mask_t, stats, valid, anomaly_ratio=0.3, rng=rng_eval,
            )

            logits = model(corrupt_stats.to(device), valid.to(device))
            preds = (torch.sigmoid(logits) > 0.5).float()

            valid_np = valid.numpy().astype(bool)
            preds_np = preds.cpu().numpy()
            labels_np = labels.numpy()

            for b in range(valid_np.shape[0]):
                v = valid_np[b]
                all_correct += int((preds_np[b, v] == labels_np[b, v]).sum())
                all_total += int(v.sum())

    accuracy = all_correct / max(all_total, 1)
    logger.info(f"  LSTM anomaly accuracy: {accuracy:.3f} (n={all_total})")
    return accuracy


# ── Summary table ────────────────────────────────────────────────────


def print_comparison_table(results: list):
    """Print the full comparison table."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE — Contrastive Lane Encoder (10 cameras, 91 lanes)")
    print("=" * 100)

    header = f"{'Method':<28} {'Supervision':<18} {'match_sim':>10} {'lat_diff':>10} {'edge_f1':>10} {'anomaly':>10} {'General?':>10}"
    print(header)
    print("-" * 100)

    for r in results:
        method = r.get("method", "?")
        supervision = r.get("supervision", "?")
        generalizes = r.get("generalizes", "?")

        if "aggregate" in r:
            agg = r["aggregate"]
            match_sim = f"{agg['mean_match_sim']['mean']:.3f}"
            lat_diff = f"{agg['mean_lat_rank_diff']['mean']:.3f}"
            edge_acc = f"{agg['edge_flag_f1']['mean']:.3f}"
        elif "seen" in r:
            # Per-camera supervised: show seen/unseen
            match_sim = "—"
            lat_diff = f"{r['seen']['mean_lat_rank_diff']:.3f}/{r['unseen']['mean_lat_rank_diff']:.3f}"
            edge_acc = f"{r['seen']['edge_flag_f1']:.3f}/{r['unseen']['edge_flag_f1']:.3f}"
        else:
            match_sim = lat_diff = edge_acc = "—"

        anomaly = f"{r['anomaly_acc']:.3f}" if "anomaly_acc" in r else "—"

        print(f"{method:<28} {supervision:<18} {match_sim:>10} {lat_diff:>10} {edge_acc:>10} {anomaly:>10} {generalizes:>10}")

    print()


def render_comparison_figure(results: list, output_dir: Path):
    """Render comparison table as PDF/PNG figure and save CSV."""
    import csv
    import matplotlib.pyplot as plt

    # Build rows
    col_labels = ["Method", "Supervision", "match_sim", "lat_diff", "edge_f1", "anomaly_acc", "Generalizes?"]
    rows = []
    for r in results:
        method = r.get("method", "?")
        supervision = r.get("supervision", "?")
        generalizes = r.get("generalizes", "?")

        if "aggregate" in r:
            agg = r["aggregate"]
            match_sim = f"{agg['mean_match_sim']['mean']:.3f}"
            lat_diff = f"{agg['mean_lat_rank_diff']['mean']:.3f}"
            edge_acc = f"{agg['edge_flag_f1']['mean']:.3f}"
        elif "seen" in r:
            match_sim = "—"
            lat_diff = f"{r['seen']['mean_lat_rank_diff']:.3f} / {r['unseen']['mean_lat_rank_diff']:.3f}"
            edge_acc = f"{r['seen']['edge_flag_f1']:.3f} / {r['unseen']['edge_flag_f1']:.3f}"
        else:
            match_sim = lat_diff = edge_acc = "—"

        anomaly = f"{r['anomaly_acc']:.3f}" if "anomaly_acc" in r else "—"
        rows.append([method, supervision, match_sim, lat_diff, edge_acc, anomaly, generalizes])

    # Save CSV
    csv_path = output_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(col_labels)
        writer.writerows(rows)
    logger.info(f"Saved CSV to {csv_path}")

    # Render figure
    fig, ax = plt.subplots(figsize=(14, max(2.5, len(rows) * 0.55 + 1.5)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Highlight "ours" row
    for i, r in enumerate(results):
        if "ours" in r.get("method", ""):
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor("#d4edda")
                table[i + 1, j].set_text_props(fontweight="bold")

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#343a40")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Footnotes
    footnotes = (
        "match_sim: cosine similarity to best-matched reference lane (higher = better)\n"
        "lat_diff: |query_rank - matched_rank| (lower = better)\n"
        "edge_f1: leftmost/rightmost flag F1 score\n"
        "anomaly_acc: synthetic anomaly detection accuracy\n"
        "Per-camera supervised: seen / unseen camera results\n"
        "Scale: 10 cameras, 91 lanes, leave-one-camera-out protocol"
    )
    fig.text(
        0.5, -0.01, footnotes, ha="center", va="top", fontsize=6.5,
        fontstyle="italic", transform=fig.transFigure,
    )

    ax.set_title(
        "Comparison Table — Contrastive Lane Encoder vs Baselines",
        fontsize=12, fontweight="bold", pad=15,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    path = output_dir / "comparison_table.pdf"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison figure to {path}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Baseline & ablation evaluation")
    parser.add_argument("--config", default="configs/lane_contrastive.yaml")
    parser.add_argument("--method", default=None,
                        choices=["traj-stats", "stats-oracle", "per-camera-sup",
                                 "encoder", "no-cross-attn", "svm", "lstm"],
                        help="Single method to run")
    parser.add_argument("--all", action="store_true", help="Run all methods")
    parser.add_argument("--checkpoint", default=None,
                        help="Encoder checkpoint (for encoder/no-cross-attn methods)")
    parser.add_argument("--contrastive-checkpoint", default=None,
                        help="Contrastive-only checkpoint for comparison")
    parser.add_argument("--output-dir", default="results/baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset = _load_dataset(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    methods_to_run = []
    if args.all:
        methods_to_run = ["traj-stats", "stats-oracle", "per-camera-sup", "encoder", "svm", "lstm"]
    elif args.method:
        methods_to_run = [args.method]
    else:
        methods_to_run = ["traj-stats", "stats-oracle", "per-camera-sup", "encoder", "svm", "lstm"]

    # ── Traj stats only ──
    if "traj-stats" in methods_to_run:
        logger.info("Running: traj-stats-only baseline (5-dim, no roles, no encoder)")
        result = eval_traj_stats_only(dataset)
        agg = result["aggregate"]
        logger.info(f"  match_sim={agg['mean_match_sim']['mean']:.3f}  "
                     f"lat_diff={agg['mean_lat_rank_diff']['mean']:.3f}  "
                     f"edge_f1={agg['edge_flag_f1']['mean']:.3f}")
        # Threshold anomaly detection
        logger.info("  Computing threshold-based anomaly detection...")
        result["anomaly_acc"] = eval_anomaly_threshold(dataset)
        logger.info(f"  anomaly_acc={result['anomaly_acc']:.3f}")
        all_results.append(result)

    # ── Stats oracle ──
    if "stats-oracle" in methods_to_run:
        logger.info("Running: stats-oracle baseline (9-dim, with roles)")
        result = eval_stats_oracle(dataset)
        agg = result["aggregate"]
        logger.info(f"  match_sim={agg['mean_match_sim']['mean']:.3f}  "
                     f"lat_diff={agg['mean_lat_rank_diff']['mean']:.3f}  "
                     f"edge_f1={agg['edge_flag_f1']['mean']:.3f}")
        result["anomaly_acc"] = 0.0  # Can't do anomaly detection with static features
        all_results.append(result)

    # ── Per-camera supervised ──
    if "per-camera-sup" in methods_to_run:
        logger.info("Running: per-camera supervised classifier")
        result = eval_per_camera_supervised(dataset)
        logger.info(f"  seen: lat_diff={result['seen']['mean_lat_rank_diff']:.3f}  "
                     f"edge_f1={result['seen']['edge_flag_f1']:.3f}")
        logger.info(f"  unseen: lat_diff={result['unseen']['mean_lat_rank_diff']:.3f}  "
                     f"edge_f1={result['unseen']['edge_flag_f1']:.3f}")
        all_results.append(result)

    # ── Contrastive + cross-lane attention ──
    contrastive_ckpt = args.contrastive_checkpoint or "results/lane_contrastive/checkpoints/best.pt"
    if "encoder" in methods_to_run and Path(contrastive_ckpt).exists():
        logger.info(f"Running: contrastive + cross-attn ({contrastive_ckpt})")
        result = eval_encoder(dataset, contrastive_ckpt, config,
                              label="contrastive + cross-attn")
        agg = result["aggregate"]
        logger.info(f"  match_sim={agg['mean_match_sim']['mean']:.3f}  "
                     f"lat_diff={agg['mean_lat_rank_diff']['mean']:.3f}  "
                     f"edge_f1={agg['edge_flag_f1']['mean']:.3f}")
        all_results.append(result)

    # ── Contrastive, no cross-lane attention ──
    no_crossattn_ckpt = "results/lane_contrastive_no_crossattn/checkpoints/best.pt"
    if "encoder" in methods_to_run and Path(no_crossattn_ckpt).exists():
        logger.info(f"Running: contrastive, no cross-attn ({no_crossattn_ckpt})")
        result = eval_encoder(dataset, no_crossattn_ckpt, config,
                              label="contrastive, no cross-attn")
        agg = result["aggregate"]
        logger.info(f"  match_sim={agg['mean_match_sim']['mean']:.3f}  "
                     f"lat_diff={agg['mean_lat_rank_diff']['mean']:.3f}  "
                     f"edge_f1={agg['edge_flag_f1']['mean']:.3f}")
        all_results.append(result)

    # ── Joint encoder (window-level contrastive) ──
    joint_ckpt = args.checkpoint or "results/joint_encoder/checkpoints/best.pt"
    if "encoder" in methods_to_run and Path(joint_ckpt).exists():
        logger.info(f"Running: joint, window-level contrastive ({joint_ckpt})")
        result = eval_encoder(dataset, joint_ckpt, config,
                              label="joint (ours)")
        agg = result["aggregate"]
        logger.info(f"  match_sim={agg['mean_match_sim']['mean']:.3f}  "
                     f"lat_diff={agg['mean_lat_rank_diff']['mean']:.3f}  "
                     f"edge_f1={agg['edge_flag_f1']['mean']:.3f}")
        # Add anomaly accuracy from training history
        hist_path = Path("results/joint_encoder/history.json")
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            anomaly_acc = hist.get("train/anomaly_accuracy", [])
            result["anomaly_acc"] = max(anomaly_acc) if anomaly_acc else 0.0
        all_results.append(result)

    # ── Two-stage temporal (frozen encoder + GRU) ──
    if "encoder" in methods_to_run:
        hist_path = Path("results/temporal_encoder/history.json")
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            anomaly_acc = hist.get("train/accuracy", [])
            two_stage_result = {
                "method": "two-stage (frozen)",
                "supervision": "none",
                "generalizes": "YES",
                "anomaly_acc": max(anomaly_acc) if anomaly_acc else 0.0,
            }
            all_results.append(two_stage_result)

    # ── SVM anomaly baseline ──
    if "svm" in methods_to_run:
        logger.info("Running: One-Class SVM anomaly baseline")
        svm_acc = eval_anomaly_svm(config)
        all_results.append({
            "method": "SVM (One-Class)",
            "supervision": "none",
            "generalizes": "YES",
            "anomaly_acc": svm_acc,
        })

    # ── LSTM anomaly baseline ──
    if "lstm" in methods_to_run:
        logger.info("Running: LSTM anomaly baseline (4-dim traj_stats)")
        lstm_acc = eval_anomaly_lstm(config)
        all_results.append({
            "method": "LSTM (traj_stats)",
            "supervision": "none",
            "generalizes": "YES",
            "anomaly_acc": lstm_acc,
        })

    # ── Print comparison table ──
    if all_results:
        print_comparison_table(all_results)
        render_comparison_figure(all_results, output_dir)

    # ── Save JSON ──
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "per_camera"}
        save_results.append(sr)

    out_path = output_dir / "comparison_table.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
