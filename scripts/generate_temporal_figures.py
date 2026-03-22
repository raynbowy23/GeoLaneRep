#!/usr/bin/env python3
"""Generate temporal encoder figures for the paper.

Figures:
    T1 — Window size comparison: anomaly detection at 5/15/30-sec windows
    T2 — Anomaly score timeline with injected incident overlay
    T3 — Embedding trajectory (UMAP): single lane traced through incident
    T4 — Embedding delta heatmap ||e(t) - e(t-1)|| per lane per window

Legacy aliases: 2a=T3(UMAP), 2b=T2(timeline), 2c=T4(heatmap)

Usage (two-stage):
    python scripts/generate_temporal_figures.py \
        --config configs/lane_contrastive.yaml \
        --checkpoint results/temporal_encoder/checkpoints/best.pt \
        --encoder-checkpoint results/lane_contrastive/checkpoints/best.pt

Usage (joint):
    python scripts/generate_temporal_figures.py \
        --config configs/lane_contrastive.yaml \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --joint
"""

import sys
import warnings
from pathlib import Path

# Suppress UMAP/numba warnings
warnings.filterwarnings("ignore", message="n_jobs value")
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_model_and_data(args):
    """Load temporal or joint model and dataset.

    Supports two modes:
        --joint: loads JointLaneEncoder from a single checkpoint
        (default): loads LaneTemporalEncoder from temporal + encoder checkpoints
    """
    import yaml

    from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
    from src.models.lane_encoder import LaneEncoder
    from src.models.temporal_encoder import LaneTemporalEncoder
    from src.training.temporal_trainer import inject_anomalies

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(args, "joint", False):
        # --- Joint mode: single checkpoint contains everything ---
        from src.models.joint_encoder import JointLaneEncoder

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", config)
        mc = ckpt_config.get("model", {})
        tc = ckpt_config.get("temporal", {})

        lane_encoder = LaneEncoder(
            polyline_k=mc.get("polyline_k", 16),
            d_model=mc.get("polyline_encoder_dim", 64),
            embed_dim=mc.get("embed_dim", 128),
            proj_dim=mc.get("proj_dim", 64),
            polyline_mode=mc.get("polyline_encoder", "transformer"),
            polyline_layers=mc.get("polyline_encoder_layers", 2),
            polyline_heads=mc.get("polyline_encoder_heads", 4),
            stats_dim=mc.get("stats_dim", 9),
            geometry_dropout=0.0,
            dropout=mc.get("dropout", 0.1),
            use_cross_lane_attention=False,
        )

        model = JointLaneEncoder(
            lane_encoder=lane_encoder,
            embed_dim=mc.get("embed_dim", 128),
            gru_layers=tc.get("gru_layers", 1),
            dropout=tc.get("dropout", 0.1),
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info("Loaded JointLaneEncoder from checkpoint")

    else:
        # --- Two-stage mode: separate encoder + temporal checkpoints ---
        enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
        enc_config = enc_ckpt.get("config", config)
        mc = enc_config.get("model", {})

        lane_encoder = LaneEncoder(
            polyline_k=mc.get("polyline_k", 16),
            d_model=mc.get("polyline_encoder_dim", 64),
            embed_dim=mc.get("embed_dim", 128),
            proj_dim=mc.get("proj_dim", 64),
            polyline_mode=mc.get("polyline_encoder", "transformer"),
            polyline_layers=mc.get("polyline_encoder_layers", 2),
            polyline_heads=mc.get("polyline_encoder_heads", 4),
            stats_dim=mc.get("stats_dim", 9),
            geometry_dropout=0.0,
            dropout=mc.get("dropout", 0.1),
            use_cross_lane_attention=False,
        )
        lane_encoder.load_state_dict(enc_ckpt["model_state_dict"], strict=False)

        tc = config.get("temporal", {})
        embed_dim = mc.get("embed_dim", 128)

        model = LaneTemporalEncoder(
            lane_encoder=lane_encoder,
            embed_dim=embed_dim,
            freeze_encoder=True,
            gru_layers=tc.get("gru_layers", 1),
            dropout=tc.get("dropout", 0.1),
        ).to(device)

        temporal_ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(temporal_ckpt["model_state_dict"])
        model.eval()
        logger.info("Loaded LaneTemporalEncoder (two-stage) from checkpoint")

    mc = config.get("model", {})

    # Dataset
    dataset = TemporalLaneDataset(
        config=config,
        polyline_k=mc.get("polyline_k", 16),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get("temporal", {}).get("batch_size", 16),
        shuffle=False,
        collate_fn=temporal_collate_fn,
    )

    return model, dataset, loader, config, device


# ── Figure T1: Window size comparison ────────────────────────────────


def figure_t1(model, config, device, output_dir: Path):
    """Window size comparison: anomaly detection accuracy at 1/2/5/10/15-min windows.

    Re-creates the temporal dataset with each window size, injects anomalies,
    and measures detection accuracy. Uses real operational time (no normalization).
    Shows how temporal resolution affects the model's ability to detect
    behavioral shifts.
    """
    import copy
    from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
    from src.training.temporal_trainer import inject_anomalies

    window_sizes = [60.0, 120.0, 300.0, 600.0, 900.0]  # 1, 2, 5, 10, 15 min
    rng = np.random.default_rng(42)
    mc = config.get("model", {})

    results = {}  # window_size -> {accuracy, precision, recall, f1, scores}

    for ws in window_sizes:
        # Build dataset with this window size
        cfg = copy.deepcopy(config)
        cfg.setdefault("temporal", {})["window_size_sec"] = ws
        # Keep stride at half the window size for 50% overlap
        cfg["temporal"]["window_stride_sec"] = ws / 2.0
        # Use real 30-min observation cap
        cfg["temporal"]["total_time_sec"] = 1800.0

        try:
            dataset = TemporalLaneDataset(
                config=cfg,
                polyline_k=mc.get("polyline_k", 16),
            )
        except Exception as e:
            logger.warning(f"Failed to create dataset with window_size={ws}: {e}")
            continue

        if len(dataset) == 0:
            logger.warning(f"Empty dataset for window_size={ws}, skipping")
            continue

        loader = DataLoader(
            dataset,
            batch_size=min(len(dataset), config.get("temporal", {}).get("batch_size", 16)),
            shuffle=False,
            collate_fn=temporal_collate_fn,
        )

        all_preds = []
        all_labels = []
        all_scores_clean = []
        all_scores_anom = []

        with torch.no_grad():
            for batch in loader:
                geometry = batch["geometry"].to(device)
                poly = batch["window_traj_polylines"].to(device)
                mask = batch["window_traj_mask"].to(device)
                stats = batch["window_traj_stats"].to(device)
                valid = batch["window_valid"].to(device)
                roles = batch["roles"].to(device)

                # Clean scores
                clean_out = model(geometry, poly, mask, stats, valid, roles)
                clean_scores = torch.sigmoid(clean_out["anomaly_scores"])

                # Inject anomalies
                c_poly, c_mask, c_stats, labels = inject_anomalies(
                    poly, mask, stats, valid, anomaly_ratio=0.3,
                    rng=np.random.default_rng(42),
                )
                anom_out = model(geometry, c_poly, c_mask, c_stats, valid, roles)
                anom_scores = torch.sigmoid(anom_out["anomaly_scores"])

                valid_np = valid.cpu().numpy().astype(bool)
                labels_np = labels.cpu().numpy()

                for b in range(valid_np.shape[0]):
                    v = valid_np[b]
                    all_labels.extend(labels_np[b, v].tolist())
                    all_scores_clean.extend(clean_scores[b, v].cpu().tolist())
                    all_scores_anom.extend(anom_scores[b, v].cpu().tolist())

        if not all_labels:
            continue

        scores_arr = np.array(all_scores_anom)
        labels_arr = np.array(all_labels)

        # Find optimal threshold via Youden's J
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]

        preds = (scores_arr > best_thresh).astype(float)

        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()
        tn = ((preds == 0) & (labels_arr == 0)).sum()

        acc = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        results[ws] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "threshold": float(best_thresh),
            "n_windows": len(all_labels),
            "mean_clean": np.mean(all_scores_clean),
            "mean_anom": np.mean(all_scores_anom),
        }
        logger.info(
            f"  Window {ws / 60:.1f}min: acc={acc:.3f} prec={prec:.3f} "
            f"rec={rec:.3f} f1={f1:.3f} t*={best_thresh:.2f} (n={len(all_labels)})"
        )

    if not results:
        logger.warning("No results for T1, skipping")
        return

    # Grouped bar chart (discrete window sizes, no interpolation)
    fig, ax = plt.subplots(figsize=(8, 5))

    ws_list = sorted(results.keys())
    accs = [results[w]["accuracy"] for w in ws_list]
    f1s = [results[w]["f1"] for w in ws_list]
    precs = [results[w]["precision"] for w in ws_list]
    recs = [results[w]["recall"] for w in ws_list]

    x = np.arange(len(ws_list))
    width = 0.2
    ax.bar(x - 1.5 * width, accs, width, label="Accuracy", color="#2196F3", edgecolor="black")
    ax.bar(x - 0.5 * width, f1s, width, label="F1 Score", color="#F44336", edgecolor="black")
    ax.bar(x + 0.5 * width, precs, width, label="Precision", color="#4CAF50", edgecolor="black")
    ax.bar(x + 1.5 * width, recs, width, label="Recall", color="#9C27B0", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{w / 60:.0f} min\n(t*={results[w]['threshold']:.2f})" for w in ws_list
    ])
    ax.set_xlabel("Window Size (optimal threshold)")
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Detection vs Operational Window Size")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    path = output_dir / "T1_window_size_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save raw metrics
    thresholds = [results[w]["threshold"] for w in ws_list]
    np.savez(
        str(output_dir / "T1_metrics.npz"),
        window_sizes=np.array(ws_list),
        accuracies=np.array(accs),
        f1_scores=np.array(f1s),
        precisions=np.array(precs),
        recalls=np.array(recs),
        thresholds=np.array(thresholds),
    )
    logger.info(f"Saved T1 to {path}")

    # Return best window results for T3 (ROC + confusion matrix)
    best_ws = max(results, key=lambda w: results[w]["f1"])
    return results.get(best_ws, {})


# ── Figure T2: Anomaly score timeline ────────────────────────────────


def _classify_lane_role(roles_vec) -> str:
    """Classify a lane's role from its role vector into a human-readable label.

    Role vector: [lateral_rank, is_leftmost, is_rightmost, has_successor, group_size, ...]
    """
    lat_rank = roles_vec[0]
    is_left = roles_vec[1] > 0.5
    is_right = roles_vec[2] > 0.5
    has_succ = roles_vec[3] > 0.5

    if has_succ and is_right:
        return "Merge (rightmost)"
    if has_succ:
        return "Merge"
    if is_left:
        return "Leftmost"
    if is_right:
        return "Rightmost"
    if 0.35 <= lat_rank <= 0.65:
        return "Middle"
    if lat_rank < 0.35:
        return "Inner"
    return "Outer"


def figure_t2(model, loader, device, config, output_dir: Path):
    """Anomaly score timeline with injected incident overlay.

    Shows one lane per structural role (leftmost, rightmost, middle, merge),
    comparing clean vs anomaly-injected scores over time.
    """
    from src.training.temporal_trainer import inject_anomalies

    tc = config.get("temporal", {})
    window_stride = tc.get("window_stride_sec", 5.0)
    rng = np.random.default_rng(42)

    # Collect all batches to find one lane per role type
    all_clean = []
    all_anom = []
    all_labels = []
    all_valid = []
    all_roles = []
    all_keys = []

    with torch.no_grad():
        for batch in loader:
            geometry = batch["geometry"].to(device)
            poly = batch["window_traj_polylines"].to(device)
            mask = batch["window_traj_mask"].to(device)
            stats = batch["window_traj_stats"].to(device)
            valid = batch["window_valid"].to(device)
            roles = batch["roles"].to(device)

            clean_out = model(geometry, poly, mask, stats, valid, roles)
            clean_scores = torch.sigmoid(clean_out["anomaly_scores"]).cpu().numpy()

            c_poly, c_mask, c_stats, labels = inject_anomalies(
                poly, mask, stats, valid, anomaly_ratio=0.3, rng=rng,
            )
            anom_out = model(geometry, c_poly, c_mask, c_stats, valid, roles)
            anom_scores = torch.sigmoid(anom_out["anomaly_scores"]).cpu().numpy()

            B = clean_scores.shape[0]
            for b in range(B):
                all_clean.append(clean_scores[b])
                all_anom.append(anom_scores[b])
                all_labels.append(labels[b].cpu().numpy())
                all_valid.append(batch["window_valid"][b].numpy())
                all_roles.append(batch["roles"][b].numpy())
                all_keys.append(batch["lane_keys"][b])

    if not all_clean:
        logger.warning("No data for T2")
        return

    W = all_clean[0].shape[0]
    time_axis = np.arange(W) * window_stride / 60.0  # minutes

    # Classify each lane and pick one per role (prefer lowest clean baseline)
    target_roles = ["Leftmost", "Rightmost", "Middle", "Merge"]
    role_candidates = {r: [] for r in target_roles}

    for i, roles_vec in enumerate(all_roles):
        v = all_valid[i].astype(bool)
        if v.sum() == 0:
            continue
        label = _classify_lane_role(roles_vec)
        clean_mean = all_clean[i][v].mean()
        # Match to target roles
        for target in target_roles:
            if label.startswith(target) or label == target:
                role_candidates[target].append((clean_mean, i))
                break

    # Pick best candidate per role (lowest clean baseline = clearest signal)
    show_items = []  # (index, role_label)
    for target in target_roles:
        cands = role_candidates[target]
        if cands:
            cands.sort(key=lambda x: x[0])
            show_items.append((cands[0][1], target))

    if not show_items:
        logger.warning("Could not find lanes for any target role, falling back")
        show_items = [(0, "Lane")]

    n_show = len(show_items)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for plot_idx, (i, role_label) in enumerate(show_items):
        ax = axes[plot_idx]
        v = all_valid[i].astype(bool)

        ax.plot(time_axis[v], all_clean[i][v],
                "b-o", markersize=4, label="Clean", alpha=0.7)
        ax.plot(time_axis[v], all_anom[i][v],
                "r-s", markersize=4, label="With anomalies", alpha=0.7)

        stride_min = window_stride / 60.0
        for w in range(W):
            if all_labels[i][w] > 0.5:
                ax.axvspan(
                    time_axis[w] - stride_min / 2,
                    time_axis[w] + stride_min / 2,
                    alpha=0.2, color="red",
                )

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"{role_label} Lane", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if plot_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle("Anomaly Score Timeline by Lane Role", fontsize=13, y=1.01)
    fig.tight_layout()

    path = output_dir / "T2_anomaly_timeline.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved T2 to {path}")


# ── Figure T3: ROC curve + confusion matrix ──────────────────────────


def figure_t3(model, loader, device, config, output_dir: Path):
    """ROC curve and confusion matrix at best threshold.

    Left: ROC curve with AUROC.
    Right: Confusion matrix at threshold=0.5.
    """
    from sklearn.metrics import roc_curve, auc
    from src.training.temporal_trainer import inject_anomalies

    rng = np.random.default_rng(42)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            geometry = batch["geometry"].to(device)
            poly = batch["window_traj_polylines"].to(device)
            mask = batch["window_traj_mask"].to(device)
            stats = batch["window_traj_stats"].to(device)
            valid = batch["window_valid"].to(device)
            roles = batch["roles"].to(device)

            corrupt_poly, corrupt_mask, corrupt_stats, labels = inject_anomalies(
                poly, mask, stats, valid, anomaly_ratio=0.3, rng=rng,
            )
            anom_out = model(geometry, corrupt_poly, corrupt_mask, corrupt_stats, valid, roles)
            scores = torch.sigmoid(anom_out["anomaly_scores"])

            valid_np = valid.cpu().numpy().astype(bool)
            labels_np = labels.cpu().numpy()
            scores_np = scores.cpu().numpy()

            for b in range(valid_np.shape[0]):
                v = valid_np[b]
                all_scores.extend(scores_np[b, v].tolist())
                all_labels.extend(labels_np[b, v].tolist())

    if not all_labels:
        logger.warning("No data for T3")
        return

    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ROC curve
    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)

    # Mark optimal threshold (Youden's J = max(TPR - FPR))
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    ax1.scatter(fpr[best_idx], tpr[best_idx], c="red", s=100, zorder=5,
                edgecolors="black", linewidths=1.5,
                label=f"Optimal (t={best_thresh:.2f})")

    # Mark threshold=0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax1.scatter(fpr[idx_05], tpr[idx_05], c="orange", s=80, zorder=5,
                marker="D", edgecolors="black", linewidths=1.5,
                label="t=0.5")

    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Right: Confusion matrix at optimal threshold
    preds = (scores_arr > best_thresh).astype(int)
    tp = int(((preds == 1) & (labels_arr == 1)).sum())
    fp = int(((preds == 1) & (labels_arr == 0)).sum())
    fn = int(((preds == 0) & (labels_arr == 1)).sum())
    tn = int(((preds == 0) & (labels_arr == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    im = ax2.imshow(cm, cmap="Blues", vmin=0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Normal", "Anomaly"], fontsize=10)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Normal", "Anomaly"], fontsize=10)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("Ground Truth", fontsize=11)
    ax2.set_title(f"Confusion Matrix (threshold={best_thresh:.2f})")

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = 100 * val / total if total > 0 else 0
            color = "white" if val > cm.max() / 2 else "black"
            ax2.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                     fontsize=12, fontweight="bold", color=color)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    ax2.text(
        0.5, -0.15,
        f"P={precision:.1%}  R={recall:.1%}  F1={f1:.1%}  (n={total})",
        transform=ax2.transAxes, ha="center", fontsize=10,
    )

    fig.suptitle("Anomaly Detection Performance", fontsize=14)
    fig.tight_layout()

    path = output_dir / "T3_roc_confusion.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved T3 to {path}")


# ── Figure T4: Embedding delta heatmap ────────────────────────────────


def figure_t4(model, loader, device, config, output_dir: Path):
    """Embedding delta heatmap: ||e(t) - e(t-1)|| per lane per window.

    Rows = lanes (sorted by mean delta, highest first), columns = window
    transitions. Bright bands highlight lanes/windows with large embedding
    shifts, localizing temporal instability. Red border marks injected
    anomaly windows for ground truth comparison.
    """
    from src.training.temporal_trainer import inject_anomalies

    tc = config.get("temporal", {})
    window_stride = tc.get("window_stride_sec", 5.0)
    rng = np.random.default_rng(42)

    all_deltas = []
    all_labels = []  # anomaly labels per lane per window
    all_role_labels = []

    with torch.no_grad():
        for batch in loader:
            geometry = batch["geometry"].to(device)
            poly = batch["window_traj_polylines"].to(device)
            mask = batch["window_traj_mask"].to(device)
            stats = batch["window_traj_stats"].to(device)
            valid = batch["window_valid"].to(device)
            roles = batch["roles"].to(device)

            # Inject anomalies (same protocol as training)
            c_poly, c_mask, c_stats, labels = inject_anomalies(
                poly, mask, stats, valid, anomaly_ratio=0.3, rng=rng,
            )

            # Encode corrupted data
            output = model(geometry, c_poly, c_mask, c_stats, valid, roles)
            emb = output["window_embeddings"].cpu()  # (B, W, D)
            valid_np = batch["window_valid"].cpu().numpy().astype(bool)
            labels_np = labels.cpu().numpy()
            B, W, D = emb.shape

            # Compute ||e(t) - e(t-1)|| for consecutive windows
            deltas = torch.norm(emb[:, 1:] - emb[:, :-1], dim=-1)  # (B, W-1)

            for b in range(B):
                v = valid_np[b]
                trans_valid = v[:-1] & v[1:]
                delta_row = deltas[b].numpy().copy()
                delta_row[~trans_valid] = 0.0
                all_deltas.append(delta_row)
                all_labels.append(labels_np[b])  # (W,)
                role_label = _classify_lane_role(batch["roles"][b].numpy())
                all_role_labels.append(role_label)

    if not all_deltas:
        logger.warning("No data for T4")
        return

    delta_matrix = np.stack(all_deltas)  # (N_lanes, W-1)
    label_matrix = np.stack(all_labels)  # (N_lanes, W)
    W_minus_1 = delta_matrix.shape[1]

    # Sort by mean delta for visual clarity (highest instability at top)
    sort_idx = np.argsort(delta_matrix.mean(axis=1))[::-1]
    delta_matrix = delta_matrix[sort_idx]
    label_matrix = label_matrix[sort_idx]
    sorted_labels = [all_role_labels[i] for i in sort_idx]

    # Deduplicate labels by appending index within same role
    label_counts = {}
    display_labels = []
    for lbl in sorted_labels:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        display_labels.append(f"{lbl} #{label_counts[lbl]}")

    # Limit to top 30 lanes for readability
    n_show = min(30, len(delta_matrix))
    delta_matrix = delta_matrix[:n_show]
    label_matrix = label_matrix[:n_show]
    display_labels = display_labels[:n_show]

    fig, ax = plt.subplots(figsize=(10, max(4, n_show * 0.25)))

    transition_labels = [
        f"{i * window_stride / 60:.1f}\u2192{(i + 1) * window_stride / 60:.1f}m"
        for i in range(W_minus_1)
    ]

    im = ax.imshow(delta_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="||e(t) - e(t-1)||")

    # Overlay anomaly injection markers: red border on cells where either
    # adjacent window was injected with an anomaly
    for row in range(n_show):
        for col in range(W_minus_1):
            # Transition col spans window col → col+1
            if label_matrix[row, col] > 0.5 or label_matrix[row, col + 1] > 0.5:
                rect = plt.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    linewidth=1.5, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)

    ax.set_xticks(range(W_minus_1))
    ax.set_xticklabels(transition_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(display_labels, fontsize=6)

    ax.set_xlabel("Window Transition")
    ax.set_ylabel("Lane")
    ax.set_title("Embedding Delta Heatmap (red border = injected anomaly)")
    fig.tight_layout()

    path = output_dir / "T4_embedding_delta_heatmap.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved T4 to {path}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate temporal encoder figures")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to temporal/joint encoder checkpoint",
    )
    parser.add_argument(
        "--encoder-checkpoint", default=None,
        help="Path to pre-trained LaneEncoder checkpoint (required for two-stage, ignored for --joint)",
    )
    parser.add_argument(
        "--joint", action="store_true",
        help="Load as JointLaneEncoder (single checkpoint) instead of two-stage",
    )
    parser.add_argument(
        "--output-dir", default="results/joint_encoder/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figures", nargs="+", default=["T1", "T2", "T3", "T4"],
        choices=["T1", "T2", "T3", "T4"],
        help="Which figures to generate. T1=window sweep, T2=anomaly timeline, T3=ROC+confusion, T4=embedding delta heatmap.",
    )
    args = parser.parse_args()

    if not args.joint and not args.encoder_checkpoint:
        parser.error("--encoder-checkpoint is required for two-stage mode (use --joint for joint checkpoint)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, dataset, loader, config, device = _load_model_and_data(args)

    if "T1" in args.figures:
        logger.info("Generating T1: Window size comparison...")
        figure_t1(model, config, device, output_dir)

    if "T2" in args.figures:
        logger.info("Generating T2: Anomaly score timeline...")
        figure_t2(model, loader, device, config, output_dir)

    if "T3" in args.figures:
        logger.info("Generating T3: ROC curve + confusion matrix...")
        figure_t3(model, loader, device, config, output_dir)

    if "T4" in args.figures:
        logger.info("Generating T4: Embedding delta heatmap...")
        figure_t4(model, loader, device, config, output_dir)

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
