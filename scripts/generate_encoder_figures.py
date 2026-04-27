#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]
_CAMERA_COLORS = plt.cm.tab10.colors


def _load_eval_data(args):
    """Load encoder evaluation data from checkpoint (full recompute) or results JSON.

    Returns:
        all_results: dict mapping camera -> cross-camera eval output
        projections: (N, proj_dim) tensor or None
        roles: (N, 5) tensor or None
        cameras: list[str] or None
        lane_keys: list[str] or None
        dataset: LaneDataset or None
    """
    if args.results_json:
        with open(args.results_json) as f:
            all_results = json.load(f)
        return all_results, None, None, None, None, None

    # Full recompute from checkpoint
    from torch.utils.data import DataLoader

    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import (
        encode_lanes,
        evaluate_zero_shot,
        load_trained_encoder,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_encoder(args.checkpoint, device)

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    dataset = LaneDataset(
        config=config,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )

    # Leave-one-camera-out evaluation
    all_results = {}
    for camera in dataset.cameras:
        logger.info(f"Evaluating held-out camera: {camera}")
        metrics = evaluate_zero_shot(
            model, dataset, camera, device,
            batch_size=train_cfg.get("batch_size", 32),
        )
        if metrics:
            all_results[camera] = metrics

    # Full embedding bank (all cameras, geometry included) for E4
    full_loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
    )
    projections, roles, lane_keys = encode_lanes(
        model, full_loader, device, drop_geometry=False,
    )
    cameras = [s.camera for s in dataset.samples]

    return all_results, projections, roles, cameras, lane_keys, dataset


def figure_e2(all_results: dict, output_dir: Path):
    """Scatter plot: matched lateral rank vs ground truth across cameras.

    Each point is one lane from a held-out camera matched to its nearest
    embedding neighbor in the reference set. Points on the diagonal show
    that cross-camera alignment preserves lateral position.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    unique_cams = sorted(all_results.keys())
    cam_to_idx = {c: i for i, c in enumerate(unique_cams)}

    all_gt = []
    all_pred = []

    for cam, metrics in all_results.items():
        cidx = cam_to_idx[cam]
        color = _CAMERA_COLORS[cidx % len(_CAMERA_COLORS)]
        marker = _MARKERS[cidx % len(_MARKERS)]

        per_lane = metrics.get("per_lane", [])
        if not per_lane:
            continue

        for lane in per_lane:
            # Extract GT rank from query lane key and matched ref rank
            gt_rank = lane.get("query_lat_rank")
            pred_rank = lane.get("matched_lat_rank")

            if gt_rank is None or pred_rank is None:
                # Fall back: gt_rank = query role[0], pred_rank = matched role[0]
                # These are stored as lat_rank_diff = |gt - pred|
                # Can't reconstruct individual values from diff alone
                continue

            all_gt.append(gt_rank)
            all_pred.append(pred_rank)

        # If per_lane has the raw values, plot them
        gt_ranks = [l["query_lat_rank"] for l in per_lane if "query_lat_rank" in l]
        pred_ranks = [l["matched_lat_rank"] for l in per_lane if "matched_lat_rank" in l]

        if gt_ranks and pred_ranks:
            ax.scatter(
                gt_ranks, pred_ranks,
                c=[color], marker=marker, s=80, edgecolors="black",
                linewidths=0.5, label=cam, zorder=2,
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="perfect")
    if all_gt:
        mae = np.mean(np.abs(np.array(all_gt) - np.array(all_pred)))
        ax.text(
            0.05, 0.92, f"MAE = {mae:.3f}\nn = {len(all_gt)}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax.set_xlabel("Ground Truth Lateral Rank", fontsize=12)
    ax.set_ylabel("Matched Lateral Rank (cross-camera)", fontsize=12)
    ax.set_title("Cross-Camera Lateral Rank Alignment", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "E2_lateral_rank.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved E2 to {path}")


def figure_e3(all_results: dict, output_dir: Path):
    """Confusion matrices for leftmost/rightmost edge prediction.

    Also shows per-camera bar chart of edge flag accuracy.
    """
    gt_leftmost, pred_leftmost = [], []
    gt_rightmost, pred_rightmost = [], []
    per_camera_acc = {}

    cameras = sorted(all_results.keys()) if all_results else []

    for cam in cameras:
        per_lane = all_results.get(cam, {}).get("per_lane", [])

        cam_gt_left, cam_pr_left = [], []
        cam_gt_right, cam_pr_right = [], []

        if per_lane and "query_is_leftmost" in per_lane[0]:
            for l in per_lane:
                gt_l = l["query_is_leftmost"]
                gt_r = l["query_is_rightmost"]
                pr_l = l["matched_is_leftmost"]
                pr_r = l["matched_is_rightmost"]
                gt_leftmost.append(int(gt_l))
                pred_leftmost.append(int(pr_l))
                gt_rightmost.append(int(gt_r))
                pred_rightmost.append(int(pr_r))
                cam_gt_left.append(int(gt_l))
                cam_pr_left.append(int(pr_l))
                cam_gt_right.append(int(gt_r))
                cam_pr_right.append(int(pr_r))
        elif cam in all_results:
            per_camera_acc[cam] = all_results[cam].get("edge_flag_accuracy", None)
            continue

        # Compute macro-averaged F1 across leftmost + rightmost
        if cam_gt_left:
            f1s = []
            for gt_arr, pr_arr in [(cam_gt_left, cam_pr_left), (cam_gt_right, cam_pr_right)]:
                gt_a, pr_a = np.array(gt_arr), np.array(pr_arr)
                tp = int(((gt_a == 1) & (pr_a == 1)).sum())
                fp = int(((gt_a == 0) & (pr_a == 1)).sum())
                fn = int(((gt_a == 1) & (pr_a == 0)).sum())
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
            per_camera_acc[cam] = np.mean(f1s)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Subplot 1: Leftmost confusion matrix
    _plot_confusion(
        axes[0], gt_leftmost, pred_leftmost,
        title="Leftmost Flag", labels=["Not Left", "Leftmost"],
    )

    # Subplot 2: Rightmost confusion matrix
    _plot_confusion(
        axes[1], gt_rightmost, pred_rightmost,
        title="Rightmost Flag", labels=["Not Right", "Rightmost"],
    )

    # Subplot 3: Per-camera edge F1 bar chart
    if per_camera_acc:
        cams = sorted(per_camera_acc.keys())
        f1s = [per_camera_acc[c] for c in cams]
        bars = axes[2].barh(cams, f1s, color=plt.cm.tab10.colors[:len(cams)], edgecolor="black")
        axes[2].set_xlim(0, 1.05)
        axes[2].set_xlabel("Edge Flag F1 (macro avg)", fontsize=11)
        axes[2].set_title("Per-Camera Edge F1", fontsize=12)
        for bar, f1 in zip(bars, f1s):
            if f1 is not None:
                axes[2].text(
                    f1 + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{f1:.2f}", va="center", fontsize=9,
                )
        axes[2].axvline(1.0, color="green", linestyle="--", alpha=0.5)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)

    fig.suptitle("Edge Prediction (Lane Adjacency)", fontsize=14, y=1.02)
    fig.tight_layout()

    path = output_dir / "E3_edge_prediction.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved E3 to {path}")


def _plot_confusion(ax, gt, pred, title, labels):
    """Plot a 2x2 confusion matrix on the given axes."""
    if not gt:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    gt = np.array(gt)
    pred = np.array(pred)

    # 2x2 confusion matrix
    tp = int(((gt == 1) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)
    ax.set_title(title, fontsize=12)

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = 100 * val / total if total > 0 else 0
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    # Precision / Recall / F1 annotation (for the positive class)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    ax.text(
        0.5, -0.15,
        f"P={precision:.1%}  R={recall:.1%}  F1={f1:.1%}  (n={total})",
        transform=ax.transAxes, ha="center", fontsize=9,
    )


def figure_e4(
    projections: torch.Tensor,
    roles: torch.Tensor,
    cameras: list,
    lane_keys: list,
    dataset,
    output_dir: Path,
):
    """Box/violin plot: cosine similarity within-group vs across-group vs same-rank-cross-camera.

    Three categories:
    1. Same group: lanes from the same (camera, group_id) -- siblings
    2. Different group, same camera: different groups within same camera
    3. Cross-camera, same lateral rank: matching rank from different cameras
    """
    N = len(projections)
    sim_matrix = (projections @ projections.T).numpy()

    # Build metadata arrays
    group_ids = np.array([dataset.samples[i].group_id for i in range(N)])
    cam_arr = np.array(cameras)
    lat_ranks = roles[:, 0].numpy()

    # Sample pairs (computing all N^2 is fine for <200 lanes)
    same_group_sims = []
    diff_group_same_cam_sims = []
    same_rank_cross_cam_sims = []
    diff_rank_cross_cam_sims = []

    for i in range(N):
        for j in range(i + 1, N):
            s = sim_matrix[i, j]
            same_cam = cam_arr[i] == cam_arr[j]
            same_grp = same_cam and group_ids[i] == group_ids[j]
            rank_close = abs(lat_ranks[i] - lat_ranks[j]) < 0.15

            if same_grp:
                same_group_sims.append(s)
            elif same_cam:
                diff_group_same_cam_sims.append(s)
            elif rank_close:
                same_rank_cross_cam_sims.append(s)
            else:
                diff_rank_cross_cam_sims.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Subplot 1: Box plot of similarity categories
    ax = axes[0]
    categories = [
        ("Same group\n(siblings)", same_group_sims),
        ("Diff group\n(same camera)", diff_group_same_cam_sims),
        ("Same rank\n(cross-camera)", same_rank_cross_cam_sims),
        ("Diff rank\n(cross-camera)", diff_rank_cross_cam_sims),
    ]

    data = [sims for _, sims in categories]
    labels = [name for name, _ in categories]
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#95a5a6"]

    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, widths=0.6,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points (jittered)
    for i, sims in enumerate(data):
        if sims:
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sims))
            ax.scatter(
                np.full(len(sims), i + 1) + jitter, sims,
                c=colors[i], s=8, alpha=0.4, edgecolors="none", zorder=1,
            )

    # Summary stats
    for i, sims in enumerate(data):
        if sims:
            med = np.median(sims)
            ax.text(
                i + 1, ax.get_ylim()[1] - 0.02,
                f"n={len(sims)}\nmed={med:.2f}",
                ha="center", va="top", fontsize=7,
            )

    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Cross-Camera Alignment by Pair Type", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Subplot 2: Similarity heatmap (sorted by camera then group)
    ax2 = axes[1]

    # Sort by (camera, group_id, cls_id) for block structure
    sample_metadata = [
        (cam_arr[i], group_ids[i], dataset.samples[i].cls_id, i)
        for i in range(N)
    ]
    sample_metadata.sort()
    order = [m[3] for m in sample_metadata]

    sim_sorted = sim_matrix[np.ix_(order, order)]

    im = ax2.imshow(sim_sorted, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

    # Camera boundary lines
    sorted_cams = [cam_arr[i] for i in order]
    boundaries = []
    cam_ticks = []
    prev_cam = None
    for idx, cam in enumerate(sorted_cams):
        if cam != prev_cam:
            if prev_cam is not None:
                boundaries.append(idx)
            cam_ticks.append((idx, cam))
            prev_cam = cam

    for b in boundaries:
        ax2.axhline(b - 0.5, color="white", linewidth=1.5)
        ax2.axvline(b - 0.5, color="white", linewidth=1.5)

    # Group boundaries (thinner)
    sorted_groups = [(cam_arr[i], group_ids[i]) for i in order]
    prev_cg = None
    for idx, cg in enumerate(sorted_groups):
        if cg != prev_cg and prev_cg is not None and cg[0] == prev_cg[0]:
            ax2.axhline(idx - 0.5, color="white", linewidth=0.5, alpha=0.6)
            ax2.axvline(idx - 0.5, color="white", linewidth=0.5, alpha=0.6)
        prev_cg = cg

    tick_positions = []
    tick_labels = []
    for start_idx, cam in cam_ticks:
        end_idx = len(sorted_cams)
        for b in boundaries:
            if b > start_idx:
                end_idx = b
                break
        tick_positions.append((start_idx + end_idx) / 2.0)
        tick_labels.append(cam)

    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=7)

    plt.colorbar(im, ax=ax2, label="Cosine Similarity", shrink=0.8)
    ax2.set_title("Similarity Matrix (by camera/group)", fontsize=12)

    fig.suptitle("Cross-Camera Behavioral Alignment", fontsize=14, y=1.02)
    fig.tight_layout()

    path = output_dir / "E4_embedding_similarity.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved E4 to {path}")


def figure_e4_from_results(all_results: dict, output_dir: Path):
    """Simplified E4 using only match similarity from results JSON.

    Shows per-camera match similarity distribution as a box plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    cameras = sorted(all_results.keys())
    data = []
    labels = []

    for cam in cameras:
        per_lane = all_results[cam].get("per_lane", [])
        sims = [l["cosine_sim"] for l in per_lane if "cosine_sim" in l]
        if sims:
            data.append(sims)
            labels.append(cam)

    if data:
        bp = ax.boxplot(
            data, tick_labels=labels, patch_artist=True, widths=0.6,
            showfliers=True,
        )
        colors = [_CAMERA_COLORS[i % len(_CAMERA_COLORS)] for i in range(len(data))]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Cosine Similarity to Best Match", fontsize=11)
        ax.set_title("Match Similarity per Held-Out Camera", fontsize=12)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = output_dir / "E4_match_similarity.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved E4 (match sim only) to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate encoder evaluation figures")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to trained encoder checkpoint (recomputes all metrics)",
    )
    parser.add_argument(
        "--results-json", default=None,
        help="Path to pre-saved cross-camera results JSON (skip recomputation)",
    )
    parser.add_argument(
        "--output-dir", default="results/joint_encoder/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figures", nargs="+", default=["E2", "E4"],
        choices=["E2", "E3", "E4"],
        help="Which figures to generate (default: E2, E4). E3 retained for backward compat.",
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.results_json:
        parser.error("Must provide either --checkpoint or --results-json")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results, projections, roles, cameras, lane_keys, dataset = _load_eval_data(args)

    if "E2" in args.figures:
        logger.info("Generating E2: Lateral Rank Prediction...")
        figure_e2(all_results, output_dir)

    if "E3" in args.figures:
        logger.info("Generating E3: Edge Prediction...")
        figure_e3(all_results, output_dir)

    if "E4" in args.figures:
        if projections is not None and dataset is not None:
            logger.info("Generating E4: Embedding Similarity (full)...")
            figure_e4(projections, roles, cameras, lane_keys, dataset, output_dir)
        elif all_results:
            logger.info("Generating E4: Embedding Similarity (from results JSON)...")
            figure_e4_from_results(all_results, output_dir)
        else:
            logger.warning("Skipping E4: no embeddings or results available")

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
