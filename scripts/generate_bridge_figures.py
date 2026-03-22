#!/usr/bin/env python3
"""Generate CrossTraffic bridge figures (C-series).

Reframed around: "The encoder embedding space preserves traffic metric
structure — lanes with similar behavioral embeddings have similar speed,
density, and LOS."

Figures:
    C1 — Embedding similarity vs traffic metric distance: lanes with
         higher cosine similarity have more similar speed and density.
    C2 — LOS agreement by embedding proximity: nearby embeddings predict
         same or adjacent LOS grade.
    C3 — Per-camera summary heatmap of embedding-metric correlation.

Usage:
    python scripts/generate_bridge_figures.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --figures C1 C2 C3
"""

import sys
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data(args):
    """Load encoder, encode all lanes, return embeddings + dataset."""
    import yaml
    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import load_trained_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_encoder(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    dataset = LaneDataset(
        config=config,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )

    # Encode all lanes
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        stats_input = torch.cat([
            batch["traj_stats"].to(device),
            batch["roles"].to(device),
        ], dim=-1)
        output = model(
            geometry=batch["geometry"].to(device),
            traj_polylines=batch["traj_polylines"].to(device),
            traj_mask=batch["traj_mask"].to(device),
            traj_stats=stats_input,
        )

    # Move outputs to CPU
    encoder_output = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in output.items()
    }

    return model, config, dataset, encoder_output, device


# ---------------------------------------------------------------------------
# Reference metrics from raw trajectories
# ---------------------------------------------------------------------------

def _compute_reference_metrics_trajectory(dataset, config):
    """Compute reference speed/density/flow/LOS directly from raw trajectories.

    Independent computation: raw (x,y,t) → traffic engineering formulas.
    Returns dict: lane_key -> {speed, density, flow, los, source}
    """
    import polars as pl
    from src.bridge.traffic_translator import classify_los, TYPICAL_CAPACITY
    from src.data.annotation_loader import get_group_lanes, load_annotation_json
    from src.data.lane_dataset import point_to_polyline_dist

    data_cfg = config.get("data", {})
    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    image_w = data_cfg.get("image_width", 1920)
    image_h = data_cfg.get("image_height", 1080)
    wh = np.array([image_w, image_h], dtype=np.float64)

    assign_cfg = config.get("assignment", {})
    lat_thresh_px = assign_cfg.get("lateral_threshold_px", 60.0)
    lat_thresh_norm = lat_thresh_px / image_w
    min_pts = assign_cfg.get("min_tracklet_points", 5)

    fps = 30.0

    reference = {}
    cameras = sorted(set(s.camera for s in dataset.samples))

    for cam in cameras:
        annot_path = annot_dir / cam / "annotation.json"
        traj_path = annot_dir / cam / "trajectory.csv"
        if not annot_path.exists() or not traj_path.exists():
            continue

        annotation = load_annotation_json(annot_path)
        traj_df = pl.read_csv(str(traj_path))
        if traj_df.is_empty():
            continue

        id_col = "id" if "id" in traj_df.columns else "track_id"
        time_col = "time" if "time" in traj_df.columns else None
        has_time = time_col is not None

        if has_time:
            obs_period = max(traj_df[time_col].max() - traj_df[time_col].min(), 1.0)
        else:
            obs_period = max((traj_df["frame_num"].max() - traj_df["frame_num"].min()) / fps, 1.0)

        # Estimate pixels_per_meter from longest annotation lane
        max_lane_px = 0.0
        for lg in annotation["lane_groups"]:
            lanes = get_group_lanes(annotation, lg["group_id"],
                                    image_wh=(image_w, image_h))
            for lane in lanes:
                wp_px = lane["waypoints"] * wh
                diffs = np.diff(wp_px, axis=0)
                max_lane_px = max(max_lane_px, np.sum(np.linalg.norm(diffs, axis=1)))
        ppm = max_lane_px / 150.0 if max_lane_px > 50 else 10.0

        for lg in annotation["lane_groups"]:
            gid = lg["group_id"]
            lanes = get_group_lanes(annotation, gid, image_wh=(image_w, image_h))
            if not lanes:
                continue

            lane_tracks = {l["cls_id"]: [] for l in lanes}

            for track_id_val, group in traj_df.group_by(id_col):
                pts_px = np.column_stack([
                    group["x"].to_numpy(), group["y"].to_numpy(),
                ]).astype(np.float64)
                pts_norm = pts_px / wh

                if len(pts_norm) < min_pts:
                    continue

                if has_time:
                    times = group[time_col].to_numpy().astype(np.float64)
                else:
                    times = group["frame_num"].to_numpy().astype(np.float64) / fps

                best_cls, best_dist = None, float("inf")
                for lane in lanes:
                    wp = lane["waypoints"]
                    if len(wp) < 2:
                        continue
                    md = point_to_polyline_dist(pts_norm, wp).mean()
                    if md < best_dist:
                        best_dist = md
                        best_cls = lane["cls_id"]

                if best_cls is not None and best_dist < lat_thresh_norm:
                    lane_tracks[best_cls].append({
                        "pts_px": pts_px, "times": times,
                        "t_start": times[0], "t_end": times[-1],
                    })

            for lane in lanes:
                cls_id = lane["cls_id"]
                lane_key = f"{cam}_{gid}_{cls_id}"
                tracks = lane_tracks.get(cls_id, [])

                if not tracks:
                    reference[lane_key] = {
                        "speed": 0.0, "density": 0.0, "flow": 0.0,
                        "los": "A", "n_vehicles": 0, "source": "trajectory",
                    }
                    continue

                # Speed: median of per-trajectory speeds
                traj_speeds_mph = []
                for tr in tracks:
                    pts, ts = tr["pts_px"], tr["times"]
                    if len(pts) < 2:
                        continue
                    dx, dt = np.diff(pts, axis=0), np.diff(ts)
                    valid = dt > 1e-6
                    if not valid.any():
                        continue
                    px_per_sec = np.linalg.norm(dx[valid], axis=1) / dt[valid]
                    mph = np.median(px_per_sec) / ppm * 2.237
                    traj_speeds_mph.append(mph)

                ref_speed = min(max(float(np.median(traj_speeds_mph)) if traj_speeds_mph else 0.0, 0.0), 120.0)

                # Flow and density from fundamental equation
                n_vehicles = len(tracks)
                ref_flow = (n_vehicles / obs_period) * 3600.0 if obs_period > 0 else 0.0
                ref_density = min(ref_flow / ref_speed if ref_speed > 1.0 else 0.0, 200.0)

                ref_vc = ref_flow / TYPICAL_CAPACITY if TYPICAL_CAPACITY > 0 else 0
                ref_los = classify_los(ref_density, ref_vc)

                reference[lane_key] = {
                    "speed": ref_speed, "density": ref_density,
                    "flow": ref_flow, "los": ref_los,
                    "n_vehicles": n_vehicles, "source": "trajectory",
                }

        logger.info(f"  {cam}: ppm={ppm:.1f}, obs_period={obs_period:.0f}s")

    logger.info(f"Trajectory reference: {len(reference)} lanes from {len(cameras)} cameras")
    return reference


# ---------------------------------------------------------------------------
# Figure C1: Embedding similarity vs traffic metric distance
# ---------------------------------------------------------------------------

def figure_c1(embeddings, reference, dataset, output_dir: Path):
    """C1: Embedding similarity predicts traffic metric similarity.

    Shows that lanes with higher cosine similarity in embedding space
    also have more similar speed, density, and LOS. Uses binned means
    with error bars for clarity.
    """
    from scipy.stats import spearmanr

    n = len(dataset.samples)
    lane_keys = [s.lane_key for s in dataset.samples]

    # Cosine similarity matrix
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    cos_sim = emb_norm @ emb_norm.T

    # Traffic metric vectors
    speeds = np.array([reference.get(lk, {}).get("speed", 0.0) for lk in lane_keys])
    densities = np.array([reference.get(lk, {}).get("density", 0.0) for lk in lane_keys])
    los_idx = np.array([
        {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}.get(
            reference.get(lk, {}).get("los", "A"), 0
        ) for lk in lane_keys
    ])

    # Collect all pairs (upper triangle only)
    sims, speed_diffs, dens_diffs, los_diffs = [], [], [], []
    for i in range(n):
        if speeds[i] <= 0:
            continue
        for j in range(i + 1, n):
            if speeds[j] <= 0:
                continue
            sims.append(cos_sim[i, j])
            speed_diffs.append(abs(speeds[i] - speeds[j]))
            dens_diffs.append(abs(densities[i] - densities[j]))
            los_diffs.append(abs(int(los_idx[i]) - int(los_idx[j])))

    sims = np.array(sims)
    speed_diffs = np.array(speed_diffs)
    dens_diffs = np.array(dens_diffs)
    los_diffs = np.array(los_diffs)

    # Bin by cosine similarity
    n_bins = 10
    bin_edges = np.linspace(sims.min(), sims.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(sims, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    def binned_stats(values):
        means, stds, counts = [], [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                means.append(values[mask].mean())
                stds.append(values[mask].std() / np.sqrt(mask.sum()))  # SEM
                counts.append(mask.sum())
            else:
                means.append(np.nan)
                stds.append(np.nan)
                counts.append(0)
        return np.array(means), np.array(stds), np.array(counts)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Speed difference vs similarity ──
    spd_means, spd_sems, _ = binned_stats(speed_diffs)
    ax1.errorbar(bin_centers, spd_means, yerr=spd_sems, fmt="o-",
                 color="#2196F3", capsize=3, markersize=5, linewidth=1.5)

    rho_spd, _ = spearmanr(sims, speed_diffs)
    ax1.set_xlabel("Embedding Cosine Similarity")
    ax1.set_ylabel("|Speed Difference| (mph)")
    ax1.set_title(f"Speed Difference (ρ = {rho_spd:.3f})")
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Density difference vs similarity ──
    den_means, den_sems, _ = binned_stats(dens_diffs)
    ax2.errorbar(bin_centers, den_means, yerr=den_sems, fmt="s-",
                 color="#FF5722", capsize=3, markersize=5, linewidth=1.5)

    rho_den, _ = spearmanr(sims, dens_diffs)
    ax2.set_xlabel("Embedding Cosine Similarity")
    ax2.set_ylabel("|Density Difference| (veh/mi/ln)")
    ax2.set_title(f"Density Difference (ρ = {rho_den:.3f})")
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: LOS difference vs similarity ──
    los_means, los_sems, _ = binned_stats(los_diffs.astype(float))
    ax3.errorbar(bin_centers, los_means, yerr=los_sems, fmt="D-",
                 color="#4CAF50", capsize=3, markersize=5, linewidth=1.5)

    rho_los, _ = spearmanr(sims, los_diffs)
    ax3.set_xlabel("Embedding Cosine Similarity")
    ax3.set_ylabel("|LOS Grade Difference|")
    ax3.set_title(f"LOS Difference (ρ = {rho_los:.3f})")
    ax3.grid(True, alpha=0.2)

    fig.suptitle(
        "C1: Embedding Similarity Predicts Traffic Metric Similarity",
        fontsize=13,
    )
    fig.tight_layout()

    path = output_dir / "C1_embedding_vs_traffic.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"Saved C1 to {path}\n"
        f"  Speed ρ={rho_spd:.3f}, Density ρ={rho_den:.3f}, LOS ρ={rho_los:.3f}\n"
        f"  ({len(sims)} pairs from {n} lanes)"
    )


# ---------------------------------------------------------------------------
# Figure C2: LOS agreement by embedding proximity
# ---------------------------------------------------------------------------

def figure_c2(embeddings, reference, dataset, output_dir: Path):
    """C2: LOS agreement rate by embedding proximity tier.

    Left: For each proximity tier (top-10%, top-25%, top-50%, all),
    shows the % of pairs with same LOS or adjacent LOS.

    Right: Cross-camera retrieval — for each lane, find nearest embedding
    from a *different camera* and check LOS agreement.
    """
    from src.bridge.traffic_translator import LOS_THRESHOLDS

    n = len(dataset.samples)
    lane_keys = [s.lane_key for s in dataset.samples]
    cameras = [s.camera for s in dataset.samples]

    los_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    los_grades = ["A", "B", "C", "D", "E", "F"]
    los_colors = {
        "A": "#2ecc71", "B": "#27ae60", "C": "#f1c40f",
        "D": "#e67e22", "E": "#e74c3c", "F": "#8e44ad",
    }

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    cos_sim = emb_norm @ emb_norm.T

    los_arr = np.array([
        los_to_idx.get(reference.get(lk, {}).get("los", "A"), 0)
        for lk in lane_keys
    ])

    # Valid lanes (have reference speed > 0)
    valid = np.array([reference.get(lk, {}).get("speed", 0) > 0 for lk in lane_keys])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel 1: LOS agreement by proximity tier ──
    # Collect all valid pairs
    pair_sims, pair_los_diff = [], []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            pair_sims.append(cos_sim[i, j])
            pair_los_diff.append(abs(int(los_arr[i]) - int(los_arr[j])))

    pair_sims = np.array(pair_sims)
    pair_los_diff = np.array(pair_los_diff)

    # Tiers by similarity percentile
    tiers = [
        ("Top 10%", np.percentile(pair_sims, 90)),
        ("Top 25%", np.percentile(pair_sims, 75)),
        ("Top 50%", np.percentile(pair_sims, 50)),
        ("All pairs", pair_sims.min() - 1),
    ]

    tier_labels, exact_accs, adj_accs, tier_counts = [], [], [], []
    for label, threshold in tiers:
        mask = pair_sims >= threshold
        if mask.sum() == 0:
            continue
        tier_labels.append(label)
        tier_counts.append(mask.sum())
        exact_accs.append((pair_los_diff[mask] == 0).mean())
        adj_accs.append((pair_los_diff[mask] <= 1).mean())

    x = np.arange(len(tier_labels))
    w = 0.35
    ax1.bar(x - w/2, exact_accs, w, label="Exact LOS match",
            color="#2196F3", edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, adj_accs, w, label="Adjacent (±1)",
            color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n(n={c:,})" for l, c in zip(tier_labels, tier_counts)],
                        fontsize=9)
    ax1.set_ylabel("Agreement Rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("LOS Agreement by Embedding Proximity")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2, axis="y")

    # Annotate percentages
    for i in range(len(tier_labels)):
        ax1.text(x[i] - w/2, exact_accs[i] + 0.02, f"{exact_accs[i]:.0%}",
                 ha="center", fontsize=8)
        ax1.text(x[i] + w/2, adj_accs[i] + 0.02, f"{adj_accs[i]:.0%}",
                 ha="center", fontsize=8)

    # ── Panel 2: Cross-camera nearest-neighbor LOS retrieval ──
    # For each lane, find NN from different camera, check LOS match
    nn_exact, nn_adj, nn_total = 0, 0, 0
    nn_los_pairs = []  # (true_los, nn_los)

    for i in range(n):
        if not valid[i]:
            continue
        best_sim, best_j = -1, -1
        for j in range(n):
            if j == i or cameras[j] == cameras[i] or not valid[j]:
                continue
            if cos_sim[i, j] > best_sim:
                best_sim = cos_sim[i, j]
                best_j = j
        if best_j >= 0:
            nn_total += 1
            diff = abs(int(los_arr[i]) - int(los_arr[best_j]))
            if diff == 0:
                nn_exact += 1
            if diff <= 1:
                nn_adj += 1
            nn_los_pairs.append((int(los_arr[i]), int(los_arr[best_j])))

    # Build cross-camera confusion matrix
    cm = np.zeros((6, 6), dtype=int)
    for true_i, pred_i in nn_los_pairs:
        cm[true_i, pred_i] += 1

    im = ax2.imshow(cm, interpolation="nearest", cmap="Blues")
    ax2.set_xticks(range(6))
    ax2.set_xticklabels(los_grades, fontsize=10)
    ax2.set_yticks(range(6))
    ax2.set_yticklabels(los_grades, fontsize=10)
    ax2.set_xlabel("Nearest Neighbor LOS (other camera)")
    ax2.set_ylabel("Query Lane LOS")

    thresh_val = cm.max() / 2.0
    for i in range(6):
        for j in range(6):
            if cm[i, j] > 0:
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=9, color="white" if cm[i, j] > thresh_val else "black")

    nn_exact_rate = nn_exact / nn_total if nn_total > 0 else 0
    nn_adj_rate = nn_adj / nn_total if nn_total > 0 else 0
    ax2.set_title(
        f"Cross-Camera NN Retrieval (n={nn_total})\n"
        f"Exact: {nn_exact_rate:.1%} | Adjacent(±1): {nn_adj_rate:.1%}",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("C2: Embedding Proximity Preserves Traffic State", fontsize=13)
    fig.tight_layout()

    path = output_dir / "C2_los_agreement.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"Saved C2 to {path}\n"
        f"  Cross-camera NN: exact={nn_exact_rate:.1%}, adj={nn_adj_rate:.1%} (n={nn_total})"
    )


# ---------------------------------------------------------------------------
# Figure C3: Per-camera embedding-metric correlation summary
# ---------------------------------------------------------------------------

def figure_c3(embeddings, reference, dataset, output_dir: Path):
    """C3: Per-camera summary of embedding-traffic metric correlation.

    Left: Heatmap — per-camera Spearman ρ between embedding distance
    and speed/density/LOS difference within each camera.

    Right: Scatter of all per-camera (embedding_sim, speed_diff) pairs
    colored by camera, showing the global trend.
    """
    from scipy.stats import spearmanr

    n = len(dataset.samples)
    lane_keys = [s.lane_key for s in dataset.samples]
    cam_list = [s.camera for s in dataset.samples]
    cameras = sorted(set(cam_list))

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    cos_sim = emb_norm @ emb_norm.T

    speeds = np.array([reference.get(lk, {}).get("speed", 0.0) for lk in lane_keys])
    densities = np.array([reference.get(lk, {}).get("density", 0.0) for lk in lane_keys])
    los_idx = np.array([
        {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}.get(
            reference.get(lk, {}).get("los", "A"), 0
        ) for lk in lane_keys
    ])
    valid = speeds > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # ── Panel 1: Per-camera heatmap ──
    metric_names = ["Speed ρ", "Density ρ", "LOS ρ", "n_lanes"]
    heatmap = np.full((len(cameras), len(metric_names)), np.nan)

    for ci, cam in enumerate(cameras):
        idxs = [i for i in range(n) if cam_list[i] == cam and valid[i]]
        heatmap[ci, 3] = len(idxs)
        if len(idxs) < 3:
            continue

        # Intra-camera pairs
        sims_c, spd_d, den_d, los_d = [], [], [], []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                sims_c.append(cos_sim[i, j])
                spd_d.append(abs(speeds[i] - speeds[j]))
                den_d.append(abs(densities[i] - densities[j]))
                los_d.append(abs(int(los_idx[i]) - int(los_idx[j])))

        if len(sims_c) >= 3:
            sims_c = np.array(sims_c)
            rho_s, _ = spearmanr(sims_c, spd_d)
            rho_d, _ = spearmanr(sims_c, den_d)
            rho_l, _ = spearmanr(sims_c, los_d)
            heatmap[ci, 0] = rho_s
            heatmap[ci, 1] = rho_d
            heatmap[ci, 2] = rho_l

    # Plot heatmap (correlation columns only, n_lanes as text)
    corr_data = heatmap[:, :3]
    im = ax1.imshow(corr_data, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    # _r because negative ρ = good (higher similarity → smaller difference)

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(metric_names[:3], fontsize=10)
    cam_labels = [f"{cam} (n={int(heatmap[ci, 3])})" for ci, cam in enumerate(cameras)]
    ax1.set_yticks(range(len(cameras)))
    ax1.set_yticklabels(cam_labels, fontsize=8)

    for i in range(len(cameras)):
        for j in range(3):
            val = corr_data[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "—"
            color = "white" if (not np.isnan(val) and abs(val) > 0.5) else "black"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    means = np.nanmean(corr_data, axis=0)
    ax1.set_title("Per-Camera Embedding-Metric Correlation (Spearman ρ)", fontsize=10)
    ax1.set_xlabel(
        f"Mean:  Speed ρ={means[0]:.3f}  Density ρ={means[1]:.3f}  LOS ρ={means[2]:.3f}",
        fontsize=9,
    )
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.04,
                 label="ρ (negative = similarity predicts closeness)")

    # ── Panel 2: Global embedding sim vs speed diff (sampled) ──
    cam_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cameras), 1)))
    cam_color_map = {c: cam_colors[i] for i, c in enumerate(cameras)}

    # Sample pairs to avoid overplotting
    rng = np.random.default_rng(42)
    all_pairs = []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            all_pairs.append((i, j))

    if len(all_pairs) > 2000:
        sample_idx = rng.choice(len(all_pairs), 2000, replace=False)
        sampled = [all_pairs[k] for k in sample_idx]
    else:
        sampled = all_pairs

    for i, j in sampled:
        # Color by whether same or cross camera
        if cam_list[i] == cam_list[j]:
            c = cam_color_map[cam_list[i]]
            alpha = 0.3
        else:
            c = "gray"
            alpha = 0.15
        ax2.scatter(cos_sim[i, j], abs(speeds[i] - speeds[j]),
                    c=[c], s=8, alpha=alpha, linewidth=0)

    # Add trend line
    all_sims_arr = np.array([cos_sim[i, j] for i, j in sampled])
    all_spd_arr = np.array([abs(speeds[i] - speeds[j]) for i, j in sampled])
    z = np.polyfit(all_sims_arr, all_spd_arr, 1)
    x_line = np.linspace(all_sims_arr.min(), all_sims_arr.max(), 50)
    ax2.plot(x_line, np.polyval(z, x_line), "r-", linewidth=2, alpha=0.8,
             label=f"Trend (slope={z[0]:.1f})")

    rho_global, _ = spearmanr(all_sims_arr, all_spd_arr)
    ax2.set_xlabel("Embedding Cosine Similarity")
    ax2.set_ylabel("|Speed Difference| (mph)")
    ax2.set_title(f"Embedding Similarity vs Speed Distance (ρ={rho_global:.3f})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.suptitle("C3: Per-Camera Embedding-Traffic Metric Structure", fontsize=13)
    fig.tight_layout()

    path = output_dir / "C3_per_camera_summary.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"Saved C3 to {path}\n"
        f"  Mean ρ: speed={means[0]:.3f}, density={means[1]:.3f}, los={means[2]:.3f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate CrossTraffic bridge figures (C-series)"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained encoder checkpoint",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="results/bridge/figures")
    parser.add_argument(
        "--figures", nargs="+", default=["C1", "C2", "C3"],
        choices=["C1", "C2", "C3"],
    )
    parser.add_argument(
        "--calibration", default=None,
        help="Path to camera calibration YAML (optional)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, dataset, encoder_output, device = _load_data(args)

    # Get embeddings (N, 128)
    embeddings = encoder_output["embedding"].numpy()

    # Compute trajectory-derived traffic metrics as ground truth
    reference = _compute_reference_metrics_trajectory(dataset, config)

    n_valid = sum(1 for lk in reference if reference[lk]["speed"] > 0)
    logger.info(f"Valid lanes with traffic metrics: {n_valid}/{len(dataset.samples)}")

    if "C1" in args.figures:
        logger.info("Generating C1: Embedding similarity vs traffic metrics...")
        figure_c1(embeddings, reference, dataset, output_dir)

    if "C2" in args.figures:
        logger.info("Generating C2: LOS agreement by proximity...")
        figure_c2(embeddings, reference, dataset, output_dir)

    if "C3" in args.figures:
        logger.info("Generating C3: Per-camera summary...")
        figure_c3(embeddings, reference, dataset, output_dir)

    logger.info(f"All bridge figures saved to {output_dir}")


if __name__ == "__main__":
    main()
