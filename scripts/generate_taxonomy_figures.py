#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import logging

import cv2
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

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]
_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]


def _load_data(args):
    """Load encoder, encode all lanes, return embeddings + metadata."""
    import yaml

    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import encode_lanes, load_trained_encoder

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

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
    )

    projections, roles, lane_keys = encode_lanes(
        model, loader, device, drop_geometry=False,
    )
    cameras = [s.camera for s in dataset.samples]

    return projections, roles, cameras, lane_keys, dataset, config


def _cluster_embeddings(embeddings: np.ndarray, method: str = "hdbscan",
                        n_clusters: int = 5) -> np.ndarray:
    """Cluster lane embeddings.

    Args:
        embeddings: (N, D) array.
        method: 'hdbscan' or 'kmeans'.
        n_clusters: Number of clusters for k-means.

    Returns:
        (N,) cluster labels (int). -1 for noise (HDBSCAN only).
    """
    if method == "hdbscan":
        try:
            from hdbscan import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
            labels = clusterer.fit_predict(embeddings)
            n_found = len(set(labels) - {-1})
            logger.info(f"HDBSCAN found {n_found} clusters ({(labels == -1).sum()} noise)")
            return labels
        except ImportError:
            logger.warning("hdbscan not installed, falling back to k-means")
            method = "kmeans"

    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        logger.info(f"K-means: {n_clusters} clusters")
        return labels

    raise ValueError(f"Unknown clustering method: {method}")


def figure_m1(embeddings: np.ndarray, labels: np.ndarray,
              cameras: list, lane_keys: list, output_dir: Path):
    """UMAP scatter of all lane embeddings colored by cluster, markers by camera."""
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed, skipping M1")
        return

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    unique_cams = sorted(set(cameras))
    cam_to_idx = {c: i for i, c in enumerate(unique_cams)}
    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(coords)):
        cidx = cam_to_idx[cameras[i]]
        marker = _MARKERS[cidx % len(_MARKERS)]
        label = labels[i]
        if label == -1:
            color = "#cccccc"
        else:
            color = _CLUSTER_COLORS[label % len(_CLUSTER_COLORS)]
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, marker=marker, s=70,
            edgecolors="black", linewidths=0.5, zorder=2,
        )

    # Cluster legend
    cluster_handles = []
    for lbl in unique_labels:
        name = f"Cluster {lbl}" if lbl >= 0 else "Noise"
        color = "#cccccc" if lbl == -1 else _CLUSTER_COLORS[lbl % len(_CLUSTER_COLORS)]
        h = ax.scatter([], [], c=color, s=50, edgecolors="black",
                        linewidths=0.5, label=name)
        cluster_handles.append(h)

    # Camera legend
    cam_handles = []
    for cam in unique_cams:
        cidx = cam_to_idx[cam]
        h = ax.scatter([], [], marker=_MARKERS[cidx % len(_MARKERS)],
                        c="gray", s=40, label=cam)
        cam_handles.append(h)

    leg1 = ax.legend(handles=cluster_handles, loc="upper left", fontsize=7,
                     title="Clusters", title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=cam_handles, loc="upper right", fontsize=7,
              title="Cameras", title_fontsize=8)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Lane Embeddings by Behavioral Cluster")
    fig.tight_layout()

    path = output_dir / "M1_umap_clusters.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved M1 to {path}")


def figure_m2(labels: np.ndarray, roles: np.ndarray, dataset,
              output_dir: Path):
    """Table of cluster behavioral statistics with inferred HCM type.

    Columns: cluster, n_lanes, mean_speed, mean_density (traj_count_norm),
             mean_lat_rank, edge_ratio, inferred_type.
    """
    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        logger.warning("No clusters found, skipping M2")
        return

    rows = []
    for lbl in unique_labels:
        mask = labels == lbl
        n = mask.sum()
        indices = np.where(mask)[0]

        # Gather stats from samples
        speeds = [dataset.samples[i].traj_stats[0] for i in indices]
        densities = [dataset.samples[i].traj_stats[3] for i in indices]  # traj_count_norm
        lat_ranks = roles[mask, 0]
        is_left = roles[mask, 1]
        is_right = roles[mask, 2]
        edge_ratio = (is_left.sum() + is_right.sum()) / (2 * n) if n > 0 else 0

        mean_speed = np.mean(speeds)
        mean_density = np.mean(densities)
        mean_lat = lat_ranks.mean()

        # Infer HCM type heuristic
        hcm_type = _infer_hcm_type(mean_speed, mean_density, mean_lat, edge_ratio)

        rows.append({
            "cluster": lbl,
            "n_lanes": int(n),
            "mean_speed": mean_speed,
            "mean_density": mean_density,
            "mean_lat_rank": float(mean_lat),
            "edge_ratio": float(edge_ratio),
            "inferred_type": hcm_type,
        })

    # Save CSV
    csv_path = output_dir / "M2_cluster_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved M2 table to {csv_path}")

    # Render as figure
    fig, ax = plt.subplots(figsize=(10, max(2, len(rows) * 0.6 + 1.5)))
    ax.axis("off")

    col_labels = ["Cluster", "N", "Speed", "Density", "Lat Rank", "Edge%", "Inferred Type"]
    cell_text = []
    colors_per_row = []
    for r in rows:
        cell_text.append([
            str(r["cluster"]),
            str(r["n_lanes"]),
            f"{r['mean_speed']:.4f}",
            f"{r['mean_density']:.3f}",
            f"{r['mean_lat_rank']:.2f}",
            f"{r['edge_ratio']:.2f}",
            r["inferred_type"],
        ])
        colors_per_row.append(
            _CLUSTER_COLORS[r["cluster"] % len(_CLUSTER_COLORS)]
        )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color first column by cluster
    for i, color in enumerate(colors_per_row):
        table[i + 1, 0].set_facecolor(color)
        table[i + 1, 0].set_text_props(color="white", fontweight="bold")

    ax.set_title("Cluster Behavioral Statistics", fontsize=13, pad=20)

    # Column definition footnotes
    footnotes = (
        "Speed: mean per-frame displacement in normalized pixel space [0,1] "
        "(higher ≈ faster vehicles; not calibrated to m/s)\n"
        "Density: normalized trajectory count per lane "
        "(traj_count / max_traj_count across lanes in group)\n"
        "Lat Rank: lateral position within lane group "
        "(0.0 = leftmost, 1.0 = rightmost)\n"
        "Edge%: fraction of cluster lanes that are boundary lanes "
        "((n_leftmost + n_rightmost) / (2 × n_lanes))"
    )
    fig.text(
        0.5, -0.02, footnotes, ha="center", va="top", fontsize=7,
        fontstyle="italic", wrap=True,
        transform=fig.transFigure,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    path = output_dir / "M2_cluster_table.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved M2 figure to {path}")


def _infer_hcm_type(mean_speed: float, mean_density: float,
                     mean_lat: float, edge_ratio: float) -> str:
    """Heuristic HCM lane type inference from cluster statistics.

    Based on speed/density patterns typical of HCM lane types.
    These thresholds are approximate and should be calibrated with real data.
    """
    if mean_density < 0.1:
        return "Low-volume"
    if mean_speed > 0.015 and mean_density > 0.3:
        return "Free-flow (high vol)"
    if mean_speed > 0.01:
        if edge_ratio > 0.5:
            return "Edge lane (free)"
        return "Free-flow"
    if mean_speed > 0.005:
        return "Synchronized"
    return "Congested"


def figure_m3(projections: np.ndarray, cameras: list, lane_keys: list,
              dataset, roles: np.ndarray, output_dir: Path,
              n_queries: int = 6, top_k: int = 3):
    """Cross-camera retrieval: query lane → top-K from other cameras.

    Shows query lane polyline on its camera frame alongside top-K matches
    on their respective camera frames, with behavioral annotations
    (lateral rank, speed, density) to demonstrate alignment quality.
    """
    data_cfg = dataset.config.get("data", {})
    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    W, H = dataset.image_wh

    # Cosine similarity matrix
    proj_norm = projections / (np.linalg.norm(projections, axis=1, keepdims=True) + 1e-8)
    sim_matrix = proj_norm @ proj_norm.T

    N = len(cameras)
    cam_arr = np.array(cameras)

    # Select diverse query lanes (one per camera, spread across clusters)
    unique_cams = sorted(set(cameras))
    query_indices = []
    for cam in unique_cams:
        cam_mask = cam_arr == cam
        cam_indices = np.where(cam_mask)[0]
        if len(cam_indices) > 0:
            # Pick the one with highest mean cross-camera similarity
            cross_cam_mask = ~cam_mask
            if cross_cam_mask.any():
                cross_sims = sim_matrix[np.ix_(cam_indices, np.where(cross_cam_mask)[0])]
                best_local = cross_sims.mean(axis=1).argmax()
                query_indices.append(cam_indices[best_local])
        if len(query_indices) >= n_queries:
            break

    if not query_indices:
        logger.warning("No query lanes for M3")
        return

    n_show = min(len(query_indices), n_queries)
    fig, axes = plt.subplots(n_show, top_k + 1, figsize=(4 * (top_k + 1), 3.5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, q_idx in enumerate(query_indices[:n_show]):
        q_cam = cameras[q_idx]

        # Find top-K from other cameras
        sims = sim_matrix[q_idx].copy()
        for i in range(N):
            if cameras[i] == q_cam:
                sims[i] = -1.0
        top_indices = np.argsort(sims)[::-1][:top_k]

        # Draw query
        q_sample = dataset.samples[q_idx]
        q_stats = _lane_stats_str(q_sample, roles[q_idx])
        _draw_lane_on_frame(
            axes[row, 0], q_sample, annot_dir, W, H,
            title=f"Query: {q_cam}",
            subtitle=q_stats,
            is_query=True,
        )

        # Draw matches
        for k, ref_idx in enumerate(top_indices):
            sim_val = sim_matrix[q_idx, ref_idx]
            ref_sample = dataset.samples[ref_idx]
            ref_stats = _lane_stats_str(ref_sample, roles[ref_idx])
            _draw_lane_on_frame(
                axes[row, k + 1], ref_sample, annot_dir, W, H,
                title=f"#{k+1}: {cameras[ref_idx]} (sim={sim_val:.2f})",
                subtitle=ref_stats,
                is_query=False,
            )

    fig.suptitle("Cross-Camera Lane Retrieval", fontsize=14, y=1.01)
    fig.tight_layout()

    path = output_dir / "M3_cross_camera_retrieval.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved M3 to {path}")


def _lane_stats_str(sample, role_vec: np.ndarray) -> str:
    """Build a compact annotation string for a lane."""
    lat_rank = role_vec[0]
    is_left = bool(role_vec[1] > 0.5)
    is_right = bool(role_vec[2] > 0.5)
    speed = sample.traj_stats[0]
    density = sample.traj_stats[3]

    edge_tag = ""
    if is_left:
        edge_tag = " [L]"
    elif is_right:
        edge_tag = " [R]"

    return f"rank={lat_rank:.2f}{edge_tag}  spd={speed:.4f}  den={density:.2f}"


def _draw_lane_on_frame(ax, sample, annot_dir: Path, W: int, H: int,
                         title: str = "", subtitle: str = "",
                         is_query: bool = False):
    """Draw a lane polyline on its camera frame with annotations."""
    from src.utils.visualization import SLOT_COLORS

    frame_path = annot_dir / sample.camera / "last_frame.npy"
    if frame_path.exists():
        frame = np.load(str(frame_path))
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = np.zeros((H, W, 3), dtype=np.uint8)

    overlay = frame.copy()
    pts = (sample.geometry * np.array([W, H])).astype(np.int32)

    # Thicker line + glow for visibility
    line_color = (255, 255, 0) if is_query else (0, 255, 255)
    if len(pts) >= 2:
        pts_cv = pts.reshape(-1, 1, 2)
        # Dark outline for contrast
        cv2.polylines(overlay, [pts_cv], False, (0, 0, 0), 7)
        # Bright lane
        cv2.polylines(overlay, [pts_cv], False, line_color, 4)

    ax.imshow(overlay)
    ax.set_title(title, fontsize=8, fontweight="bold" if is_query else "normal")
    if subtitle:
        ax.text(
            0.5, -0.02, subtitle, transform=ax.transAxes, ha="center",
            fontsize=7, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Generate Module 3: Behavioral Taxonomy figures")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained encoder checkpoint",
    )
    parser.add_argument(
        "--config", default=None,
        help="Optional config YAML override",
    )
    parser.add_argument(
        "--output-dir", default="results/taxonomy/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--cluster-method", default="hdbscan",
        choices=["hdbscan", "kmeans"],
        help="Clustering method (default: hdbscan)",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=5,
        help="Number of clusters for k-means (ignored for HDBSCAN)",
    )
    parser.add_argument(
        "--figures", nargs="+", default=["M1", "M2", "M3"],
        choices=["M1", "M2", "M3"],
        help="Which figures to generate (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projections, roles, cameras, lane_keys, dataset, config = _load_data(args)

    embeddings = projections.numpy()
    roles_np = roles.numpy()

    # Cluster
    labels = _cluster_embeddings(
        embeddings, method=args.cluster_method, n_clusters=args.n_clusters,
    )

    # Save cluster assignments
    assign_path = output_dir / "cluster_assignments.csv"
    with open(assign_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lane_key", "camera", "cluster"])
        for i in range(len(lane_keys)):
            writer.writerow([lane_keys[i], cameras[i], int(labels[i])])
    logger.info(f"Saved cluster assignments to {assign_path}")

    if "M1" in args.figures:
        logger.info("Generating M1: UMAP cluster visualization...")
        figure_m1(embeddings, labels, cameras, lane_keys, output_dir)

    if "M2" in args.figures:
        logger.info("Generating M2: Cluster statistics table...")
        figure_m2(labels, roles_np, dataset, output_dir)

    if "M3" in args.figures:
        logger.info("Generating M3: Cross-camera retrieval...")
        figure_m3(embeddings, cameras, lane_keys, dataset, roles_np, output_dir)

    logger.info(f"All taxonomy figures saved to {output_dir}")


if __name__ == "__main__":
    main()
