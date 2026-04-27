import logging
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.visualization import SLOT_COLORS

logger = logging.getLogger(__name__)

_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]

def plot_embedding_space(
    projections: torch.Tensor,
    roles: torch.Tensor,
    cameras: List[str],
    lane_keys: List[str],
    save_path: str,
) -> None:
    """t-SNE visualization of lane embeddings colored by lateral rank.

    Args:
        projections: (N, proj_dim) L2-normalized projections.
        roles: (N, 5) role tensors [lat_rank, leftmost, rightmost, successor, group_size].
        cameras: Length-N list of camera names.
        lane_keys: Length-N list of lane key strings.
        save_path: Output PNG path.
    """
    from sklearn.manifold import TSNE

    proj_np = projections.numpy()
    coords = TSNE(n_components=2, perplexity=min(30, len(proj_np) - 1), random_state=42).fit_transform(proj_np)

    lat_rank = roles[:, 0].numpy()
    is_leftmost = roles[:, 1].numpy().astype(bool)
    is_rightmost = roles[:, 2].numpy().astype(bool)

    unique_cams = sorted(set(cameras))
    cam_to_idx = {c: i for i, c in enumerate(unique_cams)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(coords)):
        cidx = cam_to_idx[cameras[i]]
        marker = _MARKERS[cidx % len(_MARKERS)]
        edge_color = "black" if (is_leftmost[i] or is_rightmost[i]) else "none"
        edge_width = 1.5 if (is_leftmost[i] or is_rightmost[i]) else 0.0
        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=[lat_rank[i]], cmap="viridis", vmin=0, vmax=1,
            marker=marker, s=60, edgecolors=edge_color, linewidths=edge_width,
            zorder=2,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="lateral_rank")

    # Camera legend
    handles = []
    for cam in unique_cams:
        cidx = cam_to_idx[cam]
        h = ax.scatter([], [], marker=_MARKERS[cidx % len(_MARKERS)],
                        c="gray", s=40, label=cam)
        handles.append(h)
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=1)

    ax.set_title("Lane Embedding Space (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved embedding space plot to {save_path}")

def plot_matching_grid(
    dataset,
    query_indices: List[int],
    matched_ref_indices: List[int],
    similarities: List[float],
    save_path: str,
    max_pairs: int = 12,
) -> None:
    """Grid of (query lane, matched reference lane) pairs drawn on camera frames.

    Args:
        dataset: LaneDataset instance.
        query_indices: Dataset indices of query lanes.
        matched_ref_indices: Dataset indices of matched reference lanes.
        similarities: Cosine similarities for each pair.
        save_path: Output PNG path.
        max_pairs: Max number of pairs to show.
    """
    n_pairs = min(len(query_indices), max_pairs)
    if n_pairs == 0:
        logger.warning("No pairs to plot")
        return

    data_cfg = dataset.config.get("data", {})
    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    W, H = dataset.image_wh

    fig, axes = plt.subplots(n_pairs, 2, figsize=(12, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_pairs):
        q_idx = query_indices[row]
        r_idx = matched_ref_indices[row]
        sim = similarities[row]

        for col, idx in enumerate([q_idx, r_idx]):
            sample = dataset.samples[idx]
            ax = axes[row, col]

            # Load frame
            frame_path = annot_dir / sample.camera / "last_frame.npy"
            if frame_path.exists():
                frame = np.load(str(frame_path))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 and frame.shape[2] == 3 else frame
            else:
                frame_rgb = np.zeros((H, W, 3), dtype=np.uint8)

            # Draw lane polyline
            pts = (sample.geometry * np.array([W, H])).astype(np.int32)
            color_idx = sample.cls_id % len(SLOT_COLORS)
            bgr_color = SLOT_COLORS[color_idx]
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])  # BGR->RGB

            overlay = frame_rgb.copy()
            if len(pts) >= 2:
                # Draw on BGR copy then convert
                cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, rgb_color, 3)

            ax.imshow(overlay)
            label = "query" if col == 0 else "ref"
            ax.set_title(f"{label}: {sample.lane_key}\nsim={sim:.2f}", fontsize=8)
            ax.axis("off")

    fig.suptitle("Contrastive Matching Pairs", fontsize=12)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved matching grid to {save_path}")

def plot_similarity_heatmap(
    projections: torch.Tensor,
    cameras: List[str],
    lane_keys: List[str],
    save_path: str,
) -> None:
    """Cosine similarity heatmap across all lanes, grouped by camera.

    Args:
        projections: (N, proj_dim) L2-normalized projections.
        cameras: Length-N list of camera names.
        lane_keys: Length-N list of lane key strings.
        save_path: Output PNG path.
    """
    # Sort by camera for block structure
    order = sorted(range(len(cameras)), key=lambda i: cameras[i])
    projections = projections[order]
    cameras_sorted = [cameras[i] for i in order]
    keys_sorted = [lane_keys[i] for i in order]

    # Cosine similarity
    sim = (projections @ projections.T).numpy()

    # Find camera boundaries
    boundaries = []
    prev_cam = None
    cam_ticks = []
    for i, cam in enumerate(cameras_sorted):
        if cam != prev_cam:
            if prev_cam is not None:
                boundaries.append(i)
            cam_ticks.append((i, cam))
            prev_cam = cam

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")

    # Camera boundary lines
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=1)
        ax.axvline(b - 0.5, color="white", linewidth=1)

    # Camera tick labels
    tick_positions = []
    tick_labels = []
    for start_idx, cam in cam_ticks:
        # Find end of this camera block
        end_idx = len(cameras_sorted)
        for b in boundaries:
            if b > start_idx:
                end_idx = b
                break
        mid = (start_idx + end_idx) / 2.0
        tick_positions.append(mid)
        tick_labels.append(cam)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=7)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Lane Embedding Similarity Matrix")
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved similarity heatmap to {save_path}")
