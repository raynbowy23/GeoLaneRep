#!/usr/bin/env python
"""Side-by-side comparison: geometric assignment (GT) vs zero-shot prediction.

Left panel:  trajectories colored by geometric lane assignment (uses annotation)
Right panel: trajectories colored by zero-shot prediction (no annotation)

If left looks clean but right is messy → encoder/matching problem
If both look messy → clustering problem (upstream of encoder)

Usage:
    python scripts/compare_gt_vs_zero_shot.py --camera I43_Walnut
    python scripts/compare_gt_vs_zero_shot.py --camera I43_Walnut --config configs/lane_contrastive.yaml
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import cv2
import numpy as np
import polars as pl
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _draw_gt_assignment(
    frame: np.ndarray,
    assignments: pl.DataFrame,
    annotation: dict,
    image_wh: tuple,
) -> np.ndarray:
    """Draw trajectories colored by geometric lane assignment (ground truth)."""
    from src.data.annotation_loader import get_group_lanes
    from src.utils.visualization import SLOT_COLORS

    vis = frame.copy()
    W, H = image_wh

    # Build lane_key -> color mapping from annotation order
    lane_colors = {}
    color_idx = 0
    for lg in annotation["lane_groups"]:
        gid = lg["group_id"]
        lanes = get_group_lanes(annotation, gid, image_wh)
        for lane in lanes:
            key = (gid, lane["cls_id"])
            lane_colors[key] = SLOT_COLORS[color_idx % len(SLOT_COLORS)]
            color_idx += 1

            # Draw GT lane polyline
            pts_px = (lane["waypoints"] * np.array([W, H])).astype(np.int32)
            cv2.polylines(vis, [pts_px], False, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.polylines(vis, [pts_px], False, lane_colors[key], 2, cv2.LINE_AA)

    # Draw track trails colored by assigned lane
    id_col = "track_id" if "track_id" in assignments.columns else "id"
    for tid in assignments[id_col].unique().to_list():
        track = assignments.filter(pl.col(id_col) == tid)
        sort_col = "time" if "time" in track.columns else "frame_num"
        track = track.sort(sort_col)

        lane_id = track["lane_id"][0]
        gid = track["group_id"][0]
        pts = track.select("x", "y").to_numpy().astype(np.int32)

        if lane_id >= 0:
            color = lane_colors.get((gid, lane_id), (128, 128, 128))
        else:
            color = (80, 80, 80)  # rejected = dark gray

        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, color, 1, cv2.LINE_AA)

    # Title
    n_lanes = len(lane_colors)
    cv2.putText(vis, f"GT Assignment ({n_lanes} lanes)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, f"GT Assignment ({n_lanes} lanes)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return vis


def _draw_zero_shot(
    frame: np.ndarray,
    pseudo_lanes: list,
    camera: str,
) -> np.ndarray:
    """Draw trajectories colored by zero-shot discovered lanes."""
    from src.utils.visualization import SLOT_COLORS

    vis = frame.copy()
    h, w = vis.shape[:2]
    image_wh = np.array([w, h], dtype=np.float64)

    for lane_i, pl_lane in enumerate(pseudo_lanes):
        color = SLOT_COLORS[lane_i % len(SLOT_COLORS)]

        for traj in pl_lane.trajectories:
            pts_px = (traj * image_wh).astype(np.int32)
            if len(pts_px) >= 2:
                cv2.polylines(vis, [pts_px], False, color, 1, cv2.LINE_AA)

        # Draw mean trajectory thicker
        if pl_lane.trajectories:
            from src.zero_shot_lanes import _compute_mean_polyline
            mean_traj = _compute_mean_polyline(pl_lane.trajectories, 32)
            mean_px = (mean_traj * image_wh).astype(np.int32)
            cv2.polylines(vis, [mean_px], False, color, 3, cv2.LINE_AA)

            mid = mean_px[len(mean_px) // 2]
            label = f"G{pl_lane.group_id}L{pl_lane.lane_idx}"
            cv2.putText(vis, label, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    n_lanes = len(pseudo_lanes)
    cv2.putText(vis, f"Zero-Shot ({n_lanes} lanes)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, f"Zero-Shot ({n_lanes} lanes)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Compare GT assignment vs zero-shot lane detection")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--config", default="configs/lane_contrastive.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    cam_dir = annot_dir / args.camera

    # Load frame
    frame_path = cam_dir / "last_frame.npy"
    frame = np.load(str(frame_path))
    H, W = frame.shape[:2]

    # --- Left panel: GT geometric assignment ---
    from src.data.annotation_loader import load_annotation_json

    annot_path = cam_dir / "annotation.json"
    annotation = load_annotation_json(str(annot_path))

    saving_path = Path(config.get("experiment", {}).get("saving_path", "./results/"))
    assign_csv = saving_path / "lane_assignment" / args.camera / "lane_assignments.csv"

    if assign_csv.exists():
        assignments = pl.read_csv(str(assign_csv))
        left = _draw_gt_assignment(frame, assignments, annotation, (W, H))
    else:
        left = frame.copy()
        cv2.putText(left, "No assignment data — run `make assign` first",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        logger.warning(f"No assignment CSV at {assign_csv}")

    # --- Right panel: zero-shot discovered lanes ---
    traj_df = pl.read_csv(str(cam_dir / "trajectory.csv"))

    from src.zero_shot_lanes import discover_lanes
    pseudo_lanes = discover_lanes(traj_df, (H, W), args.camera, config)
    right = _draw_zero_shot(frame, pseudo_lanes, args.camera)

    # --- Combine side by side ---
    # Scale down to fit side-by-side
    scale = 0.5
    left_small = cv2.resize(left, None, fx=scale, fy=scale)
    right_small = cv2.resize(right, None, fx=scale, fy=scale)
    combined = np.hstack([left_small, right_small])

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "results/lane_contrastive/zero_shot"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.camera}_gt_vs_zero_shot.png"
    cv2.imwrite(str(out_path), combined)
    logger.info(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
