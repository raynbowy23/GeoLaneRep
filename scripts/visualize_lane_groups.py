#!/usr/bin/env python3
"""Visualize lane groups with per-group colored contours and heading labels.

Runs density-contour lane group detection and draws each lane group as a
distinct colored contour boundary with heading annotation on the camera frame.

Usage:
    python scripts/visualize_lane_groups.py --config configs/lanelet_core.yaml
    python scripts/visualize_lane_groups.py --config configs/lanelet_core.yaml --camera US12_Stoughton
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import polars as pl
import yaml

from src.data.density_contours import detect_lane_groups, fill_uncovered_contours

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _hsv_color(index: int, total: int) -> tuple:
    """Generate HSV-spaced BGR color for group index."""
    if total <= 0:
        total = 1
    hue = int(180 * index / total) % 180
    hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


def draw_lane_group_contours(
    frame: np.ndarray,
    contours: list,
    group_headings: dict,
) -> np.ndarray:
    """Draw per-group colored contour boundaries with heading labels."""
    n_groups = len(contours)
    for i, cnt in enumerate(contours):
        color = _hsv_color(i, n_groups)
        # Filled semi-transparent overlay
        overlay = frame.copy()
        cv2.drawContours(overlay, [cnt.astype(np.int32)], -1, color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        # Thick boundary
        cv2.drawContours(frame, [cnt.astype(np.int32)], -1, color, 2, cv2.LINE_AA)

        # Heading label at centroid
        M = cv2.moments(cnt.astype(np.int32))
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            pts = cnt.reshape(-1, 2)
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())

        heading_rad = group_headings.get(i, 0.0)
        heading_deg = np.degrees(heading_rad) % 360
        label = f"G{i} {heading_deg:.0f}deg"
        cv2.putText(frame, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize lane groups")
    parser.add_argument("--config", default="configs/lanelet_core.yaml")
    parser.add_argument("--camera", default=None, help="Single camera (default: all)")
    parser.add_argument("--out-dir", default="results/lanegroup_viz")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
    preprocess_dir = v1_dir / "results" / "preprocess"

    if not preprocess_dir.exists():
        logger.error(f"Preprocess dir not found: {preprocess_dir}")
        sys.exit(1)

    if args.camera:
        cameras = [args.camera]
    else:
        cam_list_path = Path(data_cfg.get(
            "camera_locations",
            str(v1_dir / "dataset" / "camera_location_list.txt"),
        ))
        if cam_list_path.exists():
            cameras = [l.strip() for l in cam_list_path.read_text().splitlines() if l.strip()]
        else:
            cameras = sorted([
                d.name for d in preprocess_dir.iterdir()
                if d.is_dir() and (d / "trajectory.csv").exists()
            ])

    if not cameras:
        logger.error("No cameras found.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method = data_cfg.get("lane_group_method", "density")
    lg_kwargs = dict(
        min_track_points=data_cfg.get("tracklet_min_points", 5),
        min_gap_deg=data_cfg.get("direction_min_gap_deg", 45.0),
        min_vehicles_per_group=data_cfg.get("min_vehicles_per_group", 3),
    )
    if method == "density":
        lg_kwargs["tracklet_length"] = data_cfg.get("tracklet_length", 15)
        lg_kwargs["neck_ratio"] = data_cfg.get("neck_ratio", 0.4)
        lg_kwargs["sigma_along"] = data_cfg.get("density_sigma_along", 25.0)
        lg_kwargs["sigma_across"] = data_cfg.get("density_sigma_across", 6.0)
    elif method == "dbscan":
        lg_kwargs["hull_buffer_px"] = data_cfg.get("hull_buffer_px", 20.0)
        lg_kwargs["dbscan_min_samples"] = data_cfg.get("dbscan_min_samples", 3)
        if "dbscan_eps" in data_cfg:
            lg_kwargs["dbscan_eps"] = data_cfg["dbscan_eps"]

    for cam in cameras:
        cam_dir = preprocess_dir / cam
        traj_path = cam_dir / "trajectory.csv"
        frame_path = cam_dir / "last_frame.npy"
        if not traj_path.exists():
            logger.warning(f"Skipping {cam}: no trajectory.csv")
            continue

        traj = pl.read_csv(str(traj_path))
        frame = np.load(str(frame_path)) if frame_path.exists() else np.zeros((1080, 1920, 3), dtype=np.uint8)
        H, W = frame.shape[:2]

        # Time window filtering
        time_window = data_cfg.get("detection_period", 3600)
        if "time" in traj.columns:
            t_min = traj["time"].min()
            traj = traj.filter(pl.col("time") < t_min + time_window)

        logger.info(f"Processing {cam} ({len(traj)} points, {W}x{H})")

        contours, group_headings, _ = detect_lane_groups(
            traj, (H, W), method=method, **lg_kwargs,
        )

        # Fill uncovered contours from v1 collect_cars if available
        cars_path = cam_dir / "collect_cars.npy"
        if cars_path.exists():
            vehicles = np.load(str(cars_path), allow_pickle=True)
            if len(vehicles) > 0:
                v1_contours = _generate_v1_contours(frame, vehicles)
                if v1_contours:
                    contours, group_headings = fill_uncovered_contours(
                        contours, group_headings, v1_contours, traj, (H, W),
                        min_track_points=data_cfg.get("tracklet_min_points", 5),
                    )

        vis = frame.copy()

        if contours:
            draw_lane_group_contours(vis, contours, group_headings)

        # Stats overlay
        n_groups = len(contours)
        y_off = 25
        for text in [cam, f"Lane groups: {n_groups}"]:
            cv2.putText(vis, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y_off += 22

        # Per-group legend
        for i in range(n_groups):
            color = _hsv_color(i, n_groups)
            heading_deg = np.degrees(group_headings.get(i, 0.0)) % 360
            cv2.putText(vis, f"G{i} {heading_deg:.0f}deg", (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y_off += 16

        save_path = out_dir / f"{cam}_lane_groups.png"
        cv2.imwrite(str(save_path), vis)
        logger.info(f"  Saved: {save_path}")

    logger.info(f"Done. Results in {out_dir}")


def _generate_v1_contours(frame: np.ndarray, vehicles: np.ndarray) -> list:
    """Generate v1-style contours from collect_cars.npy."""
    fh, fw = frame.shape[:2]
    margin_x = int(fw * 0.05)
    margin_y = int(fh * 0.05)

    confs = np.array([v[5] for v in vehicles if len(v) >= 6], dtype=np.float32)
    conf_threshold = float(np.percentile(confs, 50)) if len(confs) > 0 else 0.5

    binary = np.zeros((fh, fw), dtype=np.uint8)
    for veh in vehicles:
        if len(veh) < 6 or veh[5] <= conf_threshold:
            continue
        x, y, w, h = int(veh[0]), int(veh[1]), int(veh[2]), int(veh[3])
        if x < margin_x or x > fw - margin_x or y < margin_y or y > fh - margin_y:
            continue
        axes = (max(1, w // 4), max(1, h // 4))
        cv2.ellipse(binary, (x, y), axes, 0, 0, 360, 255, -1)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if 12000 < cv2.contourArea(c) < 6000000]


if __name__ == "__main__":
    main()
