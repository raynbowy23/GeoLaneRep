#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.annotation_loader import get_group_lanes, load_annotation_json

logger = logging.getLogger(__name__)

PALETTE = [
    (0, 0, 255),      # red
    (255, 165, 0),     # blue-ish
    (0, 255, 0),       # green
    (0, 255, 255),     # yellow
    (255, 0, 255),     # magenta
    (255, 255, 0),     # cyan
    (128, 0, 255),     # purple
    (0, 128, 255),     # orange
    (0, 180, 0),       # dark green
    (180, 0, 180),     # dark magenta
    (255, 128, 128),   # light blue
    (128, 255, 128),   # light green
]
REJECTED_COLOR = (128, 128, 128)  # gray
GT_LINE_ALPHA = 0.6  # transparency hint (drawn as dashed)


def _color_for(group_id: int, lane_id: int, max_lanes: int = 4) -> tuple:
    """Unique color per (group, lane) pair."""
    idx = group_id * max_lanes + lane_id
    return PALETTE[idx % len(PALETTE)]


def _draw_diamond(img, center, color, size=6):
    """Draw a filled diamond marker."""
    cx, cy = center
    pts = np.array([
        [cx, cy - size], [cx + size, cy],
        [cx, cy + size], [cx - size, cy],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_assignment(
    frame: np.ndarray,
    assignments: pl.DataFrame,
    annotation: dict,
    image_wh: tuple,
    show_gt: bool = True,
) -> np.ndarray:
    """Draw lane assignments on a camera frame.

    Args:
        frame: (H, W, 3) BGR image.
        assignments: DataFrame with track_id, x, y, lane_id, group_id, lane_score.
        annotation: Parsed annotation dict.
        image_wh: (width, height).

    Returns:
        Annotated frame copy.
    """
    vis = frame.copy()
    W, H = image_wh

    # 1) Draw GT annotation lane polylines as dashed white-bordered lines
    #    with diamond markers at waypoints — visually distinct from tracklets.
    if show_gt:
        for lg in annotation["lane_groups"]:
            gid = lg["group_id"]
            lanes = get_group_lanes(annotation, gid, image_wh)
            for lane in lanes:
                color = _color_for(gid, lane["cls_id"])
                pts_px = (lane["waypoints"] * np.array([W, H])).astype(np.int32)
                # White border for contrast
                cv2.polylines(vis, [pts_px], False, (255, 255, 255), 5, cv2.LINE_AA)
                # Colored line on top
                cv2.polylines(vis, [pts_px], False, color, 3, cv2.LINE_AA)
                # Diamond markers at each waypoint
                for pt in pts_px:
                    _draw_diamond(vis, tuple(pt), color, size=6)
                # Label at start
                if len(pts_px) > 0:
                    cv2.putText(
                        vis, f"G{gid}L{lane['cls_id']}",
                        tuple(pts_px[0] + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3,
                    )
                    cv2.putText(
                        vis, f"G{gid}L{lane['cls_id']}",
                        tuple(pts_px[0] + np.array([8, -8])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )

    # 2) Draw tracklet points as small circles, colored by assigned lane.
    #    Same color as the GT lane they're assigned to.
    for row in assignments.iter_rows(named=True):
        x, y = int(row["x"]), int(row["y"])
        lane_id = row["lane_id"]
        gid = row["group_id"]
        if lane_id >= 0:
            color = _color_for(gid, lane_id)
            radius = 3
        else:
            color = REJECTED_COLOR
            radius = 2
        cv2.circle(vis, (x, y), radius, color, -1)

    # 3) Draw track trails (thin lines connecting consecutive points per track)
    track_ids = assignments["track_id"].unique().to_list()
    for tid in track_ids:
        track = assignments.filter(pl.col("track_id") == tid).sort("time")
        pts = track.select("x", "y").to_numpy().astype(np.int32)
        lane_id = track["lane_id"][0]
        gid = track["group_id"][0]
        if lane_id >= 0:
            color = _color_for(gid, lane_id)
        else:
            color = REJECTED_COLOR
        if len(pts) > 1:
            cv2.polylines(vis, [pts], False, color, 1, cv2.LINE_AA)

    # Stats overlay
    total = assignments["track_id"].n_unique()
    assigned = assignments.filter(pl.col("lane_id") >= 0)["track_id"].n_unique()
    rejected = total - assigned
    lane_changes = assignments.filter(pl.col("lane_change"))["track_id"].n_unique()

    stats_text = [
        f"Tracks: {total}",
        f"Assigned: {assigned} ({100*assigned/max(total,1):.0f}%)",
        f"Rejected: {rejected}",
        f"Lane changes: {lane_changes}",
    ]
    y_off = 30
    for line in stats_text:
        cv2.putText(vis, line, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, line, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_off += 25

    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize lane assignments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--no-gt", action="store_true", help="Hide GT annotation lines")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, cfg.get("system", {}).get("log_level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    saving_path = Path(cfg.get("experiment", {}).get("saving_path", "./results/"))
    assign_dir = saving_path / "preprocessing" / "lane_assignment"
    output_dir = Path(args.output) if args.output else assign_dir / "viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    annot_dir = Path(cfg["data"].get("annotation_dir", "../dataset/preprocess"))

    # Discover cameras with assignment results
    if args.camera:
        cameras = [args.camera]
    else:
        cameras = [p.name for p in sorted(assign_dir.iterdir())
                   if p.is_dir() and (p / "lane_assignments.csv").exists()]

    for cam in cameras:
        csv_path = assign_dir / cam / "lane_assignments.csv"
        if not csv_path.exists():
            logger.warning(f"No assignments for {cam}, run `make assign` first")
            continue

        assignments = pl.read_csv(csv_path)
        if len(assignments) == 0:
            continue

        # Load annotation
        annot_path = annot_dir / cam / "annotation.json"
        if not annot_path.exists():
            logger.warning(f"No annotation for {cam}")
            continue
        annotation = load_annotation_json(str(annot_path))

        # Load frame
        frame_path = annot_dir / cam / "last_frame.npy"
        if not frame_path.exists():
            logger.warning(f"No frame for {cam}")
            continue
        frame = np.load(frame_path)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        H, W = frame.shape[:2]

        vis = draw_assignment(frame, assignments, annotation, (W, H), show_gt=not args.no_gt)

        out_path = output_dir / f"{cam}_assignment.png"
        cv2.imwrite(str(out_path), vis)
        logger.info(f"Saved {out_path}")

    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
