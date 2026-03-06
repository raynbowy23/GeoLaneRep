#!/usr/bin/env python3
"""Visualize data-driven lanelet graph vs annotation polylines on camera frame.

Draws both side-by-side for comparison:
  - Annotation lanes: dashed white-bordered lines with diamond waypoints
  - Data-driven lanes: solid colored lines with circle waypoints
  - Adjacent edges: thin dotted lines between paired lanes

Usage:
    python scripts/visualize_lanelet_graph.py --config configs/lanelet_core.yaml
    python scripts/visualize_lanelet_graph.py --config configs/lanelet_core.yaml --camera US12_Greenway
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
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


def _color_for(group_id: int, lane_id: int, max_lanes: int = 6) -> tuple:
    idx = group_id * max_lanes + lane_id
    return PALETTE[idx % len(PALETTE)]


def _draw_diamond(img, center, color, size=6):
    cx, cy = int(center[0]), int(center[1])
    pts = np.array([
        [cx, cy - size], [cx + size, cy],
        [cx, cy + size], [cx - size, cy],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_lanelet_comparison(
    frame: np.ndarray,
    lanelet_data: dict,
    annotation: dict,
    image_wh: tuple,
) -> np.ndarray:
    """Draw annotation lanes and data-driven lanelets on the same frame.

    Left legend shows which is which:
      - Dashed thick lines + diamonds = annotation (GT geometry)
      - Solid lines + circles = data-driven (from trajectories)
    """
    W, H = image_wh
    vis = frame.copy()

    # ── 1) Annotation lanes (dashed, diamonds) ──────────────
    for lg in annotation["lane_groups"]:
        gid = lg["group_id"]
        lanes = get_group_lanes(annotation, gid, image_wh)
        for lane in lanes:
            color = _color_for(gid, lane["cls_id"])
            pts_px = (lane["waypoints"] * np.array([W, H])).astype(np.int32)
            # White border
            cv2.polylines(vis, [pts_px], False, (255, 255, 255), 5, cv2.LINE_AA)
            # Dashed colored line (draw segments with gaps)
            for i in range(0, len(pts_px) - 1, 2):
                end = min(i + 2, len(pts_px))
                cv2.polylines(vis, [pts_px[i:end]], False, color, 3, cv2.LINE_AA)
            # Diamond markers
            for pt in pts_px:
                _draw_diamond(vis, pt, color, size=5)
            # Label
            if len(pts_px) > 0:
                label = f"GT G{gid}L{lane['cls_id']}"
                pos = tuple(pts_px[0] + np.array([8, -12]))
                cv2.putText(vis, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            color, 1, cv2.LINE_AA)

    # ── 2) Data-driven lanelet centerlines (solid, circles) ──
    positions = lanelet_data["positions"]  # (G, 2) normalized
    tangents = lanelet_data["tangents"]
    lane_ids = lanelet_data["lane_ids"]
    group_ids = lanelet_data["group_ids"]
    edge_index = lanelet_data["edge_index"]  # (2, E)
    edge_types = lanelet_data["edge_types"]  # (E,)

    # De-normalize to pixel
    wh = np.array([W, H], dtype=np.float64)
    pos_px = positions * wh

    # Identify unique lanes by (group_id, lane_id) contiguous ranges
    # We use lane_meta-like grouping from lane_ids
    unique_lanes = np.unique(lane_ids)

    # Map internal lane_id → (group_id, original cls_id) from the npz
    # Since npz doesn't store cls_id, reconstruct from the build order:
    # each unique lane_id maps to its group_id (all nodes in that lane share it)
    # and we need the annotation cls_id. We can recover it from lane_assignments.csv
    # but for viz, we just use lane_id as the color index within its group.

    # Group waypoints by lane for polyline drawing
    for lid in unique_lanes:
        mask = lane_ids == lid
        lane_pos = pos_px[mask]
        gid = int(group_ids[mask][0])

        # Find the cls_id: count which lane this is within its group
        lanes_in_group = np.unique(lane_ids[group_ids == gid])
        cls_idx = int(np.searchsorted(lanes_in_group, lid))

        color = _color_for(gid, cls_idx)

        pts = lane_pos.astype(np.int32)
        if len(pts) > 1:
            # Solid colored line
            cv2.polylines(vis, [pts], False, color, 2, cv2.LINE_AA)

        # Circle markers at waypoints (subsample for readability)
        step = max(1, len(pts) // 20)
        for i in range(0, len(pts), step):
            cv2.circle(vis, tuple(pts[i]), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(pts[i]), 4, (255, 255, 255), 1, cv2.LINE_AA)

        # Label at start
        if len(pts) > 0:
            label = f"DD G{gid}L{cls_idx}"
            pos = tuple(pts[0] + np.array([-8, 12]))
            cv2.putText(vis, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        color, 1, cv2.LINE_AA)

    # ── 3) Draw tangent arrows (subsample) ───────────────────
    arrow_step = max(1, len(pos_px) // 40)
    for i in range(0, len(pos_px), arrow_step):
        pt = pos_px[i].astype(int)
        t = tangents[i]
        end = (pt + t * 15).astype(int)
        gid = int(group_ids[i])
        lid = int(lane_ids[i])
        lanes_in_group = np.unique(lane_ids[group_ids == gid])
        cls_idx = int(np.searchsorted(lanes_in_group, lid))
        color = _color_for(gid, cls_idx)
        cv2.arrowedLine(vis, tuple(pt), tuple(end), color, 1, cv2.LINE_AA, tipLength=0.4)

    # ── 4) Draw adjacent edges (thin gray) ───────────────────
    if edge_index.shape[1] > 0:
        adj_mask = edge_types == 4
        adj_edges = edge_index[:, adj_mask]
        # Subsample adjacent edges for readability
        n_adj = adj_edges.shape[1]
        step = max(1, n_adj // 200)
        for k in range(0, n_adj, step):
            i, j = adj_edges[0, k], adj_edges[1, k]
            pt1 = pos_px[i].astype(int)
            pt2 = pos_px[j].astype(int)
            cv2.line(vis, tuple(pt1), tuple(pt2), (200, 200, 200), 1, cv2.LINE_AA)

    # ── Legend ────────────────────────────────────────────────
    legend_y = 30
    cv2.putText(vis, "--- Diamond: Annotation (GT)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis, "--- Diamond: Annotation (GT)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
    legend_y += 22
    cv2.putText(vis, "--- Circle:  Data-driven (DD)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis, "--- Circle:  Data-driven (DD)", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    legend_y += 22
    n_wp = len(positions)
    n_lanes = len(unique_lanes)
    n_edges = edge_index.shape[1]
    cv2.putText(vis, f"DD: {n_lanes} lanes, {n_wp} wpts, {n_edges} edges",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis, f"DD: {n_lanes} lanes, {n_wp} wpts, {n_edges} edges",
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize lanelet graph vs annotation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=getattr(logging, cfg.get("system", {}).get("log_level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    saving_path = Path(cfg.get("experiment", {}).get("saving_path", "./results/"))
    assign_dir = saving_path / "lane_assignment"
    output_dir = Path(args.output) if args.output else assign_dir / "viz_lanelet"
    output_dir.mkdir(parents=True, exist_ok=True)

    annot_dir = Path(cfg["data"].get("annotation_dir", "../dataset/preprocess"))

    if args.camera:
        cameras = [args.camera]
    else:
        cameras = [p.name for p in sorted(assign_dir.iterdir())
                   if p.is_dir() and (p / "lanelet_graph.npz").exists()]

    if not cameras:
        logger.error("No cameras with lanelet_graph.npz found. Run `make lanelet-graph` first.")
        sys.exit(1)

    for cam in cameras:
        npz_path = assign_dir / cam / "lanelet_graph.npz"
        if not npz_path.exists():
            logger.warning(f"No lanelet_graph.npz for {cam}")
            continue

        annot_path = annot_dir / cam / "annotation.json"
        if not annot_path.exists():
            logger.warning(f"No annotation for {cam}")
            continue

        frame_path = annot_dir / cam / "last_frame.npy"
        if not frame_path.exists():
            logger.warning(f"No frame for {cam}")
            continue

        # Load data
        lanelet_data = dict(np.load(npz_path))
        annotation = load_annotation_json(str(annot_path))
        frame = np.load(frame_path)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        H, W = frame.shape[:2]

        vis = draw_lanelet_comparison(frame, lanelet_data, annotation, (W, H))

        out_path = output_dir / f"{cam}_lanelet.png"
        cv2.imwrite(str(out_path), vis)
        logger.info(f"Saved {out_path}")

    logger.info(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
