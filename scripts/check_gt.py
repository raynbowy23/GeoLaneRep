#!/usr/bin/env python3
"""Check pseudo GT and SUMO-matched targets before training.

Visualizes:
  - Tracklet centroids colored by pseudo-lane label
  - GT lanelet waypoints (from SUMO matching)
  - GT lanelet edges (successor/adjacent)
  - SUMO target polylines

Usage:
    python scripts/check_gt.py --config configs/lanelet.yaml --split val
    python scripts/check_gt.py --config configs/lanelet_core.yaml --split val
    python scripts/check_gt.py --cache_dir results/lanelet_discovery/graph_cache --split val
"""

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import yaml

from src.utils.homography_scale import local_meters_to_pixels

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LANE_COLORS = [
    (0, 0, 255), (255, 128, 0), (0, 200, 0), (0, 200, 255),
    (200, 0, 200), (200, 200, 0), (128, 0, 0), (0, 128, 128),
    (100, 100, 255), (255, 200, 100), (100, 255, 100), (255, 100, 200),
]

GT_NODE_COLOR = (0, 255, 255)  # yellow
GT_EDGE_COLOR = (0, 200, 200)


def m2px(pts_m, pixel_hom, ref_gps):
    """Convert meter-space points to pixel coordinates via GPS intermediate.

    Path: meters → GPS (equirectangular) → pixels (via pinv(pixel_hom)).
    """
    pts_m = np.asarray(pts_m, dtype=np.float64)
    if pts_m.ndim == 1:
        pts_m = pts_m.reshape(1, -1)
    pixels = local_meters_to_pixels(pixel_hom, pts_m[:, :2], ref_gps)
    valid = np.isfinite(pixels).all(axis=1)
    return pixels, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output_dir", default="results/gt_check")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        data_cfg = config.get("data", {})
    else:
        data_cfg = {"v1_dir": "../graph_geolane"}

    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
    preprocess_dir = v1_dir / "results" / "preprocess"

    # Load graph cache
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif args.config:
        from src.data.dataset import TrackletDataset
        cache_path = TrackletDataset._cache_path(config, args.split)
        cache_dir = cache_path.parent if cache_path else None
    else:
        cache_dir = None

    if cache_dir:
        candidates = list(cache_dir.glob(f"{args.split}_*.pt"))
        if candidates:
            cache_path = candidates[0]
        else:
            logger.error(f"No cache found in {cache_dir}")
            sys.exit(1)
    else:
        logger.error("Provide --config or --cache_dir")
        sys.exit(1)

    samples = torch.load(cache_path, weights_only=False)
    logger.info(f"Loaded {len(samples)} graphs from {cache_path}")

    # Load camera frames
    camera_frames = {}
    for cam_dir in preprocess_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        frame_path = cam_dir / "last_frame.npy"
        if frame_path.exists():
            camera_frames[cam_dir.name] = np.load(str(frame_path))

    # Group by camera
    cam_groups = defaultdict(list)
    for s in samples:
        cam_groups[getattr(s, "camera_loc", "?")].append(s)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cam in sorted(cam_groups.keys()):
        frame = camera_frames.get(cam)
        if frame is None:
            continue

        vis = frame.copy()
        logger.info(f"\n{'='*60}")
        logger.info(f"Camera: {cam}")
        logger.info(f"{'='*60}")

        for gi, s in enumerate(cam_groups[cam]):
            heading = float(getattr(s, "lane_group_heading", 0)) * 57.3
            N = s.num_nodes

            # Get homography + ref_gps for meter→pixel conversion
            hom = getattr(s, "pixel_hom_matrix", None)
            ref_gps = getattr(s, "ref_gps", None)

            if hom is not None and ref_gps is not None:
                pixel_hom = hom.numpy().astype(np.float64)
                ref_gps_np = ref_gps.numpy().astype(np.float64)
                has_hom = True
            else:
                # Fallback: try pixel_centroids directly
                px_centroids = getattr(s, "pixel_centroids", None)
                if px_centroids is not None:
                    px_c = px_centroids.numpy()
                else:
                    logger.warning(f"  Dir {gi}: No homography/ref_gps or pixel centroids, skipping")
                    continue
                has_hom = False

            # --- Pseudo-lane labels ---
            pseudo = s.pseudo_labels.numpy()
            unique_labels = sorted(set(pseudo.tolist()))
            n_lanes = len([l for l in unique_labels if l >= 0])

            logger.info(f"\n  Dir {gi}: heading={heading:.0f}°, N={N}, pseudo_lanes={n_lanes}")
            for lbl in unique_labels:
                count = (pseudo == lbl).sum()
                logger.info(f"    Label {lbl}: {count} tracklets")

            # Draw tracklets colored by pseudo-lane label
            centroids = s.centroids.numpy()
            tangents = s.tangents.numpy()

            for j in range(N):
                lbl = pseudo[j]
                color = LANE_COLORS[lbl % len(LANE_COLORS)] if lbl >= 0 else (128, 128, 128)

                if has_hom:
                    px, valid = m2px(centroids[j:j+1], pixel_hom, ref_gps_np)
                    if not valid[0]:
                        continue
                    px_pt = (int(px[0, 0]), int(px[0, 1]))
                else:
                    px_pt = (int(px_c[j, 0]), int(px_c[j, 1]))

                if 0 <= px_pt[0] < vis.shape[1] and 0 <= px_pt[1] < vis.shape[0]:
                    cv2.circle(vis, px_pt, 3, color, -1)

                    # Draw tangent arrow
                    if has_hom:
                        end_m = centroids[j] + tangents[j] * 5.0
                        end_px, ev = m2px(end_m.reshape(1, -1), pixel_hom, ref_gps_np)
                        if ev[0]:
                            epx = (int(end_px[0, 0]), int(end_px[0, 1]))
                            cv2.arrowedLine(vis, px_pt, epx, color, 1, tipLength=0.3)

            # --- GT lanelet nodes ---
            gt_pos = getattr(s, "gt_lanelet_positions", None)
            gt_tan = getattr(s, "gt_lanelet_tangents", None)
            gt_edges = getattr(s, "gt_lanelet_edge_index", None)
            gt_edge_types = getattr(s, "gt_lanelet_edge_types", None)
            gt_lane_ids = getattr(s, "gt_lanelet_lane_ids", None)

            if gt_pos is not None and len(gt_pos) > 0:
                gt_pos_np = gt_pos.numpy()
                gt_tan_np = gt_tan.numpy() if gt_tan is not None else None
                n_gt = len(gt_pos_np)

                logger.info(f"    GT lanelet nodes: {n_gt}")
                if gt_lane_ids is not None and len(gt_lane_ids) > 0:
                    unique_gt_lanes = sorted(set(gt_lane_ids.numpy().tolist()))
                    logger.info(f"    GT lanes: {len(unique_gt_lanes)} ({unique_gt_lanes})")

                gt_range_x = gt_pos_np[:, 0].max() - gt_pos_np[:, 0].min()
                gt_range_y = gt_pos_np[:, 1].max() - gt_pos_np[:, 1].min()
                logger.info(f"    GT extent: {gt_range_x:.1f} x {gt_range_y:.1f}m")

                if gt_edges is not None and len(gt_edges[0]) > 0:
                    edge_types = gt_edge_types.numpy() if gt_edge_types is not None else None
                    type_names = ["no_edge", "successor", "merge", "diverge", "adjacent"]
                    if edge_types is not None:
                        type_counts = Counter(edge_types.tolist())
                        logger.info(f"    GT edges: {len(gt_edges[0])} " +
                                    ", ".join(f"{type_names[t]}={c}" for t, c in sorted(type_counts.items())))

                # Draw GT nodes
                if has_hom:
                    gt_px, gt_valid = m2px(gt_pos_np, pixel_hom, ref_gps_np)

                    for k in range(n_gt):
                        if not gt_valid[k]:
                            continue
                        pt = (int(gt_px[k, 0]), int(gt_px[k, 1]))
                        if 0 <= pt[0] < vis.shape[1] and 0 <= pt[1] < vis.shape[0]:
                            lane_id = int(gt_lane_ids[k]) if gt_lane_ids is not None else 0
                            gt_color = LANE_COLORS[lane_id % len(LANE_COLORS)]
                            cv2.circle(vis, pt, 8, gt_color, 2)
                            cv2.circle(vis, pt, 3, (255, 255, 255), -1)

                            # Heading arrow
                            if gt_tan_np is not None:
                                end_m = gt_pos_np[k] + gt_tan_np[k] * 8.0
                                end_px, ev = m2px(end_m.reshape(1, -1), pixel_hom, ref_gps_np)
                                if ev[0]:
                                    epx = (int(end_px[0, 0]), int(end_px[0, 1]))
                                    cv2.arrowedLine(vis, pt, epx, gt_color, 2, tipLength=0.2)

                    # Draw GT edges
                    if gt_edges is not None and len(gt_edges[0]) > 0:
                        edge_idx = gt_edges.numpy()
                        for e in range(edge_idx.shape[1]):
                            i, j_idx = edge_idx[0, e], edge_idx[1, e]
                            if not gt_valid[i] or not gt_valid[j_idx]:
                                continue
                            pt1 = (int(gt_px[i, 0]), int(gt_px[i, 1]))
                            pt2 = (int(gt_px[j_idx, 0]), int(gt_px[j_idx, 1]))
                            etype = int(edge_types[e]) if edge_types is not None else 1
                            if etype == 1:  # successor
                                cv2.arrowedLine(vis, pt1, pt2, GT_EDGE_COLOR, 1, tipLength=0.15)
                            elif etype == 4:  # adjacent
                                cv2.line(vis, pt1, pt2, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                logger.info(f"    GT lanelet nodes: NONE")

            # --- Target polylines ---
            target_poly = getattr(s, "target_polylines", None)
            target_mask = getattr(s, "target_valid_mask", None)
            if target_poly is not None and target_mask is not None:
                n_valid = target_mask.sum().item()
                logger.info(f"    Target polylines: {n_valid}/{len(target_mask)} valid")

                if has_hom:
                    for ti in range(len(target_mask)):
                        if not target_mask[ti]:
                            continue
                        poly = target_poly[ti].numpy()  # (K, 2)
                        poly_px, pv = m2px(poly, pixel_hom, ref_gps_np)
                        pts = []
                        for k in range(len(poly)):
                            if pv[k]:
                                pts.append((int(poly_px[k, 0]), int(poly_px[k, 1])))
                        if len(pts) >= 2:
                            for k in range(len(pts) - 1):
                                cv2.line(vis, pts[k], pts[k+1], (0, 255, 0), 2)

        # Add legend
        y = 30
        cv2.putText(vis, "Small dots: tracklets (color=lane)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(vis, "Large circles+arrows: GT lanelet nodes", (10, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GT_NODE_COLOR, 1)
        cv2.putText(vis, "Green lines: SUMO target polylines", (10, y+36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        save_path = str(out_dir / f"{cam}_gt.png")
        cv2.imwrite(save_path, vis)
        logger.info(f"\nSaved: {save_path}")

    logger.info(f"\nAll GT visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
