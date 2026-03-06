#!/usr/bin/env python3
"""Build graph cache and inspect tracklets without training.

Usage:
    python scripts/build_cache.py --config configs/lanelet_core.yaml
    python scripts/build_cache.py --config configs/lanelet_core.yaml --split val --visualize
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None, help="Build only this split (default: all)")
    parser.add_argument("--visualize", action="store_true", help="Render tracklets on camera frames")
    parser.add_argument("--camera", default=None, help="Process only this camera location")
    parser.add_argument("--output_dir", default="results/tracklet_check")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
    preprocess_dir = v1_dir / "results" / "preprocess"
    camera_list_path = Path(data_cfg.get(
        "camera_locations",
        str(v1_dir / "dataset" / "camera_location_list.txt"),
    ))

    cameras = []
    if camera_list_path.exists():
        cameras = [l.strip() for l in camera_list_path.read_text().splitlines() if l.strip()]
    if not cameras:
        cameras = [d.name for d in preprocess_dir.iterdir() if d.is_dir()]
    if args.camera:
        cameras = [c for c in cameras if c == args.camera]

    # Build processed_data dict (same as trainer._load_data)
    processed = {}
    camera_frames = {}
    for cam in cameras:
        cam_dir = preprocess_dir / cam
        traj_path = cam_dir / "trajectory.csv"
        frame_path = cam_dir / "last_frame.npy"
        if not traj_path.exists():
            logger.warning(f"Skipping {cam}: no trajectory.csv")
            continue
        traj = pl.read_csv(str(traj_path))
        frame = np.load(str(frame_path)) if frame_path.exists() else np.zeros((720, 1280, 3), dtype=np.uint8)
        camera_frames[cam] = frame

        class _PData:
            pass
        pdata = _PData()
        pdata.camera_loc = cam
        pdata.trajectories = traj
        pdata.frame = frame
        pdata.graph_data = {}
        pdata.contours = []
        pdata.metadata = {}

        # Extract SUMO metadata only if SUMO targets are used
        if data_cfg.get("use_sumo_targets", False):
            # Add v1 source path so OSMConnection can be imported
            v1_src = str((v1_dir / "src").resolve())
            if v1_src not in sys.path:
                sys.path.insert(0, v1_src)
            from src.training.trainer import TrainingPipeline
            exp_dir = Path(config.get("experiment", {}).get("saving_path", "./results"))
            dummy = type('T', (), {'config': config, 'exp_dir': exp_dir})()
            extracted = TrainingPipeline._extract_sumo_metadata(dummy, cam, traj, v1_dir)
            if extracted:
                pdata.metadata = extracted
                logger.info(f"  {cam}: SUMO metadata loaded ({len(extracted.get('gps_lane_geom', {}))} lanes)")

        processed[cam] = pdata

    logger.info(f"Loaded {len(processed)} cameras")

    # Build datasets (this triggers graph cache building)
    from src.data.dataset import TrackletDataset

    splits = [args.split] if args.split else ["train", "val", "test"]
    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Building {split} split...")
        logger.info(f"{'='*60}")

        # Disable cache when filtering by camera (avoid loading stale full cache)
        if args.camera:
            config.setdefault("data", {})["cache_graphs"] = False
        dataset = TrackletDataset(processed, config, split=split)
        samples = dataset.samples

        logger.info(f"\n{split}: {len(samples)} graphs total")

        # Group by camera
        cam_groups = defaultdict(list)
        for s in samples:
            cam_groups[getattr(s, "camera_loc", "?")].append(s)

        # Print stats per camera
        logger.info(f"\n{'Camera':<20} {'Dir':>5} {'Head':>6} {'N':>4} {'E':>5} "
                     f"{'Extent':>12} {'PolyLen':>16} {'GT':>3}")
        logger.info("-" * 80)

        total_tracklets = 0
        small_groups = 0
        for cam in sorted(cam_groups.keys()):
            for i, s in enumerate(cam_groups[cam]):
                heading = float(getattr(s, "lane_group_heading", 0)) * 57.3
                N = s.num_nodes
                E = s.edge_index.shape[1]
                total_tracklets += N

                centroids = s.centroids.numpy()
                ext_x = centroids[:, 0].max() - centroids[:, 0].min()
                ext_y = centroids[:, 1].max() - centroids[:, 1].min()

                polys = s.polylines.numpy()
                poly_lens = np.linalg.norm(np.diff(polys, axis=1), axis=-1).sum(axis=1)

                gt_n = len(s.gt_lanelet_positions) if hasattr(s, "gt_lanelet_positions") and s.gt_lanelet_positions is not None and len(s.gt_lanelet_positions) > 0 else 0

                flag = " *" if N < 8 else ""
                if N < 8:
                    small_groups += 1

                logger.info(f"{cam:<20} {i:>5} {heading:>5.0f}° {N:>4} {E:>5} "
                             f"{ext_x:>5.0f}×{ext_y:<5.0f}m "
                             f"{poly_lens.min():>5.1f}-{poly_lens.max():<5.1f}m  "
                             f"{gt_n:>3}{flag}")

        logger.info(f"\nSummary: {len(samples)} graphs, {total_tracklets} total tracklets, "
                     f"{small_groups} small groups (<8 tracklets)")

        # Visualize tracklets on camera frames
        if args.visualize and split in ("val", "test"):
            import cv2
            from src.data.density_contours import detect_lane_groups
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            def _hsv_color(index, total):
                if total <= 0:
                    total = 1
                hue = int(180 * index / total) % 180
                hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                return tuple(int(c) for c in bgr[0, 0])

            for cam in sorted(cam_groups.keys()):
                frame = camera_frames.get(cam)
                if frame is None:
                    continue
                vis = frame.copy()
                H_frame, W_frame = vis.shape[:2]
                n_groups = len(cam_groups[cam])

                # Draw lane group contours behind tracklets
                pdata = processed.get(cam)
                if pdata is not None:
                    traj = pdata.trajectories
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
                    contours, group_headings, _ = detect_lane_groups(
                        traj, (H_frame, W_frame), method=method, **lg_kwargs,
                    )
                    for ci, cnt in enumerate(contours):
                        color = _hsv_color(ci, len(contours))
                        overlay = vis.copy()
                        cv2.drawContours(overlay, [cnt.astype(np.int32)], -1, color, -1)
                        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
                        cv2.drawContours(vis, [cnt.astype(np.int32)], -1, color, 2, cv2.LINE_AA)
                        # Heading label
                        M = cv2.moments(cnt.astype(np.int32))
                        if M["m00"] > 0:
                            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        else:
                            pts = cnt.reshape(-1, 2)
                            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                        hdeg = np.degrees(group_headings.get(ci, 0.0)) % 360
                        label = f"G{ci} {hdeg:.0f}deg"
                        cv2.putText(vis, label, (cx - 30, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(vis, label, (cx - 30, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Draw tracklets per lane group (no edges — just points + arrows)
                for gi, s in enumerate(cam_groups[cam]):
                    image_wh_np = s.image_wh.numpy() if hasattr(s, "image_wh") and s.image_wh is not None else np.array([1920.0, 1080.0])

                    centroids = s.centroids.numpy()  # (N, 2) in (s,d) rotated frame
                    tangents = s.tangents.numpy()
                    track_ids = s.track_ids.numpy() if hasattr(s, "track_ids") else np.arange(len(centroids))

                    # Inverse-rotate (s,d) frame back to [0,1] normalized space
                    if hasattr(s, 'sd_heading') and s.sd_heading is not None:
                        h = float(s.sd_heading)
                        cos_h, sin_h = np.cos(h), np.sin(h)
                        R_inv = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                        origin = s.sd_origin.numpy()
                        centroids = (centroids @ R_inv.T) + origin
                        tangents = tangents @ R_inv.T

                    # Denormalize [0,1] to pixel coordinates
                    all_px = centroids * image_wh_np

                    # Color by vehicle ID
                    unique_vids = np.unique(track_ids)
                    rng = np.random.RandomState(42 + gi)
                    vid_colors = {vid: tuple(rng.randint(50, 256, 3).tolist()) for vid in unique_vids}

                    for j in range(len(centroids)):
                        px, py = int(all_px[j, 0]), int(all_px[j, 1])
                        vcolor = vid_colors[track_ids[j]]

                        if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
                            cv2.circle(vis, (px, py), 4, vcolor, -1)
                            cv2.circle(vis, (px, py), 4, (0, 0, 0), 1)

                            # Draw tangent arrow
                            end_px_pt = all_px[j] + tangents[j] * 20.0
                            epx, epy = int(end_px_pt[0]), int(end_px_pt[1])
                            cv2.arrowedLine(vis, (px, py), (epx, epy), vcolor, 1,
                                            tipLength=0.3)

                # Draw raw annotation GT directly from JSON (bypasses cache)
                # This verifies annotations are correct independently of the GT pipeline
                total_gt = 0
                annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
                annot_path = annot_dir / cam / "annotation.json"
                if annot_path.exists():
                    import json
                    with open(annot_path) as f:
                        annot_data = json.load(f)

                    lane_colors = [
                        (0, 220, 0), (0, 200, 255), (255, 100, 0),
                        (200, 0, 200), (0, 150, 255), (255, 255, 0),
                        (100, 255, 100), (255, 150, 150), (150, 150, 255),
                        (255, 200, 100), (100, 255, 255), (200, 100, 255),
                    ]
                    lane_idx = 0
                    for lg in annot_data.get("lane_groups", []):
                        gid = lg["group_id"]
                        for lane in lg.get("lanes", []):
                            waypoints = lane.get("waypoints", [])
                            if len(waypoints) < 2:
                                continue
                            pts_px = np.array([[wp["x"], wp["y"]] for wp in waypoints])
                            color = lane_colors[lane_idx % len(lane_colors)]
                            lane_idx += 1
                            total_gt += len(pts_px)

                            # Draw lane polyline
                            cv2.polylines(vis, [pts_px.astype(np.int32)],
                                          isClosed=False, color=color,
                                          thickness=3, lineType=cv2.LINE_AA)

                            # Draw diamond markers at each waypoint
                            for wp in pts_px:
                                gx, gy = int(wp[0]), int(wp[1])
                                diamond = np.array([
                                    [gx, gy - 7], [gx + 7, gy], [gx, gy + 7], [gx - 7, gy]
                                ], dtype=np.int32)
                                cv2.fillPoly(vis, [diamond], color)
                                cv2.polylines(vis, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)

                            # Label at first waypoint
                            lx, ly = int(pts_px[0, 0]), int(pts_px[0, 1])
                            label = f"G{gid}L{lane['cls_id']}"
                            cv2.putText(vis, label, (lx + 10, ly),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(vis, label, (lx + 10, ly),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                    logger.info(f"  {cam}: {total_gt} annotation waypoints from {lane_idx} lanes")

                # Draw cached GT (green diamonds) — this is what the training loss uses
                # Inverse-rotates from (s,d) frame back to normalized [0,1], then to pixels
                cached_gt_total = 0
                for gi, s in enumerate(cam_groups[cam]):
                    has_attr = hasattr(s, "gt_lanelet_positions")
                    is_none = s.gt_lanelet_positions is None if has_attr else True
                    n_pts = s.gt_lanelet_positions.shape[0] if (has_attr and not is_none) else 0
                    lg_id = getattr(s, "lane_group_id", "?")
                    logger.info(f"    {cam} lg={lg_id} window={gi}: has_gt_attr={has_attr} is_none={is_none} n_pts={n_pts}")
                    if not has_attr or is_none:
                        continue
                    gt_pos = s.gt_lanelet_positions.numpy()  # (G, 2) in (s,d) frame
                    if len(gt_pos) == 0:
                        continue

                    # Inverse-rotate (s,d) back to normalized [0,1]
                    gt_vis = gt_pos.copy()
                    if hasattr(s, 'sd_heading') and s.sd_heading is not None:
                        h = float(s.sd_heading)
                        cos_h, sin_h = np.cos(h), np.sin(h)
                        R_inv = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                        origin = s.sd_origin.numpy()
                        gt_vis = (gt_vis @ R_inv.T) + origin

                    image_wh_np = s.image_wh.numpy() if hasattr(s, "image_wh") and s.image_wh is not None else np.array([1920.0, 1080.0])
                    gt_px = gt_vis * image_wh_np
                    logger.info(f"      GT px range: x=[{gt_px[:,0].min():.0f},{gt_px[:,0].max():.0f}] y=[{gt_px[:,1].min():.0f},{gt_px[:,1].max():.0f}] image_wh={image_wh_np}")

                    # Draw green diamonds (same style as training visualization)
                    lane_ids = s.gt_lanelet_lane_ids.numpy() if hasattr(s, "gt_lanelet_lane_ids") and s.gt_lanelet_lane_ids is not None else np.zeros(len(gt_pos))
                    for lid in np.unique(lane_ids):
                        mask = lane_ids == lid
                        lane_pts = gt_px[mask]
                        if len(lane_pts) >= 2:
                            cv2.polylines(vis, [lane_pts.astype(np.int32)],
                                          isClosed=False, color=(0, 220, 0),
                                          thickness=2, lineType=cv2.LINE_AA)

                    for g in range(len(gt_px)):
                        gx, gy = int(gt_px[g, 0]), int(gt_px[g, 1])
                        diamond = np.array([
                            [gx, gy - 7], [gx + 7, gy], [gx, gy + 7], [gx - 7, gy]
                        ], dtype=np.int32)
                        cv2.fillPoly(vis, [diamond], (0, 220, 0))
                        cv2.polylines(vis, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)
                    cached_gt_total += len(gt_pos)

                if cached_gt_total > 0:
                    logger.info(f"  {cam}: {cached_gt_total} cached GT waypoints (training target)")
                else:
                    logger.warning(f"  {cam}: NO cached GT waypoints!")

                # Header with stats
                total_nodes = sum(s.num_nodes for s in cam_groups[cam])
                total_edges = sum(s.edge_index.shape[1] for s in cam_groups[cam])
                total_vids = sum(len(np.unique(s.track_ids.numpy())) for s in cam_groups[cam] if hasattr(s, "track_ids"))
                y_off = 25
                for text in [
                    f"Nodes: {total_nodes}",
                    f"Edges: {total_edges}",
                    f"Vehicles: {total_vids}",
                    f"Lane groups: {n_groups}",
                    f"Annot waypoints: {total_gt}",
                    f"Cached GT: {cached_gt_total} (green diamonds)",
                ]:
                    cv2.putText(vis, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(vis, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 0), 1, cv2.LINE_AA)
                    y_off += 22

                save_path = str(out_dir / f"{cam}_{split}_tracklets.png")
                cv2.imwrite(save_path, vis)
                logger.info(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
