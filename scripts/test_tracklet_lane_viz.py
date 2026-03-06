"""Visualize tracklet graph + SUMO lane boundaries + lane group contours.

Produces two figures per camera:
  1. {cam}.png -- tracklet graph + SUMO lane boundaries
     - Tracklet centroids (colored by vehicle) with tangent arrows
     - Per-vehicle polyline trajectories
     - Graph edges (thin gray)
     - SUMO lane boundaries (colored polylines, GPS->pixel, bbox-clipped)

  2. {cam}_lane_groups.png -- lane group contours + per-group tracklets
     - Semi-transparent filled contours per lane group
     - Tracklets colored by lane group assignment
     - Heading labels per group

Usage:
    python scripts/test_tracklet_lane_viz.py --config configs/lanelet.yaml
    python scripts/test_tracklet_lane_viz.py --config configs/lanelet.yaml --camera US12_Park
"""

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = PROJECT_ROOT.parent / "graph_geolane"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V1_ROOT))
sys.path.insert(0, str(V1_ROOT / "src"))

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Distinct colors for vehicle tracks (BGR)
TRACK_COLORS = [
    (255,  80,  80), ( 80, 255,  80), ( 80,  80, 255),
    (255, 255,  80), (255,  80, 255), ( 80, 255, 255),
    (180, 120, 255), (255, 180, 100), (100, 180, 255),
    (180, 255, 100), (255, 100, 180), (100, 255, 180),
    (200, 150,  80), ( 80, 200, 150), (150,  80, 200),
    (220, 220, 100),
]

# SUMO lane colors (BGR) -- brighter, thicker lines
LANE_COLORS = [
    (0, 200, 200), (200, 200, 0), (200, 0, 200), (0, 200, 0),
    (200, 100, 0), (0, 100, 200), (100, 200, 100), (200, 0, 100),
    (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0),
]


# -- Liang-Barsky line-segment clipping ------------------------------

def _clip_segment(p0, p1, xmin, ymin, xmax, ymax):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in [(-dx, p0[0] - xmin), (dx, xmax - p0[0]),
                 (-dy, p0[1] - ymin), (dy, ymax - p0[1])]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return None
    return (p0[0] + t0 * dx, p0[1] + t0 * dy), (p0[0] + t1 * dx, p0[1] + t1 * dy)


def clip_lane_to_bbox(lane_px, bbox, margin=30.0):
    """Clip lane polyline to bbox with segment-level Liang-Barsky clipping."""
    if lane_px is None or len(lane_px) < 2:
        return None
    xmin, ymin = bbox[0] - margin, bbox[1] - margin
    xmax, ymax = bbox[2] + margin, bbox[3] + margin
    clipped = []
    for i in range(len(lane_px) - 1):
        seg = _clip_segment(lane_px[i], lane_px[i + 1], xmin, ymin, xmax, ymax)
        if seg is None:
            if clipped:
                break
            continue
        if not clipped:
            clipped.append(seg[0])
        clipped.append(seg[1])
    return np.array(clipped, dtype=np.float32) if len(clipped) >= 2 else None


def gps_to_pixels(pixel_hom, gps_pts):
    """Convert GPS (lon, lat) -> pixel via inv(pixel_hom)."""
    gps = np.asarray(gps_pts, dtype=np.float64)
    N = gps.shape[0]
    hom_inv = np.linalg.pinv(pixel_hom)
    gps_h = np.hstack([gps[:, :2], np.ones((N, 1), dtype=np.float64)])
    px_h = (hom_inv @ gps_h.T).T
    px = px_h[:, :2] / px_h[:, 2:3]
    return px.astype(np.float32)


# HSV-spaced group colors (BGR)
def _group_color(idx, total):
    hue = int(180 * idx / max(total, 1)) % 180
    hsv = np.uint8([[[hue, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_lane_groups(frame, traj, graph, config, camera_name):
    """Draw lane group contours + per-group tracklets on frame.

    Args:
        frame: (H, W, 3) BGR camera frame.
        traj: full trajectory DataFrame (not time-windowed).
        graph: PyG Data from build_tracklet_graph() (time-windowed).
        config: YAML config dict.
        camera_name: camera name string.

    Returns:
        (H, W, 3) annotated frame.
    """
    from src.data.density_contours import detect_lane_groups, assign_vehicles_to_lane_groups

    vis = frame.copy()
    H, W = vis.shape[:2]
    data_cfg = config.get("data", config)
    method = data_cfg.get("lane_group_method", "dbscan")

    # Build kwargs for lane group detection
    lg_kwargs = dict(
        min_track_points=data_cfg.get("tracklet_min_points", 5),
        min_gap_deg=data_cfg.get("direction_min_gap_deg", 45.0),
        min_vehicles_per_group=data_cfg.get("min_vehicles_per_group", 3),
    )
    if method == "dbscan":
        lg_kwargs.update(
            hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
            dbscan_min_samples=data_cfg.get("dbscan_min_samples", 3),
        )
        if "dbscan_eps" in data_cfg:
            lg_kwargs["dbscan_eps"] = data_cfg["dbscan_eps"]
    elif method == "community":
        lg_kwargs.update(
            hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
            community_resolution=data_cfg.get("community_resolution", 1.0),
            scale_sigma=data_cfg.get("scale_penalty_sigma", 0.5),
            edge_radius=data_cfg.get("lane_group_edge_radius", 150.0),
        )
    else:  # density
        lg_kwargs.update(
            tracklet_length=data_cfg.get("tracklet_length", 15),
            sigma_along=data_cfg.get("density_sigma_along", 25.0),
            sigma_across=data_cfg.get("density_sigma_across", 6.0),
            neck_ratio=data_cfg.get("neck_ratio", 0.4),
            refine_contours=data_cfg.get("refine_contours", True),
            refine_buffer_px=data_cfg.get("refine_buffer_px", 25.0),
        )

    # Detect lane groups
    contours, group_headings, track_stats = detect_lane_groups(
        traj, (H, W), method=method, **lg_kwargs)

    if not contours:
        cv2.putText(vis, f"{camera_name}: No lane groups detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    n_groups = len(contours)
    logger.info(f"  Lane groups: {n_groups} ({method})")

    # Assign vehicles to groups
    groups = assign_vehicles_to_lane_groups(
        traj, (H, W), method=method,
        precomputed=(contours, group_headings, track_stats),
        **lg_kwargs)

    # Build vehicle -> group_id mapping
    vid_to_gid = {}
    for group_traj, heading, gid in groups:
        for vid in group_traj["id"].unique().to_list():
            vid_to_gid[vid] = gid

    # -- 1. Draw semi-transparent filled contours (bottom layer) --
    overlay = vis.copy()
    for i, cnt in enumerate(contours):
        color = _group_color(i, n_groups)
        cv2.drawContours(overlay, [cnt.astype(np.int32)], -1, color, -1)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # -- 2. Draw contour outlines + heading labels --
    for i, cnt in enumerate(contours):
        color = _group_color(i, n_groups)
        cv2.drawContours(vis, [cnt.astype(np.int32)], -1, color, 2, cv2.LINE_AA)
        # Label at contour centroid
        M = cv2.moments(cnt.astype(np.int32))
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(cnt[:, 0, 0].mean()), int(cnt[:, 0, 1].mean())
        heading_deg = np.degrees(group_headings.get(i, 0.0))
        label = f"G{i} {heading_deg:.0f}deg"
        cv2.putText(vis, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # -- 3. Draw tracklets colored by lane group --
    if graph is not None:
        pixel_c = graph.pixel_centroids.numpy()
        pixel_t = graph.pixel_tangents.numpy()
        track_ids = graph.track_ids.numpy()
        N = graph.num_nodes
        arrow_len = 18

        for i in range(N):
            cx, cy = int(pixel_c[i, 0]), int(pixel_c[i, 1])
            vid = int(track_ids[i])
            gid = vid_to_gid.get(vid, -1)
            if gid >= 0:
                color = _group_color(gid, n_groups)
            else:
                color = (120, 120, 120)  # unassigned = gray
            cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)
            dx = int(pixel_t[i, 0] * arrow_len)
            dy = int(pixel_t[i, 1] * arrow_len)
            cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy),
                            color, 1, tipLength=0.35, line_type=cv2.LINE_AA)

    # -- Stats overlay --
    n_assigned = sum(1 for v in vid_to_gid.values() if v >= 0)
    n_total_veh = traj["id"].n_unique()
    y = 25
    lines = [
        f"{camera_name} | {n_groups} lane groups ({method})",
        f"Vehicles assigned: {n_assigned}/{n_total_veh}",
    ]
    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    return vis


def draw_centroids(frame, graph, camera_name):
    """Draw only tracklet centroids -- large circles colored by vehicle with tangent arrows.

    No graph edges, no polylines, no SUMO lanes. Clean centroid-only view.
    """
    vis = frame.copy()
    if graph is None:
        return vis

    pixel_c = graph.pixel_centroids.numpy()
    pixel_t = graph.pixel_tangents.numpy()
    track_ids = graph.track_ids.numpy()
    N = graph.num_nodes

    unique_tids = np.unique(track_ids)
    tid_to_color = {tid: TRACK_COLORS[i % len(TRACK_COLORS)]
                    for i, tid in enumerate(unique_tids)}

    # Draw centroids as larger circles with tangent arrows
    arrow_len = 25
    for i in range(N):
        cx, cy = int(pixel_c[i, 0]), int(pixel_c[i, 1])
        color = tid_to_color[track_ids[i]]
        # Larger filled circle with black outline
        cv2.circle(vis, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)
        # Tangent arrow
        dx = int(pixel_t[i, 0] * arrow_len)
        dy = int(pixel_t[i, 1] * arrow_len)
        if abs(dx) + abs(dy) > 2:
            cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy),
                            color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # Stats overlay
    n_veh = len(unique_tids)
    y = 25
    lines = [
        f"{camera_name} | Centroids: {N} tracklets, {n_veh} vehicles",
    ]
    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    return vis


# -- Main drawing ----------------------------------------------------

def draw_combined(frame, graph, metadata, camera_name, traj_px):
    """Draw tracklet graph + SUMO lane boundaries on frame."""
    from src.data.canonical import decanonicalize_polyline

    vis = frame.copy()
    H, W = vis.shape[:2]

    # -- Trajectory bounding box (p2-p98 for dense region) --
    traj_bbox = (0.0, 0.0, float(W), float(H))
    if traj_px is not None and len(traj_px) >= 2:
        traj_bbox = (
            float(np.percentile(traj_px[:, 0], 2)),
            float(np.percentile(traj_px[:, 1], 2)),
            float(np.percentile(traj_px[:, 0], 98)),
            float(np.percentile(traj_px[:, 1], 98)),
        )

    # -- 1. SUMO lane boundaries (bottom layer) --
    gps_lane_geom = metadata.get("gps_lane_geom", {})
    pixel_hom = metadata.get("pixel_hom", None)
    lane_headings = metadata.get("lane_headings", {})
    n_lanes = 0

    if gps_lane_geom and pixel_hom is not None:
        pixel_hom = np.asarray(pixel_hom, dtype=np.float64)
        for idx, (lane_id, gps_pts) in enumerate(gps_lane_geom.items()):
            gps_arr = np.array(gps_pts, dtype=np.float64)
            if gps_arr.ndim != 2 or gps_arr.shape[1] < 2 or len(gps_arr) < 2:
                continue
            px_pts = gps_to_pixels(pixel_hom, gps_arr[:, :2])
            clipped = clip_lane_to_bbox(px_pts, traj_bbox, margin=30.0)
            if clipped is None or len(clipped) < 2:
                continue
            n_lanes += 1
            color = LANE_COLORS[idx % len(LANE_COLORS)]
            pts_int = clipped.astype(np.int32)
            cv2.polylines(vis, [pts_int], isClosed=False, color=color,
                          thickness=3, lineType=cv2.LINE_AA)
            # Waypoint dots
            for j in range(len(pts_int)):
                cv2.circle(vis, tuple(pts_int[j]), 4, color, -1, cv2.LINE_AA)
                cv2.circle(vis, tuple(pts_int[j]), 4, (0, 0, 0), 1, cv2.LINE_AA)
            # Label
            mid = len(pts_int) // 2
            heading = lane_headings.get(lane_id)
            h_str = f" h={heading:.1f}" if heading is not None else ""
            cv2.putText(vis, f"{lane_id}{h_str}",
                        (int(pts_int[mid, 0]) + 6, int(pts_int[mid, 1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

    # -- 2. Graph edges (thin gray, middle layer) --
    if graph is not None:
        pixel_c = graph.pixel_centroids.numpy()
        pixel_t = graph.pixel_tangents.numpy()
        track_ids = graph.track_ids.numpy()
        edge_index = graph.edge_index.numpy()
        N = graph.num_nodes
        E = edge_index.shape[1]

        unique_tids = np.unique(track_ids)
        tid_to_color = {tid: TRACK_COLORS[i % len(TRACK_COLORS)]
                        for i, tid in enumerate(unique_tids)}

        max_edges = 2000
        n_draw = min(E, max_edges)
        indices = (np.random.choice(E, n_draw, replace=False)
                   if E > max_edges else np.arange(E))
        for idx in indices:
            i, j = edge_index[0, idx], edge_index[1, idx]
            pt1 = (int(pixel_c[i, 0]), int(pixel_c[i, 1]))
            pt2 = (int(pixel_c[j, 0]), int(pixel_c[j, 1]))
            cv2.line(vis, pt1, pt2, (60, 60, 60), 1, cv2.LINE_AA)

        # -- 3. Per-vehicle polylines --
        polylines_local = graph.polylines.numpy()
        # Use pixel-space centroid/tangent for decanonicalization
        # (polylines were built in pixel space before meter conversion)
        centroids_m = graph.centroids.numpy()
        tangents_m = graph.tangents.numpy()
        for i in range(N):
            global_poly = decanonicalize_polyline(
                polylines_local[i], pixel_c[i], pixel_t[i])
            color = tid_to_color[track_ids[i]]
            pts = global_poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=False, color=color,
                          thickness=1, lineType=cv2.LINE_AA)

        # -- 4. Tracklet centroids + tangent arrows (top layer) --
        arrow_len = 18
        for i in range(N):
            cx, cy = int(pixel_c[i, 0]), int(pixel_c[i, 1])
            color = tid_to_color[track_ids[i]]
            cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)
            dx = int(pixel_t[i, 0] * arrow_len)
            dy = int(pixel_t[i, 1] * arrow_len)
            cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy),
                            color, 1, tipLength=0.35, line_type=cv2.LINE_AA)

    # -- Stats overlay --
    y = 25
    n_nodes = graph.num_nodes if graph is not None else 0
    n_edges = graph.edge_index.shape[1] if graph is not None else 0
    n_veh = len(np.unique(graph.track_ids.numpy())) if graph is not None else 0
    lines = [
        f"{camera_name} | {n_nodes} tracklets, {n_edges} edges, {n_veh} vehicles",
        f"SUMO lanes: {n_lanes} | Arrows: tangent direction",
    ]
    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracklet graph + SUMO lane boundaries")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--camera", default=None, help="Single camera")
    parser.add_argument("--time-window", type=float, default=60.0,
                        help="Seconds of trajectory for graph (default: 60)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = (Path(args.out_dir) if args.out_dir
               else PROJECT_ROOT / "results" / "tracklet_lane_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    import polars as pl
    from src.data.tracklet_graph import build_tracklet_graph
    from src.training.trainer import TrainingPipeline

    data_cfg = config.get("data", {})
    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
    preprocess_dir = v1_dir / "results" / "preprocess"
    camera_list_path = Path(data_cfg.get(
        "camera_locations",
        str(v1_dir / "dataset" / "camera_location_list.txt"),
    ))

    cameras = []
    if camera_list_path.exists():
        cameras = [l.strip() for l in camera_list_path.read_text().splitlines()
                   if l.strip()]
    if args.camera:
        cameras = [c for c in cameras if c == args.camera] or [args.camera]

    # TrainingPipeline for _extract_sumo_metadata
    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.config = config
    pipeline.exp_dir = out_dir

    for cam in cameras:
        cam_dir = preprocess_dir / cam
        traj_path = cam_dir / "trajectory.csv"
        frame_path = cam_dir / "last_frame.npy"
        if not traj_path.exists():
            logger.warning(f"Skipping {cam}: no trajectory.csv")
            continue

        traj_full = pl.read_csv(str(traj_path))
        frame = (np.load(str(frame_path)) if frame_path.exists()
                 else np.zeros((720, 1280, 3), dtype=np.uint8))
        H, W = frame.shape[:2]

        logger.info(f"\n=== {cam} === ({W}x{H})")

        # Time-windowed trajectories for tracklet graph
        traj = traj_full
        if "time" in traj.columns:
            t_min = traj["time"].min()
            traj = traj.filter(pl.col("time") < t_min + args.time_window)
        logger.info(f"  Traj: {traj.shape[0]} pts, {traj['id'].n_unique()} vehicles "
                    f"({args.time_window}s window)")

        # Build tracklet graph
        graph = build_tracklet_graph(traj, (H, W), config)
        if graph is None:
            logger.warning(f"  Skipping {cam}: too few tracklets")
            continue
        logger.info(f"  Graph: {graph.num_nodes} nodes, "
                    f"{graph.edge_index.shape[1]} edges")

        # SUMO metadata
        metadata = pipeline._extract_sumo_metadata(cam, traj_full, v1_dir)
        if not metadata:
            metadata = {}
            logger.warning(f"  No SUMO metadata for {cam}")

        # Trajectory pixel positions for bbox
        traj_px = None
        if "x" in traj_full.columns and "y" in traj_full.columns:
            traj_px = traj_full.select(["x", "y"]).to_numpy().astype(np.float64)

        # Figure 1: tracklet graph + SUMO lanes
        vis = draw_combined(frame, graph, metadata, cam, traj_px)
        save_path = out_dir / f"{cam}.png"
        cv2.imwrite(str(save_path), vis)
        logger.info(f"  Saved: {save_path}")

        # Figure 2: lane group contours + per-group tracklets
        vis_lg = draw_lane_groups(frame, traj_full, graph, config, cam)
        save_path_lg = out_dir / f"{cam}_lane_groups.png"
        cv2.imwrite(str(save_path_lg), vis_lg)
        logger.info(f"  Saved: {save_path_lg}")

        # Figure 3: centroids only (large circles + tangent arrows)
        vis_c = draw_centroids(frame, graph, cam)
        save_path_c = out_dir / f"{cam}_centroids.png"
        cv2.imwrite(str(save_path_c), vis_c)
        logger.info(f"  Saved: {save_path_c}")

    logger.info(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
