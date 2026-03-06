"""Diagnostic: render SUMO lane geometry on camera frames via GPS→pixel.

Lanes are interpolated for smooth, dense waypoints and clipped to the
detected lane group contours so they never extend beyond the observed
road boundaries.

Usage:
    python scripts/test_gt_viz.py --config configs/lanelet.yaml
    python scripts/test_gt_viz.py --config configs/lanelet.yaml --camera US12_Park
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

# Lane colors (BGR)
LANE_COLORS = [
    (0, 200, 200), (200, 200, 0), (200, 0, 200), (0, 200, 0),
    (200, 100, 0), (0, 100, 200), (100, 200, 100), (200, 0, 100),
    (0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0),
]


def gps_to_pixels(pixel_hom: np.ndarray, gps_pts: np.ndarray) -> np.ndarray:
    """Convert GPS (lon, lat) to pixel coordinates using inv(pixel_hom)."""
    gps = np.asarray(gps_pts, dtype=np.float64)
    N = gps.shape[0]
    hom_inv = np.linalg.pinv(pixel_hom)
    gps_h = np.hstack([gps[:, :2], np.ones((N, 1), dtype=np.float64)])
    px_h = (hom_inv @ gps_h.T).T
    px = px_h[:, :2] / px_h[:, 2:3]
    return px.astype(np.float32)


def interpolate_polyline(pts: np.ndarray, spacing_px: float = 5.0) -> np.ndarray:
    """Resample a polyline to evenly-spaced points along its arc length.

    Args:
        pts: (L, 2) polyline points.
        spacing_px: approximate spacing between output points in pixels.

    Returns:
        (K, 2) interpolated points.
    """
    if pts is None or len(pts) < 2:
        return pts
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1.0:
        return pts

    n_out = max(int(total / spacing_px), 2)
    targets = np.linspace(0.0, total, n_out)
    result = np.empty((n_out, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j < len(cum_len) - 2 and cum_len[j + 1] < t:
            j += 1
        seg_start = cum_len[j]
        seg_end = cum_len[j + 1] if j + 1 < len(cum_len) else cum_len[j]
        seg_range = seg_end - seg_start
        alpha = (t - seg_start) / seg_range if seg_range > 1e-8 else 0.0
        result[i] = pts[j] * (1.0 - alpha) + pts[min(j + 1, len(pts) - 1)] * alpha
    return result


def build_contour_mask(contours, H, W, dilate_px: int = 30):
    """Build a binary mask from a list of OpenCV contours.

    Args:
        contours: list of OpenCV contour arrays.
        H, W: frame dimensions.
        dilate_px: pixels to dilate the mask to tolerate calibration error.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(mask, [cnt.astype(np.int32)], -1, 255, -1)
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def clip_lane_to_contour_mask(lane_px: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clip a lane polyline to the region covered by contour mask.

    Keeps the largest contiguous run of points that are inside the mask.

    Args:
        lane_px: (L, 2) interpolated lane points in pixel coordinates.
        mask: (H, W) uint8 binary mask (255 = inside contours).

    Returns:
        Clipped (K, 2) array or None.
    """
    if lane_px is None or len(lane_px) < 2:
        return None

    H, W = mask.shape
    inside = np.zeros(len(lane_px), dtype=bool)
    for i in range(len(lane_px)):
        x, y = int(round(lane_px[i, 0])), int(round(lane_px[i, 1]))
        if 0 <= x < W and 0 <= y < H:
            inside[i] = mask[y, x] > 0

    if inside.sum() < 2:
        return None

    # Largest contiguous run
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, v in enumerate(inside):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    if best_len < 2:
        return None

    return lane_px[best_start:best_start + best_len]


def detect_contours(traj, frame_shape, config):
    """Detect lane group contours using the configured method."""
    from src.data.density_contours import detect_lane_groups

    data_cfg = config.get("data", config)
    method = data_cfg.get("lane_group_method", "dbscan")
    H, W = frame_shape

    kwargs = dict(
        min_track_points=data_cfg.get("tracklet_min_points", 5),
        min_gap_deg=data_cfg.get("direction_min_gap_deg", 45.0),
        min_vehicles_per_group=data_cfg.get("min_vehicles_per_group", 3),
    )
    if method == "dbscan":
        kwargs.update(
            hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
            dbscan_min_samples=data_cfg.get("dbscan_min_samples", 3),
        )
        if "dbscan_eps" in data_cfg:
            kwargs["dbscan_eps"] = data_cfg["dbscan_eps"]
    elif method == "community":
        kwargs.update(
            hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
            community_resolution=data_cfg.get("community_resolution", 1.0),
            scale_sigma=data_cfg.get("scale_penalty_sigma", 0.5),
            edge_radius=data_cfg.get("lane_group_edge_radius", 150.0),
        )
    else:  # density
        kwargs.update(
            tracklet_length=data_cfg.get("tracklet_length", 15),
            sigma_along=data_cfg.get("density_sigma_along", 25.0),
            sigma_across=data_cfg.get("density_sigma_across", 6.0),
            neck_ratio=data_cfg.get("neck_ratio", 0.4),
            refine_contours=data_cfg.get("refine_contours", True),
            refine_buffer_px=data_cfg.get("refine_buffer_px", 25.0),
        )

    contours, group_headings, _ = detect_lane_groups(
        traj, (H, W), method=method, **kwargs)
    return contours, group_headings


def draw_gps_lanes_on_frame(frame, metadata, camera_name, contour_mask,
                            contours=None, group_headings=None):
    """Draw interpolated, contour-clipped SUMO lanes on frame.

    Args:
        frame: (H, W, 3) BGR camera frame.
        metadata: SUMO metadata dict with gps_lane_geom, pixel_hom, etc.
        camera_name: camera location name for header.
        contour_mask: (H, W) uint8 binary mask for clipping.
        contours: optional list of contour arrays (for drawing outlines).
        group_headings: optional {gid: heading_rad} for contour labels.
    """
    vis = frame.copy()
    H, W = vis.shape[:2]

    gps_lane_geom = metadata.get("gps_lane_geom", {})
    pixel_hom = metadata.get("pixel_hom", None)
    lane_headings = metadata.get("lane_headings", {})

    # Draw contour outlines as thin dashed white for reference
    if contours:
        for i, cnt in enumerate(contours):
            cv2.drawContours(vis, [cnt.astype(np.int32)], -1,
                             (200, 200, 200), 1, cv2.LINE_AA)
            # Heading label at centroid
            if group_headings:
                M = cv2.moments(cnt.astype(np.int32))
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hdeg = np.degrees(group_headings.get(i, 0.0))
                    cv2.putText(vis, f"G{i} {hdeg:.0f}deg", (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

    if not gps_lane_geom:
        cv2.putText(vis, f"{camera_name}: NO gps_lane_geom in metadata", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    if pixel_hom is None:
        cv2.putText(vis, f"{camera_name}: NO pixel_hom in metadata", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    pixel_hom = np.asarray(pixel_hom, dtype=np.float64)

    n_lanes = 0
    n_in_frame = 0

    for idx, (lane_id, gps_pts) in enumerate(gps_lane_geom.items()):
        gps_arr = np.array(gps_pts, dtype=np.float64)
        if gps_arr.ndim != 2 or gps_arr.shape[1] < 2 or len(gps_arr) < 2:
            continue

        # Direct GPS → pixel conversion
        px_pts = gps_to_pixels(pixel_hom, gps_arr[:, :2])

        # Interpolate to dense, evenly-spaced points
        px_pts = interpolate_polyline(px_pts, spacing_px=5.0)

        color = LANE_COLORS[idx % len(LANE_COLORS)]
        n_lanes += 1

        # Clip to lane group contour mask
        clipped = clip_lane_to_contour_mask(px_pts, contour_mask)
        if clipped is None or len(clipped) < 2:
            continue
        n_in_frame += 1

        # Draw as polyline
        pts_int = clipped.astype(np.int32)
        cv2.polylines(vis, [pts_int], isClosed=False, color=color, thickness=3,
                      lineType=cv2.LINE_AA)

        # Draw waypoint dots (subsample for readability — every ~15px along arc)
        step = max(1, len(pts_int) // max(1, int(
            np.linalg.norm(np.diff(pts_int.astype(float), axis=0), axis=1).sum() / 15)))
        for j in range(0, len(pts_int), step):
            cx, cy = int(pts_int[j, 0]), int(pts_int[j, 1])
            cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)
        # Always draw last point
        cx, cy = int(pts_int[-1, 0]), int(pts_int[-1, 1])
        cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Label at midpoint
        mid = len(pts_int) // 2
        mx, my = int(pts_int[mid, 0]), int(pts_int[mid, 1])
        heading = lane_headings.get(lane_id)
        h_str = f" h={heading:.2f}" if heading is not None else ""
        label = f"{lane_id}{h_str}"
        cv2.putText(vis, label, (mx + 8, my - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        logger.info(f"  {lane_id}: {len(gps_arr)} GPS -> {len(clipped)} interp pts")

    # Header
    y = 25
    n_contours = len(contours) if contours else 0
    lines = [
        f"SUMO GT | {camera_name} | {n_in_frame}/{n_lanes} lanes (contour-clipped)",
        f"Interpolated 5px spacing | {n_contours} lane group contours",
    ]

    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    return vis


# Fitted-lane colors (BGR) — distinct from SUMO colors
FITTED_COLORS = [
    (50, 255, 50), (255, 50, 50), (50, 50, 255),
    (50, 255, 255), (255, 50, 255), (255, 255, 50),
    (100, 200, 255), (255, 200, 100), (200, 100, 255),
    (100, 255, 200), (255, 100, 200), (200, 255, 100),
]


def draw_fitted_lanes_on_frame(frame, fitted_lanes, sumo_lanes_px,
                                matches, contour_mask, camera_name,
                                contours=None, group_headings=None):
    """Draw trajectory-fitted lanelets on frame.

    Shows fitted lanelets (solid, thick) and matched SUMO lanes (dashed, thin)
    for comparison.

    Args:
        frame: (H, W, 3) BGR camera frame.
        fitted_lanes: {label: (K, 2)} fitted centerlines in pixel coords.
        sumo_lanes_px: {lane_id: (L, 2)} SUMO lanes in pixel coords.
        matches: {sumo_lane_id: fitted_label} matching.
        contour_mask: (H, W) uint8 mask for clipping.
        camera_name: camera name string.
        contours: optional contour outlines.
        group_headings: optional group headings.
    """
    vis = frame.copy()
    H, W = vis.shape[:2]

    # Draw contour outlines
    if contours:
        for i, cnt in enumerate(contours):
            cv2.drawContours(vis, [cnt.astype(np.int32)], -1,
                             (200, 200, 200), 1, cv2.LINE_AA)
            if group_headings:
                M = cv2.moments(cnt.astype(np.int32))
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hdeg = np.degrees(group_headings.get(i, 0.0))
                    cv2.putText(vis, f"G{i} {hdeg:.0f}deg", (cx - 30, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)

    # Draw matched SUMO lanes as dashed thin lines (reference)
    reverse_matches = {}  # fitted_label -> sumo_lane_id
    for sid, flbl in matches.items():
        reverse_matches.setdefault(flbl, []).append(sid)

    for sid, pts in sumo_lanes_px.items():
        if sid not in matches:
            continue
        interp = interpolate_polyline(pts, spacing_px=5.0)
        clipped = clip_lane_to_contour_mask(interp, contour_mask)
        if clipped is None or len(clipped) < 2:
            continue
        pts_int = clipped.astype(np.int32)
        # Dashed line
        for j in range(0, len(pts_int) - 1, 2):
            p1 = (int(pts_int[j, 0]), int(pts_int[j, 1]))
            p2 = (int(pts_int[min(j + 1, len(pts_int) - 1), 0]),
                   int(pts_int[min(j + 1, len(pts_int) - 1), 1]))
            cv2.line(vis, p1, p2, (150, 150, 150), 1, cv2.LINE_AA)

    # Draw fitted lanelets (solid, thick)
    for idx, (lbl, pts) in enumerate(sorted(fitted_lanes.items())):
        # Clip to contour mask
        interp = interpolate_polyline(pts, spacing_px=5.0)
        clipped = clip_lane_to_contour_mask(interp, contour_mask)
        if clipped is None or len(clipped) < 2:
            continue

        color = FITTED_COLORS[idx % len(FITTED_COLORS)]
        pts_int = clipped.astype(np.int32)
        cv2.polylines(vis, [pts_int], isClosed=False, color=color,
                      thickness=3, lineType=cv2.LINE_AA)

        # Waypoint dots (subsampled)
        step = max(1, len(pts_int) // max(1, int(
            np.linalg.norm(np.diff(pts_int.astype(float), axis=0), axis=1).sum() / 15)))
        for j in range(0, len(pts_int), step):
            cv2.circle(vis, tuple(pts_int[j]), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, tuple(pts_int[j]), 4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, tuple(pts_int[-1]), 4, color, -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(pts_int[-1]), 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Label
        mid = len(pts_int) // 2
        sumo_ids = reverse_matches.get(lbl, [])
        sumo_str = f" <-> {sumo_ids[0]}" if sumo_ids else ""
        label = f"L{lbl}{sumo_str}"
        cv2.putText(vis, label,
                    (int(pts_int[mid, 0]) + 8, int(pts_int[mid, 1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Header
    n_fitted = len(fitted_lanes)
    n_matched = len(matches)
    y = 25
    lines = [
        f"Fitted Lanelets | {camera_name} | {n_fitted} lanes from trajectories",
        f"Solid: trajectory-fitted | Dashed gray: matched SUMO | {n_matched} matches",
    ]
    for text in lines:
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22

    return vis


def main():
    parser = argparse.ArgumentParser(description="Test GT lanelet visualization (GPS→pixel)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--camera", default=None, help="Single camera (default: all)")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "results" / "gt_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    import polars as pl

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
    if args.camera:
        cameras = [c for c in cameras if c == args.camera]
        if not cameras:
            cameras = [args.camera]

    logger.info(f"Testing GT visualization for {len(cameras)} camera(s)")

    from src.training.trainer import TrainingPipeline
    from src.data.tracklet_graph import build_tracklet_graph
    from src.utils.lane_fitting import fit_lanelets_from_tracklets, match_sumo_to_fitted

    pipeline = TrainingPipeline.__new__(TrainingPipeline)
    pipeline.config = config
    pipeline.exp_dir = out_dir

    time_window = 120.0  # seconds of traj for tracklet graph

    for cam in cameras:
        cam_dir = preprocess_dir / cam
        traj_path = cam_dir / "trajectory.csv"
        frame_path = cam_dir / "last_frame.npy"
        if not traj_path.exists():
            logger.warning(f"Skipping {cam}: no trajectory.csv")
            continue

        traj = pl.read_csv(str(traj_path))
        frame = np.load(str(frame_path)) if frame_path.exists() else np.zeros((720, 1280, 3), dtype=np.uint8)
        H, W = frame.shape[:2]

        logger.info(f"\n=== {cam} === (frame: {W}x{H})")

        # Extract SUMO metadata
        metadata = pipeline._extract_sumo_metadata(cam, traj, v1_dir)
        if not metadata:
            metadata = {}
            logger.warning(f"  No SUMO metadata for {cam}")

        logger.info(f"  gps_lane_geom: {len(metadata.get('gps_lane_geom', {}))} lanes")

        # Detect lane group contours for clipping
        contours, group_headings = detect_contours(traj, (H, W), config)
        logger.info(f"  Lane group contours: {len(contours)}")
        contour_mask = build_contour_mask(contours, H, W)

        # Figure 1: SUMO lanes clipped to contours
        vis_sumo = draw_gps_lanes_on_frame(
            frame, metadata, cam, contour_mask,
            contours=contours, group_headings=group_headings)
        cv2.imwrite(str(out_dir / f"{cam}_sumo.png"), vis_sumo)
        logger.info(f"  Saved: {out_dir / f'{cam}_sumo.png'}")

        # Build tracklet graph for lane fitting
        traj_windowed = traj
        if "time" in traj.columns:
            t_min = traj["time"].min()
            traj_windowed = traj.filter(pl.col("time") < t_min + time_window)

        graph = build_tracklet_graph(traj_windowed, (H, W), config)
        if graph is None:
            logger.warning(f"  Skipping fitted lanes for {cam}: too few tracklets")
            continue

        # Fit lanelets from trajectory pseudo-lanes
        pixel_c = graph.pixel_centroids.numpy()
        pixel_t = graph.pixel_tangents.numpy()
        pseudo_labels = graph.pseudo_labels.numpy()
        fitted_lanes = fit_lanelets_from_tracklets(
            pixel_c, pixel_t, pseudo_labels, n_bins=25, smooth_window=3)

        # Convert SUMO lanes to pixel space for matching
        sumo_lanes_px = {}
        pixel_hom = metadata.get("pixel_hom")
        gps_lane_geom = metadata.get("gps_lane_geom", {})
        if pixel_hom is not None:
            pixel_hom_arr = np.asarray(pixel_hom, dtype=np.float64)
            for lid, gps_pts in gps_lane_geom.items():
                gps_arr = np.array(gps_pts, dtype=np.float64)
                if gps_arr.ndim == 2 and gps_arr.shape[1] >= 2 and len(gps_arr) >= 2:
                    sumo_lanes_px[lid] = gps_to_pixels(pixel_hom_arr, gps_arr[:, :2])

        # Match SUMO lanes to fitted lanes
        matches = match_sumo_to_fitted(sumo_lanes_px, fitted_lanes, max_dist=80.0)
        logger.info(f"  Fitted: {len(fitted_lanes)} lanes, {len(matches)} matched to SUMO")

        # Figure 2: Fitted lanelets with SUMO reference
        vis_fitted = draw_fitted_lanes_on_frame(
            frame, fitted_lanes, sumo_lanes_px, matches, contour_mask, cam,
            contours=contours, group_headings=group_headings)
        cv2.imwrite(str(out_dir / f"{cam}_fitted.png"), vis_fitted)
        logger.info(f"  Saved: {out_dir / f'{cam}_fitted.png'}")

    logger.info(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
