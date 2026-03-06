"""Visualize pooling assignment: how tracklets are grouped into lanelet nodes.

Produces per camera:
  {cam}_pooling.png -- tracklet centroids colored by assigned lanelet node,
                       with lanelet node positions as large circles and
                       lines connecting each tracklet to its lanelet.

Usage:
    python scripts/test_pooling_viz.py --config configs/lanelet.yaml
    python scripts/test_pooling_viz.py --config configs/lanelet.yaml --camera US12_Park
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
import polars as pl
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Vivid colors for lanelet nodes (BGR)
LANELET_COLORS = [
    (  0,   0, 255),  # red
    (189, 115,   0),  # blue
    ( 48, 171, 120),  # green
    ( 33, 176, 237),  # orange
    (143,  46, 125),  # purple
    (191, 191,   0),  # cyan
    ( 25,  84, 217),  # deep orange
    (  0, 128,   0),  # forest green
    (191,   0, 191),  # magenta
    (171, 179,  33),  # teal
    (  0, 181, 140),  # lime green
    ( 46,  20, 163),  # wine red
    (  0, 102, 204),  # burnt orange
    (179, 120,  31),  # deep sky blue
    (212,   0, 148),  # violet
    ( 79,  43, 237),  # cherry red
]


def draw_pooling_assignment(frame, model_output, data, conf_threshold=0.3):
    """Draw tracklets colored by their assigned lanelet node.

    Shows:
      - Thin lines from each tracklet to its assigned lanelet node
      - Tracklet centroids as circles colored by assignment
      - Lanelet node positions as large circles with white outline
      - Soft assignment strength as circle opacity/size
    """
    vis = frame.copy()

    pred_pos = model_output["lanelet_positions"].detach().cpu().numpy()    # (M, 2)
    pred_conf = model_output["lanelet_confidence"].detach().cpu().numpy()  # (M,)
    S = model_output["assignment_matrix"].detach().cpu().numpy()           # (N, M)

    M = pred_pos.shape[0]
    N = S.shape[0]

    # Check for homography
    has_hom = hasattr(data, "pixel_hom_matrix") and data.pixel_hom_matrix is not None
    has_ref = hasattr(data, "ref_gps") and data.ref_gps is not None

    # Tracklet pixel positions
    has_pixel = hasattr(data, "pixel_centroids") and data.pixel_centroids is not None
    if has_pixel:
        centroids_px = data.pixel_centroids.cpu().numpy()
    elif has_hom and has_ref:
        from src.utils.homography_scale import local_meters_to_pixels
        pixel_hom = data.pixel_hom_matrix.cpu().numpy()
        ref_gps = data.ref_gps.cpu().numpy()
        centroids_px = local_meters_to_pixels(pixel_hom, data.centroids.cpu().numpy(), ref_gps)
    else:
        centroids_px = data.centroids.cpu().numpy()

    # Lanelet positions in pixels
    if has_hom and has_ref:
        from src.utils.homography_scale import local_meters_to_pixels
        pixel_hom = data.pixel_hom_matrix.cpu().numpy()
        ref_gps = data.ref_gps.cpu().numpy()
        lanelet_pos_px = local_meters_to_pixels(pixel_hom, pred_pos, ref_gps)
    else:
        lanelet_pos_px = pred_pos

    active_mask = pred_conf > conf_threshold
    assignments = S.argmax(axis=1)       # (N,) hard assignment
    assignment_strength = S.max(axis=1)  # (N,) how confident the assignment is

    # -- 1. Draw thin lines from tracklets to their assigned lanelet (bottom layer) --
    for i in range(N):
        m = assignments[i]
        if not active_mask[m]:
            continue
        color = LANELET_COLORS[m % len(LANELET_COLORS)]
        # Fade line by assignment strength
        alpha = float(assignment_strength[i])
        faded = tuple(int(c * alpha * 0.4) for c in color)
        pt1 = (int(centroids_px[i, 0]), int(centroids_px[i, 1]))
        pt2 = (int(lanelet_pos_px[m, 0]), int(lanelet_pos_px[m, 1]))
        cv2.line(vis, pt1, pt2, faded, 1, cv2.LINE_AA)

    # -- 2. Draw tracklet centroids colored by assignment --
    for i in range(N):
        m = assignments[i]
        if active_mask[m]:
            color = LANELET_COLORS[m % len(LANELET_COLORS)]
        else:
            color = (80, 80, 80)  # unassigned to inactive lanelet
        # Size proportional to assignment strength
        strength = float(assignment_strength[i])
        radius = max(3, int(3 + 4 * strength))
        cx, cy = int(centroids_px[i, 0]), int(centroids_px[i, 1])
        cv2.circle(vis, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), radius, (0, 0, 0), 1, cv2.LINE_AA)

    # -- 3. Draw lanelet nodes on top (large circles) --
    for m in range(M):
        if not active_mask[m]:
            continue
        color = LANELET_COLORS[m % len(LANELET_COLORS)]
        cx, cy = int(lanelet_pos_px[m, 0]), int(lanelet_pos_px[m, 1])
        # Count assigned tracklets
        n_assigned = int((assignments == m).sum())
        # Size proportional to number of assigned tracklets
        radius = max(10, min(20, 8 + n_assigned))
        cv2.circle(vis, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), radius, (255, 255, 255), 2, cv2.LINE_AA)
        # Label with lanelet index and count
        cv2.putText(vis, f"L{m}", (cx - 8, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # -- Stats overlay --
    n_active = int(active_mask.sum())
    n_assigned_to_active = int(sum(1 for i in range(N) if active_mask[assignments[i]]))
    mean_strength = float(assignment_strength.mean())
    y = 25
    lines = [
        f"Pooling Assignment | {N} tracklets -> {n_active}/{M} active lanelets",
        f"Assigned to active: {n_assigned_to_active}/{N} | Mean strength: {mean_strength:.2f}",
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
        description="Visualize pooling assignment (tracklets -> lanelet nodes)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--camera", default=None, help="Single camera")
    parser.add_argument("--time-window", type=float, default=60.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Checkpoint
    save_dir = Path(config.get("experiment", {}).get("saving_path", "./results"))
    exp_name = config.get("experiment", {}).get("experiment_name", "lanelet_discovery")
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = save_dir / exp_name / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_dir = (Path(args.out_dir) if args.out_dir
               else PROJECT_ROOT / "results" / "pooling_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from src.models.lane_repr import LaneReprModel
    model = LaneReprModel(config).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded checkpoint: epoch={epoch}")
    model.eval()

    # Data setup
    from src.data.tracklet_graph import build_tracklet_graph
    from src.data.density_contours import assign_vehicles_to_lane_groups

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    conf_thresh = train_cfg.get("confidence_threshold", 0.3)
    use_lane_groups = data_cfg.get("use_lane_groups", False)
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

    for cam in cameras:
        cam_dir = preprocess_dir / cam
        traj_path = cam_dir / "trajectory.csv"
        frame_path = cam_dir / "last_frame.npy"
        if not traj_path.exists():
            logger.warning(f"Skipping {cam}: no trajectory.csv")
            continue

        traj = pl.read_csv(str(traj_path))
        frame = (np.load(str(frame_path)) if frame_path.exists()
                 else np.zeros((720, 1280, 3), dtype=np.uint8))
        H, W = frame.shape[:2]

        if "time" in traj.columns:
            t_min = traj["time"].min()
            traj = traj.filter(pl.col("time") < t_min + args.time_window)

        logger.info(f"\n=== {cam} === ({W}x{H})")

        vis = frame.copy()

        if use_lane_groups:
            # Build kwargs for lane group detection
            method = data_cfg.get("lane_group_method", "dbscan")
            lg_kwargs = dict(
                min_track_points=data_cfg.get("tracklet_min_points", 5),
                min_gap_deg=data_cfg.get("direction_min_gap_deg", 45.0),
                min_vehicles_per_group=data_cfg.get("min_vehicles_per_group", 3),
            )
            if method == "density":
                lg_kwargs.update(
                    tracklet_length=data_cfg.get("tracklet_length", 15),
                    sigma_along=data_cfg.get("density_sigma_along", 25.0),
                    sigma_across=data_cfg.get("density_sigma_across", 6.0),
                    neck_ratio=data_cfg.get("neck_ratio", 0.4),
                    refine_contours=data_cfg.get("refine_contours", True),
                    refine_buffer_px=data_cfg.get("refine_buffer_px", 25.0),
                )
            elif method == "dbscan":
                lg_kwargs.update(
                    hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
                    dbscan_min_samples=data_cfg.get("dbscan_min_samples", 3),
                )
                if "dbscan_eps" in data_cfg:
                    lg_kwargs["dbscan_eps"] = data_cfg["dbscan_eps"]

            groups = assign_vehicles_to_lane_groups(
                traj, (H, W), method=method, **lg_kwargs)

            for group_traj, heading, gid in groups:
                if len(group_traj) < 10:
                    continue
                graph = build_tracklet_graph(
                    group_traj, (H, W), config,
                    single_direction=True,
                    skip_density_contours=True,
                )
                if graph is None:
                    continue

                graph_device = graph.to(device)
                with torch.no_grad():
                    output = model(graph_device)

                vis = draw_pooling_assignment(vis, output, graph_device,
                                             conf_threshold=conf_thresh)
                n_active = int((output["lanelet_confidence"] > conf_thresh).sum().item())
                logger.info(f"  {cam} group {gid}: {graph.num_nodes} tracklets -> "
                            f"{n_active} active lanelets")
        else:
            graph = build_tracklet_graph(traj, (H, W), config)
            if graph is None:
                logger.warning(f"  Skipping {cam}: too few tracklets")
                continue

            graph_device = graph.to(device)
            with torch.no_grad():
                output = model(graph_device)

            vis = draw_pooling_assignment(vis, output, graph_device,
                                         conf_threshold=conf_thresh)
            n_active = int((output["lanelet_confidence"] > conf_thresh).sum().item())
            logger.info(f"  {cam}: {graph.num_nodes} tracklets -> {n_active} active lanelets")

        save_path = out_dir / f"{cam}_pooling.png"
        cv2.imwrite(str(save_path), vis)
        logger.info(f"  Saved: {save_path}")

    logger.info(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
