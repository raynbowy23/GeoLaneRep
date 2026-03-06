"""Inference pipeline for v3 lanelet discovery.

Runs the model per lane group (matching training), then composites
all lane group results into one visualization per camera.

Usage:
    python scripts/inference.py --config configs/lanelet.yaml [--checkpoint path]
"""

import json
import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_DIR = PROJECT_ROOT.parent / "graph_geolane"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V1_DIR / "src"))

import cv2
import numpy as np
import polars as pl
import torch
import yaml

from src.models.lane_repr import LaneReprModel
from src.data.tracklet_graph import build_tracklet_graph, split_trajectories_by_direction
from src.data.lane_extraction import extract_lanelet_lanes, lanelet_lanes_to_dict
from src.training.evaluator import compute_lanelet_metrics
from src.utils.visualization import visualize_lanelet_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> LaneReprModel:
    """Load model from checkpoint.

    Uses the config saved inside the checkpoint for model architecture
    (ensures weights match), but keeps the user-provided config for data paths.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Use checkpoint's model config to ensure architecture matches saved weights
    if "config" in ckpt and "model" in ckpt["config"]:
        model_config = {**config, "model": ckpt["config"]["model"]}
        logger.info("Using model config from checkpoint (architecture match)")
    else:
        model_config = config
    model = LaneReprModel(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    logger.info(f"Loaded checkpoint: epoch={epoch}")
    model.eval()
    return model


def load_camera_data(preprocess_dir: Path, camera: str, time_window: float = 120.0):
    """Load preprocessed trajectory + frame for a camera."""
    cam_dir = preprocess_dir / camera
    traj = pl.read_csv(str(cam_dir / "trajectory.csv"))
    frame_path = cam_dir / "last_frame.npy"
    frame = np.load(str(frame_path)) if frame_path.exists() else np.zeros((720, 1280, 3), dtype=np.uint8)

    if "time" in traj.columns:
        t_min = traj["time"].min()
        traj = traj.filter(pl.col("time") < t_min + time_window)

    return traj, frame


def extract_sumo_metadata(camera: str, traj_full: pl.DataFrame, config: dict) -> dict:
    """Extract SUMO metadata (pixel_hom, lane geom) — same as trainer._extract_sumo_metadata."""
    data_cfg = config.get("data", {})
    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))

    v1_path = str(v1_dir.resolve())
    if v1_path not in sys.path:
        sys.path.insert(0, v1_path)

    try:
        from core.osm_extraction.connect_to_osm import OSMConnection
    except ImportError:
        logger.warning("Cannot import OSMConnection from v1 — no homography available")
        return {}

    class _V1Cfg:
        pass

    cfg = _V1Cfg()
    cfg.data = _V1Cfg()
    cfg.data.dataset_path = data_cfg.get("dataset_path", str(v1_dir / "dataset"))
    cfg.data.osm_path = data_cfg.get("osm_path", str(v1_dir / "dataset" / "sumo"))
    cfg.data.include_junction_movements = False
    cfg.data.junction_min_traj_points = 30
    cfg.experiment = _V1Cfg()
    cfg.experiment.is_save = False
    cfg.experiment.saving_path = str(Path(config.get("experiment", {}).get("saving_path", "./results")))

    try:
        osm = OSMConnection(cfg)
        result = osm.get_lane_groups_from_sumo(camera, traj_full)
        lane_group_dict, pixel_hom, gps_lane_geom, lane_shape, highway_mask, lane_headings = result
    except Exception as e:
        logger.warning(f"SUMO extraction failed for {camera}: {e}")
        return {}

    if not gps_lane_geom or pixel_hom is None:
        return {}

    # Convention fix: v1 uses [lat, lon], v3 standardizes on [lon, lat]
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    pixel_hom = swap @ pixel_hom

    for lid in list(gps_lane_geom.keys()):
        pts = np.array(gps_lane_geom[lid], dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            gps_lane_geom[lid] = np.column_stack([pts[:, 1], pts[:, 0]])
            if pts.shape[1] > 2:
                gps_lane_geom[lid] = np.column_stack(
                    [gps_lane_geom[lid], pts[:, 2:]]
                )

    if data_cfg.get("merge_consecutive_sumo_lanes", True):
        from src.data.dataset import _merge_consecutive_sumo_lanes
        sumo_net_path = str(Path(data_cfg["osm_path"], camera, "osm.net.xml"))
        gps_lane_geom, lane_headings = _merge_consecutive_sumo_lanes(
            gps_lane_geom, lane_headings, sumo_net_path
        )

    return {
        "pixel_hom": pixel_hom,
        "gps_lane_geom": gps_lane_geom,
        "lane_headings": lane_headings,
        "lane_shape": lane_shape,
    }


def _build_lane_group_kwargs(data_cfg: dict, camera_loc: str = "") -> tuple:
    """Build kwargs for detect_lane_groups / assign_vehicles_to_lane_groups."""
    method = data_cfg.get("lane_group_method", "dbscan")
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
    elif method == "learned":
        preprocess_dir = Path(data_cfg.get("preprocess_path", "./results/preprocess"))
        cam_cache_dir = preprocess_dir / camera_loc if camera_loc else preprocess_dir
        kwargs.update(
            hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
            cache_dir=str(cam_cache_dir),
        )
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
    return method, kwargs


def run_inference_on_camera(
    model: LaneReprModel,
    config: dict,
    camera: str,
    traj: pl.DataFrame,
    frame: np.ndarray,
    device: torch.device,
    out_dir: Path,
    pixel_hom=None,
    ref_gps=None,
):
    """Run inference per lane group and composite results."""
    H, W = frame.shape[:2]
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    conf_thresh = train_cfg.get("confidence_threshold", 0.3)
    top_k = train_cfg.get("vis_top_k", 30)
    use_lane_groups = data_cfg.get("use_lane_groups", False)

    all_lanes = []
    total_tracklets = 0
    total_active = 0
    vis = frame.copy()

    if use_lane_groups:
        from src.data.density_contours import detect_lane_groups, assign_vehicles_to_lane_groups

        method, kwargs = _build_lane_group_kwargs(data_cfg, camera)
        groups = assign_vehicles_to_lane_groups(
            traj, (H, W), method=method, **kwargs,
        )

        if not groups:
            logger.warning(f"  {camera}: no lane groups found, falling back to single graph")
            use_lane_groups = False

    if use_lane_groups:
        for group_traj, heading, gid in groups:
            if len(group_traj) < 10:
                continue

            graph = build_tracklet_graph(
                group_traj, (H, W), config,
                single_direction=True,
                skip_density_contours=True,
                pixel_hom=pixel_hom,
                ref_gps=ref_gps,
            )
            if graph is None:
                continue

            graph.lane_group_id = gid
            graph.lane_group_heading = heading

            graph_device = graph.to(device)
            with torch.no_grad():
                output = model(graph_device)

            lanes = extract_lanelet_lanes(output, conf_threshold=conf_thresh)
            # Tag lanes with group id
            for lane in lanes:
                lane.lane_id = len(all_lanes) + lane.lane_id
            all_lanes.extend(lanes)

            total_tracklets += graph.num_nodes
            total_active += int((output["lanelet_confidence"] > conf_thresh).sum().item())

            vis, _ = visualize_lanelet_graph(
                vis, output, graph_device, conf_threshold=conf_thresh,
                top_k=top_k,
            )

            logger.info(
                f"  {camera} group {gid} (heading={np.degrees(heading):.0f}deg): "
                f"{graph.num_nodes} tracklets, {len(lanes)} lanes"
            )
    else:
        # Fallback: single graph per camera
        graph = build_tracklet_graph(
            traj, (H, W), config,
            pixel_hom=pixel_hom,
            ref_gps=ref_gps,
        )
        if graph is None:
            logger.warning(f"  {camera}: too few tracklets, skipping")
            return None

        graph_device = graph.to(device)
        with torch.no_grad():
            output = model(graph_device)

        all_lanes = extract_lanelet_lanes(output, conf_threshold=conf_thresh)
        total_tracklets = graph.num_nodes
        total_active = int((output["lanelet_confidence"] > conf_thresh).sum().item())

        vis, _ = visualize_lanelet_graph(
            vis, output, graph_device, conf_threshold=conf_thresh,
            top_k=top_k,
        )

    # Save visualization
    cam_dir = out_dir / camera
    cam_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cam_dir / "lanelet_graph.png"), vis)

    # Save lane data
    lane_data = {
        "camera": camera,
        "num_tracklets": total_tracklets,
        "active_nodes": total_active,
        "lanes": lanelet_lanes_to_dict(all_lanes),
    }
    with open(cam_dir / "lanelet_result.json", "w") as f:
        json.dump(lane_data, f, indent=2)

    logger.info(f"  {camera}: {len(all_lanes)} total lanes extracted")
    return lane_data


def main():
    parser = argparse.ArgumentParser(description="Lanelet discovery inference (v3)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--camera", default=None, help="Single camera")
    parser.add_argument("--time-window", type=float, default=120.0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config.get("experiment", {}).get("experiment_name", "lanelet_discovery")
    save_dir = Path(config.get("experiment", {}).get("saving_path", "./results"))
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = save_dir / exp_name / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else save_dir / exp_name / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, str(ckpt_path), device)

    data_cfg = config.get("data", {})
    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
    preprocess_dir = v1_dir / "results" / "preprocess"

    if args.camera:
        cameras = [args.camera]
    else:
        cameras = sorted([
            d.name for d in preprocess_dir.iterdir()
            if d.is_dir() and (d / "trajectory.csv").exists()
        ])

    logger.info(f"Running inference on {len(cameras)} cameras...")
    all_results = []
    for cam in cameras:
        # Load full trajectory (for ref_gps) and windowed trajectory (for inference)
        cam_dir = preprocess_dir / cam
        traj_full = pl.read_csv(str(cam_dir / "trajectory.csv"))
        frame_path = cam_dir / "last_frame.npy"
        frame = np.load(str(frame_path)) if frame_path.exists() else np.zeros((720, 1280, 3), dtype=np.uint8)

        # Window the trajectory
        traj = traj_full
        if "time" in traj.columns:
            t_min = traj["time"].min()
            traj = traj.filter(pl.col("time") < t_min + args.time_window)

        # Extract homography + ref_gps (same as training pipeline)
        metadata = extract_sumo_metadata(cam, traj_full, config)
        pixel_hom = metadata.get("pixel_hom", None)
        ref_gps = None
        if pixel_hom is not None:
            from src.utils.homography_scale import pixels_to_local_meters
            traj_pts = traj_full.select(["x", "y"]).to_numpy().astype(np.float64)
            _, ref_gps = pixels_to_local_meters(pixel_hom, traj_pts)
            logger.info(f"  {cam}: homography loaded, ref_gps={ref_gps}")
        else:
            logger.warning(f"  {cam}: no homography — features will be in pixel-space (quality may degrade)")

        result = run_inference_on_camera(
            model, config, cam, traj, frame, device, out_dir,
            pixel_hom=pixel_hom, ref_gps=ref_gps,
        )
        if result:
            all_results.append(result)

    # Summary
    total_lanes = sum(len(r["lanes"]) for r in all_results)
    logger.info(f"Cameras: {len(all_results)}/{len(cameras)}, Total lanes: {total_lanes}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"total_cameras": len(all_results), "total_lanes": total_lanes,
                    "per_camera": all_results}, f, indent=2)

    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
