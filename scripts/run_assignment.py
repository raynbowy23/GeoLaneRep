#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lane_assignment import LaneAssigner

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_cameras(dataset_path: str, annotation_dir: str) -> list:
    """Find cameras that have both trajectory.csv and annotation.json."""
    annot_dir = Path(annotation_dir)
    cameras = []
    for p in sorted(annot_dir.iterdir()):
        if not p.is_dir():
            continue
        annot = p / "annotation.json"
        if not annot.exists():
            continue
        # Check multiple possible trajectory locations
        traj_candidates = [
            Path(dataset_path) / "preprocess" / p.name / "trajectory.csv",
            Path(dataset_path) / p.name / "trajectory.csv",
            annot_dir / p.name / "trajectory.csv",
        ]
        if any(t.exists() for t in traj_candidates):
            cameras.append(p.name)
    return cameras


def run_camera(
    camera: str,
    cfg: dict,
    output_dir: Path,
) -> pl.DataFrame:
    """Run lane assignment for a single camera."""
    data_cfg = cfg["data"]
    assign_cfg = cfg.get("assignment", {})

    # Resolve paths
    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    dataset_path = Path(data_cfg.get("dataset_path", "./dataset/"))

    annot_path = annot_dir / camera / "annotation.json"
    if not annot_path.exists():
        logger.warning(f"No annotation for {camera}: {annot_path}")
        return pl.DataFrame()

    # Find trajectory CSV
    traj_path = dataset_path / "preprocess" / camera / "trajectory.csv"
    if not traj_path.exists():
        traj_path = annot_dir / camera / "trajectory.csv"
    if not traj_path.exists():
        logger.warning(f"No trajectory for {camera}")
        return pl.DataFrame()

    # Load frame for shape
    frame_path = annot_dir / camera / "last_frame.npy"
    if frame_path.exists():
        frame = np.load(frame_path)
        frame_shape = frame.shape[:2]  # (H, W)
        image_wh = (frame_shape[1], frame_shape[0])
    else:
        image_wh = (data_cfg.get("image_width", 1920), data_cfg.get("image_height", 1080))
        frame_shape = (image_wh[1], image_wh[0])

    logger.info(f"Processing {camera}: frame={frame_shape}, image_wh={image_wh}")

    # Load trajectories
    traj = pl.read_csv(traj_path)
    logger.info(f"  Loaded {len(traj)} trajectory rows, {traj['id'].n_unique()} tracks")

    # Create assigner
    assigner = LaneAssigner(
        annotation_path=str(annot_path),
        image_wh=image_wh,
        lateral_threshold_px=assign_cfg.get("lateral_threshold_px", 60.0),
        min_tracklet_points=assign_cfg.get("min_tracklet_points", 5),
        lane_change_std_px=assign_cfg.get("lane_change_std_px", 30.0),
        lane_group_method=data_cfg.get("lane_group_method", "density"),
    )

    # Run assignment
    results = assigner.assign(traj, frame_shape)

    # Save output
    cam_dir = output_dir / camera
    cam_dir.mkdir(parents=True, exist_ok=True)
    out_path = cam_dir / "lane_assignments.csv"
    results.write_csv(out_path)
    logger.info(f"  Saved {len(results)} rows to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Geometric lane assignment")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--camera", default=None, help="Single camera name (default: all)")
    parser.add_argument("--output", default=None, help="Output directory (default: results/preprocessing/lane_assignment/)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.get("system", {}).get("log_level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output) if args.output else Path(
        cfg.get("experiment", {}).get("saving_path", "./results/")
    ) / "preprocessing" / "lane_assignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    annot_dir = cfg["data"].get("annotation_dir", "../dataset/preprocess")

    if args.camera:
        cameras = [args.camera]
    else:
        cameras = discover_cameras(cfg["data"].get("dataset_path", "./dataset/"), annot_dir)

    if not cameras:
        logger.error("No cameras found with both trajectory.csv and annotation.json")
        sys.exit(1)

    logger.info(f"Processing {len(cameras)} cameras: {cameras}")

    all_stats = []
    for cam in cameras:
        try:
            result = run_camera(cam, cfg, output_dir)
            if len(result) > 0:
                total = result["track_id"].n_unique()
                assigned = result.filter(pl.col("lane_id") >= 0)["track_id"].n_unique()
                all_stats.append({
                    "camera": cam,
                    "total_tracks": total,
                    "assigned_tracks": assigned,
                    "rejected_tracks": total - assigned,
                    "rejection_rate": (total - assigned) / max(total, 1),
                })
        except Exception:
            logger.exception(f"Failed to process {cam}")

    # Summary
    if all_stats:
        stats_df = pl.DataFrame(all_stats)
        print("\n=== Assignment Summary ===")
        print(stats_df)
        stats_df.write_csv(output_dir / "summary.csv")


if __name__ == "__main__":
    main()
