#!/usr/bin/env python3
"""Build data-driven lanelet graphs from lane assignment results.

Usage:
    python scripts/build_lanelet_graph.py --config configs/lanelet_core.yaml
    python scripts/build_lanelet_graph.py --config configs/lanelet_core.yaml --camera US12_Greenway
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lanelet_graph import build_lanelet_graph

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_cameras(assignment_dir: Path) -> list:
    """Find cameras that have lane_assignments.csv."""
    cameras = []
    if not assignment_dir.exists():
        return cameras
    for p in sorted(assignment_dir.iterdir()):
        if p.is_dir() and (p / "lane_assignments.csv").exists():
            cameras.append(p.name)
    return cameras


def run_camera(
    camera: str,
    cfg: dict,
    assignment_dir: Path,
    output_dir: Path,
) -> bool:
    """Build lanelet graph for a single camera."""
    data_cfg = cfg["data"]
    graph_cfg = cfg.get("lanelet_graph", {})

    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    annot_path = annot_dir / camera / "annotation.json"
    assign_path = assignment_dir / camera / "lane_assignments.csv"

    if not assign_path.exists():
        logger.warning(f"No assignments for {camera}: {assign_path}")
        return False
    if not annot_path.exists():
        logger.warning(f"No annotation for {camera}: {annot_path}")
        return False

    # Resolve image dimensions
    frame_path = annot_dir / camera / "last_frame.npy"
    if frame_path.exists():
        frame = np.load(frame_path)
        image_wh = (frame.shape[1], frame.shape[0])
    else:
        image_wh = (data_cfg.get("image_width", 1920), data_cfg.get("image_height", 1080))

    logger.info(f"Processing {camera}: image_wh={image_wh}")

    result = build_lanelet_graph(
        assignments_path=str(assign_path),
        annotation_path=str(annot_path),
        image_wh=image_wh,
        waypoint_spacing_px=graph_cfg.get("waypoint_spacing_px", 20.0),
        min_points_per_lane=graph_cfg.get("min_points_per_lane", 3),
    )

    # Save
    cam_dir = output_dir / camera
    cam_dir.mkdir(parents=True, exist_ok=True)
    out_path = cam_dir / "lanelet_graph.npz"

    np.savez(
        out_path,
        positions=result["positions"],
        tangents=result["tangents"],
        lane_ids=result["lane_ids"],
        group_ids=result["group_ids"],
        edge_index=result["edge_index"],
        edge_types=result["edge_types"],
    )

    n_lanes = len(result["lane_meta"])
    n_wp = len(result["positions"])
    n_edges = result["edge_index"].shape[1] if result["edge_index"].ndim == 2 else 0
    logger.info(f"  Saved {out_path}: {n_lanes} lanes, {n_wp} waypoints, {n_edges} edges")

    return True


def main():
    parser = argparse.ArgumentParser(description="Build data-driven lanelet graphs")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--camera", default=None, help="Single camera name (default: all)")
    parser.add_argument("--output", default=None, help="Output directory override")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.get("system", {}).get("log_level", "INFO")),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    saving_path = Path(cfg.get("experiment", {}).get("saving_path", "./results/"))
    assignment_dir = saving_path / "lane_assignment"
    output_dir = Path(args.output) if args.output else assignment_dir

    if args.camera:
        cameras = [args.camera]
    else:
        cameras = discover_cameras(assignment_dir)

    if not cameras:
        logger.error("No cameras found with lane_assignments.csv")
        sys.exit(1)

    logger.info(f"Building lanelet graphs for {len(cameras)} cameras: {cameras}")

    n_ok = 0
    for cam in cameras:
        try:
            if run_camera(cam, cfg, assignment_dir, output_dir):
                n_ok += 1
        except Exception:
            logger.exception(f"Failed to process {cam}")

    logger.info(f"Done: {n_ok}/{len(cameras)} cameras processed successfully")


if __name__ == "__main__":
    main()
