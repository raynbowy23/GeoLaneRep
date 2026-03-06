#!/usr/bin/env python3
"""Visual pooling diagnostic: render pooling results on camera frames.

Renders each lane group's pooling output (nodes, headings, chains) overlaid
on the camera frame, with per-direction stats. Useful for quick visual
confirmation after training.

Usage:
    python scripts/diagnose_pooling_visual.py --config configs/lanelet_core.yaml
    python scripts/diagnose_pooling_visual.py --config configs/lanelet.yaml \
        --checkpoint results/lanelet_discovery/checkpoints/best.pt
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import yaml

from src.utils.visualization import visualize_lanelet_graph

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--cache_dir", default=None, help="Override graph cache dir")
    parser.add_argument("--output_dir", default="results/pooling_diag", help="Output dir")
    parser.add_argument("--conf_threshold", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_thresh = args.conf_threshold or config.get("training", {}).get("confidence_threshold", 0.3)

    # Load model
    from src.models.lane_repr import LaneReprModel
    model = LaneReprModel(config).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.info("No checkpoint — random init")

    # Load graph cache
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        candidates = list(cache_dir.glob(f"{args.split}_*.pt"))
        cache_path = candidates[0] if candidates else None
    else:
        from src.data.dataset import TrackletDataset
        cache_path = TrackletDataset._cache_path(config, args.split)
    if not cache_path or not cache_path.exists():
        logger.error(f"No graph cache at {cache_path}")
        sys.exit(1)
    samples = torch.load(cache_path, weights_only=False)
    logger.info(f"Loaded {len(samples)} graphs from {cache_path}")

    # Load camera frames
    data_cfg = config.get("data", {})
    v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
    preprocess_dir = v1_dir / "results" / "preprocess"
    camera_frames = {}
    for cam_dir in preprocess_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        frame_path = cam_dir / "last_frame.npy"
        if frame_path.exists():
            camera_frames[cam_dir.name] = np.load(str(frame_path))

    # Group samples by camera
    camera_samples = defaultdict(list)
    for s in samples:
        cam = getattr(s, "camera_loc", "unknown")
        camera_samples[cam].append(s)

    model.eval()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    M = config.get("model", {}).get("num_lanelet_nodes", 16)

    for cam in sorted(camera_samples.keys()):
        frame = camera_frames.get(cam)
        if frame is None:
            logger.warning(f"No frame for {cam}, skipping")
            continue

        vis = frame.copy()
        cam_info_lines = []

        for i, sample in enumerate(camera_samples[cam]):
            data = sample.to(device)
            with torch.no_grad():
                output = model(data)

            # Stats for this lane group
            heading_deg = float(getattr(data, "lane_group_heading", 0)) * 57.3
            S = output["assignment_matrix"].cpu().numpy()
            conf = output["lanelet_confidence"].cpu().numpy()
            pos = output["lanelet_positions"].cpu().numpy()
            n_active = (conf > conf_thresh).sum()
            hard_assign = S.argmax(axis=1)
            n_used_slots = len(set(hard_assign.tolist()))

            # Pairwise distance of active nodes
            active_mask = conf > conf_thresh
            if active_mask.sum() >= 2:
                from scipy.spatial.distance import pdist
                active_pos = pos[active_mask]
                pw = pdist(active_pos)
                min_dist = pw.min()
                median_dist = np.median(pw)
            else:
                min_dist = median_dist = float('nan')

            info = (f"  Dir {i}: heading={heading_deg:.0f}deg, "
                    f"N={data.num_nodes} tracklets, "
                    f"active={n_active}/{M}, "
                    f"used_slots={n_used_slots}, "
                    f"min_dist={min_dist:.1f}m")
            cam_info_lines.append(info)
            logger.info(f"[{cam}] {info}")

            # Render this lane group onto frame
            vis = visualize_lanelet_graph(
                vis, output, data,
                conf_threshold=conf_thresh,
            )

        # Add text overlay with stats
        y0 = vis.shape[0] - 15 * (len(cam_info_lines) + 1) - 5
        for j, line in enumerate(cam_info_lines):
            y = y0 + 15 * (j + 1)
            cv2.putText(vis, line.strip(), (5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        save_path = str(out_dir / f"{cam}.png")
        cv2.imwrite(save_path, vis)
        logger.info(f"Saved: {save_path}")

    logger.info(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
