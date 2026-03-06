#!/usr/bin/env python
"""Zero-shot lane detection on a new camera using trajectory data only.

Given a trained contrastive encoder:
1. Build reference bank from annotated cameras (with geometry)
2. Discover pseudo-lanes from trajectory behavior on target camera
3. Match to reference bank via cosine similarity
4. Output predicted lane properties + visualization

Usage:
    python scripts/run_zero_shot.py \
      --checkpoint results/lane_contrastive/checkpoints/best.pt \
      --camera I43_Keefe \
      --output-dir results/lane_contrastive/zero_shot
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

import cv2
import numpy as np
import polars as pl
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot lane detection")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained encoder checkpoint"
    )
    parser.add_argument(
        "--camera", required=True, help="Target camera name (trajectory-only)"
    )
    parser.add_argument(
        "--config", default=None, help="Optional config override"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/lane_contrastive/zero_shot)"
    )
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from src.data.annotation_loader import load_annotation_json
    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import encode_lanes, load_trained_encoder
    from src.zero_shot_lanes import (
        build_lanes_from_annotation,
        discover_lanes,
        predict_lane_properties,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load encoder
    logger.info(f"Loading encoder from {args.checkpoint}")
    model, ckpt_config = load_trained_encoder(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt_config

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})
    data_cfg = config.get("data", {})
    polyline_k = model_cfg.get("polyline_k", 16)

    # Step 2: Build reference bank from all annotated cameras
    logger.info("Building reference bank from annotated cameras...")
    dataset = LaneDataset(
        config=config,
        polyline_k=polyline_k,
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )

    ref_loader = DataLoader(
        dataset,
        batch_size=train_cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_fn,
    )
    ref_proj, ref_roles, ref_keys = encode_lanes(
        model, ref_loader, device, drop_geometry=False,
    )
    logger.info(f"Reference bank: {len(ref_keys)} lanes from {len(dataset.cameras)} cameras")

    # Step 3: Load target camera data
    annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
    cam_dir = annot_dir / args.camera

    traj_path = cam_dir / "trajectory.csv"
    if not traj_path.exists():
        logger.error(f"Trajectory file not found: {traj_path}")
        return

    traj_df = pl.read_csv(str(traj_path))
    logger.info(f"Loaded {len(traj_df)} trajectory points for {args.camera}")

    # Load frame for shape and visualization
    frame_path = cam_dir / "last_frame.npy"
    if frame_path.exists():
        frame = np.load(str(frame_path))
    else:
        # Fallback: try PNG
        frame_png = cam_dir / "last_frame.png"
        if frame_png.exists():
            frame = cv2.imread(str(frame_png))
        else:
            image_w = data_cfg.get("image_width", 1920)
            image_h = data_cfg.get("image_height", 1080)
            frame = np.zeros((image_h, image_w, 3), dtype=np.uint8)
            logger.warning("No frame found, using black background")

    frame_shape = (frame.shape[0], frame.shape[1])
    logger.info(f"Frame shape: {frame_shape}")

    # Step 4: Build pseudo-lanes
    # Use annotation geometry for lane boundaries (clean evaluation),
    # fall back to trajectory-only discovery if no annotation exists.
    annot_path = cam_dir / "annotation.json"
    if annot_path.exists():
        logger.info(f"Using annotation geometry for lane boundaries")
        annotation = load_annotation_json(annot_path)
        pseudo_lanes = build_lanes_from_annotation(
            annotation, traj_df, frame_shape, args.camera, config,
        )
    else:
        logger.info(f"No annotation found, using trajectory-only lane discovery")
        pseudo_lanes = discover_lanes(traj_df, frame_shape, args.camera, config)

    if not pseudo_lanes:
        logger.error("No pseudo-lanes found. Check trajectory/annotation data.")
        return

    # Step 5: Predict lane properties
    # Geometric: lateral_rank, is_leftmost/rightmost, n_lanes (always computed)
    # Encoder: has_successor, group_size (from reference bank matching)
    predictions = predict_lane_properties(
        pseudo_lanes,
        model=model, ref_proj=ref_proj, ref_roles=ref_roles, ref_keys=ref_keys,
        device=device, polyline_k=polyline_k,
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        max_group_size=data_cfg.get("max_group_size", 6),
    )

    # Step 6: Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "results/lane_contrastive/zero_shot"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions JSON
    json_path = output_dir / f"{args.camera}_predictions.json"
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info(f"Predictions saved to {json_path}")

    # Step 7: Visualization
    vis = _visualize_zero_shot(
        frame, pseudo_lanes, predictions, args.camera,
    )
    vis_path = output_dir / f"{args.camera}_zero_shot.png"
    cv2.imwrite(str(vis_path), vis)
    logger.info(f"Visualization saved to {vis_path}")

    # Summary
    logger.info("=== Zero-Shot Prediction Summary ===")
    for pred in predictions:
        parts = [
            f"  {pred['lane_key']}:",
            f"geo={pred['lateral_rank']:.2f}",
            f"left={pred['is_leftmost']}",
            f"right={pred['is_rightmost']}",
            f"n={pred['n_lanes_in_group']}",
        ]
        if "encoder_rank" in pred:
            parts.append(f"enc={pred['encoder_rank']:.2f}")
            raw = pred.get("encoder_rank_raw", pred["encoder_rank"])
            parts.append(f"raw={raw:.3f}")
            parts.append(f"sim={pred['match_similarity']:.3f}")
            parts.append(f"conf={pred.get('encoder_confidence', '?')}")
        logger.info(" ".join(parts))


def _visualize_zero_shot(
    frame: np.ndarray,
    pseudo_lanes: list,
    predictions: list,
    camera: str,
) -> np.ndarray:
    """Draw discovered pseudo-lane trajectories colored by predicted lateral rank.

    Args:
        frame: (H, W, 3) BGR camera frame.
        pseudo_lanes: List of PseudoLane objects.
        predictions: List of prediction dicts.
        camera: Camera name for title.

    Returns:
        (H, W, 3) annotated frame.
    """
    from src.utils.visualization import SLOT_COLORS

    vis = frame.copy()
    h, w = vis.shape[:2]
    image_wh = np.array([w, h], dtype=np.float64)

    # Distinct color per lane
    for lane_i, (pl_lane, pred) in enumerate(zip(pseudo_lanes, predictions)):
        color_bgr = SLOT_COLORS[lane_i % len(SLOT_COLORS)]

        # Draw trajectories
        for traj in pl_lane.trajectories:
            pts_px = (traj * image_wh).astype(np.int32)
            if len(pts_px) >= 2:
                cv2.polylines(vis, [pts_px], False, color_bgr, 1, cv2.LINE_AA)

        # Draw mean trajectory thicker
        if pl_lane.trajectories:
            from src.zero_shot_lanes import _compute_mean_polyline
            mean_traj = _compute_mean_polyline(pl_lane.trajectories, 32)
            mean_px = (mean_traj * image_wh).astype(np.int32)
            cv2.polylines(vis, [mean_px], False, color_bgr, 3, cv2.LINE_AA)

            # Label at midpoint — show encoder_rank if available, else geo_rank
            mid = mean_px[len(mean_px) // 2]
            rank_val = pred.get("encoder_rank", pred["lateral_rank"])
            rank_src = "e" if "encoder_rank" in pred else "g"
            label = (
                f"G{pred['group_id']}L{pred['lane_idx']} "
                f"{rank_src}r={rank_val:.2f}"
            )
            cv2.putText(
                vis, label, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                vis, label, tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color_bgr, 1, cv2.LINE_AA,
            )

    # Title
    title = f"Zero-Shot: {camera} ({len(pseudo_lanes)} lanes)"
    cv2.putText(
        vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 3, cv2.LINE_AA,
    )
    cv2.putText(
        vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 200, 0), 2, cv2.LINE_AA,
    )

    return vis


if __name__ == "__main__":
    main()
