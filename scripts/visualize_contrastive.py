#!/usr/bin/env python
"""Visualize contrastive lane embeddings.

Produces visualization PNGs:
- embedding_space.png  — t-SNE colored by lateral rank
- matching_grid.png    — top matched (query, ref) lane pairs
- similarity_heatmap.png — cosine similarity matrix by camera

Modes:
- --held-out CAM    : single held-out camera visualizations
- --all-cameras     : leave-one-out loop, one matching grid per camera
- (neither)         : global embedding space + heatmap only

Usage:
    python scripts/visualize_contrastive.py --checkpoint best.pt
    python scripts/visualize_contrastive.py --checkpoint best.pt --held-out I43_Keefe
    python scripts/visualize_contrastive.py --checkpoint best.pt --all-cameras
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import torch
import yaml
from torch.utils.data import DataLoader, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize contrastive lane embeddings")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--held-out", default=None, help="Held-out camera for query/ref split")
    parser.add_argument("--all-cameras", action="store_true", help="Loop over all cameras (leave-one-out)")
    parser.add_argument(
        "--output-dir",
        default="results/lane_contrastive/visualizations",
        help="Directory for output PNGs",
    )
    args = parser.parse_args()

    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import encode_lanes, load_trained_encoder
    from src.utils.contrastive_viz import (
        plot_embedding_space,
        plot_matching_grid,
        plot_similarity_heatmap,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_config = load_trained_encoder(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt_config

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    dataset = LaneDataset(
        config=config,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )

    batch_size = train_cfg.get("batch_size", 32)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_held_out(camera: str, out_dir: Path):
        """Generate matching grid + per-camera embedding/heatmap for one held-out camera."""
        held_out_indices = dataset.get_camera_indices(camera)
        if not held_out_indices:
            logger.warning(f"No samples for held-out camera: {camera}, skipping")
            return None, None, None

        held_out_set = set(held_out_indices)
        train_indices = [i for i in range(len(dataset)) if i not in held_out_set]

        ref_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )
        query_loader = DataLoader(
            Subset(dataset, held_out_indices),
            batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )

        ref_proj, ref_roles, ref_keys = encode_lanes(model, ref_loader, device, drop_geometry=False)
        query_proj, query_roles, query_keys = encode_lanes(model, query_loader, device, drop_geometry=True)

        # Matching: query -> ref
        sim_matrix = torch.mm(query_proj, ref_proj.t())
        best_match_idx = sim_matrix.argmax(dim=1)
        best_match_sim = sim_matrix.gather(1, best_match_idx.unsqueeze(1)).squeeze(1)

        q_dataset_indices = held_out_indices
        r_dataset_indices = [train_indices[idx.item()] for idx in best_match_idx]
        sims = best_match_sim.tolist()

        out_dir.mkdir(parents=True, exist_ok=True)
        plot_matching_grid(
            dataset, q_dataset_indices, r_dataset_indices, sims,
            str(out_dir / "matching_grid.png"),
        )

        # Return combined projections for global plots
        combined_proj = torch.cat([ref_proj, query_proj], dim=0)
        combined_roles = torch.cat([ref_roles, query_roles], dim=0)
        combined_keys = ref_keys + query_keys
        return combined_proj, combined_roles, combined_keys

    if args.all_cameras:
        # Leave-one-out loop: one matching grid per camera
        logger.info(f"Generating visualizations for all {len(dataset.cameras)} cameras")
        for camera in dataset.cameras:
            logger.info(f"  Held-out: {camera}")
            cam_dir = output_dir / camera
            visualize_held_out(camera, cam_dir)

        # Global embedding space + heatmap from full encode (all with geometry)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )
        all_proj, all_roles, all_keys = encode_lanes(model, loader, device, drop_geometry=False)
        all_cameras = [k.rsplit("_", 2)[0] for k in all_keys]

    elif args.held_out:
        combined_proj, combined_roles, combined_keys = visualize_held_out(
            args.held_out, output_dir,
        )
        if combined_proj is None:
            return
        all_proj, all_roles, all_keys = combined_proj, combined_roles, combined_keys
        all_cameras = [k.rsplit("_", 2)[0] for k in all_keys]

    else:
        # Encode all lanes with geometry
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )
        all_proj, all_roles, all_keys = encode_lanes(model, loader, device, drop_geometry=False)
        all_cameras = [k.rsplit("_", 2)[0] for k in all_keys]

    # Global embedding space + heatmap
    plot_embedding_space(
        all_proj, all_roles, all_cameras, all_keys,
        str(output_dir / "embedding_space.png"),
    )
    plot_similarity_heatmap(
        all_proj, all_cameras, all_keys,
        str(output_dir / "similarity_heatmap.png"),
    )

    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
