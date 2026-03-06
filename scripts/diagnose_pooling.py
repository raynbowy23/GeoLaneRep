#!/usr/bin/env python3
"""Pooling diagnostic: load a graph cache, run model, print what pooling produces.

Checks:
  1. Assignment S: is it collapsed or spread? (entropy, mass per slot)
  2. Soft centroids: do they cover the tracklet extent?
  3. Offsets: is offset_head pushing nodes off-centroid?
  4. Headings: do soft tangents match tracklet flow?
  5. Separation: pairwise distances between lanelet nodes

Usage:
    python scripts/diagnose_pooling.py --config configs/lanelet_core.yaml
    python scripts/diagnose_pooling.py --config configs/lanelet.yaml --checkpoint results/lanelet_discovery/checkpoints/best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src.…` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint (omit for random init)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample", type=int, default=0, help="Sample index")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to diagnose")
    parser.add_argument("--cache_dir", default=None, help="Override graph cache directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from src.models.lane_repr import LaneReprModel
    model = LaneReprModel(config).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        # Tolerate missing/extra keys (architecture may differ)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.info("No checkpoint — using random init (tests if pooling geometry works)")

    # Load graph cache
    if args.cache_dir:
        from glob import glob
        cache_dir = Path(args.cache_dir)
        candidates = list(cache_dir.glob(f"{args.split}_*.pt"))
        cache_path = candidates[0] if candidates else None
    else:
        from src.data.dataset import TrackletDataset
        cache_path = TrackletDataset._cache_path(config, args.split)
    if cache_path and cache_path.exists():
        samples = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded {len(samples)} cached graphs from {cache_path}")
    else:
        logger.error(f"No graph cache found at {cache_path}. Run training first to build cache.")
        sys.exit(1)

    model.eval()
    M = config.get("model", {}).get("num_lanelet_nodes", 16)

    end_idx = min(args.sample + args.num_samples, len(samples))
    for si in range(args.sample, end_idx):
        data = samples[si].to(device)
        logger.info(f"\n{'='*70}")
        logger.info(f"SAMPLE {si}: camera={getattr(data, 'camera_loc', '?')}, "
                     f"N={data.num_nodes} tracklets, E={data.edge_index.shape[1]} edges")
        logger.info(f"{'='*70}")

        # Tracklet stats
        centroids = data.centroids.cpu().numpy()
        tangents = data.tangents.cpu().numpy()
        c_min = centroids.min(axis=0)
        c_max = centroids.max(axis=0)
        c_range = c_max - c_min
        logger.info(f"\n[Tracklets]")
        logger.info(f"  Spatial extent: x=[{c_min[0]:.1f}, {c_max[0]:.1f}] y=[{c_min[1]:.1f}, {c_max[1]:.1f}]  "
                     f"range=({c_range[0]:.1f}, {c_range[1]:.1f})m")
        mean_tangent = tangents.mean(axis=0)
        logger.info(f"  Mean tangent: ({mean_tangent[0]:.3f}, {mean_tangent[1]:.3f})  "
                     f"heading={np.degrees(np.arctan2(mean_tangent[1], mean_tangent[0])):.1f}deg")

        # Forward pass
        with torch.no_grad():
            output = model(data)

        S = output["assignment_matrix"].cpu().numpy()       # (N, M)
        pos = output["lanelet_positions"].cpu().numpy()      # (M, 2)
        head = output["lanelet_headings"].cpu().numpy()      # (M, 2)
        conf = output["lanelet_confidence"].cpu().numpy()    # (M,)

        # --- 1. Assignment analysis ---
        logger.info(f"\n[Assignment S] shape=({S.shape[0]}, {S.shape[1]})")
        mass = S.sum(axis=0)  # (M,) total mass per slot
        hard_assign = S.argmax(axis=1)  # (N,) which slot each tracklet picks
        slot_counts = np.bincount(hard_assign, minlength=M)

        # Per-tracklet entropy
        S_clamped = np.clip(S, 1e-8, 1.0)
        per_tracklet_entropy = -(S_clamped * np.log(S_clamped)).sum(axis=1)
        max_entropy = np.log(M)

        logger.info(f"  Per-tracklet entropy: mean={per_tracklet_entropy.mean():.3f}  "
                     f"(max possible={max_entropy:.3f}, ratio={per_tracklet_entropy.mean()/max_entropy:.2%})")
        logger.info(f"  Mass per slot (soft): min={mass.min():.1f} max={mass.max():.1f} "
                     f"std={mass.std():.1f}  ideal={S.shape[0]/M:.1f}")
        logger.info(f"  Hard counts per slot: {slot_counts.tolist()}")

        empty_slots = (slot_counts == 0).sum()
        dominant_slots = (slot_counts > S.shape[0] * 0.2).sum()
        logger.info(f"  Empty slots: {empty_slots}/{M}  "
                     f"Dominant slots (>20% mass): {dominant_slots}/{M}")

        if empty_slots > M // 2:
            logger.warning(f"  ** PROBLEM: {empty_slots}/{M} slots empty — assignment is collapsed!")
        if dominant_slots >= 1 and dominant_slots <= 2:
            logger.warning(f"  ** PROBLEM: only {dominant_slots} slot(s) hold most tracklets — severe collapse!")

        # --- 2. Position analysis ---
        logger.info(f"\n[Lanelet Positions]")
        # Compute soft centroids manually
        S_t = S.T  # (M, N)
        mass_vec = S_t.sum(axis=1, keepdims=True)
        mass_vec = np.clip(mass_vec, 1e-8, None)
        soft_centroids = (S_t @ centroids) / mass_vec  # (M, 2)
        offsets = pos - soft_centroids  # (M, 2) — offset_head contribution
        offset_norms = np.linalg.norm(offsets, axis=1)

        logger.info(f"  Soft centroids range: x=[{soft_centroids[:,0].min():.1f}, {soft_centroids[:,0].max():.1f}] "
                     f"y=[{soft_centroids[:,1].min():.1f}, {soft_centroids[:,1].max():.1f}]")
        logger.info(f"  Final positions range: x=[{pos[:,0].min():.1f}, {pos[:,0].max():.1f}] "
                     f"y=[{pos[:,1].min():.1f}, {pos[:,1].max():.1f}]")
        logger.info(f"  Offset magnitudes: mean={offset_norms.mean():.2f}m  max={offset_norms.max():.2f}m")

        # Coverage: what fraction of tracklet spatial extent is covered?
        pos_range_x = pos[:, 0].max() - pos[:, 0].min()
        pos_range_y = pos[:, 1].max() - pos[:, 1].min()
        coverage_x = pos_range_x / max(c_range[0], 1e-4)
        coverage_y = pos_range_y / max(c_range[1], 1e-4)
        logger.info(f"  Coverage: x={coverage_x:.1%} of tracklet range, y={coverage_y:.1%}")

        if coverage_x < 0.3 and coverage_y < 0.3:
            logger.warning(f"  ** PROBLEM: lanelet nodes cover <30% of tracklet spatial extent — condensed!")

        # --- 3. Pairwise distances ---
        logger.info(f"\n[Separation]")
        from scipy.spatial.distance import pdist
        pw_dists = pdist(pos)
        logger.info(f"  Pairwise distances: min={pw_dists.min():.2f}m  median={np.median(pw_dists):.2f}m  "
                     f"max={pw_dists.max():.2f}m")
        n_close = (pw_dists < 3.5).sum()
        n_total = len(pw_dists)
        logger.info(f"  Pairs < 3.5m (one lane width): {n_close}/{n_total}")

        if pw_dists.min() < 0.5:
            logger.warning(f"  ** PROBLEM: some nodes are <0.5m apart — effectively collapsed!")

        # --- 4. Heading analysis ---
        logger.info(f"\n[Headings]")
        soft_tangents = (S_t @ tangents) / mass_vec  # (M, 2)
        soft_t_norms = np.linalg.norm(soft_tangents, axis=1, keepdims=True)
        soft_tangents_unit = soft_tangents / np.clip(soft_t_norms, 1e-8, None)

        # Cosine similarity between final heading and soft tangent
        cos_sim = (head * soft_tangents_unit).sum(axis=1)
        logger.info(f"  Heading vs soft-tangent cosine sim: mean={cos_sim.mean():.3f}  min={cos_sim.min():.3f}")
        misaligned = (cos_sim < 0.5).sum()
        if misaligned > 0:
            logger.warning(f"  ** {misaligned}/{M} nodes have heading misaligned >60deg from soft tangent")

        # All headings should roughly agree with lane group heading
        lane_heading = getattr(data, "lane_group_heading", None)
        if lane_heading is not None:
            lh = float(lane_heading)
            lh_vec = np.array([np.cos(lh), np.sin(lh)])
            head_cos = (head * lh_vec).sum(axis=1)
            logger.info(f"  Heading vs lane-group-heading cosine: mean={head_cos.mean():.3f}")

        heading_angles = np.degrees(np.arctan2(head[:, 1], head[:, 0]))
        logger.info(f"  Heading angles (deg): {np.round(heading_angles, 1).tolist()}")

        # --- 5. Confidence ---
        logger.info(f"\n[Confidence]")
        conf_thresh = config.get("training", {}).get("confidence_threshold", 0.3)
        n_active = (conf > conf_thresh).sum()
        logger.info(f"  Confidence: min={conf.min():.3f} max={conf.max():.3f} mean={conf.mean():.3f}")
        logger.info(f"  Active (>{conf_thresh}): {n_active}/{M}")

        # --- 6. GT comparison (if available) ---
        gt_pos = getattr(data, "gt_lanelet_positions", None)
        if gt_pos is not None and len(gt_pos) > 0:
            gt_pos_np = gt_pos.cpu().numpy()
            logger.info(f"\n[GT Comparison]")
            logger.info(f"  GT nodes: {len(gt_pos_np)}")
            logger.info(f"  GT range: x=[{gt_pos_np[:,0].min():.1f}, {gt_pos_np[:,0].max():.1f}] "
                         f"y=[{gt_pos_np[:,1].min():.1f}, {gt_pos_np[:,1].max():.1f}]")
            # Nearest GT node for each pred
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(pos, gt_pos_np)
            nearest_gt_dist = dist_matrix.min(axis=1)
            logger.info(f"  Pred→GT nearest: mean={nearest_gt_dist.mean():.2f}m  max={nearest_gt_dist.max():.2f}m")

        # --- Summary verdict ---
        logger.info(f"\n[VERDICT]")
        issues = []
        if empty_slots > M // 2:
            issues.append("COLLAPSED assignment (most slots empty)")
        if coverage_x < 0.3 and coverage_y < 0.3:
            issues.append("CONDENSED positions (<30% coverage)")
        if pw_dists.min() < 0.5:
            issues.append("OVERLAPPING nodes (<0.5m apart)")
        if misaligned > M // 3:
            issues.append(f"MISALIGNED headings ({misaligned}/{M} nodes)")
        if offset_norms.max() > 50:
            issues.append(f"LARGE offsets (max={offset_norms.max():.1f}m)")

        if issues:
            for issue in issues:
                logger.warning(f"  !! {issue}")
        else:
            logger.info(f"  OK — pooling looks healthy")


if __name__ == "__main__":
    main()
