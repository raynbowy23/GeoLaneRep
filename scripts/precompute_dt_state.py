#!/usr/bin/env python3
"""Pre-compute encoder state for the GeoLane Twin to consume.

Generates JSON files that the twin's encoder_bridge.rs watches:
  - lane_state.json:      per-lane LOS, speed, density, anomaly, embeddings
  - topology.json:         generated lane geometry variants
  - variant_results.json:  ranked variant proposals with metrics
  - modifications.json:    behavioral-diff lane modification commands

This enables the D-series demo without running the encoder live.
The twin polls shared/dt_state/ and picks up updates automatically.

Usage:
    # D1: Pre-compute lane state for one camera
    python scripts/precompute_dt_state.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --camera US12_Monona \
        --output-dir shared/dt_state

    # D2: Also generate and rank variants
    python scripts/precompute_dt_state.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --camera US12_Monona \
        --output-dir shared/dt_state \
        --generate-variants \
        --diffusion-checkpoint results/generation/figures/diffusion_model.pt

    # Extract a camera frame for D1 figure
    python scripts/precompute_dt_state.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --camera US12_Monona \
        --output-dir shared/dt_state \
        --extract-frame dataset/511video/US12_Monona.mp4
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_encoder(args):
    """Load encoder model and dataset."""
    import yaml
    from src.data.lane_dataset import LaneDataset
    from src.training.zero_shot_eval import load_trained_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_encoder(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    # Filter to target camera if specified
    cameras = None
    if args.camera:
        cameras = [args.camera]

    dataset = LaneDataset(
        config=config,
        cameras=cameras,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )

    return model, config, dataset, device


def _encode_all(model, dataset, device):
    """Encode all lanes, return embeddings and batch."""
    from src.data.lane_dataset import collate_fn

    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    model.eval()
    with torch.no_grad():
        stats_input = torch.cat([
            batch["traj_stats"].to(device),
            batch["roles"].to(device),
        ], dim=-1)
        output = model(
            geometry=batch["geometry"].to(device),
            traj_polylines=batch["traj_polylines"].to(device),
            traj_mask=batch["traj_mask"].to(device),
            traj_stats=stats_input,
        )

    embeddings = output["embedding"].cpu().numpy()
    return output, batch, embeddings


# ---------------------------------------------------------------------------
# D1: Lane state export
# ---------------------------------------------------------------------------

def export_lane_state(model, dataset, device, output_dir, args):
    """Encode lanes and write lane_state.json for the twin."""
    from src.bridge.dt_export import DigitalTwinExporter
    from src.bridge.traffic_translator import EncoderTrafficBridge

    t_start = time.time()

    # Encode
    enc_output, batch, embeddings = _encode_all(model, dataset, device)
    t_encode = time.time()

    # Bridge → traffic metrics
    bridge = EncoderTrafficBridge()
    enc_out_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in enc_output.items()}
    metrics = bridge.translate_batch(dataset, enc_out_cpu)
    t_bridge = time.time()

    # Compute anomaly scores (baseline = zero drift for static pre-compute)
    anomaly_scores = np.zeros(len(metrics))

    # Export
    exporter = DigitalTwinExporter(output_dir=str(output_dir))
    exporter.write_lane_state(metrics, anomaly_scores, embeddings)
    t_export = time.time()

    # Write timing metadata
    timing = {
        "camera": args.camera or "all",
        "n_lanes": len(dataset),
        "encode_time_s": round(t_encode - t_start, 3),
        "bridge_time_s": round(t_bridge - t_encode, 3),
        "export_time_s": round(t_export - t_bridge, 3),
        "total_time_s": round(t_export - t_start, 3),
        "timestamp": time.time(),
    }
    timing_path = output_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    logger.info(
        f"Lane state exported: {len(metrics)} lanes in {timing['total_time_s']:.1f}s "
        f"(encode={timing['encode_time_s']:.1f}s, "
        f"bridge={timing['bridge_time_s']:.1f}s, "
        f"export={timing['export_time_s']:.1f}s)"
    )

    return enc_output, batch, embeddings, metrics


# ---------------------------------------------------------------------------
# D2: Variant generation and ranking
# ---------------------------------------------------------------------------

def generate_and_rank_variants(
    model, dataset, device, embeddings_baseline, metrics_baseline,
    output_dir, args,
):
    """Generate lane variants, evaluate, rank, and write results for twin."""
    from src.bridge.dt_export import DigitalTwinExporter
    from src.bridge.traffic_translator import EncoderTrafficBridge
    from src.bridge.variant_comparator import VariantComparator
    from src.data.lane_dataset import collate_fn
    from src.generation.spec import LaneSpecification

    bridge = EncoderTrafficBridge()
    comparator = VariantComparator()
    exporter = DigitalTwinExporter(output_dir=str(output_dir))

    # Pick the target group (largest group in target camera)
    cam = args.camera or dataset.samples[0].camera
    cam_samples = [s for s in dataset.samples if s.camera == cam]
    group_counts = {}
    for s in cam_samples:
        group_counts.setdefault(s.group_id, []).append(s)
    target_gid = max(group_counts, key=lambda g: len(group_counts[g]))
    target_samples = group_counts[target_gid]

    logger.info(
        f"Generating variants for {cam} group {target_gid} "
        f"({len(target_samples)} lanes)"
    )

    # Build retrieval index and resolver
    from src.generation.diffusion import (
        LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer,
    )
    from src.generation.retrieval import build_retrieval_index
    from src.generation.spec import SpecEmbeddingResolver

    polyline_k = dataset.polyline_k

    index = build_retrieval_index(model, dataset, device, use_embeddings=True)
    resolver = SpecEmbeddingResolver(index, dataset)

    # Load diffusion trainer
    K = index.K
    geom_dim = K * 2
    cond_dim = index.D

    denoiser = LaneDenoiser(
        geom_dim=geom_dim, t_dim=64, cond_dim=cond_dim, hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))
    trainer.load(args.diffusion_checkpoint)
    logger.info(f"Loaded diffusion checkpoint: {args.diffusion_checkpoint}")

    from src.generation.directed import DirectedLaneGenerator
    generator = DirectedLaneGenerator(
        resolver=resolver,
        trainer=trainer,
        encoder=model,
        dataset=dataset,
        device=device,
    )

    # Define variants to generate
    variant_specs = [
        ("add_rightmost", "Add rightmost lane",
         LaneSpecification.rightmost(camera=cam, group_id=target_gid)),
        ("add_leftmost", "Add leftmost lane",
         LaneSpecification.leftmost(camera=cam, group_id=target_gid)),
    ]

    # Check if group has merge lanes
    has_merge = any(s.role.has_successor for s in target_samples)
    if has_merge:
        variant_specs.append(("add_merge", "Add merge lane",
            LaneSpecification.merge_lane(camera=cam, group_id=target_gid)))

    # Generate each variant
    variant_results = []
    generated_lanes = []

    for variant_name, description, spec in variant_specs:
        logger.info(f"Generating variant: {variant_name}")

        # Determine lane_type from spec
        if spec.is_rightmost:
            lane_type = "rightmost"
        elif spec.is_leftmost:
            lane_type = "leftmost"
        elif spec.has_successor:
            lane_type = "merge"
        else:
            lane_type = "unknown"

        try:
            result = generator.generate(spec, n_candidates=5)

            # Compute metrics for the best candidate
            best_geom = result.best  # (K, 2)
            cos_sim = float(result.scores.max())

            target_emb = result.target_embedding
            emb_norm = float(np.linalg.norm(target_emb))

            variant_results.append({
                "variant_name": variant_name,
                "description": description,
                "lane_type": lane_type,
                "camera": cam,
                "group_id": target_gid,
                "best_score": cos_sim,
                "n_candidates": len(result.candidates),
                "target_embedding_norm": emb_norm,
            })

            generated_lanes.append({
                "points": best_geom,
                "lane_type": lane_type,
                "confidence": cos_sim,
                "variant_name": variant_name,
            })

            logger.info(
                f"  {variant_name}: best_score={cos_sim:.3f}, "
                f"emb_norm={emb_norm:.2f}"
            )

        except Exception as e:
            logger.warning(f"  {variant_name} failed: {e}")
            variant_results.append({
                "variant_name": variant_name,
                "description": description,
                "lane_type": lane_type,
                "camera": cam,
                "group_id": target_gid,
                "best_score": 0.0,
                "n_candidates": 0,
                "error": str(e),
            })

    # Rank by score
    variant_results.sort(key=lambda v: v.get("best_score", 0), reverse=True)
    for rank, v in enumerate(variant_results):
        v["rank"] = rank + 1

    # Write variant_results.json (new file for twin)
    variant_payload = {
        "version": "1.0",
        "timestamp": time.time(),
        "generation": 1,
        "source": "geolane_encoder",
        "data": {
            "camera": cam,
            "group_id": target_gid,
            "n_existing_lanes": len(target_samples),
            "variants": variant_results,
        },
    }
    variant_path = output_dir / "variant_results.json"
    with open(variant_path, "w") as f:
        json.dump(variant_payload, f, indent=2, default=str)
    logger.info(f"Wrote {len(variant_results)} variant results to {variant_path}")

    # Write topology.json with generated lanes
    if generated_lanes:
        exporter.write_topology(generated_lanes, cam, target_gid)

    # Write modifications.json for the best variant
    if variant_results and variant_results[0].get("best_score", 0) > 0:
        best = variant_results[0]
        # Find the generated geometry for the best variant
        best_shape = []
        for gl in generated_lanes:
            if gl["variant_name"] == best["variant_name"]:
                pts = gl["points"]
                if isinstance(pts, np.ndarray):
                    best_shape = [[float(p[0]), float(p[1])] for p in pts]
                else:
                    best_shape = pts
                break

        from src.bridge.dt_export import ModificationEntry
        from dataclasses import asdict
        mod = ModificationEntry(
            action="add",
            edge_id=f"{cam}_g{target_gid}",
            reason=f"Variant '{best['variant_name']}': {best.get('description', '')} "
                   f"(score={best['best_score']:.3f})",
            shape=best_shape,
            width=3.2,
            speed_limit=24.6,
        )
        mod_payload = {
            "version": "1.0",
            "timestamp": time.time(),
            "generation": 1,
            "source": "geolane_encoder",
            "data": {
                "modifications": [asdict(mod)],
                "n_modifications": 1,
            },
        }
        mod_path = output_dir / "modifications.json"
        with open(mod_path, "w") as f:
            json.dump(mod_payload, f, indent=2)

    return variant_results


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_camera_frame(video_path, output_dir, camera_name):
    """Extract a representative frame from camera video for D1 figure."""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping frame extraction")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None

    # Seek to 25% of video for a representative frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = total_frames // 4
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning("Failed to read frame")
        return None

    frame_path = output_dir / f"{camera_name}_frame.png"
    cv2.imwrite(str(frame_path), frame)
    logger.info(f"Extracted frame {target_frame}/{total_frames} → {frame_path}")
    return frame_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute encoder state for GeoLane Twin"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", default=None,
                        help="Target camera (default: all)")
    parser.add_argument("--output-dir", default="shared/dt_state",
                        help="Shared directory for twin JSON export")
    parser.add_argument("--generate-variants", action="store_true",
                        help="Also generate and rank lane variants (D2)")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Diffusion model checkpoint (required for --generate-variants)")
    parser.add_argument("--extract-frame", default=None,
                        help="Path to video file to extract a representative frame")
    args = parser.parse_args()

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    model, config, dataset, device = _load_encoder(args)

    if len(dataset) == 0:
        logger.error(f"No lanes found for camera: {args.camera}")
        return

    # Determine cameras to process
    if args.camera:
        cameras = [args.camera]
    else:
        cameras = sorted(set(s.camera for s in dataset.samples))

    all_camera_results = []

    for cam in cameras:
        logger.info(f"Processing camera: {cam}")

        # Per-camera output directory
        cam_output_dir = base_output_dir / cam
        cam_output_dir.mkdir(parents=True, exist_ok=True)

        # Filter dataset to this camera
        cam_args = argparse.Namespace(**vars(args))
        cam_args.camera = cam

        cam_model, cam_config, cam_dataset, cam_device = model, config, dataset, device
        if len(cameras) > 1:
            # Re-create dataset filtered to this camera
            model_cfg = config.get("model", {})
            train_cfg = config.get("contrastive_training", {})
            from src.data.lane_dataset import LaneDataset
            cam_dataset = LaneDataset(
                config=config,
                cameras=[cam],
                polyline_k=model_cfg.get("polyline_k", 16),
                max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
                role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
            )
            if len(cam_dataset) == 0:
                logger.warning(f"No lanes found for camera: {cam}, skipping")
                continue

        # D1: Export lane state
        enc_output, batch, embeddings, metrics = export_lane_state(
            model, cam_dataset, device, cam_output_dir, cam_args,
        )

        # Optional: extract camera frame
        if args.extract_frame:
            extract_camera_frame(args.extract_frame, cam_output_dir, cam)

        # D2: Generate variants
        if args.generate_variants:
            if args.diffusion_checkpoint is None:
                logger.error("--diffusion-checkpoint required for --generate-variants")
                return
            generate_and_rank_variants(
                model, cam_dataset, device, embeddings, metrics,
                cam_output_dir, cam_args,
            )

        all_camera_results.append(cam)

        # Also write to base dir if single camera (backward compat)
        if len(cameras) == 1:
            import shutil
            for f in cam_output_dir.iterdir():
                shutil.copy2(f, base_output_dir / f.name)

    # Copy C2 re-encoding results if they exist
    c2_results_dir = PROJECT_ROOT / "results" / "c2_synthesis_loop"
    if c2_results_dir.exists():
        import shutil
        c2_summary = c2_results_dir / "c2_summary.json"
        if c2_summary.exists():
            # Write c2_metrics.json for twin consumption
            with open(c2_summary) as f:
                c2_data = json.load(f)

            c2_metrics = {
                "version": "1.0",
                "timestamp": time.time(),
                "generation": 1,
                "source": "geolane_encoder",
                "data": c2_data,
            }
            c2_path = base_output_dir / "c2_metrics.json"
            with open(c2_path, "w") as f:
                json.dump(c2_metrics, f, indent=2)
            logger.info(f"Copied C2 re-encoding metrics to {c2_path}")

        # Also copy per-camera C2 results into camera subdirs
        c2_per_cam = c2_results_dir / "per_camera"
        if c2_per_cam.exists():
            for cam in all_camera_results:
                cam_c2 = c2_per_cam / f"{cam}.json"
                if cam_c2.exists():
                    with open(cam_c2) as f:
                        cam_c2_data = json.load(f)
                    cam_c2_out = {
                        "version": "1.0",
                        "timestamp": time.time(),
                        "generation": 1,
                        "source": "geolane_encoder",
                        "data": cam_c2_data,
                    }
                    out_path = base_output_dir / cam / "c2_metrics.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, "w") as f:
                        json.dump(cam_c2_out, f, indent=2)
    else:
        logger.info("No C2 results found. Run 'make c2-experiment-all' first to include re-encoding metrics.")

    # Write camera index for twin discovery
    index = {
        "version": "1.0",
        "timestamp": time.time(),
        "source": "geolane_encoder",
        "cameras": all_camera_results,
        "default_camera": cameras[0] if cameras else None,
    }
    index_path = base_output_dir / "camera_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"All outputs written to {base_output_dir}/")
    logger.info(f"Cameras: {all_camera_results}")
    logger.info("Start geolane_twin and point it to this directory.")


if __name__ == "__main__":
    main()
