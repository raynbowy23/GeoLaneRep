#!/usr/bin/env python3
"""Planning Evaluation: Full encoder→generator→SUMO→re-encode loop.

End-to-end pipeline that uses the generation pipeline to suggest variants
and measures their traffic impact via SUMO simulation:

    1. Encode:      Real camera → encoder → embeddings (behavioral understanding)
    2. Generate:    Encoder embedding → diffusion generator → lane variant
    3. Map:         Match generated variant's group to SUMO edge
    4. Inject:      Add lane to SUMO network via netconvert
    5. Re-simulate: SUMO(modified) → FCD → encoder → new embeddings
    6. Compare:     ΔSpeed, ΔDensity, ΔLOS + embedding shift δ

This is the full Path A (Future Planning) evaluation:
    "The encoder observes traffic, the generator suggests a lane,
     SUMO simulates the impact, and we measure improvement."

Output: results/planning_eval/
    - planning_results.json   — per-camera before/after comparison
    - planning_summary.json   — aggregate impact statistics

Usage:
    # Single camera (requires diffusion checkpoint)
    python scripts/run_planning_eval.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --diffusion-checkpoint results/generation/figures/diffusion_model.pt \
        --camera US12_Monona

    # All cameras
    python scripts/run_planning_eval.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --diffusion-checkpoint results/generation/figures/diffusion_model.pt \
        --all-cameras

    # Without diffusion (falls back to position-only variant)
    python scripts/run_planning_eval.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --camera US12_Monona --no-generate
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

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

def load_encoder(args):
    """Load encoder model and config."""
    import yaml
    from src.training.zero_shot_eval import load_trained_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_encoder(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    return model, config, device


def build_dataset(config, cameras=None):
    """Build LaneDataset filtered to specific cameras."""
    from src.data.lane_dataset import LaneDataset

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    return LaneDataset(
        config=config,
        cameras=cameras,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )


def encode_all(model, dataset, device):
    """Encode all lanes, return embeddings."""
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

    return output["embedding"].cpu().numpy()


# ---------------------------------------------------------------------------
# Generation pipeline
# ---------------------------------------------------------------------------

def build_generator(model, dataset, device, diffusion_checkpoint):
    """Build the full generation pipeline: encoder → retrieval → diffusion."""
    from src.generation.retrieval import build_retrieval_index
    from src.generation.spec import SpecEmbeddingResolver
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer
    from src.generation.directed import DirectedLaneGenerator

    index = build_retrieval_index(model, dataset, device, use_embeddings=True)
    resolver = SpecEmbeddingResolver(index, dataset)

    K = index.K
    denoiser = LaneDenoiser(
        geom_dim=K * 2, t_dim=64, cond_dim=index.D, hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))
    trainer.load(diffusion_checkpoint)

    generator = DirectedLaneGenerator(
        resolver=resolver, trainer=trainer,
        encoder=model, dataset=dataset, device=device,
    )

    return generator, index


def generate_variant(generator, dataset, camera, target_gid, variant_types=None):
    """Generate lane variants using the full generation pipeline.

    Returns:
        List of (variant_name, lane_type, score, geometry) tuples.
    """
    from src.generation.spec import LaneSpecification

    cam_samples = [s for s in dataset.samples if s.camera == camera]
    group_samples = [s for s in cam_samples if s.group_id == target_gid]

    if variant_types is None:
        variant_types = ["rightmost", "leftmost"]
        # Add merge if the group has merge lanes
        if any(s.role.has_successor for s in group_samples):
            variant_types.append("merge")

    results = []
    for vtype in variant_types:
        try:
            if vtype == "rightmost":
                spec = LaneSpecification.rightmost(camera=camera, group_id=target_gid)
            elif vtype == "leftmost":
                spec = LaneSpecification.leftmost(camera=camera, group_id=target_gid)
            elif vtype == "merge":
                spec = LaneSpecification.merge_lane(camera=camera, group_id=target_gid)
            else:
                continue

            result = generator.generate(spec, n_candidates=5)
            score = float(result.scores.max())
            best_geom = result.best  # (K, 2) in image space

            results.append({
                "variant_name": f"add_{vtype}",
                "lane_type": vtype,
                "score": score,
                "geometry": best_geom,
                "n_candidates": len(result.candidates),
                "mean_score": float(result.scores.mean()),
            })
            logger.info(f"  Generated {vtype}: score={score:.3f}")

        except Exception as e:
            logger.warning(f"  Generation failed for {vtype}: {e}")
            results.append({
                "variant_name": f"add_{vtype}",
                "lane_type": vtype,
                "score": 0.0,
                "geometry": None,
                "error": str(e),
            })

    return results


# ---------------------------------------------------------------------------
# SUMO simulation helpers
# ---------------------------------------------------------------------------

def run_sumo_on_network(net_path, sim_duration, flow_rate):
    """Run SUMO on a network and return per-lane metrics."""
    from src.bridge.sumo_runner import _parse_net_lanes, _parse_edge_output

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        edge_output = tmpdir_path / "edge_output.xml"

        # Generate demand
        random_trips = Path("/usr/share/sumo/tools/randomTrips.py")
        if not random_trips.exists():
            logger.error("randomTrips.py not found")
            return []

        period = max(0.5, 3600.0 / flow_rate)
        routes_file = tmpdir_path / "routes.rou.xml"
        trips_file = tmpdir_path / "trips.xml"
        cmd_trips = [
            "python", str(random_trips),
            "-n", str(net_path),
            "-o", str(trips_file),
            "-r", str(routes_file),
            "-e", str(sim_duration),
            "-p", str(period),
            "--fringe-factor", "5",
            "--allow-fringe",
            "-t", 'departLane="free" departSpeed="max"',
            "--validate",
        ]
        trip_result = subprocess.run(
            cmd_trips, capture_output=True, text=True, timeout=60,
        )
        if trip_result.returncode != 0 or not routes_file.exists():
            logger.warning(f"Trip generation failed: {trip_result.stderr[:200]}")
            return []

        # Lane data collection
        additional_file = tmpdir_path / "additional.xml"
        additional_file.write_text(
            f'<additional>\n'
            f'  <laneData id="lane_metrics" '
            f'file="{edge_output}" '
            f'freq="300" '
            f'excludeEmpty="true"/>\n'
            f'</additional>\n'
        )

        cmd = [
            "sumo",
            "-n", str(net_path),
            "-r", str(routes_file),
            "-a", str(additional_file),
            "--end", str(sim_duration),
            "--no-step-log", "true",
            "--duration-log.statistics", "true",
            "--ignore-route-errors", "true",
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if not edge_output.exists():
            return []

        lane_geoms = _parse_net_lanes(net_path)
        return _parse_edge_output(edge_output, lane_geoms)


def summarize_metrics(metrics_list) -> dict:
    """Compute aggregate traffic statistics from SUMO lane metrics."""
    if not metrics_list:
        return {
            "n_lanes": 0, "mean_speed_mph": 0.0, "mean_density_veh_mi_ln": 0.0,
            "mean_flow_veh_hr": 0.0, "los_distribution": {},
        }

    speeds = [m.speed_mph for m in metrics_list]
    densities = [m.density_veh_mi_ln for m in metrics_list]
    flows = [m.flow_veh_hr for m in metrics_list]

    los_counts = {}
    for m in metrics_list:
        los_counts[m.los] = los_counts.get(m.los, 0) + 1

    return {
        "n_lanes": len(metrics_list),
        "mean_speed_mph": float(np.mean(speeds)),
        "std_speed_mph": float(np.std(speeds)),
        "mean_density_veh_mi_ln": float(np.mean(densities)),
        "std_density_veh_mi_ln": float(np.std(densities)),
        "mean_flow_veh_hr": float(np.mean(flows)),
        "std_flow_veh_hr": float(np.std(flows)),
        "los_distribution": los_counts,
        "per_lane": [
            {
                "lane_id": m.lane_id,
                "speed_mph": round(m.speed_mph, 1),
                "density_veh_mi_ln": round(m.density_veh_mi_ln, 1),
                "flow_veh_hr": round(m.flow_veh_hr, 1),
                "los": m.los,
            }
            for m in metrics_list
        ],
    }


def compute_deltas(baseline: dict, modified: dict) -> dict:
    """Compute before→after deltas in traffic metrics."""
    return {
        "delta_speed_mph": modified["mean_speed_mph"] - baseline["mean_speed_mph"],
        "delta_density_veh_mi_ln": modified["mean_density_veh_mi_ln"] - baseline["mean_density_veh_mi_ln"],
        "delta_flow_veh_hr": modified["mean_flow_veh_hr"] - baseline["mean_flow_veh_hr"],
        "delta_n_lanes": modified["n_lanes"] - baseline["n_lanes"],
        "los_before": baseline["los_distribution"],
        "los_after": modified["los_distribution"],
        "speed_improved": modified["mean_speed_mph"] > baseline["mean_speed_mph"],
        "density_improved": modified["mean_density_veh_mi_ln"] < baseline["mean_density_veh_mi_ln"],
    }


# ---------------------------------------------------------------------------
# Map encoder group → SUMO edge
# ---------------------------------------------------------------------------

def match_group_to_edge(dataset, camera, group_id, baseline_metrics):
    """Match an encoder lane group to the best SUMO edge.

    Uses lane count similarity and picks the busiest matching edge.
    """
    cam_samples = [s for s in dataset.samples if s.camera == camera]
    group_samples = [s for s in cam_samples if s.group_id == group_id]
    group_n_lanes = len(group_samples)

    # Group SUMO metrics by edge
    edge_metrics = {}
    for m in baseline_metrics:
        edge_metrics.setdefault(m.edge_id, []).append(m)

    # Find edge with closest lane count, break ties by density (busiest)
    best_edge = None
    best_score = -1

    for eid, ms in edge_metrics.items():
        n = len(ms)
        count_sim = 1.0 / (1.0 + abs(n - group_n_lanes))
        mean_density = np.mean([m.density_veh_mi_ln for m in ms])
        # Prefer matching lane count, then higher density
        score = count_sim * 100 + mean_density
        if score > best_score:
            best_score = score
            best_edge = eid

    return best_edge


# ---------------------------------------------------------------------------
# Full evaluation per camera
# ---------------------------------------------------------------------------

def evaluate_camera(
    model, dataset, device, camera, sumo_root,
    generator=None, diffusion_available=False,
    sim_duration=3600, flow_rate=1200,
) -> Optional[dict]:
    """Full pipeline: encode → generate → inject → simulate → compare.

    Steps:
        1. Encode real lanes → baseline embeddings
        2. Run SUMO(original) → baseline traffic metrics
        3. Generate variant via diffusion (or fallback to position-only)
        4. Map variant's group to busiest SUMO edge
        5. Inject lane into SUMO network via netconvert
        6. Run SUMO(modified) → FCD → re-encode → new embeddings + traffic metrics
        7. Compare traffic deltas + embedding shift
    """
    from src.bridge.sumo_runner import run_sumo_simulation, _parse_net_lanes
    from src.bridge.sumo_modifier import add_lane_to_network, LaneAddition
    from src.bridge.trajectory_extractor import extract_encoder_inputs
    from src.bridge.reencoder import LaneReencoder

    t_start = time.time()

    net_path = sumo_root / camera / "osm.net.xml"
    if not net_path.exists():
        logger.warning(f"[{camera}] No SUMO network found")
        return None

    # Step 1: Encode real lanes
    logger.info(f"[{camera}] Encoding real lanes...")
    embeddings_real = encode_all(model, dataset, device)
    t_encode = time.time()

    # Step 2: Baseline SUMO simulation
    logger.info(f"[{camera}] Running baseline SUMO simulation...")
    baseline_metrics = run_sumo_simulation(
        camera, sumo_root,
        sim_duration=sim_duration, flow_rate=flow_rate,
        collect_fcd=True,
    )
    if not baseline_metrics:
        logger.warning(f"[{camera}] Baseline simulation produced no metrics")
        return None

    baseline_summary = summarize_metrics(baseline_metrics)
    t_baseline = time.time()

    logger.info(
        f"[{camera}] Baseline: {baseline_summary['n_lanes']} lanes, "
        f"speed={baseline_summary['mean_speed_mph']:.1f} mph, "
        f"density={baseline_summary['mean_density_veh_mi_ln']:.1f} veh/mi/ln, "
        f"LOS={baseline_summary['los_distribution']}"
    )

    # Step 3: Generate variant via encoder→generator pipeline
    # Find the largest group for this camera
    cam_samples = [s for s in dataset.samples if s.camera == camera]
    group_counts = {}
    for s in cam_samples:
        group_counts.setdefault(s.group_id, []).append(s)
    target_gid = max(group_counts, key=lambda g: len(group_counts[g]))

    variant_info = None
    if generator is not None and diffusion_available:
        logger.info(f"[{camera}] Generating variants for group {target_gid}...")
        variants = generate_variant(generator, dataset, camera, target_gid)
        # Pick best variant by generation score
        valid_variants = [v for v in variants if v.get("score", 0) > 0]
        if valid_variants:
            variant_info = max(valid_variants, key=lambda v: v["score"])
            logger.info(
                f"[{camera}] Best variant: {variant_info['variant_name']} "
                f"(score={variant_info['score']:.3f})"
            )

    # Determine lane type to add
    if variant_info:
        lane_type = variant_info["lane_type"]
    else:
        lane_type = "rightmost"
        logger.info(f"[{camera}] No generator available, using default: add_{lane_type}")

    # Step 4: Map group to SUMO edge
    target_edge = match_group_to_edge(dataset, camera, target_gid, baseline_metrics)
    if target_edge is None:
        logger.warning(f"[{camera}] Could not match group to SUMO edge")
        return None

    logger.info(f"[{camera}] Mapped group {target_gid} → edge {target_edge}")

    # Step 5: Inject lane into SUMO network
    addition = LaneAddition(edge_id=target_edge, position=lane_type)
    try:
        modified_net = add_lane_to_network(net_path, addition)
    except Exception as e:
        logger.warning(f"[{camera}] Failed to modify network: {e}")
        return None

    t_modify = time.time()

    # Step 6: Re-simulate modified network
    logger.info(f"[{camera}] Running modified SUMO simulation...")
    modified_metrics = run_sumo_on_network(modified_net, sim_duration, flow_rate)

    if not modified_metrics:
        logger.warning(f"[{camera}] Modified simulation produced no metrics")
        modified_net.unlink(missing_ok=True)
        return None

    modified_summary = summarize_metrics(modified_metrics)
    t_modified = time.time()

    logger.info(
        f"[{camera}] Modified: {modified_summary['n_lanes']} lanes, "
        f"speed={modified_summary['mean_speed_mph']:.1f} mph, "
        f"density={modified_summary['mean_density_veh_mi_ln']:.1f} veh/mi/ln, "
        f"LOS={modified_summary['los_distribution']}"
    )

    # Step 6b: Re-encode modified FCD through encoder (embedding shift)
    embedding_shift = None
    fcd_path = sumo_root / camera / "fcd_output.xml"
    if fcd_path.exists():
        try:
            lane_geoms = _parse_net_lanes(modified_net)
            traj_data = extract_encoder_inputs(fcd_path, lane_geoms, polyline_k=16)
            if traj_data:
                reencoder = LaneReencoder(model, device)
                reencoded = reencoder.reencode(traj_data, lane_geoms)

                # Compute mean embedding shift between real and re-encoded
                from src.bridge.variant_comparator import VariantComparator
                real_emb_dict = {
                    dataset.samples[i].lane_key: embeddings_real[i]
                    for i in range(len(dataset))
                    if dataset.samples[i].camera == camera
                }
                comparator = VariantComparator()
                report = comparator.compare_nearest(
                    real_emb_dict, reencoded,
                    variant_name=f"{camera}_modified",
                )
                embedding_shift = {
                    "mean_cosine_similarity": report.mean_cosine_similarity,
                    "mean_l2_distance": report.mean_l2_distance,
                    "n_matched": len(report.lane_results),
                }
                logger.info(
                    f"[{camera}] Embedding shift: cos_sim={report.mean_cosine_similarity:.3f}"
                )
        except Exception as e:
            logger.warning(f"[{camera}] Re-encoding failed: {e}")

    modified_net.unlink(missing_ok=True)

    # Step 7: Compare
    deltas = compute_deltas(baseline_summary, modified_summary)
    t_done = time.time()

    result = {
        "camera": camera,
        "group_id": target_gid,
        "target_edge": target_edge,
        "variant": {
            "name": variant_info["variant_name"] if variant_info else f"add_{lane_type}",
            "lane_type": lane_type,
            "generation_score": variant_info["score"] if variant_info else None,
            "generated_by": "diffusion" if variant_info else "position_only",
        },
        "baseline": baseline_summary,
        "modified": modified_summary,
        "deltas": deltas,
        "embedding_shift": embedding_shift,
        "timing": {
            "encode_s": round(t_encode - t_start, 2),
            "baseline_sim_s": round(t_baseline - t_encode, 2),
            "modify_s": round(t_modify - t_baseline, 2),
            "modified_sim_s": round(t_modified - t_modify, 2),
            "total_s": round(t_done - t_start, 2),
        },
    }

    # Log impact
    ds = deltas["delta_speed_mph"]
    dd = deltas["delta_density_veh_mi_ln"]

    logger.info(
        f"[{camera}] IMPACT: "
        f"speed {'↑' if ds > 0 else '↓'}{abs(ds):.1f} mph, "
        f"density {'↓' if dd < 0 else '↑'}{abs(dd):.1f} veh/mi/ln, "
        f"LOS {deltas['los_before']} → {deltas['los_after']}"
    )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Planning Evaluation: encoder→generator→SUMO→re-encode loop"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--all-cameras", action="store_true")
    parser.add_argument("--sumo-root", default=None)
    parser.add_argument("--output-dir", default="results/planning_eval")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Diffusion model checkpoint for generation pipeline")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip generation, use position-only variant (for testing)")
    parser.add_argument("--sim-duration", type=int, default=3600)
    parser.add_argument("--flow-rate", type=int, default=1200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve SUMO root
    if args.sumo_root:
        sumo_root = Path(args.sumo_root)
    else:
        sumo_root = PROJECT_ROOT / "dataset" / "sumo"
        if not sumo_root.exists():
            sumo_root = PROJECT_ROOT.parent / "graph_geolane" / "dataset" / "sumo"

    if not sumo_root.exists():
        logger.error(f"SUMO root not found: {sumo_root}")
        return

    # Load encoder
    model, config, device = load_encoder(args)

    # Build generator if diffusion checkpoint provided
    generator = None
    diffusion_available = False
    if args.diffusion_checkpoint and not args.no_generate:
        diff_path = Path(args.diffusion_checkpoint)
        if diff_path.exists():
            logger.info(f"Loading diffusion model: {diff_path}")
            full_dataset = build_dataset(config)
            generator, index = build_generator(
                model, full_dataset, device, str(diff_path),
            )
            diffusion_available = True
        else:
            logger.warning(f"Diffusion checkpoint not found: {diff_path}")

    # Determine cameras
    if args.all_cameras:
        full_dataset = build_dataset(config)
        all_encoder_cameras = sorted(set(s.camera for s in full_dataset.samples))
        all_sumo_cameras = sorted([
            d.name for d in sumo_root.iterdir()
            if d.is_dir() and (d / "osm.net.xml").exists()
        ])
        cameras = [c for c in all_encoder_cameras if c in all_sumo_cameras]
        logger.info(f"Found {len(cameras)} cameras: {cameras}")
    elif args.camera:
        cameras = [args.camera]
    else:
        parser.error("Specify --camera or --all-cameras")
        return

    # Run evaluation
    all_results = []
    for camera in cameras:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Camera: {camera}")
        logger.info(f"{'='*60}")

        dataset = build_dataset(config, cameras=[camera])
        if len(dataset) == 0:
            logger.warning(f"[{camera}] No lanes in dataset, skipping")
            continue

        result = evaluate_camera(
            model, dataset, device, camera, sumo_root,
            generator=generator, diffusion_available=diffusion_available,
            sim_duration=args.sim_duration, flow_rate=args.flow_rate,
        )

        if result is not None:
            all_results.append(result)
            cam_path = output_dir / f"{camera}.json"
            with open(cam_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

    # Aggregate summary
    if all_results:
        delta_speeds = [r["deltas"]["delta_speed_mph"] for r in all_results]
        delta_densities = [r["deltas"]["delta_density_veh_mi_ln"] for r in all_results]
        speed_improved = sum(1 for r in all_results if r["deltas"]["speed_improved"])
        density_improved = sum(1 for r in all_results if r["deltas"]["density_improved"])

        # Embedding shift stats
        emb_shifts = [
            r["embedding_shift"]["mean_cosine_similarity"]
            for r in all_results if r.get("embedding_shift")
        ]

        summary = {
            "experiment": "planning_evaluation",
            "pipeline": "encoder→generator→SUMO→re-encode" if diffusion_available
                        else "encoder→SUMO (no generation)",
            "timestamp": time.time(),
            "n_cameras": len(all_results),
            "cameras": [r["camera"] for r in all_results],
            "aggregate": {
                "mean_delta_speed_mph": float(np.mean(delta_speeds)),
                "std_delta_speed_mph": float(np.std(delta_speeds)),
                "mean_delta_density": float(np.mean(delta_densities)),
                "std_delta_density": float(np.std(delta_densities)),
                "speed_improved_count": speed_improved,
                "density_improved_count": density_improved,
                "improvement_rate": density_improved / len(all_results),
            },
            "per_camera": [
                {
                    "camera": r["camera"],
                    "variant": r["variant"]["name"],
                    "gen_score": r["variant"]["generation_score"],
                    "baseline_speed": r["baseline"]["mean_speed_mph"],
                    "modified_speed": r["modified"]["mean_speed_mph"],
                    "delta_speed": r["deltas"]["delta_speed_mph"],
                    "delta_density": r["deltas"]["delta_density_veh_mi_ln"],
                    "los_before": r["baseline"]["los_distribution"],
                    "los_after": r["modified"]["los_distribution"],
                    "embedding_cos_sim": r.get("embedding_shift", {}).get("mean_cosine_similarity"),
                }
                for r in all_results
            ],
        }

        if emb_shifts:
            summary["aggregate"]["mean_embedding_shift"] = float(np.mean(emb_shifts))

        results_path = output_dir / "planning_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        summary_path = output_dir / "planning_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary table
        logger.info(f"\n{'='*80}")
        logger.info("PLANNING EVALUATION — TRAFFIC IMPACT SUMMARY")
        logger.info(f"Pipeline: {summary['pipeline']}")
        logger.info(f"{'='*80}")
        logger.info(
            f"{'Camera':<18} {'Variant':<16} {'GenScore':>9} "
            f"{'ΔSpeed':>8} {'ΔDensity':>10} {'EmbShift':>9} {'Impact':>8}"
        )
        logger.info(f"{'-'*90}")

        for r in all_results:
            d = r["deltas"]
            v = r["variant"]
            emb = r.get("embedding_shift", {}).get("mean_cosine_similarity")
            gs = v["generation_score"]
            impact = "YES" if d["speed_improved"] and d["density_improved"] else \
                     "PARTIAL" if d["speed_improved"] or d["density_improved"] else "NO"

            logger.info(
                f"{r['camera']:<18} {v['name']:<16} "
                f"{f'{gs:.3f}' if gs else 'N/A':>9} "
                f"{d['delta_speed_mph']:>+8.1f} "
                f"{d['delta_density_veh_mi_ln']:>+10.1f} "
                f"{f'{emb:.3f}' if emb else 'N/A':>9} "
                f"{impact:>8}"
            )

        logger.info(f"{'-'*90}")
        logger.info(
            f"{'AGGREGATE':<18} {'':16} {'':>9} "
            f"{np.mean(delta_speeds):>+8.1f} "
            f"{np.mean(delta_densities):>+10.1f} "
            f"{f'{np.mean(emb_shifts):.3f}' if emb_shifts else 'N/A':>9} "
            f"{density_improved}/{len(all_results)}"
        )
        logger.info(f"\nResults saved to {output_dir}/")
    else:
        logger.error("No cameras produced results.")


if __name__ == "__main__":
    main()
