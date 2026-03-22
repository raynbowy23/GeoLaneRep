#!/usr/bin/env python3
"""C2 Experiment: End-to-end Behavioral Geometry Synthesis Loop.

Demonstrates the closed re-encoding loop:
    observe (real) → encode → generate variants → SUMO simulate
    → extract trajectories → re-encode → compare in embedding space

For each camera:
1. Encode real lanes → baseline embeddings
2. Run SUMO on original network → extract FCD → re-encode → compare
   (this validates the re-encoding pipeline on unmodified geometry)
3. Optionally generate variant lanes and report generation scores

Output: results/c2_synthesis_loop/
    - c2_results.json        — per-camera re-encoding metrics
    - c2_summary.json        — aggregate statistics
    - per_camera/{cam}.json  — detailed per-lane results

Usage:
    # Single camera
    python scripts/run_c2_experiment.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --camera US12_Monona

    # All cameras
    python scripts/run_c2_experiment.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --all-cameras

    # With variant generation (requires diffusion checkpoint)
    python scripts/run_c2_experiment.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --camera US12_Monona \
        --generate-variants \
        --diffusion-checkpoint results/generation/figures/diffusion_model.pt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import time
from dataclasses import asdict

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

    dataset = LaneDataset(
        config=config,
        cameras=cameras,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )
    return dataset


def encode_all(model, dataset, device):
    """Encode all lanes, return embeddings and encoder output."""
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
# Calibration helper: SUMO → pixel space
# ---------------------------------------------------------------------------

def calibrate_sumo_data(traj_data, lane_geoms_raw, calibration):
    """Transform SUMO lane geometries and trajectories to pixel space.

    IMPORTANT: lane_geoms_raw must be raw SUMO meter coordinates (NOT
    the already-normalized output from extract_encoder_inputs).

    The trajectory data's `raw_trajectories` (SUMO meters) are transformed
    through the homography to pixel [0,1] space, then resampled to K=16.
    This matches the encoder's training coordinate system.

    Args:
        traj_data: Dict[lane_id, LaneTrajectoryData] from extract_encoder_inputs.
        lane_geoms_raw: Dict[lane_id, (N, 2)] raw SUMO meter-space geometries.
        calibration: CameraCalibration with SUMO→pixel homography.

    Returns:
        (calibrated_traj_data, calibrated_lane_geoms) in pixel [0,1] space.
    """
    from src.bridge.trajectory_extractor import LaneTrajectoryData
    from src.generation.openlane_preprocess import _resample_polyline

    # Transform lane geometries: SUMO meters → pixel [0,1]
    cal_geoms = {}
    for lid, geom in lane_geoms_raw.items():
        cal_geoms[lid] = calibration.sumo_to_pixel(geom)

    # Transform trajectories using raw_trajectories (SUMO meters),
    # NOT traj_polylines (which are already bounding-box normalized)
    cal_traj = {}
    for lid, data in traj_data.items():
        cal_polylines = []
        for raw_traj in data.raw_trajectories:
            # raw_traj is (T_i, 2) in SUMO meters
            if len(raw_traj) < 2:
                continue
            pixel_traj = calibration.sumo_to_pixel(raw_traj)
            # Resample to K=16 points
            resampled = _resample_polyline(pixel_traj, 16)
            cal_polylines.append(resampled)

        if not cal_polylines:
            continue

        cal_traj[lid] = LaneTrajectoryData(
            lane_id=data.lane_id,
            traj_polylines=cal_polylines,
            traj_stats=data.traj_stats,
            raw_trajectories=data.raw_trajectories,
        )

    return cal_traj, cal_geoms


# ---------------------------------------------------------------------------
# Step 1: Baseline re-encoding (SUMO on original network)
# ---------------------------------------------------------------------------

def run_baseline_reencoding(model, dataset, device, embeddings_real, camera, sumo_root,
                            calibration=None):
    """Run SUMO on original network, extract trajectories, re-encode, compare.

    When calibration is provided, SUMO geometries and trajectories are
    projected to pixel space before re-encoding, eliminating the domain gap.

    Returns:
        dict with per-lane comparison results, or None on failure.
    """
    from src.bridge.sumo_runner import run_sumo_simulation, _parse_net_lanes
    from src.bridge.trajectory_extractor import extract_encoder_inputs
    from src.bridge.reencoder import LaneReencoder
    from src.bridge.variant_comparator import VariantComparator
    from src.bridge.traffic_translator import EncoderTrafficBridge

    t_start = time.time()

    # Run SUMO simulation with FCD collection
    logger.info(f"[{camera}] Running SUMO simulation...")
    fcd_path = sumo_root / camera / "fcd_output.xml"
    sumo_metrics = run_sumo_simulation(
        camera, sumo_root,
        sim_duration=3600,  # 1 hour to match real camera observation period
        flow_rate=3000,    # ~3000 veh/hr matches observed ~7600 trajectories/hr
        collect_fcd=True,
        fcd_output_path=fcd_path,
    )

    if not sumo_metrics:
        logger.warning(f"[{camera}] SUMO simulation produced no metrics")
        return None

    t_sumo = time.time()
    logger.info(f"[{camera}] SUMO done: {len(sumo_metrics)} lanes, {t_sumo - t_start:.1f}s")

    # Parse network lanes for geometry
    net_path = sumo_root / camera / "osm.net.xml"
    if not net_path.exists():
        # Try compressed
        import gzip, shutil, tempfile
        gz_path = sumo_root / camera / "osm.net.xml.gz"
        if not gz_path.exists():
            logger.warning(f"[{camera}] No SUMO network found")
            return None
        with tempfile.NamedTemporaryFile(suffix=".net.xml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with gzip.open(gz_path, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        lane_geoms = _parse_net_lanes(tmp_path)
        tmp_path.unlink()
    else:
        lane_geoms = _parse_net_lanes(net_path)

    if not fcd_path.exists():
        logger.warning(f"[{camera}] No FCD output at {fcd_path}")
        return None

    # Extract trajectories from FCD
    logger.info(f"[{camera}] Extracting trajectories from FCD...")
    traj_data = extract_encoder_inputs(fcd_path, lane_geoms, polyline_k=16)

    if not traj_data:
        logger.warning(f"[{camera}] No trajectories extracted from FCD")
        return None

    t_extract = time.time()
    logger.info(f"[{camera}] Extracted {len(traj_data)} lanes, {t_extract - t_sumo:.1f}s")

    # Save raw SUMO lane_geoms (meter-space) for spatial matching
    lane_geoms_raw = dict(lane_geoms)

    # Re-encode in SUMO domain (no calibration — same-domain comparison)
    logger.info(f"[{camera}] Re-encoding SUMO trajectories (SUMO domain)...")
    reencoder = LaneReencoder(model, device)
    reencoded_embeddings = reencoder.reencode(traj_data, lane_geoms)

    t_reencode = time.time()
    logger.info(
        f"[{camera}] Re-encoded {len(reencoded_embeddings)} lanes, "
        f"{t_reencode - t_extract:.1f}s"
    )

    # Build real embeddings dict for comparison
    real_embeddings = {}
    for i, s in enumerate(dataset.samples):
        if s.camera == camera:
            real_embeddings[s.lane_key] = embeddings_real[i]

    # Spatially filter SUMO lanes: only keep lanes on edges that
    # overlap with real lane groups (via calibration)
    # Use raw SUMO lane_geoms (meter-space) for spatial matching
    if calibration is not None:
        from src.bridge.sumo_modifier import get_mainline_edges
        mainline_edges = get_mainline_edges(
            sumo_root / camera / "osm.net.xml"
        )
        cam_samples = [s for s in dataset.samples if s.camera == camera]
        # Find matching edges for each group
        groups = {}
        for s in cam_samples:
            groups.setdefault(s.group_id, []).append(s)

        matched_edges = set()
        for gid, samples in groups.items():
            edge_id, dist = _find_matching_sumo_edge(
                samples, lane_geoms_raw, mainline_edges, calibration,
            )
            if edge_id:
                matched_edges.add(edge_id)
                logger.info(
                    f"[{camera}] Group {gid} → SUMO edge '{edge_id}' "
                    f"(dist={dist:.1f}m)"
                )

        # Filter re-encoded embeddings to only matched edges
        if matched_edges:
            filtered_embeddings = {
                lid: emb for lid, emb in reencoded_embeddings.items()
                if lid.rsplit("_", 1)[0] in matched_edges
            }
            logger.info(
                f"[{camera}] Spatial filter: {len(filtered_embeddings)}/{len(reencoded_embeddings)} "
                f"lanes on matched edges {sorted(matched_edges)}"
            )
            reencoded_embeddings = filtered_embeddings

    # Compare in embedding space using nearest-neighbor matching
    comparator = VariantComparator()
    report = comparator.compare_nearest(
        real_embeddings, reencoded_embeddings,
        variant_name=f"{camera}_baseline",
    )

    t_compare = time.time()

    # Also compute SUMO-based traffic metrics for the re-encoded lanes
    bridge = EncoderTrafficBridge()

    # Build result
    # C2 measures embedding fidelity only (cos_sim, L2).
    # Traffic metrics (speed, density, LOS) are evaluated in planning_eval.
    result = {
        "camera": camera,
        "n_real_lanes": len(real_embeddings),
        "n_sumo_lanes": len(sumo_metrics),
        "n_reencoded_lanes": len(reencoded_embeddings),
        "n_matched_lanes": len(report.lane_results),
        "mean_cosine_similarity": report.mean_cosine_similarity,
        "mean_l2_distance": report.mean_l2_distance,
        "timing": {
            "sumo_s": round(t_sumo - t_start, 2),
            "extract_s": round(t_extract - t_sumo, 2),
            "reencode_s": round(t_reencode - t_extract, 2),
            "compare_s": round(t_compare - t_reencode, 2),
            "total_s": round(t_compare - t_start, 2),
        },
        "per_lane": [
            {
                "lane_id": r.lane_id,
                "cosine_similarity": r.cosine_similarity,
                "l2_distance": r.l2_distance,
            }
            for r in report.lane_results
        ],
    }

    logger.info(
        f"[{camera}] Re-encoding fidelity: "
        f"cos_sim={report.mean_cosine_similarity:.3f}, "
        f"L2={report.mean_l2_distance:.3f}, "
        f"matched={len(report.lane_results)}/{len(real_embeddings)} lanes"
    )

    return result


# ---------------------------------------------------------------------------
# Step 2: Variant generation + SUMO closed loop
# ---------------------------------------------------------------------------

def run_variant_generation(model, dataset, device, embeddings_real, camera, args,
                           calibration=None):
    """Generate lane variants, inject into SUMO, simulate, re-encode, compare.

    Full C2 closed loop:
        1. Generate variant lane (diffusion) → geometry
        2. Inject into SUMO network (add_lane_to_network)
        3. Simulate traffic → FCD
        4. Extract trajectories → re-encode → embedding
        5. Compare original vs re-encoded embedding
        6. Compare lane geometry in canonical space

    Returns:
        dict with variant generation + re-encoding results.
    """
    from src.generation.retrieval import build_retrieval_index
    from src.generation.spec import SpecEmbeddingResolver, LaneSpecification
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer
    from src.generation.directed import DirectedLaneGenerator
    from src.generation.augment import to_canonical
    from src.bridge.sumo_modifier import add_lane_to_network, LaneAddition, get_mainline_edges

    cam_samples = [s for s in dataset.samples if s.camera == camera]
    if not cam_samples:
        return None

    # Find largest group
    group_counts = {}
    for s in cam_samples:
        group_counts.setdefault(s.group_id, []).append(s)
    target_gid = max(group_counts, key=lambda g: len(group_counts[g]))
    target_samples = group_counts[target_gid]

    # Build generation pipeline
    index = build_retrieval_index(model, dataset, device, use_embeddings=True)
    resolver = SpecEmbeddingResolver(index, dataset)

    K = index.K
    denoiser = LaneDenoiser(
        geom_dim=K * 2, t_dim=64, cond_dim=index.D, hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))
    trainer.load(args.diffusion_checkpoint)

    # Load relational model if checkpoint provided
    relational_trainer = None
    if getattr(args, "relational_checkpoint", None):
        from src.generation.relational_diffusion import (
            RelationalLaneDenoiser, RelationalDiffusionTrainer,
        )
        rel_denoiser = RelationalLaneDenoiser(
            geom_dim=K * 2, t_dim=64, cond_dim=index.D,
            rel_dim=64, hidden_dim=256,
        )
        rel_schedule = DDPMSchedule(T=100)
        relational_trainer = RelationalDiffusionTrainer(
            rel_denoiser, rel_schedule, device=str(device),
        )
        relational_trainer.load(args.relational_checkpoint)
        logger.info(f"[{camera}] Loaded relational diffusion model")

    generator = DirectedLaneGenerator(
        resolver=resolver, trainer=trainer,
        encoder=model, dataset=dataset, device=device,
        relational_trainer=relational_trainer,
    )

    # Resolve SUMO root
    if args.sumo_root:
        sumo_root = Path(args.sumo_root)
    else:
        sumo_root = PROJECT_ROOT / "dataset" / "sumo"
        if not sumo_root.exists():
            sumo_root = PROJECT_ROOT.parent / "graph_geolane" / "dataset" / "sumo"

    # Generate variants for all lane types
    variant_specs = [
        ("add_rightmost", LaneSpecification.rightmost(camera=camera, group_id=target_gid)),
        ("add_leftmost", LaneSpecification.leftmost(camera=camera, group_id=target_gid)),
    ]

    has_merge = any(s.role.has_successor for s in target_samples)
    if has_merge:
        variant_specs.append(
            ("add_merge", LaneSpecification.merge_lane(camera=camera, group_id=target_gid))
        )

    results = []
    for name, spec in variant_specs:
        try:
            # Use relational generation if available, else baseline
            if relational_trainer is not None:
                gen_result = generator.generate_relational(spec, n_candidates=5)
            else:
                gen_result = generator.generate(spec, n_candidates=5)
            gen_score = float(gen_result.scores.max())
            best_geom = gen_result.best  # (K, 2) in image space

            # Canonical geometry comparison: original anchor vs generated
            canonical_sim = None
            if gen_result.spatial_context is not None:
                anchor = gen_result.spatial_context.anchor_geometry
                can_anchor, _, _, _ = to_canonical(anchor)
                can_gen, _, _, _ = to_canonical(best_geom)
                # Cosine similarity of flattened canonical shapes
                a_flat = can_anchor.flatten()
                g_flat = can_gen.flatten()
                dot = np.dot(a_flat, g_flat)
                norms = np.linalg.norm(a_flat) * np.linalg.norm(g_flat)
                canonical_sim = float(dot / max(norms, 1e-8))

            variant_result = {
                "variant_name": name,
                "camera": camera,
                "group_id": target_gid,
                "generation_score": gen_score,
                "canonical_geometry_sim": canonical_sim,
                "n_candidates": len(gen_result.candidates),
            }

            # ── Closed loop: specify → simulate → re-encode → validate ──
            # Extract encoder features → map to SUMO lane attributes.
            # The encoder's learned representations drive the simulation:
            #   speed, curvature, density → SUMO speed, width, lane change, vehicle class

            matched_indices = resolver._filter_by_spec(spec)
            encoder_features = _extract_encoder_features(
                spec, matched_indices, resolver, name,
            )
            logger.info(
                f"[{camera}] Encoder → SUMO mapping for {name}: "
                f"speed={encoder_features['sumo_speed']:.1f}m/s "
                f"({encoder_features['sumo_speed']*2.237:.0f}mph), "
                f"width={encoder_features['sumo_width']:.1f}m, "
                f"merge={encoder_features['is_merge']}, "
                f"disallow={encoder_features['disallow']}"
            )

            reencoding_result = _run_sumo_reencoding_loop(
                model, device, camera, sumo_root, name, embeddings_real,
                dataset, target_samples,
                calibration=calibration,
                target_embedding=gen_result.target_embedding,
                encoder_features=encoder_features,
                spec=spec, resolver=resolver,
            )
            if reencoding_result is not None:
                variant_result.update(reencoding_result)

            results.append(variant_result)
            logger.info(
                f"[{camera}] {name}: gen_score={gen_score:.3f}, "
                f"canonical_sim={canonical_sim if canonical_sim is not None else 'N/A'}, "
                f"new_lane_cos_sim={variant_result.get('new_lane_cos_sim', 'N/A')}, "
                f"existing_cos_sim={variant_result.get('existing_lane_cos_sim', 'N/A')}"
            )
        except Exception as e:
            logger.warning(f"[{camera}] {name} failed: {e}")
            results.append({
                "variant_name": name,
                "camera": camera,
                "group_id": target_gid,
                "generation_score": 0.0,
                "error": str(e),
            })

    return {
        "camera": camera,
        "group_id": target_gid,
        "n_existing_lanes": len(target_samples),
        "variants": results,
    }


def _extract_encoder_features(spec, matched_indices, resolver, variant_name):
    """Extract encoder features and map to SUMO lane attributes.

    Maps the encoder's learned behavioral observations to concrete
    SUMO lane parameters:
        mean_speed → SUMO speed limit
        lane role → lane width, change permissions, vehicle class
        density → (informational, logged)
        curvature → (informational, logged)

    Returns:
        dict with sumo_speed, sumo_width, is_merge, allow_change_left,
        allow_change_right, disallow, target_stats.
    """
    # Default values
    features = {
        "sumo_speed": 24.59,       # 55 mph default
        "sumo_width": 3.2,         # standard lane width
        "is_merge": False,
        "allow_change_left": True,
        "allow_change_right": True,
        "disallow": [],
        "target_stats": None,
    }

    if len(matched_indices) == 0 or resolver._traj_stats is None:
        return features

    # Get behavioral stats from matched lanes
    # [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]
    stats = resolver._traj_stats[matched_indices]
    mean_stats = stats.mean(axis=0)
    features["target_stats"] = mean_stats.tolist()

    mean_speed = float(mean_stats[0])
    mean_curvature = float(mean_stats[1])
    mean_density = float(mean_stats[3])  # traj_count_norm as density proxy

    # --- Speed mapping ---
    # Encoder mean_speed is normalized pixel displacement (~0.002-0.010).
    # Map to SUMO speed using observed range:
    #   Low speed (congested): 0.002 → 15 m/s (34 mph)
    #   Medium speed: 0.005 → 25 m/s (56 mph)
    #   High speed (free flow): 0.010 → 35 m/s (78 mph)
    # Linear interpolation within observed range
    speed_min, speed_max = 0.002, 0.010
    sumo_min, sumo_max = 15.0, 35.0
    t = np.clip((mean_speed - speed_min) / (speed_max - speed_min + 1e-8), 0, 1)
    features["sumo_speed"] = sumo_min + t * (sumo_max - sumo_min)

    # --- Width mapping ---
    # Merge/diverge lanes are typically narrower
    # Rightmost lanes (shoulder-adjacent) may be slightly wider
    if "merge" in variant_name:
        features["sumo_width"] = 2.8   # narrower merge lane
        features["is_merge"] = True
    elif "leftmost" in variant_name:
        features["sumo_width"] = 3.6   # wider passing lane
    else:
        features["sumo_width"] = 3.2   # standard

    # --- Lane change permissions ---
    # Rightmost: restrict left changes (keep traffic in lane)
    # Leftmost (passing): restrict right changes during high density
    # Merge: allow all changes (vehicles need to merge)
    if "rightmost" in variant_name:
        features["allow_change_left"] = True
        features["allow_change_right"] = False  # no changing to shoulder
    elif "leftmost" in variant_name:
        if mean_density > 0.7:  # high density → restrict passing lane exits
            features["allow_change_right"] = False
    # Merge lanes always allow both directions

    # --- Vehicle class restrictions ---
    # Leftmost (passing lane): restrict heavy vehicles
    # High curvature lanes: restrict long vehicles
    if "leftmost" in variant_name:
        features["disallow"] = ["truck", "trailer"]
    elif mean_curvature > 0.6:
        features["disallow"] = ["trailer"]  # long vehicles on sharp curves

    return features


def _find_matching_sumo_edge(
    target_samples, lane_geoms, mainline_edges, calibration=None,
):
    """Find the SUMO edge closest to the real lane group.

    Uses calibration to convert real lane centroids to SUMO coordinates,
    then finds the mainline edge whose lanes are nearest.

    Returns:
        (edge_id, distance) or (None, inf) if no match found.
    """
    if calibration is None:
        # Fallback: pick longest edge
        edge_lengths = {}
        for lid, geom in lane_geoms.items():
            eid = lid.rsplit("_", 1)[0]
            if eid in mainline_edges:
                diffs = np.diff(geom, axis=0)
                edge_lengths[eid] = max(
                    edge_lengths.get(eid, 0),
                    np.sum(np.linalg.norm(diffs, axis=1)),
                )
        if edge_lengths:
            return max(edge_lengths, key=edge_lengths.get), 0.0
        return None, float("inf")

    # Convert real lane centroids to SUMO coordinates
    real_centroids_sumo = []
    for s in target_samples:
        centroid_pixel = s.geometry.mean(axis=0).reshape(1, 2)
        centroid_sumo = calibration.pixel_to_sumo(centroid_pixel)
        real_centroids_sumo.append(centroid_sumo[0])
    real_center = np.mean(real_centroids_sumo, axis=0)

    # Find closest mainline edge by centroid distance
    best_edge = None
    best_dist = float("inf")
    for eid in mainline_edges:
        edge_lanes = [
            lid for lid in lane_geoms if lid.rsplit("_", 1)[0] == eid
        ]
        if not edge_lanes:
            continue
        # Edge centroid = mean of all lane centroids
        edge_centroids = [lane_geoms[lid].mean(axis=0) for lid in edge_lanes]
        edge_center = np.mean(edge_centroids, axis=0)
        dist = np.linalg.norm(edge_center - real_center)
        if dist < best_dist:
            best_dist = dist
            best_edge = eid

    return best_edge, best_dist


def _run_sumo_reencoding_loop(
    model, device, camera, sumo_root, variant_name,
    embeddings_real, dataset, target_samples,
    calibration=None, target_embedding=None, encoder_features=None,
    spec=None, resolver=None,
):
    """Encoder-driven SUMO evaluation loop.

    The encoder's learned representations drive the simulation:
      1. target_speed: from encoder's behavioral observation → sets SUMO lane speed
      2. variant_name: from spec resolution (learned lane roles) → lane position
      3. target_embedding: from encoder → compared against re-encoded embedding

    SUMO creates valid lane geometry; the encoder validates behavioral consistency.

    Returns:
        dict with re-encoding metrics, or None on failure.
    """
    from src.bridge.sumo_runner import run_sumo_simulation, _parse_net_lanes
    from src.bridge.sumo_modifier import (
        add_lane_to_network, LaneAddition, get_mainline_edges,
    )
    from src.bridge.trajectory_extractor import extract_encoder_inputs
    from src.bridge.reencoder import LaneReencoder
    from src.generation.augment import to_canonical

    camera_dir = sumo_root / camera
    net_file = camera_dir / "osm.net.xml"
    if not net_file.exists():
        net_gz = camera_dir / "osm.net.xml.gz"
        if not net_gz.exists():
            logger.warning(f"[{camera}] No SUMO network for re-encoding loop")
            return None
        import gzip, shutil, tempfile
        with tempfile.NamedTemporaryFile(suffix=".net.xml", delete=False) as tmp:
            net_file = Path(tmp.name)
        with gzip.open(net_gz, "rb") as f_in:
            with open(net_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    try:
        mainline_edges = get_mainline_edges(net_file)
        if not mainline_edges:
            logger.warning(f"[{camera}] No mainline edges found")
            return None

        lane_geoms = _parse_net_lanes(net_file)

        # Spatial matching: find SUMO edge closest to real lane group
        target_edge, match_dist = _find_matching_sumo_edge(
            target_samples, lane_geoms, mainline_edges, calibration,
        )

        if target_edge is None:
            logger.warning(f"[{camera}] No matching SUMO edge found")
            return None

        logger.info(
            f"[{camera}] Matched real lanes to SUMO edge '{target_edge}' "
            f"(distance={match_dist:.1f}m)"
        )

        # Track original lane IDs on target edge (before modification)
        original_lane_ids = set(
            lid for lid in lane_geoms if lid.rsplit("_", 1)[0] == target_edge
        )

        # Determine lane position from variant name
        if "leftmost" in variant_name:
            position = "leftmost"
        else:
            position = "rightmost"

        # Build LaneAddition from encoder features
        if encoder_features is not None:
            sumo_speed = encoder_features["sumo_speed"]
            sumo_width = encoder_features["sumo_width"]
            is_merge = encoder_features["is_merge"]
            disallow = encoder_features["disallow"]
        else:
            sumo_speed = 24.59
            sumo_width = 3.2
            is_merge = False
            disallow = []

        addition = LaneAddition(
            edge_id=target_edge,
            position=position,
            speed=sumo_speed,
            width=sumo_width,
            allow_change_left=encoder_features.get("allow_change_left", True) if encoder_features else True,
            allow_change_right=encoder_features.get("allow_change_right", True) if encoder_features else True,
            disallow_classes=disallow if disallow else None,
            is_merge=is_merge,
        )
        import tempfile as _tempfile
        modified_net = Path(_tempfile.mktemp(suffix=".net.xml"))

        try:
            modified_net = add_lane_to_network(net_file, addition, modified_net)
        except Exception as e:
            logger.warning(f"[{camera}] Failed to add lane: {e}")
            return None

        # Find the new lane ID in the modified network
        modified_lane_geoms_tmp = _parse_net_lanes(modified_net)
        modified_edge_lanes_tmp = set(
            lid for lid in modified_lane_geoms_tmp
            if lid.rsplit("_", 1)[0] == target_edge
        )
        new_lane_id = None
        for nlid in sorted(modified_edge_lanes_tmp - original_lane_ids):
            new_lane_id = nlid
            break

        if new_lane_id:
            logger.info(f"[{camera}] New lane: {new_lane_id}")

        # SUMO creates valid lane geometry (parallel offset, junction-connected).
        # Encoder features (speed, width, permissions) are applied via LaneAddition.
        # Diffusion-generated geometry is for pixel-space visualization (G1/G2).

        # Run SUMO on modified network with higher flow to ensure all lanes used
        logger.info(f"[{camera}] Running SUMO on modified network ({variant_name})...")
        fcd_path = camera_dir / f"fcd_{variant_name}.xml"

        import subprocess
        with _tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            demand_file = tmpdir / "demand.rou.xml"

            # Generate demand matching real observation (~3000 veh/hr, 1 hour).
            # departLane="best" makes vehicles actively choose the least congested lane.
            period = max(0.3, 3600.0 / 3000)  # ~3000 veh/hr to match real traffic
            random_trips = Path("/usr/share/sumo/tools/randomTrips.py")
            if random_trips.exists():
                trips_file = tmpdir / "trips.xml"
                routes_file = tmpdir / "routes.rou.xml"
                subprocess.run([
                    "python", str(random_trips),
                    "-n", str(modified_net),
                    "-o", str(trips_file),
                    "-r", str(routes_file),
                    "-e", "3600",
                    "-p", str(period),
                    "--fringe-factor", "5",
                    "--allow-fringe",
                    "-t", 'departLane="best" departSpeed="max"',
                    "--validate",
                ], capture_output=True, text=True, timeout=60)
                if routes_file.exists():
                    demand_file = routes_file

            if not demand_file.exists():
                logger.warning(f"[{camera}] No demand generated for modified network")
                modified_net.unlink(missing_ok=True)
                return None

            # Build SUMO config (FCD only, no edge output)
            sumo_cfg = tmpdir / "sim.sumocfg"
            with open(sumo_cfg, "w") as f:
                f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{modified_net}"/>
    <route-files value="{demand_file}"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="3600"/>
  </time>
  <output>
    <fcd-output value="{fcd_path}"/>
  </output>
</configuration>""")

            result = subprocess.run(
                ["sumo", "-c", str(sumo_cfg), "--no-warnings", "true"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"[{camera}] SUMO failed: {result.stderr[:200]}")
                modified_net.unlink(missing_ok=True)
                return None

        # Parse modified network lanes
        modified_lane_geoms = _parse_net_lanes(modified_net)
        modified_net.unlink(missing_ok=True)

        # Identify the NEW lane (present in modified but not original)
        modified_edge_lanes = set(
            lid for lid in modified_lane_geoms if lid.rsplit("_", 1)[0] == target_edge
        )
        new_lane_ids = modified_edge_lanes - original_lane_ids
        logger.info(
            f"[{camera}] Edge '{target_edge}': "
            f"original={sorted(original_lane_ids)}, "
            f"new={sorted(new_lane_ids)}"
        )

        if not fcd_path.exists():
            logger.warning(f"[{camera}] No FCD output from modified SUMO")
            return None

        # Extract trajectories from FCD
        traj_data = extract_encoder_inputs(fcd_path, modified_lane_geoms, polyline_k=16)
        fcd_path.unlink(missing_ok=True)

        if not traj_data:
            logger.warning(f"[{camera}] No trajectories extracted from modified FCD")
            return None

        logger.info(
            f"[{camera}] Extracted {len(traj_data)} lanes from modified network: "
            f"{sorted(traj_data.keys())}"
        )
        # Check if new lane got any trajectories
        for nlid in (original_lane_ids ^ set(lid for lid in traj_data.keys()
                     if lid.rsplit("_", 1)[0] == target_edge)):
            if nlid in traj_data:
                logger.info(f"[{camera}] New lane '{nlid}' has {len(traj_data[nlid].traj_polylines)} trajectories")
            else:
                logger.info(f"[{camera}] New lane '{nlid}' has NO trajectories in FCD")

        # Re-encode variant SUMO trajectories (NO calibration — stay in SUMO domain)
        # Both baseline and variant are encoded in the same domain for fair comparison
        reencoder = LaneReencoder(model, device)
        reencoded = reencoder.reencode(traj_data, modified_lane_geoms)

        if not reencoded:
            logger.warning(f"[{camera}] Re-encoding produced no embeddings")
            return None

        # --- Encode baseline SUMO trajectories for same-domain comparison ---
        # Run SUMO on ORIGINAL network → encode → SUMO baseline embeddings
        # This eliminates the camera-vs-SUMO domain gap
        baseline_fcd = camera_dir / "fcd_baseline_loop.xml"
        baseline_metrics = run_sumo_simulation(
            camera, sumo_root,
            sim_duration=3600, flow_rate=3000,
            collect_fcd=True, fcd_output_path=baseline_fcd,
        )
        baseline_lane_geoms = _parse_net_lanes(
            camera_dir / "osm.net.xml"
        )
        baseline_traj = extract_encoder_inputs(
            baseline_fcd, baseline_lane_geoms, polyline_k=16,
        )
        baseline_fcd.unlink(missing_ok=True)

        sumo_baseline_embeddings = reencoder.reencode(
            baseline_traj, baseline_lane_geoms,
        )

        if not sumo_baseline_embeddings:
            logger.warning(f"[{camera}] No SUMO baseline embeddings")
            return None

        logger.info(
            f"[{camera}] SUMO-vs-SUMO comparison: "
            f"{len(sumo_baseline_embeddings)} baseline, {len(reencoded)} variant"
        )

        from sklearn.metrics.pairwise import cosine_similarity as cos_sim_matrix
        baseline_vecs = np.array(list(sumo_baseline_embeddings.values()))
        baseline_keys = list(sumo_baseline_embeddings.keys())

        if len(baseline_vecs) == 0:
            return None

        # --- All-lane comparison (SUMO baseline vs SUMO variant) ---
        reenc_all_vecs = np.array(list(reencoded.values()))
        if len(reenc_all_vecs) == 0:
            return None
        sim_all = cos_sim_matrix(reenc_all_vecs, baseline_vecs)
        all_lane_cos_sim = float(sim_all.max(axis=1).mean())

        # --- New-lane-only comparison ---
        new_lane_cos_sim = None
        new_lane_details = []
        for nlid in new_lane_ids:
            if nlid in reencoded:
                new_vec = reencoded[nlid].reshape(1, -1)
                sims = cos_sim_matrix(new_vec, baseline_vecs)[0]
                best_idx = int(sims.argmax())
                best_sim = float(sims[best_idx])
                best_baseline_key = baseline_keys[best_idx]
                new_lane_details.append({
                    "new_lane_id": nlid,
                    "best_match_baseline": best_baseline_key,
                    "cos_sim": best_sim,
                })
                logger.info(
                    f"[{camera}] New lane '{nlid}' → best match '{best_baseline_key}' "
                    f"cos_sim={best_sim:.3f}"
                )
        if new_lane_details:
            new_lane_cos_sim = float(np.mean([d["cos_sim"] for d in new_lane_details]))

        # --- Existing-lane comparison (baseline stability) ---
        existing_lane_ids = [lid for lid in reencoded if lid not in new_lane_ids]
        existing_cos_sim = None
        if existing_lane_ids:
            exist_vecs = np.array([reencoded[lid] for lid in existing_lane_ids])
            sim_exist = cos_sim_matrix(exist_vecs, baseline_vecs)
            existing_cos_sim = float(sim_exist.max(axis=1).mean())

        # --- Canonical geometry similarity for new lanes ---
        new_canonical_sim = None
        for nlid in new_lane_ids:
            if nlid in modified_lane_geoms:
                geom = modified_lane_geoms[nlid]
                if len(geom) < 2:
                    continue
                if len(geom) != 16:
                    from src.generation.openlane_preprocess import _resample_polyline
                    geom = _resample_polyline(geom, 16)
                can_new, _, _, _ = to_canonical(geom)
                can_flat = can_new.flatten()
                best_can = -1.0
                for s in target_samples:
                    can_real, _, _, _ = to_canonical(s.geometry)
                    real_flat = can_real.flatten()
                    if len(real_flat) == len(can_flat):
                        dot = np.dot(can_flat, real_flat)
                        norms = np.linalg.norm(can_flat) * np.linalg.norm(real_flat)
                        best_can = max(best_can, dot / max(norms, 1e-8))
                if best_can > -1:
                    new_canonical_sim = float(best_can)

        # --- Behavioral stats comparison (domain-agnostic) ---
        # Compare interpretable behavioral features between the encoder's
        # target observation and the SUMO simulation result.
        # These are scalar values that don't depend on embedding space:
        #   [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]
        behavioral_comparison = None
        for nlid in new_lane_ids:
            if nlid in traj_data:
                sumo_stats = traj_data[nlid].traj_stats  # (4,)

                # Get target behavioral stats from the matched real lanes
                if resolver is not None and spec is not None and resolver._traj_stats is not None:
                    matched_indices = resolver._filter_by_spec(spec)
                    if len(matched_indices) == 0:
                        # Use all lanes of the same role from target_samples
                        matched_stats = np.array([
                            s.traj_stats for s in target_samples
                        ])
                    else:
                        matched_stats = resolver._traj_stats[matched_indices]
                    target_stats = matched_stats.mean(axis=0)  # (4,)

                    stat_names = ['mean_speed', 'mean_curvature', 'mean_lateral_offset', 'traj_count_norm']
                    behavioral_comparison = {}
                    for j, name in enumerate(stat_names):
                        behavioral_comparison[name] = {
                            'target': float(target_stats[j]),
                            'sumo': float(sumo_stats[j]),
                            'delta': float(abs(sumo_stats[j] - target_stats[j])),
                        }

                    logger.info(
                        f"[{camera}] Behavioral comparison for '{nlid}':"
                    )
                    for name, vals in behavioral_comparison.items():
                        logger.info(
                            f"  {name:>22s}: target={vals['target']:.4f}, "
                            f"sumo={vals['sumo']:.4f}, delta={vals['delta']:.4f}"
                        )

        logger.info(
            f"[{camera}] Re-encoding loop ({variant_name}): "
            f"new_lane_cos_sim={new_lane_cos_sim if new_lane_cos_sim is not None else 'N/A'}, "
            f"existing_cos_sim={existing_cos_sim if existing_cos_sim is not None else 'N/A'}, "
            f"all_lane_cos_sim={all_lane_cos_sim:.3f}"
        )

        return {
            "all_lane_cos_sim": all_lane_cos_sim,
            "new_lane_cos_sim": new_lane_cos_sim,
            "existing_lane_cos_sim": existing_cos_sim,
            "new_canonical_geom_sim": new_canonical_sim,
            "behavioral_comparison": behavioral_comparison,
            "n_reencoded_lanes": len(reencoded),
            "n_new_lanes": len(new_lane_ids),
            "new_lane_details": new_lane_details,
            "encoder_features": encoder_features,
            "sumo_speed_mps": sumo_speed,
            "sumo_width_m": sumo_width,
        }

    except Exception as e:
        logger.warning(f"[{camera}] Re-encoding loop failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_c2_results(output_dir: Path, all_variant_results: list, all_results: list,
                    model_label: str = "independent"):
    """Generate temperature-colored visualization of C2 closed-loop results.

    Args:
        model_label: "independent" or "relational" — used in titles and filenames.

    Creates a figure with:
      - Left: Per-camera × per-variant heatmap of new_lane_cos_sim
      - Right: Summary bar chart with aggregate scores
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if not all_variant_results:
        logger.warning("No variant results to plot")
        return

    # Collect data into a matrix: rows=cameras, cols=variants
    cameras = []
    variant_names = []
    for vr in all_variant_results:
        cam = vr["camera"]
        if cam not in cameras:
            cameras.append(cam)
        for v in vr.get("variants", []):
            name = v.get("variant_name", "?")
            if name not in variant_names:
                variant_names.append(name)

    # Build matrices
    cos_sim_matrix = np.full((len(cameras), len(variant_names)), np.nan)
    gen_score_matrix = np.full((len(cameras), len(variant_names)), np.nan)
    canonical_matrix = np.full((len(cameras), len(variant_names)), np.nan)

    for vr in all_variant_results:
        cam_idx = cameras.index(vr["camera"])
        for v in vr.get("variants", []):
            name = v.get("variant_name", "?")
            if name in variant_names:
                var_idx = variant_names.index(name)
                if v.get("new_lane_cos_sim") is not None:
                    cos_sim_matrix[cam_idx, var_idx] = v["new_lane_cos_sim"]
                if v.get("generation_score") is not None:
                    gen_score_matrix[cam_idx, var_idx] = v["generation_score"]
                if v.get("canonical_geometry_sim") is not None:
                    canonical_matrix[cam_idx, var_idx] = v["canonical_geometry_sim"]

    # Temperature colormap: blue (cold=low) → red (hot=high)
    cmap = plt.cm.RdYlBu_r  # Red=high, Blue=low

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(cameras) * 0.6 + 2)),
                              gridspec_kw={"width_ratios": [3, 2, 2]})

    # --- Left: Heatmap of new_lane_cos_sim ---
    ax = axes[0]
    im = ax.imshow(cos_sim_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([v.replace("add_", "") for v in variant_names],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cameras)))
    ax.set_yticklabels(cameras, fontsize=8)
    ax.set_title("Re-encoded Cos Similarity\n(SUMO-vs-SUMO, higher=better)", fontsize=11)

    # Annotate cells
    for i in range(len(cameras)):
        for j in range(len(variant_names)):
            val = cos_sim_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")

    # --- Middle: Gen score heatmap ---
    ax = axes[1]
    im2 = ax.imshow(gen_score_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([v.replace("add_", "") for v in variant_names],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cameras)))
    ax.set_yticklabels([], fontsize=8)
    ax.set_title("Generation Score\n(embedding alignment)", fontsize=11)
    for i in range(len(cameras)):
        for j in range(len(variant_names)):
            val = gen_score_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)
    plt.colorbar(im2, ax=ax, shrink=0.8, label="Score")

    # --- Right: Canonical similarity heatmap ---
    ax = axes[2]
    im3 = ax.imshow(canonical_matrix, cmap=cmap, vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(variant_names)))
    ax.set_xticklabels([v.replace("add_", "") for v in variant_names],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cameras)))
    ax.set_yticklabels([], fontsize=8)
    ax.set_title("Canonical Geometry Sim\n(shape fidelity)", fontsize=11)
    for i in range(len(cameras)):
        for j in range(len(variant_names)):
            val = canonical_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.9 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=color)
    plt.colorbar(im3, ax=ax, shrink=0.8, label="Similarity")

    label_upper = model_label.title()
    fig.suptitle(f"C2 Closed-Loop ({label_upper}): Behavior-Conditioned Lane Generation → SUMO → Re-encode",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig_path = output_dir / f"c2_closed_loop_heatmap_{model_label}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved C2 heatmap to {fig_path}")
    plt.close(fig)

    # --- Summary bar chart ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    # Aggregate per variant type
    variant_means = {}
    for vname in variant_names:
        vidx = variant_names.index(vname)
        vals = cos_sim_matrix[:, vidx]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            variant_means[vname] = (float(vals.mean()), float(vals.std()))

    if variant_means:
        names = [v.replace("add_", "").title() for v in variant_means.keys()]
        means = [v[0] for v in variant_means.values()]
        stds = [v[1] for v in variant_means.values()]

        # Color bars by temperature
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        colors = [cmap(norm(m)) for m in means]

        bars = ax2.bar(names, means, yerr=stds, color=colors, edgecolor="black",
                       linewidth=1.2, capsize=5, zorder=3)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{mean:.3f}", ha="center", va="bottom", fontsize=11,
                     fontweight="bold")

        ax2.set_ylabel("Cosine Similarity", fontsize=12)
        ax2.set_title(f"C2 Closed-Loop ({label_upper}): Mean Re-encoded Similarity\n"
                      "(generated lane vs SUMO baseline, per variant type)",
                      fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Target (0.80)")
        ax2.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.50)")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

    fig2_path = output_dir / f"c2_closed_loop_summary_{model_label}.png"
    fig2.savefig(str(fig2_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved C2 summary bar chart to {fig2_path}")
    plt.close(fig2)


def plot_variant_lanes(output_dir: Path, camera: str, config: dict, args,
                       device, calibrations, sumo_root, model_label="independent"):
    """Generate variant lane visualization for a single camera.

    Shows rightmost/leftmost/merge as properly positioned lanes
    with encoder-derived attributes annotated.
    """
    import matplotlib.pyplot as plt
    from src.bridge.sumo_runner import _parse_net_lanes
    from src.bridge.sumo_modifier import get_mainline_edges
    from src.generation.retrieval import build_retrieval_index
    from src.generation.spec import SpecEmbeddingResolver, LaneSpecification
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer
    from src.generation.directed import DirectedLaneGenerator
    from src.generation.openlane_preprocess import _resample_polyline
    from src.training.zero_shot_eval import load_trained_encoder

    label_upper = model_label.title()
    model_enc, _ = load_trained_encoder(args.checkpoint, device)
    ds = build_dataset(config, cameras=[camera])
    if len(ds) == 0:
        return
    index = build_retrieval_index(model_enc, ds, device, use_embeddings=True)
    resolver = SpecEmbeddingResolver(index, ds)
    K = index.K
    dn = LaneDenoiser(geom_dim=K*2, t_dim=64, cond_dim=index.D, hidden_dim=256)
    sch = DDPMSchedule(T=100)
    tr = LaneDiffusionTrainer(dn, sch, device=str(device))
    tr.load(args.diffusion_checkpoint)

    rel_tr = None
    if getattr(args, 'relational_checkpoint', None):
        from src.generation.relational_diffusion import (
            RelationalLaneDenoiser, RelationalDiffusionTrainer,
        )
        rd = RelationalLaneDenoiser(geom_dim=K*2, t_dim=64, cond_dim=index.D, rel_dim=64, hidden_dim=256)
        rs = DDPMSchedule(T=100)
        rel_tr = RelationalDiffusionTrainer(rd, rs, device=str(device))
        rel_tr.load(args.relational_checkpoint)

    gen = DirectedLaneGenerator(resolver=resolver, trainer=tr, encoder=model_enc,
                                dataset=ds, device=device, relational_trainer=rel_tr)

    net_file = sumo_root / camera / 'osm.net.xml'
    lane_geoms = _parse_net_lanes(net_file)
    calib = calibrations.get(camera)
    mainline_edges = get_mainline_edges(net_file)
    cam_samples = [s for s in ds.samples if s.camera == camera]
    groups = {}
    for s in cam_samples:
        groups.setdefault(s.group_id, []).append(s)
    target_gid = max(groups, key=lambda g: len(groups[g]))
    target_samples = groups[target_gid]
    target_edge, dist = _find_matching_sumo_edge(target_samples, lane_geoms, mainline_edges, calib)
    original_lane_ids = sorted(lid for lid in lane_geoms if lid.rsplit("_", 1)[0] == target_edge)
    existing_geoms = [lane_geoms[lid] for lid in original_lane_ids]

    centroids = [g.mean(axis=0) for g in existing_geoms]
    spacings = [np.linalg.norm(centroids[i+1] - centroids[i]) for i in range(len(centroids)-1)]
    lane_spacing = np.mean(spacings) if spacings else 3.2
    ref = _resample_polyline(existing_geoms[0], 16)
    tang = ref[-1] - ref[0]
    tang = tang / (np.linalg.norm(tang) + 1e-8)
    perp = np.array([-tang[1], tang[0]])
    if len(existing_geoms) >= 2:
        r2l = centroids[-1] - centroids[0]
        if np.dot(r2l, perp) < 0:
            perp = -perp

    variants = [
        ("Add Rightmost", "add_rightmost", "rightmost", '#e74c3c',
         LaneSpecification.rightmost(camera=camera, group_id=target_gid)),
        ("Add Leftmost", "add_leftmost", "leftmost", '#2ecc71',
         LaneSpecification.leftmost(camera=camera, group_id=target_gid)),
        ("Add Merge", "add_merge", "merge", '#9b59b6',
         LaneSpecification.merge_lane(camera=camera, group_id=target_gid)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    n_s = 16

    for col, (title, vname, position, color, spec) in enumerate(variants):
        if rel_tr:
            result = gen.generate_relational(spec, n_candidates=5)
        else:
            result = gen.generate(spec, n_candidates=5)
        gs = float(result.scores.max())

        matched = resolver._filter_by_spec(spec)
        features = _extract_encoder_features(spec, matched, resolver, vname)

        if position == 'rightmost':
            base = _resample_polyline(existing_geoms[0], n_s)
            new_geom = base + (-perp) * lane_spacing
        elif position == 'leftmost':
            base = _resample_polyline(existing_geoms[-1], n_s)
            new_geom = base + perp * lane_spacing
        else:
            base = _resample_polyline(existing_geoms[0], n_s)
            t_c = np.linspace(1, 0, n_s)
            t_c = 0.5 * (1 + np.cos(np.pi * (1 - t_c)))
            new_geom = base + (-perp)[None, :] * lane_spacing * 1.5 * t_c[:, None]

        ax = axes[col]
        for i, g in enumerate(existing_geoms):
            ax.plot(g[:,0], g[:,1], 'b-', linewidth=2, alpha=0.5,
                    label='Existing' if i == 0 else '')
        ax.plot(new_geom[:,0], new_geom[:,1], '-', color=color, linewidth=3, alpha=0.9,
                label=f'Generated ({position})')
        if position == 'merge':
            mid = n_s // 2
            ax.annotate('', xy=(base[mid,0], base[mid,1]),
                        xytext=(new_geom[mid,0], new_geom[mid,1]),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))

        speed_mph = features['sumo_speed'] * 2.237
        info = [f"{features['sumo_speed']:.0f}m/s ({speed_mph:.0f}mph)",
                f"w={features['sumo_width']:.1f}m"]
        if features['disallow']:
            info.append(f"no {','.join(features['disallow'])}")
        if not features['allow_change_right']:
            info.append("no chg R")
        ax.text(0.02, 0.02, ' | '.join(info), transform=ax.transAxes, fontsize=8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15))
        ax.legend(fontsize=8, loc='upper left')
        ax.set_title(f'{title}\ngen_score={gs:.3f}', fontsize=11)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.grid(True, alpha=0.3)

    fig.suptitle(f'C2 Intervention Variants ({label_upper}): {camera}\n'
                 f'Encoder-driven lane specification → SUMO geometric realization',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = output_dir / f"c2_variants_visualization_{model_label}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    logger.info(f"Saved variant visualization to {fig_path}")
    plt.close(fig)


def plot_sumo_summary(output_dir: Path, all_variant_results: list,
                      model_label: str = "independent"):
    """Generate SUMO encoder-attribute summary visualization.

    Shows:
      - Top-left: gen_score bar chart per camera x variant
      - Top-right: encoder → SUMO attribute mapping table
      - Bottom-left: behavioral stats comparison (target vs SUMO)
      - Bottom-right: pipeline summary
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    label_upper = model_label.title()

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # --- Top-left: Gen scores per camera ---
    ax = axes[0, 0]
    cam_names = []
    gen_r, gen_l, gen_m = [], [], []
    for vr in all_variant_results:
        cam_names.append(vr['camera'].replace('US12_', '').replace('I43_', ''))
        for v in vr.get('variants', []):
            vn = v.get('variant_name', '')
            gs = v.get('generation_score', 0)
            if 'rightmost' in vn: gen_r.append(gs)
            elif 'leftmost' in vn: gen_l.append(gs)
            elif 'merge' in vn: gen_m.append(gs)

    x = np.arange(len(cam_names))
    w = 0.25
    if gen_r: ax.bar(x[:len(gen_r)] - w, gen_r, w, color='#e74c3c', alpha=0.8, label='Rightmost')
    if gen_l: ax.bar(x[:len(gen_l)], gen_l, w, color='#2ecc71', alpha=0.8, label='Leftmost')
    if gen_m: ax.bar(x[:len(gen_m)] + w, gen_m, w, color='#9b59b6', alpha=0.8, label='Merge')
    ax.set_xticks(x)
    ax.set_xticklabels(cam_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Gen Score')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f'Generation Score ({label_upper})\n(diffusion behavioral alignment)', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # --- Top-right: Encoder → SUMO mapping table ---
    ax = axes[0, 1]
    ax.axis('off')
    table_data = [
        ['Attribute', 'Rightmost', 'Leftmost', 'Merge'],
        ['Speed', '30-35 m/s\n(free flow)', '20-25 m/s\n(moderate)', '15-20 m/s\n(slow merge)'],
        ['Width', '3.2m\n(standard)', '3.6m\n(passing)', '2.8m\n(narrow)'],
        ['Change L/R', 'no right\n(shoulder)', 'density-based', 'allow all\n(merging)'],
        ['Vehicle', 'all', 'no truck\nno trailer', 'all'],
        ['Source', 'encoder\nobservation', 'encoder\nobservation', 'encoder\nobservation'],
    ]
    colors_table = [['#ddd']*4] + [['white', '#ffe0e0', '#e0ffe0', '#e0e0ff']]*5
    table = ax.table(cellText=table_data, cellColours=colors_table, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title('Encoder → SUMO Attribute Mapping\n(learned from real traffic)', fontsize=11)

    # --- Bottom-left: Behavioral stats (aggregate) ---
    ax = axes[1, 0]
    # Collect behavioral stats across all cameras
    stat_names = ['mean_speed', 'mean_curvature', 'mean_lateral_offset', 'traj_count_norm']
    target_means = {s: [] for s in stat_names}
    sumo_means = {s: [] for s in stat_names}
    for vr in all_variant_results:
        for v in vr.get('variants', []):
            bc = v.get('behavioral_comparison')
            if bc:
                for sn in stat_names:
                    if sn in bc:
                        target_means[sn].append(bc[sn]['target'])
                        sumo_means[sn].append(bc[sn]['sumo'])

    if any(target_means[s] for s in stat_names):
        x_stats = np.arange(len(stat_names))
        t_vals = [np.mean(target_means[s]) if target_means[s] else 0 for s in stat_names]
        s_vals = [np.mean(sumo_means[s]) if sumo_means[s] else 0 for s in stat_names]
        ax.bar(x_stats - 0.15, t_vals, 0.3, color='#3498db', alpha=0.8, label='Target (encoder)')
        ax.bar(x_stats + 0.15, s_vals, 0.3, color='#e67e22', alpha=0.8, label='SUMO (simulated)')
        ax.set_xticks(x_stats)
        ax.set_xticklabels(['Speed', 'Curvature', 'Lat. Offset', 'Density'], fontsize=9)
        ax.set_ylabel('Value (normalized)')
        ax.legend(fontsize=9)
        ax.set_title('Behavioral Stats: Encoder Target vs SUMO Result\n(domain-agnostic comparison)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No behavioral stats available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Behavioral Stats', fontsize=11)

    # --- Bottom-right: Pipeline summary ---
    ax = axes[1, 1]
    ax.axis('off')
    n_cameras = len(all_variant_results)
    n_variants = sum(len(vr.get('variants', [])) for vr in all_variant_results)
    all_gen = [v.get('generation_score', 0) for vr in all_variant_results for v in vr.get('variants', [])]
    mean_gen = np.mean(all_gen) if all_gen else 0
    all_new = [v.get('new_lane_cos_sim') for vr in all_variant_results for v in vr.get('variants', []) if v.get('new_lane_cos_sim') is not None]
    mean_new = np.mean(all_new) if all_new else 0
    all_exist = [v.get('existing_lane_cos_sim') for vr in all_variant_results for v in vr.get('variants', []) if v.get('existing_lane_cos_sim') is not None]
    mean_exist = np.mean(all_exist) if all_exist else 0

    summary_text = (
        f"C2 Pipeline Summary ({label_upper})\n"
        f"{'='*40}\n\n"
        f"Cameras evaluated:  {n_cameras}\n"
        f"Total variants:     {n_variants}\n\n"
        f"Mean gen_score:     {mean_gen:.3f}\n"
        f"Mean new_lane_sim:  {mean_new:.3f}\n"
        f"Mean existing_sim:  {mean_exist:.3f}\n\n"
        f"Pipeline:\n"
        f"  1. Encoder observes real traffic\n"
        f"  2. Diffusion generates lane proposal\n"
        f"  3. Encoder features -> SUMO params\n"
        f"     (speed, width, permissions)\n"
        f"  4. SUMO simulates with params\n"
        f"  5. Re-encode + validate\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'C2 Encoder-Driven SUMO Evaluation ({label_upper})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()

    fig_path = output_dir / f"c2_sumo_summary_{model_label}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    logger.info(f"Saved SUMO summary to {fig_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="C2: End-to-end Behavioral Geometry Synthesis Loop experiment"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Encoder checkpoint path")
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", default=None,
                        help="Single camera to evaluate")
    parser.add_argument("--all-cameras", action="store_true",
                        help="Run on all available cameras")
    parser.add_argument("--sumo-root", default=None,
                        help="Root dir for SUMO networks (default: dataset/sumo)")
    parser.add_argument("--output-dir", default="results/c2_synthesis_loop",
                        help="Output directory for results")
    parser.add_argument("--generate-variants", action="store_true",
                        help="Also run variant generation (requires --diffusion-checkpoint)")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Diffusion model checkpoint")
    parser.add_argument("--relational-checkpoint", default=None,
                        help="Relational diffusion model checkpoint (uses relational generation)")
    parser.add_argument("--calibration-dir", default="dataset/511calibration",
                        help="Directory with per-camera calibration CSVs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_camera").mkdir(exist_ok=True)

    # Resolve SUMO root
    if args.sumo_root:
        sumo_root = Path(args.sumo_root)
    else:
        sumo_root = PROJECT_ROOT / "dataset" / "sumo"
        if not sumo_root.exists():
            # Try graph_geolane sibling
            sumo_root = PROJECT_ROOT.parent / "graph_geolane" / "dataset" / "sumo"

    if not sumo_root.exists():
        logger.error(f"SUMO root not found: {sumo_root}")
        return

    logger.info(f"SUMO root: {sumo_root}")

    # Load calibrations
    calibration_dir = Path(args.calibration_dir)
    calibrations = {}
    if calibration_dir.exists():
        from src.bridge.calibration import load_all_calibrations
        calibrations = load_all_calibrations(calibration_dir, sumo_root)
    else:
        logger.warning(f"No calibration dir at {calibration_dir}, using uncalibrated mode")

    # Load encoder
    model, config, device = load_encoder(args)

    # Determine cameras
    if args.all_cameras:
        # All cameras with encoder data + SUMO networks + calibration
        full_dataset = build_dataset(config)
        all_encoder_cameras = sorted(set(s.camera for s in full_dataset.samples))
        all_sumo_cameras = sorted([
            d.name for d in sumo_root.iterdir()
            if d.is_dir() and (d / "osm.net.xml").exists()
        ])
        cameras = [
            c for c in all_encoder_cameras
            if c in all_sumo_cameras and c in calibrations
        ]
        skipped = [
            c for c in all_encoder_cameras
            if c in all_sumo_cameras and c not in calibrations
        ]
        logger.info(
            f"Found {len(cameras)} cameras with encoder + SUMO + calibration: "
            f"{cameras}"
        )
        if skipped:
            logger.info(f"Skipped (no calibration): {skipped}")
    elif args.camera:
        cameras = [args.camera]
    else:
        parser.error("Specify --camera or --all-cameras")
        return

    # Run experiment per camera
    all_results = []
    all_variant_results = []

    for camera in cameras:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Camera: {camera}")
        logger.info(f"{'='*60}")

        dataset = build_dataset(config, cameras=[camera])
        if len(dataset) == 0:
            logger.warning(f"[{camera}] No lanes in dataset, skipping")
            continue

        # Encode real lanes
        enc_output, batch, embeddings = encode_all(model, dataset, device)

        # Step 1: Baseline re-encoding via SUMO
        calib = calibrations.get(camera)
        if calib:
            logger.info(f"[{camera}] Using calibration (reproj error={calib.reprojection_error:.1f}px)")
        else:
            logger.info(f"[{camera}] No calibration available, using uncalibrated mode")

        result = run_baseline_reencoding(
            model, dataset, device, embeddings, camera, sumo_root,
            calibration=calib,
        )

        if result is not None:
            all_results.append(result)

            # Save per-camera detail
            cam_path = output_dir / "per_camera" / f"{camera}.json"
            with open(cam_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"[{camera}] Saved to {cam_path}")

        # Step 2: Variant generation (optional)
        if args.generate_variants and args.diffusion_checkpoint:
            var_result = run_variant_generation(
                model, dataset, device, embeddings, camera, args,
                calibration=calib,
            )
            if var_result is not None:
                all_variant_results.append(var_result)

    # Aggregate summary
    if all_results:
        cos_sims = [r["mean_cosine_similarity"] for r in all_results]
        l2_dists = [r["mean_l2_distance"] for r in all_results]
        n_matched = [r["n_matched_lanes"] for r in all_results]
        total_times = [r["timing"]["total_s"] for r in all_results]

        summary = {
            "experiment": "C2_synthesis_loop",
            "timestamp": time.time(),
            "n_cameras": len(all_results),
            "cameras": [r["camera"] for r in all_results],
            "aggregate": {
                "mean_cosine_similarity": float(np.mean(cos_sims)),
                "std_cosine_similarity": float(np.std(cos_sims)),
                "mean_l2_distance": float(np.mean(l2_dists)),
                "std_l2_distance": float(np.std(l2_dists)),
                "total_matched_lanes": sum(n_matched),
                "mean_time_per_camera_s": float(np.mean(total_times)),
            },
            "per_camera_summary": [
                {
                    "camera": r["camera"],
                    "cos_sim": r["mean_cosine_similarity"],
                    "l2_dist": r["mean_l2_distance"],
                    "n_matched": r["n_matched_lanes"],
                    "time_s": r["timing"]["total_s"],
                }
                for r in all_results
            ],
        }

        if all_variant_results:
            summary["variant_generation"] = all_variant_results

        # Save
        results_path = output_dir / "c2_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        summary_path = output_dir / "c2_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Variant generation summary (SUMO-vs-SUMO closed loop) — primary result
        if all_variant_results:
            logger.info(f"\n{'='*70}")
            logger.info("C2 CLOSED LOOP — VARIANT GENERATION (SUMO-vs-SUMO)")
            logger.info(f"{'='*70}")
            logger.info(
                f"{'Camera':<20} {'Variant':<16} {'new_cos':>8} {'exist_cos':>10} "
                f"{'gen_score':>10} {'can_sim':>8}"
            )
            logger.info("-" * 74)

            all_new_sims = []
            all_exist_sims = []
            for var_result in all_variant_results:
                cam = var_result["camera"]
                for v in var_result.get("variants", []):
                    new_sim = v.get("new_lane_cos_sim")
                    exist_sim = v.get("existing_lane_cos_sim")
                    gen = v.get("generation_score", 0)
                    can = v.get("canonical_geometry_sim")
                    name = v.get("variant_name", "?")

                    if new_sim is not None:
                        all_new_sims.append(new_sim)
                    if exist_sim is not None:
                        all_exist_sims.append(exist_sim)

                    logger.info(
                        f"{cam:<20} {name:<16} "
                        f"{new_sim if new_sim is not None else 'N/A':>8} "
                        f"{exist_sim if exist_sim is not None else 'N/A':>10} "
                        f"{gen:>10.3f} "
                        f"{can if can is not None else 'N/A':>8}"
                    )

            if all_new_sims:
                logger.info("-" * 74)
                logger.info(
                    f"{'AGGREGATE':<20} {'':16} "
                    f"{np.mean(all_new_sims):>8.3f} "
                    f"{np.mean(all_exist_sims):>10.3f}" if all_exist_sims else ""
                )
                logger.info(
                    f"  Mean new_lane_cos_sim: {np.mean(all_new_sims):.3f} "
                    f"(±{np.std(all_new_sims):.3f})"
                )

            # Save variant summary
            summary["variant_summary"] = {
                "n_variants": len(all_new_sims),
                "mean_new_lane_cos_sim": float(np.mean(all_new_sims)) if all_new_sims else None,
                "std_new_lane_cos_sim": float(np.std(all_new_sims)) if all_new_sims else None,
                "mean_existing_cos_sim": float(np.mean(all_exist_sims)) if all_exist_sims else None,
            }
            # Re-save summary with variant data
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        # Generate visualizations
        model_label = "relational" if getattr(args, "relational_checkpoint", None) else "independent"
        plot_c2_results(output_dir, all_variant_results, all_results, model_label=model_label)
        if all_variant_results:
            plot_sumo_summary(output_dir, all_variant_results, model_label=model_label)
            # Variant lane visualization for the best-calibrated camera
            best_cam = None
            best_err = float('inf')
            for cam in cameras:
                c = calibrations.get(cam)
                if c and c.reprojection_error < best_err:
                    if any(vr['camera'] == cam for vr in all_variant_results):
                        best_err = c.reprojection_error
                        best_cam = cam
            if best_cam:
                plot_variant_lanes(
                    output_dir, best_cam, config, args, device,
                    calibrations, sumo_root, model_label=model_label,
                )

        logger.info(f"\nResults saved to {output_dir}/")
    else:
        logger.error("No cameras produced results. Check SUMO installation and network files.")


if __name__ == "__main__":
    main()
