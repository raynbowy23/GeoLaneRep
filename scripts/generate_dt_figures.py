#!/usr/bin/env python3
"""Generate Digital Twin integration figures (D-series).

Figures:
    D1 — Physical vs digital side-by-side: camera frame with lane regions
         colored by LOS alongside schematic road network with same coloring.
    D2 — Ephemeral event scenario: multi-panel timeline showing incident
         detection → behavioral shift → topology update → twin state change.
         Also exports each timestep to shared directory for live twin consumption.
    D3 — Edit-Type Failure Taxonomy: 3-row figure showing where behavioral
         generation works (Type 1) and where it fails (Type 2-3).
    D4 — Behavioral Geometry Synthesis Loop: end-to-end demonstration of
         real lanes → SUMO simulate → compare behavioral metrics (speed,
         density, LOS). Shows that the same geometry produces matching
         traffic state across camera and simulation domains.

Usage:
    # Generate figures only (no live export):
    python scripts/generate_dt_figures.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --figures D1 D2 D4

    # D3 requires diffusion checkpoint:
    python scripts/generate_dt_figures.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --diffusion-checkpoint results/diffusion/checkpoints/best.pt \
        --figures D3

    # D2 with live export to shared directory (twin watches this):
    python scripts/generate_dt_figures.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --figures D2 --export-dir shared/dt_state --live
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# LOS color map (matches twin's encoder_bridge.rs)
LOS_COLORS = {
    "A": "#00b400",
    "B": "#64c800",
    "C": "#dcdc00",
    "D": "#ffa500",
    "E": "#ff5000",
    "F": "#dc0000",
}


# ---------------------------------------------------------------------------
# Data loading (shared with bridge figures)
# ---------------------------------------------------------------------------

def _load_encoder(args):
    """Load encoder model and dataset."""
    import yaml
    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import load_trained_encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_encoder(args.checkpoint, device)

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    dataset = LaneDataset(
        config=config,
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


def _get_bridge_metrics(dataset, encoder_output):
    """Compute traffic metrics via bridge."""
    from src.bridge.traffic_translator import EncoderTrafficBridge
    bridge = EncoderTrafficBridge()
    return bridge.translate_batch(dataset, {
        k: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in encoder_output.items()
    })


# ---------------------------------------------------------------------------
# Anomaly injection for D2
# ---------------------------------------------------------------------------

def _inject_incident(batch, dataset, target_indices, severity=0.8):
    """Inject incident into specific lanes by corrupting trajectory stats.

    Simulates: speed drop, density spike, trajectory count drop.

    Args:
        batch: Collated batch dict.
        dataset: LaneDataset.
        target_indices: Lane indices to corrupt.
        severity: 0-1, how severe the incident is.

    Returns:
        Modified batch (cloned).
    """
    corrupted = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

    stats = corrupted["traj_stats"].clone()
    for idx in target_indices:
        # Speed drops
        stats[idx, 0] *= (1.0 - severity * 0.9)
        # Curvature increases (erratic driving)
        stats[idx, 1] *= (1.0 + severity * 2.0)
        # Lateral offset increases (lane changing)
        stats[idx, 2] *= (1.0 + severity * 1.5)
        # Count drops (fewer vehicles passing)
        stats[idx, 3] *= (1.0 - severity * 0.7)
    corrupted["traj_stats"] = stats

    return corrupted


# ---------------------------------------------------------------------------
# Figure D1: Physical vs Digital Side-by-Side
# ---------------------------------------------------------------------------

def figure_d1(metrics, dataset, output_dir: Path):
    """Camera view with LOS overlay alongside schematic road network."""

    # Pick a camera with multiple groups
    cam_counts = {}
    for s in dataset.samples:
        cam_counts.setdefault(s.camera, set()).add(s.group_id)
    best_cam = max(cam_counts, key=lambda c: len(cam_counts[c]))
    cam_samples = [s for s in dataset.samples if s.camera == best_cam]
    cam_metrics = [m for m in metrics if m.camera == best_cam]

    fig, (ax_phys, ax_dt) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: "Physical" view — lane geometries colored by LOS
    ax_phys.set_title(f"{best_cam}: Physical View (Lane Geometries)", fontsize=10)
    ax_phys.set_facecolor("#1a1a2e")

    for sample, metric in zip(cam_samples, cam_metrics):
        geom = sample.geometry  # (K, 2)
        color = LOS_COLORS.get(metric.los, "#808080")
        ax_phys.plot(geom[:, 0], geom[:, 1], color=color, linewidth=3, alpha=0.8)
        # Label with LOS
        mid = len(geom) // 2
        ax_phys.annotate(
            f"LOS {metric.los}",
            (geom[mid, 0], geom[mid, 1]),
            color="white", fontsize=7, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
        )

    ax_phys.set_xlabel("x (pixels)")
    ax_phys.set_ylabel("y (pixels)")
    ax_phys.invert_yaxis()
    ax_phys.set_aspect("equal")

    # Right panel: "Digital" view — schematic with traffic metrics
    ax_dt.set_title(f"{best_cam}: Digital Twin State", fontsize=10)
    ax_dt.set_facecolor("#0e1117")

    for i, (sample, metric) in enumerate(zip(cam_samples, cam_metrics)):
        geom = sample.geometry
        color = LOS_COLORS.get(metric.los, "#808080")

        # Draw lane as thick line
        ax_dt.plot(geom[:, 0], geom[:, 1], color=color, linewidth=5, alpha=0.6)
        # Draw direction arrow
        if len(geom) >= 2:
            dx = geom[-1, 0] - geom[-2, 0]
            dy = geom[-1, 1] - geom[-2, 1]
            ax_dt.annotate(
                "", xy=(geom[-1, 0], geom[-1, 1]),
                xytext=(geom[-2, 0], geom[-2, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
            )
        # Metrics label
        mid = len(geom) // 2
        label = (
            f"{metric.speed_mph:.0f} mph\n"
            f"{metric.density_veh_mi_ln:.0f} veh/mi/ln\n"
            f"V/C={metric.vc_ratio:.2f}"
        )
        ax_dt.annotate(
            label,
            (geom[mid, 0] + 10, geom[mid, 1]),
            color="white", fontsize=6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.8),
        )

    ax_dt.set_xlabel("x")
    ax_dt.set_ylabel("y")
    ax_dt.invert_yaxis()
    ax_dt.set_aspect("equal")

    # Legend
    handles = [mpatches.Patch(color=c, label=f"LOS {g}") for g, c in LOS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)

    fig.suptitle("Physical vs Digital Twin View", fontsize=13, y=1.02)
    fig.tight_layout()

    path = output_dir / "D1_physical_vs_digital.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved D1 to {path}")


# ---------------------------------------------------------------------------
# Figure D2: Ephemeral Event Scenario
# ---------------------------------------------------------------------------

def figure_d2(
    model, dataset, device, metrics_baseline, embeddings_baseline,
    output_dir: Path, exporter=None, live=False,
):
    """Multi-panel timeline: normal → incident → shift → update.

    4 panels showing the same camera/group at different timesteps:
    t0: Normal operation (LOS B/C)
    t1: Incident injected (speed drops, anomaly detected)
    t2: Encoder captures behavioral shift (embedding drift)
    t3: Twin updates (topology regenerated, LOS recalculated)
    """
    from src.data.lane_dataset import collate_fn
    from src.bridge.traffic_translator import EncoderTrafficBridge

    # Pick a camera with enough lanes
    cam_counts = {}
    for s in dataset.samples:
        cam_counts.setdefault(s.camera, []).append(s)
    best_cam = max(cam_counts, key=lambda c: len(cam_counts[c]))
    cam_indices = [i for i, s in enumerate(dataset.samples) if s.camera == best_cam]

    # Target: corrupt 1-2 lanes (simulate incident on rightmost lane)
    target_indices = cam_indices[:2]

    # Encode baseline
    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    # t0: baseline
    bridge = EncoderTrafficBridge()

    # t1-t2: incident with increasing severity
    severities = [0.0, 0.5, 0.8, 0.8]
    panel_titles = [
        "t=0: Normal Operation",
        "t=1: Incident Detected",
        "t=2: Behavioral Shift",
        "t=3: Twin Updated",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    previous_state = None

    for panel_idx, (severity, title) in enumerate(zip(severities, panel_titles)):
        ax = axes[panel_idx]
        ax.set_facecolor("#0e1117")

        if severity > 0:
            corrupted = _inject_incident(batch, dataset, target_indices, severity)
        else:
            corrupted = batch

        # Re-encode
        model.eval()
        with torch.no_grad():
            stats_input = torch.cat([
                corrupted["traj_stats"].to(device),
                corrupted["roles"].to(device) if "roles" in corrupted else batch["roles"].to(device),
            ], dim=-1)
            output = model(
                geometry=corrupted["geometry"].to(device),
                traj_polylines=corrupted["traj_polylines"].to(device),
                traj_mask=corrupted["traj_mask"].to(device),
                traj_stats=stats_input,
            )

        enc_out = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in output.items()}
        panel_metrics = bridge.translate_batch(dataset, enc_out)
        panel_embeddings = output["embedding"].cpu().numpy()

        # Compute anomaly scores (embedding drift from baseline)
        anomaly_scores = np.linalg.norm(
            panel_embeddings - embeddings_baseline, axis=1
        )
        # Normalize to [0, 1]
        if anomaly_scores.max() > 0:
            anomaly_scores = anomaly_scores / (anomaly_scores.max() + 1e-8)

        # Draw lanes for this camera
        for i, (sample, metric) in enumerate(zip(dataset.samples, panel_metrics)):
            if sample.camera != best_cam:
                continue

            geom = sample.geometry
            is_target = i in target_indices
            color = LOS_COLORS.get(metric.los, "#808080")

            lw = 5 if is_target else 3
            ax.plot(geom[:, 0], geom[:, 1], color=color, linewidth=lw,
                    alpha=0.9 if is_target else 0.5)

            mid = len(geom) // 2
            if is_target:
                anom = anomaly_scores[i] if i < len(anomaly_scores) else 0
                label = f"LOS {metric.los} | {metric.speed_mph:.0f}mph"
                if severity > 0:
                    label += f"\nanom={anom:.2f}"
                ax.annotate(
                    label,
                    (geom[mid, 0], geom[mid, 1]),
                    color="white", fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                )

        ax.set_title(title, fontsize=10, color="white",
                     bbox=dict(facecolor="#1a1a2e", alpha=0.8))
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.tick_params(colors="gray", labelsize=6)

        # Export to twin if live mode
        if exporter and live:
            exporter.write_full_update(
                metrics=panel_metrics,
                anomaly_scores=anomaly_scores,
                embeddings=panel_embeddings,
                previous_state=previous_state,
            )
            logger.info(f"Exported panel {panel_idx} ({title}) to twin")
            time.sleep(1.0)  # Give twin time to poll

    # Legend
    handles = [mpatches.Patch(color=c, label=f"LOS {g}") for g, c in LOS_COLORS.items()]
    handles.append(mpatches.Patch(color="none", label="Bold = incident lane"))
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8)

    fig.suptitle(
        "Ephemeral Event — Incident Detection and Twin Update",
        fontsize=13, color="black",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    path = output_dir / "D2_ephemeral_event.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved D2 to {path}")


# ---------------------------------------------------------------------------
# Figure D3: Edit-Type Failure Taxonomy
# ---------------------------------------------------------------------------

def _min_polyline_distance(geom_a, geom_b):
    """Minimum point-to-point distance between two polylines."""
    dists = np.linalg.norm(geom_a[:, None, :] - geom_b[None, :, :], axis=2)
    return float(dists.min())


def _mean_polyline_distance(geom_a, geom_b):
    """Mean nearest-point distance from geom_a to geom_b."""
    dists = np.linalg.norm(geom_a[:, None, :] - geom_b[None, :, :], axis=2)
    return float(dists.min(axis=1).mean())


def _check_overlap(generated, existing_lanes, min_spacing=0.008):
    """Check if generated lane overlaps any existing lane.

    Returns (has_overlap, min_distance, closest_lane_idx).
    min_spacing is in normalized [0,1] coords (~8 pixels at 1080p).
    """
    if not existing_lanes:
        return False, float("inf"), -1
    distances = [_min_polyline_distance(generated, ex) for ex in existing_lanes]
    closest_idx = int(np.argmin(distances))
    min_dist = distances[closest_idx]
    has_overlap = min_dist < min_spacing
    return has_overlap, min_dist, closest_idx


def _check_merge_connectivity(generated, neighbor_geom, max_endpoint_gap=0.03):
    """Check if merge lane endpoint reaches the neighbor lane.

    Returns (is_connected, endpoint_gap, start_offset).
    """
    # Merge lane should converge: endpoint close to neighbor, startpoint far
    end_pt = generated[-1]
    start_pt = generated[0]
    end_dists = np.linalg.norm(neighbor_geom - end_pt, axis=1)
    start_dists = np.linalg.norm(neighbor_geom - start_pt, axis=1)
    endpoint_gap = float(end_dists.min())
    start_offset = float(start_dists.min())
    is_connected = endpoint_gap < max_endpoint_gap
    return is_connected, endpoint_gap, start_offset


def figure_d3(
    model, dataset, device, output_dir: Path,
    diffusion_checkpoint=None,
    relational_checkpoint=None,
):
    """Edit-Type Failure Taxonomy: independent vs relational generation.

    Three rows (edit types) × 4 cols:
      Col 0: Existing group
      Col 1: Independent generation (baseline denoiser)
      Col 2: Relational generation (neighbor-aware denoiser)
      Col 3: Quantitative comparison

    Edit types:
      Type 1 — Topology-preserving (replace rightmost with variant)
      Type 2 — Cross-section coupling (add rightmost lane)
      Type 3 — Merge/diverge (add merge lane)
    """
    from src.generation.spec import LaneSpecification, SpecEmbeddingResolver
    from src.generation.directed import DirectedLaneGenerator
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer
    from src.data.lane_dataset import collate_fn

    # --- Load models ---
    generator = None
    rel_trainer = None
    index = None

    if diffusion_checkpoint and Path(diffusion_checkpoint).exists():
        # Build retrieval index
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
        geometries = batch["geometry"].cpu().numpy()

        from src.generation.retrieval import LaneRetrievalIndex
        index = LaneRetrievalIndex(
            embeddings=embeddings,
            geometries=geometries,
            lane_keys=[s.lane_key for s in dataset.samples],
            cameras=[s.camera for s in dataset.samples],
        )

        K = index.K
        geom_dim = K * 2
        cond_dim = index.D

        denoiser = LaneDenoiser(
            geom_dim=geom_dim, t_dim=64, cond_dim=cond_dim, hidden_dim=256,
        )
        schedule = DDPMSchedule(T=100)
        trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))
        trainer.load(diffusion_checkpoint)
        logger.info(f"Loaded independent diffusion model from {diffusion_checkpoint}")

        resolver = SpecEmbeddingResolver(index, dataset)
        generator = DirectedLaneGenerator(
            resolver, trainer, encoder=model, dataset=dataset, device=device,
        )

        # Try loading relational model
        if relational_checkpoint and Path(relational_checkpoint).exists():
            from src.generation.relational_diffusion import (
                RelationalLaneDenoiser, RelationalDiffusionTrainer,
            )
            rel_denoiser = RelationalLaneDenoiser(
                geom_dim=geom_dim, t_dim=64, cond_dim=cond_dim,
                rel_dim=64, hidden_dim=256,
            )
            rel_schedule = DDPMSchedule(T=100)
            rel_trainer = RelationalDiffusionTrainer(
                rel_denoiser, rel_schedule, device=str(device),
            )
            rel_trainer.load(relational_checkpoint)
            logger.info(f"Loaded relational diffusion model from {relational_checkpoint}")
        else:
            # Auto-discover relational checkpoint next to diffusion checkpoint
            auto_path = Path(diffusion_checkpoint).parent / "relational_diffusion_model.pt"
            if auto_path.exists():
                from src.generation.relational_diffusion import (
                    RelationalLaneDenoiser, RelationalDiffusionTrainer,
                )
                rel_denoiser = RelationalLaneDenoiser(
                    geom_dim=geom_dim, t_dim=64, cond_dim=cond_dim,
                    rel_dim=64, hidden_dim=256,
                )
                rel_schedule = DDPMSchedule(T=100)
                rel_trainer = RelationalDiffusionTrainer(
                    rel_denoiser, rel_schedule, device=str(device),
                )
                rel_trainer.load(str(auto_path))
                logger.info(f"Auto-loaded relational model from {auto_path}")
            else:
                logger.warning(
                    "No relational checkpoint found. Col 2 will show "
                    "'not available'. Pass --relational-checkpoint or place "
                    "relational_diffusion_model.pt next to diffusion checkpoint."
                )
    else:
        logger.warning(
            "D3 requires --diffusion-checkpoint. Skipping."
        )
        return

    # --- Pick a camera/group WITHOUT existing merge lanes ---
    # Stronger test: can the model generate merge lanes for a group
    # that only has regular parallel lanes?
    camera_groups = {}
    for sample in dataset.samples:
        key = (sample.camera, sample.group_id)
        camera_groups.setdefault(key, []).append(sample)

    # Prefer a group WITHOUT merge lanes but with ≥4 lanes (enough to test all types)
    best_key = None
    for key, samples in sorted(camera_groups.items(),
                                key=lambda kv: len(kv[1]), reverse=True):
        has_merge = any(s.role.has_successor for s in samples)
        if not has_merge and len(samples) >= 4:
            best_key = key
            break
    if best_key is None:
        # Fallback: largest group regardless
        best_key = max(camera_groups, key=lambda k: len(camera_groups[k]))

    target_camera, target_group = best_key
    group_samples = camera_groups[best_key]
    has_merge = any(s.role.has_successor for s in group_samples)
    logger.info(f"D3 target: {target_camera} group {target_group} "
                f"({len(group_samples)} lanes, merge={'yes' if has_merge else 'no'})")

    # Get existing geometries
    group_indices = [
        i for i, s in enumerate(dataset.samples)
        if s.camera == target_camera and s.group_id == target_group
    ]
    existing_geoms = [np.asarray(dataset.samples[i].geometry) for i in group_indices]

    # Load camera frame
    annot_dir = Path(
        dataset.config.get("data", {}).get("annotation_dir", "../dataset/preprocess")
    )
    frame_path = annot_dir / target_camera / "last_frame.npy"
    frame = None
    if frame_path.exists():
        import cv2
        frame = np.load(str(frame_path))
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Compute inter-lane spacing for violation thresholds
    if len(existing_geoms) > 1:
        centroids = np.array([g.mean(axis=0) for g in existing_geoms])
        tangents = []
        for g in existing_geoms:
            t = g[-1] - g[0]
            n = np.linalg.norm(t)
            if n > 1e-8:
                tangents.append(t / n)
        mean_t = np.mean(tangents, axis=0) if tangents else np.array([1.0, 0.0])
        mean_t /= np.linalg.norm(mean_t) + 1e-8
        perp = np.array([-mean_t[1], mean_t[0]])
        lateral_pos = centroids @ perp
        spacings = np.diff(np.sort(lateral_pos))
        median_spacing = float(np.median(spacings)) if len(spacings) > 0 else 0.02
        lateral_pos_arr = lateral_pos
    else:
        median_spacing = 0.02
        lateral_pos_arr = np.zeros(len(existing_geoms))

    # =====================================================================
    # Build the 3-row × 4-col figure
    # =====================================================================
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))

    edit_types = [
        {
            "label": "Reconstruct",
            "subtitle": "Re-generate the rightmost lane (should match original)",
            "check": "type1",
        },
        {
            "label": "Extend",
            "subtitle": "Add a new lane next to the rightmost",
            "check": "type2",
        },
        {
            "label": "Merge",
            "subtitle": "Add a lane that merges into an existing neighbor",
            "check": "type3",
        },
    ]

    def _draw_existing(ax):
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        if frame is not None:
            ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.4)
        for g in existing_geoms:
            ax.plot(g[:, 0], g[:, 1], color="#377eb8", linewidth=2, alpha=0.8)
        ax.axis("off")

    def _draw_generated(ax, gen, label_suffix=""):
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        if frame is not None:
            ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.3)
        for g in existing_geoms:
            ax.plot(g[:, 0], g[:, 1], color="#cccccc", linewidth=1.5, alpha=0.6)
        if gen is not None:
            ax.plot(gen[:, 0], gen[:, 1], color="#e41a1c", linewidth=3, alpha=0.9)
            ax.scatter(gen[0, 0], gen[0, 1], c="green", s=40, zorder=5)
            ax.scatter(gen[-1, 0], gen[-1, 1], c="red", s=40, zorder=5)
        else:
            ax.text(0.5, 0.5, "Not available",
                    transform=ax.transAxes, fontsize=11, ha="center",
                    color="#999999")
        ax.axis("off")

    def _build_spec(edit_info):
        if edit_info["check"] == "type1":
            rightmost_idx = int(np.argmax(lateral_pos_arr)) if len(existing_geoms) > 1 else 0
            rightmost_sample = group_samples[rightmost_idx]
            return LaneSpecification.replace_role(
                rightmost_sample.cls_id, "rightmost",
                camera=target_camera, group_id=target_group,
            )
        elif edit_info["check"] == "type2":
            return LaneSpecification.rightmost(target_camera, target_group)
        else:  # type3
            return LaneSpecification.merge_lane(target_camera, target_group)

    def _measure(gen, edit_info, result=None):
        """Return dict of metrics for a generated lane."""
        m = {}
        if gen is None:
            return m
        if edit_info["check"] == "type1":
            # Chamfer distance to the original lane (how similar is the variant?)
            rightmost_idx = int(np.argmax(lateral_pos_arr)) if len(existing_geoms) > 1 else 0
            original = existing_geoms[rightmost_idx]
            # Resample to same length for fair comparison
            from scipy.interpolate import interp1d
            def _resample(poly, n=32):
                t = np.linspace(0, 1, len(poly))
                t_new = np.linspace(0, 1, n)
                return np.column_stack([
                    interp1d(t, poly[:, 0], kind="linear")(t_new),
                    interp1d(t, poly[:, 1], kind="linear")(t_new),
                ])
            orig_r = _resample(original)
            gen_r = _resample(gen)
            # Symmetric chamfer: mean of (a→b nearest) + (b→a nearest)
            d_ab = np.linalg.norm(orig_r[:, None, :] - gen_r[None, :, :], axis=2)
            chamfer = float(d_ab.min(axis=1).mean() + d_ab.min(axis=0).mean()) / 2
            m["chamfer"] = chamfer
        elif edit_info["check"] == "type2":
            # Spacing error: |actual spacing to nearest - median spacing|
            _, min_dist, _ = _check_overlap(gen, existing_geoms)
            spacing_error = abs(min_dist - median_spacing)
            m["min_dist"] = min_dist
            m["spacing_error"] = spacing_error
        elif edit_info["check"] == "type3":
            neighbor_geom = None
            if result and result.spatial_context and result.spatial_context.neighbor_geometry is not None:
                neighbor_geom = result.spatial_context.neighbor_geometry
            if neighbor_geom is not None:
                is_connected, endpoint_gap, start_offset = _check_merge_connectivity(
                    gen, neighbor_geom)
                m["connected"] = is_connected
                m["endpoint_gap"] = endpoint_gap
                m["start_offset"] = start_offset
                m["neighbor_geom"] = neighbor_geom
        return m

    # Precompute rightmost index for type1 (reconstruction)
    rightmost_idx = int(np.argmax(lateral_pos_arr)) if len(existing_geoms) > 1 else 0
    gt_rightmost = existing_geoms[rightmost_idx]

    for row, edit_info in enumerate(edit_types):
        spec = _build_spec(edit_info)

        # --- Col 0: Existing group ---
        _draw_existing(axes[row, 0])
        if row == 0:
            axes[row, 0].set_title(
                f"Existing Group\n{target_camera} g{target_group} "
                f"({len(existing_geoms)} lanes)", fontsize=9)
        else:
            axes[row, 0].set_title("Existing Group", fontsize=9)

        # --- Col 1: Independent (baseline) ---
        ind_result = generator.generate(spec, n_candidates=10)
        ind_gen = ind_result.best
        _draw_generated(axes[row, 1], ind_gen)
        # For reconstruct: overlay GT as dashed green
        if edit_info["check"] == "type1" and ind_gen is not None:
            axes[row, 1].plot(gt_rightmost[:, 0], gt_rightmost[:, 1],
                              color="#4daf4a", linewidth=2, linestyle="--",
                              alpha=0.8, label="ground truth")
            axes[row, 1].legend(fontsize=7, loc="upper right")
        axes[row, 1].set_title(
            f"{edit_info['label']}\nIndependent", fontsize=9, fontweight="bold")

        # --- Col 2: Relational ---
        rel_gen = None
        rel_result = None
        if rel_trainer is not None:
            generator_with_rel = DirectedLaneGenerator(
                generator.resolver, generator.trainer,
                encoder=model, dataset=dataset, device=device,
                relational_trainer=rel_trainer,
            )
            rel_result = generator_with_rel.generate_relational(spec, n_candidates=10)
            rel_gen = rel_result.best
            _draw_generated(axes[row, 2], rel_gen)
        else:
            _draw_generated(axes[row, 2], None)
        # For reconstruct: overlay GT as dashed green
        if edit_info["check"] == "type1" and rel_gen is not None:
            axes[row, 2].plot(gt_rightmost[:, 0], gt_rightmost[:, 1],
                              color="#4daf4a", linewidth=2, linestyle="--",
                              alpha=0.8, label="ground truth")
            axes[row, 2].legend(fontsize=7, loc="upper right")
        axes[row, 2].set_title(
            f"{edit_info['label']}\nRelational", fontsize=9, fontweight="bold")

        # --- Col 3: Comparison ---
        ax = axes[row, 3]
        ind_m = _measure(ind_gen, edit_info, ind_result)
        rel_m = _measure(rel_gen, edit_info, rel_result)

        if edit_info["check"] == "type1":
            # Chamfer distance to original lane
            labels = []
            vals = []
            if "chamfer" in ind_m:
                labels.append("Independent")
                vals.append(ind_m["chamfer"])
            if "chamfer" in rel_m:
                labels.append("Relational")
                vals.append(rel_m["chamfer"])

            if vals:
                colors = ["#5b9bd5", "#ed7d31"][:len(vals)]
                bars = ax.barh(range(len(vals)), vals, color=colors, height=0.5)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlabel("Chamfer distance (lower = more similar)", fontsize=9)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f}", va="center", fontsize=9)
            if len(vals) == 2:
                winner = "Relational" if vals[1] < vals[0] else "Independent"
                ax.text(0.95, 0.95, f"Better: {winner}",
                        transform=ax.transAxes, fontsize=8, ha="right", va="top",
                        fontweight="bold", color="#333333",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", alpha=0.9))
            ax.set_title("Reconstruction Error (Chamfer to GT)", fontsize=9)

        elif edit_info["check"] == "type2":
            # Spacing error: how close to expected median spacing
            labels = []
            vals_err = []
            vals_dist = []
            if "spacing_error" in ind_m:
                labels.append("Independent")
                vals_err.append(ind_m["spacing_error"])
                vals_dist.append(ind_m["min_dist"])
            if "spacing_error" in rel_m:
                labels.append("Relational")
                vals_err.append(rel_m["spacing_error"])
                vals_dist.append(rel_m["min_dist"])

            if vals_err:
                colors = ["#5b9bd5", "#ed7d31"][:len(vals_err)]
                bars = ax.barh(range(len(vals_err)), vals_err, color=colors, height=0.5)
                ax.set_yticks(range(len(vals_err)))
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlabel("Spacing error (lower = better)", fontsize=9)
                for bar, err, dist in zip(bars, vals_err, vals_dist):
                    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                            f"{err:.4f}  (actual: {dist:.3f})", va="center", fontsize=8)
            # Show expected spacing
            ax.text(0.95, 0.05,
                    f"Expected spacing: {median_spacing:.3f}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    color="#666666", style="italic")
            if len(vals_err) == 2:
                winner = "Relational" if vals_err[1] < vals_err[0] else "Independent"
                ax.text(0.95, 0.95, f"Better: {winner}",
                        transform=ax.transAxes, fontsize=8, ha="right", va="top",
                        fontweight="bold", color="#333333",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", alpha=0.9))
            ax.set_title("|Actual Spacing − Expected Spacing|", fontsize=9)

        elif edit_info["check"] == "type3":
            # Merge connectivity comparison
            labels = []
            vals = []
            colors = []
            if "endpoint_gap" in ind_m:
                labels.append("Independent")
                vals.append(ind_m["endpoint_gap"])
                colors.append("#4daf4a" if ind_m.get("connected") else "#e41a1c")
            if "endpoint_gap" in rel_m:
                labels.append("Relational")
                vals.append(rel_m["endpoint_gap"])
                colors.append("#4daf4a" if rel_m.get("connected") else "#e41a1c")

            if vals:
                bars = ax.barh(range(len(vals)), vals, color=colors, height=0.5)
                ax.set_yticks(range(len(vals)))
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_xlabel("Endpoint gap (lower = better)", fontsize=9)
                ax.axvline(x=0.03, color="blue", linestyle="--",
                           linewidth=1, label="connectivity threshold (0.03)")
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                            f"{val:.4f}", va="center", fontsize=9)
                ax.legend(fontsize=7, loc="lower right")

            if len(vals) == 2:
                winner = "Relational" if vals[1] < vals[0] else "Independent"
                ax.text(0.95, 0.95, f"Better: {winner}",
                        transform=ax.transAxes, fontsize=8, ha="right", va="top",
                        fontweight="bold", color="#333333",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", alpha=0.9))
            ax.set_title("Endpoint Gap to Neighbor Lane", fontsize=9)

    # --- Global legend explaining edit types ---
    legend_text = (
        "Edit Types:\n"
        "  Reconstruct — Re-generate the rightmost lane from its embedding.\n"
        "                Test: Chamfer distance to ground truth (lower = better match).\n"
        "  Extend      — Add a new lane beside the outermost existing lane.\n"
        "                Test: spacing error vs expected inter-lane distance (lower = better).\n"
        "  Merge       — Add a lane that converges into an existing neighbor lane.\n"
        "                Test: endpoint gap to target lane (lower = better connectivity).\n"
        "\n"
        "Bar color: green = connected / red = disconnected (merge only)\n"
        "Dashed blue line = threshold for acceptable distance"
    )
    fig.text(0.5, -0.02, legend_text, fontsize=9, ha="center", va="top",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8",
                       edgecolor="#cccccc", alpha=0.95))

    fig.suptitle(
        "Lane Edit Taxonomy: Independent vs Relational Generation\n"
        f"{target_camera} group {target_group} "
        f"({'no existing merge lanes' if not has_merge else 'has merge lanes'})",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    path = output_dir / "D3_edit_type_taxonomy.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved D3 to {path}")


# ---------------------------------------------------------------------------
# Figure D4: Behavioral Geometry Synthesis Loop
# (Renamed from D5; old D4 system overview removed — redundant with framework figure)
# ---------------------------------------------------------------------------

def _estimate_flow_from_observed(dataset, camera, observation_minutes=60):
    """Estimate per-direction flow rate from observed trajectory counts.

    Uses total trajectory count across lanes to estimate vehicles/hour,
    which calibrates SUMO demand to match observed conditions.
    """
    cam_samples = [s for s in dataset.samples if s.camera == camera]
    if not cam_samples:
        return 1200  # fallback

    # Total observed trajectories across all lanes
    total_trajs = sum(len(s.trajectories) for s in cam_samples)
    # Trajectories are collected over ~1 hour typically
    # Each trajectory = one vehicle pass, but counted per lane
    # Divide by number of groups to avoid double-counting
    group_ids = set(s.group_id for s in cam_samples)
    n_groups = max(len(group_ids), 1)

    # Rough estimate: total unique vehicles ≈ total_trajs / n_groups
    # (each vehicle appears in ~1 lane per group)
    estimated_veh_hr = total_trajs / n_groups
    # Scale for SUMO (which distributes across OD pairs)
    flow_rate = max(int(estimated_veh_hr * 1.5), 600)

    logger.info(
        f"[{camera}] Observed: {total_trajs} trajectories across "
        f"{len(cam_samples)} lanes, {n_groups} groups → "
        f"estimated flow rate: {flow_rate} veh/hr"
    )
    return flow_rate


def figure_d4(
    model, dataset, device, embeddings_baseline, metrics_baseline,
    output_dir: Path, sumo_root=None,
):
    """Behavioral Geometry Synthesis Loop (Option A): behavioral metrics focus.

    End-to-end: real lanes → SUMO simulate (same geometry) → compare traffic
    metrics (speed, density, LOS). Demonstrates that the same road geometry
    produces consistent behavioral state across camera and simulation domains.

    Layout:
      Left:   Real camera lanes colored by SUMO-derived LOS
      Center: Before/After behavioral metrics comparison (speed, density, LOS)
      Right:  Per-lane LOS match table + cosine similarity
    """
    from src.bridge.reencoder import LaneReencoder
    from src.bridge.variant_comparator import VariantComparator

    # Pick camera with most lanes
    cam_counts = {}
    for s in dataset.samples:
        cam_counts.setdefault(s.camera, []).append(s)
    best_cam = max(cam_counts, key=lambda c: len(cam_counts[c]))
    cam_samples = [s for s in dataset.samples if s.camera == best_cam]

    reencoder = LaneReencoder(model, device)
    comparator = VariantComparator()

    # SUMO simulation with observed-trajectory-calibrated demand
    sumo_available = False
    reencoded_embeddings = {}
    sumo_metrics_by_lane = {}

    if sumo_root is not None:
        try:
            from src.bridge.sumo_runner import (
                run_sumo_simulation, _parse_net_lanes, LOS_THRESHOLDS,
            )
            from src.bridge.trajectory_extractor import extract_encoder_inputs
            from pathlib import Path as P

            sumo_root_path = P(sumo_root)
            net_path = sumo_root_path / best_cam / "osm.net.xml"

            if not net_path.exists():
                raise FileNotFoundError(f"No network at {net_path}")

            # Estimate flow rate from observed trajectories
            flow_rate = _estimate_flow_from_observed(dataset, best_cam)

            # Run SUMO with calibrated demand → baseline LOS
            logger.info(f"[D4] Running SUMO for {best_cam} (flow={flow_rate})...")
            metrics_list = run_sumo_simulation(
                best_cam, sumo_root_path,
                collect_fcd=True, flow_rate=flow_rate,
            )

            if metrics_list:
                for m in metrics_list:
                    sumo_metrics_by_lane[m.lane_id] = m

                # Re-encode FCD through encoder
                fcd_path = sumo_root_path / best_cam / "fcd_output.xml"
                lane_geoms = _parse_net_lanes(net_path)

                if fcd_path.exists():
                    traj_data = extract_encoder_inputs(
                        fcd_path, lane_geoms, polyline_k=16,
                    )
                    if traj_data:
                        norm_geoms = {
                            lid: geom for lid, geom in lane_geoms.items()
                            if lid in traj_data
                        }
                        reencoded_embeddings = reencoder.reencode(
                            traj_data, norm_geoms,
                        )
                        sumo_available = bool(reencoded_embeddings)
                        logger.info(
                            f"[D4] SUMO re-encoding: {len(reencoded_embeddings)} lanes, "
                            f"{len(metrics_list)} with metrics"
                        )

        except Exception as e:
            logger.warning(f"[D4] SUMO failed: {e}")
            import traceback
            traceback.print_exc()

    if not sumo_available:
        logger.error("[D4] SUMO required for D4. Pass --sumo-root.")
        return

    # Build real embeddings (camera-side, best_cam only)
    real_embeddings = {}
    for i, s in enumerate(dataset.samples):
        if s.camera == best_cam:
            real_embeddings[s.lane_key] = embeddings_baseline[i]

    # NN compare
    report = comparator.compare_nearest(
        real_embeddings, reencoded_embeddings, variant_name="SUMO",
    )

    # Map each real lane to its best-matched SUMO lane for metrics lookup
    real_to_sumo = {}  # real_key → sumo LaneMetrics
    for r in report.lane_results:
        real_key = r.lane_id.split("↔")[0]
        sumo_key = r.lane_id.split("↔")[1] if "↔" in r.lane_id else None
        if sumo_key and sumo_key in sumo_metrics_by_lane:
            real_to_sumo[real_key] = sumo_metrics_by_lane[sumo_key]

    # Camera-side bridge metrics (already computed, passed in as metrics_baseline)
    cam_bridge_metrics = {}
    for i, s in enumerate(dataset.samples):
        if s.camera == best_cam:
            cam_bridge_metrics[s.lane_key] = metrics_baseline[i]

    # --- Build figure: 3 panels ---
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2])
    ax_real = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])
    ax_table = fig.add_subplot(gs[2])

    # ===== Left: Real lanes colored by SUMO-derived LOS =====
    ax_real.set_title(f"{best_cam}: Observed Lanes (SUMO LOS)", fontsize=10)
    ax_real.set_facecolor("#1a1a2e")
    for sample in cam_samples:
        geom = sample.geometry
        sumo_m = real_to_sumo.get(sample.lane_key)
        los = sumo_m.los if sumo_m else "?"
        color = LOS_COLORS.get(los, "#808080")
        ax_real.plot(geom[:, 0], geom[:, 1], color=color, linewidth=3, alpha=0.8)
        mid = len(geom) // 2
        ax_real.annotate(
            f"LOS {los}",
            (geom[mid, 0], geom[mid, 1]),
            color="white", fontsize=7, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
        )
    ax_real.invert_yaxis()
    ax_real.set_aspect("equal")
    ax_real.axis("off")

    # ===== Center: Before/After behavioral metrics (grouped bars) =====
    # Aggregate camera-observed vs SUMO-simulated metrics
    cam_speeds = []
    cam_densities = []
    sumo_speeds = []
    sumo_densities = []
    for sample in cam_samples:
        bm = cam_bridge_metrics.get(sample.lane_key)
        sm = real_to_sumo.get(sample.lane_key)
        if bm and sm:
            cam_speeds.append(bm.speed_mph)
            cam_densities.append(bm.density_veh_mi_ln)
            sumo_speeds.append(sm.speed_mph)
            sumo_densities.append(sm.density_veh_mi_ln)

    if cam_speeds:
        mean_cam_speed = np.mean(cam_speeds)
        mean_sumo_speed = np.mean(sumo_speeds)
        mean_cam_density = np.mean(cam_densities)
        mean_sumo_density = np.mean(sumo_densities)

        # Camera LOS distribution
        cam_los_dist = {}
        for sample in cam_samples:
            bm = cam_bridge_metrics.get(sample.lane_key)
            if bm:
                cam_los_dist[bm.los] = cam_los_dist.get(bm.los, 0) + 1
        # SUMO LOS distribution
        sumo_los_dist = {}
        for sample in cam_samples:
            sm = real_to_sumo.get(sample.lane_key)
            if sm:
                sumo_los_dist[sm.los] = sumo_los_dist.get(sm.los, 0) + 1

        metric_names = ["Speed\n(mph)", "Density\n(veh/mi/ln)"]
        cam_vals = [mean_cam_speed, mean_cam_density]
        sumo_vals = [mean_sumo_speed, mean_sumo_density]

        x = np.arange(len(metric_names))
        w = 0.3
        bars_cam = ax_metrics.bar(
            x - w / 2, cam_vals, w, label="Camera (observed)",
            color="#5b9bd5", edgecolor="white", linewidth=0.5,
        )
        bars_sumo = ax_metrics.bar(
            x + w / 2, sumo_vals, w, label="SUMO (simulated)",
            color="#ed7d31", edgecolor="white", linewidth=0.5,
        )

        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(metric_names, fontsize=10)
        ax_metrics.set_ylabel("Value", fontsize=10)
        ax_metrics.legend(fontsize=9, loc="upper right")
        ax_metrics.grid(axis="y", alpha=0.3)

        # Value labels on bars
        for bar, val in zip(bars_cam, cam_vals):
            ax_metrics.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#5b9bd5",
            )
        for bar, val in zip(bars_sumo, sumo_vals):
            ax_metrics.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#ed7d31",
            )

        # LOS distribution annotation
        cam_los_str = ", ".join(f"{k}:{v}" for k, v in sorted(cam_los_dist.items()))
        sumo_los_str = ", ".join(f"{k}:{v}" for k, v in sorted(sumo_los_dist.items()))
        los_match = cam_los_dist == sumo_los_dist
        match_color = "#2ecc71" if los_match else "#e67e22"
        ax_metrics.text(
            0.5, -0.12,
            f"Camera LOS: [{cam_los_str}]    SUMO LOS: [{sumo_los_str}]"
            f"    {'MATCH' if los_match else 'DIFFERS'}",
            transform=ax_metrics.transAxes, fontsize=9, ha="center",
            fontweight="bold", color=match_color,
        )

    ax_metrics.set_title("Camera vs SUMO: Aggregate Traffic Metrics", fontsize=10)

    # ===== Right: Per-lane table with LOS match + cosine similarity =====
    ax_table.axis("off")
    ax_table.set_title("Per-Lane Comparison", fontsize=10)

    # Build table data
    table_rows = []
    for sample in sorted(cam_samples, key=lambda s: s.lane_key):
        bm = cam_bridge_metrics.get(sample.lane_key)
        sm = real_to_sumo.get(sample.lane_key)
        cos_sim = None
        for r in report.lane_results:
            if r.lane_id.split("↔")[0] == sample.lane_key:
                cos_sim = r.cosine_similarity
                break

        parts = sample.lane_key.split("_")
        short_label = f"g{parts[-2]}_l{parts[-1]}"
        cam_los = bm.los if bm else "?"
        sumo_los = sm.los if sm else "?"
        los_match = cam_los == sumo_los
        table_rows.append([
            short_label,
            cam_los,
            sumo_los,
            "Y" if los_match else "N",
            f"{cos_sim:.3f}" if cos_sim is not None else "—",
        ])

    if table_rows:
        col_labels = ["Lane", "Camera\nLOS", "SUMO\nLOS", "Match", "Cos\nSim"]
        table = ax_table.table(
            cellText=table_rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        # Color cells: green for match, red for mismatch
        for i, row in enumerate(table_rows):
            match_cell = table[i + 1, 3]  # +1 for header row
            if row[3] == "Y":
                match_cell.set_facecolor("#d5f5e3")
            else:
                match_cell.set_facecolor("#fadbd8")
            # Color cos_sim by value
            try:
                cs = float(row[4])
                sim_cell = table[i + 1, 4]
                sim_cell.set_facecolor(plt.cm.RdYlGn(cs))
                sim_cell.set_text_props(
                    color="white" if cs < 0.3 else "black",
                )
            except ValueError:
                pass

        # Header styling
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#34495e")
            table[0, j].set_text_props(color="white", fontweight="bold")

    # Summary stats
    n_match = sum(1 for r in table_rows if r[3] == "Y")
    n_total = len(table_rows)
    ax_table.text(
        0.5, 0.02,
        f"LOS Match: {n_match}/{n_total} ({100*n_match/n_total:.0f}%)  |  "
        f"Mean cos_sim: {report.mean_cosine_similarity:.3f}",
        transform=ax_table.transAxes, fontsize=9, ha="center",
        fontweight="bold", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eee", alpha=0.9),
    )

    # Suptitle
    sumo_los_dist = {}
    for m in sumo_metrics_by_lane.values():
        sumo_los_dist[m.los] = sumo_los_dist.get(m.los, 0) + 1
    fig.suptitle(
        f"Behavioral Geometry Synthesis Loop — {best_cam}\n"
        f"Camera observed → SUMO (same geometry, calibrated demand) → "
        f"compare traffic metrics",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])

    path = output_dir / "D4_synthesis_loop.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)

    logger.info(comparator.format_report(report))
    logger.info(f"Saved D4 to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Digital Twin integration figures (D-series)"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="results/dt/figures")
    parser.add_argument(
        "--figures", nargs="+", default=["D1", "D2", "D4"],
        choices=["D1", "D2", "D3", "D4"],
    )
    parser.add_argument(
        "--diffusion-checkpoint", default=None,
        help="Diffusion model checkpoint (required for D3)",
    )
    parser.add_argument(
        "--relational-checkpoint", default=None,
        help="Relational diffusion model checkpoint (for D3 comparison)",
    )
    parser.add_argument(
        "--export-dir", default=None,
        help="Shared directory for twin JSON export (enables D2 export)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Export each D2 timestep with delay for live twin consumption",
    )
    parser.add_argument(
        "--lane-mapping", default=None,
        help="YAML mapping lane_key → sumo_lane_id",
    )
    parser.add_argument(
        "--calibration-points", default=None,
        help="CSV with pixel_x,pixel_y,sumo_x,sumo_y reference points",
    )
    parser.add_argument(
        "--sumo-root", default=None,
        help="Root directory with per-camera SUMO networks (for D4)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build exporter if export-dir specified
    exporter = None
    if args.export_dir:
        from src.bridge.dt_export import DigitalTwinExporter, SumoCalibration

        calibration = SumoCalibration()
        if args.calibration_points:
            import csv
            points = []
            with open(args.calibration_points) as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    points.append(tuple(float(v) for v in row[:4]))
            calibration = SumoCalibration(reference_points=points)

        lane_mapping = {}
        if args.lane_mapping:
            import yaml
            with open(args.lane_mapping) as f:
                lane_mapping = yaml.safe_load(f) or {}

        exporter = DigitalTwinExporter(
            output_dir=args.export_dir,
            calibration=calibration,
            lane_mapping=lane_mapping,
        )
        logger.info(f"Twin export enabled → {args.export_dir}")

    # Load data for D1/D2/D3/D4
    needs_data = any(f in args.figures for f in ["D1", "D2", "D3", "D4"])
    if needs_data:
        model, config, dataset, device = _load_encoder(args)
        enc_output, batch, embeddings = _encode_all(model, dataset, device)
        metrics = _get_bridge_metrics(dataset, enc_output)

    if "D1" in args.figures:
        logger.info("Generating D1: Physical vs Digital side-by-side...")
        figure_d1(metrics, dataset, output_dir)

    if "D2" in args.figures:
        logger.info("Generating D2: Ephemeral event scenario...")
        figure_d2(
            model, dataset, device, metrics, embeddings,
            output_dir, exporter=exporter, live=args.live,
        )

    if "D3" in args.figures:
        logger.info("Generating D3: Edit-Type Failure Taxonomy...")
        figure_d3(
            model, dataset, device, output_dir,
            diffusion_checkpoint=args.diffusion_checkpoint,
            relational_checkpoint=args.relational_checkpoint,
        )

    if "D4" in args.figures:
        logger.info("Generating D4: Behavioral Geometry Synthesis Loop...")
        figure_d4(
            model, dataset, device, embeddings, metrics,
            output_dir, sumo_root=args.sumo_root,
        )

    # Final export: write current state for twin
    if exporter:
        anomaly_scores = np.zeros(len(metrics))
        exporter.write_full_update(
            metrics=metrics,
            anomaly_scores=anomaly_scores,
            embeddings=embeddings,
        )
        logger.info(f"Final state exported to {args.export_dir}")

    logger.info(f"All DT figures saved to {output_dir}")


if __name__ == "__main__":
    main()
