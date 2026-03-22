#!/usr/bin/env python3
"""Generate lane geometry generation figures.

Figures:
    G1 — Conditioned generation across diverse road sections (multi-scenario)
    G2 — Spec-conditioned generation (rightmost, leftmost, merge)
         + Independent vs relational conditioning comparison
    G3 — Generation quality metrics (diversity, coherence, chamfer, spec accuracy)

    Legacy G1/G3 (retrieval/constructive) retained for backward compat as _legacy_g1/_legacy_g3.

Usage:
    # Default figures (trains diffusion + generates):
    python scripts/generate_generation_figures.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --train-diffusion --diffusion-epochs 200

    # With pre-trained diffusion model:
    python scripts/generate_generation_figures.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --diffusion-checkpoint results/generation/figures/diffusion_model.pt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "p", "h"]


# ---------------------------------------------------------------------------
# Data loading (shared with taxonomy figures)
# ---------------------------------------------------------------------------

def _load_data(args):
    """Load encoder, encode all lanes, build retrieval index."""
    import yaml
    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import load_trained_encoder
    from src.generation.retrieval import build_retrieval_index

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

    index = build_retrieval_index(model, dataset, device, use_embeddings=True)

    return model, config, dataset, index, device


def _cluster_embeddings(embeddings: np.ndarray, method: str = "hdbscan",
                        n_clusters: int = 5) -> np.ndarray:
    """Cluster lane embeddings (same as taxonomy)."""
    if method == "hdbscan":
        try:
            from hdbscan import HDBSCAN
            clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
            labels = clusterer.fit_predict(embeddings)
            n_found = len(set(labels) - {-1})
            logger.info(f"HDBSCAN found {n_found} clusters ({(labels == -1).sum()} noise)")
            return labels
        except ImportError:
            logger.warning("hdbscan not installed, falling back to k-means")
            method = "kmeans"

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    logger.info(f"K-means: {n_clusters} clusters")
    return labels


# ---------------------------------------------------------------------------
# Figure G1: Retrieved geometries per cluster
# ---------------------------------------------------------------------------

def _legacy_figure_g1(index, labels: np.ndarray, output_dir: Path,
              k: int = 3, dataset=None):
    """For each cluster centroid, retrieve top-k geometries and plot them.

    Shows the retrieval quality: do similar embeddings yield similar shapes?
    """
    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        logger.warning("No clusters found, skipping G1")
        return

    n_clusters = len(unique_labels)
    fig, axes = plt.subplots(
        n_clusters, k + 1, figsize=(3.5 * (k + 1), 3 * n_clusters),
        squeeze=False,
    )

    W, H = 1.0, 1.0  # normalized space
    if dataset:
        W, H = dataset.image_wh

    for row, lbl in enumerate(unique_labels):
        cluster_mask = labels == lbl
        cluster_indices = np.where(cluster_mask)[0]

        # Compute cluster centroid
        centroid = index.embeddings[cluster_mask].mean(axis=0)

        # Retrieve top-k from the full index
        result = index.retrieve(centroid, k=k)

        # Plot centroid info
        ax = axes[row, 0]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # image coords: y increases downward
        ax.set_aspect("equal")

        # Draw all lanes in the cluster (faint)
        color = _CLUSTER_COLORS[lbl % len(_CLUSTER_COLORS)]
        for ci in cluster_indices:
            geom = index.geometries[ci]
            ax.plot(geom[:, 0], geom[:, 1], color=color, alpha=0.3, linewidth=1)

        ax.set_title(f"Cluster {lbl} (n={len(cluster_indices)})", fontsize=9)
        ax.set_xlabel("x (norm)")
        ax.set_ylabel("y (norm)")

        # Plot retrieved lanes
        for col, (ret_idx, sim) in enumerate(
            zip(result.retrieved_indices, result.similarities)
        ):
            ax = axes[row, col + 1]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect("equal")

            geom = index.geometries[ret_idx]
            ret_label = labels[ret_idx]
            ret_color = (
                _CLUSTER_COLORS[ret_label % len(_CLUSTER_COLORS)]
                if ret_label >= 0 else "#999999"
            )

            ax.plot(geom[:, 0], geom[:, 1], color=ret_color, linewidth=2)
            ax.scatter(geom[0, 0], geom[0, 1], c="green", s=30, zorder=5,
                       label="start")
            ax.scatter(geom[-1, 0], geom[-1, 1], c="red", s=30, zorder=5,
                       label="end")

            lane_key = index.lane_keys[ret_idx]
            ax.set_title(f"#{col+1}: {lane_key}\nsim={sim:.3f}", fontsize=7)

    fig.suptitle("Retrieved Geometries per Cluster Centroid", fontsize=13)
    fig.tight_layout()

    path = output_dir / "reconstruction_retrieval_per_cluster.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved reconstruction figure to {path}")


# ---------------------------------------------------------------------------
# Figure G3: Constructive demo
# ---------------------------------------------------------------------------

def _legacy_figure_g3(
    index,
    labels: np.ndarray,
    dataset,
    output_dir: Path,
    n_queries: int = 4,
    k: int = 3,
    trainer=None,
):
    """Query lane → behavioral profile + retrieved candidates side by side.

    Picks diverse query lanes and shows: query geometry, traj stats, retrieved
    candidates with similarities, and (optionally) diffusion-generated output.
    """
    annot_dir = Path(
        dataset.config.get("data", {}).get("annotation_dir", "../dataset/preprocess")
    )
    W, H = dataset.image_wh

    # Select diverse queries: one per cluster
    unique_labels = sorted(set(labels) - {-1})
    query_indices = []
    for lbl in unique_labels:
        cluster_mask = labels == lbl
        cluster_indices = np.where(cluster_mask)[0]
        # Pick the lane closest to cluster centroid
        centroid = index.embeddings[cluster_mask].mean(axis=0)
        dists = np.linalg.norm(
            index.embeddings[cluster_indices] - centroid, axis=1
        )
        best = cluster_indices[dists.argmin()]
        query_indices.append(best)
        if len(query_indices) >= n_queries:
            break

    n_show = len(query_indices)
    n_cols = k + 2  # query + k retrieved + warm-start
    fig, axes = plt.subplots(n_show, n_cols, figsize=(3.5 * n_cols, 3.5 * n_show),
                              squeeze=False)

    for row, q_idx in enumerate(query_indices):
        sample = dataset.samples[q_idx]
        result = index.retrieve_by_index(q_idx, k=k, cross_camera=False)
        q_label = labels[q_idx]
        q_color = (
            _CLUSTER_COLORS[q_label % len(_CLUSTER_COLORS)]
            if q_label >= 0 else "#999999"
        )

        # Column 0: Query lane on camera frame
        ax = axes[row, 0]
        _draw_lane_panel(
            ax, sample, annot_dir, W, H, q_color,
            title=f"Query: {index.lane_keys[q_idx]}\n"
                  f"cluster={q_label}",
        )

        # Columns 1..k: Retrieved candidates
        for col, (ret_idx, sim) in enumerate(
            zip(result.retrieved_indices, result.similarities)
        ):
            ax = axes[row, col + 1]
            ret_sample = dataset.samples[ret_idx]
            ret_label = labels[ret_idx]
            ret_color = (
                _CLUSTER_COLORS[ret_label % len(_CLUSTER_COLORS)]
                if ret_label >= 0 else "#999999"
            )
            _draw_lane_panel(
                ax, ret_sample, annot_dir, W, H, ret_color,
                title=f"#{col+1}: {index.lane_keys[ret_idx]}\n"
                      f"sim={sim:.3f}, cluster={ret_label}",
            )

        # Last column: Warm-start interpolation
        ax = axes[row, -1]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ws = result.warm_start
        ax.plot(ws[:, 0], ws[:, 1], color=q_color, linewidth=2.5, label="warm-start")
        # Overlay query geometry for comparison
        q_geom = index.geometries[q_idx]
        ax.plot(q_geom[:, 0], q_geom[:, 1], color="black", linewidth=1,
                linestyle="--", alpha=0.5, label="query (ref)")
        ax.legend(fontsize=6, loc="lower right")
        ax.set_title("Warm-start\n(interpolated)", fontsize=8)

    fig.suptitle("Query → Retrieved Candidates → Warm Start", fontsize=13)
    fig.tight_layout()

    path = output_dir / "reconstruction_constructive_demo.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved reconstruction figure to {path}")


def _draw_lane_panel(ax, sample, annot_dir: Path, W: int, H: int,
                     color: str, title: str = ""):
    """Draw a lane on its camera frame or in normalized coords."""
    frame_path = annot_dir / sample.camera / "last_frame.npy"
    if frame_path.exists():
        frame = np.load(str(frame_path))
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame, extent=[0, 1, 1, 0])
    else:
        ax.set_facecolor("#f0f0f0")

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    pts = sample.geometry
    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.5)
    ax.scatter(pts[0, 0], pts[0, 1], c="green", s=25, zorder=5)
    ax.scatter(pts[-1, 0], pts[-1, 1], c="red", s=25, zorder=5)

    # Show traj stats as text
    stats = sample.traj_stats
    stats_text = (
        f"spd={stats[0]:.4f} crv={stats[1]:.4f}\n"
        f"lat={stats[2]:.4f} cnt={stats[3]:.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Figure G2: Multi-scenario generation showcase
# ---------------------------------------------------------------------------

def figure_g1(
    index,
    dataset,
    model,
    device,
    output_dir: Path,
    trainer=None,
    relational_trainer=None,
    args=None,
):
    """Auto-generated multi-scenario directed generation showcase.

    Picks diverse camera/group locations and generates rightmost, leftmost,
    and merge lanes for each — demonstrating generation across varied road
    sections without user specification.

    When relational_trainer is provided, generates both independent and
    relational versions as separate figures.

    Layout: rows = camera/group locations, cols = context | rightmost | leftmost | merge
    """
    from src.generation.spec import LaneSpecification, SpecEmbeddingResolver
    from src.generation.directed import DirectedLaneGenerator

    if trainer is None:
        logger.warning("G1 requires a trained diffusion model, skipping")
        return

    resolver = SpecEmbeddingResolver(index, dataset)
    generator = DirectedLaneGenerator(
        resolver, trainer, encoder=model, dataset=dataset, device=device,
        relational_trainer=relational_trainer,
    )

    annot_dir = Path(
        dataset.config.get("data", {}).get("annotation_dir", "../dataset/preprocess")
    )

    # Find all camera/group combinations, classify by merge/diverge presence
    camera_groups = {}
    for sample in dataset.samples:
        key = (sample.camera, sample.group_id)
        camera_groups.setdefault(key, []).append(sample)

    def _has_successor(samples):
        return any(s.role.has_successor for s in samples)

    # Split into groups with and without merge/diverge
    with_merge = {k: v for k, v in camera_groups.items() if _has_successor(v)}
    without_merge = {k: v for k, v in camera_groups.items() if not _has_successor(v)}

    # Pick 2 groups without merge (to show adding merge), 2 with merge
    used_cameras = set()
    selected_keys = []

    # First: pick groups WITHOUT merge/diverge (shows adding merge to plain road)
    for key in sorted(without_merge.keys(), key=lambda k: len(without_merge[k]), reverse=True):
        cam, gid = key
        if cam not in used_cameras and len(without_merge[key]) >= 2:
            selected_keys.append(key)
            used_cameras.add(cam)
            if len(selected_keys) >= 2:
                break

    # Then: pick groups WITH merge/diverge (shows generating all types)
    for key in sorted(with_merge.keys(), key=lambda k: len(with_merge[k]), reverse=True):
        cam, gid = key
        if cam not in used_cameras:
            selected_keys.append(key)
            used_cameras.add(cam)
            if len(selected_keys) >= 4:
                break

    # Fill remaining slots from any group
    if len(selected_keys) < 4:
        for key in sorted(camera_groups.keys(), key=lambda k: len(camera_groups[k]), reverse=True):
            cam, gid = key
            if cam not in used_cameras:
                selected_keys.append(key)
                used_cameras.add(cam)
                if len(selected_keys) >= 4:
                    break

    if not selected_keys:
        logger.warning("No camera groups found for G2, skipping")
        return

    lane_types = [
        ("Rightmost", LaneSpecification.rightmost),
        ("Leftmost", LaneSpecification.leftmost),
        ("Merge", LaneSpecification.merge_lane),
    ]

    # Generate both independent and relational versions when available
    versions = [("independent", False)]
    if relational_trainer is not None:
        versions.append(("relational", True))

    for version_label, use_relational in versions:
        n_rows = len(selected_keys)
        n_cols = 1 + len(lane_types)  # context + lane types
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows),
                                  squeeze=False)

        for row, (camera, group_id) in enumerate(selected_keys):
            group_samples = camera_groups[(camera, group_id)]
            group_geoms = index.geometries[
                [i for i, s in enumerate(dataset.samples)
                 if s.camera == camera and s.group_id == group_id]
            ]

            # Col 0: Context — existing lanes on camera frame
            ax = axes[row, 0]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect("equal")

            frame_path = annot_dir / camera / "last_frame.npy"
            frame = None
            if frame_path.exists():
                frame = np.load(str(frame_path))
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.5)

            for g in group_geoms:
                ax.plot(g[:, 0], g[:, 1], color="#888888", linewidth=1.5, alpha=0.7)
            has_merge = _has_successor(group_samples)
            merge_tag = "has merge" if has_merge else "no merge"
            ax.set_title(f"{camera} g{group_id}\n{len(group_samples)} lanes ({merge_tag})",
                         fontsize=8)
            ax.axis("off")

            # Cols 1-3: Generate each lane type
            for col, (type_label, spec_factory) in enumerate(lane_types):
                ax = axes[row, col + 1]
                ax.set_xlim(0, 1)
                ax.set_ylim(1, 0)
                ax.set_aspect("equal")

                spec = spec_factory(camera, group_id)
                if use_relational:
                    result = generator.generate_relational(spec, n_candidates=10)
                else:
                    result = generator.generate(spec, n_candidates=10)

                # Show camera frame
                if frame is not None:
                    ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.3)

                # Existing lanes (faint)
                for g in group_geoms:
                    ax.plot(g[:, 0], g[:, 1], color="#cccccc", linewidth=1, alpha=0.5)

                # Best generated lane
                ax.plot(result.best[:, 0], result.best[:, 1], color="#e41a1c",
                        linewidth=3, label="generated")
                ax.scatter(result.best[0, 0], result.best[0, 1], c="green", s=30, zorder=5)
                ax.scatter(result.best[-1, 0], result.best[-1, 1], c="red", s=30, zorder=5)

                best_score = result.scores.max()
                ax.set_title(f"{type_label}\nscore={best_score:.3f}", fontsize=8,
                             fontweight="bold")
                ax.axis("off")

        fig.suptitle(
            f"Conditioned Lane Generation ({version_label})\n"
            "across diverse road sections",
            fontsize=13,
        )
        fig.tight_layout()

        suffix = "" if version_label == "independent" else f"_{version_label}"
        path = output_dir / f"g1_multi_scenario{suffix}.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved G1 ({version_label}) to {path}")


# ---------------------------------------------------------------------------
# Diffusion training
# ---------------------------------------------------------------------------

def _train_diffusion(index, model, dataset, device, args):
    """Train diffusion model in canonical space with augmented geolane data.

    All geometries are converted to canonical space (centered, aligned, unit-length)
    before training. 10x augmentation (rotation, lateral jitter, stretch, curvature
    noise) provides sufficient training data for the small MLP denoiser.

    FiLM conditioning uses encoder embeddings from the trained lane encoder.
    """
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer
    from src.generation.augment import batch_to_canonical, augment_geometries

    K = index.K
    geom_dim = K * 2
    cond_dim = index.D

    denoiser = LaneDenoiser(
        geom_dim=geom_dim,
        t_dim=64,
        cond_dim=cond_dim,
        hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    trainer = LaneDiffusionTrainer(
        denoiser, schedule,
        lr=args.diffusion_lr,
        device=str(device),
    )

    # ── Stage 1: Pretrain on OpenLaneV2 (unconditional, geometry only) ──
    openlane_path = getattr(args, "openlane_data", None)
    pretrain_epochs = getattr(args, "pretrain_epochs", 500)
    if openlane_path:
        openlane_npz = Path(openlane_path)
        if openlane_npz.suffix == ".npz":
            # Pre-extracted canonical lanes
            ol_data = np.load(str(openlane_npz))
            ol_canonicals = ol_data["canonicals"]
        else:
            # Extract from raw OpenLaneV2 pickle directory
            from src.generation.openlane_preprocess import extract_openlane_geometries
            ol_geoms = extract_openlane_geometries(
                openlane_path, split="train",
                max_lanes=getattr(args, "openlane_max_lanes", 20000),
            )
            ol_canonicals, _, _, _ = batch_to_canonical(ol_geoms)
            ol_canonicals = ol_canonicals.astype(np.float32)

        ol_flat = torch.tensor(
            ol_canonicals.reshape(len(ol_canonicals), -1), dtype=torch.float32,
        )
        logger.info(
            f"Stage 1: Pretraining on {len(ol_canonicals)} OpenLaneV2 canonical lanes "
            f"(unconditional), {pretrain_epochs} epochs"
        )
        for epoch in range(1, pretrain_epochs + 1):
            loss = trainer.train_epoch(
                ol_flat, cond_embeddings=None,
                batch_size=args.diffusion_batch_size,
            )
            if epoch % 50 == 0 or epoch == 1:
                logger.info(f"  [Stage 1] Epoch {epoch}/{pretrain_epochs}: loss={loss:.6f}")

        logger.info("Stage 1 complete. Starting Stage 2 (conditional fine-tuning).")

    # ── Stage 2: Fine-tune with behavioral conditioning on geolane data ──
    canonical_geoms, _, _, _ = batch_to_canonical(index.geometries)
    aug_geoms, aug_embeds = augment_geometries(
        canonical_geoms.astype(np.float32), index.embeddings, factor=10,
    )
    geometries_flat = torch.tensor(
        aug_geoms.reshape(len(aug_geoms), -1), dtype=torch.float32
    )
    cond_embeddings = torch.tensor(aug_embeds, dtype=torch.float32)

    stage_label = "Stage 2" if openlane_path else "Training"
    logger.info(
        f"{stage_label}: diffusion on {len(aug_geoms)} augmented lanes "
        f"(canonical space, {index.N} original x10), "
        f"{args.diffusion_epochs} epochs, lr={args.diffusion_lr:.1e}"
    )
    best_loss = float("inf")
    for epoch in range(1, args.diffusion_epochs + 1):
        loss = trainer.train_epoch(
            geometries_flat, cond_embeddings,
            batch_size=args.diffusion_batch_size,
        )
        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                f"  [{stage_label}] Epoch {epoch}/{args.diffusion_epochs}: loss={loss:.6f}"
            )
        if loss < best_loss:
            best_loss = loss

    # Save
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(str(save_dir / "diffusion_model.pt"))
    logger.info(f"Saved diffusion model (best_loss={best_loss:.6f})")

    return trainer


def _train_relational(index, model, dataset, device, args):
    """Train relational diffusion model with neighbor context.

    Builds relational pairs (successor + adjacent) from annotation
    relationships, augments with joint transformations, and trains
    the RelationalLaneDenoiser with FiLM conditioning on behavioral
    + relational embeddings.
    """
    from src.generation.relational_diffusion import (
        RelationalLaneDenoiser, RelationalDiffusionTrainer,
    )
    from src.generation.diffusion import DDPMSchedule
    from src.generation.relational_pairs import (
        build_relational_pairs, augment_relational_pairs,
    )

    K = index.K
    geom_dim = K * 2
    cond_dim = index.D

    # Build relational pairs from annotations
    annot_dir = Path(
        dataset.config.get("data", {}).get("annotation_dir", "../dataset/preprocess")
    )
    pairs = build_relational_pairs(dataset, index, str(annot_dir))
    logger.info(f"Built {len(pairs)} relational pairs from annotations")

    if len(pairs) == 0:
        logger.warning("No relational pairs found, skipping relational training")
        return None

    # Augment pairs — returns a dict of numpy arrays
    augment_factor = getattr(args, "augment_factor", 10)
    no_relation_ratio = getattr(args, "no_relation_ratio", 0.3)
    aug_data = augment_relational_pairs(
        pairs, factor=augment_factor, no_relation_ratio=no_relation_ratio,
    )
    n_aug = len(aug_data["geometries"])
    logger.info(
        f"Augmented to {n_aug} samples "
        f"(factor={augment_factor}, no_relation_ratio={no_relation_ratio})"
    )

    if n_aug == 0:
        logger.warning("No augmented samples, skipping relational training")
        return None

    # Prepare tensors from augmented dict
    target_geoms = torch.tensor(aug_data["geometries"], dtype=torch.float32)
    cond_embeddings = torch.tensor(aug_data["cond_embeddings"], dtype=torch.float32)
    neighbor_geoms = torch.tensor(aug_data["neighbor_geoms"], dtype=torch.float32)
    merge_points = torch.tensor(aug_data["merge_points"], dtype=torch.float32)
    offsets = torch.tensor(aug_data["offsets"], dtype=torch.float32)
    has_relations = torch.tensor(aug_data["has_relations"], dtype=torch.float32)

    # Build model
    denoiser = RelationalLaneDenoiser(
        geom_dim=geom_dim,
        t_dim=64,
        cond_dim=cond_dim,
        rel_dim=64,
        hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    rel_trainer = RelationalDiffusionTrainer(
        denoiser, schedule,
        lr=getattr(args, "diffusion_lr", 1e-3),
        device=str(device),
    )

    # ── Stage 1: Pretrain relational model on OpenLaneV2 (unconditional) ──
    # Pretrain the FULL relational architecture with zeroed conditioning,
    # so 100% of weights (including FiLM projections) get a geometry prior.
    openlane_path = getattr(args, "openlane_data", None)
    pretrain_epochs = getattr(args, "pretrain_epochs", 500)
    if openlane_path:
        from src.generation.augment import batch_to_canonical
        openlane_npz = Path(openlane_path)
        if openlane_npz.suffix == ".npz":
            ol_data = np.load(str(openlane_npz))
            ol_canonicals = ol_data["canonicals"]
        else:
            from src.generation.openlane_preprocess import extract_openlane_geometries
            ol_geoms = extract_openlane_geometries(
                openlane_path, split="train",
                max_lanes=getattr(args, "openlane_max_lanes", 20000),
            )
            ol_canonicals, _, _, _ = batch_to_canonical(ol_geoms)
            ol_canonicals = ol_canonicals.astype(np.float32)

        n_ol = len(ol_canonicals)
        ol_flat = torch.tensor(
            ol_canonicals.reshape(n_ol, -1), dtype=torch.float32,
        )
        # Zero conditioning: behavioral=0, neighbor=0, merge=0, offset=0, has_relation=0
        ol_cond = torch.zeros(n_ol, cond_dim)
        ol_neighbor = torch.zeros(n_ol, geom_dim)
        ol_merge = torch.zeros(n_ol, 1)
        ol_offset = torch.zeros(n_ol, 1)
        ol_has_rel = torch.zeros(n_ol, 1)

        logger.info(
            f"Stage 1: Pretraining relational model on {n_ol} OpenLaneV2 lanes "
            f"(unconditional, all zeros), {pretrain_epochs} epochs"
        )
        for epoch in range(1, pretrain_epochs + 1):
            loss = rel_trainer.train_epoch(
                ol_flat, ol_cond,
                ol_neighbor, ol_merge, ol_offset, ol_has_rel,
                batch_size=getattr(args, "diffusion_batch_size", 32),
            )
            if epoch % 50 == 0 or epoch == 1:
                logger.info(f"  [Stage 1 relational] Epoch {epoch}/{pretrain_epochs}: loss={loss:.6f}")

        logger.info("Stage 1 relational complete. Starting Stage 2 (relational fine-tuning).")

    n_epochs = getattr(args, "relational_epochs", 1000)
    stage_label = "Stage 2" if openlane_path else "Training"
    logger.info(
        f"{stage_label}: relational diffusion on {n_aug} samples, "
        f"{n_epochs} epochs"
    )
    best_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
        loss = rel_trainer.train_epoch(
            target_geoms, cond_embeddings,
            neighbor_geoms, merge_points, offsets, has_relations,
            batch_size=getattr(args, "diffusion_batch_size", 32),
        )
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch}/{n_epochs}: loss={loss:.6f}")
        if loss < best_loss:
            best_loss = loss

    # Save
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    rel_trainer.save(str(save_dir / "relational_diffusion_model.pt"))
    logger.info(f"Saved relational diffusion model (best_loss={best_loss:.6f})")

    return rel_trainer


def _load_relational(path: str, index, device):
    """Load a pre-trained relational diffusion model."""
    from src.generation.relational_diffusion import (
        RelationalLaneDenoiser, RelationalDiffusionTrainer,
    )
    from src.generation.diffusion import DDPMSchedule

    K = index.K
    geom_dim = K * 2
    cond_dim = index.D

    denoiser = RelationalLaneDenoiser(
        geom_dim=geom_dim, t_dim=64, cond_dim=cond_dim,
        rel_dim=64, hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    rel_trainer = RelationalDiffusionTrainer(
        denoiser, schedule, device=str(device),
    )
    rel_trainer.load(path)
    return rel_trainer


# ---------------------------------------------------------------------------
# Figure G4: User-directed generation demo
# ---------------------------------------------------------------------------

def figure_g2(
    index,
    dataset,
    model,
    device: torch.device,
    output_dir: Path,
    trainer=None,
    args=None,
    relational_trainer=None,
):
    """Directed generation: spec → candidates → best.

    When --spec or --lane-type is given, generates those specific specs.
    Otherwise defaults to rightmost/leftmost/merge for the best camera/group.

    When relational_trainer is provided, generates both independent and
    relational versions as separate figures.
    """
    from src.generation.spec import LaneSpecification, BehaviorPrefix, SpecEmbeddingResolver
    from src.generation.directed import DirectedLaneGenerator

    if trainer is None:
        logger.warning("G2 requires a trained diffusion model, skipping")
        return

    resolver = SpecEmbeddingResolver(index, dataset)
    generator = DirectedLaneGenerator(
        resolver, trainer, encoder=model, dataset=dataset, device=device,
        relational_trainer=relational_trainer,
    )

    # Find a camera/group that has multiple lanes
    camera_groups = {}
    for sample in dataset.samples:
        key = (sample.camera, sample.group_id)
        camera_groups.setdefault(key, []).append(sample)

    # Resolve target camera/group
    if args and args.camera and args.group_id is not None:
        target_camera, target_group = args.camera, args.group_id
        if (target_camera, target_group) not in camera_groups:
            logger.warning(
                f"Camera={target_camera} group={args.group_id} not found. "
                f"Available: {list(camera_groups.keys())}"
            )
            best_key = max(camera_groups, key=lambda k: len(camera_groups[k]))
            target_camera, target_group = best_key
    else:
        best_key = max(camera_groups, key=lambda k: len(camera_groups[k]))
        target_camera, target_group = best_key

    logger.info(f"Directed generation target: {target_camera} group {target_group}")

    # Build behavioral prefix from CLI args (if any)
    behavior = None
    if args and (args.mean_speed is not None or args.mean_curvature is not None
                 or args.mean_lateral_offset is not None):
        behavior = BehaviorPrefix(
            mean_speed=args.mean_speed,
            mean_curvature=args.mean_curvature,
            mean_lateral_offset=args.mean_lateral_offset,
        )

    # Build specs
    if args and args.spec:
        # Natural language spec
        specs = [
            (args.spec, LaneSpecification.from_natural_language(
                args.spec, camera=target_camera, group_id=target_group,
            )),
        ]
    elif args and args.lane_type:
        # Explicit lane types
        type_map = {
            "rightmost": ("Rightmost lane", LaneSpecification.rightmost),
            "leftmost": ("Leftmost lane", LaneSpecification.leftmost),
            "merge": ("Merge lane", LaneSpecification.merge_lane),
        }
        specs = []
        for lt in args.lane_type:
            label, factory = type_map[lt]
            spec = factory(target_camera, target_group)
            if behavior is not None:
                spec.behavior = behavior
            specs.append((label, spec))
    else:
        # Default: all three types
        specs = [
            ("Rightmost lane", LaneSpecification.rightmost(target_camera, target_group)),
            ("Leftmost lane", LaneSpecification.leftmost(target_camera, target_group)),
            ("Merge lane", LaneSpecification.merge_lane(target_camera, target_group)),
        ]
        if behavior is not None:
            for _, spec in specs:
                spec.behavior = behavior

    # Append role replacement specs if requested
    if args and args.replace_role:
        for rr in args.replace_role:
            parts = rr.split(":")
            if len(parts) != 2:
                logger.warning(f"Invalid --replace-role format '{rr}', expected CLS:ROLE")
                continue
            cls_id, role = int(parts[0]), parts[1]
            spec = LaneSpecification.replace_role(
                cls_id, role, camera=target_camera, group_id=target_group,
            )
            specs.append((f"Replace cls={cls_id} → {role}", spec))

    # Collect lane removals (no generation needed, just visualization)
    remove_cls_ids = []
    if args and args.remove_lane:
        remove_cls_ids = list(args.remove_lane)

    # Target lane count (regenerate entire group)
    target_lanes = args.target_lanes if args else None
    has_target_lanes = target_lanes is not None

    n_candidates = args.n_candidates if args else 5

    annot_dir = Path(
        dataset.config.get("data", {}).get("annotation_dir", "../dataset/preprocess")
    )

    # Generate both independent and relational versions when available
    versions = [("independent", False)]
    if relational_trainer is not None:
        versions.append(("relational", True))

    for version_label, use_relational in versions:
        _g2_render(
            generator, specs, remove_cls_ids, has_target_lanes, target_lanes,
            n_candidates, annot_dir, target_camera, target_group,
            dataset, index, use_relational, version_label, output_dir,
        )


def _g2_render(
    generator, specs, remove_cls_ids, has_target_lanes, target_lanes,
    n_candidates, annot_dir, target_camera, target_group,
    dataset, index, use_relational, version_label, output_dir,
):
    """Render a single G2 figure (independent or relational)."""
    n_rows = len(specs) + len(remove_cls_ids) + (1 if has_target_lanes else 0)
    n_cols = 4  # context | anchor | candidates | best
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              squeeze=False)

    for row, (label, spec) in enumerate(specs):
        if use_relational:
            result = generator.generate_relational(spec, n_candidates=n_candidates)
        else:
            result = generator.generate(spec, n_candidates=n_candidates)

        # Col 0: Group context (all existing lanes)
        ax = axes[row, 0]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")

        # Try to show camera frame
        frame_path = annot_dir / target_camera / "last_frame.npy"
        if frame_path.exists():
            frame = np.load(str(frame_path))
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.5)

        # Draw existing group lanes
        if result.spatial_context is not None:
            for g in result.spatial_context.group_geometries:
                ax.plot(g[:, 0], g[:, 1], color="#888888", linewidth=1.5, alpha=0.7)
        ax.set_title(f"{label}\n{target_camera} g{target_group}", fontsize=8)

        # Col 1: Anchor geometry
        ax = axes[row, 1]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        if result.spatial_context is not None:
            anchor = result.spatial_context.anchor_geometry
            ax.plot(anchor[:, 0], anchor[:, 1], color="#377eb8", linewidth=2,
                    label="anchor")
            ax.legend(fontsize=7)
        ax.set_title(f"Spatial anchor\n({result.spatial_context.anchor_side if result.spatial_context else 'N/A'})",
                     fontsize=8)

        # Col 2: All candidates
        ax = axes[row, 2]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        # Draw existing lanes faintly
        if result.spatial_context is not None:
            for g in result.spatial_context.group_geometries:
                ax.plot(g[:, 0], g[:, 1], color="#cccccc", linewidth=1, alpha=0.5)
        # Draw candidates
        for i, (cand, score) in enumerate(zip(result.candidates, result.scores)):
            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            ax.plot(cand[:, 0], cand[:, 1], color=color, linewidth=1.5,
                    alpha=0.7, label=f"c{i} ({score:.2f})")
        ax.legend(fontsize=5, loc="lower right")
        ax.set_title(f"{n_candidates} candidates", fontsize=8)

        # Col 3: Best candidate
        ax = axes[row, 3]
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        if frame_path.exists():
            ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.3)
        # Existing lanes
        if result.spatial_context is not None:
            for g in result.spatial_context.group_geometries:
                ax.plot(g[:, 0], g[:, 1], color="#888888", linewidth=1, alpha=0.5)
        # Best
        ax.plot(result.best[:, 0], result.best[:, 1], color="#e41a1c",
                linewidth=3, label="generated")
        ax.scatter(result.best[0, 0], result.best[0, 1], c="green", s=30, zorder=5)
        ax.scatter(result.best[-1, 0], result.best[-1, 1], c="red", s=30, zorder=5)
        best_score = result.scores.max()
        ax.legend(fontsize=7)
        ax.set_title(f"Best (score={best_score:.3f})", fontsize=8)

    # --- Lane removal rows (no generation, just show group with lane excluded) ---
    if remove_cls_ids:
        # Build a map: cls_id → (geometry, sample) for the target group
        group_samples = [
            s for s in dataset.samples
            if s.camera == target_camera and s.group_id == target_group
        ]
        cls_to_geom = {}
        for s in group_samples:
            idx_in_index = dataset.samples.index(s)
            if idx_in_index < len(index.geometries):
                cls_to_geom[s.cls_id] = index.geometries[idx_in_index]

        all_geoms = list(cls_to_geom.items())  # [(cls_id, geom), ...]

        for ri, rm_cls in enumerate(remove_cls_ids):
            row = len(specs) + ri

            # Col 0: Original group (highlight removed lane in red)
            ax = axes[row, 0]
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            frame_path = annot_dir / target_camera / "last_frame.npy"
            if frame_path.exists():
                frame = np.load(str(frame_path))
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.5)
            for cid, g in all_geoms:
                if cid == rm_cls:
                    ax.plot(g[:, 0], g[:, 1], color="#e41a1c", linewidth=2.5,
                            alpha=0.9, label=f"cls={cid} (remove)")
                else:
                    ax.plot(g[:, 0], g[:, 1], color="#888888", linewidth=1.5, alpha=0.7)
            ax.legend(fontsize=6)
            ax.set_title(f"Remove cls={rm_cls}\n{target_camera} g{target_group}", fontsize=8)

            # Col 1: Removed lane (crossed out)
            ax = axes[row, 1]
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            if rm_cls in cls_to_geom:
                g = cls_to_geom[rm_cls]
                ax.plot(g[:, 0], g[:, 1], color="#e41a1c", linewidth=2, alpha=0.5,
                        linestyle="--")
                # Draw X through the lane
                cx, cy = g.mean(axis=0)
                ax.plot(cx, cy, "x", color="#e41a1c", markersize=20, markeredgewidth=3)
            ax.set_title("Removed lane", fontsize=8)

            # Col 2: empty (no candidates for removal)
            ax = axes[row, 2]
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            ax.text(0.5, 0.5, "N/A\n(removal)", ha="center", va="center",
                    fontsize=10, color="#999999", transform=ax.transAxes)
            ax.set_title("No generation", fontsize=8)

            # Col 3: Result — group without the removed lane
            ax = axes[row, 3]
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            if frame_path.exists():
                ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.3)
            for cid, g in all_geoms:
                if cid != rm_cls:
                    ax.plot(g[:, 0], g[:, 1], color="#377eb8", linewidth=2, alpha=0.8)
            ax.set_title("After removal", fontsize=8)

            logger.info(f"Lane removal: cls={rm_cls} removed from {target_camera} g{target_group}")

    # --- Target lane count: regenerate entire group with N lanes ---
    if has_target_lanes:
        row = len(specs) + len(remove_cls_ids)

        # Gather existing group info
        group_samples = [
            s for s in dataset.samples
            if s.camera == target_camera and s.group_id == target_group
        ]
        cls_to_geom = {}
        cls_to_emb = {}
        for s in group_samples:
            idx_in_index = dataset.samples.index(s)
            if idx_in_index < len(index.geometries):
                cls_to_geom[s.cls_id] = index.geometries[idx_in_index]
                cls_to_emb[s.cls_id] = index.embeddings[idx_in_index]

        existing_geoms = np.array(list(cls_to_geom.values()))  # (G, K, 2)
        existing_embs = list(cls_to_emb.values())
        current_n = len(existing_geoms)
        N = target_lanes

        # Compute road extents along perpendicular direction
        # Use unfolded tangent for visual consistency
        tangents = []
        for g in existing_geoms:
            t = g[-1] - g[0]
            norm = np.linalg.norm(t)
            if norm > 1e-8:
                tangents.append(t / norm)
        if tangents:
            mean_tangent = np.mean(tangents, axis=0)
            mean_tangent /= np.linalg.norm(mean_tangent) + 1e-8
        else:
            mean_tangent = np.array([1.0, 0.0])
        perp = np.array([-mean_tangent[1], mean_tangent[0]])

        # Project existing centroids onto perp to find road bounds
        centroids = existing_geoms.mean(axis=1)  # (G, 2)
        lateral_pos = centroids @ perp
        sorted_order = np.argsort(lateral_pos)

        left_bound = lateral_pos[sorted_order[0]]
        right_bound = lateral_pos[sorted_order[-1]]
        road_width = right_bound - left_bound

        # Evenly space N positions across the road width
        if N == 1:
            target_positions = np.array([(left_bound + right_bound) / 2])
        else:
            target_positions = np.linspace(left_bound, right_bound, N)

        # For each target position, find nearest existing lane for embedding + warm-start
        # Then generate at the target centroid
        mean_centroid = centroids.mean(axis=0)
        generated_lanes = []
        for i, target_lat in enumerate(target_positions):
            # Find nearest existing lane by lateral position
            dists = np.abs(lateral_pos - target_lat)
            nearest_idx = np.argmin(dists)
            ref_geom = existing_geoms[nearest_idx]
            ref_emb = existing_embs[nearest_idx]

            # Compute target centroid: shift mean centroid to target lateral position
            # Current mean lateral = mean of centroids projected onto perp
            ref_centroid = centroids[nearest_idx]
            lateral_shift = target_lat - lateral_pos[nearest_idx]
            target_centroid = ref_centroid + lateral_shift * perp

            result = generator.generate_with_embedding(
                ref_emb, ref_geom,
                n_candidates=n_candidates,
                override_centroid=target_centroid,
            )
            generated_lanes.append(result.best)

        # Col 0: Original group
        ax = axes[row, 0]
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        frame_path = annot_dir / target_camera / "last_frame.npy"
        if frame_path.exists():
            frame = np.load(str(frame_path))
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.5)
        for g in existing_geoms:
            ax.plot(g[:, 0], g[:, 1], color="#888888", linewidth=1.5, alpha=0.7)
        ax.set_title(f"Original ({current_n} lanes)\n{target_camera} g{target_group}",
                     fontsize=8)

        # Col 1: Target positions (dots showing where lanes will be placed)
        ax = axes[row, 1]
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for g in existing_geoms:
            ax.plot(g[:, 0], g[:, 1], color="#cccccc", linewidth=1, alpha=0.4)
        for i, target_lat in enumerate(target_positions):
            nearest_idx = np.argmin(np.abs(lateral_pos - target_lat))
            ref_centroid = centroids[nearest_idx]
            lateral_shift = target_lat - lateral_pos[nearest_idx]
            tc = ref_centroid + lateral_shift * perp
            ax.scatter(tc[0], tc[1], c=_CLUSTER_COLORS[i % len(_CLUSTER_COLORS)],
                       s=60, zorder=5, edgecolors="black", linewidths=0.5)
        ax.set_title(f"Target: {N} lanes\n(evenly spaced)", fontsize=8)

        # Col 2: All generated lanes overlaid
        ax = axes[row, 2]
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for g in existing_geoms:
            ax.plot(g[:, 0], g[:, 1], color="#cccccc", linewidth=1, alpha=0.3)
        for i, lane in enumerate(generated_lanes):
            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=2,
                    alpha=0.8, label=f"lane {i}")
        ax.legend(fontsize=5, loc="lower right")
        ax.set_title(f"{N} regenerated lanes", fontsize=8)

        # Col 3: Final result on camera frame
        ax = axes[row, 3]
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        if frame_path.exists():
            ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.3)
        for i, lane in enumerate(generated_lanes):
            color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
            ax.plot(lane[:, 0], lane[:, 1], color=color, linewidth=2.5, alpha=0.9)
        ax.set_title(f"Result: {current_n} → {N} lanes", fontsize=8)

        logger.info(
            f"Target lanes: regenerated {current_n} → {N} lanes "
            f"for {target_camera} g{target_group}"
        )

    fig.suptitle(f"Spec-Conditioned Lane Generation ({version_label})", fontsize=13)
    fig.tight_layout()

    suffix = "" if version_label == "independent" else f"_{version_label}"
    path = output_dir / f"g2_spec_conditioned{suffix}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved G2 ({version_label}) to {path}")


def figure_g2_compare(
    index,
    dataset,
    model,
    device: torch.device,
    output_dir: Path,
    baseline_trainer=None,
    relational_trainer=None,
    args=None,
):
    """Side-by-side comparison: baseline vs relational diffusion for merge lanes.

    Generates the same merge-lane specs with both models and plots them
    side by side for visual A/B comparison.
    """
    from src.generation.spec import LaneSpecification, SpecEmbeddingResolver
    from src.generation.directed import DirectedLaneGenerator

    if baseline_trainer is None or relational_trainer is None:
        logger.warning("G2 compare requires both baseline and relational trainers")
        return

    resolver = SpecEmbeddingResolver(index, dataset)
    baseline_gen = DirectedLaneGenerator(
        resolver, baseline_trainer, encoder=model, dataset=dataset, device=device,
    )
    relational_gen = DirectedLaneGenerator(
        resolver, baseline_trainer, encoder=model, dataset=dataset, device=device,
        relational_trainer=relational_trainer,
    )

    # Pick target location
    camera_groups = {}
    for sample in dataset.samples:
        key = (sample.camera, sample.group_id)
        camera_groups.setdefault(key, []).append(sample)

    if args and args.camera and args.group_id is not None:
        target_camera, target_group = args.camera, args.group_id
    else:
        best_key = max(camera_groups, key=lambda k: len(camera_groups[k]))
        target_camera, target_group = best_key

    # All three lane types
    specs = [
        ("Rightmost lane", LaneSpecification.rightmost(target_camera, target_group)),
        ("Leftmost lane", LaneSpecification.leftmost(target_camera, target_group)),
        ("Merge lane", LaneSpecification.merge_lane(target_camera, target_group)),
    ]
    n_candidates = args.n_candidates if args else 5

    annot_dir = Path(
        dataset.config.get("data", {}).get("annotation_dir", "../dataset/preprocess")
    )

    # Load camera frame once
    frame = None
    frame_path = annot_dir / target_camera / "last_frame.npy"
    if frame_path.exists():
        frame = np.load(str(frame_path))
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    n_rows = len(specs)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows), squeeze=False)

    for row, (label, spec) in enumerate(specs):
        # Baseline
        baseline_result = baseline_gen.generate(spec, n_candidates=n_candidates)
        # Relational
        relational_result = relational_gen.generate_relational(
            spec, n_candidates=n_candidates,
        )

        for col, (result, model_name) in enumerate([
            (baseline_result, "Baseline"),
            (relational_result, "Relational"),
        ]):
            ax = axes[row, col]
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_aspect("equal")

            if frame is not None:
                ax.imshow(frame, extent=[0, 1, 1, 0], alpha=0.3)

            # Existing lanes (gray)
            if result.spatial_context is not None:
                for g in result.spatial_context.group_geometries:
                    ax.plot(g[:, 0], g[:, 1], color="#888888",
                            linewidth=1.5, alpha=0.6)

                # Highlight neighbor lane (blue dashed) for relational
                if col == 1 and result.spatial_context.neighbor_geometry is not None:
                    nb = result.spatial_context.neighbor_geometry
                    ax.plot(nb[:, 0], nb[:, 1], color="#377eb8",
                            linewidth=2, linestyle="--", alpha=0.8,
                            label="neighbor")

            # Best candidate (red)
            ax.plot(result.best[:, 0], result.best[:, 1], color="#e41a1c",
                    linewidth=3, label="generated")
            ax.scatter(result.best[0, 0], result.best[0, 1], c="green", s=30, zorder=5)
            ax.scatter(result.best[-1, 0], result.best[-1, 1], c="red", s=30, zorder=5)
            best_score = result.scores.max()
            ax.legend(fontsize=7, loc="lower right")
            ax.set_title(f"{model_name}: {label}\n(score={best_score:.3f})", fontsize=9)

    fig.suptitle(
        "Independent vs Relational Conditioning\n"
        f"{target_camera} group {target_group}",
        fontsize=13,
    )
    fig.tight_layout()

    path = output_dir / "g2_baseline_vs_relational.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved baseline vs relational comparison to {path}")


# ---------------------------------------------------------------------------
# Figure G5: Generation quality metrics
# ---------------------------------------------------------------------------

def _classify_lateral_position(candidate, group_geoms, group_heading):
    """Classify a generated candidate as leftmost/interior/rightmost within its group.

    Projects the candidate centroid onto the axis perpendicular to the group
    heading and compares against the real lanes' positions.

    Args:
        candidate: (K, 2) generated lane.
        group_geoms: (G, K, 2) existing real lanes.
        group_heading: float, mean heading in radians.

    Returns:
        str: "leftmost", "rightmost", or "interior"
    """
    perp = np.array([-np.sin(group_heading), np.cos(group_heading)])
    cand_proj = np.dot(candidate.mean(axis=0), perp)
    real_projs = np.array([np.dot(g.mean(axis=0), perp) for g in group_geoms])

    real_min = real_projs.min()
    real_max = real_projs.max()
    real_range = real_max - real_min

    if real_range < 1e-6:
        return "interior"

    # Normalized position: 0 = leftmost edge, 1 = rightmost edge
    norm_pos = (cand_proj - real_min) / real_range

    if norm_pos <= -0.1 or norm_pos <= 0.2:
        return "leftmost"
    elif norm_pos >= 1.1 or norm_pos >= 0.8:
        return "rightmost"
    else:
        return "interior"


def _compute_smoothness(geom):
    """Compute curvature smoothness (variance of angle changes) for a polyline."""
    diffs = np.diff(geom, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_changes = np.diff(angles)
    return float(np.var(angle_changes))


def figure_g3(
    index,
    dataset,
    model,
    device: torch.device,
    output_dir: Path,
    trainer=None,
    relational_trainer=None,
    args=None,
):
    """Generation quality metrics (all geometry-based, no re-encoding bias).

    When relational_trainer is provided, generates both independent and
    relational versions as separate figures and metrics files.

    Metrics:
        Diversity — mean pairwise L2 between candidates (per group)
        Spatial Coherence — distance from candidates to group centroid
        Chamfer Distance — min distance from each candidate to nearest real lane
        Curvature Smoothness — variance of per-segment angle changes (raw + filtered)
        FGD — Frechet Geometry Distance in raw waypoint space (16x2 flattened)
        Spec Accuracy — does the generated lane land in the correct lateral position?
    """
    from src.generation.spec import LaneSpecification, SpecEmbeddingResolver
    from src.generation.directed import DirectedLaneGenerator

    if trainer is None:
        logger.warning("G3 requires a trained diffusion model, skipping")
        return

    resolver = SpecEmbeddingResolver(index, dataset)
    generator = DirectedLaneGenerator(
        resolver, trainer, encoder=model, dataset=dataset, device=device,
        relational_trainer=relational_trainer,
    )

    # Generate for all unique camera/group combinations
    camera_groups = {}
    for idx, sample in enumerate(dataset.samples):
        key = (sample.camera, sample.group_id)
        camera_groups.setdefault(key, []).append(idx)

    spec_defs = [
        ("rightmost", LaneSpecification.rightmost),
        ("leftmost", LaneSpecification.leftmost),
        ("merge", LaneSpecification.merge_lane),
    ]

    # Generate both independent and relational versions when available
    versions = [("independent", False)]
    if relational_trainer is not None:
        versions.append(("relational", True))

    for version_label, use_relational in versions:
        _g3_render(
            generator, index, camera_groups, spec_defs,
            use_relational, version_label, output_dir,
        )

    # Generate comparison table if both versions exist
    ind_path = output_dir / "generation_metrics.npz"
    rel_path = output_dir / "generation_metrics_relational.npz"
    if ind_path.exists() and rel_path.exists():
        _g3_comparison_table(ind_path, rel_path, output_dir)


def _g3_comparison_table(ind_path, rel_path, output_dir):
    """Render a side-by-side comparison table of independent vs relational metrics."""
    ind = np.load(str(ind_path))
    rel = np.load(str(rel_path))

    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else float("nan")

    # Build rows: (metric_name, independent_val, relational_val, direction)
    # direction: "lower" means lower is better, "higher" means higher is better
    rows = [
        ("Candidate Diversity (mean pairwise L2)",
         _safe_mean(ind["diversities"]), _safe_mean(rel["diversities"]), "higher"),
        ("Spatial Coherence (dist to group centroid)",
         _safe_mean(ind["coherences"]), _safe_mean(rel["coherences"]), "lower"),
        ("Chamfer Distance (mean, all specs)",
         _safe_mean(ind["chamfer_dists"]), _safe_mean(rel["chamfer_dists"]), "lower"),
        ("Chamfer — Rightmost",
         _safe_mean(ind["chamfer_rightmost"]), _safe_mean(rel["chamfer_rightmost"]), "lower"),
        ("Chamfer — Leftmost",
         _safe_mean(ind["chamfer_leftmost"]), _safe_mean(rel["chamfer_leftmost"]), "lower"),
        ("Chamfer — Merge",
         _safe_mean(ind["chamfer_merge"]), _safe_mean(rel["chamfer_merge"]), "lower"),
        ("Curvature Smoothness (raw)",
         _safe_mean(ind["smoothnesses_raw"]), _safe_mean(rel["smoothnesses_raw"]), "lower"),
        ("Curvature Smoothness (filtered)",
         _safe_mean(ind["smoothnesses_filtered"]), _safe_mean(rel["smoothnesses_filtered"]), "lower"),
        ("FGD (Frechet Geometry Distance)",
         float(ind["fgd"][0]), float(rel["fgd"][0]), "lower"),
        ("FGD (filtered)",
         float(ind["fgd_filtered"][0]), float(rel["fgd_filtered"][0]), "lower"),
        ("Spec Accuracy — Overall",
         float(ind["spec_acc_overall"][0]), float(rel["spec_acc_overall"][0]), "higher"),
        ("Spec Accuracy — Rightmost",
         float(ind["spec_acc_rightmost"][0]), float(rel["spec_acc_rightmost"][0]), "higher"),
        ("Spec Accuracy — Leftmost",
         float(ind["spec_acc_leftmost"][0]), float(rel["spec_acc_leftmost"][0]), "higher"),
        ("Spec Accuracy — Merge",
         float(ind["spec_acc_merge"][0]), float(rel["spec_acc_merge"][0]), "higher"),
    ]

    # Render as matplotlib table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    col_labels = ["Metric", "Independent", "Relational", "Better"]
    cell_text = []
    cell_colors = []

    for name, iv, rv, direction in rows:
        # Format values
        if "Accuracy" in name:
            i_str = f"{iv:.1%}"
            r_str = f"{rv:.1%}"
        else:
            i_str = f"{iv:.4f}"
            r_str = f"{rv:.4f}"

        # Determine winner
        if np.isnan(iv) or np.isnan(rv):
            winner = "—"
            i_color = "white"
            r_color = "white"
        elif direction == "lower":
            if rv < iv:
                winner = "Relational"
                i_color = "white"
                r_color = "#e8f5e9"
            elif iv < rv:
                winner = "Independent"
                i_color = "#e8f5e9"
                r_color = "white"
            else:
                winner = "Tie"
                i_color = "white"
                r_color = "white"
        else:  # higher is better
            if rv > iv:
                winner = "Relational"
                i_color = "white"
                r_color = "#e8f5e9"
            elif iv > rv:
                winner = "Independent"
                i_color = "#e8f5e9"
                r_color = "white"
            else:
                winner = "Tie"
                i_color = "white"
                r_color = "white"

        cell_text.append([name, i_str, r_str, winner])
        cell_colors.append(["#f5f5f5", i_color, r_color, "white"])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#d4e6f1"] * 4,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Bold header
    for j in range(4):
        table[0, j].set_text_props(fontweight="bold")
    # Left-align metric names
    for i in range(len(cell_text)):
        table[i + 1, 0].set_text_props(ha="left")

    ax.set_title(
        "G3: Generation Quality — Independent vs Relational",
        fontsize=13, fontweight="bold", pad=20,
    )

    # Add note at bottom
    fig.text(0.5, 0.02,
             "Green highlight = better result. "
             "Diversity: higher = more varied candidates. "
             "All other distances: lower = better.",
             fontsize=8, ha="center", color="#666666", style="italic")

    path = output_dir / "g3_comparison_table.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also save as CSV for paper
    csv_path = output_dir / "g3_comparison_table.csv"
    with open(str(csv_path), "w") as f:
        f.write("Metric,Independent,Relational,Better\n")
        for row in cell_text:
            f.write(",".join(row) + "\n")

    logger.info(f"Saved G3 comparison table to {path} and {csv_path}")


def _g3_render(generator, index, camera_groups, spec_defs,
               use_relational, version_label, output_dir):
    """Render a single G3 figure (independent or relational)."""
    SMOOTHNESS_THRESHOLD = 1.0  # Outlier threshold for smoothness

    diversities = []
    coherences = []
    chamfer_dists = []
    smoothnesses_raw = []
    all_gen_geoms_flat = []  # for FGD
    all_gen_geoms_filtered_flat = []  # for FGD (filtered)

    # Per-spec Chamfer tracking
    chamfer_by_spec = {"rightmost": [], "leftmost": [], "merge": []}

    # Spec accuracy tracking
    spec_correct = {"rightmost": 0, "leftmost": 0, "merge": 0}
    spec_total = {"rightmost": 0, "leftmost": 0, "merge": 0}
    # Per-spec predicted position distribution (for lateral position panel)
    spec_predicted = {
        "rightmost": {"leftmost": 0, "interior": 0, "rightmost": 0},
        "leftmost": {"leftmost": 0, "interior": 0, "rightmost": 0},
        "merge": {"leftmost": 0, "interior": 0, "rightmost": 0},
    }

    for (camera, group_id), sample_indices in camera_groups.items():
        # Real lane geometries in this group
        group_real_geoms = index.geometries[sample_indices]  # (G, K, 2)

        # Compute group heading for lateral classification
        tangents = []
        for geom in group_real_geoms:
            t = geom[-1] - geom[0]
            norm = np.linalg.norm(t)
            if norm > 1e-6:
                tangents.append(t / norm)
        mean_tangent = np.mean(tangents, axis=0) if tangents else np.array([1.0, 0.0])
        group_heading = float(np.arctan2(mean_tangent[1], mean_tangent[0]))

        for spec_name, spec_factory in spec_defs:
            spec = spec_factory(camera=camera, group_id=group_id)
            if use_relational:
                result = generator.generate_relational(spec, n_candidates=5)
            else:
                result = generator.generate(spec, n_candidates=5)
            candidates = result.candidates  # (n, K, 2)

            # 1. Diversity: mean pairwise L2 between candidates
            n = len(candidates)
            pairwise_dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(candidates[i] - candidates[j])
                    pairwise_dists.append(d)
            if pairwise_dists:
                diversities.append(np.mean(pairwise_dists))

            # 2. Spatial coherence: distance to group centroid
            if result.spatial_context is not None:
                group_centroid = result.spatial_context.group_geometries.mean(
                    axis=(0, 1)
                )
                cand_centroids = candidates.mean(axis=1)
                dists_to_group = np.linalg.norm(
                    cand_centroids - group_centroid, axis=1
                )
                coherences.append(np.mean(dists_to_group))

            # 3. Chamfer distance: min L2 to nearest real lane in group
            for cand in candidates:
                min_dist = min(
                    np.mean(np.linalg.norm(cand - real, axis=1))
                    for real in group_real_geoms
                )
                chamfer_dists.append(min_dist)
                chamfer_by_spec[spec_name].append(min_dist)

            # 4. Curvature smoothness: variance of angle changes
            for cand in candidates:
                s = _compute_smoothness(cand)
                smoothnesses_raw.append(s)

            # 5. Spec accuracy: does generated lane land in correct position?
            expected = spec_name  # "rightmost", "leftmost", or "merge"
            for cand in candidates:
                predicted = _classify_lateral_position(
                    cand, group_real_geoms, group_heading,
                )
                if expected == "merge":
                    # Merge lanes should be on the rightmost side (exit/on-ramp)
                    is_correct = predicted == "rightmost"
                else:
                    is_correct = predicted == expected
                spec_total[expected] += 1
                if is_correct:
                    spec_correct[expected] += 1
                spec_predicted[expected][predicted] += 1

            # Collect flattened geometries for FGD
            for cand in candidates:
                all_gen_geoms_flat.append(cand.flatten())
                if _compute_smoothness(cand) < SMOOTHNESS_THRESHOLD:
                    all_gen_geoms_filtered_flat.append(cand.flatten())

    # --- Smoothness: raw vs filtered ---
    smoothnesses_filtered = [s for s in smoothnesses_raw if s < SMOOTHNESS_THRESHOLD]
    n_outliers = len(smoothnesses_raw) - len(smoothnesses_filtered)
    outlier_pct = n_outliers / max(len(smoothnesses_raw), 1)

    # --- Compute FGD (Frechet Geometry Distance) ---
    all_gen_flat = np.array(all_gen_geoms_flat)  # (M, K*2)
    real_flat = index.geometries.reshape(index.N, -1)  # (N, K*2)

    mu_real = real_flat.mean(axis=0)
    mu_gen = all_gen_flat.mean(axis=0)
    sigma_real = np.cov(real_flat, rowvar=False)
    sigma_gen = np.cov(all_gen_flat, rowvar=False)

    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fgd = float(np.dot(diff, diff) + np.trace(
        sigma_real + sigma_gen - 2.0 * covmean
    ))

    # FGD on filtered candidates
    if len(all_gen_geoms_filtered_flat) > 1:
        gen_filt = np.array(all_gen_geoms_filtered_flat)
        mu_gen_f = gen_filt.mean(axis=0)
        sigma_gen_f = np.cov(gen_filt, rowvar=False)
        diff_f = mu_real - mu_gen_f
        covmean_f = sqrtm(sigma_real @ sigma_gen_f)
        if np.iscomplexobj(covmean_f):
            covmean_f = covmean_f.real
        fgd_filtered = float(np.dot(diff_f, diff_f) + np.trace(
            sigma_real + sigma_gen_f - 2.0 * covmean_f
        ))
    else:
        fgd_filtered = fgd

    # --- Spec accuracy summary ---
    spec_acc = {}
    for k in spec_correct:
        total = spec_total[k]
        correct = spec_correct[k]
        spec_acc[k] = correct / total if total > 0 else 0.0

    overall_correct = sum(spec_correct.values())
    overall_total = sum(spec_total.values())
    spec_acc["overall"] = overall_correct / overall_total if overall_total > 0 else 0.0

    # --- Plot 3x3 ---
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 1: Diversity, Coherence, Chamfer
    ax1, ax2, ax3 = axes[0]

    if diversities:
        ax1.hist(diversities, bins=15, color="#4daf4a", edgecolor="black", alpha=0.7)
        ax1.axvline(np.mean(diversities), color="red", linestyle="--",
                    label=f"mean={np.mean(diversities):.4f}")
        ax1.set_xlabel("Mean Pairwise L2 Distance")
        ax1.set_ylabel("Count (camera-groups)")
        ax1.set_title("Candidate Diversity\n(higher = more varied candidates)", fontsize=10)
        ax1.legend(fontsize=8)

    if coherences:
        ax2.hist(coherences, bins=15, color="#377eb8", edgecolor="black", alpha=0.7)
        ax2.axvline(np.mean(coherences), color="red", linestyle="--",
                    label=f"mean={np.mean(coherences):.4f}")
        ax2.set_xlabel("Distance to Group Centroid")
        ax2.set_ylabel("Count (camera-groups)")
        ax2.set_title("Spatial Coherence\n(lower = closer to group center)", fontsize=10)
        ax2.legend(fontsize=8)

    if chamfer_dists:
        ax3.hist(chamfer_dists, bins=20, color="#ff7f00", edgecolor="black", alpha=0.7)
        ax3.axvline(np.mean(chamfer_dists), color="red", linestyle="--",
                    label=f"mean={np.mean(chamfer_dists):.4f}")
        ax3.set_xlabel("Mean Point Distance to Nearest Real Lane")
        ax3.set_ylabel("Count (candidates)")
        ax3.set_title("Chamfer Distance\n(lower = closer to real lanes)", fontsize=10)
        ax3.legend(fontsize=8)

    # Row 2: Smoothness (raw + filtered), FGD, Spec Accuracy bar
    ax4, ax5, ax6 = axes[1]

    if smoothnesses_raw:
        # Raw smoothness histogram with outlier threshold
        ax4.hist(smoothnesses_raw, bins=30, color="#984ea3", edgecolor="black",
                 alpha=0.4, label=f"raw mean={np.mean(smoothnesses_raw):.4f}")
        if smoothnesses_filtered:
            ax4.hist(smoothnesses_filtered, bins=30, color="#984ea3",
                     edgecolor="black", alpha=0.7,
                     label=f"filtered mean={np.mean(smoothnesses_filtered):.4f}")
        ax4.axvline(SMOOTHNESS_THRESHOLD, color="red", linestyle="--",
                    label=f"threshold={SMOOTHNESS_THRESHOLD}")
        ax4.set_xlabel("Variance of Angle Changes")
        ax4.set_ylabel("Count (candidates)")
        ax4.set_title(f"Curvature Smoothness ({n_outliers} outliers, {outlier_pct:.0%})\n(lower = smoother lanes)", fontsize=10)
        ax4.legend(fontsize=7)

    # FGD panel
    fgd_labels = ["FGD (all)", "FGD (filtered)"]
    fgd_values = [fgd, fgd_filtered]
    colors_fgd = ["#e41a1c", "#377eb8"]
    ax5.barh(fgd_labels, fgd_values, color=colors_fgd, edgecolor="black", alpha=0.7)
    ax5.set_xlabel("Frechet Geometry Distance")
    ax5.set_title(f"FGD = {fgd:.4f} (filtered: {fgd_filtered:.4f})\n(lower = more realistic distribution)", fontsize=10)
    ax5.set_xlim(0, max(max(fgd_values) * 1.3, 0.001))
    for spine in ["top", "right"]:
        ax5.spines[spine].set_visible(False)

    # Spec accuracy bar chart
    spec_labels = ["rightmost", "leftmost", "merge", "overall"]
    spec_values = [spec_acc[k] for k in spec_labels]
    spec_colors = ["#4daf4a", "#377eb8", "#ff7f00", "#333333"]
    bars = ax6.bar(spec_labels, spec_values, color=spec_colors, edgecolor="black", alpha=0.7)
    ax6.set_ylabel("Accuracy")
    ax6.set_ylim(0, 1.15)
    ax6.set_title(f"Spec Accuracy (overall: {spec_acc['overall']:.1%})\n(correct lateral position classification)", fontsize=10)
    ax6.axhline(1.0 / 3.0, color="gray", linestyle=":", alpha=0.5, label="chance (33%)")
    ax6.legend(fontsize=7, loc="lower right")
    # Annotate bars with count INSIDE the bar
    for bar, label in zip(bars, spec_labels):
        total = spec_total.get(label, overall_total)
        correct = spec_correct.get(label, overall_correct)
        y_pos = bar.get_height() / 2
        ax6.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{correct}/{total}\n({spec_acc[label]:.0%})",
                 ha="center", va="center", fontsize=7,
                 fontweight="bold", color="white")

    # Row 3: Summary table, per-spec Chamfer, per-spec Diversity
    ax7, ax8, ax9 = axes[2]

    # Summary
    ax7.axis("off")
    mean_smooth_filt = np.mean(smoothnesses_filtered) if smoothnesses_filtered else 0
    summary = (
        f"Diversity (mean):       {np.mean(diversities):.4f}\n"
        f"Coherence (mean):       {np.mean(coherences):.4f}\n"
        f"Chamfer dist (mean):    {np.mean(chamfer_dists):.4f}\n"
        f"Smoothness (raw):       {np.mean(smoothnesses_raw):.4f}\n"
        f"Smoothness (filtered):  {mean_smooth_filt:.4f}\n"
        f"Smoothness outliers:    {n_outliers}/{len(smoothnesses_raw)} ({outlier_pct:.0%})\n"
        f"FGD (all):              {fgd:.4f}\n"
        f"FGD (filtered):         {fgd_filtered:.4f}\n"
        f"\n"
        f"Spec accuracy:\n"
        f"  rightmost:  {spec_acc['rightmost']:.1%} ({spec_correct['rightmost']}/{spec_total['rightmost']})\n"
        f"  leftmost:   {spec_acc['leftmost']:.1%} ({spec_correct['leftmost']}/{spec_total['leftmost']})\n"
        f"  merge:      {spec_acc['merge']:.1%} ({spec_correct['merge']}/{spec_total['merge']})\n"
        f"  overall:    {spec_acc['overall']:.1%} ({overall_correct}/{overall_total})\n"
        f"\n"
        f"N_real={index.N}, N_gen={len(all_gen_flat)}\n"
        f"N_gen_filtered={len(all_gen_geoms_filtered_flat)}\n"
        f"Groups={len(camera_groups)}, Specs=3"
    )
    ax7.text(0.05, 0.5, summary, fontsize=9, family="monospace",
             verticalalignment="center", transform=ax7.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
    ax7.set_title("Summary")

    # Per-spec lateral position distribution (stacked bar chart)
    spec_names = ["rightmost", "leftmost", "merge"]
    positions = ["leftmost", "interior", "rightmost"]
    pos_colors = {"leftmost": "#2196F3", "interior": "#9E9E9E", "rightmost": "#FF5722"}
    x_pos = np.arange(len(spec_names))
    bar_width = 0.6
    bottoms = np.zeros(len(spec_names))
    for pos in positions:
        counts = []
        for sn in spec_names:
            total = spec_total[sn] if spec_total[sn] > 0 else 1
            counts.append(spec_predicted[sn][pos] / total)
        ax8.bar(x_pos, counts, bar_width, bottom=bottoms,
                label=pos, color=pos_colors[pos], alpha=0.85)
        bottoms += np.array(counts)
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(spec_names, fontsize=9)
    ax8.set_ylabel("Fraction")
    ax8.set_ylim(0, 1.05)
    ax8.legend(fontsize=8, title="Predicted", loc="upper right")
    ax8.set_title("Lateral Position Distribution\n(where candidates land per spec)", fontsize=10)

    # Per-spec Chamfer distance breakdown (box plot)
    chamfer_data = []
    chamfer_labels = []
    for sn in ["rightmost", "leftmost", "merge"]:
        if chamfer_by_spec[sn]:
            chamfer_data.append(chamfer_by_spec[sn])
            chamfer_labels.append(sn)
    if chamfer_data:
        bp = ax9.boxplot(chamfer_data, tick_labels=chamfer_labels, patch_artist=True,
                         widths=0.5, medianprops=dict(color="black", linewidth=1.5))
        spec_colors = {"rightmost": "#FF5722", "leftmost": "#2196F3", "merge": "#4CAF50"}
        for patch, label in zip(bp["boxes"], chamfer_labels):
            patch.set_facecolor(spec_colors.get(label, "#9E9E9E"))
            patch.set_alpha(0.7)
        ax9.set_ylabel("Chamfer Distance")
        # Annotate medians
        for i, (data, label) in enumerate(zip(chamfer_data, chamfer_labels)):
            median = np.median(data)
            ax9.text(i + 1, median, f" {median:.3f}", fontsize=7,
                     verticalalignment="bottom", color="black")
    ax9.set_title("Chamfer by Spec\n(lower = better per lane type)", fontsize=10)

    fig.suptitle(f"Lane Generation Quality ({version_label})", fontsize=14)
    fig.tight_layout()

    suffix = "" if version_label == "independent" else f"_{version_label}"
    path = output_dir / f"g3_quality_metrics{suffix}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save metrics
    np.savez(
        str(output_dir / f"generation_metrics{suffix}.npz"),
        diversities=np.array(diversities) if diversities else np.array([]),
        coherences=np.array(coherences) if coherences else np.array([]),
        chamfer_dists=np.array(chamfer_dists) if chamfer_dists else np.array([]),
        smoothnesses_raw=np.array(smoothnesses_raw) if smoothnesses_raw else np.array([]),
        smoothnesses_filtered=np.array(smoothnesses_filtered) if smoothnesses_filtered else np.array([]),
        fgd=np.array([fgd]),
        fgd_filtered=np.array([fgd_filtered]),
        spec_acc_rightmost=np.array([spec_acc["rightmost"]]),
        spec_acc_leftmost=np.array([spec_acc["leftmost"]]),
        spec_acc_merge=np.array([spec_acc["merge"]]),
        spec_acc_overall=np.array([spec_acc["overall"]]),
        chamfer_rightmost=np.array(chamfer_by_spec["rightmost"]) if chamfer_by_spec["rightmost"] else np.array([]),
        chamfer_leftmost=np.array(chamfer_by_spec["leftmost"]) if chamfer_by_spec["leftmost"] else np.array([]),
        chamfer_merge=np.array(chamfer_by_spec["merge"]) if chamfer_by_spec["merge"] else np.array([]),
    )
    logger.info(
        f"Saved G3 ({version_label}) to {path} | diversity={np.mean(diversities):.4f}, "
        f"coherence={np.mean(coherences):.4f}, "
        f"chamfer={np.mean(chamfer_dists):.4f}, "
        f"smoothness_raw={np.mean(smoothnesses_raw):.4f}, "
        f"smoothness_filt={mean_smooth_filt:.4f}, "
        f"FGD={fgd:.4f}, FGD_filt={fgd_filtered:.4f}, "
        f"spec_acc={spec_acc['overall']:.1%}"
    )


def _load_diffusion(path: str, index, device):
    """Load a pre-trained diffusion model."""
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer

    K = index.K
    geom_dim = K * 2
    cond_dim = index.D

    denoiser = LaneDenoiser(
        geom_dim=geom_dim, t_dim=64, cond_dim=cond_dim, hidden_dim=256,
    )
    schedule = DDPMSchedule(T=100)
    trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))
    trainer.load(path)
    return trainer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate lane geometry generation figures (G-series)"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained lane encoder checkpoint",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="results/generation/figures")
    parser.add_argument(
        "--figures", nargs="+", default=["G1", "G2", "G3"],
        choices=["G1", "G2", "G3"],
    )
    parser.add_argument(
        "--cluster-method", default="hdbscan",
        choices=["hdbscan", "kmeans"],
    )
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--retrieval-k", type=int, default=3)

    # Diffusion options
    parser.add_argument("--train-diffusion", action="store_true",
                        help="Train diffusion model before generating figures")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Load pre-trained diffusion model")
    parser.add_argument("--diffusion-epochs", type=int, default=1000)
    parser.add_argument("--diffusion-lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-batch-size", type=int, default=32)

    # Two-stage pretraining (OpenLaneV2)
    parser.add_argument("--openlane-data", default=None,
                        help="Path to OpenLaneV2 canonical .npz or raw data dir for Stage 1 pretraining")
    parser.add_argument("--pretrain-epochs", type=int, default=500,
                        help="Number of unconditional pretraining epochs on OpenLaneV2")
    parser.add_argument("--openlane-max-lanes", type=int, default=20000,
                        help="Max lanes to extract from OpenLaneV2 (if using raw dir)")

    # Directed generation options (G2)
    parser.add_argument("--list-locations", action="store_true",
                        help="Print available camera/group locations and exit")
    parser.add_argument("--camera", default=None,
                        help="Target camera for directed generation (default: auto-pick)")
    parser.add_argument("--group-id", type=int, default=None,
                        help="Target group ID for directed generation (default: auto-pick)")
    parser.add_argument("--lane-type", nargs="+", default=None,
                        choices=["rightmost", "leftmost", "merge"],
                        help="Lane types to generate (default: all three)")
    parser.add_argument("--spec", default=None,
                        help="Natural language spec, e.g. 'fast straight rightmost lane'")
    parser.add_argument("--mean-speed", type=float, default=None,
                        help="Target mean speed for behavioral prefix (normalized)")
    parser.add_argument("--mean-curvature", type=float, default=None,
                        help="Target mean curvature for behavioral prefix")
    parser.add_argument("--mean-lateral-offset", type=float, default=None,
                        help="Target mean lateral offset for behavioral prefix")
    parser.add_argument("--n-candidates", type=int, default=5,
                        help="Number of candidates per spec (default: 5)")

    # Relational diffusion options
    parser.add_argument("--train-relational", action="store_true",
                        help="Train relational diffusion model")
    parser.add_argument("--relational-checkpoint", default=None,
                        help="Load pre-trained relational diffusion model")
    parser.add_argument("--relational-epochs", type=int, default=1000)
    parser.add_argument("--no-relation-ratio", type=float, default=0.3,
                        help="Fraction of training samples with zeroed relational context")
    parser.add_argument("--augment-factor", type=int, default=10,
                        help="Augmentation factor for relational pairs")
    parser.add_argument("--relational", action="store_true",
                        help="Use relational model for directed generation (G2/G3)")
    parser.add_argument("--compare-relational", action="store_true",
                        help="Side-by-side baseline vs relational comparison")
    parser.add_argument("--replace-role", nargs="+", default=None,
                        help="Replace lane role: CLS_ID:ROLE (e.g. 5:rightmost 3:merge)")
    parser.add_argument("--remove-lane", nargs="+", type=int, default=None,
                        help="Remove lane(s) by cls_id (e.g. --remove-lane 5 3)")
    parser.add_argument("--target-lanes", type=int, default=None,
                        help="Regenerate group with exactly N lanes (e.g. --target-lanes 3)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and build retrieval index
    model, config, dataset, index, device = _load_data(args)

    # List available locations and exit if requested
    if args.list_locations:
        camera_groups = {}
        for sample in dataset.samples:
            key = (sample.camera, sample.group_id)
            camera_groups.setdefault(key, []).append(sample)

        print("\nAvailable locations (camera, group_id, n_lanes):")
        print("-" * 55)
        for (cam, gid), samples in sorted(camera_groups.items()):
            roles = [s.role for s in samples]
            has_succ = sum(1 for r in roles if r.has_successor)
            print(f"  {cam:30s}  group={gid}  lanes={len(samples):2d}"
                  f"  (merge={has_succ})")
        print(f"\nTotal: {len(camera_groups)} camera-groups, "
              f"{len(dataset.samples)} lanes")
        return

    # Cluster embeddings
    labels = _cluster_embeddings(
        index.embeddings,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
    )

    # Save retrieval index data
    np.savez(
        str(output_dir / "retrieval_index.npz"),
        embeddings=index.embeddings,
        geometries=index.geometries,
        lane_keys=index.lane_keys,
        cameras=index.cameras,
        cluster_labels=labels,
    )
    logger.info(f"Saved retrieval index to {output_dir / 'retrieval_index.npz'}")

    # Optionally train or load diffusion
    trainer = None
    if args.train_diffusion:
        trainer = _train_diffusion(index, model, dataset, device, args)
    elif args.diffusion_checkpoint:
        trainer = _load_diffusion(args.diffusion_checkpoint, index, device)

    # Optionally train or load relational diffusion
    rel_trainer = None
    if args.train_relational:
        rel_trainer = _train_relational(index, model, dataset, device, args)
    elif args.relational_checkpoint:
        rel_trainer = _load_relational(args.relational_checkpoint, index, device)

    # When --relational flag is set, use relational trainer as primary
    # The baseline trainer is always needed for DirectedLaneGenerator.generate().
    # The relational trainer is passed separately and used only by generate_relational().
    # Do NOT substitute rel_trainer as the primary trainer — its sample() signature differs.
    active_trainer = trainer

    # G1: Multi-scenario generation
    if "G1" in args.figures:
        logger.info("Generating G1: Multi-scenario showcase...")
        figure_g1(index, dataset, model, device, output_dir,
                  trainer=active_trainer, relational_trainer=rel_trainer,
                  args=args)

    # G2: Spec-conditioned generation
    if "G2" in args.figures:
        logger.info("Generating G2: Spec-conditioned generation...")
        figure_g2(
            index, dataset, model, device, output_dir,
            trainer=active_trainer, args=args,
            relational_trainer=rel_trainer,
        )

    # G3: Quality metrics
    if "G3" in args.figures:
        logger.info("Generating G3: Quality metrics...")
        figure_g3(index, dataset, model, device, output_dir,
                  trainer=active_trainer, relational_trainer=rel_trainer,
                  args=args)

    # G2 comparison: baseline vs relational (auto-generate when both trainers available)
    if trainer and rel_trainer:
        logger.info("Generating G2 comparison: baseline vs relational...")
        figure_g2_compare(
            index, dataset, model, device, output_dir,
            baseline_trainer=trainer,
            relational_trainer=rel_trainer,
            args=args,
        )

    logger.info(f"All generation figures saved to {output_dir}")


if __name__ == "__main__":
    main()
