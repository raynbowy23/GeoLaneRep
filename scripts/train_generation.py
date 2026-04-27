#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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

def main():
    parser = argparse.ArgumentParser(
        description="Train baseline + (optional) relational diffusion models."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained lane encoder checkpoint")
    parser.add_argument("--config", default=None,
                        help="Optional config YAML override")
    parser.add_argument("--output-dir", default="results/generation/figures",
                        help="Where to save diffusion_model.pt and relational_diffusion_model.pt")

    # Baseline diffusion
    parser.add_argument("--diffusion-epochs", type=int, default=1000)
    parser.add_argument("--diffusion-lr", type=float, default=1e-3)
    parser.add_argument("--diffusion-batch-size", type=int, default=32)

    # Optional Stage-1 OpenLaneV2 pretraining
    parser.add_argument("--openlane-data", default=None,
                        help="Path to OpenLaneV2 canonical .npz or raw data dir (enables Stage-1 pretrain)")
    parser.add_argument("--pretrain-epochs", type=int, default=500,
                        help="Epochs for unconditional OpenLaneV2 pretraining (ignored if --openlane-data omitted)")
    parser.add_argument("--openlane-max-lanes", type=int, default=20000,
                        help="Max lanes to extract from raw OpenLaneV2 data")

    # Relational diffusion
    parser.add_argument("--train-relational", action="store_true",
                        help="Also train the relational diffusion model")
    parser.add_argument("--relational-epochs", type=int, default=1000)
    parser.add_argument("--no-relation-ratio", type=float, default=0.3,
                        help="Fraction of training samples with zeroed relational context")
    parser.add_argument("--augment-factor", type=int, default=10,
                        help="Augmentation factor for relational pairs")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, dataset, index, device = _load_data(args)

    logger.info("Training baseline diffusion model...")
    _train_diffusion(index, model, dataset, device, args)

    if args.train_relational:
        logger.info("Training relational diffusion model...")
        _train_relational(index, model, dataset, device, args)

    logger.info(f"All checkpoints saved under {output_dir}")


if __name__ == "__main__":
    main()
