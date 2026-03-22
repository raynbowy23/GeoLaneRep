#!/usr/bin/env python3
"""Diagnose whether FiLM conditioning is actually used by the diffusion model.

Tests:
1. FiLM magnitude check — Are gamma/beta near zero (= conditioning ignored)?
2. Conditioning sensitivity — Does changing the conditioning vector change the output?
3. Embedding variance check — Are different lanes producing meaningfully different embeddings?
4. Noise prediction delta — Does the predicted noise differ across conditions at the same timestep?

Usage:
    python scripts/diagnose_conditioning.py \
        --checkpoint results/joint_encoder/checkpoints/best.pt \
        --diffusion-checkpoint results/generation/figures/diffusion_model.pt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_models(args):
    """Load encoder and diffusion models + dataset."""
    import yaml
    from src.data.lane_dataset import LaneDataset, collate_fn
    from src.training.zero_shot_eval import load_trained_encoder
    from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
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

    # Encode all lanes
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

    embeddings = output["embedding"].cpu()

    # Load diffusion model
    K = model_cfg.get("polyline_k", 16)
    geom_dim = K * 2
    cond_dim = model_cfg.get("embedding_dim", 128)

    denoiser = LaneDenoiser(geom_dim=geom_dim, cond_dim=cond_dim).to(device)
    schedule = DDPMSchedule(T=100, device=str(device))
    trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))
    trainer.load(args.diffusion_checkpoint)

    # Canonical geometries
    from src.generation.augment import to_canonical
    K = model_cfg.get("polyline_k", 16)
    canonical = []
    for sample in dataset.samples:
        c, *_ = to_canonical(sample.geometry)
        # Ensure consistent size (K, 2) → (K*2,)
        if len(c) >= K:
            c = c[:K]
        else:
            c = np.pad(c, ((0, K - len(c)), (0, 0)), mode="edge")
        canonical.append(c.flatten())
    canonical = torch.tensor(np.array(canonical), dtype=torch.float32)

    return trainer, embeddings, canonical, dataset, device


# ---------------------------------------------------------------------------
# Test 1: FiLM magnitude check
# ---------------------------------------------------------------------------

def test_film_magnitudes(trainer, embeddings, device):
    """Check the magnitude of FiLM gamma/beta for real conditioning vectors.

    If gamma ≈ 0 and beta ≈ 0, the model has learned to ignore conditioning.
    """
    logger.info("=" * 60)
    logger.info("TEST 1: FiLM Gamma/Beta Magnitudes")
    logger.info("=" * 60)

    denoiser = trainer.model
    denoiser.eval()

    # Sample some conditioning vectors
    cond = embeddings[:min(50, len(embeddings))].to(device)

    results = {}
    for name, layer in [("layer1", denoiser.layer1),
                         ("layer2", denoiser.layer2),
                         ("layer3", denoiser.layer3)]:
        with torch.no_grad():
            gamma = layer.film_scale(cond)  # (B, hidden_dim)
            beta = layer.film_shift(cond)   # (B, hidden_dim)

        gamma_abs = gamma.abs()
        beta_abs = beta.abs()

        results[name] = {
            "gamma_mean": gamma_abs.mean().item(),
            "gamma_max": gamma_abs.max().item(),
            "gamma_std": gamma.std().item(),
            "beta_mean": beta_abs.mean().item(),
            "beta_max": beta_abs.max().item(),
            "beta_std": beta.std().item(),
        }

        # What fraction of gamma values are > 0.1?
        frac_active = (gamma_abs > 0.1).float().mean().item()
        results[name]["gamma_frac_active"] = frac_active

        logger.info(
            f"  {name}: gamma |mean|={gamma_abs.mean():.4f} max={gamma_abs.max():.4f} "
            f"std={gamma.std():.4f} active(>0.1)={frac_active:.1%}"
        )
        logger.info(
            f"  {name}: beta  |mean|={beta_abs.mean():.4f} max={beta_abs.max():.4f} "
            f"std={beta.std():.4f}"
        )

    # Verdict
    avg_gamma = np.mean([r["gamma_mean"] for r in results.values()])
    avg_active = np.mean([r["gamma_frac_active"] for r in results.values()])

    if avg_gamma < 0.01:
        logger.warning("VERDICT: FiLM gamma near zero — conditioning likely collapsed")
    elif avg_active < 0.1:
        logger.warning("VERDICT: Most gamma values inactive — weak conditioning")
    else:
        logger.info(f"VERDICT: FiLM active (avg |gamma|={avg_gamma:.4f}, {avg_active:.1%} active)")

    return results


# ---------------------------------------------------------------------------
# Test 2: Conditioning sensitivity
# ---------------------------------------------------------------------------

def test_conditioning_sensitivity(trainer, embeddings, device):
    """Generate samples with different conditioning vectors and measure output variance.

    If conditioning works, different embeddings should produce different geometries.
    If collapsed, all outputs look similar regardless of conditioning.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 2: Conditioning Sensitivity (output variance)")
    logger.info("=" * 60)

    n_conds = min(20, len(embeddings))
    n_samples_per = 5

    all_outputs = []
    for i in range(n_conds):
        cond = embeddings[i]
        samples = trainer.sample(cond.to(device), n_samples=n_samples_per)
        all_outputs.append(samples.cpu().numpy())

    all_outputs = np.array(all_outputs)  # (n_conds, n_samples_per, geom_dim)

    # Within-condition variance (should be moderate — stochastic generation)
    within_var = np.mean([np.var(all_outputs[i], axis=0).mean() for i in range(n_conds)])

    # Between-condition variance (should be HIGH if conditioning works)
    per_cond_means = all_outputs.mean(axis=1)  # (n_conds, geom_dim)
    between_var = np.var(per_cond_means, axis=0).mean()

    # Total variance
    flat = all_outputs.reshape(-1, all_outputs.shape[-1])
    total_var = np.var(flat, axis=0).mean()

    ratio = between_var / (total_var + 1e-8)

    logger.info(f"  Within-condition variance:  {within_var:.6f}")
    logger.info(f"  Between-condition variance: {between_var:.6f}")
    logger.info(f"  Total variance:             {total_var:.6f}")
    logger.info(f"  Between/Total ratio:        {ratio:.4f}")

    if ratio < 0.1:
        logger.warning("VERDICT: Between-condition variance < 10% of total — conditioning likely collapsed")
    elif ratio < 0.3:
        logger.warning("VERDICT: Between-condition variance low (< 30%) — conditioning weak")
    else:
        logger.info(f"VERDICT: Conditioning active — {ratio:.1%} of variance explained by conditioning")

    # Also measure: same cond vs different cond pairwise distances
    same_dists = []
    diff_dists = []
    for i in range(min(n_conds, 10)):
        for j in range(n_samples_per):
            for k in range(j + 1, n_samples_per):
                same_dists.append(np.linalg.norm(all_outputs[i, j] - all_outputs[i, k]))
        for i2 in range(i + 1, min(n_conds, 10)):
            d = np.linalg.norm(per_cond_means[i] - per_cond_means[i2])
            diff_dists.append(d)

    mean_same = np.mean(same_dists) if same_dists else 0
    mean_diff = np.mean(diff_dists) if diff_dists else 0
    sep_ratio = mean_diff / (mean_same + 1e-8)

    logger.info(f"  Mean same-cond pairwise dist: {mean_same:.4f}")
    logger.info(f"  Mean diff-cond pairwise dist: {mean_diff:.4f}")
    logger.info(f"  Separation ratio (diff/same): {sep_ratio:.4f}")

    if sep_ratio < 1.5:
        logger.warning("VERDICT: Inter-condition distance not much larger than intra — poor separation")

    return {
        "within_var": within_var,
        "between_var": between_var,
        "total_var": total_var,
        "ratio": ratio,
        "mean_same_dist": mean_same,
        "mean_diff_dist": mean_diff,
        "separation_ratio": sep_ratio,
    }


# ---------------------------------------------------------------------------
# Test 3: Embedding variance
# ---------------------------------------------------------------------------

def test_embedding_variance(embeddings, dataset):
    """Check if encoder embeddings are diverse enough to provide meaningful conditioning.

    If all embeddings are nearly identical, FiLM can't do anything useful.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 3: Encoder Embedding Variance")
    logger.info("=" * 60)

    emb_np = embeddings.numpy()

    # Overall stats
    norms = np.linalg.norm(emb_np, axis=1)
    per_dim_var = np.var(emb_np, axis=0)
    mean_var = per_dim_var.mean()
    active_dims = (per_dim_var > 0.01).sum()

    logger.info(f"  N embeddings:    {len(emb_np)}")
    logger.info(f"  Norm range:      [{norms.min():.3f}, {norms.max():.3f}] mean={norms.mean():.3f}")
    logger.info(f"  Per-dim variance: mean={mean_var:.6f} max={per_dim_var.max():.6f}")
    logger.info(f"  Active dims (var > 0.01): {active_dims}/{emb_np.shape[1]}")

    # Pairwise cosine similarity
    emb_norm = emb_np / (norms[:, None] + 1e-8)
    n = min(100, len(emb_norm))
    sim_matrix = emb_norm[:n] @ emb_norm[:n].T
    off_diag = sim_matrix[np.triu_indices(n, k=1)]

    logger.info(f"  Pairwise cosine sim: mean={off_diag.mean():.4f} std={off_diag.std():.4f}")
    logger.info(f"  Sim range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")

    # Per-camera variance
    cameras = list(set(s.camera for s in dataset.samples))
    for cam in cameras:
        cam_idx = [i for i, s in enumerate(dataset.samples) if s.camera == cam]
        cam_emb = emb_np[cam_idx]
        cam_var = np.var(cam_emb, axis=0).mean()
        logger.info(f"  Camera {cam}: n={len(cam_idx)} mean_var={cam_var:.6f}")

    if off_diag.mean() > 0.95:
        logger.warning("VERDICT: Embeddings nearly identical (mean cos sim > 0.95) — no diversity for conditioning")
    elif active_dims < emb_np.shape[1] * 0.1:
        logger.warning(f"VERDICT: Only {active_dims}/{emb_np.shape[1]} active dims — embedding space underutilized")
    else:
        logger.info("VERDICT: Embeddings show healthy variance")

    return {
        "norm_range": (float(norms.min()), float(norms.max())),
        "mean_var": float(mean_var),
        "active_dims": int(active_dims),
        "mean_cosine_sim": float(off_diag.mean()),
    }


# ---------------------------------------------------------------------------
# Test 4: Noise prediction delta
# ---------------------------------------------------------------------------

def test_noise_prediction_delta(trainer, embeddings, canonical, device):
    """Fix x_t and t, vary only the conditioning — measure prediction change.

    This directly tests whether the network's output depends on the conditioning.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 4: Noise Prediction Delta (fixed x_t, varying cond)")
    logger.info("=" * 60)

    denoiser = trainer.model
    denoiser.eval()

    # Pick a fixed noisy sample
    x_0 = canonical[0:1].to(device)
    n_conds = min(30, len(embeddings))

    deltas_by_t = {}

    for t_val in [0, 10, 25, 50, 75, 99]:
        t = torch.tensor([t_val], device=device)
        x_t, _ = trainer.schedule.q_sample(x_0, t)
        x_t = x_t.expand(n_conds, -1)
        t = t.expand(n_conds)

        cond = embeddings[:n_conds].to(device)

        with torch.no_grad():
            preds = denoiser(x_t, t, cond)  # (n_conds, geom_dim)

        preds_np = preds.cpu().numpy()
        pred_var = np.var(preds_np, axis=0).mean()
        pred_range = np.ptp(preds_np, axis=0).mean()

        # Compare to zero-conditioning
        zero_cond = torch.zeros_like(cond)
        with torch.no_grad():
            pred_zero = denoiser(x_t, t, zero_cond)

        diff_from_zero = (preds - pred_zero).abs().mean().item()

        deltas_by_t[t_val] = {
            "pred_variance": float(pred_var),
            "pred_range": float(pred_range),
            "diff_from_zero": float(diff_from_zero),
        }

        logger.info(
            f"  t={t_val:3d}: pred_var={pred_var:.6f} "
            f"range={pred_range:.4f} "
            f"diff_from_zero_cond={diff_from_zero:.4f}"
        )

    avg_var = np.mean([d["pred_variance"] for d in deltas_by_t.values()])
    avg_diff = np.mean([d["diff_from_zero"] for d in deltas_by_t.values()])

    if avg_var < 1e-4:
        logger.warning("VERDICT: Noise predictions nearly identical across conditions — FiLM is dead")
    elif avg_diff < 1e-3:
        logger.warning("VERDICT: Predictions same as zero-conditioning — FiLM ignored")
    else:
        logger.info(f"VERDICT: Conditioning affects predictions (avg var={avg_var:.6f}, avg diff from zero={avg_diff:.4f})")

    return deltas_by_t


# ---------------------------------------------------------------------------
# Test 5: Unconditional vs conditional generation comparison
# ---------------------------------------------------------------------------

def test_unconditional_vs_conditional(trainer, embeddings, device):
    """Generate with real embeddings vs zero vectors — compare outputs.

    If they're the same, conditioning has no effect.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST 5: Unconditional vs Conditional Generation")
    logger.info("=" * 60)

    n_samples = 20

    # Conditional: use diverse embeddings
    cond_outputs = []
    for i in range(min(n_samples, len(embeddings))):
        out = trainer.sample(embeddings[i].to(device), n_samples=1)
        cond_outputs.append(out.cpu().numpy())
    cond_outputs = np.array(cond_outputs).squeeze(1)  # (n, geom_dim)

    # Unconditional: zero embedding
    zero_cond = torch.zeros(1, embeddings.shape[1]).to(device)
    uncond_outputs = trainer.sample(zero_cond, n_samples=n_samples).cpu().numpy()

    # Compare distributions
    cond_mean = cond_outputs.mean(axis=0)
    uncond_mean = uncond_outputs.mean(axis=0)
    mean_diff = np.linalg.norm(cond_mean - uncond_mean)

    cond_var = np.var(cond_outputs, axis=0).mean()
    uncond_var = np.var(uncond_outputs, axis=0).mean()

    logger.info(f"  Conditional mean norm:     {np.linalg.norm(cond_mean):.4f}")
    logger.info(f"  Unconditional mean norm:   {np.linalg.norm(uncond_mean):.4f}")
    logger.info(f"  Mean difference L2:        {mean_diff:.4f}")
    logger.info(f"  Conditional variance:      {cond_var:.6f}")
    logger.info(f"  Unconditional variance:    {uncond_var:.6f}")

    if mean_diff < 0.01:
        logger.warning("VERDICT: Conditional and unconditional outputs nearly identical — conditioning collapsed")
    else:
        logger.info(f"VERDICT: Conditioning shifts output distribution (diff={mean_diff:.4f})")

    return {
        "mean_diff": float(mean_diff),
        "cond_var": float(cond_var),
        "uncond_var": float(uncond_var),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diagnose FiLM conditioning collapse")
    parser.add_argument("--checkpoint", required=True, help="Encoder checkpoint")
    parser.add_argument("--diffusion-checkpoint", required=True, help="Diffusion model checkpoint")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    logger.info("Loading models...")
    trainer, embeddings, canonical, dataset, device = load_models(args)

    logger.info(f"Loaded {len(embeddings)} embeddings, {len(canonical)} geometries")
    logger.info(f"Device: {device}")
    logger.info("")

    # Run all diagnostics
    r1 = test_film_magnitudes(trainer, embeddings, device)
    r2 = test_conditioning_sensitivity(trainer, embeddings, device)
    r3 = test_embedding_variance(embeddings, dataset)
    r4 = test_noise_prediction_delta(trainer, embeddings, canonical, device)
    r5 = test_unconditional_vs_conditional(trainer, embeddings, device)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    issues = []

    avg_gamma = np.mean([r1[l]["gamma_mean"] for l in r1])
    if avg_gamma < 0.01:
        issues.append("FiLM gamma near zero")
    if r2["ratio"] < 0.1:
        issues.append("Between-condition variance < 10%")
    if r2["separation_ratio"] < 1.5:
        issues.append("Poor inter/intra-condition separation")
    if r3["mean_cosine_sim"] > 0.95:
        issues.append("Embeddings nearly identical")
    if r3["active_dims"] < 13:  # < 10% of 128
        issues.append("Embedding space underutilized")

    avg_pred_var = np.mean([r4[t]["pred_variance"] for t in r4])
    if avg_pred_var < 1e-4:
        issues.append("Noise predictions identical across conditions")

    if r5["mean_diff"] < 0.01:
        issues.append("Conditional = unconditional output")

    if not issues:
        logger.info("No conditioning collapse detected. FiLM appears healthy.")
        logger.info("If generation quality is still low, investigate:")
        logger.info("  - Training duration (try more epochs)")
        logger.info("  - Canonical space quality")
        logger.info("  - Data quantity / diversity")
    else:
        logger.warning(f"Found {len(issues)} issue(s):")
        for issue in issues:
            logger.warning(f"  - {issue}")

        if "Embeddings nearly identical" in issues or "Embedding space underutilized" in issues:
            logger.warning("")
            logger.warning("ROOT CAUSE likely in ENCODER, not diffusion model.")
            logger.warning("Fix: Check encoder training, verify contrastive loss produces diverse embeddings.")
        elif "FiLM gamma near zero" in issues or "Noise predictions identical" in issues:
            logger.warning("")
            logger.warning("ROOT CAUSE likely in DIFFUSION TRAINING.")
            logger.warning("Fix options:")
            logger.warning("  1. Add conditioning dropout (10-20%) for classifier-free guidance")
            logger.warning("  2. Increase learning rate for FiLM parameters")
            logger.warning("  3. Add auxiliary embedding reconstruction loss")
            logger.warning("  4. Check if Stage 1 pretraining freezes FiLM weights (it shouldn't)")


if __name__ == "__main__":
    main()
