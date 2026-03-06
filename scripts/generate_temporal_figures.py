#!/usr/bin/env python3
"""Generate temporal encoder figures for the paper.

Figures:
    2a — UMAP of per-window embeddings colored by time window index
    2b — Anomaly score timeline with injected incident overlay
    2c — Embedding delta heatmap ||e(t) - e(t-1)|| per lane per window

Usage:
    python scripts/generate_temporal_figures.py \
        --config configs/lane_contrastive.yaml \
        --checkpoint results/temporal_encoder/checkpoints/best.pt \
        --encoder-checkpoint results/lane_contrastive/checkpoints/best.pt \
        --output-dir results/temporal_encoder/figures
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_model_and_data(args):
    """Load temporal model, encoder, and dataset."""
    import yaml

    from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
    from src.models.lane_encoder import LaneEncoder
    from src.models.temporal_encoder import LaneTemporalEncoder
    from src.training.temporal_trainer import inject_anomalies

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    enc_ckpt = torch.load(args.encoder_checkpoint, map_location=device)
    enc_config = enc_ckpt.get("config", config)
    mc = enc_config.get("model", {})

    lane_encoder = LaneEncoder(
        polyline_k=mc.get("polyline_k", 16),
        d_model=mc.get("polyline_encoder_dim", 64),
        embed_dim=mc.get("embed_dim", 128),
        proj_dim=mc.get("proj_dim", 64),
        polyline_mode=mc.get("polyline_encoder", "transformer"),
        polyline_layers=mc.get("polyline_encoder_layers", 2),
        polyline_heads=mc.get("polyline_encoder_heads", 4),
        stats_dim=mc.get("stats_dim", 9),
        geometry_dropout=0.0,
        dropout=mc.get("dropout", 0.1),
        use_cross_lane_attention=False,
    )
    lane_encoder.load_state_dict(enc_ckpt["model_state_dict"], strict=False)

    tc = config.get("temporal", {})
    embed_dim = mc.get("embed_dim", 128)

    model = LaneTemporalEncoder(
        lane_encoder=lane_encoder,
        embed_dim=embed_dim,
        freeze_encoder=True,
        gru_layers=tc.get("gru_layers", 1),
        dropout=tc.get("dropout", 0.1),
    ).to(device)

    # Load temporal checkpoint
    temporal_ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(temporal_ckpt["model_state_dict"])
    model.eval()

    # Dataset
    dataset = TemporalLaneDataset(
        config=config,
        polyline_k=mc.get("polyline_k", 16),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get("temporal", {}).get("batch_size", 16),
        shuffle=False,
        collate_fn=temporal_collate_fn,
    )

    return model, dataset, loader, config, device


# ── Figure 2a: UMAP of per-window embeddings ────────────────────────


def figure_2a(model, loader, device, output_dir: Path):
    """UMAP of per-window embeddings colored by time window index.

    Shows how lane representations drift over time windows.
    """
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed, skipping figure 2a")
        return

    all_embeddings = []
    all_window_ids = []
    all_lane_keys = []

    with torch.no_grad():
        for batch in loader:
            output = model(
                geometry=batch["geometry"].to(device),
                window_traj_polylines=batch["window_traj_polylines"].to(device),
                window_traj_mask=batch["window_traj_mask"].to(device),
                window_traj_stats=batch["window_traj_stats"].to(device),
                window_valid=batch["window_valid"].to(device),
                roles=batch["roles"].to(device),
            )
            # window_embeddings: (B, W, D)
            emb = output["window_embeddings"].cpu().numpy()
            valid = batch["window_valid"].numpy()
            B, W, D = emb.shape

            for b in range(B):
                for w in range(W):
                    if valid[b, w]:
                        all_embeddings.append(emb[b, w])
                        all_window_ids.append(w)
                        all_lane_keys.append(batch["lane_keys"][b])

    if not all_embeddings:
        logger.warning("No valid embeddings for figure 2a")
        return

    embeddings = np.stack(all_embeddings)
    window_ids = np.array(all_window_ids)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15)
    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=window_ids, cmap="viridis", s=10, alpha=0.6,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Window Index (time)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("2a: Per-Window Embeddings Colored by Time")
    fig.tight_layout()

    path = output_dir / "2a_umap_temporal.pdf"
    fig.savefig(str(path), dpi=300)
    fig.savefig(str(path.with_suffix(".png")), dpi=150)
    plt.close(fig)
    logger.info(f"Saved 2a to {path}")


# ── Figure 2b: Anomaly score timeline ───────────────────────────────


def figure_2b(model, loader, device, config, output_dir: Path):
    """Anomaly score timeline with injected incident overlay.

    Shows anomaly scores over time for a few example lanes, with
    synthetically injected anomalies marked.
    """
    from src.training.temporal_trainer import inject_anomalies

    tc = config.get("temporal", {})
    window_stride = tc.get("window_stride_sec", 5.0)
    rng = np.random.default_rng(42)

    # Get first batch
    batch = next(iter(loader))

    geometry = batch["geometry"].to(device)
    poly = batch["window_traj_polylines"].to(device)
    mask = batch["window_traj_mask"].to(device)
    stats = batch["window_traj_stats"].to(device)
    valid = batch["window_valid"].to(device)
    roles = batch["roles"].to(device)

    # Get clean scores
    with torch.no_grad():
        clean_out = model(geometry, poly, mask, stats, valid, roles)
        clean_scores = torch.sigmoid(clean_out["anomaly_scores"]).cpu().numpy()

    # Inject anomalies and get corrupted scores
    corrupt_poly, corrupt_mask, corrupt_stats, labels = inject_anomalies(
        poly, mask, stats, valid, anomaly_ratio=0.3, rng=rng,
    )
    with torch.no_grad():
        anom_out = model(geometry, corrupt_poly, corrupt_mask, corrupt_stats, valid, roles)
        anom_scores = torch.sigmoid(anom_out["anomaly_scores"]).cpu().numpy()

    labels_np = labels.cpu().numpy()
    valid_np = batch["window_valid"].numpy()

    B, W = clean_scores.shape
    n_show = min(4, B)
    time_axis = np.arange(W) * window_stride

    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        v = valid_np[i]

        # Clean scores
        ax.plot(
            time_axis[v], clean_scores[i, v],
            "b-o", markersize=4, label="Clean", alpha=0.7,
        )
        # Anomaly scores
        ax.plot(
            time_axis[v], anom_scores[i, v],
            "r-s", markersize=4, label="With anomalies", alpha=0.7,
        )

        # Shade injected anomaly windows
        for w in range(W):
            if labels_np[i, w] > 0.5:
                ax.axvspan(
                    time_axis[w] - window_stride / 2,
                    time_axis[w] + window_stride / 2,
                    alpha=0.2, color="red", label="Injected" if w == 0 else None,
                )

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Anomaly Score")
        ax.set_title(f"Lane: {batch['lane_keys'][i]}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("2b: Anomaly Score Timeline", fontsize=13, y=1.01)
    fig.tight_layout()

    path = output_dir / "2b_anomaly_timeline.pdf"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 2b to {path}")


# ── Figure 2c: Embedding delta heatmap ──────────────────────────────


def figure_2c(model, loader, device, config, output_dir: Path):
    """Embedding delta heatmap: ||e(t) - e(t-1)|| per lane per window.

    Rows = lanes, columns = window transitions. Highlights temporal instability.
    """
    tc = config.get("temporal", {})
    window_stride = tc.get("window_stride_sec", 5.0)

    all_deltas = []
    all_lane_keys = []

    with torch.no_grad():
        for batch in loader:
            output = model(
                geometry=batch["geometry"].to(device),
                window_traj_polylines=batch["window_traj_polylines"].to(device),
                window_traj_mask=batch["window_traj_mask"].to(device),
                window_traj_stats=batch["window_traj_stats"].to(device),
                window_valid=batch["window_valid"].to(device),
                roles=batch["roles"].to(device),
            )
            emb = output["window_embeddings"].cpu()  # (B, W, D)
            valid = batch["window_valid"]
            B, W, D = emb.shape

            # Compute ||e(t) - e(t-1)|| for consecutive windows
            deltas = torch.norm(emb[:, 1:] - emb[:, :-1], dim=-1)  # (B, W-1)

            for b in range(B):
                all_deltas.append(deltas[b].numpy())
                all_lane_keys.append(batch["lane_keys"][b])

    if not all_deltas:
        logger.warning("No data for figure 2c")
        return

    delta_matrix = np.stack(all_deltas)  # (N_lanes, W-1)
    W_minus_1 = delta_matrix.shape[1]

    # Sort by mean delta for visual clarity
    sort_idx = np.argsort(delta_matrix.mean(axis=1))[::-1]
    delta_matrix = delta_matrix[sort_idx]
    sorted_keys = [all_lane_keys[i] for i in sort_idx]

    # Limit to top 30 lanes for readability
    n_show = min(30, len(delta_matrix))
    delta_matrix = delta_matrix[:n_show]
    sorted_keys = sorted_keys[:n_show]

    fig, ax = plt.subplots(figsize=(10, max(4, n_show * 0.25)))

    transition_labels = [
        f"{i * window_stride:.0f}→{(i + 1) * window_stride:.0f}s"
        for i in range(W_minus_1)
    ]

    im = ax.imshow(delta_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="||e(t) - e(t-1)||")

    ax.set_xticks(range(W_minus_1))
    ax.set_xticklabels(transition_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(sorted_keys, fontsize=6)

    ax.set_xlabel("Window Transition")
    ax.set_ylabel("Lane")
    ax.set_title("2c: Embedding Delta Heatmap (temporal instability)")
    fig.tight_layout()

    path = output_dir / "2c_delta_heatmap.pdf"
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved 2c to {path}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate temporal encoder figures")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to temporal encoder checkpoint",
    )
    parser.add_argument(
        "--encoder-checkpoint", required=True,
        help="Path to pre-trained LaneEncoder checkpoint",
    )
    parser.add_argument(
        "--output-dir", default="results/temporal_encoder/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figures", nargs="+", default=["2a", "2b", "2c"],
        choices=["2a", "2b", "2c"],
        help="Which figures to generate (default: all)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, dataset, loader, config, device = _load_model_and_data(args)

    if "2a" in args.figures:
        logger.info("Generating 2a: UMAP of temporal embeddings...")
        figure_2a(model, loader, device, output_dir)

    if "2b" in args.figures:
        logger.info("Generating 2b: Anomaly score timeline...")
        figure_2b(model, loader, device, config, output_dir)

    if "2c" in args.figures:
        logger.info("Generating 2c: Embedding delta heatmap...")
        figure_2c(model, loader, device, config, output_dir)

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
