import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.temporal_dataset import TemporalLaneDataset, temporal_collate_fn
from src.models.lane_encoder import LaneEncoder
from src.models.temporal_encoder import LaneTemporalEncoder

logger = logging.getLogger(__name__)

def inject_anomalies(
    window_traj_polylines: torch.Tensor,
    window_traj_mask: torch.Tensor,
    window_traj_stats: torch.Tensor,
    window_valid: torch.Tensor,
    anomaly_ratio: float = 0.3,
    rng: np.random.Generator = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inject synthetic anomalies into windowed trajectory data.

    Anomaly types:
        1. Speed anomaly: Scale trajectory polylines by 0.2x (compress movement).
        2. Count anomaly: Zero out 90% of trajectories (simulate blocked lane).
        3. Lateral anomaly: Add random offset to trajectories (swerving).

    Args:
        window_traj_polylines: (B, W, T, K, 2) trajectory data.
        window_traj_mask: (B, W, T) validity mask.
        window_traj_stats: (B, W, 4) per-window stats.
        window_valid: (B, W) window validity.
        anomaly_ratio: Fraction of windows to corrupt per sample.
        rng: Random number generator.

    Returns:
        (corrupted_polylines, corrupted_mask, corrupted_stats, labels)
        labels: (B, W) float tensor, 1.0 = anomalous, 0.0 = normal.
    """
    if rng is None:
        rng = np.random.default_rng()

    B, W, T, K, _ = window_traj_polylines.shape
    device = window_traj_polylines.device

    # Clone inputs
    poly = window_traj_polylines.clone()
    mask = window_traj_mask.clone()
    stats = window_traj_stats.clone()
    labels = torch.zeros(B, W, device=device)

    for b in range(B):
        # Select windows to corrupt (only valid ones)
        valid_windows = window_valid[b].nonzero(as_tuple=True)[0].cpu().numpy()
        if len(valid_windows) == 0:
            continue

        n_corrupt = max(1, int(len(valid_windows) * anomaly_ratio))
        n_corrupt = min(n_corrupt, 3)  # Cap at 3 windows
        corrupt_windows = rng.choice(valid_windows, size=n_corrupt, replace=False)

        for w in corrupt_windows:
            labels[b, w] = 1.0

            # Choose anomaly type
            anomaly_type = rng.integers(0, 3)

            if anomaly_type == 0:
                # Speed anomaly: compress trajectory movement to 20%
                # Shrink toward centroid of each trajectory
                valid_trajs = mask[b, w].nonzero(as_tuple=True)[0]
                if len(valid_trajs) > 0:
                    centroid = poly[b, w, valid_trajs].mean(dim=(0, 1), keepdim=True)
                    poly[b, w, valid_trajs] = (
                        centroid + 0.2 * (poly[b, w, valid_trajs] - centroid)
                    )
                    # Update stats: reduce speed
                    stats[b, w, 0] *= 0.2

            elif anomaly_type == 1:
                # Count anomaly: zero out 90% of trajectories
                valid_trajs = mask[b, w].nonzero(as_tuple=True)[0]
                if len(valid_trajs) > 1:
                    n_keep = max(1, len(valid_trajs) // 10)
                    keep_idx = torch.tensor(
                        rng.choice(valid_trajs.cpu().numpy(), n_keep, replace=False),
                        device=device,
                    )
                    drop_mask = torch.ones(T, dtype=torch.bool, device=device)
                    drop_mask[keep_idx] = False
                    mask[b, w] = mask[b, w] & ~drop_mask
                    poly[b, w, drop_mask] = 0.0
                    # Update stats: reduce count
                    stats[b, w, 3] *= 0.1

            else:
                # Lateral anomaly: add random offset (swerving)
                valid_trajs = mask[b, w].nonzero(as_tuple=True)[0]
                if len(valid_trajs) > 0:
                    offset = torch.tensor(
                        rng.normal(0, 0.02, size=2),
                        dtype=torch.float32,
                        device=device,
                    )
                    poly[b, w, valid_trajs] = poly[b, w, valid_trajs] + offset
                    # Update stats: increase lateral offset
                    stats[b, w, 2] += abs(offset[0].item()) + abs(offset[1].item())

    return poly, mask, stats, labels

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TemporalTrainer:
    """Training loop for temporal lane encoder.

    Trains GRU + anomaly_head on synthetic anomaly detection, with the
    underlying LaneEncoder frozen.
    """

    def __init__(self, config_path: str, encoder_checkpoint: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.encoder_checkpoint = encoder_checkpoint
        self._setup()

    def _setup(self):
        sys_cfg = self.config.get("system", {})
        temporal_cfg = self.config.get("temporal", {})
        exp_cfg = self.config.get("experiment", {})
        model_cfg = self.config.get("model", {})

        # Device
        device_str = sys_cfg.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        logger.info(f"Device: {self.device}")

        # Seed
        seed = sys_cfg.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._rng = np.random.default_rng(seed)

        # Load pre-trained LaneEncoder
        logger.info(f"Loading encoder from {self.encoder_checkpoint}")
        checkpoint = torch.load(self.encoder_checkpoint, map_location=self.device)
        ckpt_config = checkpoint.get("config", self.config)

        ckpt_model_cfg = ckpt_config.get("model", {})
        lane_encoder = LaneEncoder(
            polyline_k=ckpt_model_cfg.get("polyline_k", 16),
            d_model=ckpt_model_cfg.get("polyline_encoder_dim", 64),
            embed_dim=ckpt_model_cfg.get("embed_dim", 128),
            proj_dim=ckpt_model_cfg.get("proj_dim", 64),
            polyline_mode=ckpt_model_cfg.get("polyline_encoder", "transformer"),
            polyline_layers=ckpt_model_cfg.get("polyline_encoder_layers", 2),
            polyline_heads=ckpt_model_cfg.get("polyline_encoder_heads", 4),
            stats_dim=ckpt_model_cfg.get("stats_dim", 9),
            geometry_dropout=0.0,  # No geometry dropout at temporal stage
            dropout=ckpt_model_cfg.get("dropout", 0.1),
            use_cross_lane_attention=False,  # Per-lane encoding only
        )
        lane_encoder.load_state_dict(checkpoint["model_state_dict"], strict=False)

        embed_dim = ckpt_model_cfg.get("embed_dim", 128)

        # Build temporal model
        self.model = LaneTemporalEncoder(
            lane_encoder=lane_encoder,
            embed_dim=embed_dim,
            freeze_encoder=temporal_cfg.get("freeze_encoder", True),
            gru_layers=temporal_cfg.get("gru_layers", 1),
            dropout=temporal_cfg.get("dropout", 0.1),
        ).to(self.device)

        # Count trainable params
        n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Temporal model: {n_trainable:,} trainable / {n_total:,} total params")

        # Optimizer (only trainable params)
        lr = float(temporal_cfg.get("learning_rate", 0.001))
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

        # Training config
        self.epochs = temporal_cfg.get("epochs", 50)
        self.batch_size = temporal_cfg.get("batch_size", 16)
        self.anomaly_ratio = temporal_cfg.get("anomaly_ratio", 0.3)

        # Paths
        self.save_dir = Path(exp_cfg.get("saving_path", "./results"))
        self.exp_name = "temporal_encoder"
        self.exp_dir = self.save_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.history = defaultdict(list)

    def _load_data(self) -> TemporalLaneDataset:
        """Load temporal lane dataset."""
        model_cfg = self.config.get("model", {})
        dataset = TemporalLaneDataset(
            config=self.config,
            polyline_k=model_cfg.get("polyline_k", 16),
            max_traj_per_window=50,
        )
        return dataset

    def _train_epoch(self, epoch: int, loader: DataLoader) -> dict:
        """Run one training epoch with synthetic anomaly injection."""
        self.model.train()
        epoch_losses = defaultdict(float)
        n_batches = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

        for batch in pbar:
            geometry = batch["geometry"].to(self.device)
            window_traj_polylines = batch["window_traj_polylines"].to(self.device)
            window_traj_mask = batch["window_traj_mask"].to(self.device)
            window_traj_stats = batch["window_traj_stats"].to(self.device)
            window_valid = batch["window_valid"].to(self.device)
            roles = batch["roles"].to(self.device)

            # Inject synthetic anomalies
            corrupt_poly, corrupt_mask, corrupt_stats, labels = inject_anomalies(
                window_traj_polylines,
                window_traj_mask,
                window_traj_stats,
                window_valid,
                anomaly_ratio=self.anomaly_ratio,
                rng=self._rng,
            )

            # Forward
            output = self.model(
                geometry=geometry,
                window_traj_polylines=corrupt_poly,
                window_traj_mask=corrupt_mask,
                window_traj_stats=corrupt_stats,
                window_valid=window_valid,
                roles=roles,
            )

            # BCE loss on valid windows only
            anomaly_logits = output["anomaly_scores"]  # (B, W)
            valid_mask = window_valid.float()

            # Masked BCE
            loss = F.binary_cross_entropy_with_logits(
                anomaly_logits, labels, weight=valid_mask, reduction="sum"
            )
            n_valid = valid_mask.sum().clamp(min=1.0)
            loss = loss / n_valid

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                1.0,
            )
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = (torch.sigmoid(anomaly_logits) > 0.5).float()
                valid_preds = preds[window_valid]
                valid_labels = labels[window_valid]
                if len(valid_labels) > 0:
                    accuracy = (valid_preds == valid_labels).float().mean().item()
                    precision_num = ((valid_preds == 1) & (valid_labels == 1)).sum().float()
                    precision_den = (valid_preds == 1).sum().float().clamp(min=1)
                    recall_num = ((valid_preds == 1) & (valid_labels == 1)).sum().float()
                    recall_den = (valid_labels == 1).sum().float().clamp(min=1)
                    precision = (precision_num / precision_den).item()
                    recall = (recall_num / recall_den).item()
                else:
                    accuracy = 0.0
                    precision = 0.0
                    recall = 0.0

            epoch_losses["loss"] += loss.item()
            epoch_losses["accuracy"] += accuracy
            epoch_losses["precision"] += precision
            epoch_losses["recall"] += recall
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.3f}")

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        return dict(epoch_losses)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "encoder_checkpoint": self.encoder_checkpoint,
        }
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch}.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info(f"  Saved best checkpoint at epoch {epoch}")

    def run(self):
        """Full temporal training pipeline."""
        logger.info("Loading temporal lane dataset...")
        dataset = self._load_data()

        num_workers = self.config.get("system", {}).get("num_workers", 0)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=temporal_collate_fn,
            num_workers=num_workers,
            drop_last=True,
        )

        logger.info(
            f"Training: {len(dataset)} lanes, {dataset.n_windows} windows/lane, "
            f"batch_size={self.batch_size}"
        )

        # MLflow
        mlflow.set_tracking_uri(str(self.save_dir / "mlruns"))
        mlflow.set_experiment(self.exp_name)

        best_acc = 0.0

        with mlflow.start_run(run_name=f"temporal_{time.strftime('%m%d_%H%M')}"):
            # Log temporal-specific params
            temporal_cfg = self.config.get("temporal", {})
            mlflow.log_params({
                f"temporal.{k}": v for k, v in temporal_cfg.items()
            })

            for epoch in range(self.epochs):
                t0 = time.time()
                metrics = self._train_epoch(epoch, loader)
                dt = time.time() - t0

                for k, v in metrics.items():
                    self.history[f"train/{k}"].append(v)

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"loss={metrics['loss']:.4f} "
                    f"acc={metrics['accuracy']:.3f} "
                    f"prec={metrics['precision']:.3f} "
                    f"rec={metrics['recall']:.3f} | {dt:.1f}s"
                )

                mlflow.log_metrics(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=epoch + 1,
                )

                # Checkpoint
                is_best = metrics["accuracy"] > best_acc
                if is_best:
                    best_acc = metrics["accuracy"]

                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(epoch + 1, is_best=is_best)

            # Final checkpoint
            self._save_checkpoint(self.epochs, is_best=True)

            # Save history
            history_path = self.exp_dir / "history.json"
            with open(history_path, "w") as f:
                json.dump(
                    {k: [float(x) for x in v] for k, v in self.history.items()},
                    f,
                    indent=2,
                )
            mlflow.log_artifact(str(history_path))

            logger.info(
                f"Training complete. Best accuracy: {best_acc:.3f}. "
                f"Results in {self.exp_dir}"
            )
