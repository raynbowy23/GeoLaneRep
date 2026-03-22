"""Joint training loop for contrastive + temporal lane encoder.

Trains both objectives simultaneously through a single trainable encoder:
  - Contrastive (InfoNCE + role regression): shapes encoder to produce
    structurally meaningful lane embeddings.
  - Temporal (BCE on synthetic anomalies): shapes encoder + GRU to detect
    per-window behavioral changes.

Total loss = alpha * temporal_loss + beta * contrastive_loss
"""

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
from src.models.joint_encoder import JointLaneEncoder
from src.training.contrastive import ContrastiveLaneLoss
from src.training.temporal_trainer import inject_anomalies

logger = logging.getLogger(__name__)


class JointTrainer:
    """Joint contrastive + temporal training loop.

    Uses TemporalLaneDataset as the primary dataset. Both InfoNCE and
    temporal BCE losses shape the shared trainable encoder.

    Args:
        config_path: Path to YAML config file.
        encoder_checkpoint: Optional path to pre-trained contrastive encoder
            checkpoint for warm-start initialization.
    """

    def __init__(self, config_path: str, encoder_checkpoint: str = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.encoder_checkpoint = encoder_checkpoint
        self._setup()

    def _setup(self):
        sys_cfg = self.config.get("system", {})
        joint_cfg = self.config.get("joint_training", {})
        temporal_cfg = self.config.get("temporal", {})
        model_cfg = self.config.get("model", {})
        exp_cfg = self.config.get("experiment", {})

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

        # Build LaneEncoder
        lane_encoder = LaneEncoder(
            polyline_k=model_cfg.get("polyline_k", 16),
            d_model=model_cfg.get("polyline_encoder_dim", 64),
            embed_dim=model_cfg.get("embed_dim", 128),
            proj_dim=model_cfg.get("proj_dim", 64),
            polyline_mode=model_cfg.get("polyline_encoder", "transformer"),
            polyline_layers=model_cfg.get("polyline_encoder_layers", 2),
            polyline_heads=model_cfg.get("polyline_encoder_heads", 4),
            stats_dim=model_cfg.get("stats_dim", 9),
            geometry_dropout=model_cfg.get("geometry_dropout", 0.5),
            dropout=model_cfg.get("dropout", 0.1),
            use_cross_lane_attention=False,  # Cross-attn requires grouped batching incompatible with temporal windows
        )

        # Optional warm-start from contrastive checkpoint
        if self.encoder_checkpoint:
            logger.info(f"Warm-starting encoder from {self.encoder_checkpoint}")
            checkpoint = torch.load(
                self.encoder_checkpoint, map_location=self.device
            )
            lane_encoder.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )

        embed_dim = model_cfg.get("embed_dim", 128)

        # Build joint model (all params trainable)
        self.model = JointLaneEncoder(
            lane_encoder=lane_encoder,
            embed_dim=embed_dim,
            gru_layers=temporal_cfg.get("gru_layers", 1),
            dropout=temporal_cfg.get("dropout", 0.1),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"JointLaneEncoder: {n_params:,} trainable params")

        # Contrastive loss (reuse existing)
        train_cfg = self.config.get("contrastive_training", {})
        self.contrastive_criterion = ContrastiveLaneLoss(
            temperature=train_cfg.get("temperature", 0.07),
            role_weight=joint_cfg.get("role_weight", 2.0),
            total_epochs=joint_cfg.get("epochs", 200),
        )

        # Optimizer: all params
        lr = float(joint_cfg.get("learning_rate", 3e-4))
        wd = float(joint_cfg.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # Scheduler
        self.epochs = joint_cfg.get("epochs", 200)
        warmup = joint_cfg.get("warmup_epochs", 10)
        self.warmup_epochs = warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.epochs - warmup, 1)
        )

        # Loss weights
        self.temporal_weight = joint_cfg.get("temporal_weight", 1.0)
        self.contrastive_weight = joint_cfg.get("contrastive_weight", 1.0)

        # Training config
        self.batch_size = joint_cfg.get("batch_size", 16)
        self.anomaly_ratio = joint_cfg.get("anomaly_ratio", 0.3)
        self.grad_clip = joint_cfg.get("grad_clip_norm", 1.0)
        self.val_freq = joint_cfg.get("val_frequency", 5)
        self.ckpt_freq = joint_cfg.get("checkpoint_frequency", 10)

        # Paths
        self.save_dir = Path(exp_cfg.get("saving_path", "./results"))
        self.exp_name = joint_cfg.get("experiment_name",
                                      exp_cfg.get("experiment_name", "joint_encoder"))
        self.exp_dir = self.save_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.history = defaultdict(list)

    def _load_data(
        self, held_out_cameras: Optional[List[str]] = None
    ) -> TemporalLaneDataset:
        """Load temporal lane dataset with positive pair mining."""
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("contrastive_training", {})

        # Exclude held-out cameras from training
        cameras = None
        if held_out_cameras:
            data_cfg = self.config.get("data", {})
            camera_list_path = Path(data_cfg.get(
                "camera_locations", "./dataset/camera_location_list.txt"
            ))
            if camera_list_path.exists():
                all_cameras = [
                    l.strip() for l in camera_list_path.read_text().splitlines()
                    if l.strip()
                ]
            else:
                annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
                all_cameras = sorted(
                    d.name for d in annot_dir.iterdir()
                    if d.is_dir() and (d / "annotation.json").exists()
                )
            cameras = [c for c in all_cameras if c not in held_out_cameras]
            logger.info(
                f"Held out cameras: {held_out_cameras}. "
                f"Training on {len(cameras)} cameras."
            )

        dataset = TemporalLaneDataset(
            config=self.config,
            cameras=cameras,
            polyline_k=model_cfg.get("polyline_k", 16),
            max_traj_per_window=50,
            role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
        )
        return dataset

    def _train_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        positive_pairs: List[Tuple[int, int]],
    ) -> dict:
        """Run one training epoch with both losses."""
        self.model.train()
        epoch_metrics = defaultdict(float)
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
            indices = batch["idx"]

            # 1. Inject synthetic anomalies for temporal loss
            corrupt_poly, corrupt_mask, corrupt_stats, labels = inject_anomalies(
                window_traj_polylines,
                window_traj_mask,
                window_traj_stats,
                window_valid,
                anomaly_ratio=self.anomaly_ratio,
                rng=self._rng,
            )

            # 2. Forward through JointLaneEncoder
            output = self.model(
                geometry=geometry,
                window_traj_polylines=corrupt_poly,
                window_traj_mask=corrupt_mask,
                window_traj_stats=corrupt_stats,
                window_valid=window_valid,
                roles=roles,
            )

            # 3. Temporal loss: BCE on anomaly scores (valid windows only)
            anomaly_logits = output["anomaly_scores"]
            valid_mask_float = window_valid.float()
            temporal_loss = F.binary_cross_entropy_with_logits(
                anomaly_logits, labels, weight=valid_mask_float, reduction="sum"
            )
            n_valid = valid_mask_float.sum().clamp(min=1.0)
            temporal_loss = temporal_loss / n_valid

            # 4. Window-level contrastive loss: InfoNCE per valid window, averaged
            window_proj = output["window_projections"]  # (B, W, proj_dim)
            W = window_proj.shape[1]

            window_ctr_losses = []
            window_ctr_metrics_accum = defaultdict(float)
            n_valid_windows = 0

            for w in range(W):
                # Only compute on windows where enough lanes are valid
                w_valid = window_valid[:, w]  # (B,)
                if w_valid.sum() < 2:
                    continue

                w_valid_cpu = w_valid.cpu()
                w_proj = window_proj[w_valid, w]  # (B_valid, proj_dim)
                w_indices = indices[w_valid_cpu]

                # Role regression only on first window to avoid redundant computation
                if w == 0:
                    w_loss, w_metrics = self.contrastive_criterion(
                        w_proj, w_indices, positive_pairs,
                        roles=roles[w_valid],
                        pred_rank=output["pred_rank"][w_valid],
                        pred_edge=output["pred_edge"][w_valid],
                        pred_size=output["pred_size"][w_valid],
                        epoch=epoch,
                    )
                else:
                    w_loss, w_metrics = self.contrastive_criterion(
                        w_proj, w_indices, positive_pairs,
                        roles=None, pred_rank=None, pred_edge=None, pred_size=None,
                        epoch=epoch,
                    )

                if not (torch.isnan(w_loss) or torch.isinf(w_loss)):
                    window_ctr_losses.append(w_loss)
                    for mk, mv in w_metrics.items():
                        window_ctr_metrics_accum[mk] += mv
                    n_valid_windows += 1

            if window_ctr_losses:
                contrastive_loss = torch.stack(window_ctr_losses).mean()
                ctr_metrics = {
                    k: v / n_valid_windows
                    for k, v in window_ctr_metrics_accum.items()
                }
            else:
                contrastive_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                ctr_metrics = {}

            # 5. Total loss
            total_loss = (
                self.temporal_weight * temporal_loss
                + self.contrastive_weight * contrastive_loss
            )

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.optimizer.zero_grad()
                continue

            # 6. Backward through everything
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.optimizer.step()

            # Metrics
            with torch.no_grad():
                preds = (torch.sigmoid(anomaly_logits) > 0.5).float()
                valid_preds = preds[window_valid]
                valid_labels = labels[window_valid]
                if len(valid_labels) > 0:
                    accuracy = (valid_preds == valid_labels).float().mean().item()
                else:
                    accuracy = 0.0

            epoch_metrics["total_loss"] += total_loss.item()
            epoch_metrics["temporal_loss"] += temporal_loss.item()
            epoch_metrics["contrastive_loss"] += ctr_metrics.get("contrastive_loss", 0)
            epoch_metrics["role_loss"] += ctr_metrics.get("role_loss", 0)
            epoch_metrics["anomaly_accuracy"] += accuracy
            epoch_metrics["mean_pos_sim"] += ctr_metrics.get("mean_pos_sim", 0)
            epoch_metrics["mean_neg_sim"] += ctr_metrics.get("mean_neg_sim", 0)
            epoch_metrics["n_positive_pairs"] += ctr_metrics.get("n_positive_pairs", 0)
            n_batches += 1

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                acc=f"{accuracy:.3f}",
                pos=f"{ctr_metrics.get('mean_pos_sim', 0):.3f}",
            )

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)
        return dict(epoch_metrics)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "lane_encoder_state_dict": self.model.lane_encoder.state_dict(),
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

    def run(self, held_out_cameras: Optional[List[str]] = None):
        """Full joint training pipeline."""
        logger.info("Loading temporal lane dataset with positive pair mining...")
        dataset = self._load_data(held_out_cameras)

        num_workers = self.config.get("system", {}).get("num_workers", 0)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=temporal_collate_fn,
            num_workers=num_workers,
            drop_last=True,
        )

        joint_cfg = self.config.get("joint_training", {})
        lr = float(joint_cfg.get("learning_rate", 3e-4))

        held_out_str = ", ".join(held_out_cameras) if held_out_cameras else "none"
        logger.info(
            f"Joint training: {len(dataset)} lanes, "
            f"{dataset.n_windows} windows/lane, "
            f"{len(dataset.positive_pairs)} positive pairs, "
            f"held out: {held_out_str}, "
            f"alpha={self.temporal_weight}, beta={self.contrastive_weight}"
        )

        # MLflow
        mlflow.set_tracking_uri(str(self.save_dir / "mlruns"))
        mlflow.set_experiment(self.exp_name)

        best_acc = 0.0

        with mlflow.start_run(run_name=f"joint_{time.strftime('%m%d_%H%M')}"):
            mlflow.log_params({
                f"joint.{k}": v
                for k, v in joint_cfg.items()
            })
            if self.encoder_checkpoint:
                mlflow.log_param("warm_start", self.encoder_checkpoint)

            for epoch in range(self.epochs):
                t0 = time.time()

                # Warmup LR
                if epoch < self.warmup_epochs:
                    warmup_factor = (epoch + 1) / self.warmup_epochs
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr * warmup_factor

                # Train
                metrics = self._train_epoch(
                    epoch, loader, dataset.positive_pairs
                )
                dt = time.time() - t0

                for k, v in metrics.items():
                    self.history[f"train/{k}"].append(v)

                lr_current = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"total={metrics['total_loss']:.4f} "
                    f"temp={metrics['temporal_loss']:.4f} "
                    f"ctr={metrics['contrastive_loss']:.4f} "
                    f"role={metrics['role_loss']:.4f} "
                    f"acc={metrics['anomaly_accuracy']:.3f} "
                    f"pos_sim={metrics['mean_pos_sim']:.3f} "
                    f"neg_sim={metrics['mean_neg_sim']:.3f} "
                    f"lr={lr_current:.6f} | {dt:.1f}s"
                )

                mlflow.log_metrics(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=epoch + 1,
                )

                # Checkpoint
                is_best = metrics["anomaly_accuracy"] > best_acc
                if is_best:
                    best_acc = metrics["anomaly_accuracy"]

                if (epoch + 1) % self.ckpt_freq == 0:
                    self._save_checkpoint(epoch + 1, is_best=is_best)

                # LR schedule
                if epoch >= self.warmup_epochs:
                    self.scheduler.step()

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
                f"Joint training complete. Best anomaly accuracy: {best_acc:.3f}. "
                f"Results in {self.exp_dir}"
            )
