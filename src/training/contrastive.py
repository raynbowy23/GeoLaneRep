"""Contrastive training for lane representation learning.

InfoNCE loss with structural positive mining. Positives are lanes with
similar structural roles across different cameras. Negatives are all other
lanes in the batch (including same-camera lanes with different roles).
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.lane_dataset import GroupBatchSampler, LaneDataset, collate_fn
from src.models.lane_encoder import LaneEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class ContrastiveLaneLoss(nn.Module):
    """InfoNCE contrastive loss with structural positive pair mining + role regression.

    For each anchor, positives are lanes with similar structural role from
    different cameras. All other lanes in the batch are negatives.

    Role regression heads provide direct supervision on lateral_rank, edge
    flags, and group_size to prevent representation collapse.

    Args:
        temperature: Softmax temperature for InfoNCE.
        role_weight: Weight for role regression loss.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        role_weight: float = 2.0,
        total_epochs: int = 100,
        group_consistency_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.role_weight = role_weight
        self.total_epochs = total_epochs
        self.group_consistency_weight = group_consistency_weight

    @staticmethod
    def _group_rank_consistency_loss(
        pred_rank: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predicted ranks within a group should be monotonically ordered and uniformly spaced."""
        loss = torch.tensor(0.0, device=pred_rank.device)
        n_groups = 0
        for gid in group_ids.unique():
            mask = group_ids == gid
            group_ranks = torch.sigmoid(pred_rank[mask])
            n = len(group_ranks)
            if n < 2:
                continue
            target = torch.linspace(0, 1, n, device=group_ranks.device)
            sorted_preds, _ = group_ranks.sort()
            loss = loss + F.mse_loss(sorted_preds, target)
            n_groups += 1
        return loss / max(n_groups, 1)

    def forward(
        self,
        projections: torch.Tensor,
        indices: torch.Tensor,
        positive_pairs: List[Tuple[int, int]],
        roles: Optional[torch.Tensor] = None,
        pred_rank: Optional[torch.Tensor] = None,
        pred_edge: Optional[torch.Tensor] = None,
        pred_size: Optional[torch.Tensor] = None,
        epoch: int = 0,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute InfoNCE loss + role regression loss with contrastive warmup.

        Role loss dominates early training to force the encoder to learn
        meaningful rank/edge structure first. Contrastive loss ramps up
        over the first 33% of training for cross-camera alignment.

        Args:
            projections: (B, proj_dim) L2-normalized projection vectors.
            indices: (B,) global sample indices into the dataset.
            positive_pairs: List of (i, j) global index pairs that are positives.
            roles: (B, 5) GT role descriptors [lat_rank, left, right, succ, size].
            pred_rank: (B,) predicted lateral rank.
            pred_edge: (B, 2) predicted is_leftmost, is_rightmost logits.
            pred_size: (B,) predicted group size.
            epoch: Current epoch number (for contrastive warmup schedule).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        B = projections.shape[0]
        device = projections.device

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "contrastive_loss": 0.0,
                "n_positive_pairs": 0,
                "mean_pos_sim": 0.0,
                "mean_neg_sim": 0.0,
            }

        # Build positive pair mask in batch coordinates
        idx_to_batch = {}
        for b, idx in enumerate(indices.tolist()):
            idx_to_batch[idx] = b

        pos_mask = torch.zeros(B, B, dtype=torch.bool, device=device)
        for i, j in positive_pairs:
            if i in idx_to_batch and j in idx_to_batch:
                bi, bj = idx_to_batch[i], idx_to_batch[j]
                pos_mask[bi, bj] = True
                pos_mask[bj, bi] = True

        n_pos = pos_mask.sum().item() // 2  # each pair counted twice

        # Compute contrastive loss only when positive pairs exist in batch
        contrastive_loss = torch.tensor(0.0, device=device)
        pos_sims_val = 0.0
        neg_sims_val = 0.0

        if n_pos > 0:
            # Similarity matrix
            sim = torch.mm(projections, projections.t()) / self.temperature  # (B, B)

            # Mask out self-similarity
            self_mask = torch.eye(B, dtype=torch.bool, device=device)
            sim.masked_fill_(self_mask, -1e9)

            has_pos = pos_mask.any(dim=1)  # (B,)

            if has_pos.any():
                # Log-sum-exp over all non-self entries (denominator)
                neg_mask = ~self_mask  # all except self
                log_denom = torch.logsumexp(sim + neg_mask.float().log(), dim=1)

                # Log-sum-exp over positives (numerator)
                pos_sim = sim.clone()
                pos_sim.masked_fill_(~pos_mask, -1e9)
                log_numer = torch.logsumexp(pos_sim, dim=1)

                per_anchor_loss = log_denom - log_numer
                contrastive_loss = per_anchor_loss[has_pos].mean()

            # Metrics
            with torch.no_grad():
                sim_vals = torch.mm(projections, projections.t())
                pos_sims_val = sim_vals[pos_mask].mean().item() if pos_mask.any() else 0.0
                neg_mask_strict = ~torch.eye(B, dtype=torch.bool, device=device) & ~pos_mask
                neg_sims_val = sim_vals[neg_mask_strict].mean().item() if neg_mask_strict.any() else 0.0

        # Role regression losses (direct supervision to prevent collapse)
        role_loss = torch.tensor(0.0, device=device)
        role_loss_val = 0.0
        if (roles is not None and pred_rank is not None
                and pred_edge is not None and pred_size is not None):
            gt_rank = roles[:, 0]                     # lateral_rank [0,1]
            gt_edge = roles[:, 1:3]                   # is_leftmost, is_rightmost
            gt_size = roles[:, 4]                     # group_size

            # All heads output raw logits → BCE with logits for bounded [0,1] targets.
            # Gives stronger gradients at extremes than MSE(sigmoid(x), y).
            loss_rank = F.binary_cross_entropy_with_logits(pred_rank, gt_rank)
            # Weighted BCE for edge heads: most lanes are interior (not edge),
            # so upweight the minority class to prevent always-false predictions.
            # Typical: ~2/N lanes are edge per group → pos_weight ≈ N/2 - 1.
            edge_pos_weight = torch.tensor([3.0, 3.0], device=device)
            loss_edge = F.binary_cross_entropy_with_logits(
                pred_edge, gt_edge, pos_weight=edge_pos_weight,
            )
            loss_size = F.binary_cross_entropy_with_logits(pred_size, gt_size)
            role_loss = loss_rank + loss_edge + 0.5 * loss_size
            role_loss_val = role_loss.item()

        # 3-phase loss schedule to prevent role & contrastive from fighting:
        #   Phase 1 (0-30%):  role dominates → encoder learns rank/edge first
        #   Phase 2 (30-70%): balanced → contrastive refines cross-camera alignment
        #   Phase 3 (70-100%): contrastive dominates → fine-tune matching quality
        epoch_frac = epoch / max(self.total_epochs, 1)
        if epoch_frac < 0.3:
            ctr_w, role_w = 0.3, self.role_weight       # role_weight=2.0 → 6.7x ratio
        elif epoch_frac < 0.7:
            ctr_w, role_w = 1.0, 1.0                     # balanced
        else:
            ctr_w, role_w = 2.0, 0.5                     # contrastive dominates
        loss = ctr_w * contrastive_loss + role_w * role_loss

        # Group rank consistency loss (only when cross-lane attention provides group_ids)
        group_consistency_loss = torch.tensor(0.0, device=device)
        if group_ids is not None and pred_rank is not None:
            group_consistency_loss = self._group_rank_consistency_loss(pred_rank, group_ids)
            loss = loss + self.group_consistency_weight * group_consistency_loss

        metrics = {
            "contrastive_loss": contrastive_loss.item(),
            "role_loss": role_loss_val,
            "group_consistency_loss": group_consistency_loss.item(),
            "total_loss": loss.item(),
            "n_positive_pairs": n_pos,
            "mean_pos_sim": pos_sims_val,
            "mean_neg_sim": neg_sims_val,
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ContrastiveTrainer:
    """Training loop for contrastive lane representation learning.

    Supports leave-one-camera-out evaluation splits.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self._setup()

    def _setup(self):
        sys_cfg = self.config.get("system", {})
        train_cfg = self.config.get("contrastive_training", {})
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

        # Model
        model_cfg = self.config.get("model", {})
        self.use_cross_lane_attention = model_cfg.get("use_cross_lane_attention", False)
        self.model = LaneEncoder(
            polyline_k=model_cfg.get("polyline_k", 16),
            d_model=model_cfg.get("polyline_encoder_dim", 64),
            embed_dim=model_cfg.get("embed_dim", 128),
            proj_dim=model_cfg.get("proj_dim", 64),
            polyline_mode=model_cfg.get("polyline_encoder", "transformer"),
            polyline_layers=model_cfg.get("polyline_encoder_layers", 2),
            polyline_heads=model_cfg.get("polyline_encoder_heads", 4),
            stats_dim=model_cfg.get("stats_dim", 9),
            geometry_dropout=model_cfg.get("geometry_dropout", 0.2),
            dropout=model_cfg.get("dropout", 0.1),
            use_cross_lane_attention=self.use_cross_lane_attention,
            cross_lane_heads=model_cfg.get("cross_lane_heads", 4),
            rel_feat_dim=model_cfg.get("rel_feat_dim", 3),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"LaneEncoder params: {n_params:,}")

        # Loss
        self.criterion = ContrastiveLaneLoss(
            temperature=train_cfg.get("temperature", 0.07),
            role_weight=train_cfg.get("role_weight", 2.0),
            total_epochs=train_cfg.get("epochs", 100),
            group_consistency_weight=train_cfg.get("group_consistency_weight", 0.5),
        )

        # Optimizer
        lr = float(train_cfg.get("learning_rate", 3e-4))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # Scheduler
        self.epochs = train_cfg.get("epochs", 100)
        warmup = train_cfg.get("warmup_epochs", 5)
        self.warmup_epochs = warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.epochs - warmup, 1)
        )

        # Grad clip
        self.grad_clip = train_cfg.get("grad_clip_norm", 1.0)

        # Training state
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.patience = train_cfg.get("early_stopping_patience", 30)
        self.history = defaultdict(list)

        # Paths
        self.save_dir = Path(exp_cfg.get("saving_path", "./results"))
        self.exp_name = exp_cfg.get("experiment_name", "lane_contrastive")
        self.exp_dir = self.save_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(
        self, held_out_cameras: Optional[List[str]] = None
    ) -> Tuple[LaneDataset, Optional[List[int]]]:
        """Load LaneDataset. Optionally hold out cameras for zero-shot eval.

        Args:
            held_out_cameras: List of camera names to hold out.

        Returns:
            (full_dataset, held_out_indices or None)
        """
        model_cfg = self.config.get("model", {})
        train_cfg = self.config.get("contrastive_training", {})

        dataset = LaneDataset(
            config=self.config,
            polyline_k=model_cfg.get("polyline_k", 16),
            max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
            role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
            augment=train_cfg.get("augment", True),
        )

        # Store max_group_size in config so it's saved in checkpoint
        # and available at zero-shot inference for consistent normalization
        self.config.setdefault("data", {})["max_group_size"] = dataset._max_group_size

        held_out_indices = None
        if held_out_cameras:
            held_out_indices = []
            for cam in held_out_cameras:
                cam_indices = dataset.get_camera_indices(cam)
                if cam_indices:
                    held_out_indices.extend(cam_indices)
                else:
                    logger.warning(f"No samples for held-out camera {cam}")
            if not held_out_indices:
                held_out_indices = None

        return dataset, held_out_indices

    def _make_loaders(
        self,
        dataset: LaneDataset,
        held_out_indices: Optional[List[int]] = None,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create train (and optionally eval) dataloaders."""
        train_cfg = self.config.get("contrastive_training", {})
        batch_size = train_cfg.get("batch_size", 32)
        num_workers = self.config.get("system", {}).get("num_workers", 0)

        if held_out_indices:
            held_out_set = set(held_out_indices)
            train_indices = [i for i in range(len(dataset)) if i not in held_out_set]
        else:
            train_indices = list(range(len(dataset)))

        if self.use_cross_lane_attention:
            # GroupBatchSampler ensures complete groups per batch.
            # Use max(batch_size, len(train_indices)) so all lanes fit in one
            # batch when the dataset is small (~130 lanes), preserving cross-
            # camera positive pairs. For larger datasets, batch_size applies.
            group_batch_size = max(batch_size, len(train_indices))
            train_sampler = GroupBatchSampler(
                dataset, group_batch_size, allowed_indices=set(train_indices),
            )
            train_loader = DataLoader(
                dataset,
                batch_sampler=train_sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        else:
            train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                drop_last=True,
            )

        eval_loader = None
        if held_out_indices:
            if self.use_cross_lane_attention:
                eval_batch_size = max(batch_size, len(held_out_indices))
                eval_sampler = GroupBatchSampler(
                    dataset, eval_batch_size, allowed_indices=set(held_out_indices),
                )
                eval_loader = DataLoader(
                    dataset,
                    batch_sampler=eval_sampler,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                )
            else:
                eval_loader = DataLoader(
                    Subset(dataset, held_out_indices),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                )

        return train_loader, eval_loader

    def _train_epoch(
        self, epoch: int, loader: DataLoader, positive_pairs: List[Tuple[int, int]]
    ) -> dict:
        """Run one training epoch."""
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
            # Move to device
            geometry = batch["geometry"].to(self.device)
            traj_polylines = batch["traj_polylines"].to(self.device)
            traj_mask = batch["traj_mask"].to(self.device)
            traj_stats = batch["traj_stats"].to(self.device)
            roles = batch["roles"].to(self.device)
            indices = batch["idx"]

            # Concatenate traj_stats + roles as encoder input
            stats_input = torch.cat([traj_stats, roles], dim=-1)

            # Forward
            if self.use_cross_lane_attention:
                group_ids = batch["group_ids"].to(self.device)
                output = self.model.forward_grouped(
                    geometry=geometry,
                    traj_polylines=traj_polylines,
                    traj_mask=traj_mask,
                    traj_stats=stats_input,
                    group_ids=group_ids,
                )
            else:
                group_ids = None
                output = self.model(
                    geometry=geometry,
                    traj_polylines=traj_polylines,
                    traj_mask=traj_mask,
                    traj_stats=stats_input,
                )

            # Loss (InfoNCE + role regression with contrastive warmup)
            loss, metrics = self.criterion(
                output["projection"], indices, positive_pairs,
                roles=roles,
                pred_rank=output["pred_rank"],
                pred_edge=output["pred_edge"],
                pred_size=output["pred_size"],
                epoch=epoch,
                group_ids=group_ids,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for k, v in metrics.items():
                epoch_losses[k] += v
            n_batches += 1
            pbar.set_postfix(loss=f"{metrics.get('contrastive_loss', 0):.4f}")

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        return dict(epoch_losses)

    @torch.no_grad()
    def _eval_epoch(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        dataset: LaneDataset,
    ) -> dict:
        """Evaluate: encode all training lanes as reference, match held-out lanes."""
        self.model.eval()

        # Encode training lanes
        train_embeddings = []
        train_roles = []
        for batch in train_loader:
            stats_input = torch.cat([
                batch["traj_stats"].to(self.device),
                batch["roles"].to(self.device),
            ], dim=-1)
            if self.use_cross_lane_attention:
                output = self.model.forward_grouped(
                    geometry=batch["geometry"].to(self.device),
                    traj_polylines=batch["traj_polylines"].to(self.device),
                    traj_mask=batch["traj_mask"].to(self.device),
                    traj_stats=stats_input,
                    group_ids=batch["group_ids"].to(self.device),
                )
            else:
                output = self.model(
                    geometry=batch["geometry"].to(self.device),
                    traj_polylines=batch["traj_polylines"].to(self.device),
                    traj_mask=batch["traj_mask"].to(self.device),
                    traj_stats=stats_input,
                )
            train_embeddings.append(output["projection"])
            train_roles.append(batch["roles"])

        train_emb = torch.cat(train_embeddings, dim=0).cpu()    # (N_train, proj_dim)
        train_role = torch.cat(train_roles, dim=0)              # (N_train, 5)

        # Encode held-out lanes (trajectory-only: drop geometry)
        eval_embeddings = []
        eval_roles = []
        for batch in eval_loader:
            stats_input = torch.cat([
                batch["traj_stats"].to(self.device),
                batch["roles"].to(self.device),
            ], dim=-1)
            if self.use_cross_lane_attention:
                output = self.model.forward_grouped(
                    geometry=batch["geometry"].to(self.device),
                    traj_polylines=batch["traj_polylines"].to(self.device),
                    traj_mask=batch["traj_mask"].to(self.device),
                    traj_stats=stats_input,
                    group_ids=batch["group_ids"].to(self.device),
                    drop_geometry=True,
                )
            else:
                output = self.model(
                    geometry=batch["geometry"].to(self.device),
                    traj_polylines=batch["traj_polylines"].to(self.device),
                    traj_mask=batch["traj_mask"].to(self.device),
                    traj_stats=stats_input,
                    drop_geometry=True,  # zero-shot: no annotation geometry
                )
            eval_embeddings.append(output["projection"])
            eval_roles.append(batch["roles"])

        eval_emb = torch.cat(eval_embeddings, dim=0).cpu()  # (N_eval, proj_dim)
        eval_role = torch.cat(eval_roles, dim=0)              # (N_eval, 5)

        # Match: for each eval lane, find nearest training lane by cosine sim
        sim_matrix = torch.mm(eval_emb, train_emb.t())  # (N_eval, N_train)
        best_match_idx = sim_matrix.argmax(dim=1)        # (N_eval,)
        best_match_sim = sim_matrix.gather(1, best_match_idx.unsqueeze(1)).squeeze(1)

        # Evaluate: compare matched role similarity
        matched_train_roles = train_role[best_match_idx]  # (N_eval, 5)
        # Role similarity: lateral_rank difference
        lat_rank_diff = (eval_role[:, 0] - matched_train_roles[:, 0]).abs()
        edge_match = (
            (eval_role[:, 1] == matched_train_roles[:, 1]).float()  # leftmost
            + (eval_role[:, 2] == matched_train_roles[:, 2]).float()  # rightmost
        ) / 2.0

        return {
            "mean_match_sim": best_match_sim.mean().item(),
            "mean_lat_rank_diff": lat_rank_diff.mean().item(),
            "edge_flag_accuracy": edge_match.mean().item(),
        }

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        ckpt_dir = self.exp_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch}.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")
            logger.info(f"  Saved best checkpoint at epoch {epoch}")

    def run(self, held_out_cameras: Optional[List[str]] = None):
        """Full training pipeline.

        Args:
            held_out_cameras: Cameras to hold out for zero-shot evaluation.
                If None, trains on all cameras without evaluation.
        """
        logger.info("Loading lane dataset...")
        dataset, held_out_indices = self._load_data(held_out_cameras)
        train_loader, eval_loader = self._make_loaders(dataset, held_out_indices)

        train_cfg = self.config.get("contrastive_training", {})
        val_freq = train_cfg.get("val_frequency", 5)
        ckpt_freq = train_cfg.get("checkpoint_frequency", 10)
        lr = train_cfg.get("learning_rate", 3e-4)

        held_out_str = ", ".join(held_out_cameras) if held_out_cameras else "none"
        logger.info(
            f"Training: {len(dataset)} total lanes, "
            f"held out: {held_out_str} "
            f"({len(held_out_indices) if held_out_indices else 0} lanes)"
        )

        # MLflow
        mlflow.set_tracking_uri(str(self.save_dir / "mlruns"))
        mlflow.set_experiment(self.exp_name)

        with mlflow.start_run(run_name=f"{self.exp_name}_{time.strftime('%m%d_%H%M')}"):
            flat_params = {}
            for section, params in self.config.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        flat_params[f"{section}.{k}"] = v
            mlflow.log_params(flat_params)

            for epoch in range(self.epochs):
                t0 = time.time()

                # Warmup LR
                if epoch < self.warmup_epochs:
                    warmup_factor = (epoch + 1) / self.warmup_epochs
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr * warmup_factor

                # Train
                train_metrics = self._train_epoch(
                    epoch, train_loader, dataset.positive_pairs
                )
                dt = time.time() - t0

                for k, v in train_metrics.items():
                    self.history[f"train/{k}"].append(v)

                loss_val = train_metrics.get("total_loss", train_metrics.get("contrastive_loss", 0))
                ctr_loss = train_metrics.get("contrastive_loss", 0)
                role_loss = train_metrics.get("role_loss", 0)
                pos_sim = train_metrics.get("mean_pos_sim", 0)
                neg_sim = train_metrics.get("mean_neg_sim", 0)
                n_pos = train_metrics.get("n_positive_pairs", 0)
                lr_current = self.optimizer.param_groups[0]["lr"]

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"loss={loss_val:.4f} ctr={ctr_loss:.4f} role={role_loss:.4f} "
                    f"pos_sim={pos_sim:.3f} neg_sim={neg_sim:.3f} "
                    f"n_pos={n_pos:.0f} lr={lr_current:.6f} | {dt:.1f}s"
                )

                mlflow.log_metrics(
                    {f"train/{k}": v for k, v in train_metrics.items()},
                    step=epoch + 1,
                )

                # Validation
                is_best = False
                if eval_loader and (epoch + 1) % val_freq == 0:
                    eval_metrics = self._eval_epoch(
                        train_loader, eval_loader, dataset
                    )
                    for k, v in eval_metrics.items():
                        self.history[f"eval/{k}"].append(v)

                    lat_diff = eval_metrics["mean_lat_rank_diff"]
                    logger.info(
                        f"  Eval: match_sim={eval_metrics['mean_match_sim']:.3f} "
                        f"lat_rank_diff={lat_diff:.3f} "
                        f"edge_acc={eval_metrics['edge_flag_accuracy']:.3f}"
                    )
                    mlflow.log_metrics(
                        {f"eval/{k}": v for k, v in eval_metrics.items()},
                        step=epoch + 1,
                    )

                    is_best = lat_diff < self.best_metric
                    if is_best:
                        self.best_metric = lat_diff
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

                # Checkpoint
                if (epoch + 1) % ckpt_freq == 0:
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

            logger.info(f"Training complete. Results in {self.exp_dir}")
