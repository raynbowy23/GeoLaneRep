"""Training pipeline for v3 lanelet discovery via differentiable graph coarsening."""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import _merge_consecutive_sumo_lanes
from src.models.lane_repr import LaneReprModel
from src.training.losses import compute_total_loss
from src.training.evaluator import compute_lanelet_metrics
from src.utils.visualization import SLOT_COLORS

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Main training loop for v3 lanelet discovery."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self._setup()

    def _setup(self):
        sys_cfg = self.config.get("system", {})
        train_cfg = self.config.get("training", {})
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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Model
        self.model = LaneReprModel(self.config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model params: {n_params:,}")

        # Optimizer
        lr = train_cfg.get("learning_rate", 3e-4)
        wd = train_cfg.get("weight_decay", 5e-4)
        opt_name = train_cfg.get("optimizer", "adamw")
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        epochs = train_cfg.get("epochs", 100)
        warmup = train_cfg.get("warmup_epochs", 10)
        sched_name = train_cfg.get("scheduler", "cosine")
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs - warmup)
        else:
            self.scheduler = None

        # Grad clip
        self.grad_clip = train_cfg.get("grad_clip_norm", 1.0)

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision (AMP) enabled")

        # Training state
        self.epochs = epochs
        self.warmup_epochs = warmup
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.patience = train_cfg.get("early_stopping_patience", 30)
        self.history = defaultdict(list)

        # Paths
        self.save_dir = Path(exp_cfg.get("saving_path", "./results"))
        self.exp_name = exp_cfg.get("experiment_name", "lanelet_discovery")
        self.exp_dir = self.save_dir / self.exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Camera frames and SUMO metadata for visualization
        self.camera_frames = {}
        self.sumo_metadata = {}

    def _load_data(self):
        """Load preprocessed trajectory data and build TrackletDatasets."""
        from src.data.dataset import TrackletDataset

        data_cfg = self.config.get("data", {})
        v1_dir = Path(data_cfg.get("v1_dir", "../graph_geolane"))
        preprocess_dir = v1_dir / "results" / "preprocess"
        use_sumo = data_cfg.get("use_sumo_targets", False)
        camera_list_path = Path(data_cfg.get(
            "camera_locations",
            str(v1_dir / "dataset" / "camera_location_list.txt"),
        ))

        cameras = []
        if camera_list_path.exists():
            cameras = [l.strip() for l in camera_list_path.read_text().splitlines() if l.strip()]
        if not cameras:
            cameras = [d.name for d in preprocess_dir.iterdir() if d.is_dir()]

        import polars as pl
        processed = {}
        for cam in cameras:
            cam_dir = preprocess_dir / cam
            traj_path = cam_dir / "trajectory.csv"
            frame_path = cam_dir / "last_frame.npy"
            if not traj_path.exists():
                logger.warning(f"Skipping {cam}: no trajectory.csv")
                continue
            traj = pl.read_csv(str(traj_path))
            frame = np.load(str(frame_path)) if frame_path.exists() else np.zeros((720, 1280, 3), dtype=np.uint8)
            self.camera_frames[cam] = frame

            class _PData:
                pass
            pdata = _PData()
            pdata.camera_loc = cam
            pdata.trajectories = traj
            pdata.frame = frame
            pdata.graph_data = {}
            pdata.contours = []
            pdata.metadata = {}
            if data_cfg.get("use_sumo_targets", False):
                extracted = self._extract_sumo_metadata(cam, traj, v1_dir)
                if extracted:
                    pdata.metadata = extracted
                    self.sumo_metadata[cam] = extracted
            processed[cam] = pdata

        if not processed:
            raise RuntimeError(f"No preprocessed data found in {preprocess_dir}.")

        logger.info(f"Loaded {len(processed)} cameras from {preprocess_dir}")

        train_cfg = self.config.get("training", {})
        batch_size = train_cfg.get("batch_size", 1)
        num_workers = self.config.get("system", {}).get("num_workers", 0)

        self.train_dataset = TrackletDataset(processed, self.config, split="train")
        self.val_dataset = TrackletDataset(processed, self.config, split="val")
        self.test_dataset = TrackletDataset(processed, self.config, split="test")

        mp_ctx = "spawn" if num_workers > 0 else None
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=TrackletDataset.collate_fn, num_workers=num_workers,
            multiprocessing_context=mp_ctx, persistent_workers=num_workers > 0,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=TrackletDataset.collate_fn, num_workers=num_workers,
            multiprocessing_context=mp_ctx, persistent_workers=num_workers > 0,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=TrackletDataset.collate_fn, num_workers=num_workers,
            multiprocessing_context=mp_ctx, persistent_workers=num_workers > 0,
        )
        logger.info(f"Data: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")

    def _extract_sumo_metadata(self, camera_loc: str, traj_df, v1_dir) -> dict:
        """Extract SUMO lane geometries and convert GPS -> pixel-space."""
        import sys
        v1_path = str(Path(v1_dir).resolve())
        if v1_path not in sys.path:
            sys.path.insert(0, v1_path)

        try:
            from core.osm_extraction.connect_to_osm import OSMConnection
        except ImportError:
            logger.warning("Cannot import OSMConnection from v1 — SUMO targets unavailable")
            return {}

        data_cfg = self.config.get("data", {})

        class _V1Cfg:
            pass

        cfg = _V1Cfg()
        cfg.data = _V1Cfg()
        cfg.data.dataset_path = data_cfg.get("dataset_path", str(Path(v1_dir) / "dataset"))
        cfg.data.osm_path = data_cfg.get("osm_path", str(Path(v1_dir) / "dataset" / "sumo"))
        cfg.data.include_junction_movements = False
        cfg.data.junction_min_traj_points = 30
        cfg.experiment = _V1Cfg()
        cfg.experiment.is_save = False
        cfg.experiment.saving_path = str(self.exp_dir)

        try:
            osm = OSMConnection(cfg)
            result = osm.get_lane_groups_from_sumo(camera_loc, traj_df)
            lane_group_dict, pixel_hom, gps_lane_geom, lane_shape, highway_mask, lane_headings = result
        except Exception as e:
            logger.warning(f"SUMO extraction failed for {camera_loc}: {e}")
            return {}

        if not gps_lane_geom or pixel_hom is None:
            return {}

        # --- Convention fix: v1 uses [lat, lon], v3 standardizes on [lon, lat] ---
        # pixel_hom from v1 maps pixel → [lat, lon].
        # Swap rows 0,1 so it maps pixel → [lon, lat].
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        pixel_hom = swap @ pixel_hom

        # gps_lane_geom from v1 is [lat, lon] — swap to [lon, lat].
        for lid in list(gps_lane_geom.keys()):
            pts = np.array(gps_lane_geom[lid], dtype=np.float64)
            if pts.ndim == 2 and pts.shape[1] >= 2:
                gps_lane_geom[lid] = np.column_stack([pts[:, 1], pts[:, 0]])
                if pts.shape[1] > 2:
                    gps_lane_geom[lid] = np.column_stack(
                        [gps_lane_geom[lid], pts[:, 2:]]
                    )

        if data_cfg.get("merge_consecutive_sumo_lanes", True):
            sumo_net_path = str(Path(data_cfg["osm_path"], camera_loc, "osm.net.xml"))
            gps_lane_geom, lane_headings = _merge_consecutive_sumo_lanes(
                gps_lane_geom, lane_headings, sumo_net_path
            )

        hom_inv = np.linalg.pinv(pixel_hom)
        pixel_lane_geom = {}
        for lane_id, gps_pts in gps_lane_geom.items():
            gps_arr = np.array(gps_pts, dtype=np.float64)
            if gps_arr.ndim != 2 or gps_arr.shape[1] < 2:
                continue
            pts_h = np.hstack([gps_arr[:, :2], np.ones((len(gps_arr), 1), dtype=np.float64)])
            transformed = hom_inv @ pts_h.T
            transformed /= transformed[2]
            pixel_lane_geom[lane_id] = transformed[:2].T

        pixel_lane_headings = {}
        for lane_id, pts in pixel_lane_geom.items():
            if len(pts) >= 2:
                dx = pts[-1, 0] - pts[0, 0]
                dy = pts[-1, 1] - pts[0, 1]
                pixel_lane_headings[lane_id] = np.arctan2(dy, dx)

        return {
            "pixel_lane_geom": pixel_lane_geom,
            "gps_lane_geom": gps_lane_geom,
            "pixel_hom": pixel_hom,
            "lane_headings": pixel_lane_headings,
            "lane_shape": lane_shape,
        }

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        accum_steps = self.config.get("training", {}).get("gradient_accumulation_steps", 1)
        epoch_losses = defaultdict(float)
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.epochs}",
            leave=False,
            bar_format="{l_bar}{bar:30}{r_bar}",
        )
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(batch)
                gt_labels = batch.gt_labels if hasattr(batch, "gt_labels") else None
                loss, loss_dict = compute_total_loss(output, batch, self.config, gt_labels, epoch=epoch)
                loss = loss / accum_steps

            # Skip NaN batches to prevent poisoning the entire run
            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k, v in loss_dict.items():
                if not (isinstance(v, float) and (v != v)):  # skip NaN
                    epoch_losses[k] += v
            n_batches += 1

            pbar.set_postfix(loss=f"{loss_dict.get('total', 0):.3f}")

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        return dict(epoch_losses)

    @torch.no_grad()
    def _validate(self, loader, epoch: int = 0) -> dict:
        self.model.eval()
        train_cfg = self.config.get("training", {})
        conf_thresh = train_cfg.get("confidence_threshold", 0.3)

        all_losses = defaultdict(float)
        all_metrics = defaultdict(float)
        metrics_n = defaultdict(int)
        n = 0

        for batch in loader:
            batch = batch.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(batch)
                gt_labels = batch.gt_labels if hasattr(batch, "gt_labels") else None
                _, loss_dict = compute_total_loss(output, batch, self.config, gt_labels, epoch=epoch)

            lanelet_metrics = compute_lanelet_metrics(
                output, batch, gt_labels,
                confidence_threshold=conf_thresh,
            )

            for k, v in loss_dict.items():
                all_losses[k] += v
            for k, v in lanelet_metrics.items():
                if not (isinstance(v, float) and v != v):  # skip NaN
                    all_metrics[k] += v
                    metrics_n[k] += 1
            n += 1

        result = {}
        for k in all_losses:
            result[f"loss/{k}"] = all_losses[k] / max(n, 1)
        for k in all_metrics:
            result[f"metric/{k}"] = all_metrics[k] / max(metrics_n[k], 1)

        return result

    @torch.no_grad()
    def _visualize_epoch(self, epoch: int, dataset=None, tag: str = None):
        """Run model on one representative time window per lane group per camera."""
        self.model.eval()
        if tag is None:
            tag = f"epoch_{epoch}"
        vis_dir = self.exp_dir / "visualizations" / tag
        vis_dir.mkdir(parents=True, exist_ok=True)

        if dataset is None:
            dataset = self.val_dataset

        train_cfg = self.config.get("training", {})
        conf_thresh = train_cfg.get("confidence_threshold", 0.3)
        vis_top_k = train_cfg.get("vis_top_k", 0)

        from src.utils.visualization import visualize_lanelet_graph

        # Group samples by (camera, lane_group_id)
        camera_lg_samples = defaultdict(list)
        for sample in dataset:
            cam = sample.camera_loc
            lg = getattr(sample, "lane_group_id", 0)
            camera_lg_samples[(cam, lg)].append(sample)

        # Pick one representative per lane group: the window with most tracklets
        camera_representatives = defaultdict(list)
        for (cam, lg), samples in camera_lg_samples.items():
            best = max(samples, key=lambda s: s.x.shape[0])
            camera_representatives[cam].append(best)

        n_saved = 0
        for cam in sorted(camera_representatives.keys()):
            frame = self.camera_frames.get(cam)
            if frame is None:
                continue

            vis = frame.copy()
            color_offset = 0
            for sample in camera_representatives[cam]:
                data = sample.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    output = self.model(data)

                vis, n_chains = visualize_lanelet_graph(
                    vis, output, data,
                    conf_threshold=conf_thresh,
                    color_offset=color_offset,
                    top_k=vis_top_k,
                )
                color_offset += n_chains

            save_path = str(vis_dir / f"{cam}.png")
            cv2.imwrite(save_path, vis)
            try:
                mlflow.log_artifact(save_path, artifact_path=f"visualizations/{tag}")
            except Exception:
                pass

            n_saved += 1
            if n_saved >= 10:
                break

        logger.info(f"  Saved {n_saved} lanelet visualizations to {vis_dir}")

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

    def run(self):
        """Full training pipeline."""
        logger.info("Loading data...")
        self._load_data()

        train_cfg = self.config.get("training", {})
        val_freq = train_cfg.get("val_frequency", 5)
        test_freq = train_cfg.get("test_frequency", 50)
        ckpt_freq = train_cfg.get("checkpoint_frequency", 10)
        vis_freq = train_cfg.get("vis_frequency", 10)

        # MLflow setup
        mlflow.set_tracking_uri(str(self.save_dir / "mlruns"))
        mlflow.set_experiment(self.exp_name)
        with mlflow.start_run(run_name=f"{self.exp_name}_{time.strftime('%m%d_%H%M')}"):
            flat_params = {}
            for section, params in self.config.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        flat_params[f"{section}.{k}"] = v
                else:
                    flat_params[section] = params
            mlflow.log_params(flat_params)
            mlflow.log_artifact(self.config_path)

            logger.info(f"Training for {self.epochs} epochs...")
            for epoch in range(self.epochs):
                t0 = time.time()

                # Warmup LR
                if epoch < self.warmup_epochs:
                    warmup_factor = (epoch + 1) / self.warmup_epochs
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = train_cfg.get("learning_rate", 3e-4) * warmup_factor

                train_losses = self._train_epoch(epoch)
                dt = time.time() - t0

                for k, v in train_losses.items():
                    self.history[f"train/{k}"].append(v)

                total = train_losses.get("total", 0)
                lr_current = self.optimizer.param_groups[0]["lr"]
                pos_loss = train_losses.get("lanelet_position", 0)
                head_loss = train_losses.get("lanelet_heading", 0)
                edge_loss = train_losses.get("lanelet_edge", 0)
                conf_loss = train_losses.get("lanelet_confidence", 0)
                n_matched = train_losses.get("lanelet_n_matched", 0)
                n_gt = train_losses.get("lanelet_n_gt", 0)
                match_dist = train_losses.get("match_mean_dist", 0)
                median_gt_dist = train_losses.get("match_median_min_gt_dist", 0)
                pred_range = train_losses.get("pred_range", 0)
                gt_range = train_losses.get("gt_range", 0)

                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} | "
                    f"loss={total:.4f} pos={pos_loss:.4f} head={head_loss:.4f}"
                    f" edge={edge_loss:.4f} conf={conf_loss:.4f}"
                    f" matched={n_matched:.0f}/{n_gt:.0f}"
                    f" match_dist={match_dist:.1f}m gt_nearest={median_gt_dist:.1f}m"
                    f" pred_rng={pred_range:.0f} gt_rng={gt_range:.0f}"
                    f" lr={lr_current:.6f} | {dt:.1f}s"
                )

                mlflow.log_metrics({
                    f"train/{k}": v for k, v in train_losses.items()
                }, step=epoch + 1)
                mlflow.log_metrics({
                    "lr": lr_current,
                    "epoch_time_s": dt,
                }, step=epoch + 1)

                # Validation
                is_best = False
                if (epoch + 1) % val_freq == 0:
                    val_metrics = self._validate(self.val_loader, epoch=epoch)
                    for k, v in val_metrics.items():
                        self.history[f"val/{k}"].append(v)

                    val_loss = val_metrics.get("loss/total", float("inf"))
                    npe = val_metrics.get("metric/node_position_error", float("nan"))
                    hae = val_metrics.get("metric/heading_angular_error", float("nan"))
                    ari = val_metrics.get("metric/assignment_ari", float("nan"))
                    topo_f1 = val_metrics.get("metric/lane_topology_f1", float("nan"))

                    logger.info(
                        f"  Val: loss={val_loss:.4f} NPE={npe:.2f}m HAE={hae:.1f}deg"
                        f" ARI={ari:.3f} TopologyF1={topo_f1:.3f}"
                    )

                    mlflow.log_metrics({
                        f"val/{k}": v for k, v in val_metrics.items()
                        if not (isinstance(v, float) and v != v)
                    }, step=epoch + 1)

                    is_best = val_loss < self.best_metric
                    if is_best:
                        self.best_metric = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

                # Periodic test evaluation
                if (epoch + 1) % test_freq == 0 and len(self.test_dataset) > 0:
                    test_metrics = self._validate(self.test_loader, epoch=epoch)
                    test_loss = test_metrics.get("loss/total", float("inf"))
                    test_npe = test_metrics.get("metric/node_position_error", float("nan"))
                    logger.info(f"  Test: loss={test_loss:.4f} NPE={test_npe:.2f}m")
                    mlflow.log_metrics({
                        f"test/{k}": v for k, v in test_metrics.items()
                        if not (isinstance(v, float) and v != v)
                    }, step=epoch + 1)

                # Visualization (val + train cameras)
                if (epoch + 1) % vis_freq == 0:
                    self._visualize_epoch(epoch + 1)
                    self._visualize_epoch(epoch + 1, dataset=self.train_dataset, tag=f"epoch_{epoch + 1}_train")

                # Checkpoint
                if (epoch + 1) % ckpt_freq == 0:
                    self._save_checkpoint(epoch + 1, is_best=is_best)

                # LR schedule (after warmup)
                if self.scheduler and epoch >= self.warmup_epochs:
                    self.scheduler.step()

            # Final
            self._save_checkpoint(self.epochs, is_best=True)
            self._visualize_epoch(self.epochs)

            if len(self.test_dataset) > 0:
                test_metrics = self._validate(self.test_loader, epoch=self.epochs)
                logger.info("Test metrics:")
                for k, v in sorted(test_metrics.items()):
                    logger.info(f"  {k}: {v:.4f}")
                # Visualize holdout/test cameras
                self._visualize_epoch(
                    self.epochs, dataset=self.test_dataset, tag=f"epoch_{self.epochs}_test"
                )
                mlflow.log_metrics({
                    f"test/{k}": v for k, v in test_metrics.items()
                    if not (isinstance(v, float) and v != v)
                })

            history_path = self.exp_dir / "history.json"
            with open(history_path, "w") as f:
                json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, f, indent=2)
            mlflow.log_artifact(str(history_path))

            logger.info(f"Training complete. Results in {self.exp_dir}")
