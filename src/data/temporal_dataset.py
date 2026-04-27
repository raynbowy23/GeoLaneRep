import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.annotation_loader import (
    get_annotation_relationships,
    get_group_lanes,
    load_annotation_json,
)
from src.data.lane_dataset import (
    LaneRole,
    _compute_lane_roles,
    _compute_traj_stats,
    _add_group_relative_features,
    point_to_polyline_dist,
    LaneDataset,
)

logger = logging.getLogger(__name__)

@dataclass
class TemporalLaneSample:
    """One lane with time-windowed trajectory data."""

    camera: str
    group_id: int
    cls_id: int
    lane_key: str

    geometry: np.ndarray                        # (K, 2) static lane waypoints
    window_trajectories: List[List[np.ndarray]] # W windows, each list of (T_i, 2)
    window_traj_stats: List[np.ndarray]         # W x (4,) stats per window
    role: LaneRole
    window_valid: List[bool]                    # W bools (enough trajs in window?)

# ---------------------------------------------------------------------------
# Polyline resampling (copied from lane_dataset to avoid circular import issues)
# ---------------------------------------------------------------------------

def _resample_polyline(pts: np.ndarray, k: int) -> np.ndarray:
    """Resample a polyline to exactly k evenly-spaced points."""
    if len(pts) < 2:
        return np.zeros((k, 2), dtype=np.float64)

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]

    if total < 1e-10:
        return np.tile(pts[0], (k, 1))

    target_dists = np.linspace(0, total, k)
    resampled = np.zeros((k, 2), dtype=np.float64)
    for i, d in enumerate(target_dists):
        seg_idx = np.searchsorted(cum_len, d, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(pts) - 2)
        seg_start = cum_len[seg_idx]
        seg_end = cum_len[seg_idx + 1]
        seg_len = seg_end - seg_start
        t = (d - seg_start) / seg_len if seg_len > 1e-10 else 0.0
        resampled[i] = pts[seg_idx] * (1 - t) + pts[seg_idx + 1] * t

    return resampled

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TemporalLaneDataset(Dataset):
    """Dataset of time-windowed lane trajectories for temporal encoding.

    Splits trajectories into overlapping time windows and computes per-window
    statistics. Geometry is static across windows.

    Args:
        config: Full config dict (same as LaneDataset).
        cameras: Optional list of camera names.
        polyline_k: Number of resampled waypoints per polyline.
        max_traj_per_window: Max trajectories to keep per window.
    """

    def __init__(
        self,
        config: dict,
        cameras: Optional[List[str]] = None,
        polyline_k: int = 16,
        max_traj_per_window: int = 50,
        role_similarity_threshold: float = 0.8,
    ):
        self.config = config
        self.polyline_k = polyline_k
        self.max_traj_per_window = max_traj_per_window
        self.role_sim_thresh = role_similarity_threshold
        self._rng = np.random.default_rng(42)

        model_cfg = config.get("model", {})
        self.use_group_relative = model_cfg.get("stats_dim", 9) > 9

        data_cfg = config.get("data", {})
        annot_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
        image_w = data_cfg.get("image_width", 1920)
        image_h = data_cfg.get("image_height", 1080)
        self.image_wh = (image_w, image_h)

        assign_cfg = config.get("assignment", {})
        self.lateral_threshold_px = assign_cfg.get("lateral_threshold_px", 60.0)
        self.min_tracklet_points = assign_cfg.get("min_tracklet_points", 5)

        temporal_cfg = config.get("temporal", {})
        self.window_size_sec = temporal_cfg.get("window_size_sec", 10.0)
        self.window_stride_sec = temporal_cfg.get("window_stride_sec", 5.0)
        self.total_time_sec = temporal_cfg.get("total_time_sec", 60.0)
        self.min_traj_per_window = temporal_cfg.get("min_traj_per_window", 2)

        # Compute number of windows
        self.n_windows = int(
            (self.total_time_sec - self.window_size_sec) / self.window_stride_sec
        ) + 1

        # Discover cameras
        if cameras is None:
            camera_list_path = Path(data_cfg.get(
                "camera_locations", "./dataset/camera_location_list.txt"
            ))
            if camera_list_path.exists():
                cameras = [
                    l.strip() for l in camera_list_path.read_text().splitlines()
                    if l.strip()
                ]
            else:
                cameras = sorted(
                    d.name for d in annot_dir.iterdir()
                    if d.is_dir() and (d / "annotation.json").exists()
                )

        # First pass: count max group size for normalization
        self._max_group_size = 1
        for cam in cameras:
            annot_path = annot_dir / cam / "annotation.json"
            if not annot_path.exists():
                continue
            annotation = load_annotation_json(annot_path)
            for lg in annotation["lane_groups"]:
                n_lanes = len([l for l in lg["lanes"] if len(l.get("waypoints", [])) >= 2])
                self._max_group_size = max(self._max_group_size, n_lanes)

        # Build samples
        self.samples: List[TemporalLaneSample] = []
        self._camera_indices: Dict[str, List[int]] = {}

        for cam in cameras:
            annot_path = annot_dir / cam / "annotation.json"
            traj_path = annot_dir / cam / "trajectory.csv"
            if not annot_path.exists() or not traj_path.exists():
                logger.warning(f"Skipping {cam}: missing annotation or trajectory")
                continue

            annotation = load_annotation_json(annot_path)
            traj_df = pl.read_csv(str(traj_path))

            cam_indices = []
            for lg in annotation["lane_groups"]:
                gid = lg["group_id"]
                roles = _compute_lane_roles(
                    annotation, gid, self.image_wh, self._max_group_size
                )
                lanes = get_group_lanes(annotation, gid, image_wh=self.image_wh)

                # Assign trajectories with time info
                lane_traj_with_time = self._assign_trajectories_with_time(
                    lanes, traj_df
                )

                # Compute aggregate stats per lane for group-relative features
                group_agg_stats = {}
                for lane in lanes:
                    cls_id = lane["cls_id"]
                    if cls_id not in roles:
                        continue
                    all_trajs = [xy for xy, _ in lane_traj_with_time.get(cls_id, [])]
                    max_count = max(
                        (len(lane_traj_with_time.get(l["cls_id"], []))
                         for l in lanes if l["cls_id"] in roles), default=1
                    )
                    group_agg_stats[cls_id] = _compute_traj_stats(
                        all_trajs, lane["waypoints"], max(max_count, 1)
                    )

                _add_group_relative_features(roles, group_agg_stats)

                for lane in lanes:
                    cls_id = lane["cls_id"]
                    if cls_id not in roles:
                        continue

                    lane_key = f"{cam}_{gid}_{cls_id}"
                    traj_time_pairs = lane_traj_with_time.get(cls_id, [])

                    # Bin into windows
                    window_trajs, window_stats, window_valid = self._bin_into_windows(
                        traj_time_pairs, lane["waypoints"]
                    )

                    sample = TemporalLaneSample(
                        camera=cam,
                        group_id=gid,
                        cls_id=cls_id,
                        lane_key=lane_key,
                        geometry=lane["waypoints"],
                        window_trajectories=window_trajs,
                        window_traj_stats=window_stats,
                        role=roles[cls_id],
                        window_valid=window_valid,
                    )
                    cam_indices.append(len(self.samples))
                    self.samples.append(sample)

            self._camera_indices[cam] = cam_indices

        # Pre-compute positive pairs for contrastive learning
        self._positive_pairs = self._mine_positive_pairs()

        logger.info(
            f"TemporalLaneDataset: {len(self.samples)} lanes, "
            f"{self.n_windows} windows per lane, "
            f"{len(self._positive_pairs)} positive pairs"
        )

    def _assign_trajectories_with_time(
        self,
        lanes: list,
        traj_df: pl.DataFrame,
    ) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
        """Assign trajectories to lanes, preserving time information.

        Returns:
            Dict mapping cls_id -> list of (xy_array, time_array) tuples.
        """
        result: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
            l["cls_id"]: [] for l in lanes
        }

        if traj_df.is_empty() or not lanes:
            return result

        wh = np.array(self.image_wh, dtype=np.float64)

        # Determine time column
        time_col = None
        for col_name in ["time", "timestamp", "frame", "t"]:
            if col_name in traj_df.columns:
                time_col = col_name
                break

        id_col = "id" if "id" in traj_df.columns else "track_id"
        tracks = traj_df.group_by(id_col)

        for track_id, group in tracks:
            pts = np.column_stack([
                group["x"].to_numpy(),
                group["y"].to_numpy(),
            ]).astype(np.float64) / wh

            if len(pts) < self.min_tracklet_points:
                continue

            # Extract time values
            if time_col is not None:
                times = group[time_col].to_numpy().astype(np.float64)
            else:
                # Fallback: use row index as proxy for time
                times = np.arange(len(pts), dtype=np.float64)

            # Find nearest lane
            best_cls = None
            best_dist = float("inf")
            thresh = self.lateral_threshold_px / wh[0]

            for lane in lanes:
                wp = lane["waypoints"]
                if len(wp) < 2:
                    continue
                dists = point_to_polyline_dist(pts, wp)
                mean_dist = dists.mean()
                if mean_dist < best_dist:
                    best_dist = mean_dist
                    best_cls = lane["cls_id"]

            if best_cls is not None and best_dist < thresh:
                result[best_cls].append((pts, times))

        return result

    def _bin_into_windows(
        self,
        traj_time_pairs: List[Tuple[np.ndarray, np.ndarray]],
        lane_waypoints: np.ndarray,
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[bool]]:
        """Split trajectory-time pairs into time windows.

        Args:
            traj_time_pairs: List of (xy, time) pairs for this lane.
            lane_waypoints: Lane geometry for stats computation.

        Returns:
            (window_trajectories, window_stats, window_valid)
        """
        W = self.n_windows
        window_trajs: List[List[np.ndarray]] = [[] for _ in range(W)]
        window_stats: List[np.ndarray] = []
        window_valid: List[bool] = []

        # Offset times to start at 0 (real seconds, no scaling)
        if traj_time_pairs:
            all_times = np.concatenate([t for _, t in traj_time_pairs])
            t_min = all_times.min()
        else:
            t_min = 0.0

        # Assign each trajectory to windows based on its mean time
        for xy, times in traj_time_pairs:
            # Real seconds from start (no normalization)
            norm_times = times - t_min

            # Use mean time to determine primary window
            mean_t = norm_times.mean()

            for w in range(W):
                w_start = w * self.window_stride_sec
                w_end = w_start + self.window_size_sec
                if w_start <= mean_t < w_end:
                    window_trajs[w].append(xy)

        # Compute per-window stats and validity
        # Max traj count across windows for normalization
        max_count = max((len(wt) for wt in window_trajs), default=1)
        max_count = max(max_count, 1)

        for w in range(W):
            trajs = window_trajs[w]
            valid = len(trajs) >= self.min_traj_per_window
            window_valid.append(valid)

            stats = _compute_traj_stats(trajs, lane_waypoints, max_count)
            window_stats.append(stats)

        return window_trajs, window_stats, window_valid

    def _mine_positive_pairs(self) -> List[Tuple[int, int]]:
        """Mine positive pairs: same structural role, different camera.

        Same criteria as LaneDataset._mine_positive_pairs().
        """
        pairs = []
        n = len(self.samples)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = self.samples[i], self.samples[j]
                if si.camera == sj.camera:
                    continue
                rank_diff = abs(si.role.lateral_rank - sj.role.lateral_rank)
                if rank_diff > 0.15:
                    continue
                if si.role.is_leftmost != sj.role.is_leftmost:
                    continue
                if si.role.is_rightmost != sj.role.is_rightmost:
                    continue
                sim = LaneDataset._role_similarity(si.role, sj.role)
                if sim >= self.role_sim_thresh:
                    pairs.append((i, j))
        return pairs

    @property
    def positive_pairs(self) -> List[Tuple[int, int]]:
        return self._positive_pairs

    @property
    def cameras(self) -> List[str]:
        return list(self._camera_indices.keys())

    def get_camera_indices(self, camera: str) -> List[int]:
        return self._camera_indices.get(camera, [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        W = self.n_windows
        K = self.polyline_k

        # Resample geometry (static across windows)
        geometry = _resample_polyline(sample.geometry, K)

        # Process each window's trajectories
        all_window_traj_polylines = []  # W lists of (K, 2) tensors
        all_window_n_trajs = []

        for w in range(W):
            trajs = sample.window_trajectories[w]

            # Subsample if too many
            if len(trajs) > self.max_traj_per_window:
                indices = self._rng.choice(
                    len(trajs), self.max_traj_per_window, replace=False
                )
                trajs = [trajs[i] for i in indices]

            # Resample each trajectory to K points
            traj_polylines = []
            for t in trajs:
                resampled = _resample_polyline(t, K)
                traj_polylines.append(torch.tensor(resampled, dtype=torch.float32))

            # Zero placeholder if empty
            if not traj_polylines:
                traj_polylines = [torch.zeros(K, 2)]

            all_window_traj_polylines.append(traj_polylines)
            all_window_n_trajs.append(len(trajs))

        # Stack window stats
        window_traj_stats = torch.tensor(
            np.stack(sample.window_traj_stats), dtype=torch.float32
        )  # (W, 4)

        window_valid = torch.tensor(sample.window_valid, dtype=torch.bool)  # (W,)

        return {
            "idx": idx,
            "camera": sample.camera,
            "group_id": sample.group_id,
            "lane_key": sample.lane_key,
            "geometry": torch.tensor(geometry, dtype=torch.float32),   # (K, 2)
            "window_traj_polylines": all_window_traj_polylines,        # W x [list of (K,2)]
            "window_traj_stats": window_traj_stats,                    # (W, 4)
            "window_valid": window_valid,                              # (W,)
            "role": sample.role.to_tensor(include_group_relative=self.use_group_relative),
            "n_windows": W,
        }

def temporal_collate_fn(batch: List[dict]) -> dict:
    """Collate temporal samples, padding trajectories across batch and windows.

    Returns tensors:
        geometry: (B, K, 2)
        window_traj_polylines: (B, W, T_max, K, 2)
        window_traj_mask: (B, W, T_max) bool
        window_traj_stats: (B, W, 4)
        window_valid: (B, W) bool
        roles: (B, R) where R is role dimension (5 structural + 3 group-relative)
    """
    B = len(batch)
    K = batch[0]["geometry"].shape[0]
    W = batch[0]["n_windows"]

    geometry = torch.stack([b["geometry"] for b in batch])                # (B, K, 2)
    window_traj_stats = torch.stack([b["window_traj_stats"] for b in batch])  # (B, W, 4)
    window_valid = torch.stack([b["window_valid"] for b in batch])        # (B, W)
    roles = torch.stack([b["role"] for b in batch])                       # (B, R)
    indices = torch.tensor([b["idx"] for b in batch], dtype=torch.long)

    # Find max trajectory count across all windows and all samples
    T_max = 1
    for b in batch:
        for w_trajs in b["window_traj_polylines"]:
            T_max = max(T_max, len(w_trajs))

    # Build padded trajectory tensor
    traj_padded = torch.zeros(B, W, T_max, K, 2)
    traj_mask = torch.zeros(B, W, T_max, dtype=torch.bool)

    for bi, b in enumerate(batch):
        for w in range(W):
            w_trajs = b["window_traj_polylines"][w]
            n = len(w_trajs)
            for j in range(n):
                traj_padded[bi, w, j] = w_trajs[j]
            traj_mask[bi, w, :n] = True

    return {
        "idx": indices,
        "cameras": [b["camera"] for b in batch],
        "lane_keys": [b["lane_key"] for b in batch],
        "geometry": geometry,                        # (B, K, 2)
        "window_traj_polylines": traj_padded,        # (B, W, T_max, K, 2)
        "window_traj_mask": traj_mask,               # (B, W, T_max)
        "window_traj_stats": window_traj_stats,      # (B, W, 4)
        "window_valid": window_valid,                # (B, W)
        "roles": roles,                              # (B, R)
    }
