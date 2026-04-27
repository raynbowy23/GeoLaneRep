import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from src.data.annotation_loader import (
    get_annotation_relationships,
    get_group_lanes,
    load_annotation_json,
)

logger = logging.getLogger(__name__)

def point_to_polyline_dist(pts: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Compute min distance from each point to the nearest segment of a polyline.

    Uses point-to-segment projection (not just nearest-waypoint), which gives
    accurate distances even with sparse waypoints.

    Args:
        pts: (N, 2) query points.
        polyline: (M, 2) ordered waypoints defining the polyline.

    Returns:
        (N,) array of distances to the nearest polyline segment.
    """
    if len(polyline) < 2:
        # Degenerate: single point — fall back to point distance
        return np.linalg.norm(pts - polyline[0:1], axis=1)

    # Segment start/end: (M-1, 2)
    seg_a = polyline[:-1]  # (S, 2)
    seg_b = polyline[1:]   # (S, 2)
    seg_d = seg_b - seg_a  # (S, 2)
    seg_len_sq = np.sum(seg_d ** 2, axis=1)  # (S,)

    # For each point, project onto each segment
    # pts: (N, 1, 2), seg_a: (1, S, 2)
    ap = pts[:, None, :] - seg_a[None, :, :]  # (N, S, 2)
    # Parameter t of projection clamped to [0, 1]
    t = np.sum(ap * seg_d[None, :, :], axis=2)  # (N, S)
    t = np.clip(t / np.maximum(seg_len_sq[None, :], 1e-12), 0.0, 1.0)  # (N, S)
    # Nearest point on segment
    proj = seg_a[None, :, :] + t[:, :, None] * seg_d[None, :, :]  # (N, S, 2)
    # Distance to nearest point on each segment
    seg_dists = np.linalg.norm(pts[:, None, :] - proj, axis=2)  # (N, S)
    # Min across segments
    return seg_dists.min(axis=1)  # (N,)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LaneRole:
    """Structural role descriptor derived from annotation topology."""

    lateral_rank: float       # 0.0 (leftmost) to 1.0 (rightmost) within group
    is_leftmost: bool
    is_rightmost: bool
    has_successor: bool       # lane continues into next segment (merge/exit)
    group_size: float         # normalized lane count (n_lanes / max_across_dataset)

    # Group-relative behavioral features (z-scores within group)
    relative_speed: float = 0.0
    relative_density: float = 0.0
    relative_curvature: float = 0.0

    def to_tensor(self, include_group_relative: bool = False) -> torch.Tensor:
        """Role tensor. 5 dims structural, or 8 dims with group-relative."""
        vals = [
            self.lateral_rank,
            float(self.is_leftmost),
            float(self.is_rightmost),
            float(self.has_successor),
            self.group_size,
        ]
        if include_group_relative:
            vals.extend([
                self.relative_speed,
                self.relative_density,
                self.relative_curvature,
            ])
        return torch.tensor(vals, dtype=torch.float32)

    def structural_tensor(self) -> torch.Tensor:
        """Structural role tensor (5 dims) for similarity computation."""
        return self.to_tensor(include_group_relative=False)

@dataclass
class LaneSample:
    """One annotated lane with its geometry, trajectories, and role."""

    camera: str
    group_id: int
    cls_id: int
    lane_key: str              # unique id: "{camera}_{group_id}_{cls_id}"

    geometry: np.ndarray       # (G, 2) normalized waypoints
    trajectories: List[np.ndarray]  # list of (T_i, 2) trajectory arrays
    traj_stats: np.ndarray     # (4,) [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]

    role: LaneRole

# ---------------------------------------------------------------------------
# Lane role computation
# ---------------------------------------------------------------------------

def _compute_lane_roles(
    annotation: dict,
    group_id: int,
    image_wh: Tuple[int, int],
    max_group_size: int,
) -> Dict[int, LaneRole]:
    """Compute structural role for each lane in a group.

    Lateral rank is based on perpendicular offset from the group's mean
    direction vector (cross-product ordering).
    """
    lanes = get_group_lanes(annotation, group_id, image_wh=image_wh)
    if not lanes:
        return {}

    relationships = get_annotation_relationships(annotation, group_id)
    successor_cls_ids = set()
    for rel in relationships:
        if rel.get("type") == "successor":
            successor_cls_ids.add(rel.get("from_cls"))

    # Compute group heading from mean of all lane tangent vectors
    tangents = []
    centroids = []
    for lane in lanes:
        wp = lane["waypoints"]  # (N, 2) normalized
        if len(wp) >= 2:
            t = wp[-1] - wp[0]
            norm = np.linalg.norm(t)
            if norm > 1e-8:
                tangents.append(t / norm)
            centroids.append(wp.mean(axis=0))

    if not tangents:
        return {}

    mean_tangent = np.mean(tangents, axis=0)
    mean_tangent /= np.linalg.norm(mean_tangent) + 1e-8

    # Fold tangent to upper half-plane [0, π) so opposite-direction
    # groups on the same road get consistent lateral rank ordering.
    if mean_tangent[1] < 0 or (mean_tangent[1] == 0 and mean_tangent[0] < 0):
        mean_tangent = -mean_tangent

    # Perpendicular: rotate tangent 90 degrees CCW
    perp = np.array([-mean_tangent[1], mean_tangent[0]])

    # Project each lane centroid onto the perpendicular axis
    lateral_offsets = {}
    for lane, centroid in zip(lanes, centroids):
        lateral_offsets[lane["cls_id"]] = np.dot(centroid, perp)

    # Sort by lateral offset to determine rank
    sorted_cls = sorted(lateral_offsets.keys(), key=lambda c: lateral_offsets[c])
    n = len(sorted_cls)
    group_size_norm = n / max(max_group_size, 1)

    roles = {}
    for rank_idx, cls_id in enumerate(sorted_cls):
        lat_rank = rank_idx / max(n - 1, 1) if n > 1 else 0.5
        roles[cls_id] = LaneRole(
            lateral_rank=lat_rank,
            is_leftmost=(rank_idx == 0),
            is_rightmost=(rank_idx == n - 1),
            has_successor=(cls_id in successor_cls_ids),
            group_size=group_size_norm,
        )

    return roles

def _add_group_relative_features(
    roles: Dict[int, LaneRole],
    group_stats: Dict[int, np.ndarray],
) -> None:
    """Add group-relative behavioral features to roles in-place.

    For each lane, computes z-score of speed, density, and curvature
    relative to its group. Provides cross-lane context without attention.

    Args:
        roles: Dict mapping cls_id -> LaneRole (modified in-place).
        group_stats: Dict mapping cls_id -> (4,) traj stats array.
    """
    common_ids = [c for c in roles if c in group_stats]
    if len(common_ids) < 2:
        return  # single-lane group: relative features stay at 0

    stats_matrix = np.stack([group_stats[c] for c in common_ids])  # (N, 4)
    # stats: [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]
    speeds = stats_matrix[:, 0]
    curvatures = stats_matrix[:, 1]
    densities = stats_matrix[:, 3]

    eps = 1e-6
    speed_mean, speed_std = speeds.mean(), speeds.std()
    curv_mean, curv_std = curvatures.mean(), curvatures.std()
    dens_mean, dens_std = densities.mean(), densities.std()

    for i, cls_id in enumerate(common_ids):
        roles[cls_id].relative_speed = float(
            (speeds[i] - speed_mean) / (speed_std + eps)
        )
        roles[cls_id].relative_curvature = float(
            (curvatures[i] - curv_mean) / (curv_std + eps)
        )
        roles[cls_id].relative_density = float(
            (densities[i] - dens_mean) / (dens_std + eps)
        )

# ---------------------------------------------------------------------------
# Trajectory statistics
# ---------------------------------------------------------------------------

def _compute_traj_stats(
    trajectories: List[np.ndarray],
    lane_waypoints: np.ndarray,
    max_traj_count: int,
) -> np.ndarray:
    """Compute aggregate statistics for trajectories assigned to a lane.

    Returns (4,): [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]
    """
    if not trajectories:
        return np.zeros(4, dtype=np.float32)

    speeds = []
    curvatures = []
    lateral_offsets = []

    # Lane direction for lateral offset
    if len(lane_waypoints) >= 2:
        lane_tangent = lane_waypoints[-1] - lane_waypoints[0]
        lane_tangent_norm = np.linalg.norm(lane_tangent)
        if lane_tangent_norm > 1e-8:
            lane_tangent = lane_tangent / lane_tangent_norm
        lane_perp = np.array([-lane_tangent[1], lane_tangent[0]])
        lane_center = lane_waypoints.mean(axis=0)
    else:
        lane_perp = np.array([0.0, 1.0])
        lane_center = np.zeros(2)

    for traj in trajectories:
        if len(traj) < 2:
            continue
        # Speed: mean displacement between consecutive points
        diffs = np.diff(traj, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        speeds.append(dists.mean())

        # Curvature: mean angle change
        if len(traj) >= 3:
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            curvatures.append(angle_diffs.mean())

        # Lateral offset from lane center
        offsets = np.dot(traj - lane_center, lane_perp)
        lateral_offsets.append(np.abs(offsets).mean())

    mean_speed = np.mean(speeds) if speeds else 0.0
    mean_curv = np.mean(curvatures) if curvatures else 0.0
    mean_lat = np.mean(lateral_offsets) if lateral_offsets else 0.0
    count_norm = len(trajectories) / max(max_traj_count, 1)

    return np.array([mean_speed, mean_curv, mean_lat, count_norm], dtype=np.float32)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LaneDataset(Dataset):
    """Dataset of annotated lanes across all cameras.

    Each item is a LaneSample containing geometry, trajectories, stats, and role.
    The dataset also pre-computes positive pairs for contrastive training.
    """

    def __init__(
        self,
        config: dict,
        cameras: Optional[List[str]] = None,
        polyline_k: int = 16,
        max_traj_per_lane: int = 50,
        role_similarity_threshold: float = 0.8,
        augment: bool = False,
    ):
        self.config = config
        self.polyline_k = polyline_k
        self.max_traj_per_lane = max_traj_per_lane
        self.role_sim_thresh = role_similarity_threshold
        self.augment = augment
        self._rng = np.random.default_rng(42)

        # stats_dim > 4 means group-relative features are enabled (legacy)
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

        # First pass: count max group size across all cameras for normalization
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
        self.samples: List[LaneSample] = []
        self._camera_indices: Dict[str, List[int]] = {}  # camera -> [sample indices]

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

                # Assign trajectories to lanes using simple geometric scoring
                lane_trajectories = self._assign_trajectories(lanes, traj_df)

                # Compute stats for all lanes in group first
                max_traj_count = max(
                    len(v) for v in lane_trajectories.values()
                ) if lane_trajectories else 1

                group_stats = {}
                for lane in lanes:
                    cls_id = lane["cls_id"]
                    if cls_id not in roles:
                        continue
                    trajs = lane_trajectories.get(cls_id, [])
                    group_stats[cls_id] = _compute_traj_stats(
                        trajs, lane["waypoints"], max_traj_count
                    )

                # Add group-relative features to roles
                _add_group_relative_features(roles, group_stats)

                for lane in lanes:
                    cls_id = lane["cls_id"]
                    if cls_id not in roles:
                        continue

                    lane_key = f"{cam}_{gid}_{cls_id}"
                    trajs = lane_trajectories.get(cls_id, [])

                    sample = LaneSample(
                        camera=cam,
                        group_id=gid,
                        cls_id=cls_id,
                        lane_key=lane_key,
                        geometry=lane["waypoints"],
                        trajectories=trajs,
                        traj_stats=group_stats[cls_id],
                        role=roles[cls_id],
                    )
                    cam_indices.append(len(self.samples))
                    self.samples.append(sample)

            self._camera_indices[cam] = cam_indices

        logger.info(
            f"LaneDataset: {len(self.samples)} lanes from "
            f"{len(self._camera_indices)} cameras"
        )

        # Pre-compute positive pairs
        self._positive_pairs = self._mine_positive_pairs()
        logger.info(f"Mined {len(self._positive_pairs)} positive pairs")

    def _assign_trajectories(
        self,
        lanes: list,
        traj_df: pl.DataFrame,
    ) -> Dict[int, List[np.ndarray]]:
        """Simple geometric assignment: each track -> nearest lane by mean distance."""
        result: Dict[int, List[np.ndarray]] = {l["cls_id"]: [] for l in lanes}

        if traj_df.is_empty() or not lanes:
            return result

        # Normalize pixel coordinates
        wh = np.array(self.image_wh, dtype=np.float64)

        # Group by track id
        id_col = "id" if "id" in traj_df.columns else "track_id"
        tracks = traj_df.group_by(id_col)

        for track_id, group in tracks:
            pts = np.column_stack([
                group["x"].to_numpy(),
                group["y"].to_numpy(),
            ]).astype(np.float64) / wh

            if len(pts) < self.min_tracklet_points:
                continue

            # Find nearest lane
            best_cls = None
            best_dist = float("inf")
            thresh = self.lateral_threshold_px / wh[0]  # normalize threshold

            for lane in lanes:
                wp = lane["waypoints"]
                if len(wp) < 2:
                    continue
                # Mean distance from trajectory points to lane polyline
                dists = point_to_polyline_dist(pts, wp)
                mean_dist = dists.mean()
                if mean_dist < best_dist:
                    best_dist = mean_dist
                    best_cls = lane["cls_id"]

            if best_cls is not None and best_dist < thresh:
                result[best_cls].append(pts)

        return result

    def _mine_positive_pairs(self) -> List[Tuple[int, int]]:
        """Mine positive pairs: same structural role, different camera.

        Criteria (all must hold):
        - Different camera
        - Role cosine similarity >= threshold (default 0.8)
        - Lateral rank difference < 0.15 (prevents leftmost-rightmost pairing)
        - Same edge type (both leftmost, both rightmost, or both interior)
        """
        pairs = []
        n = len(self.samples)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = self.samples[i], self.samples[j]
                # Must be from different cameras
                if si.camera == sj.camera:
                    continue
                # Lateral rank must be close
                rank_diff = abs(si.role.lateral_rank - sj.role.lateral_rank)
                if rank_diff > 0.15:
                    continue
                # Edge type must match (both leftmost, both rightmost, or both interior)
                if si.role.is_leftmost != sj.role.is_leftmost:
                    continue
                if si.role.is_rightmost != sj.role.is_rightmost:
                    continue
                # Role cosine similarity
                sim = self._role_similarity(si.role, sj.role)
                if sim >= self.role_sim_thresh:
                    pairs.append((i, j))
        return pairs

    @staticmethod
    def _role_similarity(a: LaneRole, b: LaneRole) -> float:
        """Cosine-like similarity between two lane roles (structural only)."""
        va = a.structural_tensor()
        vb = b.structural_tensor()
        dot = (va * vb).sum()
        norm_a = va.norm() + 1e-8
        norm_b = vb.norm() + 1e-8
        return (dot / (norm_a * norm_b)).item()

    @property
    def cameras(self) -> List[str]:
        return list(self._camera_indices.keys())

    @property
    def positive_pairs(self) -> List[Tuple[int, int]]:
        return self._positive_pairs

    def get_camera_indices(self, camera: str) -> List[int]:
        return self._camera_indices.get(camera, [])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Resample geometry to fixed K points
        geometry = self._resample_polyline(sample.geometry, self.polyline_k)

        # Sample and pad trajectories
        trajs = sample.trajectories
        if len(trajs) > self.max_traj_per_lane:
            indices = self._rng.choice(len(trajs), self.max_traj_per_lane, replace=False)
            trajs = [trajs[i] for i in indices]

        # Apply augmentation during training
        if self.augment and trajs:
            geometry, trajs = self._augment(geometry, trajs)

        # Resample each trajectory to polyline_k points
        traj_polylines = []
        for t in trajs:
            resampled = self._resample_polyline(t, self.polyline_k)
            traj_polylines.append(torch.tensor(resampled, dtype=torch.float32))

        # If no trajectories, use a zero placeholder
        if not traj_polylines:
            traj_polylines = [torch.zeros(self.polyline_k, 2)]

        return {
            "idx": idx,
            "camera": sample.camera,
            "group_id": sample.group_id,
            "lane_key": sample.lane_key,
            "geometry": torch.tensor(geometry, dtype=torch.float32),    # (K, 2)
            "traj_polylines": traj_polylines,                            # list of (K, 2)
            "traj_stats": torch.tensor(sample.traj_stats, dtype=torch.float32),  # (4,)
            "role": sample.role.to_tensor(include_group_relative=self.use_group_relative),
            "n_trajs": len(trajs),
        }

    def _augment(
        self,
        geometry: np.ndarray,
        trajectories: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Apply random augmentations to geometry and trajectories.

        Augmentations (applied center-relative to preserve [0,1] range):
        1. Random rotation (±15°) — simulates different camera angles
        2. Random lateral jitter — simulates annotation offset
        3. Trajectory subsampling (drop 30%) — simulates occlusion
        4. Point-level noise on trajectories — simulates tracker jitter
        """
        rng = self._rng
        center = geometry.mean(axis=0) if len(geometry) > 0 else np.array([0.5, 0.5])

        # 1. Random rotation (±15°)
        angle = rng.uniform(-math.pi / 12, math.pi / 12)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        geometry = (geometry - center) @ R.T + center
        trajectories = [(t - center) @ R.T + center for t in trajectories]

        # 2. Random lateral jitter (small shift perpendicular to lane)
        jitter = rng.normal(0, 0.003, size=2)
        geometry = geometry + jitter
        trajectories = [t + jitter for t in trajectories]

        # 3. Trajectory subsampling (drop 30% of trajectories)
        if len(trajectories) > 2:
            n_keep = max(2, int(len(trajectories) * 0.7))
            keep_idx = rng.choice(len(trajectories), n_keep, replace=False)
            trajectories = [trajectories[i] for i in keep_idx]

        # 4. Point-level noise on trajectory points
        trajectories = [t + rng.normal(0, 0.002, size=t.shape)
                        for t in trajectories]

        return geometry, trajectories

    @staticmethod
    def _resample_polyline(pts: np.ndarray, k: int) -> np.ndarray:
        """Resample a polyline to exactly k evenly-spaced points."""
        if len(pts) < 2:
            return np.zeros((k, 2), dtype=np.float64)

        # Compute cumulative arc length
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum_len[-1]

        if total < 1e-10:
            return np.tile(pts[0], (k, 1))

        # Interpolate at uniform spacing
        target_dists = np.linspace(0, total, k)
        resampled = np.zeros((k, 2), dtype=np.float64)
        for i, d in enumerate(target_dists):
            # Find segment
            seg_idx = np.searchsorted(cum_len, d, side="right") - 1
            seg_idx = np.clip(seg_idx, 0, len(pts) - 2)
            seg_start = cum_len[seg_idx]
            seg_end = cum_len[seg_idx + 1]
            seg_len = seg_end - seg_start
            t = (d - seg_start) / seg_len if seg_len > 1e-10 else 0.0
            resampled[i] = pts[seg_idx] * (1 - t) + pts[seg_idx + 1] * t

        return resampled

def collate_fn(batch: List[dict]) -> dict:
    """Collate variable-length trajectory lists into padded tensors."""
    # Stack fixed-size tensors
    geometry = torch.stack([b["geometry"] for b in batch])           # (B, K, 2)
    traj_stats = torch.stack([b["traj_stats"] for b in batch])      # (B, 4)
    roles = torch.stack([b["role"] for b in batch])                  # (B, R)
    indices = torch.tensor([b["idx"] for b in batch], dtype=torch.long)

    # Remap (camera, group_id) pairs to unique integers for group_ids tensor
    cam_group_pairs = [(b["camera"], b["group_id"]) for b in batch]
    unique_pairs = {}
    group_ids_list = []
    for pair in cam_group_pairs:
        if pair not in unique_pairs:
            unique_pairs[pair] = len(unique_pairs)
        group_ids_list.append(unique_pairs[pair])
    group_ids = torch.tensor(group_ids_list, dtype=torch.long)

    # Pad trajectory polylines: gather all per-lane, pad to max count
    max_n_trajs = max(len(b["traj_polylines"]) for b in batch)
    k = batch[0]["geometry"].shape[0]
    traj_padded = torch.zeros(len(batch), max_n_trajs, k, 2)
    traj_mask = torch.zeros(len(batch), max_n_trajs, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = len(b["traj_polylines"])
        for j in range(n):
            traj_padded[i, j] = b["traj_polylines"][j]
        traj_mask[i, :n] = True

    return {
        "idx": indices,
        "cameras": [b["camera"] for b in batch],
        "lane_keys": [b["lane_key"] for b in batch],
        "geometry": geometry,              # (B, K, 2)
        "traj_polylines": traj_padded,     # (B, max_T, K, 2)
        "traj_mask": traj_mask,            # (B, max_T)
        "traj_stats": traj_stats,          # (B, 4)
        "roles": roles,                    # (B, R)
        "group_ids": group_ids,            # (B,)
    }

class GroupBatchSampler(Sampler):
    """Batch sampler that ensures all lanes from the same (camera, group_id) appear together.

    Greedily packs complete groups into batches until batch_size is reached.
    Groups are shuffled each epoch for randomness.

    Args:
        dataset: LaneDataset instance.
        batch_size: Maximum batch size.
        allowed_indices: If provided, only sample from these indices (e.g. train split).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        allowed_indices: Optional[Set[int]] = None,
    ):
        self.batch_size = batch_size

        # Build groups: (camera, group_id) -> list of sample indices
        groups: Dict[tuple, List[int]] = defaultdict(list)
        for i in range(len(dataset)):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            sample = dataset.samples[i]
            key = (sample.camera, sample.group_id)
            groups[key].append(i)

        self.groups = list(groups.values())
        self._rng = np.random.default_rng(42)

    def __iter__(self):
        # Shuffle group order
        order = self._rng.permutation(len(self.groups))

        batch = []
        for g_idx in order:
            group = self.groups[g_idx]
            # If adding this group would exceed batch size, yield current batch
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            batch.extend(group)

        # Yield last batch if non-empty
        if batch:
            yield batch

    def __len__(self):
        # Estimate number of batches
        total = sum(len(g) for g in self.groups)
        return max(1, (total + self.batch_size - 1) // self.batch_size)
