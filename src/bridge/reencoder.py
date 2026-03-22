"""Re-encode SUMO simulation trajectories through the trained encoder.

Takes trajectory data extracted from SUMO FCD output (via trajectory_extractor)
and runs it through the LaneEncoder to get embeddings in the same space as
real camera observations. This closes the synthesis loop:

    edit lanes → SUMO simulate → extract trajectories → RE-ENCODE → compare

The re-encoded embeddings are directly comparable to real-lane embeddings
via cosine similarity, enabling quantitative evaluation of lane edits.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.bridge.trajectory_extractor import LaneTrajectoryData

logger = logging.getLogger(__name__)


class LaneReencoder:
    """Re-encode SUMO trajectories through the trained LaneEncoder.

    Args:
        model: Trained LaneEncoder (loaded from checkpoint).
        device: Torch device.
        polyline_k: Number of points per polyline (must match training).
        stats_dim: Stats input dimension (4 traj_stats + 5 role = 9).
    """

    def __init__(self, model, device="cpu", polyline_k=16, stats_dim=9):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.polyline_k = polyline_k
        self.stats_dim = stats_dim
        self.model.eval()

    def reencode(
        self,
        lane_traj_data: Dict[str, LaneTrajectoryData],
        lane_geometries: Dict[str, np.ndarray],
        lane_roles: Optional[Dict[str, np.ndarray]] = None,
        max_traj_per_lane: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Re-encode extracted SUMO trajectories to get embeddings.

        Args:
            lane_traj_data: Dict from trajectory_extractor.extract_encoder_inputs().
            lane_geometries: Dict mapping lane_id -> (K, 2) normalized geometry.
            lane_roles: Optional dict mapping lane_id -> (5,) role tensor values.
                If None, uses default mid-rank interior lane role.
            max_traj_per_lane: Max trajectories per lane for encoding.

        Returns:
            Dict mapping lane_id -> (embed_dim,) embedding array.
        """
        if not lane_traj_data:
            return {}

        lane_ids = list(lane_traj_data.keys())
        B = len(lane_ids)

        # Build batched tensors
        geometry_batch = torch.zeros(B, self.polyline_k, 2)
        stats_batch = torch.zeros(B, self.stats_dim)

        # Find max traj count for padding
        traj_counts = []
        for lid in lane_ids:
            data = lane_traj_data[lid]
            n = min(len(data.traj_polylines), max_traj_per_lane)
            traj_counts.append(n)
        max_T = max(traj_counts) if traj_counts else 1

        traj_batch = torch.zeros(B, max_T, self.polyline_k, 2)
        traj_mask = torch.zeros(B, max_T, dtype=torch.bool)

        for i, lid in enumerate(lane_ids):
            data = lane_traj_data[lid]

            # Geometry
            if lid in lane_geometries:
                geom = lane_geometries[lid]
                if len(geom) != self.polyline_k:
                    geom = self._resample(geom)
                geometry_batch[i] = torch.tensor(geom, dtype=torch.float32)

            # Trajectory polylines
            n_traj = min(len(data.traj_polylines), max_traj_per_lane)
            for j in range(n_traj):
                poly = data.traj_polylines[j]
                if len(poly) != self.polyline_k:
                    poly = self._resample(poly)
                traj_batch[i, j] = torch.tensor(poly, dtype=torch.float32)
                traj_mask[i, j] = True

            # Stats: traj_stats (4) + role (5) = 9
            traj_stats_4 = torch.tensor(data.traj_stats, dtype=torch.float32)

            if lane_roles is not None and lid in lane_roles:
                role = torch.tensor(lane_roles[lid], dtype=torch.float32)
            else:
                # Default role: mid-rank interior lane, group_size=0.5
                role = torch.tensor(
                    [0.5, 0.0, 0.0, 0.0, 0.5], dtype=torch.float32
                )

            stats_batch[i] = torch.cat([traj_stats_4, role])

        # Run encoder
        with torch.no_grad():
            output = self.model(
                geometry=geometry_batch.to(self.device),
                traj_polylines=traj_batch.to(self.device),
                traj_mask=traj_mask.to(self.device),
                traj_stats=stats_batch.to(self.device),
            )

        embeddings = output["embedding"].cpu().numpy()

        result = {}
        for i, lid in enumerate(lane_ids):
            result[lid] = embeddings[i]

        logger.info(f"Re-encoded {len(result)} lanes → {embeddings.shape[1]}-dim embeddings")
        return result

    def reencode_with_matched_roles(
        self,
        lane_traj_data: Dict[str, LaneTrajectoryData],
        lane_geometries: Dict[str, np.ndarray],
        matched_samples: Dict[str, "LaneSample"],
        max_traj_per_lane: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Re-encode using roles from matched real lanes.

        Uses the SUMO→encoder lane matching from sumo_runner to copy
        the real lane's role descriptor (lateral_rank, edge flags, group_size)
        into the re-encoding input. This ensures role-consistency between
        the original and re-encoded embeddings.

        Args:
            lane_traj_data: Extracted trajectory data from SUMO.
            lane_geometries: Normalized lane geometries.
            matched_samples: Dict mapping sumo_lane_id -> LaneSample.
            max_traj_per_lane: Max trajectories per lane.

        Returns:
            Dict mapping lane_id -> (embed_dim,) embedding array.
        """
        lane_roles = {}
        for lid in lane_traj_data:
            if lid in matched_samples:
                sample = matched_samples[lid]
                role_tensor = sample.role.to_tensor(include_group_relative=False)
                lane_roles[lid] = role_tensor.numpy()

        return self.reencode(
            lane_traj_data, lane_geometries, lane_roles, max_traj_per_lane
        )

    def _resample(self, pts: np.ndarray) -> np.ndarray:
        """Resample polyline to self.polyline_k points."""
        if len(pts) < 2:
            return np.zeros((self.polyline_k, 2), dtype=np.float64)

        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum_len[-1]

        if total < 1e-10:
            return np.tile(pts[0], (self.polyline_k, 1))

        target_dists = np.linspace(0, total, self.polyline_k)
        resampled = np.zeros((self.polyline_k, 2), dtype=np.float64)
        for i, d in enumerate(target_dists):
            seg_idx = np.searchsorted(cum_len, d, side="right") - 1
            seg_idx = np.clip(seg_idx, 0, len(pts) - 2)
            seg_start = cum_len[seg_idx]
            seg_end = cum_len[seg_idx + 1]
            seg_len = seg_end - seg_start
            t = (d - seg_start) / seg_len if seg_len > 1e-10 else 0.0
            resampled[i] = pts[seg_idx] * (1 - t) + pts[seg_idx + 1] * t

        return resampled
