"""Geometric lane assignment: assign tracklets to annotated lanes.

No ML training required. Each tracklet is assigned to the annotation lane
with the smallest perpendicular projection distance (in pixel space).

Pipeline:
  trajectory.csv
    → per track: project onto every annotation lane (all groups)
    → assign to nearest lane (or reject if too far)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from src.data.annotation_loader import (
    get_group_lanes,
    load_annotation_json,
)

logger = logging.getLogger(__name__)


def _resample_polyline(polyline: np.ndarray, spacing: float = 5.0) -> np.ndarray:
    """Resample a polyline to uniform spacing (default 5px)."""
    if len(polyline) < 2:
        return polyline
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]
    if total_len < 1e-8:
        return polyline
    n_out = max(int(total_len / spacing) + 1, 2)
    t_out = np.linspace(0.0, total_len, n_out)
    return np.column_stack([
        np.interp(t_out, cum_len, polyline[:, 0]),
        np.interp(t_out, cum_len, polyline[:, 1]),
    ])


def _project_all_points_to_polyline(
    points: np.ndarray, polyline: np.ndarray, chunk_size: int = 50000
) -> np.ndarray:
    """Project N points onto a polyline, returning unsigned min distances.

    Vectorized over both points and segments using broadcasting.
    Processes in chunks to limit memory usage.

    Args:
        points: (N, 2) query points in pixel space.
        polyline: (M, 2) polyline vertices.
        chunk_size: Max points per batch (controls memory).

    Returns:
        (N,) unsigned minimum distance from each point to the polyline.
    """
    n_pts = len(points)
    n_seg = len(polyline) - 1
    if n_seg < 1:
        return np.full(n_pts, np.inf)

    # Segment start/end: (S, 2)
    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a
    ab_len_sq = np.sum(ab ** 2, axis=1)

    valid = ab_len_sq > 1e-12
    if not np.any(valid):
        return np.full(n_pts, np.inf)

    a_v = a[valid]
    ab_v = ab[valid]
    ab_ls_v = ab_len_sq[valid]

    result = np.empty(n_pts, dtype=np.float64)

    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        pts = points[start:end]  # (C, 2)

        # Broadcast: (C, 1, 2) - (1, V, 2) → (C, V, 2)
        ap = pts[:, None, :] - a_v[None, :, :]
        t = np.clip(np.sum(ap * ab_v[None, :, :], axis=2) / ab_ls_v[None, :], 0.0, 1.0)
        proj = a_v[None, :, :] + t[:, :, None] * ab_v[None, :, :]
        dist = np.linalg.norm(pts[:, None, :] - proj, axis=2)
        result[start:end] = dist.min(axis=1)

    return result


def _project_points_signed(
    points: np.ndarray, polyline: np.ndarray
) -> np.ndarray:
    """Project points onto polyline, return signed lateral distances.

    Used for lane-change detection (std of signed lateral).
    Small N (single track), so no chunking needed.
    """
    n_pts = len(points)
    n_seg = len(polyline) - 1
    if n_seg < 1:
        return np.full(n_pts, np.inf)

    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a
    ab_len_sq = np.sum(ab ** 2, axis=1)
    valid = ab_len_sq > 1e-12
    if not np.any(valid):
        return np.full(n_pts, np.inf)

    a_v, ab_v, ab_ls_v = a[valid], ab[valid], ab_len_sq[valid]

    ap = points[:, None, :] - a_v[None, :, :]
    t = np.clip(np.sum(ap * ab_v[None, :, :], axis=2) / ab_ls_v[None, :], 0.0, 1.0)
    proj = a_v[None, :, :] + t[:, :, None] * ab_v[None, :, :]
    diff = points[:, None, :] - proj
    dist = np.linalg.norm(diff, axis=2)

    best_seg = dist.argmin(axis=1)
    row_idx = np.arange(n_pts)
    min_dist = dist[row_idx, best_seg]
    cross = (ab_v[best_seg, 0] * diff[row_idx, best_seg, 1]
             - ab_v[best_seg, 1] * diff[row_idx, best_seg, 0])
    return np.sign(cross) * min_dist


class LaneAssigner:
    """Assigns each tracklet to its nearest annotation lane across all groups.

    Fully vectorized: projects all trajectory points against each lane at once,
    aggregates per-track mean distance, picks the closest lane.

    Usage:
        assigner = LaneAssigner(annotation_path, image_wh=(1920, 1080))
        results = assigner.assign(trajectories, frame_shape)
    """

    def __init__(
        self,
        annotation_path: str,
        image_wh: Tuple[int, int] = (1920, 1080),
        lateral_threshold_px: float = 60.0,
        min_tracklet_points: int = 5,
        lane_change_std_px: float = 30.0,
        lane_group_method: str = "density",  # unused, kept for API compat
    ):
        self.annotation = load_annotation_json(annotation_path)
        self.image_wh = image_wh
        self.lateral_threshold_px = lateral_threshold_px
        self.min_tracklet_points = min_tracklet_points
        self.lane_change_std_px = lane_change_std_px

        # Build flat list of all lanes (pixel, densely resampled)
        wh = np.array(image_wh, dtype=np.float64)
        self._all_lanes: List[dict] = []
        for lg in self.annotation["lane_groups"]:
            gid = lg["group_id"]
            lanes_norm = get_group_lanes(self.annotation, gid, image_wh)
            for lane in lanes_norm:
                wp_px = lane["waypoints"] * wh
                wp_px_dense = _resample_polyline(wp_px, spacing=5.0)
                self._all_lanes.append({
                    "group_id": gid,
                    "cls_id": lane["cls_id"],
                    "waypoints_px": wp_px_dense,
                    "color": lane["color"],
                })

        n_groups = len(self.annotation["lane_groups"])
        logger.info(
            f"LaneAssigner: camera={self.annotation['camera']}, "
            f"{n_groups} groups, {len(self._all_lanes)} total lanes"
        )

    def assign(
        self,
        trajectories: pl.DataFrame,
        frame_shape: Tuple[int, int],
    ) -> pl.DataFrame:
        """Assign each tracklet to its nearest annotation lane.

        Vectorized batch approach:
        1. For each lane, project ALL trajectory points at once → (N,) distances
        2. Attach track IDs, aggregate mean distance per (track, lane)
        3. Pick the lane with smallest mean distance per track
        4. Join back to full trajectories

        Args:
            trajectories: DataFrame with columns: id, time, x, y.
            frame_shape: (height, width) — unused, kept for API compat.

        Returns:
            DataFrame: track_id, time, x, y, lane_id, lane_score, group_id, lane_change.
        """
        empty_schema = {
            "track_id": pl.Int64, "time": pl.Float64,
            "x": pl.Float64, "y": pl.Float64,
            "lane_id": pl.Int64, "lane_score": pl.Float64,
            "group_id": pl.Int64, "lane_change": pl.Boolean,
        }

        if not self._all_lanes:
            return pl.DataFrame(schema=empty_schema)

        # Filter to tracks with enough points
        valid_tracks = (
            trajectories
            .group_by("id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= self.min_tracklet_points)
            .select("id")
        )
        traj = trajectories.join(valid_tracks, on="id", how="inner")
        n_tracks = traj["id"].n_unique()

        if n_tracks == 0:
            return pl.DataFrame(schema=empty_schema)

        logger.info(f"Assigning {n_tracks} tracks to {len(self._all_lanes)} lanes...")

        # All points as numpy: (total_pts, 2)
        all_points = traj.select("x", "y").to_numpy().astype(np.float64)
        all_ids = traj["id"].to_numpy()

        # For each lane, compute distance from every point (vectorized)
        n_lanes = len(self._all_lanes)
        # dist_matrix: (total_pts, n_lanes) — min distance per point per lane
        dist_matrix = np.empty((len(all_points), n_lanes), dtype=np.float64)

        for li, lane in enumerate(self._all_lanes):
            dist_matrix[:, li] = _project_all_points_to_polyline(
                all_points, lane["waypoints_px"]
            )

        # Build a Polars DataFrame for fast group_by aggregation
        lane_names = [f"d_{li}" for li in range(n_lanes)]
        dist_df = pl.DataFrame({
            "id": all_ids,
            **{name: dist_matrix[:, li] for li, name in enumerate(lane_names)},
        })

        # Mean distance per (track, lane)
        agg_exprs = [pl.col(name).mean().alias(name) for name in lane_names]
        track_means = dist_df.group_by("id").agg(agg_exprs)

        # Find best lane per track
        mean_arr = track_means.select(lane_names).to_numpy()  # (n_tracks, n_lanes)
        best_lane_idx = mean_arr.argmin(axis=1)  # (n_tracks,)
        best_score = mean_arr[np.arange(len(mean_arr)), best_lane_idx]  # (n_tracks,)

        # Map lane index → group_id, cls_id
        lane_group_ids = np.array([l["group_id"] for l in self._all_lanes])
        lane_cls_ids = np.array([l["cls_id"] for l in self._all_lanes])

        assigned_group = lane_group_ids[best_lane_idx]
        assigned_lane = lane_cls_ids[best_lane_idx]

        # Apply threshold: reject if best score > threshold
        rejected = best_score > self.lateral_threshold_px
        assigned_group = np.where(rejected, -1, assigned_group)
        assigned_lane = np.where(rejected, -1, assigned_lane)

        # Lane change detection: use std of per-point distances to best lane.
        # Build a lookup: track_id → best_lane_idx
        lane_change = np.zeros(len(track_means), dtype=bool)
        if self.lane_change_std_px > 0:
            tm_ids = track_means["id"].to_numpy()

            # Map each point's track_id → index in track_means (vectorized via Polars join)
            point_idx_df = (
                pl.DataFrame({"id": all_ids, "_pt": np.arange(len(all_ids))})
                .join(
                    pl.DataFrame({"id": tm_ids, "_tidx": np.arange(len(tm_ids))}),
                    on="id", how="inner",
                )
                .sort("_pt")
            )
            point_lane_idx = best_lane_idx[point_idx_df["_tidx"].to_numpy()]
            # Get distance to best lane for each point
            best_lane_dists = dist_matrix[np.arange(len(all_points)), point_lane_idx]

            # Std per track via Polars group_by
            std_df = (
                pl.DataFrame({"id": all_ids, "d": best_lane_dists})
                .group_by("id")
                .agg(pl.col("d").std().alias("std_d"))
            )
            # Join back to track_means order
            std_joined = (
                pl.DataFrame({"id": tm_ids, "_idx": np.arange(len(tm_ids))})
                .join(std_df, on="id", how="left")
                .sort("_idx")
            )
            std_arr = std_joined["std_d"].fill_null(0.0).to_numpy()
            lane_change = (std_arr > self.lane_change_std_px) & ~rejected

        # Build assignment DataFrame
        assign_df = pl.DataFrame({
            "id": track_means["id"],
            "lane_id": assigned_lane.astype(np.int64),
            "group_id": assigned_group.astype(np.int64),
            "lane_score": best_score,
            "lane_change": lane_change,
        })

        # Stats
        n_assigned = int(np.sum(~rejected))
        n_rejected = int(np.sum(rejected))
        n_lc = int(np.sum(lane_change))
        logger.info(
            f"Assignment complete: {n_assigned}/{n_assigned + n_rejected} assigned, "
            f"{n_rejected} rejected, {n_lc} lane-change flagged"
        )

        # Join back to full trajectories
        result = (
            traj
            .join(assign_df, on="id", how="inner")
            .rename({"id": "track_id"})
            .select("track_id", "time", "x", "y", "lane_id", "lane_score", "group_id", "lane_change")
        )

        return result
