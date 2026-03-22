"""Extract per-lane trajectories from SUMO FCD output for re-encoding.

Converts SUMO Floating Car Data (FCD) into the encoder's expected format:
- traj_polylines: list of (K, 2) normalized trajectory polylines per lane
- traj_stats: (4,) [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]

This bridges the gap between SUMO simulation output and the encoder input,
enabling the re-encoding loop: edit lanes → simulate → extract → re-encode → compare.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LaneTrajectoryData:
    """Encoder-compatible trajectory data for one lane."""

    lane_id: str
    traj_polylines: List[np.ndarray]  # list of (K, 2) normalized polylines
    traj_stats: np.ndarray            # (4,) [mean_speed, mean_curv, mean_lat, count_norm]
    raw_trajectories: List[np.ndarray]  # list of (T_i, 2) raw SUMO coords (for debug)


def parse_fcd_output(
    fcd_path: Path,
    allowed_lanes: Optional[set] = None,
) -> Dict[str, List[List[Tuple[float, float, float]]]]:
    """Parse SUMO FCD XML into per-vehicle trajectories grouped by lane.

    Args:
        fcd_path: Path to FCD output XML.
        allowed_lanes: If set, only include vehicles on these lane IDs.

    Returns:
        Dict mapping lane_id -> list of vehicle trajectories.
        Each trajectory is [(x, y, speed), ...] in SUMO coordinates.
    """
    tree = ET.parse(str(fcd_path))
    root = tree.getroot()

    # Collect per-vehicle data: vehicle_id -> [(time, x, y, speed, lane)]
    vehicle_traces: Dict[str, List[Tuple[float, float, float, float, str]]] = {}

    for timestep in root.findall("timestep"):
        t = float(timestep.get("time", "0"))
        for vehicle in timestep.findall("vehicle"):
            vid = vehicle.get("id")
            x = float(vehicle.get("x", "0"))
            y = float(vehicle.get("y", "0"))
            speed = float(vehicle.get("speed", "0"))
            lane = vehicle.get("lane", "")

            if allowed_lanes and lane not in allowed_lanes:
                continue
            # Skip internal junction lanes
            if lane.startswith(":"):
                continue

            vehicle_traces.setdefault(vid, []).append((t, x, y, speed, lane))

    # Group trajectories by dominant lane (the lane where the vehicle
    # spends the most timesteps)
    lane_trajectories: Dict[str, List[List[Tuple[float, float, float]]]] = {}

    for vid, trace in vehicle_traces.items():
        if len(trace) < 3:
            continue

        # Find dominant lane
        lane_counts: Dict[str, int] = {}
        for _, _, _, _, lane in trace:
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
        dominant_lane = max(lane_counts, key=lane_counts.get)

        # Extract (x, y, speed) for timesteps on dominant lane
        points = [
            (x, y, speed)
            for _, x, y, speed, lane in trace
            if lane == dominant_lane
        ]

        if len(points) >= 3:
            lane_trajectories.setdefault(dominant_lane, []).append(points)

    logger.info(
        f"Parsed FCD: {len(vehicle_traces)} vehicles → "
        f"{sum(len(v) for v in lane_trajectories.values())} trajectories "
        f"across {len(lane_trajectories)} lanes"
    )
    return lane_trajectories


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


def _compute_traj_stats(
    trajectories: List[np.ndarray],
    lane_geometry: np.ndarray,
    max_traj_count: int,
) -> np.ndarray:
    """Compute aggregate traj_stats matching the encoder's expected format.

    Returns (4,): [mean_speed, mean_curvature, mean_lateral_offset, traj_count_norm]
    """
    if not trajectories:
        return np.zeros(4, dtype=np.float32)

    speeds = []
    curvatures = []
    lateral_offsets = []

    # Lane direction for lateral offset
    if len(lane_geometry) >= 2:
        lane_tangent = lane_geometry[-1] - lane_geometry[0]
        lane_tangent_norm = np.linalg.norm(lane_tangent)
        if lane_tangent_norm > 1e-8:
            lane_tangent = lane_tangent / lane_tangent_norm
        lane_perp = np.array([-lane_tangent[1], lane_tangent[0]])
        lane_center = lane_geometry.mean(axis=0)
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


def extract_encoder_inputs(
    fcd_path: Path,
    lane_geometries: Dict[str, np.ndarray],
    sumo_to_pixel: Optional[callable] = None,
    polyline_k: int = 16,
    max_traj_per_lane: int = 50,
    allowed_lanes: Optional[set] = None,
) -> Dict[str, LaneTrajectoryData]:
    """Extract encoder-compatible trajectory data from SUMO FCD output.

    This is the main entry point for D-T1. Takes raw SUMO FCD output and
    converts it into the format expected by the LaneEncoder.

    Args:
        fcd_path: Path to FCD output XML.
        lane_geometries: Dict mapping lane_id -> (N, 2) SUMO coordinates.
        sumo_to_pixel: Optional transform function (N, 2) SUMO → (N, 2) pixel.
            If None, normalizes SUMO coords to [0, 1] using bounding box.
        polyline_k: Number of points per resampled polyline.
        max_traj_per_lane: Max trajectories to keep per lane.
        allowed_lanes: If set, only process these lane IDs.

    Returns:
        Dict mapping lane_id -> LaneTrajectoryData.
    """
    # Parse FCD
    raw_trajs = parse_fcd_output(fcd_path, allowed_lanes)

    if not raw_trajs:
        logger.warning("No trajectories extracted from FCD output")
        return {}

    # Compute normalization bounds from all trajectory + geometry points
    all_points = []
    for lane_id, trajs in raw_trajs.items():
        for traj in trajs:
            all_points.extend([(x, y) for x, y, _ in traj])
    for geom in lane_geometries.values():
        all_points.extend(geom.tolist())

    all_points = np.array(all_points)
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    bounds_range = bounds_max - bounds_min
    bounds_range[bounds_range < 1e-8] = 1.0  # prevent division by zero

    def normalize(pts: np.ndarray) -> np.ndarray:
        """Normalize SUMO coordinates to [0, 1] range."""
        if sumo_to_pixel is not None:
            return sumo_to_pixel(pts)
        return (pts - bounds_min) / bounds_range

    # Process each lane
    results = {}
    max_traj_count = max(len(v) for v in raw_trajs.values()) if raw_trajs else 1

    for lane_id, trajs in raw_trajs.items():
        # Get lane geometry (normalized)
        if lane_id in lane_geometries:
            lane_geom_norm = normalize(lane_geometries[lane_id])
        else:
            # Estimate lane geometry from mean trajectory
            # Resample each trajectory to polyline_k points before averaging
            # (trajectories may have different lengths)
            resampled = []
            for t in trajs[:10]:
                pts = np.array([(x, y) for x, y, _ in t])
                if len(pts) < 2:
                    continue
                resampled.append(_resample_polyline(pts, polyline_k))
            if resampled:
                mean_traj = np.mean(resampled, axis=0)
            else:
                continue  # skip lane with no usable trajectories
            lane_geom_norm = normalize(mean_traj)

        # Convert raw trajectories to normalized (x, y) arrays
        traj_arrays = []
        raw_arrays = []
        for traj in trajs:
            pts = np.array([(x, y) for x, y, _ in traj])
            raw_arrays.append(pts)
            pts_norm = normalize(pts)
            traj_arrays.append(pts_norm)

        # Limit trajectory count
        if len(traj_arrays) > max_traj_per_lane:
            indices = np.random.default_rng(42).choice(
                len(traj_arrays), max_traj_per_lane, replace=False
            )
            traj_arrays = [traj_arrays[i] for i in indices]
            raw_arrays = [raw_arrays[i] for i in indices]

        # Resample to K points
        traj_polylines = [
            _resample_polyline(t, polyline_k) for t in traj_arrays
        ]

        # Compute stats using normalized trajectories + geometry
        lane_geom_resampled = _resample_polyline(lane_geom_norm, polyline_k)
        traj_stats = _compute_traj_stats(
            traj_arrays, lane_geom_resampled, max_traj_count,
        )

        results[lane_id] = LaneTrajectoryData(
            lane_id=lane_id,
            traj_polylines=traj_polylines,
            traj_stats=traj_stats,
            raw_trajectories=raw_arrays,
        )

    logger.info(
        f"Extracted encoder inputs for {len(results)} lanes "
        f"(total {sum(len(r.traj_polylines) for r in results.values())} trajectories)"
    )
    return results
