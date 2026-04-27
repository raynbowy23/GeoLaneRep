from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryLane:
    """A lane centerline extracted from trajectory data."""
    centerline: np.ndarray      # (K, 2) normalized [0,1] image coords
    lateral_rank: float         # 0 = leftmost, 1 = rightmost
    n_trajectories: int         # number of vehicle trajectories assigned

@dataclass
class TrajectoryGenResult:
    """Result of trajectory-based lane generation."""
    generated: np.ndarray           # (K, 2) generated lane centerline
    existing_lanes: List[np.ndarray]  # existing lane centerlines for context
    spec: str                       # "rightmost", "leftmost", or "merge"
    anchor_lane: np.ndarray         # anchor lane used for generation
    perp: np.ndarray                # (2,) perpendicular direction
    spacing: float                  # estimated inter-lane spacing

# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _resample(pts: np.ndarray, k: int) -> np.ndarray:
    """Resample a polyline to k evenly-spaced points (arc-length)."""
    if len(pts) < 2:
        return np.tile(pts[0], (k, 1))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total < 1e-10:
        return np.tile(pts[0], (k, 1))
    targets = np.linspace(0, total, k)
    out = np.zeros((k, 2))
    for i, d in enumerate(targets):
        idx = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(pts) - 2)
        seg = cum[idx + 1] - cum[idx]
        t = (d - cum[idx]) / seg if seg > 1e-10 else 0.0
        out[i] = pts[idx] * (1 - t) + pts[idx + 1] * t
    return out

def _smooth(pts: np.ndarray, passes: int = 4) -> np.ndarray:
    """Multi-pass 3-point moving average."""
    for _ in range(passes):
        s = pts.copy()
        for j in range(1, len(pts) - 1):
            s[j] = (pts[j - 1] + pts[j] + pts[j + 1]) / 3.0
        pts = s
    return pts

def _mean_polyline(trajectories: List[np.ndarray], k: int = 32) -> np.ndarray:
    """Compute mean centerline from a list of trajectories via resampling."""
    resampled = [_resample(t, k) for t in trajectories if len(t) >= 2]
    if not resampled:
        return np.zeros((k, 2))
    return np.mean(resampled, axis=0)

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_lane_centerlines(
    pseudo_lanes,
    k: int = 32,
) -> List[TrajectoryLane]:
    """Extract centerlines from PseudoLane objects.

    Args:
        pseudo_lanes: List of PseudoLane from zero_shot_lanes.build_lanes_from_annotation.
        k: Number of waypoints per centerline.

    Returns:
        List of TrajectoryLane sorted left to right.
    """
    lanes = []
    n = len(pseudo_lanes)
    for i, pl in enumerate(pseudo_lanes):
        if not pl.trajectories:
            continue
        cl = _mean_polyline(pl.trajectories, k)
        cl = _smooth(cl)
        lanes.append(TrajectoryLane(
            centerline=cl,
            lateral_rank=i / max(n - 1, 1),
            n_trajectories=len(pl.trajectories),
        ))
    return lanes

def estimate_geometry(lanes: List[TrajectoryLane]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate group tangent, perpendicular, and inter-lane spacing.

    Args:
        lanes: Sorted list of TrajectoryLane.

    Returns:
        tangent (2,), perp (2,), median_spacing (float)
    """
    # Tangent: mean of per-lane (end - start)
    tangents = []
    for lane in lanes:
        cl = lane.centerline
        t = cl[-1] - cl[0]
        n = np.linalg.norm(t)
        if n > 1e-8:
            tangents.append(t / n)

    if tangents:
        tangent = np.mean(tangents, axis=0)
        tangent /= np.linalg.norm(tangent) + 1e-8
    else:
        tangent = np.array([1.0, 0.0])

    # Perpendicular: 90° CCW
    perp = np.array([-tangent[1], tangent[0]])

    # Validate: rightmost lane (rank=1) should have MAX perp projection
    if len(lanes) >= 2:
        centroids = np.array([l.centerline.mean(axis=0) for l in lanes])
        projections = centroids @ perp
        # lanes are sorted left→right; rightmost should have max projection
        # if not, flip perp
        if projections[0] > projections[-1]:
            perp = -perp

    # Spacing: median inter-lane distance along perp
    spacing = 0.05  # default
    if len(lanes) >= 2:
        centroids = np.array([l.centerline.mean(axis=0) for l in lanes])
        lat = centroids @ perp
        spacings = np.diff(np.sort(lat))
        if len(spacings) > 0:
            spacing = float(np.median(spacings))

    return tangent, perp, spacing

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_rightmost(
    lanes: List[TrajectoryLane],
    perp: np.ndarray,
    spacing: float,
    k: int = 32,
) -> np.ndarray:
    """Generate one new lane to the right of the rightmost existing lane."""
    anchor = lanes[-1].centerline  # rightmost
    generated = anchor + perp * spacing
    generated = np.clip(generated, 0.0, 1.0)
    return _smooth(_resample(generated, k))

def generate_leftmost(
    lanes: List[TrajectoryLane],
    perp: np.ndarray,
    spacing: float,
    k: int = 32,
) -> np.ndarray:
    """Generate one new lane to the left of the leftmost existing lane."""
    anchor = lanes[0].centerline  # leftmost
    generated = anchor - perp * spacing
    generated = np.clip(generated, 0.0, 1.0)
    return _smooth(_resample(generated, k))

def generate_merge(
    lanes: List[TrajectoryLane],
    perp: np.ndarray,
    spacing: float,
    k: int = 32,
    convergence_start: float = 3.5,
) -> np.ndarray:
    """Generate a merge lane converging into the rightmost existing lane.

    The merge lane starts offset (convergence_start * spacing) from the
    rightmost lane and converges to join it at the end.

    Args:
        lanes: Sorted lane list.
        perp: Perpendicular direction (pointing right).
        spacing: Inter-lane spacing.
        k: Output waypoints.
        convergence_start: How many lane-widths out to start the merge.
    """
    anchor = lanes[-1].centerline  # rightmost — target to merge into
    K = len(anchor)

    # Taper offset from convergence_start at start to 0 at end
    taper = np.linspace(convergence_start, 0.0, K)[:, None]  # (K, 1)
    generated = anchor + perp * spacing * taper

    generated = np.clip(generated, 0.0, 1.0)
    return _smooth(_resample(generated, k))

# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def generate(
    pseudo_lanes,
    spec: str,
    k: int = 32,
) -> Optional[TrajectoryGenResult]:
    """Generate a new lane from trajectory data.

    Args:
        pseudo_lanes: List of PseudoLane objects from build_lanes_from_annotation.
        spec: One of "rightmost", "leftmost", "merge".
        k: Number of output waypoints.

    Returns:
        TrajectoryGenResult or None if insufficient data.
    """
    lanes = extract_lane_centerlines(pseudo_lanes, k=k)
    if len(lanes) < 2:
        logger.warning(f"Need ≥2 lanes, got {len(lanes)}")
        return None

    tangent, perp, spacing = estimate_geometry(lanes)
    logger.info(
        f"Trajectory geometry: tangent={tangent.round(3)}, "
        f"perp={perp.round(3)}, spacing={spacing:.4f}"
    )

    if spec == "rightmost":
        generated = generate_rightmost(lanes, perp, spacing, k)
        anchor = lanes[-1].centerline
    elif spec == "leftmost":
        generated = generate_leftmost(lanes, perp, spacing, k)
        anchor = lanes[0].centerline
    elif spec == "merge":
        generated = generate_merge(lanes, perp, spacing, k)
        anchor = lanes[-1].centerline
    else:
        raise ValueError(f"Unknown spec: {spec}. Choose rightmost/leftmost/merge.")

    return TrajectoryGenResult(
        generated=generated,
        existing_lanes=[l.centerline for l in lanes],
        spec=spec,
        anchor_lane=anchor,
        perp=perp,
        spacing=spacing,
    )
