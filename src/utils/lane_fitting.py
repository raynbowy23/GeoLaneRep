"""Fit smooth lanelet centerlines from trajectory pseudo-lane clusters.

Given tracklet centroids and pseudo-lane labels (from cross-track clustering),
computes a smooth polyline centerline per pseudo-lane by:
  1. Computing the dominant direction per cluster
  2. Projecting centroids onto the along-track axis
  3. Binning along-track and taking median cross-track position per bin
  4. Fitting a smooth polyline through the bin medians

The resulting "fitted lanelets" are positioned where vehicles actually drive,
complementing the SUMO structural prior with observed trajectory data.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _fit_single_lane(
    centroids: np.ndarray,
    tangents: np.ndarray,
    n_bins: int = 20,
    smooth_window: int = 3,
) -> Optional[np.ndarray]:
    """Fit a smooth centerline through a cluster of tracklet centroids.

    Args:
        centroids: (M, 2) pixel positions of tracklets in this lane.
        tangents: (M, 2) unit tangent vectors.
        n_bins: number of along-track bins for median computation.
        smooth_window: moving-average window for smoothing.

    Returns:
        (K, 2) smooth centerline points, or None if too few points.
    """
    if len(centroids) < 5:
        return None

    # Dominant direction from mean tangent
    mean_t = tangents.mean(axis=0)
    norm = np.linalg.norm(mean_t)
    if norm < 1e-6:
        return None
    along = mean_t / norm
    perp = np.array([-along[1], along[0]])

    # Project centroids onto along-track and cross-track axes
    center = centroids.mean(axis=0)
    rel = centroids - center
    along_proj = rel @ along
    cross_proj = rel @ perp

    # Bin along the along-track axis
    a_min, a_max = along_proj.min(), along_proj.max()
    if a_max - a_min < 10.0:  # too short
        return None

    bin_edges = np.linspace(a_min, a_max, n_bins + 1)
    bin_centers_along = []
    bin_centers_cross = []

    for b in range(n_bins):
        mask = (along_proj >= bin_edges[b]) & (along_proj < bin_edges[b + 1])
        if b == n_bins - 1:  # include right edge in last bin
            mask = mask | (along_proj == bin_edges[b + 1])
        if mask.sum() < 2:
            continue
        bin_centers_along.append(np.median(along_proj[mask]))
        bin_centers_cross.append(np.median(cross_proj[mask]))

    if len(bin_centers_along) < 3:
        return None

    bin_along = np.array(bin_centers_along)
    bin_cross = np.array(bin_centers_cross)

    # Smooth cross-track values with moving average
    if smooth_window > 1 and len(bin_cross) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        # Pad edges to avoid shrinkage
        pad = smooth_window // 2
        padded = np.concatenate([
            np.full(pad, bin_cross[0]),
            bin_cross,
            np.full(pad, bin_cross[-1]),
        ])
        bin_cross = np.convolve(padded, kernel, mode='valid')[:len(bin_along)]

    # Convert back to pixel coordinates
    pts = center + np.outer(bin_along, along) + np.outer(bin_cross, perp)
    return pts.astype(np.float32)


def fit_lanelets_from_tracklets(
    pixel_centroids: np.ndarray,
    pixel_tangents: np.ndarray,
    pseudo_labels: np.ndarray,
    n_bins: int = 20,
    smooth_window: int = 3,
) -> Dict[int, np.ndarray]:
    """Fit smooth lanelet centerlines from tracklet pseudo-lane clusters.

    Args:
        pixel_centroids: (N, 2) tracklet centroids in pixel coordinates.
        pixel_tangents: (N, 2) tracklet tangent vectors.
        pseudo_labels: (N,) integer lane labels (-1 = unassigned).
        n_bins: along-track bins for median computation.
        smooth_window: moving-average smoothing window.

    Returns:
        Dict mapping lane_label -> (K, 2) centerline points in pixel coords.
    """
    unique_labels = sorted(set(pseudo_labels[pseudo_labels >= 0].tolist()))
    fitted = {}

    for lbl in unique_labels:
        mask = pseudo_labels == lbl
        pts = pixel_centroids[mask]
        tgts = pixel_tangents[mask]

        centerline = _fit_single_lane(pts, tgts, n_bins=n_bins,
                                       smooth_window=smooth_window)
        if centerline is not None:
            fitted[lbl] = centerline

    logger.info(f"Fitted {len(fitted)}/{len(unique_labels)} pseudo-lane centerlines")
    return fitted


def match_sumo_to_fitted(
    sumo_lanes_px: Dict[str, np.ndarray],
    fitted_lanes: Dict[int, np.ndarray],
    max_dist: float = 60.0,
) -> Dict[str, int]:
    """Match SUMO lanes to fitted trajectory lanes by proximity + heading.

    Args:
        sumo_lanes_px: {lane_id: (L, 2) pixel points} SUMO lane geometry.
        fitted_lanes: {label: (K, 2) pixel points} fitted centerlines.
        max_dist: max centroid distance for a valid match.

    Returns:
        Dict mapping sumo_lane_id -> fitted_lane_label.
    """
    if not sumo_lanes_px or not fitted_lanes:
        return {}

    # Compute centroid and heading for each SUMO lane
    sumo_info = {}
    for lid, pts in sumo_lanes_px.items():
        if len(pts) < 2:
            continue
        centroid = pts.mean(axis=0)
        direction = pts[-1] - pts[0]
        heading = np.arctan2(direction[1], direction[0])
        sumo_info[lid] = (centroid, heading)

    # Compute centroid and heading for each fitted lane
    fitted_info = {}
    for lbl, pts in fitted_lanes.items():
        centroid = pts.mean(axis=0)
        direction = pts[-1] - pts[0]
        heading = np.arctan2(direction[1], direction[0])
        fitted_info[lbl] = (centroid, heading)

    # Greedy matching: for each SUMO lane, find closest fitted lane
    matches = {}
    for lid, (s_cent, s_head) in sumo_info.items():
        best_lbl = None
        best_score = float('inf')
        for lbl, (f_cent, f_head) in fitted_info.items():
            dist = np.linalg.norm(s_cent - f_cent)
            # Heading similarity (penalize opposing directions)
            head_cos = np.cos(s_head - f_head)
            if head_cos < 0.3:  # > ~72 degrees apart
                continue
            if dist < max_dist and dist < best_score:
                best_score = dist
                best_lbl = lbl
        if best_lbl is not None:
            matches[lid] = best_lbl

    return matches
