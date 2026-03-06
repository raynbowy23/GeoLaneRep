"""Oriented density splatting and road contour extraction.

Ported from V1 hybrid_lane_detector.py as standalone pure functions.
Used to compute spatial road boundaries that constrain pseudo-label clustering.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import polars as pl
from scipy import ndimage

logger = logging.getLogger(__name__)


def build_tracklets_for_density(
    trajectories: pl.DataFrame,
    min_track_points: int = 5,
    tracklet_length: int = 15,
    return_ids: bool = False,
) -> np.ndarray:
    """Build oriented tracklet points from vehicle tracks.

    Each track is split into non-overlapping segments of `tracklet_length`
    points. Per segment the endpoint displacement gives the tangent direction
    theta in [0, pi) (folded so opposing lanes align for splatting).

    All points along each segment are emitted (not just centroids), so
    density is constrained to where vehicles actually drove.

    Args:
        trajectories: DataFrame with 'id', 'x', 'y' columns.
        min_track_points: Minimum points per track to include.
        tracklet_length: Number of points per tracklet segment.
        return_ids: If True, return (tracklets, track_ids) tuple.

    Returns:
        (N_points, 4) array: [x, y, cos(theta), sin(theta)].
        If return_ids: tuple of (tracklets, track_ids_array).
    """
    # Vectorised: add within-track row index and segment id
    df = trajectories.select("id", "x", "y").with_columns(
        pl.col("x").cum_count().over("id").alias("_row"),
    )
    # Filter short tracks
    track_len = df.group_by("id").agg(pl.len().alias("_n"))
    valid_ids = track_len.filter(pl.col("_n") >= min_track_points)["id"]
    df = df.filter(pl.col("id").is_in(valid_ids))
    if df.is_empty():
        return np.empty((0, 4), dtype=np.float32)

    # Segment id within each track
    df = df.with_columns(
        ((pl.col("_row") - 1) // tracklet_length).alias("_seg"),
    )

    # Per-segment first/last point for displacement direction
    seg_stats = df.group_by("id", "_seg").agg(
        pl.first("x").alias("x0"), pl.first("y").alias("y0"),
        pl.last("x").alias("x1"), pl.last("y").alias("y1"),
        pl.len().alias("_n"),
    ).filter(pl.col("_n") >= 2)

    dx = (seg_stats["x1"] - seg_stats["x0"]).to_numpy()
    dy = (seg_stats["y1"] - seg_stats["y0"]).to_numpy()
    mag = np.sqrt(dx * dx + dy * dy)
    valid = mag > 1e-6
    seg_stats = seg_stats.filter(pl.Series(valid))
    dx, dy = dx[valid], dy[valid]
    mag = mag[valid]

    theta = np.arctan2(dy, dx) % np.pi
    cos_th = np.cos(theta).astype(np.float32)
    sin_th = np.sin(theta).astype(np.float32)

    # Map (id, seg) -> (cos, sin)
    seg_keys = seg_stats.select("id", "_seg").with_columns(
        pl.Series("_cos", cos_th),
        pl.Series("_sin", sin_th),
    )

    # Join back to all points
    result = df.join(seg_keys, on=["id", "_seg"], how="inner")
    empty = np.empty((0, 4), dtype=np.float32)
    if result.is_empty():
        return (empty, np.empty(0, dtype=np.int64)) if return_ids else empty

    out = np.column_stack([
        result["x"].to_numpy(zero_copy_only=False).astype(np.float32),
        result["y"].to_numpy(zero_copy_only=False).astype(np.float32),
        result["_cos"].to_numpy(zero_copy_only=False),
        result["_sin"].to_numpy(zero_copy_only=False),
    ])
    if return_ids:
        ids = result["id"].to_numpy(zero_copy_only=False)
        return out, ids
    return out


def _build_oriented_kernel(
    theta: float,
    sigma_along: float,
    sigma_across: float,
) -> np.ndarray:
    """Build a 2D anisotropic Gaussian kernel oriented along theta.

    Args:
        theta: Orientation angle in radians (direction of elongation).
        sigma_along: sigma along the road direction (large).
        sigma_across: sigma across the road direction (small).

    Returns:
        Normalized 2D kernel (float32).
    """
    size = int(np.ceil(6 * sigma_along)) | 1
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    u = x * cos_t - y * sin_t # along road
    v = x * sin_t + y * cos_t # across road

    kernel = np.exp(-0.5 * (u ** 2 / (sigma_along ** 2) + v ** 2 / (sigma_across ** 2)))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel.astype(np.float32)


def splat_oriented_density(
    tracklets: np.ndarray,
    frame_shape: Tuple[int, int],
    sigma_along: float = 25.0,
    sigma_across: float = 6.0,
    n_angle_bins: int = 12,
    downsample: int = 1,
) -> np.ndarray:
    """Splat tracklets into an oriented density field.

    Quantizes tracklet angles into bins, builds one rotated kernel per bin,
    accumulates tracklet centroids into point maps, and convolves each bin
    with its oriented kernel. Sum across bins and log-compress.

    Args:
        tracklets: (N, 4) array [cx, cy, cos(theta), sin(theta)].
        frame_shape: (height, width).
        sigma_along: sigma along road direction.
        sigma_across: sigma across road direction.
        n_angle_bins: Number of angle bins over [0, pi).
        downsample: Compute at 1/downsample resolution, then upscale.

    Returns:
        (h, w) float32 log-density field.
    """
    h, w = frame_shape
    ds = max(1, int(downsample))
    sh, sw = (h + ds - 1) // ds, (w + ds - 1) // ds
    density = np.zeros((sh, sw), dtype=np.float32)

    if len(tracklets) == 0:
        if ds > 1:
            return np.zeros((h, w), dtype=np.float32)
        return density

    # Scale coordinates and sigmas for downsampled grid
    scale_inv = 1.0 / ds
    coords_all = tracklets[:, :2] * scale_inv
    eff_sigma_along = sigma_along * scale_inv
    eff_sigma_across = sigma_across * scale_inv

    angles = np.arctan2(tracklets[:, 3], tracklets[:, 2]) % np.pi
    bin_edges = np.linspace(0, np.pi, n_angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(
        (angles / np.pi * n_angle_bins).astype(int),
        0, n_angle_bins - 1,
    )

    kernels = [_build_oriented_kernel(bc, eff_sigma_along, eff_sigma_across)
               for bc in bin_centers]

    for b in range(n_angle_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue

        point_map = np.zeros((sh, sw), dtype=np.float32)
        coords = np.round(coords_all[mask]).astype(np.int32)
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < sw) & \
                (coords[:, 1] >= 0) & (coords[:, 1] < sh)
        coords = coords[valid]
        np.add.at(point_map, (coords[:, 1], coords[:, 0]), 1)

        filtered = cv2.filter2D(point_map, -1, kernels[b])
        density += filtered

    result = np.log1p(density).astype(np.float32)
    if ds > 1:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
    return result


def extract_road_contours(
    density: np.ndarray,
    threshold_pct: float = 45.0,
    min_contour_area: float = 500.0,
) -> List[np.ndarray]:
    """Extract contours from a log-density field via hysteresis threshold.

    Seeds at high percentile, expands to low percentile via connected-component
    flood fill. Produces smooth contours via morphological operations.

    Args:
        density: (h, w) float32 log-density field.
        threshold_pct: Top N% of nonzero values kept as road.
        min_contour_area: Minimum contour area to keep.

    Returns:
        List of OpenCV contour arrays.
    """
    nonzero = density[density > 0]
    if len(nonzero) == 0:
        return []

    lo_pct = 100 - threshold_pct
    hi_pct = 100 - threshold_pct * 0.4
    thr_lo = float(np.percentile(nonzero, lo_pct))
    thr_hi = float(np.percentile(nonzero, hi_pct))
    if thr_lo <= 0:
        thr_lo = float(np.percentile(nonzero, 50))
    if thr_hi <= thr_lo:
        thr_hi = thr_lo

    seeds = density >= thr_hi
    low_mask = density >= thr_lo

    labeled, n_components = ndimage.label(low_mask)
    if n_components == 0:
        return []

    component_has_seed = np.zeros(n_components + 1, dtype=bool)
    seed_labels = labeled[seeds]
    if seed_labels.size > 0:
        component_has_seed[np.unique(seed_labels[seed_labels > 0])] = True

    binary = component_has_seed[labeled].astype(np.uint8) * 255

    found, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.astype(np.float32) for c in found if cv2.contourArea(c) >= min_contour_area]


def assign_tracklets_to_contours(
    centroids: np.ndarray,
    contours: List[np.ndarray],
    frame_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Assign each tracklet centroid to the containing road contour.

    Uses rasterized label-map lookup for O(C) drawContours + O(N) index
    instead of O(N*C) pointPolygonTest calls.

    Args:
        centroids: (N, 2) pixel positions [x, y].
        contours: List of OpenCV contour arrays from extract_road_contours.
        frame_shape: (H, W) of the frame.  When None, inferred from contour
            bounding boxes (slight overhead).

    Returns:
        (N,) int array — contour index per tracklet, -1 if outside all contours.
    """
    N = len(centroids)
    if N == 0:
        return np.full(0, -1, dtype=np.int32)

    # Determine label-map dimensions
    if frame_shape is not None:
        H, W = frame_shape
    else:
        # Infer from contour extents + centroid extents
        max_x = int(centroids[:, 0].max()) + 1
        max_y = int(centroids[:, 1].max()) + 1
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            max_x = max(max_x, int(pts[:, 0].max()) + 1)
            max_y = max(max_y, int(pts[:, 1].max()) + 1)
        H, W = max_y, max_x

    # Rasterize contours into a label map (later contours overwrite earlier,
    # so draw in reverse so lower-index contours win ties)
    label_map = np.full((H, W), -1, dtype=np.int32)
    for ci in reversed(range(len(contours))):
        cv2.drawContours(label_map, [contours[ci].astype(np.int32)], -1, ci, -1)

    # Vectorized integer-index lookup
    xs = np.clip(np.round(centroids[:, 0]).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(centroids[:, 1]).astype(np.int32), 0, H - 1)
    return label_map[ys, xs]


# ---------------------------------------------------------------------------
# Lane group detection via contour-first + per-contour direction split
# ---------------------------------------------------------------------------

def _compute_track_stats(
    trajectories: pl.DataFrame,
    min_track_points: int = 5,
) -> pl.DataFrame:
    """Compute per-vehicle heading, position, and scale statistics.

    Derives instantaneous heading from consecutive (x, y) displacements,
    then averages per track to get (mean_sin, mean_cos, mean_x, mean_y,
    mean_scale).

    ``mean_scale`` is ``log1p(mean(bbox_h))`` when the ``bbox_h`` column
    is present, otherwise falls back to ``mean_y`` (a weak depth proxy).
    """
    traj = trajectories.sort("id", "time" if "time" in trajectories.columns else "frame_num")

    traj = traj.with_columns([
        (pl.col("x").shift(-1).over("id") - pl.col("x")).alias("dx"),
        (pl.col("y").shift(-1).over("id") - pl.col("y")).alias("dy"),
    ])
    traj = traj.filter(pl.col("dx").is_not_null())

    speed = (pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt()
    traj = traj.filter(speed > 0.5)

    norm = (pl.col("dx") ** 2 + pl.col("dy") ** 2).sqrt()
    traj = traj.with_columns([
        (pl.col("dy") / norm).alias("dir_sin"),
        (pl.col("dx") / norm).alias("dir_cos"),
    ])

    has_bbox_h = "bbox_h" in traj.columns

    agg_exprs = [
        pl.col("dir_sin").mean().alias("mean_sin"),
        pl.col("dir_cos").mean().alias("mean_cos"),
        pl.col("x").mean().alias("mean_x"),
        pl.col("y").mean().alias("mean_y"),
        pl.len().alias("n_points"),
    ]
    if has_bbox_h:
        agg_exprs.append(pl.col("bbox_h").mean().alias("_raw_bbox_h"))

    stats = traj.group_by("id").agg(agg_exprs)
    stats = stats.filter(pl.col("n_points") >= min_track_points)

    # Compute mean_scale: log1p(mean(bbox_h)) or fallback to mean_y
    if has_bbox_h:
        stats = stats.with_columns(
            pl.col("_raw_bbox_h").log1p().alias("mean_scale"),
        ).drop("_raw_bbox_h")
    else:
        stats = stats.with_columns(
            pl.col("mean_y").alias("mean_scale"),
        )

    return stats


def _cluster_headings(
    headings: np.ndarray,
    min_gap_deg: float = 45.0,
) -> np.ndarray:
    """Circular peak-based clustering on vehicle headings.

    Builds a smoothed circular histogram, finds peaks (heading modes),
    and assigns each vehicle to the nearest peak.  Peaks closer than
    *min_gap_deg* are merged when the valley between them is shallow.

    This replaces the old gap-between-sorted-headings approach, which
    always fell back to a binary split when the number of vehicles was
    large (avg gap << min_gap_deg).

    Args:
        headings: (N,) array of heading angles in radians (-pi, pi].
        min_gap_deg: minimum angular separation (degrees) to keep two
            peaks as separate groups. Closer peaks are merged unless
            a deep valley separates them.

    Returns:
        (N,) int array of group labels (0-indexed).
    """
    n = len(headings)

    def _binary_fallback():
        dom = np.arctan2(
            np.sin(2 * headings).mean(),
            np.cos(2 * headings).mean(),
        ) / 2
        return np.where(np.cos(headings - dom) >= 0, 0, 1)

    if n < 6:
        return _binary_fallback()

    # ---- Smoothed circular histogram ----
    n_bins = 72                              # 5-degree bins
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts, _ = np.histogram(headings, bins=bin_edges)

    sigma_bins = 1.5                         # Gaussian σ ≈ 7.5°
    smooth = ndimage.gaussian_filter1d(
        counts.astype(np.float64), sigma=sigma_bins, mode="wrap")

    peak_max = smooth.max()
    if peak_max == 0:
        return _binary_fallback()

    # ---- Find peaks (local maxima above 5 % of tallest) ----
    peak_thresh = peak_max * 0.05
    peaks = []
    for i in range(n_bins):
        p = (i - 1) % n_bins
        nx = (i + 1) % n_bins
        if (smooth[i] > smooth[p]
                and smooth[i] >= smooth[nx]
                and smooth[i] > peak_thresh):
            peaks.append(i)

    if len(peaks) < 2:
        return _binary_fallback()

    # ---- Merge peaks closer than min_gap_deg with shallow valley ----
    min_gap_rad = np.radians(min_gap_deg)
    bin_width = 2 * np.pi / n_bins

    def _circ_dist_bins(a, b):
        """Circular distance in radians between bin centres a and b."""
        d = abs(bin_centers[a] - bin_centers[b])
        return min(d, 2 * np.pi - d)

    def _valley_depth(a, b):
        """Min smoothed value on the shorter arc between bins a and b."""
        if b >= a:
            fwd = smooth[a:b + 1].min()
            bwd = min(smooth[a::-1].min(), smooth[b:].min())
        else:
            fwd = min(smooth[a:].min(), smooth[:b + 1].min())
            bwd = smooth[b:a + 1].min()
        return min(fwd, bwd)

    # Iteratively merge the closest pair with a shallow valley
    changed = True
    while changed and len(peaks) > 1:
        changed = False
        best_i = -1
        best_ratio = -1.0
        for i in range(len(peaks)):
            j = (i + 1) % len(peaks)
            d = _circ_dist_bins(peaks[i], peaks[j])
            if d >= min_gap_rad:
                continue
            v = _valley_depth(peaks[i], peaks[j])
            smaller = min(smooth[peaks[i]], smooth[peaks[j]])
            ratio = v / smaller if smaller > 0 else 1.0
            # Shallow valley (ratio > 0.3) → merge
            if ratio > 0.3 and (best_i < 0 or ratio > best_ratio):
                best_i = i
                best_ratio = ratio
        if best_i >= 0:
            j = (best_i + 1) % len(peaks)
            # Keep the taller peak
            drop = j if smooth[peaks[best_i]] >= smooth[peaks[j]] else best_i
            peaks.pop(drop)
            changed = True

    if len(peaks) < 2:
        return _binary_fallback()

    # ---- Assign each heading to nearest peak ----
    peak_angles = np.array([bin_centers[p] for p in peaks])  # (K,)
    diffs = headings[:, None] - peak_angles[None, :]         # (N, K)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi           # wrap [-π, π]
    labels = np.argmin(np.abs(diffs), axis=1)

    return labels


def _smooth_binary_mask(binary: np.ndarray) -> np.ndarray:
    """Morph close -> Gaussian blur -> morph open to smooth a binary mask."""
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), 3.0)
    smoothed = (smoothed >= 127).astype(np.uint8) * 255
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, k_open)
    return smoothed


def _refine_contour_with_vehicles(
    contour: np.ndarray,
    vehicle_positions: np.ndarray,
    frame_shape: Tuple[int, int],
    buffer_px: float = 25.0,
    min_vehicles: int = 5,
) -> Optional[np.ndarray]:
    """Intersect a density contour with a buffered mask of vehicle positions.

    Tightens contours to only cover areas where vehicles actually drove,
    preventing bleed into off-road regions.

    Args:
        contour: OpenCV contour array.
        vehicle_positions: (N, 2) array of [x, y] vehicle centroids.
        frame_shape: (height, width).
        buffer_px: Radius of circles drawn at each vehicle position.
        min_vehicles: Minimum vehicles required; return None if fewer.

    Returns:
        Refined contour, or None if result is too small.
    """
    if len(vehicle_positions) < min_vehicles:
        return None

    h, w = frame_shape

    # Rasterize density contour
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour.astype(np.int32)], -1, 255, -1)

    # Build vehicle trajectory mask — filled circles at each position
    vehicle_mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(1, int(round(buffer_px)))
    for px, py in vehicle_positions:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(vehicle_mask, (ix, iy), radius, 255, -1)

    # Morphological closing to bridge gaps between vehicle circles
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, k_close)

    # AND-intersect: keep only contour pixels near vehicles
    refined_mask = cv2.bitwise_and(contour_mask, vehicle_mask)

    # Extract refined contour
    found, _ = cv2.findContours(
        refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None

    best = max(found, key=cv2.contourArea)
    min_area = cv2.contourArea(contour) * 0.15  # must retain >= 15% of original
    if cv2.contourArea(best) < min_area:
        return None

    eps = 0.002 * cv2.arcLength(best, True)
    return cv2.approxPolyDP(best, eps, True).astype(np.float32)


def _compute_hull_contour(
    xs: np.ndarray,
    ys: np.ndarray,
    frame_shape: Tuple[int, int],
    buffer_px: float = 20.0,
) -> Optional[np.ndarray]:
    """Compute convex hull contour from vehicle positions with buffer.

    Args:
        xs: (N,) array of x positions.
        ys: (N,) array of y positions.
        frame_shape: (height, width).
        buffer_px: Dilation radius around the hull.

    Returns:
        OpenCV contour (float32), or None if too few points.
    """
    if len(xs) < 3:
        return None

    h, w = frame_shape
    points = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(points)

    if cv2.contourArea(hull) < 100:
        return None

    # Rasterize hull and dilate to add buffer
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [hull.astype(np.int32)], -1, 255, -1)

    buf = max(1, int(round(buffer_px)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (buf * 2 + 1, buf * 2 + 1))
    mask = cv2.dilate(mask, kernel)

    found, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not found:
        return None

    best = max(found, key=cv2.contourArea)
    eps = 0.002 * cv2.arcLength(best, True)
    return cv2.approxPolyDP(best, eps, True).astype(np.float32)


def _split_contour_at_tracklet_gaps(
    contour: np.ndarray,
    veh_xs: np.ndarray,
    veh_ys: np.ndarray,
    veh_headings: np.ndarray,
    frame_shape: Tuple[int, int],
    min_area: float,
    n_slices: int = 30,
    gap_ratio: float = 0.25,
) -> List[np.ndarray]:
    """Split a contour where vehicle tracklet density drops (overpass/underpass).

    Projects vehicle positions inside the contour onto the dominant heading
    axis, builds a 1D histogram, and splits at valleys where count drops
    below gap_ratio * median.

    Args:
        contour: OpenCV contour array.
        veh_xs, veh_ys: All vehicle centroid positions.
        veh_headings: All vehicle headings (rad).
        frame_shape: (height, width).
        min_area: Minimum sub-contour area to keep.
        n_slices: Number of slices along the road axis.
        gap_ratio: Split where count < gap_ratio * median_count.

    Returns:
        List of sub-contours, or [contour] if no valid split found.
    """
    fh, fw = frame_shape

    # Find vehicles inside this contour
    inside = np.array([
        cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
        for x, y in zip(veh_xs, veh_ys)
    ])
    n_in = int(inside.sum())
    if n_in < 20:
        return [contour]

    in_xs = veh_xs[inside]
    in_ys = veh_ys[inside]
    in_h = veh_headings[inside]

    # Dominant heading of vehicles in this contour
    dom_h = float(np.arctan2(np.sin(in_h).mean(), np.cos(in_h).mean()))
    along = np.array([np.cos(dom_h), np.sin(dom_h)])
    perp = np.array([-np.sin(dom_h), np.cos(dom_h)])

    # Project along dominant direction
    along_proj = in_xs * along[0] + in_ys * along[1]
    pmin, pmax = along_proj.min(), along_proj.max()
    if pmax - pmin < 50:
        return [contour]

    # 1D histogram
    bins = np.linspace(pmin, pmax, n_slices + 1)
    counts, _ = np.histogram(along_proj, bins=bins)
    median_count = float(np.median(counts))
    if median_count < 5:
        return [contour]

    threshold = gap_ratio * median_count

    # Find gap zones (consecutive slices below threshold)
    is_gap = counts < threshold
    # Don't count edge slices as gaps (vehicles just thin out at frame edges)
    edge_margin = max(2, n_slices // 10)
    is_gap[:edge_margin] = False
    is_gap[-edge_margin:] = False

    if not is_gap.any():
        return [contour]

    # Find contiguous gap regions
    gap_starts = []
    gap_ends = []
    in_gap = False
    for i in range(len(is_gap)):
        if is_gap[i] and not in_gap:
            gap_starts.append(i)
            in_gap = True
        elif not is_gap[i] and in_gap:
            gap_ends.append(i)
            in_gap = False
    if in_gap:
        gap_ends.append(len(is_gap))

    if not gap_starts:
        return [contour]

    # Use the deepest gap (lowest count) as split point
    best_gap_idx = -1
    best_gap_min = float('inf')
    for gi in range(len(gap_starts)):
        gap_min = float(counts[gap_starts[gi]:gap_ends[gi]].min())
        if gap_min < best_gap_min:
            best_gap_min = gap_min
            best_gap_idx = gi

    # Split boundary: middle of the deepest gap zone
    gap_s = gap_starts[best_gap_idx]
    gap_e = gap_ends[best_gap_idx]
    split_bin = (gap_s + gap_e) // 2
    split_val = 0.5 * (bins[split_bin] + bins[split_bin + 1])

    logger.info(
        f"Tracklet gap detected: slices {gap_s}-{gap_e}, "
        f"min_count={best_gap_min:.0f} (threshold={threshold:.0f}, "
        f"median={median_count:.0f})")

    # Split vehicles into two groups: before and after the gap
    before = along_proj < split_val
    after = along_proj >= split_val
    n_before = int(before.sum())
    n_after = int(after.sum())

    if n_before < 10 or n_after < 10:
        return [contour]

    # Generate sub-contours from each group of vehicles
    contour_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour.astype(np.int32)], -1, 255, -1)

    # All vehicle positions projected
    all_along = veh_xs * along[0] + veh_ys * along[1]

    sub_contours = []
    for mask_sel, group_name in [(before, "before"), (after, "after")]:
        # Create point mask from the vehicle positions in this group
        grp_xs = in_xs[mask_sel]
        grp_ys = in_ys[mask_sel]
        pt_mask = np.zeros((fh, fw), dtype=np.uint8)
        for x, y in zip(grp_xs, grp_ys):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < fw and 0 <= iy < fh:
                cv2.circle(pt_mask, (ix, iy), 8, 255, -1)

        # Dilate to connect nearby points, then intersect with parent contour
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        pt_mask = cv2.dilate(pt_mask, kern, iterations=3)
        pt_mask = cv2.bitwise_and(pt_mask, contour_mask)

        # Close holes
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        pt_mask = cv2.morphologyEx(pt_mask, cv2.MORPH_CLOSE, close_k)

        found, _ = cv2.findContours(
            pt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            continue
        best = max(found, key=cv2.contourArea)
        if cv2.contourArea(best) < min_area:
            continue
        eps = 0.002 * cv2.arcLength(best, True)
        sub_contours.append(cv2.approxPolyDP(best, eps, True).astype(np.float32))

    if len(sub_contours) < 2:
        return [contour]

    logger.info(
        f"Split contour at tracklet gap into {len(sub_contours)} parts")
    return sub_contours


def _split_contour_at_gaps(
    contour: np.ndarray,
    density: np.ndarray,
    min_area: float,
    neck_ratio: float = 0.4,
) -> List[np.ndarray]:
    """Split a contour at density valleys (overpass/underpass separation).

    Uses Otsu threshold on within-contour density to find high-density cores,
    then assigns ALL contour pixels to the nearest core (full coverage).
    Validates the split via distance-transform neck measurement at the
    inter-territory boundary — not the valley edge.

    Args:
        contour: OpenCV contour array.
        density: (h, w) log-compressed density field.
        min_area: Minimum sub-contour area to keep.
        neck_ratio: Neck thickness / max thickness threshold. Below this
            ratio the gap is considered a genuine neck and the split is kept.

    Returns:
        List of sub-contours, or [contour] if no valid split found.
    """
    h, w = density.shape[:2]

    # Rasterize contour to binary mask
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour.astype(np.int32)], -1, 255, -1)

    # Extract density values within contour
    within = density[contour_mask > 0]
    nonzero = within[within > 0]
    if len(nonzero) < 50:
        return [contour]

    # Otsu threshold on within-contour density
    vals_u8 = np.clip(
        (nonzero / nonzero.max() * 255), 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_OTSU)
    real_thr = (otsu_thr / 255.0) * nonzero.max()

    # High-density mask clipped to contour
    high_mask = ((density >= real_thr) & (contour_mask > 0)).astype(np.uint8)

    # Connected components of high-density region
    labeled, n_cc = ndimage.label(high_mask)
    if n_cc <= 1:
        return [contour]

    # Keep only CCs that are >= 15% of parent contour area (discard noise)
    parent_pixels = int((contour_mask > 0).sum())
    min_cc_pixels = int(parent_pixels * 0.15)
    cc_sizes = ndimage.sum(high_mask > 0, labeled, range(1, n_cc + 1))
    keep_ids = [i + 1 for i, sz in enumerate(cc_sizes) if sz >= min_cc_pixels]
    if len(keep_ids) < 2:
        return [contour]

    # Re-label with only the large CCs
    filtered = np.zeros_like(labeled)
    for new_id, old_id in enumerate(keep_ids, 1):
        filtered[labeled == old_id] = new_id
    labeled = filtered
    n_cc = len(keep_ids)

    # Assign every contour pixel to nearest CC (full coverage, no shrinkage)
    _, nearest_idx = ndimage.distance_transform_edt(
        labeled == 0, return_indices=True)
    territory = labeled[nearest_idx[0], nearest_idx[1]]
    territory = np.where(contour_mask > 0, territory, 0)

    # Neck validation at the inter-territory boundary
    dist = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
    max_thickness = float(dist.max())
    if max_thickness < 1.0:
        return [contour]

    # Boundary = contour pixels where neighboring territories differ
    terr_max = ndimage.maximum_filter(territory, size=3)
    terr_min = ndimage.minimum_filter(territory, size=3)
    boundary = (territory > 0) & (terr_max != terr_min)

    if boundary.any():
        neck_thickness = float(dist[boundary].min())
    else:
        neck_thickness = 0.0

    if neck_thickness > neck_ratio * max_thickness:
        logger.debug(
            f"Neck too thick ({neck_thickness:.1f}/{max_thickness:.1f}"
            f"={neck_thickness / max_thickness:.2f}), skip split")
        return [contour]

    # Extract sub-contours from each territory (covers full parent area)
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sub_contours = []
    for cc_id in range(1, n_cc + 1):
        terr_mask = ((territory == cc_id) * 255).astype(np.uint8)
        terr_mask = cv2.morphologyEx(terr_mask, cv2.MORPH_CLOSE, k_smooth)
        found, _ = cv2.findContours(
            terr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in found:
            if cv2.contourArea(c) >= min_area:
                sub_contours.append(c.astype(np.float32))

    if len(sub_contours) < 2:
        return [contour]

    logger.info(
        f"Split contour (area={cv2.contourArea(contour):.0f}) into "
        f"{len(sub_contours)} sub-contours at density gaps "
        f"(neck={neck_thickness:.1f}, max_thick={max_thickness:.1f}, "
        f"ratio={neck_thickness / max_thickness:.2f})")
    return sub_contours


def _detect_lane_groups_density(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    tracklet_length: int = 15,
    min_track_points: int = 5,
    sigma_along: float = 25.0,
    sigma_across: float = 6.0,
    min_gap_deg: float = 45.0,
    neck_ratio: float = 0.4,
    min_vehicles_per_group: int = 5,
    refine_contours: bool = True,
    refine_buffer_px: float = 25.0,
    min_extent_ratio: float = 0.25,
    min_road_vehicle_pct: float = 0.05,
    **kwargs,
) -> Tuple[List[np.ndarray], Dict[int, float], Optional[pl.DataFrame]]:
    """Detect lane groups via density splatting + contour extraction (legacy).

    Pipeline:
    1. Compute per-vehicle heading and position from trajectory displacements
    2. Build tracklets from ALL vehicles -> splat oriented density -> extract road contours
       (combined density gives spatial separation from the density field)
    3. Per contour: find vehicles inside -> _cluster_headings() for direction labels
       - If unimodal or < 10 vehicles: keep parent contour as single lane group
       - If 2+ groups with >= 80 deg separation: per-direction density within parent
         mask -> argmax ownership -> non-overlapping sub-contours
    4. Return contours + headings dict

    Returns:
        lane_group_contours: list of OpenCV contour arrays (one per group)
        group_headings: {group_id: heading_rad}
        track_stats: pl.DataFrame from _compute_track_stats (or None)
    """
    fh, fw = frame_shape
    min_area = fh * fw * 0.02  # 2% of frame — filters tiny fragments

    # Step 1: Per-vehicle heading + position stats
    track_stats = _compute_track_stats(trajectories, min_track_points)
    if len(track_stats) < 4:
        return [], {}, None

    track_ids = track_stats["id"].to_numpy()
    mean_sins = track_stats["mean_sin"].to_numpy()
    mean_coss = track_stats["mean_cos"].to_numpy()
    mean_xs = track_stats["mean_x"].to_numpy()
    mean_ys = track_stats["mean_y"].to_numpy()
    headings = np.arctan2(mean_sins, mean_coss)
    n_vehicles = len(track_ids)

    # Step 2: Combined density from ALL vehicles -> road contours
    # Build tracklets ONCE with IDs — reused for per-direction density later.
    all_tracklets, all_tracklet_ids = build_tracklets_for_density(
        trajectories, min_track_points=min_track_points,
        tracklet_length=tracklet_length, return_ids=True)
    if len(all_tracklets) < 3:
        return [], {}, track_stats

    density = splat_oriented_density(
        all_tracklets, frame_shape,
        sigma_along=sigma_along, sigma_across=sigma_across,
        downsample=2)
    if density.max() == 0:
        return [], {}, track_stats

    road_contours = extract_road_contours(
        density, threshold_pct=45.0, min_contour_area=min_area)
    if not road_contours:
        return [], {}, track_stats

    # Smooth density-derived contours
    smoothed_contours = []
    for cnt in road_contours:
        mask_img = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(mask_img, [cnt.astype(np.int32)], -1, 255, -1)
        mask_img = _smooth_binary_mask(mask_img)
        found, _ = cv2.findContours(
            mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in found:
            if cv2.contourArea(c) >= min_area:
                eps = 0.002 * cv2.arcLength(c, True)
                smoothed_contours.append(
                    cv2.approxPolyDP(c, eps, True).astype(np.float32))
    if smoothed_contours:
        road_contours = smoothed_contours

    # Step 2.5: Split contours at density necks (e.g. overpass/underpass)
    split_contours = []
    for cnt in road_contours:
        split_contours.extend(
            _split_contour_at_gaps(cnt, density, min_area, neck_ratio))
    road_contours = split_contours

    # Global heading clustering (computed once on ALL vehicles so that
    # peaks visible only globally — e.g. overpass vs underpass at ~36° apart —
    # are preserved when we later look at a single road contour).
    global_dir_labels = _cluster_headings(headings, min_gap_deg)

    logger.info(
        f"Contour-first pipeline: {n_vehicles} vehicles, "
        f"{len(road_contours)} road contours")

    # Step 3: Per contour -> heading analysis -> lane groups
    contours: List[np.ndarray] = []
    group_headings: Dict[int, float] = {}
    gid = 0

    # Effective min vehicles: at least config value or 3% of total vehicles
    eff_min_vehicles = max(min_vehicles_per_group, int(n_vehicles * 0.03))

    # Minimum bounding box diagonal — a real lane should span a significant
    # portion of the frame, not be a compact blob.
    frame_diag = float(np.sqrt(fh ** 2 + fw ** 2))
    min_extent = frame_diag * min_extent_ratio

    # Helper: append a lane group with optional refinement and min-vehicles check
    def _append_group(cnt_arg, heading, veh_xs, veh_ys):
        nonlocal gid
        n_vehs = len(veh_xs)
        if n_vehs < eff_min_vehicles:
            logger.debug(
                f"Skipping group: only {n_vehs} vehicles "
                f"(min={eff_min_vehicles})")
            return False
        # Check spatial extent — reject compact blobs
        _, _, bw, bh = cv2.boundingRect(cnt_arg.astype(np.int32))
        extent = float(np.sqrt(bw ** 2 + bh ** 2))
        if extent < min_extent:
            logger.debug(
                f"Skipping group: extent {extent:.0f}px < {min_extent:.0f}px "
                f"(15% of frame diagonal)")
            return False
        refined_cnt = cnt_arg
        if refine_contours:
            positions = np.column_stack([veh_xs, veh_ys])
            result = _refine_contour_with_vehicles(
                cnt_arg, positions, (fh, fw),
                buffer_px=refine_buffer_px,
                min_vehicles=min_vehicles_per_group)
            if result is not None:
                refined_cnt = result
        contours.append(refined_cnt)
        group_headings[gid] = heading
        gid += 1
        return True

    # Pre-rasterize road contours into a label map for fast vehicle-to-contour lookup
    road_label_map = np.full((fh, fw), -1, dtype=np.int32)
    for ci_tmp in reversed(range(len(road_contours))):
        cv2.drawContours(road_label_map,
                         [road_contours[ci_tmp].astype(np.int32)], -1,
                         int(ci_tmp), -1)
    veh_ix = np.clip(np.round(mean_xs).astype(np.int32), 0, fw - 1)
    veh_iy = np.clip(np.round(mean_ys).astype(np.int32), 0, fh - 1)
    veh_road_labels = road_label_map[veh_iy, veh_ix]

    for ci, cnt in enumerate(road_contours):
        # Find vehicles with centroid inside this contour (vectorized)
        inside_mask = veh_road_labels == ci
        n_inside = int(inside_mask.sum())

        # Skip road contours with too few vehicles (side roads, parking lots).
        # Road contours need a stricter threshold than individual lane groups:
        # a real road should carry at least 5% of observed traffic.
        min_road_vehicles = max(eff_min_vehicles, int(n_vehicles * min_road_vehicle_pct))
        if n_inside < min_road_vehicles:
            logger.debug(
                f"Skipping road contour {ci}: {n_inside} vehicles "
                f"(min={eff_min_vehicles})")
            continue

        in_headings = headings[inside_mask]
        in_ids = track_ids[inside_mask]
        in_xs = mean_xs[inside_mask]
        in_ys = mean_ys[inside_mask]

        # Use global heading labels (preserves overpass/underpass separation)
        if n_inside < 10:
            # Too few for reliable split — single group
            mean_h = float(np.arctan2(
                np.sin(in_headings).mean(), np.cos(in_headings).mean()))
            _append_group(cnt, mean_h, in_xs, in_ys)
            continue

        dir_labels = global_dir_labels[inside_mask]
        unique_dirs = sorted(set(dir_labels.tolist()))

        # Compute per-direction mean heading and check group sizes
        dir_info = [] # (dir_label, heading, vehicle_mask, n_vehicles)
        min_dir_vehicles = max(3, n_inside // 20)
        for dlbl in unique_dirs:
            dmask = dir_labels == dlbl
            n_dir = int(dmask.sum())
            if n_dir < min_dir_vehicles:
                continue
            dh = float(np.arctan2(
                np.sin(in_headings[dmask]).mean(),
                np.cos(in_headings[dmask]).mean()))
            dir_info.append((dlbl, dh, dmask, n_dir))

        if len(dir_info) <= 1:
            # Single effective direction — keep parent contour
            mean_h = float(np.arctan2(
                np.sin(in_headings).mean(), np.cos(in_headings).mean()))
            if dir_info:
                mean_h = dir_info[0][1]
            _append_group(cnt, mean_h, in_xs, in_ys)
            continue

        # Check angular separation between all direction pairs
        dir_headings = np.array([di[1] for di in dir_info])
        max_sep = 0.0
        for i in range(len(dir_headings)):
            for j in range(i + 1, len(dir_headings)):
                sep = abs(dir_headings[i] - dir_headings[j])
                if sep > np.pi:
                    sep = 2 * np.pi - sep
                max_sep = max(max_sep, sep)

        if max_sep < np.pi * 4 / 9: # < 80 degrees
            # Insufficient angular separation — treat as unimodal
            mean_h = float(np.arctan2(
                np.sin(in_headings).mean(), np.cos(in_headings).mean()))
            _append_group(cnt, mean_h, in_xs, in_ys)
            continue

        # Multi-direction within same contour — argmax for non-overlapping sub-contours
        parent_mask_img = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(parent_mask_img, [cnt.astype(np.int32)], -1, 255, -1)

        # Build per-direction densities by filtering cached tracklets (fast)
        sub_densities = []  # (raw_density, heading)
        # Build ID set lookup for fast filtering
        id_set_cache = {}
        for dlbl, dh, dmask, n_dir in dir_info:
            d_ids = in_ids[dmask]
            id_set_cache[dlbl] = set(d_ids.tolist())

        for dlbl, dh, dmask, n_dir in dir_info:
            d_id_set = id_set_cache[dlbl]
            # Filter pre-built tracklets by track ID (no rebuild)
            id_mask = np.isin(all_tracklet_ids, list(d_id_set))
            sub_tracklets = all_tracklets[id_mask]
            if len(sub_tracklets) < 3:
                sub_densities.append((np.zeros((fh, fw), np.float32), dh))
                continue

            # Splat raw density (no log1p) masked to parent contour
            raw_density = _splat_raw_density(
                sub_tracklets, frame_shape,
                sigma_along=sigma_along, sigma_across=sigma_across,
                downsample=2)
            raw_density = np.where(
                parent_mask_img > 0, raw_density, 0).astype(np.float32)
            sub_densities.append((raw_density, dh))

        # Check we have at least 2 non-empty densities
        valid_count = sum(1 for d, _ in sub_densities if d.max() > 0)
        if valid_count < 2:
            # Fallback: use parent contour with dominant heading
            if valid_count == 1:
                dh = next(h for d, h in sub_densities if d.max() > 0)
            else:
                dh = float(np.arctan2(
                    np.sin(in_headings).mean(), np.cos(in_headings).mean()))
            _append_group(cnt, dh, in_xs, in_ys)
            continue

        # Stack and argmax — normalise each direction by its peak so that
        # low-traffic directions (e.g. crossing road at overpass) can compete
        # with high-traffic directions (highway) at their crossing point.
        stack = np.stack([d for d, _ in sub_densities], axis=0)  # (K, h, w)
        for k in range(stack.shape[0]):
            pk = stack[k].max()
            if pk > 0:
                stack[k] /= pk          # normalise to [0, 1]
        ownership = np.argmax(stack, axis=0)  # (h, w)
        all_zero = stack.max(axis=0) == 0
        ownership[all_zero] = -1

        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        n_added = 0
        for k, (raw_d, dh) in enumerate(sub_densities):
            territory = (ownership == k) & ~all_zero & (parent_mask_img > 0)
            if not territory.any():
                continue

            terr_u8 = (territory * 255).astype(np.uint8)

            # Large morph close bridges small gaps between same-direction lanes
            terr_u8 = cv2.morphologyEx(terr_u8, cv2.MORPH_CLOSE, close_k)
            # Clip back to argmax ownership — prevent leaking into other
            # directions' territory (overpass/underpass overlap fix)
            own_mask = ((ownership == k) * 255).astype(np.uint8)
            terr_u8 = cv2.bitwise_and(terr_u8, own_mask)

            # Emit each spatially disconnected CC as a separate lane group
            # (e.g. main highway + parallel ramp with same heading)
            labeled, n_cc = ndimage.label(terr_u8 > 0)
            if n_cc == 0:
                continue

            dlbl, _, dmask, n_dir = dir_info[k]
            dir_veh_xs = in_xs[dmask]
            dir_veh_ys = in_ys[dmask]

            for cc_id in range(1, n_cc + 1):
                cc_mask = (labeled == cc_id).astype(np.uint8) * 255

                # Smooth edges
                cc_mask = _smooth_binary_mask(cc_mask)

                found, _ = cv2.findContours(
                    cc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not found:
                    continue
                best = max(found, key=cv2.contourArea)
                if cv2.contourArea(best) < min_area:
                    continue
                eps = 0.002 * cv2.arcLength(best, True)
                best = cv2.approxPolyDP(best, eps, True).astype(np.float32)

                # Filter vehicles to those inside this CC's contour (vectorized)
                cc_label = np.full((fh, fw), 0, dtype=np.uint8)
                cv2.drawContours(cc_label, [best.astype(np.int32)], -1, 1, -1)
                d_ix = np.clip(np.round(dir_veh_xs).astype(np.int32), 0, fw - 1)
                d_iy = np.clip(np.round(dir_veh_ys).astype(np.int32), 0, fh - 1)
                cc_inside = cc_label[d_iy, d_ix] == 1
                if _append_group(best, dh, dir_veh_xs[cc_inside], dir_veh_ys[cc_inside]):
                    n_added += 1

        # Fallback: if argmax produced no valid sub-contours, keep parent
        if n_added == 0:
            mean_h = float(np.arctan2(
                np.sin(in_headings).mean(), np.cos(in_headings).mean()))
            _append_group(cnt, mean_h, in_xs, in_ys)

        logger.info(
            f"Contour {ci}: {n_inside} vehicles, "
            f"{len(dir_info)} directions, {n_added} sub-contours")

    # ---- Post-process: remove overlaps between lane group contours ----
    # Lane groups (especially at overpasses) must not cross/overlap.
    # For overlapping pixels, the larger contour keeps them.
    if len(contours) > 1:
        contours, group_headings = _remove_contour_overlaps(
            contours, group_headings, (fh, fw), min_area)

    logger.info(
        f"Lane group detection: {n_vehicles} vehicles -> "
        f"{len(road_contours)} road contours -> {len(contours)} lane groups")
    return contours, group_headings, track_stats


def _remove_contour_overlaps(
    contours: List[np.ndarray],
    group_headings: Dict[int, float],
    frame_shape: Tuple[int, int],
    min_area: float,
) -> Tuple[List[np.ndarray], Dict[int, float]]:
    """Remove overlapping regions between contours.

    For each overlapping pixel pair (i, j), the smaller contour loses
    the overlap.  Contours that become too small are dropped.
    Returns updated contours list and re-indexed group_headings.
    """
    fh, fw = frame_shape
    n = len(contours)
    masks = []
    for cnt in contours:
        m = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(m, [cnt.astype(np.int32)], -1, 255, -1)
        masks.append(m)

    # Subtract overlaps: smaller contour loses
    areas = [cv2.contourArea(c) for c in contours]
    for i in range(n):
        for j in range(i + 1, n):
            overlap = cv2.bitwise_and(masks[i], masks[j])
            if overlap.sum() == 0:
                continue
            if areas[i] <= areas[j]:
                masks[i] = cv2.bitwise_and(masks[i], cv2.bitwise_not(overlap))
            else:
                masks[j] = cv2.bitwise_and(masks[j], cv2.bitwise_not(overlap))

    # Re-extract contours from cleaned masks, preserving heading mapping
    result_contours: List[np.ndarray] = []
    result_headings: Dict[int, float] = {}
    new_gid = 0
    for i, m in enumerate(masks):
        if m.sum() == 0:
            continue
        found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            continue
        best = max(found, key=cv2.contourArea)
        if cv2.contourArea(best) < min_area:
            continue
        eps = 0.002 * cv2.arcLength(best, True)
        result_contours.append(
            cv2.approxPolyDP(best, eps, True).astype(np.float32))
        result_headings[new_gid] = group_headings.get(i, 0.0)
        new_gid += 1
    return result_contours, result_headings


# ---------------------------------------------------------------------------
# DBSCAN-based lane group detection (simpler, auto-adaptive)
# ---------------------------------------------------------------------------

def _detect_lane_groups_dbscan(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    min_track_points: int = 5,
    min_gap_deg: float = 45.0,
    min_vehicles_per_group: int = 3,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: int = 3,
    hull_buffer_px: float = 20.0,
    **kwargs,
) -> Tuple[List[np.ndarray], Dict[int, float], Optional[pl.DataFrame]]:
    """Detect lane groups via DBSCAN spatial clustering + direction split.

    Simpler alternative to density splatting. Auto-adapts to each camera's
    perspective by auto-tuning DBSCAN eps from the data distribution.

    Pipeline:
    1. Compute per-vehicle heading and position from trajectory displacements
    2. DBSCAN on vehicle centroids -> spatial clusters
    3. Per cluster: gap-based heading clustering -> direction groups
    4. Per (spatial, direction) group: convex hull + buffer -> contour

    Returns:
        lane_group_contours: list of OpenCV contour arrays (one per group)
        group_headings: {group_id: heading_rad}
        track_stats: pl.DataFrame from _compute_track_stats (or None)
    """
    from sklearn.cluster import DBSCAN as _DBSCAN
    from scipy.spatial import cKDTree

    fh, fw = frame_shape

    # Step 1: Per-vehicle heading + position stats
    track_stats = _compute_track_stats(trajectories, min_track_points)
    if len(track_stats) < 4:
        return [], {}, None

    mean_xs = track_stats["mean_x"].to_numpy()
    mean_ys = track_stats["mean_y"].to_numpy()
    mean_sins = track_stats["mean_sin"].to_numpy()
    mean_coss = track_stats["mean_cos"].to_numpy()
    headings = np.arctan2(mean_sins, mean_coss)
    positions = np.column_stack([mean_xs, mean_ys])
    n_vehicles = len(positions)

    # Step 2: Auto-tune DBSCAN eps via knee detection on k-NN distances
    if dbscan_eps is None:
        from sklearn.neighbors import NearestNeighbors
        k = min(20, n_vehicles - 1)
        if k < 2:
            return [], {}, None
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(positions)
        distances, _ = nn.kneighbors()

        # Sort k-th NN distances — the "knee" is the inter-road gap
        sorted_dists = np.sort(distances[:, -1]).astype(np.float64)
        n_pts = len(sorted_dists)
        x = np.arange(n_pts, dtype=np.float64)

        # Knee = point of max perpendicular distance from chord
        dx = x[-1] - x[0]
        dy = sorted_dists[-1] - sorted_dists[0]
        line_len = np.sqrt(dx ** 2 + dy ** 2)
        if line_len > 1e-8:
            perp_dist = np.abs(
                dy * x - dx * sorted_dists
                + x[-1] * sorted_dists[0] - sorted_dists[-1] * x[0]
            ) / line_len
            knee_idx = int(np.argmax(perp_dist))
            dbscan_eps = float(sorted_dists[knee_idx])
        else:
            dbscan_eps = float(sorted_dists[-1])

        # Bound to reasonable fraction of frame diagonal
        diag = np.sqrt(float(fh ** 2 + fw ** 2))
        dbscan_eps = max(dbscan_eps, diag * 0.04)  # floor ~59px for 720x1280
        dbscan_eps = min(dbscan_eps, diag * 0.20)  # cap ~294px
        logger.info(f"Auto-tuned DBSCAN eps={dbscan_eps:.1f} (knee detection, k={k})")

    # Step 3: Spatial clustering
    spatial_labels = _DBSCAN(
        eps=dbscan_eps, min_samples=dbscan_min_samples,
    ).fit_predict(positions)

    # Assign noise points to nearest non-noise cluster
    noise_mask = spatial_labels == -1
    if noise_mask.any() and not noise_mask.all():
        core_mask = ~noise_mask
        tree = cKDTree(positions[core_mask])
        _, nearest = tree.query(positions[noise_mask])
        spatial_labels[noise_mask] = spatial_labels[core_mask][nearest]

    unique_clusters = sorted(set(spatial_labels.tolist()))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    logger.info(
        f"DBSCAN: {n_vehicles} vehicles -> {len(unique_clusters)} spatial "
        f"clusters (eps={dbscan_eps:.1f})")

    # Step 4: Per cluster -> direction split -> lane groups
    contours: List[np.ndarray] = []
    group_headings: Dict[int, float] = {}
    gid = 0

    for clust_id in unique_clusters:
        mask = spatial_labels == clust_id
        n_cluster = int(mask.sum())
        if n_cluster < min_vehicles_per_group:
            logger.info(
                f"Skipping spatial cluster {clust_id}: only {n_cluster} "
                f"vehicles (min={min_vehicles_per_group})")
            continue

        cluster_headings = headings[mask]
        cluster_xs = mean_xs[mask]
        cluster_ys = mean_ys[mask]

        # Too few for reliable direction split — single group
        if n_cluster < 10:
            mean_h = float(np.arctan2(
                np.sin(cluster_headings).mean(),
                np.cos(cluster_headings).mean()))
            hull = _compute_hull_contour(
                cluster_xs, cluster_ys, (fh, fw), hull_buffer_px)
            if hull is not None:
                contours.append(hull)
                group_headings[gid] = mean_h
                gid += 1
            continue

        dir_labels = _cluster_headings(cluster_headings, min_gap_deg)
        unique_dirs = sorted(set(dir_labels.tolist()))

        # Build direction info
        dir_info = []
        for dlbl in unique_dirs:
            dmask = dir_labels == dlbl
            n_dir = int(dmask.sum())
            if n_dir < min_vehicles_per_group:
                continue
            dh = float(np.arctan2(
                np.sin(cluster_headings[dmask]).mean(),
                np.cos(cluster_headings[dmask]).mean()))
            dir_info.append((dlbl, dh, dmask, n_dir))

        if len(dir_info) <= 1:
            # Single effective direction
            mean_h = float(np.arctan2(
                np.sin(cluster_headings).mean(),
                np.cos(cluster_headings).mean()))
            if dir_info:
                mean_h = dir_info[0][1]
            hull = _compute_hull_contour(
                cluster_xs, cluster_ys, (fh, fw), hull_buffer_px)
            if hull is not None:
                contours.append(hull)
                group_headings[gid] = mean_h
                gid += 1
            continue

        # Check angular separation between direction pairs
        dir_headings_arr = np.array([di[1] for di in dir_info])
        max_sep = 0.0
        for i in range(len(dir_headings_arr)):
            for j in range(i + 1, len(dir_headings_arr)):
                sep = abs(dir_headings_arr[i] - dir_headings_arr[j])
                if sep > np.pi:
                    sep = 2 * np.pi - sep
                max_sep = max(max_sep, sep)

        if max_sep < np.pi * 4 / 9:  # < 80 degrees
            # Insufficient angular separation — treat as unimodal
            mean_h = float(np.arctan2(
                np.sin(cluster_headings).mean(),
                np.cos(cluster_headings).mean()))
            hull = _compute_hull_contour(
                cluster_xs, cluster_ys, (fh, fw), hull_buffer_px)
            if hull is not None:
                contours.append(hull)
                group_headings[gid] = mean_h
                gid += 1
            continue

        # Multi-direction: Voronoi-style partition (non-overlapping)
        # 1. Compute combined hull for entire spatial cluster
        combined_hull = _compute_hull_contour(
            cluster_xs, cluster_ys, (fh, fw), hull_buffer_px)
        if combined_hull is None:
            continue

        # 2. Rasterize combined hull as the territory to partition
        territory_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(
            territory_mask,
            [combined_hull.astype(np.int32)], -1, 255, -1)

        # 3. Build per-direction point masks and compute distance transforms
        n_dirs = len(dir_info)
        dist_maps = np.full((n_dirs, fh, fw), np.inf, dtype=np.float32)
        for di, (dlbl, dh, dmask, n_dir) in enumerate(dir_info):
            dir_xs = cluster_xs[dmask]
            dir_ys = cluster_ys[dmask]
            pt_mask = np.zeros((fh, fw), dtype=np.uint8)
            for x, y in zip(dir_xs, dir_ys):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < fw and 0 <= iy < fh:
                    cv2.circle(pt_mask, (ix, iy), 3, 255, -1)
            # Distance from every pixel to nearest vehicle of this direction
            dist_maps[di] = ndimage.distance_transform_edt(pt_mask == 0)

        # 4. Assign each territory pixel to nearest direction (argmin)
        ownership = np.argmin(dist_maps, axis=0)  # (fh, fw)

        # 5. Extract per-direction contours from territory
        for di, (dlbl, dh, dmask, n_dir) in enumerate(dir_info):
            dir_mask = np.zeros((fh, fw), dtype=np.uint8)
            dir_mask[(territory_mask > 0) & (ownership == di)] = 255

            # Morphological close to smooth jagged Voronoi edges
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dir_mask = cv2.morphologyEx(dir_mask, cv2.MORPH_CLOSE, kern)

            found, _ = cv2.findContours(
                dir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not found:
                continue

            best = max(found, key=cv2.contourArea)
            if cv2.contourArea(best) < 100:
                continue

            eps_approx = 0.002 * cv2.arcLength(best, True)
            cnt = cv2.approxPolyDP(best, eps_approx, True).astype(np.float32)
            contours.append(cnt)
            group_headings[gid] = dh
            gid += 1

    logger.info(
        f"Lane group detection (DBSCAN): {n_vehicles} vehicles -> "
        f"{len(unique_clusters)} spatial clusters -> {gid} lane groups")
    return contours, group_headings, track_stats


def _detect_lane_groups_learned(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    cache_dir: str = "",
    hull_buffer_px: float = 20.0,
    min_track_points: int = 5,
    min_vehicles_per_group: int = 3,
    **kwargs,
) -> Tuple[List[np.ndarray], Dict[int, float], Optional[pl.DataFrame]]:
    """Load cached learned lane groups and generate contours via convex hull.

    Requires lane_groups.npz to exist in cache_dir (produced by
    scripts/learn_lane_groups.py).
    """
    from pathlib import Path
    from src.data.lane_group_net import detect_lane_groups_learned

    cache_path = Path(cache_dir) if cache_dir else Path(".")
    contours, headings = detect_lane_groups_learned(
        trajectories, frame_shape, cache_path,
        hull_buffer_px=hull_buffer_px,
        min_track_points=min_track_points,
        min_vehicles_per_group=min_vehicles_per_group,
    )
    return contours, headings, None


# ---------------------------------------------------------------------------
# Community detection lane grouping (WP3)
# ---------------------------------------------------------------------------

def _build_vehicle_affinity_graph(
    positions: np.ndarray,
    headings: np.ndarray,
    scales: np.ndarray,
    lateral_weights: Dict[Tuple[int, int], float],
    vids: np.ndarray,
    edge_radius: float = 500.0,
    scale_sigma: float = 0.5,
):
    """Build a weighted vehicle affinity graph for community detection.

    Nodes = vehicles within a heading group.
    Edges connect K-nearest neighbors (not just radius, so long lanes stay
    connected).  Weight emphasises cross-track proximity (same-lane vehicles
    are close perpendicular to heading) while being lenient along-track.

    Args:
        positions: (N, 2) vehicle centroid positions.
        headings: (N,) heading angles in radians.
        scales: (N,) scale values (log1p(bbox_h) or mean_y).
        lateral_weights: Dict mapping (vid_i, vid_j) -> consistency float.
        vids: (N,) vehicle IDs (for lateral_weights lookup).
        edge_radius: Maximum distance for edge creation.
        scale_sigma: Scale difference penalty sigma.

    Returns:
        networkx.Graph with weighted edges.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    G = nx.Graph()
    n = len(positions)
    for i in range(n):
        G.add_node(i)

    if n < 2:
        return G

    # Dominant heading for along/cross decomposition
    dom_h = float(np.arctan2(np.sin(headings).mean(), np.cos(headings).mean()))
    along = np.array([np.cos(dom_h), np.sin(dom_h)], dtype=np.float64)
    perp = np.array([-np.sin(dom_h), np.cos(dom_h)], dtype=np.float64)

    # Cross-track positions (lane-discriminative axis)
    cross = positions @ perp

    # KDTree for spatial neighbours + K-NN to keep long lanes connected
    tree = cKDTree(positions)
    radius_pairs = tree.query_pairs(r=edge_radius, output_type="ndarray")

    k_nn = min(15, n - 1)
    _, knn_idx = tree.query(positions, k=k_nn + 1)  # includes self

    # Collect unique pairs from both radius and KNN
    pair_set = set()
    for p in range(len(radius_pairs)):
        i, j = int(radius_pairs[p, 0]), int(radius_pairs[p, 1])
        pair_set.add((min(i, j), max(i, j)))
    for i in range(n):
        for j_idx in knn_idx[i, 1:]:
            pair_set.add((min(i, int(j_idx)), max(i, int(j_idx))))

    # Normalise scale values for similarity
    scale_range = max(scales.max() - scales.min(), 1e-6)
    scales_norm = (scales - scales.min()) / scale_range

    for i, j in pair_set:
        # Cross-track distance (lane separation signal)
        d_cross = abs(cross[i] - cross[j])

        # Along-track distance (should NOT penalise heavily)
        d_along = abs((positions[i] - positions[j]) @ along)

        # Heading similarity
        cos_dh = max(0.0, float(np.cos(headings[i] - headings[j])))
        if cos_dh < 0.1:
            continue

        # Scale similarity
        d_scale = abs(scales_norm[i] - scales_norm[j])
        scale_sim = float(np.exp(-d_scale ** 2 / (2 * scale_sigma ** 2)))

        # Lateral ordering consistency
        vi, vj = int(vids[i]), int(vids[j])
        lat_key = (min(vi, vj), max(vi, vj))
        lat = lateral_weights.get(lat_key, 0.5)

        # Weight: penalise cross-track distance strongly (same lane = small
        # cross-track), penalise along-track distance weakly (same lane can
        # span the whole frame).
        # Typical lane width ~40-80px in perspective, so sigma_cross ~40px.
        sigma_cross = max(edge_radius * 0.08, 30.0)
        sigma_along = edge_radius  # very lenient
        spatial = float(np.exp(
            -d_cross ** 2 / (2 * sigma_cross ** 2)
            - d_along ** 2 / (2 * sigma_along ** 2)
        ))

        weight = cos_dh * lat * scale_sim * spatial
        if weight > 0.005:
            G.add_edge(i, j, weight=weight)

    return G


def _merge_small_communities(
    communities: list,
    positions: np.ndarray,
    min_size: int,
) -> list:
    """Merge communities smaller than min_size into nearest larger neighbour.

    Uses centroid distance between the small community and each large
    community to decide the merge target.
    """
    large = [c for c in communities if len(c) >= min_size]
    small = [c for c in communities if len(c) < min_size]

    if not large:
        # Everything is small — merge all into one
        merged = set()
        for c in communities:
            merged |= c
        return [merged] if merged else []

    if not small:
        return large

    # Compute centroids of large communities
    large_centroids = []
    for c in large:
        idx = sorted(c)
        large_centroids.append(positions[idx].mean(axis=0))
    large_centroids = np.array(large_centroids)

    for sc in small:
        sc_idx = sorted(sc)
        sc_centroid = positions[sc_idx].mean(axis=0)
        dists = np.linalg.norm(large_centroids - sc_centroid, axis=1)
        nearest = int(np.argmin(dists))
        large[nearest] = large[nearest] | sc

    return large


def _detect_lane_groups_community(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    min_track_points: int = 5,
    min_gap_deg: float = 45.0,
    min_vehicles_per_group: int = 3,
    hull_buffer_px: float = 20.0,
    community_resolution: float = 0.1,
    scale_sigma: float = 0.5,
    edge_radius: float = 0.0,
    **kwargs,
) -> Tuple[List[np.ndarray], Dict[int, float], Optional[pl.DataFrame]]:
    """Detect lane groups via Louvain community detection on vehicle affinity graph.

    Pipeline:
    1. Per-vehicle stats via _compute_track_stats() (with mean_scale).
    2. Cheap first pass: _cluster_headings() to separate opposing directions.
    3. Per heading group: build weighted affinity graph (networkx).
       - Edge weight emphasises cross-track proximity (same lane = close
         perpendicular to heading) while being lenient along-track.
       - K-NN edges guarantee long lanes stay connected.
    4. Louvain community detection with LOW resolution (few large groups).
    5. Merge tiny communities into nearest large neighbour.
    6. Per community: convex hull contour.

    Args:
        edge_radius: KDTree radius for edge creation.  When 0 (default),
            auto-set to 40% of frame diagonal so it spans the whole road.
        community_resolution: Louvain resolution. Lower = fewer, larger
            communities.  Default 0.1 targets 1-4 groups per heading direction.

    Returns:
        lane_group_contours: list of OpenCV contour arrays (one per group)
        group_headings: {group_id: heading_rad}
        track_stats: pl.DataFrame from _compute_track_stats (or None)
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    fh, fw = frame_shape

    # Auto-set edge radius from frame diagonal if not specified
    if edge_radius <= 0:
        diag = float(np.sqrt(fh ** 2 + fw ** 2))
        edge_radius = diag * 0.4
        logger.debug(f"Community: auto edge_radius={edge_radius:.0f} (40% of diag={diag:.0f})")

    # Step 1: Per-vehicle stats (heading, position, scale)
    track_stats = _compute_track_stats(trajectories, min_track_points)
    if len(track_stats) < 4:
        return [], {}, None

    stat_ids = track_stats["id"].to_numpy()
    mean_xs = track_stats["mean_x"].to_numpy()
    mean_ys = track_stats["mean_y"].to_numpy()
    mean_sins = track_stats["mean_sin"].to_numpy()
    mean_coss = track_stats["mean_cos"].to_numpy()
    mean_scales = track_stats["mean_scale"].to_numpy()
    headings = np.arctan2(mean_sins, mean_coss)
    positions = np.column_stack([mean_xs, mean_ys])
    n_vehicles = len(positions)

    # Step 2: Heading clustering to separate opposing directions
    if n_vehicles < 10:
        dir_labels = np.zeros(n_vehicles, dtype=int)
    else:
        dir_labels = _cluster_headings(headings, min_gap_deg)

    unique_dirs = sorted(set(dir_labels.tolist()))

    # Skip expensive lateral ordering for community detection — the affinity
    # graph already separates lanes well via cross-track distance + heading +
    # scale.  Lateral ordering is used in WP2 edge features, not here.
    lateral_weights: Dict[Tuple[int, int], float] = {}

    # Step 3+4: Per heading group -> affinity graph -> Louvain
    # Collect all communities globally, then Voronoi partition at the end
    all_community_xs: List[np.ndarray] = []    # per-community x positions
    all_community_ys: List[np.ndarray] = []    # per-community y positions
    all_community_headings: List[float] = []   # per-community mean heading

    # Minimum community size: at least 5% of direction group or min_vehicles
    min_comm = max(min_vehicles_per_group, 5)

    for dlbl in unique_dirs:
        dmask = dir_labels == dlbl
        n_dir = int(dmask.sum())
        if n_dir < min_vehicles_per_group:
            continue

        dir_positions = positions[dmask]
        dir_headings = headings[dmask]
        dir_scales = mean_scales[dmask]
        dir_ids = stat_ids[dmask]
        dir_xs = mean_xs[dmask]
        dir_ys = mean_ys[dmask]

        mean_h = float(np.arctan2(
            np.sin(dir_headings).mean(),
            np.cos(dir_headings).mean()))

        # Too few for community detection — single group
        if n_dir < 6:
            all_community_xs.append(dir_xs)
            all_community_ys.append(dir_ys)
            all_community_headings.append(mean_h)
            continue

        # Build affinity graph
        G = _build_vehicle_affinity_graph(
            dir_positions, dir_headings, dir_scales,
            lateral_weights, dir_ids,
            edge_radius=edge_radius, scale_sigma=scale_sigma,
        )

        # Handle disconnected graph: if too few edges, fall back to single group
        if G.number_of_edges() < 3:
            all_community_xs.append(dir_xs)
            all_community_ys.append(dir_ys)
            all_community_headings.append(mean_h)
            continue

        # Louvain community detection
        try:
            communities = louvain_communities(
                G, weight="weight", resolution=community_resolution, seed=42,
            )
        except Exception as e:
            logger.warning(f"Louvain failed: {e}, using single group")
            all_community_xs.append(dir_xs)
            all_community_ys.append(dir_ys)
            all_community_headings.append(mean_h)
            continue

        # Step 5: Merge tiny communities into nearest large neighbour
        communities = list(communities)
        min_comm_size = max(min_comm, int(n_dir * 0.05))
        communities = _merge_small_communities(
            communities, dir_positions, min_comm_size,
        )

        logger.debug(
            f"Direction {dlbl} ({n_dir} vehicles): "
            f"Louvain -> {len(communities)} communities after merge")

        # Collect valid communities
        for comm in communities:
            comm_indices = sorted(comm)
            if len(comm_indices) >= min_vehicles_per_group:
                all_community_xs.append(dir_xs[comm_indices])
                all_community_ys.append(dir_ys[comm_indices])
                comm_h = float(np.arctan2(
                    np.sin(dir_headings[comm_indices]).mean(),
                    np.cos(dir_headings[comm_indices]).mean()))
                all_community_headings.append(comm_h)

    # Step 6: Independent convex hull per community (like DBSCAN heuristic)
    # Each group is its own isolated corridor — no space-filling partition.
    contours: List[np.ndarray] = []
    group_headings: Dict[int, float] = {}
    gid = 0
    n_total = len(all_community_xs)

    for ci_idx in range(n_total):
        hull = _compute_hull_contour(
            all_community_xs[ci_idx], all_community_ys[ci_idx],
            (fh, fw), hull_buffer_px)
        if hull is not None:
            contours.append(hull)
            group_headings[gid] = all_community_headings[ci_idx]
            gid += 1

    logger.info(
        f"Lane group detection (community): {n_vehicles} vehicles -> "
        f"{len(unique_dirs)} heading groups -> {gid} lane groups")
    return contours, group_headings, track_stats


# ---------------------------------------------------------------------------
# Post-processing: fill uncovered reference contours
# ---------------------------------------------------------------------------

def fill_uncovered_contours(
    contours: List[np.ndarray],
    group_headings: Dict[int, float],
    reference_contours: List[np.ndarray],
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    min_track_points: int = 5,
    min_overlap_ratio: float = 0.3,
) -> Tuple[List[np.ndarray], Dict[int, float]]:
    """Promote uncovered reference contours (e.g. v1) to lane groups.

    For each reference contour, check if any existing lane group overlaps it
    significantly.  If not, create a new lane group from that contour with the
    mean heading of vehicles inside it.

    Args:
        contours: Existing lane group contours.
        group_headings: Existing {group_id: heading_rad}.
        reference_contours: External contours (e.g. from v1 detection).
        trajectories: Full trajectory DataFrame.
        frame_shape: (height, width).
        min_track_points: Minimum points per vehicle track.
        min_overlap_ratio: Minimum area overlap ratio to consider "covered".

    Returns:
        Updated (contours, group_headings) with new groups appended.
    """
    if not reference_contours:
        return contours, group_headings

    fh, fw = frame_shape
    contours = list(contours)
    group_headings = dict(group_headings)
    next_gid = max(group_headings.keys()) + 1 if group_headings else 0

    # Precompute union mask of all existing lane groups
    union_mask = np.zeros((fh, fw), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(union_mask, [cnt.astype(np.int32)], -1, 255, -1)

    # Per-vehicle stats for heading computation
    track_stats = _compute_track_stats(trajectories, min_track_points)
    if len(track_stats) > 0:
        stat_xs = track_stats["mean_x"].to_numpy()
        stat_ys = track_stats["mean_y"].to_numpy()
        stat_sins = track_stats["mean_sin"].to_numpy()
        stat_coss = track_stats["mean_cos"].to_numpy()
    else:
        stat_xs = stat_ys = stat_sins = stat_coss = np.array([])

    for ref_cnt in reference_contours:
        ref_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(ref_mask, [ref_cnt.astype(np.int32)], -1, 255, -1)
        ref_area = float(ref_mask.sum()) / 255.0
        if ref_area < 100:
            continue

        # Check overlap with union of all existing lane groups
        overlap = float(cv2.bitwise_and(ref_mask, union_mask).sum()) / 255.0
        covered = (overlap / ref_area) > min_overlap_ratio

        if covered:
            continue

        # Apply same size filters as density method
        _, _, bw, bh = cv2.boundingRect(ref_cnt.astype(np.int32))
        extent = float(np.sqrt(bw ** 2 + bh ** 2))
        frame_diag = float(np.sqrt(fh ** 2 + fw ** 2))
        if extent < frame_diag * 0.25:
            logger.debug(
                f"Skipping small uncovered v1 contour: extent {extent:.0f}px")
            continue

        # Check vehicle count inside
        n_inside = 0
        heading = 0.0
        if len(stat_xs) > 0:
            inside = np.array([
                cv2.pointPolygonTest(ref_cnt, (float(x), float(y)), False) >= 0
                for x, y in zip(stat_xs, stat_ys)
            ])
            n_inside = int(inside.sum())
            if n_inside > 0:
                heading = float(np.arctan2(
                    stat_sins[inside].mean(), stat_coss[inside].mean()))

        n_total = len(stat_xs)
        # Lower threshold for v1 contours (externally validated roads)
        min_vehs = max(min_track_points, int(n_total * 0.01))
        if n_inside < min_vehs:
            logger.debug(
                f"Skipping uncovered v1 contour: {n_inside} vehicles < {min_vehs}")
            continue

        contours.append(ref_cnt.astype(np.float32))
        group_headings[next_gid] = heading
        next_gid += 1
        logger.info(f"Promoted uncovered v1 contour to lane group G{next_gid - 1}")

    return contours, group_headings


# ---------------------------------------------------------------------------
# Public router + vehicle assignment
# ---------------------------------------------------------------------------

def detect_lane_groups(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    method: str = "dbscan",
    **kwargs,
) -> Tuple[List[np.ndarray], Dict[int, float], Optional[pl.DataFrame]]:
    """Detect lane groups via the specified method.

    Args:
        trajectories: DataFrame with 'id', 'x', 'y' columns.
        frame_shape: (height, width).
        method: "dbscan" (spatial clustering, recommended),
                "density" (oriented density splatting, legacy),
                "learned" (GNN-based, requires cached lane_groups.npz), or
                "community" (Louvain on vehicle affinity graph).
        **kwargs: Method-specific parameters forwarded to the implementation.

    Returns:
        lane_group_contours: list of OpenCV contour arrays
        group_headings: {group_id: heading_rad}
        track_stats: pl.DataFrame from _compute_track_stats (or None).
            Returned so that assign_vehicles_to_lane_groups() can reuse it
            without a redundant aggregation pass.
    """
    if method == "dbscan":
        return _detect_lane_groups_dbscan(trajectories, frame_shape, **kwargs)
    elif method == "density":
        return _detect_lane_groups_density(trajectories, frame_shape, **kwargs)
    elif method == "learned":
        return _detect_lane_groups_learned(trajectories, frame_shape, **kwargs)
    elif method == "community":
        return _detect_lane_groups_community(trajectories, frame_shape, **kwargs)
    else:
        raise ValueError(f"Unknown lane group method: {method!r}")


def assign_vehicles_to_lane_groups(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    method: str = "dbscan",
    min_track_points: int = 5,
    precomputed: Optional[Tuple[List[np.ndarray], Dict[int, float]]] = None,
    **kwargs,
) -> List[Tuple[pl.DataFrame, float, int]]:
    """Assign vehicles to lane groups.

    Returns the same shape as split_trajectories_by_direction():
    list of (filtered_traj, heading_rad, group_id) tuples.

    Vehicles that fall outside all lane group contours are dropped.

    Args:
        trajectories: DataFrame with 'id', 'x', 'y' columns.
        frame_shape: (height, width).
        method: "dbscan", "density", or "learned".
        min_track_points: Minimum points per track.
        precomputed: Optional (contours, group_headings) from a prior
            detect_lane_groups() call.  When supplied the expensive
            detection step is skipped entirely.
        **kwargs: Method-specific parameters forwarded to detect_lane_groups.
            For method="learned": must include cache_dir (str or Path).

    Returns:
        List of (sub_traj, heading_rad, group_id) tuples.
        Empty list if no lane groups detected.
    """
    # "learned" method bypasses contour-based assignment and uses cached labels directly
    if method == "learned" and precomputed is None:
        from pathlib import Path
        from src.data.lane_group_net import assign_vehicles_to_lane_groups_learned
        cache_dir = Path(kwargs.get("cache_dir", "."))
        return assign_vehicles_to_lane_groups_learned(
            trajectories, frame_shape, cache_dir,
            min_track_points=min_track_points,
            hull_buffer_px=kwargs.get("hull_buffer_px", 20.0),
            min_vehicles_per_group=kwargs.get("min_vehicles_per_group", 3),
        )

    track_stats = None
    if precomputed is not None:
        if len(precomputed) == 3:
            contours, group_headings, track_stats = precomputed
        else:
            contours, group_headings = precomputed
    else:
        contours, group_headings, track_stats = detect_lane_groups(
            trajectories, frame_shape,
            method=method,
            min_track_points=min_track_points,
            **kwargs,
        )

    if not contours:
        logger.warning("No lane groups detected, returning single group with all trajectories")
        return [(trajectories, 0.0, 0)]

    # Per-vehicle mean position — reuse track_stats from detect_lane_groups
    # to avoid a redundant Polars aggregation pass
    stats = track_stats if track_stats is not None else _compute_track_stats(trajectories, min_track_points)
    if len(stats) == 0:
        return [(trajectories, 0.0, 0)]

    vids = stats["id"].to_numpy()
    mean_xs = stats["mean_x"].to_numpy().astype(np.float32)
    mean_ys = stats["mean_y"].to_numpy().astype(np.float32)

    # Vectorized assignment: rasterize each contour into a label mask
    h, w = frame_shape
    label_map = np.full((h, w), -1, dtype=np.int32)
    # Draw later groups first so earlier (lower-id) groups win ties
    for gid in reversed(range(len(contours))):
        cv2.drawContours(label_map, [contours[gid].astype(np.int32)], -1, int(gid), -1)

    # Lookup each vehicle's group from the label map
    ix = np.clip(np.round(mean_xs).astype(np.int32), 0, w - 1)
    iy = np.clip(np.round(mean_ys).astype(np.int32), 0, h - 1)
    vehicle_group = label_map[iy, ix]

    # Build per-group trajectory subsets
    results: List[Tuple[pl.DataFrame, float, int]] = []
    for gid in range(len(contours)):
        mask = vehicle_group == gid
        if mask.sum() == 0:
            continue
        group_vids = vids[mask].tolist()
        group_traj = trajectories.filter(pl.col("id").is_in(group_vids))
        if len(group_traj) < 10:
            continue
        heading = group_headings.get(gid, 0.0)
        results.append((group_traj, heading, gid))

    n_dropped = int((vehicle_group == -1).sum())
    if n_dropped > 0:
        logger.info(f"Lane group assignment: {n_dropped}/{len(vids)} vehicles outside all contours (dropped)")
    logger.info(
        f"Lane group assignment: {len(vids)} vehicles -> "
        f"{len(results)} groups ({[len(r[0]) for r in results]} rows each)")

    if not results:
        logger.warning("All vehicles outside lane groups, returning single group")
        return [(trajectories, 0.0, 0)]

    return results


def _splat_raw_density(
    tracklets: np.ndarray,
    frame_shape: Tuple[int, int],
    sigma_along: float = 25.0,
    sigma_across: float = 6.0,
    n_angle_bins: int = 12,
    downsample: int = 1,
) -> np.ndarray:
    """Splat tracklets into a raw (pre-log) density field.

    Same as splat_oriented_density but returns the raw sum without log1p,
    needed for argmax ownership comparison across direction groups.
    """
    h, w = frame_shape
    ds = max(1, int(downsample))
    sh, sw = (h + ds - 1) // ds, (w + ds - 1) // ds
    density = np.zeros((sh, sw), dtype=np.float32)

    if len(tracklets) == 0:
        if ds > 1:
            return np.zeros((h, w), dtype=np.float32)
        return density

    scale_inv = 1.0 / ds
    coords_all = tracklets[:, :2] * scale_inv
    eff_sigma_along = sigma_along * scale_inv
    eff_sigma_across = sigma_across * scale_inv

    angles = np.arctan2(tracklets[:, 3], tracklets[:, 2]) % np.pi
    bin_edges = np.linspace(0, np.pi, n_angle_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(
        (angles / np.pi * n_angle_bins).astype(int),
        0, n_angle_bins - 1,
    )

    kernels = [_build_oriented_kernel(bc, eff_sigma_along, eff_sigma_across)
               for bc in bin_centers]

    for b in range(n_angle_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue

        point_map = np.zeros((sh, sw), dtype=np.float32)
        coords = np.round(coords_all[mask]).astype(np.int32)
        valid = (coords[:, 0] >= 0) & (coords[:, 0] < sw) & \
                (coords[:, 1] >= 0) & (coords[:, 1] < sh)
        coords = coords[valid]
        np.add.at(point_map, (coords[:, 1], coords[:, 0]), 1)

        filtered = cv2.filter2D(point_map, -1, kernels[b])
        density += filtered

    if ds > 1:
        density = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    return density
