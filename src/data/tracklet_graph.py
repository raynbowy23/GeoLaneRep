"""Tracklet graph construction: trajectories -> PyG Data with tracklet nodes."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from src.data.canonical import canonicalize_polyline, decanonicalize_polyline_batch

logger = logging.getLogger(__name__)


def _compute_lateral_ordering_weights(
    trajectories: pl.DataFrame,
    min_track_points: int = 5,
    min_overlap_frames: int = 5,
) -> Dict[Tuple[int, int], float]:
    """Compute pairwise lateral ordering consistency between vehicles.

    For each vehicle pair (A, B) with similar heading:
    1. Find time frames where both have observations.
    2. Project positions onto the perpendicular of the shared heading.
    3. consistency = |mean(sign(perp_A - perp_B))| in [0, 1].

    A value of 1.0 means stable ordering (same road, adjacent lanes).
    A value near 0.0 means randomly switching sides (overpass/underpass,
    crossing roads).

    Args:
        trajectories: DataFrame with 'id', 'x', 'y', 'time' columns.
        min_track_points: Minimum points per track.
        min_overlap_frames: Minimum co-visible frames to compute consistency.

    Returns:
        Dict mapping (vid_i, vid_j) with vid_i < vid_j to consistency float.
    """
    from scipy.spatial import cKDTree

    has_time = "time" in trajectories.columns
    if not has_time:
        return {}

    # Build per-vehicle arrays: heading + time-indexed (x, y)
    vehicle_data: Dict[int, Dict] = {}
    for (tid,), group in trajectories.group_by("id"):
        pts = group.select(["x", "y"]).to_numpy().astype(np.float32)
        if len(pts) < min_track_points:
            continue
        t = group.select("time").to_numpy().flatten()
        order = np.argsort(t)
        pts = pts[order]
        t = t[order]
        dx = pts[-1, 0] - pts[0, 0]
        dy = pts[-1, 1] - pts[0, 1]
        if dx * dx + dy * dy < 1e-6:
            continue
        heading = np.arctan2(dy, dx)
        vehicle_data[int(tid)] = {
            "heading": heading,
            "times": t,
            "xy": pts,
            "centroid": pts.mean(axis=0),
        }

    if len(vehicle_data) < 2:
        return {}

    vids = sorted(vehicle_data.keys())
    centroids = np.array([vehicle_data[v]["centroid"] for v in vids])

    # Use KDTree to limit pairs to spatially nearby vehicles
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=200.0, output_type="ndarray")

    weights: Dict[Tuple[int, int], float] = {}
    heading_thresh = np.cos(np.radians(45.0))  # similar heading

    for p in range(len(pairs)):
        i_idx, j_idx = int(pairs[p, 0]), int(pairs[p, 1])
        vi, vj = vids[i_idx], vids[j_idx]
        di, dj = vehicle_data[vi], vehicle_data[vj]

        # Check heading similarity
        cos_dh = np.cos(di["heading"] - dj["heading"])
        if cos_dh < heading_thresh:
            continue

        # Shared heading for perpendicular projection
        mean_h = np.arctan2(
            np.sin(di["heading"]) + np.sin(dj["heading"]),
            np.cos(di["heading"]) + np.cos(dj["heading"]),
        )
        perp = np.array([-np.sin(mean_h), np.cos(mean_h)], dtype=np.float32)

        # Find overlapping time frames via nearest-time matching
        ti, tj = di["times"], dj["times"]
        xi, xj = di["xy"], dj["xy"]

        # For each frame of i, find nearest frame of j within tolerance
        dt_tol = 0.5  # half-second tolerance
        signs = []
        j_ptr = 0
        for k in range(len(ti)):
            while j_ptr < len(tj) - 1 and tj[j_ptr + 1] <= ti[k]:
                j_ptr += 1
            # Check both j_ptr and j_ptr+1
            best_dt = abs(ti[k] - tj[j_ptr])
            best_j = j_ptr
            if j_ptr + 1 < len(tj) and abs(ti[k] - tj[j_ptr + 1]) < best_dt:
                best_dt = abs(ti[k] - tj[j_ptr + 1])
                best_j = j_ptr + 1
            if best_dt <= dt_tol:
                perp_i = xi[k] @ perp
                perp_j = xj[best_j] @ perp
                signs.append(np.sign(perp_i - perp_j))

        if len(signs) >= min_overlap_frames:
            consistency = float(abs(np.mean(signs)))
            key = (min(vi, vj), max(vi, vj))
            weights[key] = consistency

    logger.debug(
        f"Lateral ordering: {len(weights)} vehicle pairs with "
        f"consistency scores (from {len(vids)} vehicles)")
    return weights


def _resample_polyline(pts: np.ndarray, k: int) -> np.ndarray:
    """Resample a polyline to exactly k evenly-spaced points via linear interp.

    Args:
        pts: (M, 2) raw points (M >= 2).
        k: desired output length.

    Returns:
        (k, 2) resampled points.
    """
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1e-6:
        return np.tile(pts[0], (k, 1))
    targets = np.linspace(0.0, total, k)
    result = np.empty((k, 2), dtype=np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j < len(cum_len) - 2 and cum_len[j + 1] < t:
            j += 1
        seg_start = cum_len[j]
        seg_end = cum_len[j + 1] if j + 1 < len(cum_len) else cum_len[j]
        seg_range = seg_end - seg_start
        alpha = (t - seg_start) / seg_range if seg_range > 1e-8 else 0.0
        result[i] = pts[j] * (1.0 - alpha) + pts[min(j + 1, len(pts) - 1)] * alpha
    return result


def _compute_tangent(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """PCA first component -> unit tangent + curvature proxy.

    Returns:
        tangent: (2,) unit direction vector (NOT folded, keeps full 2pi).
        curvature: eigenvalue ratio lambda2/lambda1 (higher = more curved).
    """
    centered = pts - pts.mean(axis=0)
    if np.unique(centered, axis=0).shape[0] < 2:
        dx = pts[-1, 0] - pts[0, 0]
        dy = pts[-1, 1] - pts[0, 1]
        norm = max(np.sqrt(dx * dx + dy * dy), 1e-8)
        return np.array([dx / norm, dy / norm], dtype=np.float32), 0.0
    cov = centered.T @ centered / len(centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals sorted ascending: [lambda_small, lambda_large]
    pc = eigvecs[:, -1]
    # Orient tangent to match trajectory direction (start->end)
    direction = pts[-1] - pts[0]
    if np.dot(pc, direction) < 0:
        pc = -pc
    curvature = eigvals[0] / max(eigvals[1], 1e-8)
    return pc.astype(np.float32), float(curvature)


def _split_by_gaps(values: np.ndarray, gap_factor: float, min_size: int, min_gap_floor: float = 15.0) -> np.ndarray:
    """1D gap-based clustering: split sorted values at large gaps.

    Gaps larger than median_gap * gap_factor (or at least min_gap_floor) define cluster boundaries.
    Clusters smaller than min_size are labeled -1.
    """
    order = np.argsort(values)
    sorted_v = values[order]
    gaps = np.diff(sorted_v)

    if len(gaps) == 0:
        return np.zeros(len(values), dtype=np.int64)

    positive_gaps = gaps[gaps > 0]
    med_gap = np.median(positive_gaps) if len(positive_gaps) > 0 else 1.0
    thresh = max(med_gap * gap_factor, min_gap_floor)

    labels = np.zeros(len(values), dtype=np.int64)
    current = 0
    labels[order[0]] = 0
    for i in range(1, len(sorted_v)):
        if gaps[i - 1] > thresh:
            current += 1
        labels[order[i]] = current

    # Remove small clusters
    for lbl in range(current + 1):
        if (labels == lbl).sum() < min_size:
            labels[labels == lbl] = -1

    # Re-number valid labels contiguously
    valid = sorted(set(labels.tolist()) - {-1})
    mapping = {old: new for new, old in enumerate(valid)}
    mapping[-1] = -1
    return np.array([mapping[l] for l in labels], dtype=np.int64)


def _dbscan_kdtree(
    points: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN using cKDTree (avoids sklearn dependency).

    Args:
        points: (N, D) feature array.
        eps: neighborhood radius.
        min_samples: minimum neighbors for a core point.

    Returns:
        (N,) integer labels, -1 = noise.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, eps)
    N = len(points)
    labels = np.full(N, -1, dtype=np.int64)
    cluster_id = 0
    visited = np.zeros(N, dtype=bool)

    for i in range(N):
        if visited[i]:
            continue
        if len(neighbors[i]) < min_samples:
            continue
        # BFS expansion
        visited[i] = True
        labels[i] = cluster_id
        queue = list(neighbors[i])
        while queue:
            j = queue.pop(0)
            if visited[j]:
                continue
            visited[j] = True
            labels[j] = cluster_id
            if len(neighbors[j]) >= min_samples:
                queue.extend(neighbors[j])
        cluster_id += 1

    return labels


def _compute_pseudo_labels(
    centroids: np.ndarray,
    tangents: np.ndarray,
    track_ids: np.ndarray,
    gap_factor: float = 2.5,
    min_cluster: int = 5,
    lane_coord_labels: Optional[Dict[int, int]] = None,
    single_direction: bool = False,
    contour_ids: Optional[np.ndarray] = None,
    lane_width_hint: float = 15.0,
) -> np.ndarray:
    """Assign pseudo lane labels via per-vehicle cross-track clustering.

    If lane_coord_labels is provided (from LaneCoordNet preprocessing),
    uses those directly — each tracklet inherits its vehicle's label.
    Otherwise falls back to per-vehicle gap-based clustering.

    When contour_ids is provided, clustering runs independently per contour
    group so tracklets from spatially disjoint road segments (e.g. overpass
    vs highway) never merge into the same pseudo-lane.

    Args:
        centroids: (N, 2) tracklet pixel positions.
        tangents: (N, 2) unit tangent vectors.
        track_ids: (N,) integer vehicle IDs per tracklet.
        gap_factor: multiplier on median gap for lane boundary detection.
        min_cluster: minimum tracklets per pseudo-lane.
        lane_coord_labels: optional {vehicle_id: lane_label} from Phi.
        single_direction: if True, skip fwd/bwd split — all tracklets share one direction.
        contour_ids: optional (N,) int — contour index per tracklet, -1 = outside.
        lane_width_hint: approximate lane width in pixels — used as a
            reference scale for gap detection and DBSCAN radius.

    Returns:
        (N,) integer labels, -1 = unassigned.
    """
    N = len(centroids)

    # Use pre-computed lane coord labels if available
    if lane_coord_labels is not None:
        labels = np.full(N, -1, dtype=np.int64)
        for i in range(N):
            vid = int(track_ids[i])
            if vid in lane_coord_labels:
                labels[i] = lane_coord_labels[vid]
        n_assigned = (labels >= 0).sum()
        n_unique = len(set(labels[labels >= 0].tolist()))
        logger.info(f"Pseudo-GT from Phi: {n_assigned}/{N} assigned, {n_unique} lanes")
        return labels

    if N < 2 * min_cluster:
        return np.full(N, -1, dtype=np.int64)

    # Dominant heading (folded to remove 180° ambiguity)
    h = np.arctan2(tangents[:, 1], tangents[:, 0])
    dom = np.arctan2(np.sin(2 * h).mean(), np.cos(2 * h).mean()) / 2

    # Along-track and cross-track directions
    along = np.array([np.cos(dom), np.sin(dom)])
    perp = np.array([-np.sin(dom), np.cos(dom)])

    labels = np.full(N, -1, dtype=np.int64)
    next_label = 0

    if single_direction:
        direction_masks = [np.ones(N, dtype=bool)]
    else:
        fwd = np.cos(h - dom) >= 0
        direction_masks = [fwd, ~fwd]

    # Build contour group masks (outer loop around direction masks)
    if contour_ids is not None:
        unique_cids = sorted(set(contour_ids[contour_ids >= 0].tolist()))
        contour_masks = [contour_ids == cid for cid in unique_cids]
    else:
        contour_masks = [np.ones(N, dtype=bool)]

    for c_mask in contour_masks:
        for mask in direction_masks:
            combined = c_mask & mask
            idx = np.where(combined)[0]
            if len(idx) < min_cluster:
                continue

            sub_centroids = centroids[idx]
            cross_pos = sub_centroids @ perp
            sub_track_ids = track_ids[idx]

            # Per-vehicle median cross-track position AND 2D pixel position
            unique_vids = np.unique(sub_track_ids)
            vid_median = np.empty(len(unique_vids), dtype=np.float64)
            vid_positions = np.empty((len(unique_vids), 2), dtype=np.float64)
            vid_to_idx = {} # vehicle_id -> indices into idx array
            for vi, vid in enumerate(unique_vids):
                vmask = sub_track_ids == vid
                vid_median[vi] = np.median(cross_pos[vmask])
                vid_positions[vi] = np.median(sub_centroids[vmask], axis=0)
                vid_to_idx[vid] = np.where(vmask)[0]

            # Stage 1: 1D cross-track gap split (for parallel lanes)
            # Gap floor = fraction of lane width: parallel lanes separated by
            # ~1 lane width should be detected even when intra-lane noise is low.
            min_vehicles = max(2, min_cluster // 3)
            gap_floor = lane_width_hint * 0.5 if single_direction else lane_width_hint
            if len(unique_vids) >= min_vehicles:
                veh_labels = _split_by_gaps(vid_median, gap_factor, min_size=min_vehicles, min_gap_floor=gap_floor)
            else:
                veh_labels = np.zeros(len(unique_vids), dtype=np.int64)

            n_clusters = len(set(veh_labels.tolist()) - {-1})

            # Stage 2: 2D DBSCAN on per-vehicle median pixel positions
            # eps = 1.5 * lane_width so vehicles on the same lane merge but
            # vehicles on adjacent lanes (>1 lane width apart) stay separate.
            dbscan_eps = lane_width_hint * 1.5
            if n_clusters <= 1 and len(unique_vids) >= min_vehicles:
                veh_labels = _dbscan_kdtree(vid_positions, eps=dbscan_eps, min_samples=min_vehicles)
                n_clusters = len(set(veh_labels.tolist()) - {-1})

            if n_clusters > 1:
                for vi, vid in enumerate(unique_vids):
                    if veh_labels[vi] < 0:
                        continue
                    tracklet_indices = vid_to_idx[vid]
                    labels[idx[tracklet_indices]] = next_label + veh_labels[vi]
                next_label += n_clusters
            else:
                sub_labels = _dbscan_kdtree(sub_centroids, eps=dbscan_eps, min_samples=min_cluster)
                for sl in sorted(set(sub_labels.tolist()) - {-1}):
                    labels[idx[sub_labels == sl]] = next_label
                    next_label += 1

    return labels


def split_trajectories_by_direction(
    trajectories: pl.DataFrame,
    min_gap_deg: float = 45.0,
) -> List[Tuple[pl.DataFrame, float, int]]:
    """Split trajectories into direction groups by per-vehicle heading.

    Uses circular gap-based clustering on vehicle headings to detect
    2+ direction groups (handles highway + overpass/underpass crossings).

    Args:
        trajectories: DataFrame with columns [id, x, y] at minimum.
        min_gap_deg: minimum angular gap (degrees) to define a group boundary.

    Returns:
        List of (sub_traj, heading_rad, group_id) tuples.
    """
    has_time = "time" in trajectories.columns
    headings = []
    vids = []

    for (tid,), group in trajectories.group_by("id"):
        pts = group.select(["x", "y"]).to_numpy().astype(np.float32)
        if len(pts) < 2:
            continue
        if has_time:
            t = group.select("time").to_numpy().flatten()
            order = np.argsort(t)
            pts = pts[order]
        dx = pts[-1, 0] - pts[0, 0]
        dy = pts[-1, 1] - pts[0, 1]
        if dx * dx + dy * dy < 1e-6:
            continue
        headings.append(np.arctan2(dy, dx))
        vids.append(tid)

    if len(headings) < 4:
        logger.warning("Too few vehicles for direction split, returning single group")
        dom = np.arctan2(np.sin(2 * np.mean(headings)).item(), np.cos(2 * np.mean(headings)).item()) / 2 if headings else 0.0
        return [(trajectories, dom, 0)]

    headings = np.array(headings, dtype=np.float32)
    vids = np.array(vids)

    # Circular gap-based clustering on headings
    order = np.argsort(headings)
    sorted_h = headings[order]
    n = len(sorted_h)

    # Compute circular gaps (including wrap-around)
    gaps = np.empty(n, dtype=np.float32)
    gaps[:n - 1] = np.diff(sorted_h)
    gaps[n - 1] = (sorted_h[0] + 2 * np.pi) - sorted_h[-1]

    min_gap_rad = np.radians(min_gap_deg)
    big_gap_mask = gaps > min_gap_rad

    if big_gap_mask.sum() < 2:
        # Fall back to binary forward/backward (original behavior)
        dom = np.arctan2(np.sin(2 * headings).mean(), np.cos(2 * headings).mean()) / 2
        group_labels = np.where(np.cos(headings - dom) >= 0, 0, 1)
    else:
        # Multi-direction: split at big gaps
        # Start from right after the biggest gap and walk circularly
        biggest_gap_pos = int(np.argmax(gaps))
        group_labels_sorted = np.zeros(n, dtype=int)
        current_group = 0
        for step in range(n):
            idx = (biggest_gap_pos + 1 + step) % n
            group_labels_sorted[idx] = current_group
            if big_gap_mask[idx] and step < n - 1:
                current_group += 1
        # Map back to original order
        group_labels = np.empty(n, dtype=int)
        group_labels[order] = group_labels_sorted

    # Build direction groups, filter small groups
    unique_groups = sorted(set(group_labels.tolist()))
    min_vehicles = max(3, len(headings) // 50)
    groups = []
    gid_remap = 0
    for glbl in unique_groups:
        mask = group_labels == glbl
        g_vids = vids[mask]
        g_headings = headings[mask]
        if len(g_vids) < min_vehicles:
            continue
        sub = trajectories.filter(pl.col("id").is_in(g_vids.tolist()))
        if len(sub) < 10:
            continue
        mean_h = float(np.arctan2(np.sin(g_headings).mean(), np.cos(g_headings).mean()))
        groups.append((sub, mean_h, gid_remap))
        gid_remap += 1

    if not groups:
        dom = np.arctan2(np.sin(2 * headings).mean(), np.cos(2 * headings).mean()) / 2
        return [(trajectories, float(dom), 0)]

    logger.info(
        f"Direction split: {len(groups)} groups, "
        + ", ".join(f"g{g[2]}={len(g[0])} pts ({np.degrees(g[1]):.0f}°)" for g in groups)
    )
    return groups


def build_tracklet_graph(
    trajectories: pl.DataFrame,
    frame_shape: Tuple[int, int],
    config: Dict,
    lane_coord_labels: Optional[Dict[int, int]] = None,
    single_direction: bool = False,
    pixel_hom: Optional[np.ndarray] = None,
    ref_gps: Optional[np.ndarray] = None,
    skip_density_contours: bool = False,
    image_wh: Optional[Tuple[int, int]] = None,
) -> Optional[Data]:
    """Build a tracklet-level graph from trajectory data.

    Each node is a tracklet (short vehicle trajectory segment).
    Edges connect spatially and temporally proximal tracklets.

    When data.use_density_contours is true, oriented density splatting is
    performed on the constructed tracklet centroids+tangents to extract road
    boundary contours.  Pseudo-label clustering then runs independently per
    contour so spatially disjoint road segments never merge.

    Args:
        trajectories: DataFrame with columns [id, x, y, time] at minimum.
                      Optional: [bbox_h] for scale feature.
        frame_shape: (H, W) of the camera frame.
        config: dict from YAML with data.* keys.
        lane_coord_labels: optional {vehicle_id: lane_label} from LaneCoordNet.
        single_direction: if True, all tracklets share one heading direction;
                          skips fwd/bwd split in pseudo-GT and enforces tangent threshold on edges.

    Returns:
        PyG Data object, or None if too few tracklets.
    """
    data_cfg = config.get("data", config)
    tracklet_length = data_cfg.get("tracklet_length", 15)
    min_track_points = data_cfg.get("tracklet_min_points", 5)
    polyline_k = data_cfg.get("tracklet_polyline_k", 10)
    edge_radius = data_cfg.get("edge_radius", 80.0)
    edge_temporal_max = data_cfg.get("edge_temporal_max", 30.0)
    edge_tangent_thresh = data_cfg.get("edge_tangent_threshold", 0.5)
    max_neighbors = data_cfg.get("max_neighbors_per_node", 30)

    # Lateral ordering + scale penalty (WP2)
    use_lateral = data_cfg.get("use_lateral_ordering", False)
    lateral_min_overlap = data_cfg.get("lateral_min_overlap_frames", 5)
    scale_penalty_sigma = data_cfg.get("scale_penalty_sigma", 0.5)

    H, W = frame_shape

    # Step 0: Precompute lateral ordering weights if enabled
    lateral_weights: Dict[Tuple[int, int], float] = {}
    if use_lateral:
        lateral_weights = _compute_lateral_ordering_weights(
            trajectories,
            min_track_points=min_track_points,
            min_overlap_frames=lateral_min_overlap,
        )

    # Step 1: Build tracklets from trajectories
    tracklets_meta = [] # (centroid, tangent, curvature, speed, scale, track_id, polyline_raw, time_mid)
    has_time = "time" in trajectories.columns
    has_bbox = "bbox_h" in trajectories.columns

    for (tid,), group in trajectories.group_by("id"):
        cols = ["x", "y"]
        if has_time:
            cols.append("time")
        if has_bbox:
            cols.append("bbox_h")
        arr = group.select(cols).to_numpy().astype(np.float32)
        pts_xy = arr[:, :2]
        times = arr[:, 2] if has_time else np.arange(len(arr), dtype=np.float32)
        bbox_h = arr[:, 3] if has_bbox else np.full(len(arr), 20.0, dtype=np.float32)

        if len(pts_xy) < min_track_points:
            continue

        n_segs = max(1, len(pts_xy) // tracklet_length)
        for seg_idx in range(n_segs):
            start = seg_idx * tracklet_length
            end = min(start + tracklet_length, len(pts_xy))
            seg = pts_xy[start:end]
            seg_t = times[start:end]
            seg_bh = bbox_h[start:end]
            if len(seg) < 2:
                continue

            centroid = seg.mean(axis=0)
            tangent, curvature = _compute_tangent(seg)

            # Speed: total displacement / total time
            total_dist = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
            dt = seg_t[-1] - seg_t[0]
            speed = total_dist / max(dt, 1e-6)

            # Scale: log(bbox_h) - expected from y-position (linear depth model)
            mean_bh = seg_bh.mean()
            y_norm = centroid[1] / max(H, 1)
            expected_log_bh = np.log(max(mean_bh, 1.0)) # simplified: just log bbox
            scale = expected_log_bh

            time_mid = seg_t.mean()

            # Raw polyline for encoding
            poly_raw = _resample_polyline(seg, polyline_k) if len(seg) >= 2 else np.tile(seg[0], (polyline_k, 1))

            tracklets_meta.append({
                "centroid": centroid,
                "tangent": tangent,
                "curvature": curvature,
                "speed": speed,
                "scale": scale,
                "track_id": int(tid) if isinstance(tid, (int, float, np.integer)) else hash(tid) % (2**31),
                "polyline_raw": poly_raw,
                "time_mid": time_mid,
            })

    if len(tracklets_meta) < 3:
        logger.warning(f"Too few tracklets ({len(tracklets_meta)}), skipping graph.")
        return None

    N = len(tracklets_meta)

    # Step 2: Canonicalize polylines into tangent-aligned local frames
    polylines_local = np.empty((N, polyline_k, 2), dtype=np.float32)
    centroids = np.empty((N, 2), dtype=np.float32)
    tangents = np.empty((N, 2), dtype=np.float32)
    curvatures = np.empty(N, dtype=np.float32)
    speeds = np.empty(N, dtype=np.float32)
    scales = np.empty(N, dtype=np.float32)
    track_ids = np.empty(N, dtype=np.int64)
    times_mid = np.empty(N, dtype=np.float32)

    for i, t in enumerate(tracklets_meta):
        centroids[i] = t["centroid"]
        tangents[i] = t["tangent"]
        curvatures[i] = t["curvature"]
        speeds[i] = t["speed"]
        scales[i] = t["scale"]
        track_ids[i] = t["track_id"]
        times_mid[i] = t["time_mid"]
        polylines_local[i] = canonicalize_polyline(t["polyline_raw"], t["centroid"], t["tangent"])

    # Save pixel-space copies for visualization and density splatting
    pixel_centroids = centroids.copy()
    pixel_tangents = tangents.copy()

    # Normalize to [0,1] pixel-space: centroids / [W, H]
    data_cfg = config.get("data", config)
    if image_wh is not None:
        norm_W, norm_H = float(image_wh[0]), float(image_wh[1])
    else:
        norm_W = float(data_cfg.get("image_width", W))
        norm_H = float(data_cfg.get("image_height", H))
    wh = np.array([norm_W, norm_H], dtype=np.float32)

    centroids = centroids / wh  # (N, 2) in [0, 1]

    # Re-compute tangents from normalized centroids (unit vectors stay the same)
    # Polylines: decanonicalize from pixel, normalize, re-canonicalize
    for i in range(N):
        global_px = decanonicalize_polyline_batch(
            torch.from_numpy(polylines_local[i:i+1]),
            torch.from_numpy(pixel_centroids[i:i+1]),
            torch.from_numpy(pixel_tangents[i:i+1]),
        ).numpy()[0]  # (K, 2) in pixels
        global_norm = global_px / wh  # (K, 2) in [0, 1]
        tangents[i], _ = _compute_tangent(global_norm)
        polylines_local[i] = canonicalize_polyline(global_norm, centroids[i], tangents[i])

    # Normalize speeds: pixel displacement per frame → normalized displacement
    speeds = speeds / np.maximum(np.linalg.norm(wh), 1e-6)

    # Step 3: Build node features (excluding polyline — that's separate)
    # Features: [curvature, speed_norm, scale, tangent_cos, tangent_sin, cross_track_norm]
    # cross_track_norm: position perpendicular to dominant heading (lane-discriminative, no depth bias)
    speed_max = max(speeds.max(), 1e-6)

    # Compute cross-track coordinate relative to dominant heading
    h = np.arctan2(tangents[:, 1], tangents[:, 0])
    dom = np.arctan2(np.sin(2 * h).mean(), np.cos(2 * h).mean()) / 2
    perp = np.array([-np.sin(dom), np.cos(dom)])
    cross_track = (centroids - centroids.mean(axis=0)) @ perp
    ct_range = max(np.abs(cross_track).max(), 1e-6)
    cross_track_norm = cross_track / ct_range # [-1, 1]

    node_features = np.stack([
        curvatures,
        speeds / speed_max,
        scales / max(scales.max(), 1e-6),
        tangents[:, 0],
        tangents[:, 1],
        cross_track_norm,
    ], axis=1).astype(np.float32) # (N, 6)

    # Step 4: Build edges using KDTree for O(N log N) spatial lookup
    from scipy.spatial import cKDTree

    src_list, dst_list = [], []
    edge_feat_list = []

    tree = cKDTree(centroids)

    # k-NN: query k nearest neighbors per node (capped by edge_radius)
    k_query = min(max_neighbors + 1, N)  # +1 because query includes self
    nn_dists, nn_indices = tree.query(centroids, k=k_query)  # (N, k)

    # Build unique (i, j) pairs from k-NN results (i < j to avoid duplicates)
    pair_set = set()
    for i in range(N):
        for ki in range(1, k_query):  # skip self (index 0)
            j = int(nn_indices[i, ki])
            if nn_dists[i, ki] <= edge_radius:
                pair_set.add((min(i, j), max(i, j)))
    pairs = np.array(sorted(pair_set), dtype=np.int64) if pair_set else np.empty((0, 2), dtype=np.int64)

    # Precompute per-node rotation for local frame transform
    thetas = np.arctan2(tangents[:, 1], tangents[:, 0]) # (N,)
    cos_neg = np.cos(-thetas)
    sin_neg = np.sin(-thetas)

    # Edge feature dimensionality: 7 base + 2 optional (lateral + scale_penalty)
    edge_dim = 9 if use_lateral else 7

    for p in range(len(pairs)):
        i, j = int(pairs[p, 0]), int(pairs[p, 1])

        # Temporal proximity
        dt = abs(times_mid[j] - times_mid[i])
        if dt > edge_temporal_max:
            continue

        dx = centroids[j, 0] - centroids[i, 0]
        dy = centroids[j, 1] - centroids[i, 1]

        # Relative heading
        cos_dtheta = tangents[i, 0] * tangents[j, 0] + tangents[i, 1] * tangents[j, 1]
        sin_dtheta = tangents[i, 0] * tangents[j, 1] - tangents[i, 1] * tangents[j, 0]

        # Filter cross-direction edges (cos_dtheta < threshold means opposing headings)
        if cos_dtheta < edge_tangent_thresh:
            continue

        same_track = 1.0 if track_ids[i] == track_ids[j] else 0.0
        d_speed = (speeds[j] - speeds[i]) / max(speed_max, 1e-6)
        d_scale = scales[j] - scales[i]

        # Lateral ordering consistency + scale penalty (WP2)
        if use_lateral:
            vid_i, vid_j = int(track_ids[i]), int(track_ids[j])
            lat_key = (min(vid_i, vid_j), max(vid_i, vid_j))
            lat_consistency = lateral_weights.get(lat_key, 0.5)
            sp = float(np.exp(-d_scale ** 2 / (2 * scale_penalty_sigma ** 2)))
        else:
            lat_consistency = 0.0  # unused
            sp = 0.0  # unused

        # Forward edge i->j: relative position in local frame of node i
        dx_local_i = cos_neg[i] * dx - sin_neg[i] * dy
        dy_local_i = sin_neg[i] * dx + cos_neg[i] * dy

        # Normalize by edge_radius (now in [0,1] normalized space)
        norm_ij = max(edge_radius, 0.001)

        base_ij = [
            dx_local_i / norm_ij,
            dy_local_i / norm_ij,
            cos_dtheta, sin_dtheta, d_speed, d_scale, same_track,
        ]
        if use_lateral:
            base_ij.extend([lat_consistency, sp])
        feat_ij = np.array(base_ij, dtype=np.float32)

        # Reverse edge j->i: relative position in local frame of node j
        dx_local_j = cos_neg[j] * (-dx) - sin_neg[j] * (-dy)
        dy_local_j = sin_neg[j] * (-dx) + cos_neg[j] * (-dy)
        base_ji = [
            dx_local_j / norm_ij,
            dy_local_j / norm_ij,
            cos_dtheta, sin_dtheta, -d_speed, -d_scale, same_track,
        ]
        if use_lateral:
            base_ji.extend([lat_consistency, sp])
        feat_ji = np.array(base_ji, dtype=np.float32)

        src_list.extend([i, j])
        dst_list.extend([j, i])
        edge_feat_list.extend([feat_ij, feat_ji])

    if len(src_list) == 0:
        # Fallback: connect 3-nearest neighbors
        logger.warning("No edges found, connecting 3-nearest neighbors.")
        _, nn_idx = tree.query(centroids, k=min(4, N))
        for i in range(N):
            for j_idx in nn_idx[i, 1:]:
                dx = centroids[j_idx, 0] - centroids[i, 0]
                dy = centroids[j_idx, 1] - centroids[i, 1]
                cos_dtheta = tangents[i, 0] * tangents[j_idx, 0] + tangents[i, 1] * tangents[j_idx, 1]
                sin_dtheta = tangents[i, 0] * tangents[j_idx, 1] - tangents[i, 1] * tangents[j_idx, 0]
                dx_local = cos_neg[i] * dx - sin_neg[i] * dy
                dy_local = sin_neg[i] * dx + cos_neg[i] * dy
                same_track = 1.0 if track_ids[i] == track_ids[j_idx] else 0.0
                d_speed = (speeds[j_idx] - speeds[i]) / max(speed_max, 1e-6)
                d_scale = scales[j_idx] - scales[i]
                base_feat = [
                    dx_local / max(edge_radius, 0.001), dy_local / max(edge_radius, 0.001),
                    cos_dtheta, sin_dtheta, d_speed, d_scale, same_track,
                ]
                if use_lateral:
                    vid_i, vid_j = int(track_ids[i]), int(track_ids[j_idx])
                    lat_key = (min(vid_i, vid_j), max(vid_i, vid_j))
                    base_feat.extend([
                        lateral_weights.get(lat_key, 0.5),
                        float(np.exp(-d_scale ** 2 / (2 * scale_penalty_sigma ** 2))),
                    ])
                feat = np.array(base_feat, dtype=np.float32)
                src_list.append(i)
                dst_list.append(j_idx)
                edge_feat_list.append(feat)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) # (2, E)
    edge_attr = torch.from_numpy(np.stack(edge_feat_list)) if edge_feat_list else torch.empty((0, edge_dim), dtype=torch.float32)

    # Step 5: Compute next-displacement targets for predictive loss
    # For each tracklet, find the next tracklet from the same vehicle
    next_displacement = np.full((N, 2), np.nan, dtype=np.float32)
    has_next = np.zeros(N, dtype=bool)

    # Group tracklets by track_id and sort by time
    tid_to_indices = {}
    for i in range(N):
        tid = track_ids[i]
        if tid not in tid_to_indices:
            tid_to_indices[tid] = []
        tid_to_indices[tid].append(i)

    for tid, indices in tid_to_indices.items():
        if len(indices) < 2:
            continue
        indices_sorted = sorted(indices, key=lambda i: times_mid[i])
        for k_idx in range(len(indices_sorted) - 1):
            curr = indices_sorted[k_idx]
            nxt = indices_sorted[k_idx + 1]
            # Displacement in local frame of current tracklet
            dx = centroids[nxt, 0] - centroids[curr, 0]
            dy = centroids[nxt, 1] - centroids[curr, 1]
            theta = np.arctan2(tangents[curr, 1], tangents[curr, 0])
            cos_t, sin_t = np.cos(-theta), np.sin(-theta)
            next_displacement[curr, 0] = cos_t * dx - sin_t * dy
            next_displacement[curr, 1] = sin_t * dx + cos_t * dy
            has_next[curr] = True

    # Step 5b: Density-based road contours from the tracklets themselves
    # NOTE: density splatting needs pixel-space rasters, so use pixel_centroids
    contour_ids = None
    road_contours = None
    use_density_contours = data_cfg.get("use_density_contours", False)
    if use_density_contours and not skip_density_contours:
        from src.data.density_contours import (
            splat_oriented_density,
            extract_road_contours,
            assign_tracklets_to_contours,
        )
        # Build (N,4) [cx, cy, cos θ, sin θ] from pixel-space arrays
        h_angles = np.arctan2(pixel_tangents[:, 1], pixel_tangents[:, 0]) % np.pi
        density_tracklets = np.column_stack([
            pixel_centroids, np.cos(h_angles), np.sin(h_angles),
        ]).astype(np.float32)
        density = splat_oriented_density(density_tracklets, frame_shape)
        road_contours = extract_road_contours(density)
        if road_contours:
            contour_ids = assign_tracklets_to_contours(pixel_centroids, road_contours, frame_shape)
            n_outside = int((contour_ids < 0).sum())
            logger.debug(
                f"Density contours: {len(road_contours)} contours, "
                f"{N - n_outside}/{N} tracklets assigned"
            )

    # Step 6: Compute pseudo lane labels for symmetry-breaking
    gap_factor = data_cfg.get("pseudo_gt_min_gap_factor", 2.5)
    lane_width_hint = data_cfg.get("fixed_lane_width", data_cfg.get("fixed_lane_width_m", 3.5))
    pseudo_labels = _compute_pseudo_labels(
        centroids, tangents, track_ids,
        gap_factor=gap_factor,
        lane_coord_labels=lane_coord_labels,
        single_direction=single_direction,
        contour_ids=contour_ids,
        lane_width_hint=lane_width_hint,
    )

    n_pseudo = len(set(pseudo_labels[pseudo_labels >= 0].tolist()))
    n_assigned = int((pseudo_labels >= 0).sum())
    n_edges = len(src_list)
    logger.debug(
        f"Graph: {N} nodes, {n_edges} edges, "
        f"pseudo_gt: {n_pseudo} labels ({n_assigned}/{N} assigned), single_dir={single_direction}"
    )

    # Pre-compute global point positions (for point-level slot attention)
    polylines_global = decanonicalize_polyline_batch(
        torch.from_numpy(polylines_local),
        torch.from_numpy(centroids),
        torch.from_numpy(tangents),
    ).numpy()  # (N, K, 2)

    data = Data(
        x=torch.from_numpy(node_features), # (N, 6)
        polylines=torch.from_numpy(polylines_local), # (N, K, 2)
        edge_index=edge_index, # (2, E)
        edge_attr=edge_attr, # (E, 7 or 9)
        centroids=torch.from_numpy(centroids), # (N, 2) — normalized [0,1] pixel-space
        tangents=torch.from_numpy(tangents), # (N, 2) — unit vectors in normalized space
        global_points=torch.from_numpy(polylines_global.reshape(-1, 2).astype(np.float32)),  # (N*K, 2)
        pixel_centroids=torch.from_numpy(pixel_centroids),  # (N, 2) always pixel-space
        pixel_tangents=torch.from_numpy(pixel_tangents),     # (N, 2) always pixel-space
        track_ids=torch.from_numpy(track_ids), # (N,)
        next_displacement=torch.from_numpy(next_displacement), # (N, 2)
        has_next_mask=torch.from_numpy(has_next), # (N,)
        pseudo_labels=torch.from_numpy(pseudo_labels), # (N,)
        contour_ids=torch.from_numpy(contour_ids) if contour_ids is not None else None,
        road_contours=road_contours, # List[ndarray] or None (not a tensor, for viz only)
        num_nodes=N,
    )
    # Store image dimensions for denormalization (viz)
    data.image_wh = torch.tensor([norm_W, norm_H], dtype=torch.float32)
    return data
