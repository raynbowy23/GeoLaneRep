"""Zero-shot lane detection from contrastive encoder.

Given a new camera with only trajectory.csv (no annotation), discovers
pseudo-lanes from trajectory behavior and predicts lane properties by
matching to a reference bank of annotated camera embeddings.

Pipeline:
    trajectory.csv (new camera)
        → density contour detection → lane groups + headings
        → per group: lateral clustering → pseudo-lanes
        → encode trajectory-only (geometry=zeros)
        → cosine match to reference bank
        → output: predicted lateral_rank, edge flags, lane_count
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from src.data.density_contours import (
    _compute_track_stats,
    _detect_lane_groups_density,
    assign_tracklets_to_contours,
)
from src.data.lane_dataset import LaneDataset, _compute_traj_stats, point_to_polyline_dist
from src.models.lane_encoder import LaneEncoder

logger = logging.getLogger(__name__)


@dataclass
class PseudoLane:
    """A lane discovered purely from trajectory behavior."""

    group_id: int
    lane_idx: int              # lateral index within group (0 = leftmost)
    lane_key: str              # "{camera}_{group_id}_{lane_idx}"
    heading_rad: float         # group heading
    trajectories: List[np.ndarray]  # list of (T_i, 2) normalized
    traj_stats: np.ndarray     # (4,) from _compute_traj_stats
    lateral_center: float      # perpendicular offset (for ordering)
    n_lanes_in_group: int


def build_lanes_from_annotation(
    annotation: dict,
    traj_df: pl.DataFrame,
    frame_shape: Tuple[int, int],
    camera: str,
    config: Optional[dict] = None,
) -> List[PseudoLane]:
    """Build pseudo-lanes using annotation geometry for lane boundaries.

    Uses annotation waypoints to define where lanes are (trajectory assignment),
    but computes all features geometrically — no annotation labels are leaked.
    The encoder will still run with drop_geometry=True (trajectory-only).

    This separates lane *discovery* (annotation-assisted) from lane
    *representation* (encoder zero-shot), enabling clean evaluation.

    Args:
        annotation: Parsed annotation dict from load_annotation_json().
        traj_df: Trajectory DataFrame with 'id', 'x', 'y' columns.
        frame_shape: (height, width) of camera frame.
        camera: Camera name for lane_key generation.
        config: Optional config dict.

    Returns:
        List of PseudoLane objects with trajectories assigned via annotation
        geometry but roles computed geometrically (has_successor=0).
    """
    from src.data.annotation_loader import get_group_lanes

    fh, fw = frame_shape
    image_wh = np.array([fw, fh], dtype=np.float64)
    cfg = config or {}
    assign_cfg = cfg.get("assignment", {})
    lateral_threshold_px = assign_cfg.get("lateral_threshold_px", 60.0)
    min_tracklet_points = assign_cfg.get("min_tracklet_points", 5)

    pseudo_lanes: List[PseudoLane] = []

    for lg in annotation.get("lane_groups", []):
        gid = lg["group_id"]
        lanes = get_group_lanes(annotation, gid, image_wh=(fw, fh))
        if not lanes:
            continue

        # Assign trajectories to lanes using annotation waypoints
        # (same logic as LaneDataset._assign_trajectories)
        lane_trajs: Dict[int, List[np.ndarray]] = {l["cls_id"]: [] for l in lanes}
        id_col = "id" if "id" in traj_df.columns else "track_id"
        thresh = lateral_threshold_px / fw

        for track_id, group in traj_df.group_by(id_col):
            pts = np.column_stack([
                group["x"].to_numpy(),
                group["y"].to_numpy(),
            ]).astype(np.float64) / image_wh

            if len(pts) < min_tracklet_points:
                continue

            best_cls = None
            best_dist = float("inf")
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
                lane_trajs[best_cls].append(pts)

        # Compute max traj count across lanes in this group
        max_traj_count = max((len(t) for t in lane_trajs.values()), default=1)

        # Compute group heading from lane tangents
        tangents = []
        for lane in lanes:
            wp = lane["waypoints"]
            if len(wp) >= 2:
                t = wp[-1] - wp[0]
                norm = np.linalg.norm(t)
                if norm > 1e-8:
                    tangents.append(t / norm)

        if not tangents:
            continue

        mean_tangent = np.mean(tangents, axis=0)
        mean_tangent /= np.linalg.norm(mean_tangent) + 1e-8
        heading_rad = float(np.arctan2(mean_tangent[1], mean_tangent[0]))

        # Fold heading for perpendicular axis (consistent with training)
        if mean_tangent[1] < 0 or (mean_tangent[1] == 0 and mean_tangent[0] < 0):
            mean_tangent = -mean_tangent
        perp = np.array([-mean_tangent[1], mean_tangent[0]])

        # Build pseudo-lanes sorted by lateral position
        lane_entries = []
        for lane in lanes:
            wp = lane["waypoints"]
            cls_id = lane["cls_id"]
            trajs = lane_trajs.get(cls_id, [])
            if not trajs:
                continue

            centroid = wp.mean(axis=0)
            lateral_center = float(np.dot(centroid, perp))
            stats = _compute_traj_stats(trajs, wp, max_traj_count)
            lane_entries.append((lateral_center, cls_id, trajs, stats, wp))

        lane_entries.sort(key=lambda x: x[0])
        n_lanes = len(lane_entries)

        for lane_idx, (lat_center, cls_id, trajs, stats, wp) in enumerate(lane_entries):
            lane_key = f"{camera}_{gid}_{lane_idx}"
            pseudo_lanes.append(PseudoLane(
                group_id=gid,
                lane_idx=lane_idx,
                lane_key=lane_key,
                heading_rad=heading_rad,
                trajectories=trajs,
                traj_stats=stats,
                lateral_center=lat_center,
                n_lanes_in_group=n_lanes,
            ))

    logger.info(
        f"Built {len(pseudo_lanes)} lanes from annotation "
        f"({len(annotation.get('lane_groups', []))} groups)"
    )
    return pseudo_lanes


def discover_lanes(
    traj_df: pl.DataFrame,
    frame_shape: Tuple[int, int],
    camera: str,
    config: Optional[dict] = None,
) -> List[PseudoLane]:
    """Discover lanes from trajectory data alone (no annotation).

    1. Detect lane groups via density contours
    2. Per group: lateral clustering of trajectories → pseudo-lanes
    3. Return list of PseudoLane dataclasses

    Args:
        traj_df: DataFrame with 'id', 'x', 'y' columns (pixel coordinates).
        frame_shape: (height, width) of the camera frame.
        camera: Camera name (for lane_key generation).
        config: Optional config dict for density contour parameters.

    Returns:
        List of discovered PseudoLane objects.
    """
    fh, fw = frame_shape
    image_wh = np.array([fw, fh], dtype=np.float64)

    cfg = config or {}
    density_cfg = cfg.get("density_contours", {})

    # Step 1: Detect lane groups
    # Use lower thresholds for zero-shot: smaller groups and shorter extents
    # are valid lane groups that would otherwise be filtered.
    contours, group_headings, track_stats = _detect_lane_groups_density(
        traj_df, frame_shape,
        min_track_points=density_cfg.get("min_track_points", 5),
        min_gap_deg=density_cfg.get("min_gap_deg", 45.0),
        min_vehicles_per_group=density_cfg.get("min_vehicles_per_group", 3),
        min_extent_ratio=density_cfg.get("min_extent_ratio", 0.15),
        min_road_vehicle_pct=density_cfg.get("min_road_vehicle_pct", 0.02),
    )

    if not contours or track_stats is None or len(track_stats) == 0:
        logger.warning("No lane groups detected")
        return []

    logger.info(f"Detected {len(contours)} lane groups")

    # Get per-vehicle stats
    veh_ids = track_stats["id"].to_numpy()
    mean_xs = track_stats["mean_x"].to_numpy()
    mean_ys = track_stats["mean_y"].to_numpy()

    # Assign vehicles to contours
    centroids_px = np.column_stack([mean_xs, mean_ys])
    veh_contour_labels = assign_tracklets_to_contours(
        centroids_px, contours, frame_shape,
    )

    # Build per-vehicle trajectory lookup (normalized)
    id_col = "id" if "id" in traj_df.columns else "track_id"
    veh_trajectories: Dict[int, np.ndarray] = {}
    min_pts = density_cfg.get("min_track_points", 5)
    for vid, group in traj_df.group_by(id_col):
        vid_val = vid[0] if isinstance(vid, tuple) else vid
        pts = np.column_stack([
            group["x"].to_numpy(),
            group["y"].to_numpy(),
        ]).astype(np.float64) / image_wh
        if len(pts) >= min_pts:
            veh_trajectories[vid_val] = pts

    # Build id -> index mapping for track_stats
    id_to_idx = {int(vid): i for i, vid in enumerate(veh_ids)}

    pseudo_lanes: List[PseudoLane] = []

    for gid, heading_rad in group_headings.items():
        if gid >= len(contours):
            continue

        # Get vehicles in this group
        in_group = veh_contour_labels == gid
        group_veh_ids = veh_ids[in_group]

        if len(group_veh_ids) < 3:
            continue

        group_xs = mean_xs[in_group]
        group_ys = mean_ys[in_group]

        # Compute perpendicular axis from group heading.
        # Fold heading to [0, π) so opposite-direction groups on the same
        # road use the same perpendicular axis → consistent lateral rank.
        canonical_heading = heading_rad % np.pi
        tangent = np.array([np.cos(canonical_heading), np.sin(canonical_heading)])
        perp = np.array([-tangent[1], tangent[0]])

        # Project vehicle centroids onto perpendicular axis (normalized coords)
        centroids_norm = np.column_stack([group_xs, group_ys]) / image_wh
        lateral_offsets = centroids_norm @ perp

        # Step 2: 1D peak finding for lane detection
        # Lane width from model config, or fallback to 0.012 (~15px at 1280w)
        model_cfg = cfg.get("model", {})
        lane_width = model_cfg.get("fixed_lane_width", 0.012)
        n_lanes, lane_assignments, lane_centers = _lateral_clustering(
            lateral_offsets, lane_width=lane_width,
        )

        logger.info(
            f"Group {gid}: heading={np.degrees(heading_rad):.1f}°, "
            f"{len(group_veh_ids)} vehicles, {n_lanes} lanes"
        )

        # Step 3: Build pseudo-lanes
        for lane_idx in range(n_lanes):
            lane_mask = lane_assignments == lane_idx
            lane_veh_ids = group_veh_ids[lane_mask]

            # Collect trajectories for this lane
            trajs = []
            for vid in lane_veh_ids:
                vid_int = int(vid)
                if vid_int in veh_trajectories:
                    trajs.append(veh_trajectories[vid_int])

            if not trajs:
                continue

            # Compute representative geometry (mean trajectory) for traj_stats
            # Use mean of all trajectory centroids as pseudo-geometry
            all_pts = np.concatenate(trajs, axis=0)
            pseudo_wp = _compute_mean_polyline(trajs)

            max_traj_count = max(
                int(np.sum(lane_assignments == li)) for li in range(n_lanes)
            )
            stats = _compute_traj_stats(trajs, pseudo_wp, max_traj_count)

            lane_key = f"{camera}_{gid}_{lane_idx}"
            pseudo_lanes.append(PseudoLane(
                group_id=gid,
                lane_idx=lane_idx,
                lane_key=lane_key,
                heading_rad=heading_rad,
                trajectories=trajs,
                traj_stats=stats,
                lateral_center=float(lane_centers[lane_idx]),
                n_lanes_in_group=n_lanes,
            ))

    logger.info(f"Discovered {len(pseudo_lanes)} pseudo-lanes")
    return pseudo_lanes


def _lateral_clustering(
    offsets: np.ndarray,
    lane_width: float = 0.012,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Cluster vehicles laterally using 1D Gaussian KDE + peak finding.

    Bandwidth is adaptive: uses the smaller of a lane-width-based estimate
    and a data-driven estimate (IQR-based), so it works both for wide US12
    lanes (~0.012 spacing) and narrow I43 lanes (~0.003-0.01 spacing).

    Args:
        offsets: (N,) perpendicular offsets in normalized coordinates.
        lane_width: Expected lane width in normalized coords (~0.012).
            Used as upper bound for bandwidth.

    Returns:
        (n_lanes, assignments, lane_centers)
        - n_lanes: number of detected lanes
        - assignments: (N,) lane index per vehicle
        - lane_centers: (n_lanes,) lateral center of each lane
    """
    N = len(offsets)
    if N < 3:
        return 1, np.zeros(N, dtype=int), np.array([offsets.mean()])

    offset_range = offsets.max() - offsets.min()
    if offset_range < 1e-6:
        return 1, np.zeros(N, dtype=int), np.array([offsets.mean()])

    # Adaptive bandwidth: pick the narrower of two estimates.
    # 1) Lane-width based: half the expected lane width
    bw_lane = lane_width * 0.5
    # 2) Data-driven: IQR-based (robust to outliers), scaled down to
    #    resolve structure within the distribution rather than smooth it.
    #    Factor 0.15 is empirically chosen to resolve lanes with gaps
    #    as small as ~0.005 in normalized coords.
    iqr = np.percentile(offsets, 75) - np.percentile(offsets, 25)
    bw_data = max(iqr * 0.15, 1e-4)
    bw = min(bw_lane, bw_data)

    std = offsets.std()
    if std < 1e-8:
        return 1, np.zeros(N, dtype=int), np.array([offsets.mean()])

    try:
        kde = gaussian_kde(offsets, bw_method=bw / std)
    except (np.linalg.LinAlgError, ZeroDivisionError):
        return 1, np.zeros(N, dtype=int), np.array([offsets.mean()])

    # Evaluate KDE on fine grid
    margin = max(lane_width, bw * 4)
    grid = np.linspace(offsets.min() - margin,
                       offsets.max() + margin, 2000)
    density = kde(grid)

    # Minimum peak distance: adaptive — use half the median nearest-neighbor
    # distance among the offsets, floored at a very small value.
    # This adapts to the actual lane spacing in the data.
    sorted_offsets = np.sort(offsets)
    nn_dists = np.diff(sorted_offsets)
    nn_dists = nn_dists[nn_dists > 1e-6]  # filter duplicates
    if len(nn_dists) > 0:
        min_peak_dist = max(np.median(nn_dists) * 0.3, bw * 0.5)
    else:
        min_peak_dist = bw
    grid_spacing = grid[1] - grid[0]
    min_dist_idx = max(1, int(min_peak_dist / grid_spacing))

    # Low prominence: 2% of max to catch minority-traffic lanes
    prominence = density.max() * 0.02

    peak_indices, _ = find_peaks(
        density, distance=min_dist_idx, prominence=prominence,
    )

    if len(peak_indices) == 0:
        return 1, np.zeros(N, dtype=int), np.array([offsets.mean()])

    lane_centers = grid[peak_indices]

    # Assign each vehicle to nearest peak
    dists = np.abs(offsets[:, None] - lane_centers[None, :])  # (N, K)
    assignments = dists.argmin(axis=1)

    # Sort lanes by lateral position (leftmost = 0)
    sort_order = np.argsort(lane_centers)
    new_centers = lane_centers[sort_order]
    remap = np.zeros(len(sort_order), dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        remap[old_idx] = new_idx
    assignments = remap[assignments]

    return len(new_centers), assignments, new_centers


def _compute_mean_polyline(
    trajectories: List[np.ndarray],
    n_points: int = 16,
) -> np.ndarray:
    """Compute a mean representative polyline from a set of trajectories.

    Resamples each trajectory to n_points, then averages.

    Returns:
        (n_points, 2) mean polyline.
    """
    if not trajectories:
        return np.zeros((n_points, 2), dtype=np.float64)

    resampled = []
    for traj in trajectories:
        r = LaneDataset._resample_polyline(traj, n_points)
        resampled.append(r)

    return np.mean(resampled, axis=0)


def predict_lane_properties(
    pseudo_lanes: List[PseudoLane],
    model: Optional[LaneEncoder] = None,
    ref_proj: Optional[torch.Tensor] = None,
    ref_roles: Optional[torch.Tensor] = None,
    ref_keys: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    polyline_k: int = 16,
    max_traj_per_lane: int = 50,
    max_group_size: int = 6,
) -> List[dict]:
    """Predict lane properties: geometric for rank/edges, encoder for semantics.

    Lateral rank, is_leftmost, is_rightmost, and lane_count are computed
    geometrically from lateral_center ordering within each group — no
    encoder needed. The encoder is optionally used for semantic properties
    (has_successor) that can't be determined from position alone.

    Args:
        pseudo_lanes: Discovered pseudo-lanes from discover_lanes().
        model: Optional trained LaneEncoder (for semantic matching).
        ref_proj: Optional (N_ref, proj_dim) reference projections.
        ref_roles: Optional (N_ref, 5) reference role descriptors.
        ref_keys: Optional reference lane keys.
        device: torch device (required if model is provided).
        polyline_k: Number of points per polyline.
        max_traj_per_lane: Max trajectories to use per lane.

    Returns:
        List of prediction dicts, one per pseudo-lane.
    """
    if not pseudo_lanes:
        return []

    # --- Geometric properties (no encoder) ---
    # Group lanes by group_id, sort by lateral_center within each group
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, pl_lane in enumerate(pseudo_lanes):
        groups[pl_lane.group_id].append(i)

    geo_rank = np.zeros(len(pseudo_lanes))
    geo_is_left = np.zeros(len(pseudo_lanes), dtype=bool)
    geo_is_right = np.zeros(len(pseudo_lanes), dtype=bool)
    geo_n_lanes = np.zeros(len(pseudo_lanes), dtype=int)

    for gid, lane_indices in groups.items():
        # Sort by lateral center within group
        sorted_indices = sorted(
            lane_indices, key=lambda i: pseudo_lanes[i].lateral_center
        )
        n = len(sorted_indices)
        for rank_idx, li in enumerate(sorted_indices):
            geo_rank[li] = rank_idx / max(n - 1, 1) if n > 1 else 0.5
            geo_is_left[li] = (rank_idx == 0)
            geo_is_right[li] = (rank_idx == n - 1)
            geo_n_lanes[li] = n

    # --- Encoder-based semantic properties (optional) ---
    encoder_results = None
    use_encoder = (model is not None and ref_proj is not None
                   and ref_roles is not None and ref_keys is not None
                   and device is not None)

    if use_encoder:
        encoder_results = _encode_and_match(
            model, pseudo_lanes, ref_proj, ref_roles, ref_keys,
            device, polyline_k, max_traj_per_lane, max_group_size,
        )

    # --- Rescale encoder_rank per group via min-max ---
    # The rank_head learns correct ordering but compressed range (e.g.
    # [0.077, 0.101]). Min-max rescaling to [0, 1] preserves the encoder's
    # actual distance ratios between lanes — unlike rank-based rescaling
    # which would produce values identical to geo_rank.
    encoder_rank_rescaled = {}
    if encoder_results is not None:
        for gid, lane_indices in groups.items():
            raw_ranks = {i: encoder_results[i]["predicted_rank"] for i in lane_indices}
            vals = list(raw_ranks.values())
            vmin, vmax = min(vals), max(vals)
            spread = vmax - vmin
            if len(lane_indices) == 1:
                encoder_rank_rescaled[lane_indices[0]] = 0.5
            elif spread < 1e-6:
                # All identical — fall through to raw values
                for li in lane_indices:
                    encoder_rank_rescaled[li] = raw_ranks[li]
            else:
                for li in lane_indices:
                    encoder_rank_rescaled[li] = (raw_ranks[li] - vmin) / spread

    # --- Build predictions ---
    # Minimum trajectories for trusting encoder predictions
    min_traj_for_encoder = 50

    predictions = []
    for i, pl_lane in enumerate(pseudo_lanes):
        n_trajs = len(pl_lane.trajectories)
        pred = {
            "lane_key": pl_lane.lane_key,
            "group_id": pl_lane.group_id,
            "lane_idx": pl_lane.lane_idx,
            "heading_deg": float(np.degrees(pl_lane.heading_rad)),
            "n_trajectories": n_trajs,
            "lateral_center": pl_lane.lateral_center,
            # Geometric properties (reliable, no encoder needed)
            "lateral_rank": float(geo_rank[i]),
            "is_leftmost": bool(geo_is_left[i]),
            "is_rightmost": bool(geo_is_right[i]),
            "n_lanes_in_group": int(geo_n_lanes[i]),
        }

        if encoder_results is not None:
            enc = encoder_results[i]
            # Fall back to geometric rank for sparse lanes
            confident = n_trajs >= min_traj_for_encoder
            rescaled_rank = encoder_rank_rescaled.get(i, enc["predicted_rank"])
            pred.update({
                # Regression head predictions (from encoder)
                "encoder_rank": rescaled_rank if confident else float(geo_rank[i]),
                "encoder_rank_raw": enc["predicted_rank"],  # unscaled for diagnostics
                "encoder_is_leftmost": enc["predicted_is_leftmost"] if confident else bool(geo_is_left[i]),
                "encoder_is_rightmost": enc["predicted_is_rightmost"] if confident else bool(geo_is_right[i]),
                "encoder_group_size": enc["predicted_group_size"],
                "encoder_confidence": "high" if confident else "low",
                # Cosine matching diagnostic
                "match_similarity": enc["match_similarity"],
                "matched_ref_key": enc["matched_ref_key"],
            })

        predictions.append(pred)

    return predictions


@torch.no_grad()
def _encode_and_match(
    model: LaneEncoder,
    pseudo_lanes: List[PseudoLane],
    ref_proj: torch.Tensor,
    ref_roles: torch.Tensor,
    ref_keys: List[str],
    device: torch.device,
    polyline_k: int,
    max_traj_per_lane: int,
    max_group_size: int = 6,
) -> List[dict]:
    """Encode pseudo-lanes and predict semantic properties via regression heads.

    Uses the model's rank_head, edge_head, and size_head directly for
    predictions instead of relying on cosine similarity matching to a
    reference bank (which suffers from representation collapse).

    Cosine matching to the reference bank is still computed for diagnostics
    (match_similarity, matched_ref_key) but is NOT used for property prediction.

    Returns list of dicts with encoder-derived properties per lane.
    """
    model.eval()

    geometries = []
    traj_polylines_list = []
    traj_stats_list = []

    for pl_lane in pseudo_lanes:
        geometries.append(torch.zeros(polyline_k, 2))

        trajs = pl_lane.trajectories
        if len(trajs) > max_traj_per_lane:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(trajs), max_traj_per_lane, replace=False)
            trajs = [trajs[i] for i in indices]

        traj_polys = []
        for t in trajs:
            r = LaneDataset._resample_polyline(t, polyline_k)
            traj_polys.append(torch.tensor(r, dtype=torch.float32))
        if not traj_polys:
            traj_polys = [torch.zeros(polyline_k, 2)]
        traj_polylines_list.append(traj_polys)

        traj_stats_list.append(
            torch.tensor(pl_lane.traj_stats, dtype=torch.float32))

    B = len(pseudo_lanes)
    geometry = torch.stack(geometries)
    traj_stats = torch.stack(traj_stats_list)

    # Build geometric role estimates for pseudo-lanes
    # Group pseudo-lanes by group_id, compute lateral rank from lateral_center ordering
    # Uses training max_group_size for consistent normalization
    group_lanes_map = defaultdict(list)
    for i, pl in enumerate(pseudo_lanes):
        group_lanes_map[pl.group_id].append(i)

    role_list = torch.zeros(B, 5)
    for gid, lane_indices_in_group in group_lanes_map.items():
        # Sort by lateral_center
        sorted_indices = sorted(lane_indices_in_group, key=lambda i: pseudo_lanes[i].lateral_center)
        n = len(sorted_indices)
        for rank_idx, i in enumerate(sorted_indices):
            lat_rank = rank_idx / max(n - 1, 1) if n > 1 else 0.5
            role_list[i, 0] = lat_rank
            role_list[i, 1] = float(rank_idx == 0)        # is_leftmost
            role_list[i, 2] = float(rank_idx == n - 1)    # is_rightmost
            role_list[i, 3] = 0.0                          # has_successor (unknown)
            role_list[i, 4] = n / max(max_group_size, 1)   # group_size_norm (training scale)

    # Concatenate traj_stats + roles as encoder input (matches training)
    stats_input = torch.cat([traj_stats, role_list], dim=-1)  # (B, 9)

    max_n_trajs = max(len(tp) for tp in traj_polylines_list)
    traj_padded = torch.zeros(B, max_n_trajs, polyline_k, 2)
    traj_mask = torch.zeros(B, max_n_trajs, dtype=torch.bool)
    for i, tps in enumerate(traj_polylines_list):
        for j, tp in enumerate(tps):
            traj_padded[i, j] = tp
        traj_mask[i, :len(tps)] = True

    if model.use_cross_lane_attention:
        # Build group_ids from pseudo-lanes
        group_id_list = [pl_lane.group_id for pl_lane in pseudo_lanes]
        group_ids = torch.tensor(group_id_list, dtype=torch.long)
        output = model.forward_grouped(
            geometry=geometry.to(device),
            traj_polylines=traj_padded.to(device),
            traj_mask=traj_mask.to(device),
            traj_stats=stats_input.to(device),
            group_ids=group_ids.to(device),
            drop_geometry=True,
        )
    else:
        output = model(
            geometry=geometry.to(device),
            traj_polylines=traj_padded.to(device),
            traj_mask=traj_mask.to(device),
            traj_stats=stats_input.to(device),
            drop_geometry=True,
        )

    # --- Use regression heads directly for predictions ---
    # All heads output raw logits; apply sigmoid for bounded [0,1] output.
    pred_rank = torch.sigmoid(output["pred_rank"]).cpu()  # [0, 1]
    pred_edge = torch.sigmoid(output["pred_edge"]).cpu()  # (B, 2) logits -> prob
    pred_size = torch.sigmoid(output["pred_size"]).cpu()  # (B,) bounded [0, 1]

    # --- Cosine matching for diagnostics only ---
    query_proj = output["projection"].cpu()
    sim_matrix = torch.mm(query_proj, ref_proj.t())
    best_match_idx = sim_matrix.argmax(dim=1)
    best_match_sim = sim_matrix.gather(1, best_match_idx.unsqueeze(1)).squeeze(1)

    results = []
    for i in range(B):
        results.append({
            # Regression head predictions (primary)
            "predicted_rank": float(pred_rank[i]),
            "predicted_is_leftmost": bool(pred_edge[i, 0] > 0.5),
            "predicted_is_rightmost": bool(pred_edge[i, 1] > 0.5),
            "predicted_group_size": float(pred_size[i]),
            # Cosine matching (diagnostic only)
            "match_similarity": float(best_match_sim[i]),
            "matched_ref_key": ref_keys[best_match_idx[i].item()],
        })
    return results
