import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch

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

    # Concatenate traj_stats + roles as encoder input
    stats_input = torch.cat([traj_stats, role_list], dim=-1)

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
