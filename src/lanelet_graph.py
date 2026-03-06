"""Build data-driven lanelet graph from lane assignment results.

Constructs lane centerlines from actual vehicle trajectories grouped by
assigned lane, rather than from annotation geometry. Annotation is only
used for topology (successor/adjacent edges).

Output format matches the GT lanelet graph:
  positions, tangents, edge_index, edge_types, lane_ids, group_ids
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from src.data.annotation_loader import (
    get_annotation_relationships,
    load_annotation_json,
)

logger = logging.getLogger(__name__)


def _resample_at_arclength(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at uniform arc-length intervals.

    Args:
        polyline: (N, 2) ordered points.
        spacing: desired distance between consecutive output points (pixels).

    Returns:
        (M, 2) resampled polyline with uniform spacing.
    """
    if len(polyline) < 2:
        return polyline
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]
    if total_len < 1e-8:
        return polyline[:1]
    n_out = max(int(total_len / spacing) + 1, 2)
    t_out = np.linspace(0.0, total_len, n_out)
    return np.column_stack([
        np.interp(t_out, cum_len, polyline[:, 0]),
        np.interp(t_out, cum_len, polyline[:, 1]),
    ])


def _compute_centerline(
    points_sorted: np.ndarray,
    spacing_px: float,
    smooth_window: Optional[int] = None,
) -> np.ndarray:
    """Compute a smooth centerline from time-sorted trajectory points.

    1. Moving-average smooth (handles curves naturally).
    2. Resample at uniform arc-length spacing.

    Args:
        points_sorted: (N, 2) points sorted by time.
        spacing_px: arc-length spacing in pixels.
        smooth_window: moving average window size. If None, auto = max(3, N//20).

    Returns:
        (M, 2) resampled centerline in pixel coords.
    """
    n = len(points_sorted)
    if n < 2:
        return points_sorted

    if smooth_window is None:
        smooth_window = max(3, n // 20)
    # Ensure odd window
    smooth_window = max(3, smooth_window | 1)

    if n <= smooth_window:
        smoothed = points_sorted
    else:
        kernel = np.ones(smooth_window) / smooth_window
        x_smooth = np.convolve(points_sorted[:, 0], kernel, mode="valid")
        y_smooth = np.convolve(points_sorted[:, 1], kernel, mode="valid")
        smoothed = np.column_stack([x_smooth, y_smooth])

    if len(smoothed) < 2:
        return smoothed

    return _resample_at_arclength(smoothed, spacing_px)


def _compute_tangents(positions: np.ndarray) -> np.ndarray:
    """Compute unit tangent vectors via finite differences.

    Forward diff at start, backward at end, central in interior.
    """
    n = len(positions)
    tangents = np.zeros_like(positions)
    if n < 2:
        return tangents

    # Central differences for interior
    tangents[1:-1] = positions[2:] - positions[:-2]
    # Forward/backward at endpoints
    tangents[0] = positions[1] - positions[0]
    tangents[-1] = positions[-1] - positions[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    tangents = tangents / norms
    return tangents


def build_lanelet_graph(
    assignments_path: str,
    annotation_path: str,
    image_wh: Tuple[int, int],
    waypoint_spacing_px: float = 20.0,
    min_points_per_lane: int = 3,
) -> dict:
    """Build a data-driven lanelet graph from lane assignment results.

    For each (group_id, lane_id) with assigned tracklets:
    1. Collect all (x, y) points, sort by time
    2. Smooth via moving average → mean path (handles curves)
    3. Resample at uniform arc-length intervals → centerline waypoints
    4. Normalize positions to [0,1], compute unit tangents
    5. Build topology edges from annotation relationships

    Args:
        assignments_path: Path to lane_assignments.csv.
        annotation_path: Path to annotation.json (topology only).
        image_wh: (width, height) of camera frame.
        waypoint_spacing_px: Pixel arc-length between waypoints.
        min_points_per_lane: Skip lanes with fewer trajectory points.

    Returns:
        Dict with keys:
            positions: (G, 2) float — normalized [0,1]
            tangents: (G, 2) float — unit vectors
            lane_ids: (G,) int
            group_ids: (G,) int
            edge_index: (2, E) int
            edge_types: (E,) int — 1=successor, 4=adjacent
            lane_meta: list of {group_id, cls_id, node_range: (start, end)}
    """
    # Load assignments
    df = pl.read_csv(assignments_path)

    if len(df) == 0:
        logger.warning(f"Empty assignments file: {assignments_path}")
        return _empty_graph()

    df = df.filter(pl.col("lane_id") >= 0)

    if len(df) == 0:
        logger.warning(f"No valid assignments in {assignments_path}")
        return _empty_graph()

    # Load annotation for topology
    annotation = load_annotation_json(annotation_path)

    wh = np.array(image_wh, dtype=np.float64)

    all_positions: List[np.ndarray] = []
    all_tangents: List[np.ndarray] = []
    all_lane_ids: List[int] = []
    all_group_ids: List[int] = []
    lane_meta: List[dict] = []
    lane_node_ranges: Dict[str, Tuple[int, int]] = {}

    # Process each (group_id, lane_id) pair
    grouped = df.group_by(["group_id", "lane_id"]).agg([
        pl.col("x").alias("xs"),
        pl.col("y").alias("ys"),
        pl.col("time").alias("times"),
    ])

    lane_counter = 0
    for row in grouped.iter_rows(named=True):
        gid = row["group_id"]
        lid = row["lane_id"]

        xs = np.array(row["xs"], dtype=np.float64)
        ys = np.array(row["ys"], dtype=np.float64)
        times = np.array(row["times"], dtype=np.float64)

        if len(xs) < min_points_per_lane:
            logger.debug(f"Skipping G{gid}/L{lid}: only {len(xs)} points")
            continue

        # Sort by time
        order = np.argsort(times)
        points = np.column_stack([xs[order], ys[order]])

        # Compute smoothed centerline (pixel space)
        centerline = _compute_centerline(points, waypoint_spacing_px)
        if len(centerline) < 2:
            logger.warning(f"G{gid}/L{lid}: centerline too short after smoothing")
            continue

        # Normalize to [0,1]
        centerline_norm = centerline / wh

        # Compute tangents on normalized positions
        tangents = _compute_tangents(centerline_norm)

        start_idx = sum(len(p) for p in all_positions)
        n_wp = len(centerline_norm)

        all_positions.append(centerline_norm)
        all_tangents.append(tangents)
        all_lane_ids.extend([lane_counter] * n_wp)
        all_group_ids.extend([gid] * n_wp)

        lane_key = f"annot_{gid}_{lid}"
        lane_node_ranges[lane_key] = (start_idx, start_idx + n_wp)
        lane_meta.append({
            "group_id": int(gid),
            "cls_id": int(lid),
            "node_range": (start_idx, start_idx + n_wp),
        })

        lane_counter += 1

    if not all_positions:
        logger.warning("No valid lanes after processing")
        return _empty_graph()

    positions = np.concatenate(all_positions, axis=0)
    tangents = np.concatenate(all_tangents, axis=0)
    lane_ids = np.array(all_lane_ids, dtype=np.int64)
    group_ids = np.array(all_group_ids, dtype=np.int64)
    G = len(positions)

    logger.info(f"Built {lane_counter} lanes, {G} total waypoints")

    # ── Build edges ──────────────────────────────────────────
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[int] = []

    # Intra-lane: consecutive waypoints (bidirectional, type=1)
    for lm in lane_meta:
        s, e = lm["node_range"]
        for i in range(s, e - 1):
            edge_src.append(i)
            edge_dst.append(i + 1)
            edge_types.append(1)
            edge_src.append(i + 1)
            edge_dst.append(i)
            edge_types.append(1)

    # Cross-lane edges from annotation relationships
    unique_groups = set(int(m["group_id"]) for m in lane_meta)
    for gid in unique_groups:
        relationships = get_annotation_relationships(annotation, gid)
        for rel in relationships:
            from_key = f"annot_{rel['from_group']}_{rel['from_cls']}"
            to_key = f"annot_{rel['to_group']}_{rel['to_cls']}"

            if from_key not in lane_node_ranges or to_key not in lane_node_ranges:
                continue

            from_s, from_e = lane_node_ranges[from_key]
            to_s, to_e = lane_node_ranges[to_key]

            if rel["type"] == "successor":
                # Last waypoint of from → first waypoint of to
                edge_src.append(from_e - 1)
                edge_dst.append(to_s)
                edge_types.append(1)
                edge_src.append(to_s)
                edge_dst.append(from_e - 1)
                edge_types.append(1)

            elif rel["type"] == "adjacent":
                # Connect waypoints at matching normalized arc-length t
                from_len = from_e - from_s
                to_len = to_e - to_s
                n_pairs = min(from_len, to_len)
                for k in range(n_pairs):
                    fi = from_s + int(k * from_len / n_pairs)
                    ti = to_s + int(k * to_len / n_pairs)
                    edge_src.append(fi)
                    edge_dst.append(ti)
                    edge_types.append(4)
                    edge_src.append(ti)
                    edge_dst.append(fi)
                    edge_types.append(4)

    if edge_src:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_type_arr = np.array(edge_types, dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_type_arr = np.zeros(0, dtype=np.int64)

    n_succ = int(np.sum(edge_type_arr == 1)) if len(edge_type_arr) > 0 else 0
    n_adj = int(np.sum(edge_type_arr == 4)) if len(edge_type_arr) > 0 else 0
    logger.info(f"Edges: {len(edge_type_arr)} total ({n_succ} successor, {n_adj} adjacent)")

    return {
        "positions": positions.astype(np.float32),
        "tangents": tangents.astype(np.float32),
        "lane_ids": lane_ids,
        "group_ids": group_ids,
        "edge_index": edge_index,
        "edge_types": edge_type_arr,
        "lane_meta": lane_meta,
    }


def _empty_graph() -> dict:
    """Return an empty lanelet graph dict."""
    return {
        "positions": np.zeros((0, 2), dtype=np.float32),
        "tangents": np.zeros((0, 2), dtype=np.float32),
        "lane_ids": np.zeros(0, dtype=np.int64),
        "group_ids": np.zeros(0, dtype=np.int64),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_types": np.zeros(0, dtype=np.int64),
        "lane_meta": [],
    }
