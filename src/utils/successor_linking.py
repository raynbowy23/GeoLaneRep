"""Deterministic successor linking: chain lanelet nodes into lanes using geometry.

Given predicted lanelet positions + headings + confidence, produce long lane
chains without requiring the learned edge classifier to converge.

Scoring function for candidate edge (u -> v):
    1. Downstream: dot(heading_u, v - u) > 0  (v is ahead of u)
    2. Small lateral offset: |cross(heading_u, v - u)| < threshold
    3. Heading consistency: |cos(heading_u, heading_v)| > threshold
    4. Distance penalty: prefer closer nodes

This produces the same lane visualization as the learned edge classifier
but works from epoch 1.  Later the learned successor head can replace or
refine these links.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def deterministic_successor_linking(
    positions: np.ndarray,
    headings: np.ndarray,
    confidence: np.ndarray,
    conf_threshold: float = 0.3,
    max_distance: float = 0.20,
    max_lateral_offset: float = 0.02,
    min_heading_cos: float = 0.7,
    min_downstream: float = 0.004,
) -> Tuple[List[List[int]], Dict[int, int]]:
    """Chain active lanelet nodes into lane polylines using geometry.

    Args:
        positions: (M, 2) lanelet node positions in meters.
        headings: (M, 2) unit tangent vectors at each node.
        confidence: (M,) confidence scores.
        conf_threshold: minimum confidence for active nodes.
        max_distance: maximum distance (meters) for successor candidates.
        max_lateral_offset: max perpendicular offset (meters) from heading.
        min_heading_cos: minimum |cos(angle)| between headings.
        min_downstream: minimum downstream projection (meters).

    Returns:
        chains: list of node-index lists (each is a lane).
        successor_map: dict mapping node -> its successor node.
    """
    M = len(positions)
    active_mask = confidence > conf_threshold
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        return [], {}

    # Score all candidate directed edges between active nodes
    successor_map: Dict[int, int] = {}  # u -> best_v

    for u in active_indices:
        pos_u = positions[u]
        head_u = headings[u]
        # Perpendicular direction
        perp_u = np.array([-head_u[1], head_u[0]])

        best_score = -np.inf
        best_v = -1

        for v in active_indices:
            if v == u:
                continue

            diff = positions[v] - pos_u  # vector from u to v
            dist = np.linalg.norm(diff)

            if dist > max_distance or dist < 1e-6:
                continue

            # 1. Downstream: projection along heading_u
            downstream = np.dot(diff, head_u)
            if downstream < min_downstream:
                continue  # v is not ahead of u

            # 2. Lateral offset: projection perpendicular to heading_u
            lateral = abs(np.dot(diff, perp_u))
            if lateral > max_lateral_offset:
                continue  # v is too far to the side

            # 3. Heading consistency
            head_cos = abs(np.dot(head_u, headings[v]))
            if head_cos < min_heading_cos:
                continue  # headings diverge too much

            # Composite score: prefer close, downstream, heading-consistent
            # Higher = better
            score = (
                downstream / dist          # reward downstream-ness (0-1)
                + head_cos                  # reward heading alignment (0-1)
                - lateral / max_lateral_offset  # penalize lateral offset (0-1)
                - dist / max_distance       # penalize distance (0-1)
            )

            if score > best_score:
                best_score = score
                best_v = v

        if best_v >= 0:
            successor_map[u] = best_v

    # Build chains by following successor links from roots
    chains = _extract_chains(active_indices, successor_map)
    return chains, successor_map


def _extract_chains(
    active_indices: np.ndarray,
    successor_map: Dict[int, int],
) -> List[List[int]]:
    """Extract lane chains from successor map.

    Finds root nodes (no predecessor) and follows chains greedily.
    Handles cycles by tracking visited nodes.
    """
    # Find nodes that are successors (have a predecessor)
    has_predecessor = set(successor_map.values())
    active_set = set(active_indices.tolist())

    # Roots: active nodes with no predecessor
    roots = sorted(n for n in active_set if n not in has_predecessor)

    visited = set()
    chains = []

    for root in roots:
        if root in visited:
            continue
        chain = [root]
        visited.add(root)
        current = root
        while current in successor_map:
            nxt = successor_map[current]
            if nxt in visited:
                break  # avoid cycles
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        chains.append(chain)

    # Remaining unvisited active nodes as single-node chains
    for n in active_set:
        if n not in visited:
            chains.append([n])
            visited.add(n)

    return chains


def compute_chain_metrics(
    chains: List[List[int]],
    positions: np.ndarray,
    gt_positions: Optional[np.ndarray] = None,
    gt_lane_ids: Optional[np.ndarray] = None,
    match_radius: float = 0.02,
) -> dict:
    """Evaluate successor chain quality.

    Args:
        chains: list of node-index lists (predicted lanes).
        positions: (M, 2) lanelet positions.
        gt_positions: (G, 2) GT waypoint positions (optional).
        gt_lane_ids: (G,) GT lane id per waypoint (optional).
        match_radius: distance threshold for matching.

    Returns:
        dict with:
            n_chains: number of multi-node chains.
            mean_chain_length: average number of nodes per chain.
            max_chain_length: longest chain.
            chain_lane_recall: fraction of GT lanes hit by a chain.
            chain_lane_precision: fraction of chains that hit a GT lane.
    """
    metrics = {}

    multi_chains = [c for c in chains if len(c) >= 2]
    metrics["n_chains"] = len(multi_chains)
    metrics["n_chains_total"] = len(chains)

    if multi_chains:
        lengths = [len(c) for c in multi_chains]
        metrics["mean_chain_length"] = float(np.mean(lengths))
        metrics["max_chain_length"] = int(np.max(lengths))
    else:
        metrics["mean_chain_length"] = 0.0
        metrics["max_chain_length"] = 0

    # GT matching (if available)
    if gt_positions is not None and gt_lane_ids is not None and len(gt_positions) > 0:
        unique_gt_lanes = np.unique(gt_lane_ids)
        n_gt_lanes = len(unique_gt_lanes)

        # Lane recall: GT lane recalled if any multi-node chain passes within radius
        recalled = 0
        for lid in unique_gt_lanes:
            lane_pts = gt_positions[gt_lane_ids == lid]
            for chain in multi_chains:
                chain_pts = positions[chain]
                # Check if any chain point is near any GT lane point
                dists = np.linalg.norm(
                    chain_pts[:, None, :] - lane_pts[None, :, :], axis=2
                )
                if dists.min() < match_radius:
                    recalled += 1
                    break

        metrics["chain_lane_recall"] = float(recalled) / max(n_gt_lanes, 1)

        # Lane precision: fraction of multi-node chains near any GT lane
        if multi_chains:
            precise = 0
            for chain in multi_chains:
                chain_pts = positions[chain]
                dists = np.linalg.norm(
                    chain_pts[:, None, :] - gt_positions[None, :, :], axis=2
                )
                if dists.min(axis=1).mean() < match_radius:
                    precise += 1
            metrics["chain_lane_precision"] = float(precise) / len(multi_chains)
        else:
            metrics["chain_lane_precision"] = 0.0
    else:
        metrics["chain_lane_recall"] = float("nan")
        metrics["chain_lane_precision"] = float("nan")

    return metrics
