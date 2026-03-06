"""Lane extraction from lanelet graph output.

Converts lanelet graph (nodes + edges) into lane centerlines using
scored path extraction through the successor graph.
"""

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ExtractedLaneletLane:
    """A lane extracted from the lanelet graph by following successor chains."""
    lane_id: int
    waypoints: np.ndarray  # (P, 2) ordered waypoint positions
    headings: np.ndarray  # (P, 2) unit tangent at each waypoint
    confidence: float  # mean confidence of constituent nodes
    node_indices: np.ndarray  # indices into the lanelet node array
    score: float = 0.0  # path score (higher is better)


def extract_lanelet_lanes(
    model_output: dict,
    conf_threshold: float = 0.3,
    min_nodes: int = 2,
) -> List[ExtractedLaneletLane]:
    """Extract lane centerlines as maximum-likelihood paths.

    Algorithm:
    1. Build directed graph from successor edges with probabilities.
    2. Find source nodes (no incoming successor) and sink nodes (no outgoing).
    3. For each source, find best path to any sink via greedy best-first:
       score = sum(log(edge_prob)) + sum(log(conf)) - lambda * sum(turn_penalty)
    4. Extract top-N non-overlapping paths.

    Args:
        model_output: dict from LaneReprModel.forward() (v3 output).
        conf_threshold: minimum node confidence for inclusion.
        min_nodes: minimum waypoints to form a lane.

    Returns:
        List of ExtractedLaneletLane.
    """
    pred_pos = model_output["lanelet_positions"].detach().cpu().numpy()    # (M, 2)
    pred_head = model_output["lanelet_headings"].detach().cpu().numpy()    # (M, 2)
    pred_conf = model_output["lanelet_confidence"].detach().cpu().numpy()  # (M,)
    succ_logits = model_output["successor_logits"].detach()                 # (E_c,)
    edge_index = model_output["edge_type_index"].detach().cpu().numpy()    # (2, E_c)

    M = pred_pos.shape[0]

    # Active nodes (above confidence threshold)
    active = pred_conf > conf_threshold
    active_set = set(np.where(active)[0])

    if len(active_set) < min_nodes:
        return []

    # Build scored directed edge graph from successor probabilities
    # scored_edges[src] -> list of (dst, score)
    scored_adj: Dict[int, List[tuple]] = {}
    has_incoming = set()

    if succ_logits.shape[0] > 0:
        succ_probs = torch.sigmoid(succ_logits).cpu().numpy()  # (E_c,)
        for e in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            succ_prob = float(succ_probs[e])  # binary successor probability

            if src not in active_set or dst not in active_set:
                continue
            if succ_prob < 0.1:
                continue

            # Score = log(succ_prob) + log(conf_dst) - turn_penalty
            heading_i = pred_head[src]
            heading_j = pred_head[dst]
            cos_turn = float(np.dot(heading_i, heading_j))
            turn_penalty = max(0.0, 1.0 - cos_turn) * 0.5

            score = (np.log(succ_prob + 1e-8)
                     + np.log(pred_conf[dst] + 1e-8)
                     - turn_penalty)

            scored_adj.setdefault(src, []).append((dst, score))
            has_incoming.add(dst)

    # Find source nodes (active, no incoming successor edge)
    sources = [n for n in active_set if n not in has_incoming]
    # If no natural sources, use all active nodes as potential starts
    if not sources:
        sources = sorted(active_set, key=lambda n: -pred_conf[n])

    # Extract paths via greedy best-first from each source
    used_nodes = set()
    lanes = []
    lane_id = 0

    # Sort sources by confidence (best starts first)
    sources.sort(key=lambda n: -pred_conf[n])

    for source in sources:
        if source in used_nodes:
            continue

        # Greedy best-first path from source
        path = [source]
        path_score = np.log(pred_conf[source] + 1e-8)
        visited_in_path = {source}
        current = source

        while current in scored_adj:
            # Pick the best-scored unvisited successor
            candidates = [
                (dst, sc) for dst, sc in scored_adj[current]
                if dst not in visited_in_path and dst not in used_nodes
            ]
            if not candidates:
                break

            best_dst, best_sc = max(candidates, key=lambda x: x[1])
            path.append(best_dst)
            path_score += best_sc
            visited_in_path.add(best_dst)
            current = best_dst

        if len(path) >= min_nodes:
            indices = np.array(path)
            lanes.append(ExtractedLaneletLane(
                lane_id=lane_id,
                waypoints=pred_pos[indices],
                headings=pred_head[indices],
                confidence=float(pred_conf[indices].mean()),
                node_indices=indices,
                score=float(path_score),
            ))
            used_nodes.update(path)
            lane_id += 1

    # Sort by score (best lanes first)
    lanes.sort(key=lambda l: l.score, reverse=True)

    # Re-assign sequential lane IDs
    for i, lane in enumerate(lanes):
        lane.lane_id = i

    return lanes


def lanelet_lanes_to_dict(lanes: List[ExtractedLaneletLane]) -> List[Dict]:
    """Convert extracted lanelet lanes to JSON-serializable dicts."""
    return [
        {
            "lane_id": lane.lane_id,
            "waypoints": lane.waypoints.tolist(),
            "headings": lane.headings.tolist(),
            "confidence": lane.confidence,
            "score": lane.score,
            "num_nodes": len(lane.node_indices),
        }
        for lane in lanes
    ]
