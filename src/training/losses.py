"""Lanelet loss components for v3 lanelet discovery.

Loss components:
  1.  lanelet_node_loss            — Hungarian-matched L1 on positions.
  2.  lanelet_heading_loss         — 1 - cos(angle) on matched pairs.
  3.  lanelet_edge_loss            — Focal cross-entropy on edge types.
  4.  lanelet_confidence           — BCE: matched->1, empty->0.
  5.  pooling_entropy              — Sharp assignment regularization.
  5b. balancing                    — Anti-collapse: uniform slot utilization.
  6.  pooling_link                 — Graph structure preservation.
  7.  lanelet_assign_consistency   — Same-GT-lane tracklets -> same lanelet (KL, confidence-gated).
  8.  drivable_loss                — BCE: lanelet is drivable if trajectory support > threshold.
  9.  data_fit_loss                — Soft assignment-weighted distance (tracklets near lanelets).
  10. sumo_topology_prior_loss     — Gated SUMO topology CE prior.
  11. separation_loss              — Penalize lanelet nodes closer than min_distance.
  12. heading_smoothness_loss       — Penalize heading jumps along successor edges.
  13. collinearity_loss             — Penalize zigzag: middle node deviation from predecessor-successor line.
  14. lane_classification_loss      — Cross-entropy on assignment logits (with warmup gate).
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def lanelet_node_loss(
    pred_positions: torch.Tensor,
    gt_positions: torch.Tensor,
    pred_confidence: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Hungarian-matched L1 loss on lanelet node positions.

    Args:
        pred_positions: (M, 2) predicted waypoint positions.
        gt_positions: (G, 2) GT waypoint positions.
        pred_confidence: (M,) predicted confidence scores.

    Returns:
        position_loss: scalar L1 loss on matched pairs.
        matched_pred_idx: (K,) indices of matched predicted nodes.
        matched_gt_idx: (K,) indices of matched GT nodes.
        match_stats: dict with diagnostic distance statistics.
    """
    M = pred_positions.shape[0]
    G = gt_positions.shape[0]

    empty_stats = {"mean_match_dist": 0.0, "median_min_gt_dist": 0.0,
                   "pred_range": 0.0, "gt_range": 0.0}

    if G == 0:
        return torch.tensor(0.0, device=pred_positions.device), \
               torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), empty_stats

    # Cost matrix: pairwise L1 distance (float32 to avoid AMP overflow)
    # (M, 1, 2) - (1, G, 2) -> (M, G)
    cost = (pred_positions.float().unsqueeze(1) - gt_positions.float().unsqueeze(0)).abs().sum(dim=-1)
    cost_np = cost.detach().cpu().float().numpy()
    np.nan_to_num(cost_np, nan=1e6, posinf=1e6, neginf=1e6, copy=False)

    # Diagnostic: coordinate scale and alignment stats
    pred_np = pred_positions.detach().cpu().float().numpy()
    gt_np = gt_positions.detach().cpu().float().numpy()
    pred_range = float(np.ptp(pred_np, axis=0).max()) if M > 0 else 0.0
    gt_range = float(np.ptp(gt_np, axis=0).max()) if G > 0 else 0.0
    min_dist_per_gt = float(np.median(cost_np.min(axis=0)))  # median of min pred->GT dist

    row_ind, col_ind = linear_sum_assignment(cost_np)

    # Filter out clearly unmatched pairs (distance > 0.20 in normalized space)
    valid = cost_np[row_ind, col_ind] < 0.20
    row_ind = row_ind[valid]
    col_ind = col_ind[valid]

    match_dists = cost_np[row_ind, col_ind] if len(row_ind) > 0 else np.array([])
    match_stats = {
        "mean_match_dist": float(match_dists.mean()) if len(match_dists) > 0 else 0.0,
        "median_min_gt_dist": min_dist_per_gt,
        "pred_range": pred_range,
        "gt_range": gt_range,
        "match_distances": match_dists,  # (K,) L2 distances for soft conf targets
    }

    if len(row_ind) == 0:
        return torch.tensor(0.0, device=pred_positions.device), \
               torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long), match_stats

    matched_pred = torch.tensor(row_ind, dtype=torch.long, device=pred_positions.device)
    matched_gt = torch.tensor(col_ind, dtype=torch.long, device=pred_positions.device)

    # L1 loss on matched positions
    pos_loss = (pred_positions[matched_pred] - gt_positions[matched_gt]).abs().sum(dim=-1).mean()

    return pos_loss, matched_pred, matched_gt, match_stats


def lanelet_coverage_loss(
    pred_positions: torch.Tensor,
    gt_positions: torch.Tensor,
    threshold: float = 8.0,
) -> torch.Tensor:
    """Coverage loss: penalize GT waypoints with no nearby predicted slot.

    For each GT waypoint, compute distance to nearest predicted slot.
    Penalize distances beyond threshold (soft margin).

    Args:
        pred_positions: (M, 2) predicted waypoint positions.
        gt_positions: (G, 2) GT waypoint positions.
        threshold: margin in meters — GT within this distance is considered covered.

    Returns:
        Scalar coverage loss.
    """
    if len(gt_positions) == 0 or len(pred_positions) == 0:
        return torch.tensor(0.0, device=pred_positions.device)

    # (G, M) pairwise L2 distance
    dist = torch.cdist(gt_positions.float(), pred_positions.float())  # (G, M)
    min_dist = dist.min(dim=1).values  # (G,) nearest pred for each GT

    # Penalize GT waypoints beyond threshold (smooth L1-like)
    coverage = F.relu(min_dist - threshold)
    return coverage.mean()


def lanelet_heading_loss(
    pred_headings: torch.Tensor,
    gt_tangents: torch.Tensor,
    matched_pred_idx: torch.Tensor,
    matched_gt_idx: torch.Tensor,
) -> torch.Tensor:
    """Heading loss: 1 - cos(angle) on matched pairs.

    Args:
        pred_headings: (M, 2) predicted unit tangents.
        gt_tangents: (G, 2) GT unit tangents.
        matched_pred_idx: (K,) matched pred indices.
        matched_gt_idx: (K,) matched GT indices.

    Returns:
        Scalar heading loss.
    """
    if len(matched_pred_idx) == 0:
        return torch.tensor(0.0, device=pred_headings.device)

    pred_h = pred_headings[matched_pred_idx]  # (K, 2)
    gt_h = gt_tangents[matched_gt_idx]  # (K, 2)

    # Cosine similarity (dot product of unit vectors)
    cos_sim = (pred_h * gt_h).sum(dim=-1)  # (K,)
    # 1 - |cos| to handle direction ambiguity
    loss = (1.0 - cos_sim.abs()).mean()
    return loss


def binary_successor_loss(
    successor_logits: torch.Tensor,
    edge_type_index: torch.Tensor,
    gt_edge_index: torch.Tensor,
    gt_edge_types: torch.Tensor,
    matched_pred_idx: torch.Tensor,
    matched_gt_idx: torch.Tensor,
    num_lanelet_nodes: int,
    num_gt_nodes: int,
    pred_pos: torch.Tensor = None,
    pred_head: torch.Tensor = None,
    hard_neg_k: int = 4,
) -> torch.Tensor:
    """Binary successor loss with hard-negative mining.

    For each matched pred node that has a GT successor:
      - Positive: the edge to the GT successor destination (if in candidate set)
      - Hard negatives: K candidates closest by lateral distance
      - Weighted BCE: pos_weight = n_neg/n_pos, capped at 10

    When no GT successors match: train all candidates as negative.

    Args:
        successor_logits: (E_c,) binary logits for each candidate edge.
        edge_type_index: (2, E_c) candidate edge indices (in lanelet space).
        gt_edge_index: (2, E_gt) GT edge indices (in GT space).
        gt_edge_types: (E_gt,) GT edge type labels (1=successor).
        matched_pred_idx: (K,) matched pred node indices.
        matched_gt_idx: (K,) matched GT node indices.
        num_lanelet_nodes: M.
        num_gt_nodes: G.
        pred_pos: (M, 2) predicted lanelet positions (for hard-neg geometry).
        pred_head: (M, 2) predicted lanelet headings (for hard-neg geometry).
        hard_neg_k: number of hard negatives per positive.

    Returns:
        Scalar binary cross-entropy loss.
    """
    E_c = successor_logits.shape[0]
    if E_c == 0 or len(matched_pred_idx) == 0:
        return torch.tensor(0.0, device=successor_logits.device)

    device = successor_logits.device

    # Build GT->pred node mapping
    gt_to_pred = torch.full((num_gt_nodes,), -1, dtype=torch.long, device=device)
    gt_to_pred[matched_gt_idx] = matched_pred_idx

    # Find GT successor edges (type == 1) where both endpoints are matched
    succ_mask = gt_edge_types == 1
    gt_src = gt_edge_index[0]
    gt_dst = gt_edge_index[1]
    gt_src_mapped = gt_to_pred[gt_src]
    gt_dst_mapped = gt_to_pred[gt_dst]
    valid_succ = succ_mask & (gt_src_mapped >= 0) & (gt_dst_mapped >= 0)

    # Build set of positive (pred_src, pred_dst) successor pairs
    positive_pairs = set()
    if valid_succ.any():
        src_m = gt_src_mapped[valid_succ].cpu().tolist()
        dst_m = gt_dst_mapped[valid_succ].cpu().tolist()
        for s, d in zip(src_m, dst_m):
            positive_pairs.add((s, d))

    # Build edge lookup: (src, dst) -> index into candidate edges
    edge_src = edge_type_index[0].cpu().tolist()
    edge_dst = edge_type_index[1].cpu().tolist()
    edge_lookup = {}
    for idx, (s, d) in enumerate(zip(edge_src, edge_dst)):
        edge_lookup[(s, d)] = idx

    # Also build src -> list of candidate edge indices (for hard-neg selection)
    src_to_edges = {}
    for idx, s in enumerate(edge_src):
        src_to_edges.setdefault(s, []).append(idx)

    # Collect training indices and targets
    train_indices = []
    train_targets = []

    matched_pred_set = set(matched_pred_idx.cpu().tolist())

    for pair in positive_pairs:
        src, dst = pair
        if pair in edge_lookup:
            # Positive sample
            train_indices.append(edge_lookup[pair])
            train_targets.append(1.0)

            # Hard negatives: K closest candidates from same source by lateral distance
            if src in src_to_edges and pred_pos is not None:
                cand_indices = src_to_edges[src]
                neg_indices = [ci for ci in cand_indices if (src, edge_dst[ci]) not in positive_pairs]
                if len(neg_indices) > hard_neg_k:
                    # Rank by lateral distance (across) to source — hardest negatives
                    src_pos = pred_pos[src]
                    src_heading = pred_head[src] if pred_head is not None else None
                    dists = []
                    for ci in neg_indices:
                        dst_node = edge_dst[ci]
                        delta = pred_pos[dst_node] - src_pos
                        if src_heading is not None:
                            # Lateral distance = |delta - (delta . heading) * heading|
                            along = (delta * src_heading).sum()
                            lateral = (delta - along * src_heading).norm()
                        else:
                            lateral = delta.norm()
                        dists.append((lateral.item(), ci))
                    dists.sort(key=lambda x: x[0])
                    neg_indices = [ci for _, ci in dists[:hard_neg_k]]

                for ci in neg_indices:
                    train_indices.append(ci)
                    train_targets.append(0.0)

    # Also train candidates from matched nodes without GT successors as negative
    for pred_idx in matched_pred_set:
        # Check if this node has any positive outgoing successor
        has_positive = any((pred_idx, d) in positive_pairs for d in range(num_lanelet_nodes))
        if not has_positive and pred_idx in src_to_edges:
            for ci in src_to_edges[pred_idx]:
                train_indices.append(ci)
                train_targets.append(0.0)

    if len(train_indices) == 0:
        return torch.tensor(0.0, device=device)

    train_indices = torch.tensor(train_indices, dtype=torch.long, device=device)
    train_targets = torch.tensor(train_targets, dtype=torch.float, device=device)

    logits = successor_logits[train_indices]

    # Weighted BCE: pos_weight = n_neg / n_pos, capped at 10
    n_pos = (train_targets > 0.5).sum().float().clamp(min=1)
    n_neg = (train_targets < 0.5).sum().float().clamp(min=1)
    pos_weight = (n_neg / n_pos).clamp(max=10.0)

    loss = F.binary_cross_entropy_with_logits(
        logits, train_targets,
        pos_weight=pos_weight,
        reduction='mean',
    )

    return loss


def lanelet_assign_consistency_loss(
    assignment_matrix: torch.Tensor,
    gt_labels: torch.Tensor,
) -> torch.Tensor:
    """Encourage tracklets of same GT lane to be assigned to same lanelet.

    Uses KL divergence between assignment distributions of tracklets
    in the same GT lane.

    Args:
        assignment_matrix: (N, M) soft assignment.
        gt_labels: (N,) GT lane labels (-1 = unlabeled).

    Returns:
        Scalar KL divergence loss.
    """
    valid = gt_labels >= 0
    if valid.sum() < 2:
        return torch.tensor(0.0, device=assignment_matrix.device)

    valid_S = assignment_matrix[valid]  # (V, M)
    valid_labels = gt_labels[valid]

    unique_lanes = valid_labels.unique()
    if len(unique_lanes) < 1:
        return torch.tensor(0.0, device=assignment_matrix.device)

    total_kl = torch.tensor(0.0, device=assignment_matrix.device)
    n_pairs = 0

    for lane in unique_lanes:
        mask = valid_labels == lane
        if mask.sum() < 2:
            continue
        lane_S = valid_S[mask]  # (K, M) — assignment vectors for this lane's tracklets

        # Mean assignment distribution for this lane
        mean_dist = lane_S.mean(dim=0, keepdim=True)  # (1, M)
        mean_dist = mean_dist.clamp(min=1e-8)
        lane_S_clamped = lane_S.clamp(min=1e-8)

        # KL divergence from each tracklet's distribution to the mean
        kl = (lane_S_clamped * (lane_S_clamped.log() - mean_dist.log())).sum(dim=-1)
        total_kl = total_kl + kl.mean()
        n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=assignment_matrix.device)

    return total_kl / n_pairs


def lanelet_confidence_loss(
    confidence_logits: torch.Tensor,
    pred_confidence: torch.Tensor,
    matched_pred_idx: torch.Tensor,
    num_lanelet_nodes: int,
    match_distances: torch.Tensor = None,
    neg_sample_ratio: float = 2.0,
    soft_target_sigma: float = 5.0,
) -> torch.Tensor:
    """Confidence loss with sampled negatives and soft match-quality targets.

    Matched nodes get target = exp(-dist/sigma), unmatched get 0.
    Only a sampled subset of negatives is used (neg_sample_ratio × n_pos)
    to prevent the "predict 0 everywhere" shortcut.

    Args:
        confidence_logits: (M,) raw logits before sigmoid.
        pred_confidence: (M,) sigmoid confidence (unused, kept for API compat).
        matched_pred_idx: (K,) indices of matched pred nodes.
        num_lanelet_nodes: M.
        match_distances: (K,) L1 distance of each matched pair (for soft targets).
        neg_sample_ratio: how many negatives per positive to sample.
        soft_target_sigma: distance scale for soft targets (meters).

    Returns:
        Scalar BCE loss.
    """
    M = num_lanelet_nodes
    device = confidence_logits.device
    K = len(matched_pred_idx)

    if K == 0:
        return torch.tensor(0.0, device=device)

    # --- Positive targets: soft match quality ---
    if match_distances is not None:
        pos_targets = torch.exp(-match_distances / soft_target_sigma)
    else:
        pos_targets = torch.ones(K, device=device)

    pos_logits = confidence_logits[matched_pred_idx]  # (K,)

    # --- Sampled negatives ---
    matched_set = set(matched_pred_idx.cpu().tolist())
    unmatched = [i for i in range(M) if i not in matched_set]
    n_neg = min(len(unmatched), max(int(K * neg_sample_ratio), 1))

    if n_neg > 0:
        # Shuffle to avoid bias toward low indices
        perm = torch.randperm(len(unmatched), device=device)[:n_neg]
        neg_indices = torch.tensor(unmatched, dtype=torch.long, device=device)[perm]

        neg_logits = confidence_logits[neg_indices]
        neg_targets = torch.zeros(n_neg, device=device)

        # Combine positives and negatives
        all_logits = torch.cat([pos_logits, neg_logits])
        all_targets = torch.cat([pos_targets, neg_targets])
    else:
        all_logits = pos_logits
        all_targets = pos_targets

    return F.binary_cross_entropy_with_logits(all_logits, all_targets)


def drivable_loss(
    drivable_logits: torch.Tensor,
    traj_support: torch.Tensor,
    threshold: float = 0.1,
) -> torch.Tensor:
    """BCE: lanelet is 'drivable' if trajectory support > threshold.

    Args:
        drivable_logits: (M,) raw logits from drivable_head.
        traj_support: (M,) normalized trajectory support per lanelet.
        threshold: support threshold for positive label.

    Returns:
        Scalar BCE loss.
    """
    target = (traj_support > threshold).float()
    return F.binary_cross_entropy_with_logits(drivable_logits, target)


def data_fit_loss(
    S: torch.Tensor,
    centroids: torch.Tensor,
    lanelet_positions: torch.Tensor,
) -> torch.Tensor:
    """Soft assignment-weighted distance: tracklets should be near their lanelet.

    L_fit = sum_i sum_k S[i,k] * ||c_i - p_k|| / N

    Args:
        S: (N, M) soft assignment matrix.
        centroids: (N, 2) tracklet positions in meters.
        lanelet_positions: (M, 2) predicted lanelet positions.

    Returns:
        Scalar data-fit loss.
    """
    # (N, 1, 2) - (1, M, 2) -> (N, M)
    dist = (centroids.unsqueeze(1) - lanelet_positions.unsqueeze(0)).norm(dim=-1)
    weighted_dist = (S * dist).sum(dim=1)  # (N,)
    return weighted_dist.mean()


def separation_loss(
    lanelet_positions: torch.Tensor,
    min_distance: float = 3.5,
) -> torch.Tensor:
    """Penalize lanelet nodes closer than min_distance (one lane width).

    Uses mean over *violating* pairs only, so the gradient isn't diluted
    by the many non-violating (distant) pairs.  For M=16 with only 10
    close pairs out of 240 total, mean-over-all would dilute the signal
    by ~24×.

    Args:
        lanelet_positions: (M, 2) predicted lanelet positions.
        min_distance: minimum desired pairwise distance in meters.

    Returns:
        Scalar mean violation loss (over violating pairs only).
    """
    M = lanelet_positions.shape[0]
    if M < 2:
        return torch.tensor(0.0, device=lanelet_positions.device)

    dist = torch.cdist(lanelet_positions.unsqueeze(0), lanelet_positions.unsqueeze(0)).squeeze(0)  # (M, M)
    mask = ~torch.eye(M, dtype=torch.bool, device=lanelet_positions.device)
    violations = F.relu(min_distance - dist[mask])

    # Mean over violating pairs only — avoids dilution by distant pairs
    violating = violations > 0
    if violating.any():
        return violations[violating].mean()
    return torch.tensor(0.0, device=lanelet_positions.device)


def heading_smoothness_loss(
    pred_headings: torch.Tensor,
    successor_logits: torch.Tensor,
    edge_type_index: torch.Tensor,
) -> torch.Tensor:
    """Penalize heading inconsistency along predicted successor edges.

    For each candidate edge i→j, computes 1 - cos(heading_i, heading_j)
    weighted by the successor probability. This regularizes curvature
    along chains without touching the heading correction range.

    Args:
        pred_headings: (M, 2) predicted unit tangents.
        successor_logits: (E_c,) binary logits for candidate edges.
        edge_type_index: (2, E_c) candidate edge indices.

    Returns:
        Scalar probability-weighted heading smoothness loss.
    """
    E_c = successor_logits.shape[0]
    if E_c == 0:
        return torch.tensor(0.0, device=pred_headings.device)

    edge_src = edge_type_index[0]  # (E_c,)
    edge_dst = edge_type_index[1]  # (E_c,)

    h_src = pred_headings[edge_src]  # (E_c, 2)
    h_dst = pred_headings[edge_dst]  # (E_c, 2)

    # Heading difference: 1 - cos(heading_src, heading_dst)
    cos_sim = (h_src * h_dst).sum(dim=-1)  # (E_c,)
    heading_diff = 1.0 - cos_sim  # (E_c,)

    # Weight by successor probability — focus on edges the model believes in
    succ_prob = torch.sigmoid(successor_logits.detach())  # (E_c,) detach to not backprop through logits
    weighted_loss = (succ_prob * heading_diff).sum() / succ_prob.sum().clamp(min=1e-8)
    return weighted_loss


def collinearity_loss(
    pred_positions: torch.Tensor,
    successor_logits: torch.Tensor,
    edge_type_index: torch.Tensor,
    prob_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize zigzag lane shapes: middle node should lie near the line between predecessor and successor.

    For each triple A→B→C (connected by high-probability successor edges),
    computes perpendicular distance from B to line segment AC.

    Args:
        pred_positions: (M, 2) predicted lanelet positions.
        successor_logits: (E_c,) binary logits for candidate edges.
        edge_type_index: (2, E_c) candidate edge indices.
        prob_threshold: min successor probability for an edge to be considered active.

    Returns:
        Scalar weighted collinearity deviation loss (in meters).
    """
    E_c = successor_logits.shape[0]
    if E_c == 0:
        return torch.tensor(0.0, device=pred_positions.device)

    probs = torch.sigmoid(successor_logits)
    active_mask = probs > prob_threshold
    if not active_mask.any():
        return torch.tensor(0.0, device=pred_positions.device)

    active_edges = edge_type_index[:, active_mask]  # (2, E_active)
    active_probs = probs[active_mask]

    # Build incoming/outgoing maps: node -> list of (neighbor, prob)
    src_list = active_edges[0].cpu().tolist()
    dst_list = active_edges[1].cpu().tolist()
    prob_list = active_probs.cpu().tolist()

    incoming = {}  # dst -> [(src, prob)]
    outgoing = {}  # src -> [(dst, prob)]
    for s, d, p in zip(src_list, dst_list, prob_list):
        incoming.setdefault(d, []).append((s, p))
        outgoing.setdefault(s, []).append((d, p))

    # Find triples A→B→C: B appears as both dst (incoming) and src (outgoing)
    middle_nodes = set(incoming.keys()) & set(outgoing.keys())

    deviations = []
    weights = []

    for B in middle_nodes:
        A, p_ab = max(incoming[B], key=lambda x: x[1])
        C, p_bc = max(outgoing[B], key=lambda x: x[1])

        pos_A = pred_positions[A]
        pos_B = pred_positions[B]
        pos_C = pred_positions[C]

        AC = pos_C - pos_A
        AB = pos_B - pos_A
        ac_len_sq = (AC * AC).sum()
        if ac_len_sq < 1e-6:
            continue

        t = (AB * AC).sum() / ac_len_sq
        t = t.clamp(0.0, 1.0)
        projection = pos_A + t * AC
        deviation = (pos_B - projection).norm()

        deviations.append(deviation)
        weights.append(min(p_ab, p_bc))

    if len(deviations) == 0:
        return torch.tensor(0.0, device=pred_positions.device)

    dev_tensor = torch.stack(deviations)
    weight_tensor = torch.tensor(weights, device=pred_positions.device, dtype=dev_tensor.dtype)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp(min=1e-8)

    return (dev_tensor * weight_tensor).sum()


def lane_classification_loss(
    assign_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    matched_pred_idx: torch.Tensor,
    matched_gt_idx: torch.Tensor,
    gt_lanelet_lane_ids: torch.Tensor,
    tracklet_centroids: torch.Tensor,
    gt_positions: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy lane classification on assignment logits (Stage 1).

    Per-tracklet targeting: each tracklet is supervised to go to the slot
    matched to the nearest GT waypoint of its lane. This preserves along-lane
    spatial diversity (tracklets at different positions along the same lane
    get different target slots).

    Args:
        assign_logits: (N, M) raw logits before softmax.
        gt_labels: (N,) tracklet lane labels (-1 = unlabeled).
        matched_pred_idx: (K,) matched pred slot indices from Hungarian matching.
        matched_gt_idx: (K,) matched GT waypoint indices from Hungarian matching.
        gt_lanelet_lane_ids: (G,) GT waypoint lane IDs (same indexing as gt_labels).
        tracklet_centroids: (N, 2) tracklet positions in meters.
        gt_positions: (G, 2) GT waypoint positions in meters.

    Returns:
        Scalar cross-entropy loss.
    """
    device = assign_logits.device

    if len(matched_pred_idx) == 0 or (gt_labels >= 0).sum() < 2:
        return torch.tensor(0.0, device=device)

    # Build GT waypoint index -> matched slot mapping
    gt_to_slot = {}  # gt_waypoint_idx -> pred_slot_idx
    for pred_idx, gt_idx in zip(matched_pred_idx.cpu().tolist(), matched_gt_idx.cpu().tolist()):
        if gt_idx < len(gt_lanelet_lane_ids):
            gt_to_slot[gt_idx] = pred_idx

    if len(gt_to_slot) == 0:
        return torch.tensor(0.0, device=device)

    # Group matched GT waypoints by lane
    lane_matched_gts: dict = {}  # lane_id -> [(gt_idx, slot_idx)]
    for gt_idx, slot_idx in gt_to_slot.items():
        lane_id = gt_lanelet_lane_ids[gt_idx].item()
        if lane_id not in lane_matched_gts:
            lane_matched_gts[lane_id] = []
        lane_matched_gts[lane_id].append((gt_idx, slot_idx))

    if len(lane_matched_gts) == 0:
        return torch.tensor(0.0, device=device)

    # For each labeled tracklet, find its nearest matched GT waypoint
    # within its lane, and use that waypoint's slot as the target
    centroids_np = tracklet_centroids.detach().cpu().numpy()
    gt_pos_np = gt_positions.detach().cpu().numpy()

    valid_indices = []
    targets = []

    labeled_mask = gt_labels >= 0
    for idx in torch.where(labeled_mask)[0].cpu().tolist():
        lane_id = gt_labels[idx].item()
        if lane_id not in lane_matched_gts:
            continue

        # Find nearest matched GT waypoint of this lane to this tracklet
        best_slot = None
        best_dist = float("inf")
        tc = centroids_np[idx]
        for gt_idx, slot_idx in lane_matched_gts[lane_id]:
            d = float(((tc - gt_pos_np[gt_idx]) ** 2).sum() ** 0.5)
            if d < best_dist:
                best_dist = d
                best_slot = slot_idx

        if best_slot is not None:
            valid_indices.append(idx)
            targets.append(best_slot)

    if len(valid_indices) == 0:
        return torch.tensor(0.0, device=device)

    valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=device)
    target_t = torch.tensor(targets, dtype=torch.long, device=device)

    return F.cross_entropy(assign_logits[valid_idx_t], target_t)


def sumo_topology_prior_loss(
    edge_type_logits: torch.Tensor,
    edge_type_index: torch.Tensor,
    sumo_adj_types: torch.Tensor,
    gate: torch.Tensor,
    matched_pred_idx: torch.Tensor,
    matched_gt_idx: torch.Tensor,
    num_gt_nodes: int,
) -> torch.Tensor:
    """Soft SUMO topology prior: encourage predicted edges to match SUMO when gated on.

    Maps predicted edge endpoints through Hungarian matching to look up
    SUMO GT edge types as soft targets.

    Args:
        edge_type_logits: (E_c, num_types) predicted edge type logits.
        edge_type_index: (2, E_c) predicted edge indices (in pred space).
        sumo_adj_types: (G, G) SUMO adjacency type matrix (in GT space).
        gate: scalar [0, 1] evidence gate.
        matched_pred_idx: (K,) matched predicted node indices.
        matched_gt_idx: (K,) matched GT node indices.
        num_gt_nodes: G.

    Returns:
        Scalar gated cross-entropy loss.
    """
    if edge_type_logits.shape[0] == 0 or len(matched_pred_idx) == 0:
        return torch.tensor(0.0, device=edge_type_logits.device)

    device = edge_type_logits.device

    # Build pred->GT node mapping (size must cover both matched indices AND edge indices)
    map_size = max(edge_type_index.max().item(), matched_pred_idx.max().item()) + 1
    pred_to_gt = torch.full(
        (map_size,), -1, dtype=torch.long, device=device
    )
    pred_to_gt[matched_pred_idx] = matched_gt_idx

    # Map predicted edge endpoints to GT space
    pred_src = edge_type_index[0]
    pred_dst = edge_type_index[1]
    gt_src = pred_to_gt[pred_src]
    gt_dst = pred_to_gt[pred_dst]

    # Only use edges where both endpoints are matched to GT
    valid = (gt_src >= 0) & (gt_dst >= 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    sumo_adj_types = sumo_adj_types.to(device)
    gt_src_valid = gt_src[valid]
    gt_dst_valid = gt_dst[valid]

    # Bounds check: GT indices must be within sumo_adj_types dimensions
    G_adj = sumo_adj_types.shape[0]
    in_bounds = (gt_src_valid < G_adj) & (gt_dst_valid < G_adj)
    if in_bounds.sum() == 0:
        return torch.tensor(0.0, device=device)
    gt_src_valid = gt_src_valid[in_bounds]
    gt_dst_valid = gt_dst_valid[in_bounds]

    sumo_targets = sumo_adj_types[gt_src_valid, gt_dst_valid].long()
    num_classes = edge_type_logits.shape[1]
    sumo_targets = sumo_targets.clamp(0, num_classes - 1)
    valid_logits = edge_type_logits[valid][in_bounds]

    ce = F.cross_entropy(valid_logits, sumo_targets, reduction='mean')
    return gate * ce


def compute_total_loss(
    model_output: dict,
    data,
    config: dict,
    gt_labels: torch.Tensor = None,
    epoch: int = 0,
) -> tuple:
    """Compute total lanelet loss from all components.

    Args:
        model_output: dict from LaneReprModel.forward() (v3 lanelet output).
        data: PyG Data with GT lanelet attributes.
        config: full config dict.
        gt_labels: (N,) optional GT lane labels for assignment consistency.
        epoch: current epoch (unused, kept for API compat).

    Returns:
        total_loss: scalar.
        loss_dict: dict of individual loss component values.
    """
    train_cfg = config.get("training", {})
    device = model_output["lanelet_positions"].device

    # Phase-based weight scheduling ----------------------------------------
    def _phase_weight(key: str, base: float) -> float:
        """Return weight for `key` from phase_schedule if present, else base."""
        phases = train_cfg.get("phase_schedule", None)
        if phases is None:
            return base
        for phase in sorted(phases, key=lambda p: p["start_epoch"], reverse=True):
            if epoch >= phase["start_epoch"]:
                return phase.get("weights", {}).get(key, base)
        return base

    # Phase transition logging (once per phase change)
    phases = train_cfg.get("phase_schedule", None)
    if phases is not None:
        sorted_phases = sorted(phases, key=lambda p: p["start_epoch"])
        current_phase = 0
        for i, phase in enumerate(sorted_phases):
            if epoch >= phase["start_epoch"]:
                current_phase = i + 1
        last_phase = getattr(compute_total_loss, "_last_phase", -1)
        if current_phase != last_phase:
            compute_total_loss._last_phase = current_phase
            if current_phase > 0:
                phase_info = sorted_phases[current_phase - 1]
                active = {k: v for k, v in phase_info.get("weights", {}).items() if v > 0}
                logger.info(
                    f"=== Phase {current_phase}/{len(sorted_phases)} "
                    f"(epoch {epoch}, start={phase_info['start_epoch']}) === "
                    f"Active losses: {list(active.keys())}"
                )
    # ----------------------------------------------------------------------

    loss_dict = {}

    pred_pos = model_output["lanelet_positions"]        # (M, 2)
    pred_head = model_output["lanelet_headings"]         # (M, 2)
    pred_conf = model_output["lanelet_confidence"]       # (M,)
    S = model_output["assignment_matrix"]                # (N, M)
    succ_logits = model_output["successor_logits"]         # (E_c,)
    edge_index = model_output["edge_type_index"]         # (2, E_c)
    entropy_reg = model_output["pooling_entropy_loss"]   # scalar
    balancing_reg = model_output["pooling_balancing_loss"]  # scalar
    link_reg = model_output["pooling_link_loss"]         # scalar

    M = pred_pos.shape[0]

    # Anchor gradient flow: ensures total has grad_fn even if no GT loss fires
    total = pred_conf.sum() * 0.0

    # GT lanelet graph
    gt_pos = getattr(data, "gt_lanelet_positions", None)
    gt_tan = getattr(data, "gt_lanelet_tangents", None)
    gt_edge_idx = getattr(data, "gt_lanelet_edge_index", None)
    gt_edge_types = getattr(data, "gt_lanelet_edge_types", None)

    has_gt = gt_pos is not None and len(gt_pos) > 0
    G = len(gt_pos) if has_gt else 0

    # 1. Lanelet node position loss (Hungarian matching)
    w = _phase_weight("lanelet_position_weight", train_cfg.get("lanelet_position_weight", 5.0))
    if w > 0 and has_gt:
        pos_loss, matched_pred, matched_gt, match_stats = lanelet_node_loss(pred_pos, gt_pos, pred_conf)
        loss_dict["lanelet_position"] = pos_loss.item()
        loss_dict["lanelet_n_matched"] = len(matched_pred)
        loss_dict["lanelet_n_gt"] = G
        loss_dict["match_mean_dist"] = match_stats["mean_match_dist"]
        loss_dict["match_median_min_gt_dist"] = match_stats["median_min_gt_dist"]
        loss_dict["pred_range"] = match_stats["pred_range"]
        loss_dict["gt_range"] = match_stats["gt_range"]
        total = total + w * pos_loss
    else:
        pos_loss = torch.tensor(0.0, device=device)
        matched_pred = torch.zeros(0, dtype=torch.long)
        matched_gt = torch.zeros(0, dtype=torch.long)
        match_stats = {}
        loss_dict["lanelet_position"] = 0.0
        loss_dict["lanelet_n_matched"] = 0
        loss_dict["lanelet_n_gt"] = G

    # 1b. Coverage loss (GT→pred direction: penalize uncovered GT waypoints)
    w = _phase_weight("coverage_weight", train_cfg.get("coverage_weight", 0.0))
    coverage_thresh = train_cfg.get("coverage_threshold", train_cfg.get("coverage_threshold_m", 8.0))
    if w > 0 and has_gt:
        cov_loss = lanelet_coverage_loss(pred_pos, gt_pos, threshold=coverage_thresh)
        loss_dict["coverage"] = cov_loss.item()
        total = total + w * cov_loss
    else:
        loss_dict["coverage"] = 0.0

    # 2. Heading loss
    w = _phase_weight("lanelet_heading_weight", train_cfg.get("lanelet_heading_weight", 2.0))
    if w > 0 and has_gt and gt_tan is not None:
        h_loss = lanelet_heading_loss(pred_head, gt_tan, matched_pred, matched_gt)
        loss_dict["lanelet_heading"] = h_loss.item()
        total = total + w * h_loss
    else:
        loss_dict["lanelet_heading"] = 0.0

    # 3. Binary successor loss (replaces 5-class edge type loss)
    w = _phase_weight("lanelet_edge_weight", train_cfg.get("lanelet_edge_weight", 3.0))
    hard_neg_k = train_cfg.get("hard_neg_k", 4)
    if w > 0 and has_gt and gt_edge_idx is not None and len(gt_edge_types) > 0:
        e_loss = binary_successor_loss(
            succ_logits, edge_index, gt_edge_idx, gt_edge_types,
            matched_pred, matched_gt, M, G,
            pred_pos=pred_pos, pred_head=pred_head,
            hard_neg_k=hard_neg_k,
        )
        loss_dict["lanelet_edge"] = e_loss.item()
        total = total + w * e_loss
    else:
        loss_dict["lanelet_edge"] = 0.0

    # 4. Confidence loss — sampled negatives + soft match-quality targets
    conf_logits = model_output["confidence_logits"]  # (M,) raw logits
    w = _phase_weight("lanelet_confidence_weight", train_cfg.get("lanelet_confidence_weight", 1.0))
    conf_min_matches = train_cfg.get("confidence_min_matches", 3)
    if w > 0 and len(matched_pred) >= conf_min_matches:
        # Get match distances for soft targets
        match_dists_np = match_stats.get("match_distances", None) if has_gt else None
        match_dists_t = None
        if match_dists_np is not None and len(match_dists_np) > 0:
            match_dists_t = torch.tensor(match_dists_np, dtype=torch.float, device=device)

        neg_sample_ratio = train_cfg.get("conf_neg_sample_ratio", 2.0)
        soft_sigma = train_cfg.get("conf_soft_target_sigma", 0.02)

        c_loss = lanelet_confidence_loss(
            conf_logits, pred_conf, matched_pred, M,
            match_distances=match_dists_t,
            neg_sample_ratio=neg_sample_ratio,
            soft_target_sigma=soft_sigma,
        )
        loss_dict["lanelet_confidence"] = c_loss.item()
        total = total + w * c_loss
    else:
        loss_dict["lanelet_confidence"] = 0.0

    # 5. Pooling entropy regularization (encourage sharp assignment)
    w = _phase_weight("pooling_entropy_weight", train_cfg.get("pooling_entropy_weight", 0.1))
    if w > 0:
        total = total + w * entropy_reg
        loss_dict["pooling_entropy"] = entropy_reg.item()
    else:
        loss_dict["pooling_entropy"] = 0.0

    # 5b. Balancing loss (prevent slot collapse — uniform utilization)
    w = _phase_weight("balancing_weight", train_cfg.get("balancing_weight", 1.0))
    if w > 0:
        total = total + w * balancing_reg
        loss_dict["balancing"] = balancing_reg.item()
    else:
        loss_dict["balancing"] = 0.0

    # 6. Pooling link regularization (clean clustering)
    w = _phase_weight("pooling_link_weight", train_cfg.get("pooling_link_weight", 0.1))
    if w > 0:
        total = total + w * link_reg
        loss_dict["pooling_link"] = link_reg.item()
    else:
        loss_dict["pooling_link"] = 0.0

    # 7. Assignment consistency (confidence-gated, low weight)
    w = _phase_weight("assign_consistency_weight", train_cfg.get("assign_consistency_weight", 0.3))
    if w > 0 and gt_labels is not None:
        a_loss = lanelet_assign_consistency_loss(S, gt_labels)
        gate = pred_conf.mean().detach().clamp(0.0, 1.0)
        total = total + w * gate * a_loss
        loss_dict["assign_consistency"] = a_loss.item()
    else:
        loss_dict["assign_consistency"] = 0.0

    # 8. Drivable probability loss
    w = _phase_weight("drivable_weight", train_cfg.get("drivable_weight", 1.0))
    if w > 0 and "drivable_logits" in model_output:
        d_loss = drivable_loss(
            model_output["drivable_logits"], model_output["traj_support"]
        )
        total = total + w * d_loss
        loss_dict["drivable"] = d_loss.item()
    else:
        loss_dict["drivable"] = 0.0

    # 9. Data fit (tracklets explained by lanelets)
    w = _phase_weight("data_fit_weight", train_cfg.get("data_fit_weight", 2.0))
    if w > 0 and hasattr(data, "centroids"):
        fit_loss = data_fit_loss(S, data.centroids, pred_pos)
        total = total + w * fit_loss
        loss_dict["data_fit"] = fit_loss.item()
    else:
        loss_dict["data_fit"] = 0.0

    # 10. SUMO topology prior — disabled for binary successor (was 5-class)
    loss_dict["sumo_topology"] = 0.0

    # 11. Separation loss (scatter lanelet nodes apart)
    w = _phase_weight("separation_weight", train_cfg.get("separation_weight", 0.0))
    if w > 0:
        min_dist = config.get("data", {}).get("fixed_lane_width", config.get("data", {}).get("fixed_lane_width_m", 3.5))
        sep_loss = separation_loss(pred_pos, min_distance=min_dist)
        total = total + w * sep_loss
        loss_dict["separation"] = sep_loss.item()
    else:
        loss_dict["separation"] = 0.0

    # 12. Heading smoothness loss (curvature regularization on successor chains)
    w = _phase_weight("heading_smoothness_weight", train_cfg.get("heading_smoothness_weight", 0.0))
    if w > 0 and succ_logits.shape[0] > 0:
        hs_loss = heading_smoothness_loss(pred_head, succ_logits, edge_index)
        total = total + w * hs_loss
        loss_dict["heading_smoothness"] = hs_loss.item()
    else:
        loss_dict["heading_smoothness"] = 0.0

    # 13. Collinearity loss (lane shape regularization: penalize zigzag)
    w = _phase_weight("collinearity_weight", train_cfg.get("collinearity_weight", 0.0))
    if w > 0 and succ_logits.shape[0] > 0:
        col_loss = collinearity_loss(pred_pos, succ_logits, edge_index)
        total = total + w * col_loss
        loss_dict["collinearity"] = col_loss.item()
    else:
        loss_dict["collinearity"] = 0.0

    # 14. Lane classification loss (cross-entropy on assignment logits, with warmup)
    w = _phase_weight("lane_cls_weight", train_cfg.get("lane_cls_weight", 0.0))
    lane_cls_warmup = train_cfg.get("lane_cls_warmup_epoch", 0)
    if epoch < lane_cls_warmup:
        w = 0.0
    gt_lane_ids = getattr(data, "gt_lanelet_lane_ids", None)
    if w > 0 and gt_labels is not None and gt_lane_ids is not None and has_gt and "assign_logits" in model_output:
        cls_loss = lane_classification_loss(
            model_output["assign_logits"], gt_labels,
            matched_pred, matched_gt, gt_lane_ids,
            data.centroids, gt_pos,
        )
        total = total + w * cls_loss
        loss_dict["lane_cls"] = cls_loss.item()
    else:
        loss_dict["lane_cls"] = 0.0

    loss_dict["total"] = total.item()
    return total, loss_dict
