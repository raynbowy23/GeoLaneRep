"""Evaluator: lanelet-specific metrics for v3 lanelet discovery.

Metrics:
  - Node Position Error (NPE): mean L2 between matched pred/GT nodes.
  - Heading Angular Error (HAE): mean angle error in degrees.
  - Edge Type Accuracy: weighted accuracy on edge types.
  - Lane Topology F1: per-type F1 for successor/merge/diverge/adjacent.
  - Assignment ARI: backward-compatible tracklet->lane from S matrix.
  - Lane Recall / Precision: GT lane recalled if >=2 pred nodes within 5m.
"""

import logging

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

logger = logging.getLogger(__name__)


def compute_lanelet_metrics(
    model_output: dict,
    data,
    gt_labels: torch.Tensor = None,
    confidence_threshold: float = 0.3,
    match_radius_m: float = 0.02,
) -> dict:
    """Compute all lanelet evaluation metrics.

    Args:
        model_output: dict from LaneReprModel.forward() (v3).
        data: PyG Data with GT lanelet attributes.
        gt_labels: (N,) GT lane labels for ARI.
        confidence_threshold: min confidence for active lanelet nodes.
        match_radius_m: distance threshold for lane recall/precision.

    Returns:
        dict of metric name -> float value.
    """
    metrics = {}

    pred_pos = model_output["lanelet_positions"].detach().cpu()      # (M, 2)
    pred_head = model_output["lanelet_headings"].detach().cpu()       # (M, 2)
    pred_conf = model_output["lanelet_confidence"].detach().cpu()     # (M,)
    S = model_output["assignment_matrix"].detach().cpu()              # (N, M)
    succ_logits = model_output["successor_logits"].detach().cpu()      # (E_c,)
    edge_index = model_output["edge_type_index"].detach().cpu()       # (2, E_c)

    M = pred_pos.shape[0]
    N = S.shape[0]

    gt_pos = getattr(data, "gt_lanelet_positions", None)
    gt_tan = getattr(data, "gt_lanelet_tangents", None)
    gt_lane_ids = getattr(data, "gt_lanelet_lane_ids", None)
    gt_edge_idx = getattr(data, "gt_lanelet_edge_index", None)
    gt_edge_types = getattr(data, "gt_lanelet_edge_types", None)

    # Move GT tensors to CPU for metric computation
    if gt_pos is not None:
        gt_pos = gt_pos.cpu()
    if gt_tan is not None:
        gt_tan = gt_tan.cpu()
    if gt_lane_ids is not None:
        gt_lane_ids = gt_lane_ids.cpu()
    if gt_edge_idx is not None:
        gt_edge_idx = gt_edge_idx.cpu()
    if gt_edge_types is not None:
        gt_edge_types = gt_edge_types.cpu()

    has_gt = gt_pos is not None and len(gt_pos) > 0
    G = len(gt_pos) if has_gt else 0

    # Active lanelet nodes
    active_mask = pred_conf > confidence_threshold
    n_active = int(active_mask.sum().item())
    metrics["n_active_lanelet_nodes"] = n_active
    metrics["n_gt_lanelet_nodes"] = G

    # Confidence diagnostics — detect collapse early
    metrics["conf_mean"] = float(pred_conf.mean().item())
    metrics["conf_max"] = float(pred_conf.max().item())
    metrics["conf_above_0.5"] = int((pred_conf > 0.5).sum().item())
    metrics["conf_above_0.2"] = int((pred_conf > 0.2).sum().item())
    metrics["conf_above_0.1"] = int((pred_conf > 0.1).sum().item())

    # --- Shared matching: only match active pred nodes to GT ---
    active_indices = torch.where(active_mask)[0]  # indices of active pred nodes
    matched_row = np.array([], dtype=int)   # indices into active pred
    matched_col = np.array([], dtype=int)   # indices into GT
    matched_cost = np.array([], dtype=float)
    active_to_full = active_indices.numpy()  # map active index -> full pred index

    if has_gt and n_active > 0:
        active_pos = pred_pos[active_indices]  # (n_active, 2)
        cost = torch.cdist(active_pos, gt_pos, p=2).numpy()  # (n_active, G)
        row_ind, col_ind = linear_sum_assignment(cost)
        valid = cost[row_ind, col_ind] < 0.20
        matched_row = row_ind[valid]
        matched_col = col_ind[valid]
        matched_cost = cost[matched_row, matched_col] if len(matched_row) > 0 else np.array([])

    # --- 1. Node Position Error (NPE) ---
    if len(matched_cost) > 0:
        metrics["node_position_error"] = float(np.mean(matched_cost))
    else:
        metrics["node_position_error"] = float("nan")

    # --- 2. Heading Angular Error (HAE) ---
    if len(matched_row) > 0 and gt_tan is not None:
        pred_h = pred_head[torch.tensor(active_to_full[matched_row])].numpy()
        gt_h = gt_tan[torch.tensor(matched_col)].numpy()
        cos_sim = np.clip(np.sum(pred_h * gt_h, axis=1), -1, 1)
        angles_deg = np.degrees(np.arccos(np.abs(cos_sim)))
        metrics["heading_angular_error"] = float(np.mean(angles_deg))
    else:
        metrics["heading_angular_error"] = float("nan")

    # --- 3. Binary Successor F1 (set intersection of pred vs GT pairs) ---
    if has_gt and succ_logits.shape[0] > 0 and gt_edge_idx is not None and gt_edge_types is not None:
        # Map GT nodes to pred (full index space) through matching
        gt_to_pred = torch.full((G,), -1, dtype=torch.long)
        for r, c in zip(matched_row, matched_col):
            gt_to_pred[c] = int(active_to_full[r])

        # GT successor pairs (type == 1) with both endpoints matched
        succ_mask = gt_edge_types == 1
        gt_src_mapped = gt_to_pred[gt_edge_idx[0]]
        gt_dst_mapped = gt_to_pred[gt_edge_idx[1]]
        valid_succ = succ_mask & (gt_src_mapped >= 0) & (gt_dst_mapped >= 0)

        gt_succ_pairs = set()
        if valid_succ.any():
            src_m = gt_src_mapped[valid_succ].numpy()
            dst_m = gt_dst_mapped[valid_succ].numpy()
            for s, d in zip(src_m, dst_m):
                gt_succ_pairs.add((int(s), int(d)))

        # Predicted successor pairs: sigmoid > 0.5
        pred_succ_pairs = set()
        if succ_logits.shape[0] > 0:
            pred_mask = torch.sigmoid(succ_logits) > 0.5
            pred_src = edge_index[0].numpy()
            pred_dst = edge_index[1].numpy()
            for i in range(len(pred_src)):
                if pred_mask[i]:
                    pred_succ_pairs.add((int(pred_src[i]), int(pred_dst[i])))

        # Set intersection for F1
        tp = len(gt_succ_pairs & pred_succ_pairs)
        fp = len(pred_succ_pairs - gt_succ_pairs)
        fn = len(gt_succ_pairs - pred_succ_pairs)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        metrics["edge_f1_successor"] = float(f1)
        metrics["edge_precision_successor"] = float(precision)
        metrics["edge_recall_successor"] = float(recall)
        metrics["lane_topology_f1"] = float(f1)  # successor F1 only
        metrics["n_pred_successor_edges"] = len(pred_succ_pairs)
        metrics["n_gt_successor_edges"] = len(gt_succ_pairs)
    else:
        metrics["edge_f1_successor"] = float("nan")
        metrics["edge_precision_successor"] = float("nan")
        metrics["edge_recall_successor"] = float("nan")
        metrics["lane_topology_f1"] = float("nan")
        metrics["n_pred_successor_edges"] = 0
        metrics["n_gt_successor_edges"] = 0

    # --- 4. Assignment ARI (backward-compatible with v2) ---
    if gt_labels is not None:
        valid_gt = gt_labels >= 0
        if valid_gt.sum() >= 5:
            # Derive tracklet->lane assignment from S matrix
            pred_assign = S.argmax(dim=1).cpu().numpy()  # (N,)
            gt_np = gt_labels.cpu().numpy()
            valid_mask = gt_np >= 0
            try:
                metrics["assignment_ari"] = float(adjusted_rand_score(
                    gt_np[valid_mask], pred_assign[valid_mask]))
            except Exception:
                metrics["assignment_ari"] = float("nan")
        else:
            metrics["assignment_ari"] = float("nan")
    else:
        metrics["assignment_ari"] = float("nan")

    # --- 4b. Lane classification accuracy (Stage 1 metric) ---
    # Per-tracklet: each tracklet should be assigned to the slot matched to
    # the nearest GT waypoint of its lane (preserves along-lane diversity)
    centroids_data = getattr(data, "centroids", None)
    if (gt_labels is not None and gt_lane_ids is not None and has_gt
            and len(matched_row) > 0 and centroids_data is not None):
        centroids_np = centroids_data.cpu().numpy()  # (N, 2)
        gt_pos_np_cls = gt_pos.cpu().numpy()  # (G, 2)

        # Build: lane_id -> [(gt_idx, slot_idx)] from matching
        lane_matched_gts = {}
        for r, c in zip(matched_row, matched_col):
            full_pred_idx = int(active_to_full[r])
            if c < len(gt_lane_ids):
                lane_id = int(gt_lane_ids[c].item())
                if lane_id not in lane_matched_gts:
                    lane_matched_gts[lane_id] = []
                lane_matched_gts[lane_id].append((int(c), full_pred_idx))

        if lane_matched_gts:
            pred_assign_np = S.argmax(dim=1).cpu().numpy()  # (N,)
            gt_labels_np = gt_labels.cpu().numpy()
            correct = 0
            total_labeled = 0
            for i in range(N):
                lid = int(gt_labels_np[i])
                if lid < 0 or lid not in lane_matched_gts:
                    continue
                # Find nearest matched GT waypoint of this lane
                tc = centroids_np[i]
                best_slot = None
                best_dist = float("inf")
                for gt_idx, slot_idx in lane_matched_gts[lid]:
                    d = float(np.linalg.norm(tc - gt_pos_np_cls[gt_idx]))
                    if d < best_dist:
                        best_dist = d
                        best_slot = slot_idx
                if best_slot is not None:
                    total_labeled += 1
                    if pred_assign_np[i] == best_slot:
                        correct += 1
            metrics["lane_cls_acc"] = float(correct) / max(total_labeled, 1)
        else:
            metrics["lane_cls_acc"] = float("nan")
    else:
        metrics["lane_cls_acc"] = float("nan")

    # --- 5. Lane Recall / Precision ---
    if has_gt and gt_lane_ids is not None and n_active > 0:
        active_pred_pos = pred_pos[active_mask].cpu().numpy()  # (A, 2)
        gt_pos_np = gt_pos.cpu().numpy()  # (G, 2)
        gt_lid = gt_lane_ids.cpu().numpy()  # (G,)

        unique_gt_lanes = np.unique(gt_lid)
        n_gt_lanes = len(unique_gt_lanes)

        # Lane Recall: GT lane recalled if >= 2 pred nodes within match_radius
        recalled = 0
        for lid in unique_gt_lanes:
            lane_pts = gt_pos_np[gt_lid == lid]  # GT waypoints for this lane
            # Check how many active pred nodes are near any waypoint of this lane
            if len(active_pred_pos) == 0:
                continue
            dists = np.linalg.norm(
                active_pred_pos[:, None, :] - lane_pts[None, :, :], axis=2
            )  # (A, P)
            near_any = dists.min(axis=1) < match_radius_m  # (A,)
            if near_any.sum() >= 2:
                recalled += 1

        metrics["lane_recall"] = float(recalled) / max(n_gt_lanes, 1)

        # Lane Precision: pred "lane" (group of connected active nodes with same S assignment)
        # Simplified: count predicted active nodes that are near any GT node
        if len(active_pred_pos) > 0:
            dists_all = np.linalg.norm(
                active_pred_pos[:, None, :] - gt_pos_np[None, :, :], axis=2
            )  # (A, G)
            near_gt = dists_all.min(axis=1) < match_radius_m
            metrics["lane_precision"] = float(near_gt.sum()) / max(len(active_pred_pos), 1)
        else:
            metrics["lane_precision"] = 0.0
    else:
        metrics["lane_recall"] = float("nan")
        metrics["lane_precision"] = float("nan")

    # --- Slot utilization (how many of M nodes are active) ---
    metrics["slot_utilization"] = n_active / M if M > 0 else 0.0

    # --- 6. Deterministic successor chain metrics ---
    from src.utils.successor_linking import (
        deterministic_successor_linking,
        compute_chain_metrics,
    )
    chains, _ = deterministic_successor_linking(
        pred_pos.numpy(), pred_head.numpy(), pred_conf.numpy(),
        conf_threshold=confidence_threshold,
    )
    gt_pos_np = gt_pos.numpy() if has_gt else None
    gt_lid_np = gt_lane_ids.numpy() if gt_lane_ids is not None else None
    chain_m = compute_chain_metrics(
        chains, pred_pos.numpy(),
        gt_positions=gt_pos_np,
        gt_lane_ids=gt_lid_np,
        match_radius=match_radius_m,
    )
    for k, v in chain_m.items():
        metrics[f"chain_{k}"] = v

    return metrics
