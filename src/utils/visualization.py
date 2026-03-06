"""Visualization for v3 lanelet graph discovery."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def clip_to_convex_hull(positions: np.ndarray, centroids: np.ndarray, margin: float = 5.0) -> np.ndarray:
    """Clip positions to the convex hull of centroids (expanded by margin).

    Args:
        positions: (M, 2) points to clip.
        centroids: (N, 2) reference points defining the hull.
        margin: expansion in meters around the hull.

    Returns:
        (M, 2) clipped positions.
    """
    from scipy.spatial import ConvexHull, Delaunay

    if len(centroids) < 3 or len(positions) == 0:
        return positions

    # Expand hull outward by margin
    hull_center = centroids.mean(axis=0)
    dirs = centroids - hull_center
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    expanded = centroids + dirs / norms * margin

    try:
        hull = ConvexHull(expanded)
        delaunay = Delaunay(expanded[hull.vertices])
    except Exception:
        return positions

    clipped = positions.copy()
    outside = delaunay.find_simplex(positions) < 0

    for i in np.where(outside)[0]:
        # Project to nearest point on hull boundary
        best_dist = float("inf")
        best_pt = positions[i]
        for simplex in hull.simplices:
            a, b = expanded[simplex[0]], expanded[simplex[1]]
            ab = b - a
            t = np.dot(positions[i] - a, ab) / (np.dot(ab, ab) + 1e-8)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
            dist = np.linalg.norm(positions[i] - proj)
            if dist < best_dist:
                best_dist = dist
                best_pt = proj
        clipped[i] = best_pt

    return clipped

# Vivid, distinct colors (BGR) — matches fedmeta_geolane palette for consistency.
SLOT_COLORS = [
    (  0,   0, 255),  # vivid red
    (189, 115,   0),  # strong blue
    ( 48, 171, 120),  # vivid green
    ( 33, 176, 237),  # strong orange
    (143,  46, 125),  # purple
    (191, 191,   0),  # cyan
    ( 25,  84, 217),  # deep orange
    (  0, 128,   0),  # forest green
    (191,   0, 191),  # magenta
    (171, 179,  33),  # teal
    (  0, 181, 140),  # lime green
    ( 46,  20, 163),  # wine red
    (  0, 102, 204),  # burnt orange
    (179, 120,  31),  # deep sky blue
    (212,   0, 148),  # violet
    ( 79,  43, 237),  # cherry red
    (201, 161,  51),  # bright turquoise
    (143, 237, 143),  # pastel green
    (  0, 153, 255),  # vivid amber
    ( 64,  64,  64),  # dark gray
]


def visualize_lanelet_graph(
    frame: np.ndarray,
    model_output: dict,
    data,
    conf_threshold: float = 0.3,
    save_path: Optional[str] = None,
    color_offset: int = 0,
    top_k: int = 0,
) -> np.ndarray:
    """Draw predicted lanelet graph on camera frame.

    Shows:
      - Active lanelet nodes as large colored circles.
      - Heading arrows at each lanelet node.
      - Successor chains as thick colored polylines.
      - Non-successor edges (merge/diverge/adjacent) as thin lines.

    Args:
        frame: (H, W, 3) BGR camera frame.
        model_output: dict from LaneReprModel.forward() (v3 output).
        data: PyG Data with centroids, tangents.
        conf_threshold: min confidence for active lanelet nodes.
        save_path: optional file path to save.

    Returns:
        (H, W, 3) annotated frame.
    """
    vis = frame.copy()

    pred_pos = model_output["lanelet_positions"].detach().cpu().numpy()    # (M, 2)
    pred_head = model_output["lanelet_headings"].detach().cpu().numpy()    # (M, 2)
    pred_conf = model_output["lanelet_confidence"].detach().cpu().numpy()  # (M,)
    S = model_output["assignment_matrix"].detach().cpu().numpy()           # (N, M)
    succ_logits = model_output["successor_logits"].detach()                 # (E_c,)
    edge_index = model_output["edge_type_index"].detach().cpu().numpy()    # (2, E_c)

    M = pred_pos.shape[0]

    # Inverse (s,d) -> global meters for pixel conversion
    R_inv = None
    origin = None
    if hasattr(data, 'sd_heading') and data.sd_heading is not None:
        h = float(data.sd_heading)
        cos_h, sin_h = np.cos(h), np.sin(h)
        R_inv = np.array([[cos_h, -sin_h], [sin_h, cos_h]])  # forward rotation
        origin = data.sd_origin.cpu().numpy()  # (2,)
        pred_pos = (pred_pos @ R_inv.T) + origin   # (M, 2) back to global meters
        pred_head = pred_head @ R_inv.T             # (M, 2) rotate headings back

    # Clip lanelet positions to convex hull of tracklet centroids
    if hasattr(data, "centroids") and data.centroids is not None:
        centroids_np = data.centroids.detach().cpu().numpy()  # (N, 2)
        # Inverse-rotate centroids back to global frame for hull computation
        if R_inv is not None:
            centroids_np = (centroids_np @ R_inv.T) + origin
        pred_pos = clip_to_convex_hull(pred_pos, centroids_np, margin=0.02)

    # Convert normalized [0,1] positions to pixel coordinates for drawing
    image_wh = None
    if hasattr(data, "image_wh") and data.image_wh is not None:
        image_wh = data.image_wh.cpu().numpy()  # (2,) = [W, H]
    if image_wh is None:
        image_wh = np.array([frame.shape[1], frame.shape[0]], dtype=np.float32)

    lanelet_pos_px = pred_pos * image_wh  # denormalize to pixels
    heading_px = pred_head  # headings are unit vectors, no scaling needed

    # --- Node selection: Top-K by confidence (robust to miscalibration) ---
    # When top_k > 0, keep the best K nodes regardless of confidence threshold.
    # This prevents confidence collapse from hiding all predictions.
    if top_k > 0:
        # Top-K selection: keep K highest-confidence nodes
        k = min(top_k, M)
        top_indices = np.argsort(-pred_conf)[:k]
        active_mask = np.zeros(M, dtype=bool)
        active_mask[top_indices] = True
        vis_mask = active_mask.copy()
    else:
        # Original NMS + threshold path
        S_cols = S.T  # (M, N)
        norms = np.linalg.norm(S_cols, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        S_normed = S_cols / norms
        sim = S_normed @ S_normed.T  # (M, M)

        sim_threshold = 0.3
        lane_classes = []
        assigned = np.zeros(M, dtype=bool)
        for i in range(M):
            if assigned[i] or pred_conf[i] < 0.05:
                continue
            cluster = [i]
            assigned[i] = True
            queue = [i]
            while queue:
                cur = queue.pop(0)
                for j in range(M):
                    if assigned[j] or pred_conf[j] < 0.05:
                        continue
                    if sim[cur, j] > sim_threshold:
                        cluster.append(j)
                        assigned[j] = True
                        queue.append(j)
            lane_classes.append(cluster)

        suppressed = np.ones(M, dtype=bool)
        for cluster in lane_classes:
            best = max(cluster, key=lambda i: pred_conf[i])
            suppressed[best] = False

        active_mask = (~suppressed) & (pred_conf > conf_threshold)
        vis_mask = ~suppressed

    # --- Extract lane chains ---
    # Strategy: for each active node, pick argmax successor among candidates
    # (greedy best-edge linking). Falls back to deterministic if no edges.

    active_set = set(np.where(active_mask)[0])

    # Build scored successor graph from learned logits
    # For each active src, pick the best-scoring active dst as its successor
    best_successor = {}  # src -> dst (argmax successor)
    if succ_logits.shape[0] > 0:
        succ_probs = torch.sigmoid(succ_logits).cpu().numpy()

        # Expand active set: include nodes reachable via high-confidence
        # successor edges from/to active nodes (bridges chain gaps from top-K filtering)
        expanded = set(active_set)
        for e in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            prob = float(succ_probs[e])
            if prob > 0.3:
                if src in active_set:
                    expanded.add(dst)
                if dst in active_set:
                    expanded.add(src)
        active_set = expanded

        # Collect all candidate edges with scores for active nodes
        src_candidates = {}  # src -> list of (dst, prob)
        for e in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, e]), int(edge_index[1, e])
            if src in active_set and dst in active_set:
                src_candidates.setdefault(src, []).append((dst, float(succ_probs[e])))

        # For each source, pick argmax successor (no hard threshold)
        for src, cands in src_candidates.items():
            best_dst, best_prob = max(cands, key=lambda x: x[1])
            if best_prob > 0.1:  # very low floor to avoid pure noise
                best_successor[src] = best_dst

    if len(best_successor) > 0:
        # Build chains from argmax successor graph
        predecessor_count = {}
        for src, dst in best_successor.items():
            predecessor_count[dst] = predecessor_count.get(dst, 0) + 1

        roots = [n for n in active_set if predecessor_count.get(n, 0) == 0]
        if not roots:
            # All nodes have predecessors — pick highest-confidence as root
            roots = sorted(active_set, key=lambda n: -pred_conf[n])

        visited = set()
        lane_chains = []

        for root in sorted(roots, key=lambda n: -pred_conf[n]):
            if root in visited:
                continue
            chain = [root]
            visited.add(root)
            current = root
            while current in best_successor:
                next_node = best_successor[current]
                if next_node in visited:
                    break
                chain.append(next_node)
                visited.add(next_node)
                current = next_node
            lane_chains.append(chain)

        for n in active_set:
            if n not in visited:
                lane_chains.append([n])
                visited.add(n)
    else:
        # Fall back to deterministic successor linking
        from src.utils.successor_linking import deterministic_successor_linking
        vis_conf = pred_conf.copy()
        if not (top_k > 0):
            # Only suppress if using NMS path (suppressed is not defined for top_k)
            try:
                vis_conf[suppressed] = 0.0
            except NameError:
                pass
        lane_chains, _ = deterministic_successor_linking(
            pred_pos, pred_head, vis_conf,
            conf_threshold=min(conf_threshold, 0.05),
        )

    # Assign each lane chain a distinct color from hand-picked palette
    # color_offset ensures different lane groups on the same camera get different colors
    node_lane_color = {}  # node -> color
    for chain_idx, chain in enumerate(lane_chains):
        color = SLOT_COLORS[(chain_idx + color_offset) % len(SLOT_COLORS)]
        for n in chain:
            node_lane_color[n] = color

    # --- Draw GT lanelet positions as green diamonds (if available) ---
    # Drawn AFTER predictions so GT is visible on top
    has_gt = hasattr(data, "gt_lanelet_positions") and data.gt_lanelet_positions is not None
    if has_gt:
        gt_pos_m = data.gt_lanelet_positions.cpu().numpy()  # (G, 2) in (s,d) frame
        logger.debug(f"GT lanelet positions: {gt_pos_m.shape[0]} waypoints")
    else:
        gt_pos_m = np.zeros((0, 2))
        logger.debug("No GT lanelet positions on this graph")

    # Draw lane chains: piecewise linear through nodes
    for chain_idx, chain in enumerate(lane_chains):
        color = SLOT_COLORS[(chain_idx + color_offset) % len(SLOT_COLORS)]
        if len(chain) >= 2:
            pts = np.array([[lanelet_pos_px[n, 0], lanelet_pos_px[n, 1]]
                            for n in chain], dtype=np.float64)
            cv2.polylines(vis, [pts.astype(np.int32)], isClosed=False,
                          color=color, thickness=3, lineType=cv2.LINE_AA)

    # Draw lanelet nodes on top — style by confidence
    for m in range(M):
        if not vis_mask[m]:
            continue
        color = node_lane_color.get(m, (200, 200, 200))
        cx, cy = int(lanelet_pos_px[m, 0]), int(lanelet_pos_px[m, 1])
        conf = float(pred_conf[m])
        if conf >= conf_threshold:
            # High confidence: large filled circle
            cv2.circle(vis, (cx, cy), 8, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Low confidence: small hollow circle
            cv2.circle(vis, (cx, cy), 5, color, 1, cv2.LINE_AA)

    # Draw GT on top of everything (green diamonds + lane polylines)
    if has_gt and gt_pos_m.shape[0] > 0:
        gt_vis = gt_pos_m.copy()
        if R_inv is not None and origin is not None:
            gt_vis = (gt_vis @ R_inv.T) + origin
        gt_pos_px = gt_vis * image_wh

        # Draw GT lane polylines
        if hasattr(data, "gt_lanelet_lane_ids") and data.gt_lanelet_lane_ids is not None:
            lane_ids = data.gt_lanelet_lane_ids.cpu().numpy()
            for lid in np.unique(lane_ids):
                mask = lane_ids == lid
                lane_pts = gt_pos_px[mask]
                if len(lane_pts) >= 2:
                    cv2.polylines(vis, [lane_pts.astype(np.int32)],
                                  isClosed=False, color=(0, 220, 0),
                                  thickness=2, lineType=cv2.LINE_AA)

        # Draw green diamond markers
        for g in range(gt_pos_px.shape[0]):
            gx, gy = int(gt_pos_px[g, 0]), int(gt_pos_px[g, 1])
            diamond = np.array([
                [gx, gy - 7], [gx + 7, gy], [gx, gy + 7], [gx - 7, gy]
            ], dtype=np.int32)
            cv2.fillPoly(vis, [diamond], (0, 220, 0))
            cv2.polylines(vis, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)

    # Header with diagnostics
    n_active = int(active_mask.sum())
    n_chains = len([c for c in lane_chains if len(c) >= 2])
    n_gt = gt_pos_m.shape[0]
    n_above_05 = int((pred_conf > 0.5).sum())
    n_above_02 = int((pred_conf > 0.2).sum())
    conf_mean = float(pred_conf.mean())
    conf_max = float(pred_conf.max())
    y_off = 25
    lines = [
        f"Active: {n_active}/{M} | Lanes: {n_chains} | GT: {n_gt} | conf: mean={conf_mean:.2f} max={conf_max:.2f} >0.5:{n_above_05} >0.2:{n_above_02}",
    ]
    for text in lines:
        cv2.putText(vis, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        y_off += 22

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, vis)

    return vis, len(lane_chains)
