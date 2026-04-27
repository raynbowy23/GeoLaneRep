import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.annotation_loader import (
    get_annotation_relationships,
    get_group_lanes,
    load_annotation_json,
)
from src.generation.augment import to_canonical

logger = logging.getLogger(__name__)


@dataclass
class RelationalPair:
    """One relational training pair.

    Both geometries are in the NEIGHBOR's canonical frame:
        - neighbor_geom is centered at origin, x-aligned, unit-length
          (standard canonical form)
        - target_geom is offset from origin, preserving the real
          spatial relationship between the two lanes

    Placement fields (centroid, angle, scale) describe the neighbor's
    original pose so from_canonical(generated, centroid, angle, scale)
    maps back to image space correctly.
    """

    # Target lane (the one we learn to generate) — in neighbor's canonical frame
    target_geom: np.ndarray         # (K, 2)
    target_embedding: np.ndarray    # (D,) behavioral embedding

    # Neighbor lane — in its own canonical frame (centered, aligned, unit-length)
    neighbor_geom: np.ndarray       # (K, 2)

    # Relational context
    merge_point: float              # [0, 1] where topology change occurs
    offset: float                   # starting lateral distance
    relationship_type: str          # "successor" or "adjacent"

    # Placement info: NEIGHBOR's original pose (for denormalization)
    centroid: np.ndarray            # (2,) neighbor's original centroid
    angle: float                    # neighbor's original angle
    scale: float                    # neighbor's original scale


def _resample_polyline(pts: np.ndarray, K: int) -> np.ndarray:
    """Resample a polyline to exactly K equidistant points."""
    if len(pts) == K:
        return pts
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]
    if total < 1e-8:
        return np.tile(pts[0], (K, 1))
    target_dists = np.linspace(0, total, K)
    resampled = np.zeros((K, 2))
    for i, d in enumerate(target_dists):
        idx = np.searchsorted(cum_lengths, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_len = seg_lengths[idx]
        if seg_len < 1e-8:
            resampled[i] = pts[idx]
        else:
            t = (d - cum_lengths[idx]) / seg_len
            resampled[i] = pts[idx] + t * diffs[idx]
    return resampled


def _to_shared_frame(
    points: np.ndarray,
    ref_centroid: np.ndarray,
    ref_angle: float,
    ref_scale: float,
) -> np.ndarray:
    """Express arbitrary geometry in the reference lane's canonical frame.

    Applies the same centering, rotation, and scaling that to_canonical()
    applies to the reference lane.

    Args:
        points: (K, 2) geometry in image/normalized space.
        ref_centroid: (2,) reference lane's centroid.
        ref_angle: reference lane's start→end angle.
        ref_scale: reference lane's arc length.

    Returns:
        (K, 2) geometry in the reference lane's canonical frame.
    """
    centered = points - ref_centroid
    c, s = np.cos(-ref_angle), np.sin(-ref_angle)
    R = np.array([[c, -s], [s, c]])
    rotated = centered @ R.T
    scaled = rotated / ref_scale if ref_scale > 1e-8 else rotated
    return scaled


def _compute_merge_point(
    target: np.ndarray,
    neighbor: np.ndarray,
) -> float:
    """Find where along the target lane it is closest to the neighbor.

    Computes per-point distance from target to nearest point on neighbor,
    finds the target point index with minimum distance.

    Args:
        target: (K, 2) target geometry.
        neighbor: (K, 2) neighbor geometry.

    Returns:
        merge_point in [0, 1].
    """
    K = len(target)
    min_dists = np.zeros(K)
    for i in range(K):
        dists = np.linalg.norm(neighbor - target[i], axis=1)
        min_dists[i] = dists.min()

    # merge_point = position of minimum distance along target
    idx = np.argmin(min_dists)
    return idx / max(K - 1, 1)


def _compute_offset(
    target: np.ndarray,
    neighbor: np.ndarray,
    group_perp: np.ndarray,
) -> float:
    """Compute starting lateral offset between target and neighbor.

    Projects the distance between the start of the target and nearest
    point on the neighbor onto the group's perpendicular direction.

    Args:
        target: (K, 2) target geometry.
        neighbor: (K, 2) neighbor geometry.
        group_perp: (2,) perpendicular direction of the group.

    Returns:
        Signed lateral offset.
    """
    target_start = target[0]
    # Nearest point on neighbor to target start
    dists = np.linalg.norm(neighbor - target_start, axis=1)
    nearest_idx = np.argmin(dists)
    delta = target_start - neighbor[nearest_idx]
    return float(np.dot(delta, group_perp))


def _build_pair(
    target_geom_img: np.ndarray,
    neighbor_geom_img: np.ndarray,
    target_embedding: np.ndarray,
    group_perp: np.ndarray,
    relationship_type: str,
) -> RelationalPair:
    """Build one RelationalPair in the neighbor's canonical frame.

    1. Canonicalize the NEIGHBOR → get (centroid, angle, scale).
    2. Express the TARGET in that same frame (preserves spatial offset).
    3. Compute merge_point and offset in image space (invariant to frame).
    """
    # Neighbor's canonical form and pose
    neighbor_canonical, nb_centroid, nb_angle, nb_scale = to_canonical(
        neighbor_geom_img
    )

    # Target expressed in neighbor's canonical frame
    target_in_nb_frame = _to_shared_frame(
        target_geom_img, nb_centroid, nb_angle, nb_scale,
    )

    # Merge point and offset computed in image space (frame-invariant)
    merge_point = _compute_merge_point(target_geom_img, neighbor_geom_img)
    offset = _compute_offset(target_geom_img, neighbor_geom_img, group_perp)

    return RelationalPair(
        target_geom=target_in_nb_frame.astype(np.float32),
        target_embedding=target_embedding.astype(np.float32),
        neighbor_geom=neighbor_canonical.astype(np.float32),
        merge_point=merge_point,
        offset=offset,
        relationship_type=relationship_type,
        centroid=nb_centroid.astype(np.float32),
        angle=float(nb_angle),
        scale=float(nb_scale),
    )


def build_relational_pairs(
    dataset,
    index,
    annotation_dir: str,
    polyline_k: int = 16,
    image_wh: Tuple[int, int] = (1920, 1080),
) -> List[RelationalPair]:
    """Build relational training pairs from dataset and annotations.

    Both geometries in each pair are expressed in the NEIGHBOR's canonical
    frame so the target's spatial offset is preserved in training data.

    Args:
        dataset: LaneDataset with .samples attribute.
        index: LaneRetrievalIndex with .embeddings, .geometries, .lane_keys.
        annotation_dir: Path to directory containing per-camera annotation.json.
        polyline_k: Number of points per polyline.
        image_wh: Image dimensions for normalization.

    Returns:
        List of RelationalPair instances.
    """
    annot_dir = Path(annotation_dir)

    # Build lookup: lane_key → (sample_index, sample)
    sample_lookup: Dict[str, int] = {}
    for i, sample in enumerate(dataset.samples):
        sample_lookup[sample.lane_key] = i

    # Build lane_key → index mapping for retrieval index
    key_to_idx: Dict[str, int] = {}
    for i, key in enumerate(index.lane_keys):
        key_to_idx[key] = i

    pairs: List[RelationalPair] = []

    # Iterate over cameras
    cameras = set(s.camera for s in dataset.samples)

    for camera in cameras:
        annot_path = annot_dir / camera / "annotation.json"
        if not annot_path.exists():
            logger.warning(f"No annotation.json for {camera}, skipping")
            continue

        annotation = load_annotation_json(annot_path)

        # Get all group_ids for this camera
        camera_samples = [s for s in dataset.samples if s.camera == camera]
        group_ids = set(s.group_id for s in camera_samples)

        for group_id in group_ids:
            relationships = get_annotation_relationships(annotation, group_id)
            if not relationships:
                continue

            # Get all lanes in this group
            group_lanes = get_group_lanes(annotation, group_id, image_wh)
            if not group_lanes:
                continue

            # Compute group heading and perpendicular
            tangents = []
            for lane in group_lanes:
                wp = lane["waypoints"]
                if len(wp) >= 2:
                    t = wp[-1] - wp[0]
                    n = np.linalg.norm(t)
                    if n > 1e-8:
                        tangents.append(t / n)
            if not tangents:
                continue
            mean_tangent = np.mean(tangents, axis=0)
            mean_tangent /= np.linalg.norm(mean_tangent) + 1e-8
            group_perp = np.array([-mean_tangent[1], mean_tangent[0]])

            # cls_id → resampled geometry
            cls_to_geom: Dict[int, np.ndarray] = {}
            for lane in group_lanes:
                wp = lane["waypoints"]
                if len(wp) >= 2:
                    cls_to_geom[lane["cls_id"]] = _resample_polyline(
                        wp, polyline_k,
                    )

            # Process each relationship
            for rel in relationships:
                rel_type = rel.get("type", "")
                from_cls = rel.get("from_cls")
                to_cls = rel.get("to_cls")

                if from_cls is None or to_cls is None:
                    continue
                if from_cls not in cls_to_geom or to_cls not in cls_to_geom:
                    continue

                if rel_type == "successor":
                    # from_cls merges into to_cls
                    target_geom_img = cls_to_geom[from_cls]
                    neighbor_geom_img = cls_to_geom[to_cls]
                elif rel_type == "adjacent":
                    # Bidirectional — use from as target, to as neighbor
                    target_geom_img = cls_to_geom[from_cls]
                    neighbor_geom_img = cls_to_geom[to_cls]
                else:
                    continue

                # Find matching lane_key in dataset
                target_key = f"{camera}_{group_id}_{from_cls}"
                if target_key not in key_to_idx:
                    continue

                target_idx = key_to_idx[target_key]
                target_embedding = index.embeddings[target_idx]

                pairs.append(_build_pair(
                    target_geom_img, neighbor_geom_img,
                    target_embedding, group_perp, rel_type,
                ))

            # Also create adjacent pairs for all lateral neighbors in the group
            # (even if not explicitly in relationships)
            group_samples = [
                s for s in camera_samples if s.group_id == group_id
            ]
            if len(group_samples) >= 2:
                # Sort by lateral rank
                group_samples.sort(key=lambda s: s.role.lateral_rank)
                for i in range(len(group_samples) - 1):
                    s_left = group_samples[i]
                    s_right = group_samples[i + 1]

                    left_key = s_left.lane_key
                    right_key = s_right.lane_key

                    if left_key not in key_to_idx or right_key not in key_to_idx:
                        continue

                    left_cls = s_left.cls_id
                    right_cls = s_right.cls_id

                    if left_cls not in cls_to_geom or right_cls not in cls_to_geom:
                        continue

                    left_idx = key_to_idx[left_key]
                    right_idx = key_to_idx[right_key]

                    # Left→Right pair (neighbor = right lane)
                    pairs.append(_build_pair(
                        cls_to_geom[left_cls], cls_to_geom[right_cls],
                        index.embeddings[left_idx], group_perp, "adjacent",
                    ))

                    # Right→Left pair (neighbor = left lane)
                    pairs.append(_build_pair(
                        cls_to_geom[right_cls], cls_to_geom[left_cls],
                        index.embeddings[right_idx], group_perp, "adjacent",
                    ))

    logger.info(
        f"Built {len(pairs)} relational pairs "
        f"({sum(1 for p in pairs if p.relationship_type == 'successor')} successor, "
        f"{sum(1 for p in pairs if p.relationship_type == 'adjacent')} adjacent)"
    )
    return pairs


def augment_relational_pairs(
    pairs: List[RelationalPair],
    factor: int = 10,
    rotation_deg: float = 15.0,
    lateral_jitter: float = 0.02,
    stretch_range: Tuple[float, float] = (0.9, 1.1),
    curvature_jitter: float = 0.01,
    no_relation_ratio: float = 0.3,
    seed: int = 42,
) -> dict:
    """Augment relational pairs for training.

    Applies the same random transform to both target and neighbor
    geometry to preserve their spatial relationship.

    Also includes a fraction of samples with zeroed relational context
    to teach the model to degrade gracefully when no relation is present.

    Args:
        pairs: List of RelationalPair.
        factor: Augmentation factor per pair.
        rotation_deg: Max rotation angle.
        lateral_jitter: Max lateral offset.
        stretch_range: Length stretch range.
        curvature_jitter: Per-point noise scale.
        no_relation_ratio: Fraction of samples with zeroed relational context.
        seed: Random seed.

    Returns:
        Dict with keys:
            geometries: (N, geom_dim) flattened target geometries.
            cond_embeddings: (N, D) behavioral embeddings.
            neighbor_geoms: (N, geom_dim) flattened neighbor geometries.
            merge_points: (N, 1) merge points.
            offsets: (N, 1) lateral offsets.
            has_relations: (N, 1) binary flags.
    """
    rng = np.random.RandomState(seed)

    if not pairs:
        K = 16
        D = 128
        return {
            "geometries": np.zeros((0, K * 2), dtype=np.float32),
            "cond_embeddings": np.zeros((0, D), dtype=np.float32),
            "neighbor_geoms": np.zeros((0, K * 2), dtype=np.float32),
            "merge_points": np.zeros((0, 1), dtype=np.float32),
            "offsets": np.zeros((0, 1), dtype=np.float32),
            "has_relations": np.zeros((0, 1), dtype=np.float32),
        }

    K = pairs[0].target_geom.shape[0]
    D = pairs[0].target_embedding.shape[0]
    max_rad = np.deg2rad(rotation_deg)

    all_geoms = []
    all_embeds = []
    all_neighbors = []
    all_merge_pts = []
    all_offsets = []
    all_has_rel = []

    for pair in pairs:
        for _ in range(factor):
            # Same random transform for both target and neighbor
            rot_angle = rng.uniform(-max_rad, max_rad)
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            R = np.array([[c, -s], [s, c]])
            y_off = rng.uniform(-lateral_jitter, lateral_jitter)
            stretch = rng.uniform(stretch_range[0], stretch_range[1])

            # Augment target
            tgt = pair.target_geom.copy() @ R.T
            tgt[:, 1] += y_off
            tgt[:, 0] *= stretch
            tgt += rng.normal(0, curvature_jitter, size=tgt.shape).astype(
                np.float32
            )

            # Augment neighbor with SAME transform
            nbr = pair.neighbor_geom.copy() @ R.T
            nbr[:, 1] += y_off
            nbr[:, 0] *= stretch
            nbr += rng.normal(0, curvature_jitter, size=nbr.shape).astype(
                np.float32
            )

            # Randomly zero out relational context
            if rng.random() < no_relation_ratio:
                nbr = np.zeros_like(nbr)
                mp = 0.0
                off = 0.0
                has_rel = 0.0
            else:
                mp = pair.merge_point
                off = pair.offset
                has_rel = 1.0

            all_geoms.append(tgt.flatten())
            all_embeds.append(pair.target_embedding)
            all_neighbors.append(nbr.flatten())
            all_merge_pts.append([mp])
            all_offsets.append([off])
            all_has_rel.append([has_rel])

    return {
        "geometries": np.array(all_geoms, dtype=np.float32),
        "cond_embeddings": np.array(all_embeds, dtype=np.float32),
        "neighbor_geoms": np.array(all_neighbors, dtype=np.float32),
        "merge_points": np.array(all_merge_pts, dtype=np.float32),
        "offsets": np.array(all_offsets, dtype=np.float32),
        "has_relations": np.array(all_has_rel, dtype=np.float32),
    }
