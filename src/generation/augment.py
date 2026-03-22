"""Geometry augmentation and canonical space transforms for diffusion training.

Canonical space: each lane is centered at origin, rotated so start→end aligns
with the positive x-axis, and scaled to unit arc length. This strips away
camera-specific position/orientation/scale, leaving only the curvature pattern.
Both OpenLaneV2 and annotated lanes look identical in this space.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Canonical space transforms
# ---------------------------------------------------------------------------

def to_canonical(geometry: np.ndarray):
    """Convert a single (K, 2) lane geometry to canonical space.

    Canonical space: centered at origin, start→end aligned with x-axis,
    unit arc length.

    Args:
        geometry: (K, 2) lane waypoints.

    Returns:
        (canonical, centroid, angle, scale) tuple.
        - canonical: (K, 2) in canonical space.
        - centroid: (2,) original centroid.
        - angle: float, original start→end angle in radians.
        - scale: float, original arc length.
    """
    centroid = geometry.mean(axis=0)
    centered = geometry - centroid

    # Angle from first to last point
    direction = geometry[-1] - geometry[0]
    angle = float(np.arctan2(direction[1], direction[0]))

    # Rotate to align start→end with positive x-axis
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    rotated = centered @ R.T

    # Scale to unit arc length
    diffs = np.diff(rotated, axis=0)
    arc_length = float(np.linalg.norm(diffs, axis=1).sum())
    scale = arc_length if arc_length > 1e-8 else 1.0
    canonical = rotated / scale

    return canonical, centroid, angle, scale


def from_canonical(
    canonical: np.ndarray,
    centroid: np.ndarray,
    angle: float,
    scale: float,
) -> np.ndarray:
    """Convert canonical geometry back to image space.

    Args:
        canonical: (K, 2) in canonical space.
        centroid: (2,) target centroid.
        angle: target start→end angle in radians.
        scale: target arc length.

    Returns:
        (K, 2) lane geometry in image space.
    """
    # Scale back
    scaled = canonical * scale

    # Rotate back to original orientation
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    rotated = scaled @ R.T

    # Translate to target centroid
    return rotated + centroid


def batch_to_canonical(geometries: np.ndarray):
    """Convert (N, K, 2) geometries to canonical space.

    Returns:
        (canonicals, centroids, angles, scales) tuple.
    """
    N = len(geometries)
    K = geometries.shape[1]
    canonicals = np.zeros_like(geometries)
    centroids = np.zeros((N, 2), dtype=np.float64)
    angles = np.zeros(N, dtype=np.float64)
    scales = np.zeros(N, dtype=np.float64)

    for i in range(N):
        canonicals[i], centroids[i], angles[i], scales[i] = to_canonical(
            geometries[i]
        )

    return canonicals, centroids, angles, scales


def batch_from_canonical(
    canonicals: np.ndarray,
    centroids: np.ndarray,
    angles: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Convert (N, K, 2) canonical geometries back to image space."""
    N = len(canonicals)
    result = np.zeros_like(canonicals)
    for i in range(N):
        result[i] = from_canonical(canonicals[i], centroids[i], angles[i], scales[i])
    return result


# ---------------------------------------------------------------------------
# Augmentation (operates in canonical space)
# ---------------------------------------------------------------------------

def augment_geometries(
    geometries: np.ndarray,
    embeddings: np.ndarray,
    factor: int = 10,
    rotation_deg: float = 15.0,
    lateral_jitter: float = 0.02,
    stretch_range: tuple = (0.9, 1.1),
    curvature_jitter: float = 0.01,
    seed: int = 42,
) -> tuple:
    """Augment canonical geometry-embedding pairs for diffusion training.

    Augmentations are applied in canonical space (centered, aligned, unit-length):
    - Small rotation: adds slight heading variation
    - Lateral jitter: perpendicular offset (y-axis in canonical = cross-lane)
    - Stretch: varies lane length
    - Curvature jitter: per-point noise to vary curvature

    Args:
        geometries: (N, K, 2) lane waypoints (canonical or image space).
        embeddings: (N, D) behavioral embeddings.
        factor: Number of augmented variants per original.
        rotation_deg: Max rotation angle in degrees.
        lateral_jitter: Max lateral (y) offset in canonical units.
        stretch_range: (min, max) stretch factor.
        curvature_jitter: Max per-point noise in canonical units.
        seed: Random seed.

    Returns:
        (aug_geometries, aug_embeddings) with shapes
        (N*factor, K, 2) and (N*factor, D).
    """
    rng = np.random.RandomState(seed)
    N, K, _ = geometries.shape

    aug_geoms = []
    aug_embeds = []

    max_rad = np.deg2rad(rotation_deg)

    for i in range(N):
        geom = geometries[i]  # (K, 2) already in canonical space
        emb = embeddings[i]   # (D,)

        for _ in range(factor):
            g = geom.copy()

            # 1. Small rotation (heading variation)
            rot_angle = rng.uniform(-max_rad, max_rad)
            c, s = np.cos(rot_angle), np.sin(rot_angle)
            R = np.array([[c, -s], [s, c]])
            g = g @ R.T

            # 2. Lateral jitter (y-axis in canonical = perpendicular to lane)
            y_offset = rng.uniform(-lateral_jitter, lateral_jitter)
            g[:, 1] += y_offset

            # 3. Stretch along lane direction (x-axis in canonical)
            stretch = rng.uniform(stretch_range[0], stretch_range[1])
            g[:, 0] *= stretch

            # 4. Per-point curvature noise
            noise = rng.normal(0, curvature_jitter, size=g.shape)
            g += noise

            aug_geoms.append(g)
            aug_embeds.append(emb)

    aug_geometries = np.array(aug_geoms, dtype=np.float32)
    aug_embeddings = np.array(aug_embeds, dtype=np.float32)

    return aug_geometries, aug_embeddings
