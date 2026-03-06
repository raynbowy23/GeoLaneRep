"""Homography-based coordinate transforms: pixel ↔ GPS ↔ local meter (ENU)."""

from typing import Optional, Tuple

import numpy as np


def pixels_per_meter_at_points(
    pixel_hom: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Compute local pixels-per-meter scale at N pixel positions (vectorized).

    For each point, transforms to GPS, displaces by 1m in longitude,
    transforms back to pixel, and returns the pixel displacement norm.

    Args:
        pixel_hom: (3, 3) homography mapping pixel coords -> GPS (lon, lat).
        points: (N, 2) pixel coordinates.

    Returns:
        (N,) pixels-per-meter at each point.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    N = pts.shape[0]

    # Transform pixel -> GPS  (N, 3) homogeneous
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 3)
    gps_h = (pixel_hom @ pts_h.T).T  # (N, 3)
    gps = gps_h[:, :2] / gps_h[:, 2:3]  # (N, 2)  [lon, lat]

    # Displace by 1 meter in longitude (east)
    lat_rad = np.radians(gps[:, 1])
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
    meters_per_deg_lon = np.where(meters_per_deg_lon < 1.0, 111320.0, meters_per_deg_lon)
    delta_lon = 1.0 / meters_per_deg_lon  # (N,)

    gps_displaced = np.column_stack([
        gps[:, 0] + delta_lon,
        gps[:, 1],
        np.ones(N, dtype=np.float64),
    ])  # (N, 3)

    # Transform displaced GPS -> pixel via pinv(pixel_hom)
    hom_inv = np.linalg.pinv(pixel_hom)
    px_displaced_h = (hom_inv @ gps_displaced.T).T  # (N, 3)
    px_displaced = px_displaced_h[:, :2] / px_displaced_h[:, 2:3]  # (N, 2)

    ppm = np.linalg.norm(px_displaced - pts, axis=1)  # (N,)

    # Clamp degenerate values
    ppm = np.where(ppm < 1e-6, 1.0, ppm)

    return ppm.astype(np.float32)


def meters_to_pixels(
    distance_m: float,
    pixel_hom: np.ndarray,
    ref_pixel: np.ndarray,
) -> float:
    """Convert a metric distance to pixel distance at a reference point.

    Uses the homography to estimate the local scale factor (pixels per meter)
    around ``ref_pixel``, then returns ``distance_m * pixels_per_meter``.

    Args:
        distance_m: Distance in meters to convert.
        pixel_hom: (3, 3) homography mapping pixel coords -> GPS (lon, lat).
        ref_pixel: (2,) reference pixel coordinate (x, y).

    Returns:
        Equivalent distance in pixels.
    """
    ref = np.array(ref_pixel, dtype=np.float64).ravel()[:2]

    # Transform reference pixel to GPS
    ref_h = np.array([ref[0], ref[1], 1.0], dtype=np.float64)
    gps_h = pixel_hom @ ref_h
    gps = gps_h[:2] / gps_h[2]  # (lon, lat)

    # Displace by 1 meter in longitude (east) direction
    # 1 degree latitude ~ 111320 m; 1 degree longitude ~ 111320 * cos(lat) m
    lat_rad = np.radians(gps[1])
    meters_per_deg_lon = 111320.0 * np.cos(lat_rad)
    if meters_per_deg_lon < 1.0:
        meters_per_deg_lon = 111320.0  # fallback at poles

    delta_lon = 1.0 / meters_per_deg_lon  # 1 meter in degrees longitude

    # New GPS point 1 meter east
    gps_displaced = np.array([gps[0] + delta_lon, gps[1], 1.0], dtype=np.float64)

    # Transform back to pixel via pinv(pixel_hom) (GPS -> pixel)
    hom_inv = np.linalg.pinv(pixel_hom)
    px_displaced_h = hom_inv @ gps_displaced
    px_displaced = px_displaced_h[:2] / px_displaced_h[2]

    pixels_per_meter = np.linalg.norm(px_displaced - ref)

    if pixels_per_meter < 1e-6:
        return distance_m  # degenerate homography; return raw meters as fallback

    return float(distance_m * pixels_per_meter)


# ---------------------------------------------------------------------------
# Pixel ↔ GPS ↔ Local ENU meter transforms
# ---------------------------------------------------------------------------

def _equirect_gps_to_meters(
    gps: np.ndarray,
    ref_gps: np.ndarray,
) -> np.ndarray:
    """Equirectangular projection: GPS (lon, lat) → local (x_east, y_north) meters.

    Args:
        gps: (..., 2) array of [lon, lat] in degrees.
        ref_gps: (2,) reference [lon, lat] in degrees (projection center).

    Returns:
        (..., 2) array of [x_east, y_north] in meters.
    """
    ref_lon, ref_lat = ref_gps[0], ref_gps[1]
    cos_ref = np.cos(np.radians(ref_lat))
    x_east = (gps[..., 0] - ref_lon) * 111320.0 * cos_ref
    y_north = (gps[..., 1] - ref_lat) * 111320.0
    return np.stack([x_east, y_north], axis=-1)


def _equirect_meters_to_gps(
    meters: np.ndarray,
    ref_gps: np.ndarray,
) -> np.ndarray:
    """Inverse equirectangular: local (x_east, y_north) meters → GPS (lon, lat).

    Args:
        meters: (..., 2) array of [x_east, y_north] in meters.
        ref_gps: (2,) reference [lon, lat] in degrees.

    Returns:
        (..., 2) array of [lon, lat] in degrees.
    """
    ref_lon, ref_lat = ref_gps[0], ref_gps[1]
    cos_ref = np.cos(np.radians(ref_lat))
    cos_ref = max(cos_ref, 1e-10)  # avoid division by zero at poles
    lon = meters[..., 0] / (111320.0 * cos_ref) + ref_lon
    lat = meters[..., 1] / 111320.0 + ref_lat
    return np.stack([lon, lat], axis=-1)


def pixels_to_local_meters(
    pixel_hom: np.ndarray,
    points: np.ndarray,
    ref_gps: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch pixel → GPS → local ENU meter conversion.

    Args:
        pixel_hom: (3, 3) homography mapping pixel coords → GPS (lon, lat).
        points: (N, 2) pixel coordinates.
        ref_gps: (2,) reference GPS [lon, lat] for the projection center.
            If None, computed from the mean GPS of the input points.

    Returns:
        meters: (N, 2) local meter coordinates [x_east, y_north].
        ref_gps: (2,) the reference GPS used (pass back for inverse).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    N = pts.shape[0]

    # Pixel → GPS via homography
    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 3)
    gps_h = (pixel_hom @ pts_h.T).T  # (N, 3)
    gps = gps_h[:, :2] / gps_h[:, 2:3]  # (N, 2) [lon, lat]

    # Compute ref_gps from mean if not provided
    if ref_gps is None:
        ref_gps = gps.mean(axis=0)  # (2,) [mean_lon, mean_lat]
    ref_gps = np.asarray(ref_gps, dtype=np.float64)

    meters = _equirect_gps_to_meters(gps, ref_gps).astype(np.float32)
    return meters, ref_gps.astype(np.float64)


def local_meters_to_pixels(
    pixel_hom: np.ndarray,
    meters: np.ndarray,
    ref_gps: np.ndarray,
) -> np.ndarray:
    """Inverse: local ENU meters → GPS → pixel coordinates.

    Args:
        pixel_hom: (3, 3) homography mapping pixel → GPS.
        meters: (N, 2) local meter coordinates [x_east, y_north].
        ref_gps: (2,) reference GPS [lon, lat].

    Returns:
        (N, 2) pixel coordinates.
    """
    meters = np.asarray(meters, dtype=np.float64)
    ref_gps = np.asarray(ref_gps, dtype=np.float64)

    # Meters → GPS
    gps = _equirect_meters_to_gps(meters, ref_gps)  # (N, 2)
    N = gps.shape[0]

    # GPS → pixel via pinv(pixel_hom)
    hom_inv = np.linalg.pinv(pixel_hom)
    gps_h = np.hstack([gps, np.ones((N, 1), dtype=np.float64)])  # (N, 3)
    px_h = (hom_inv @ gps_h.T).T  # (N, 3)
    pixels = px_h[:, :2] / px_h[:, 2:3]  # (N, 2)

    return pixels.astype(np.float32)


def gps_to_local_meters(
    gps_points: np.ndarray,
    ref_gps: np.ndarray,
) -> np.ndarray:
    """Convert GPS coordinates directly to local ENU meters.

    Useful for SUMO lane geometries that are already in GPS (lon, lat).

    Args:
        gps_points: (N, 2) GPS coordinates [lon, lat] in degrees.
        ref_gps: (2,) reference GPS [lon, lat] in degrees.

    Returns:
        (N, 2) local meter coordinates [x_east, y_north].
    """
    gps = np.asarray(gps_points, dtype=np.float64)
    ref = np.asarray(ref_gps, dtype=np.float64)
    return _equirect_gps_to_meters(gps, ref).astype(np.float32)
