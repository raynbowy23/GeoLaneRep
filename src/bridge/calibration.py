"""Camera calibration: pixel ↔ SUMO coordinate transforms.

Two-step calibration chain (from FedMeta-GeoLane):
    1. Pixel ↔ GPS: homography from calibration CSV (pixel_x, pixel_y, lat, lon)
    2. GPS ↔ SUMO: sumolib.net.convertLonLat2XY / convertXY2LonLat

This avoids the single-homography extrapolation problem by using sumolib's
proper UTM projection for the GPS↔SUMO step.

Pipeline for SUMO → pixel:
    SUMO (x,y) → sumolib.convertXY2LonLat → (lon,lat) → H_gps_to_pixel → pixel (x,y)

Pipeline for pixel → SUMO:
    pixel (x,y) → H_pixel_to_gps → (lon,lat) → sumolib.convertLonLat2XY → SUMO (x,y)
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraCalibration:
    """Two-step calibration for a single camera: pixel ↔ GPS ↔ SUMO.

    Uses a homography for pixel↔GPS and sumolib for GPS↔SUMO.
    """

    def __init__(
        self,
        H_gps_to_pixel: np.ndarray,
        H_pixel_to_gps: np.ndarray,
        sumo_net,
        camera: str,
        image_w: int = 1920,
        image_h: int = 1080,
        reprojection_error: float = 0.0,
    ):
        self.H_gps_to_pixel = H_gps_to_pixel  # GPS (lon,lat) → pixel
        self.H_pixel_to_gps = H_pixel_to_gps  # pixel → GPS (lon,lat)
        self.sumo_net = sumo_net
        self.camera = camera
        self.image_w = image_w
        self.image_h = image_h
        self.reprojection_error = reprojection_error

    def sumo_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Transform (N, 2) SUMO meter coordinates to normalized [0,1] pixel.

        Chain: SUMO → GPS (sumolib) → pixel (homography) → normalize.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # Step 1: SUMO → GPS via sumolib
        gps_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            lon, lat = self.sumo_net.convertXY2LonLat(pts[i, 0], pts[i, 1])
            gps_pts[i] = [lon, lat]

        # Step 2: GPS → pixel via homography
        ones = np.ones((len(gps_pts), 1))
        pts_h = np.hstack([gps_pts, ones])
        projected = (self.H_gps_to_pixel @ pts_h.T).T
        pixel_pts = projected[:, :2] / projected[:, 2:3]

        # Normalize to [0, 1]
        pixel_pts[:, 0] /= self.image_w
        pixel_pts[:, 1] /= self.image_h

        return pixel_pts

    def pixel_to_sumo(self, points: np.ndarray) -> np.ndarray:
        """Transform (N, 2) normalized [0,1] pixel to SUMO meter coordinates.

        Chain: denormalize → pixel → GPS (homography) → SUMO (sumolib).
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)

        # Denormalize from [0, 1] to pixel
        pts_px = pts.copy()
        pts_px[:, 0] *= self.image_w
        pts_px[:, 1] *= self.image_h

        # Step 1: pixel → GPS via inverse homography
        ones = np.ones((len(pts_px), 1))
        pts_h = np.hstack([pts_px, ones])
        projected = (self.H_pixel_to_gps @ pts_h.T).T
        gps_pts = projected[:, :2] / projected[:, 2:3]

        # Step 2: GPS → SUMO via sumolib
        sumo_pts = np.zeros_like(gps_pts)
        for i in range(len(gps_pts)):
            x, y = self.sumo_net.convertLonLat2XY(gps_pts[i, 0], gps_pts[i, 1])
            sumo_pts[i] = [x, y]

        return sumo_pts


def load_calibration(
    camera: str,
    calibration_dir: Path,
    net_path: Path,
    image_w: int = 1920,
    image_h: int = 1080,
) -> Optional[CameraCalibration]:
    """Load calibration for a camera from CSV + SUMO network.

    Two-step approach:
        1. Estimate homography: pixel ↔ GPS from calibration CSV
        2. Use sumolib for GPS ↔ SUMO coordinate conversion

    Args:
        camera: Camera name (e.g., "US12_Monona").
        calibration_dir: Directory with {camera}.csv files.
        net_path: Path to SUMO .net.xml.
        image_w: Image width in pixels.
        image_h: Image height in pixels.

    Returns:
        CameraCalibration or None on failure.
    """
    csv_path = calibration_dir / f"{camera}.csv"
    if not csv_path.exists():
        logger.warning(f"No calibration file for {camera} at {csv_path}")
        return None

    # Load SUMO network via sumolib
    try:
        import sumolib
        sumo_net = sumolib.net.readNet(str(net_path))
    except ImportError:
        logger.warning("sumolib not available, calibration disabled")
        return None

    # Load calibration points: pixel ↔ GPS
    pixel_pts = []
    gps_pts = []  # (lon, lat) for homography

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            px = float(row["pixel_x"])
            py = float(row["pixel_y"])
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            pixel_pts.append([px, py])
            gps_pts.append([lon, lat])  # homography works in (lon, lat)

    pixel_pts = np.array(pixel_pts, dtype=np.float64)
    gps_pts = np.array(gps_pts, dtype=np.float64)

    if len(pixel_pts) < 4:
        logger.warning(f"Need ≥4 calibration points, got {len(pixel_pts)}")
        return None

    # Estimate homography: GPS (lon,lat) → pixel
    H_gps_to_pixel, mask = cv2.findHomography(gps_pts, pixel_pts, cv2.RANSAC, 20.0)
    if H_gps_to_pixel is None:
        logger.warning(f"Homography estimation failed for {camera}")
        return None

    H_pixel_to_gps = np.linalg.inv(H_gps_to_pixel)

    # Compute reprojection error
    projected = cv2.perspectiveTransform(
        gps_pts.reshape(-1, 1, 2), H_gps_to_pixel,
    ).reshape(-1, 2)
    errors = np.linalg.norm(projected - pixel_pts, axis=1)
    mean_error = float(errors.mean())

    n_inliers = int(mask.sum()) if mask is not None else len(pixel_pts)
    logger.info(
        f"Calibration for {camera}: {n_inliers}/{len(pixel_pts)} inliers, "
        f"mean reproj error={mean_error:.1f}px"
    )

    # Verify: convert a SUMO lane point through the full chain
    lanes = sumo_net.getEdges()
    if lanes:
        test_edge = [e for e in lanes if not e.getID().startswith(":")]
        if test_edge:
            test_lane = test_edge[0].getLanes()[0]
            shape = test_lane.getShape()
            if shape:
                test_pt = np.array([[shape[0][0], shape[0][1]]])
                # SUMO → GPS
                lon, lat = sumo_net.convertXY2LonLat(test_pt[0, 0], test_pt[0, 1])
                # GPS → pixel
                gps_h = np.array([[lon, lat, 1.0]])
                pixel_h = (H_gps_to_pixel @ gps_h.T).T
                pixel_xy = pixel_h[0, :2] / pixel_h[0, 2]
                pixel_norm = pixel_xy / np.array([image_w, image_h])
                logger.info(
                    f"  Verify: SUMO ({test_pt[0,0]:.1f},{test_pt[0,1]:.1f}) → "
                    f"pixel ({pixel_norm[0]:.3f},{pixel_norm[1]:.3f})"
                )

    return CameraCalibration(
        H_gps_to_pixel=H_gps_to_pixel,
        H_pixel_to_gps=H_pixel_to_gps,
        sumo_net=sumo_net,
        camera=camera,
        image_w=image_w,
        image_h=image_h,
        reprojection_error=mean_error,
    )


def load_all_calibrations(
    calibration_dir: Path,
    sumo_root: Path,
    image_w: int = 1920,
    image_h: int = 1080,
) -> Dict[str, CameraCalibration]:
    """Load calibrations for all cameras that have CSV files."""
    calibrations = {}
    for csv_path in sorted(calibration_dir.glob("*.csv")):
        camera = csv_path.stem
        net_path = sumo_root / camera / "osm.net.xml"
        if not net_path.exists():
            continue

        calib = load_calibration(
            camera, calibration_dir, net_path, image_w, image_h,
        )
        if calib is not None:
            calibrations[camera] = calib

    logger.info(f"Loaded calibrations for {len(calibrations)} cameras")
    return calibrations
