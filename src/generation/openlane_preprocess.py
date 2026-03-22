"""Extract lane centerline geometries from OpenLaneV2 for diffusion pretraining.

Reads the OpenLaneV2 pickle dataset, projects 3D centerlines to 2D pixel
coordinates using camera intrinsics/extrinsics, normalizes to [0,1], and
resamples to K=16 points for compatibility with geolane_encoder format.

Usage:
    from src.generation.openlane_preprocess import extract_openlane_geometries
    geometries = extract_openlane_geometries(
        "/path/to/OpenLane-V2",
        split="train",
        max_lanes=10000,
    )
    # geometries.shape == (N, 16, 2)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _resample_polyline(pts: np.ndarray, k: int) -> np.ndarray:
    """Resample a polyline to exactly k evenly-spaced points (arc-length)."""
    if len(pts) < 2:
        return np.zeros((k, 2), dtype=np.float64)

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]

    if total < 1e-10:
        return np.tile(pts[0], (k, 1))

    target_dists = np.linspace(0, total, k)
    resampled = np.zeros((k, 2), dtype=np.float64)
    for i, d in enumerate(target_dists):
        seg_idx = np.searchsorted(cum_len, d, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(pts) - 2)
        seg_start = cum_len[seg_idx]
        seg_end = cum_len[seg_idx + 1]
        seg_len = seg_end - seg_start
        t = (d - seg_start) / seg_len if seg_len > 1e-10 else 0.0
        resampled[i] = pts[seg_idx] * (1 - t) + pts[seg_idx + 1] * t

    return resampled


def _project_to_pixel(
    centerline_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> Optional[np.ndarray]:
    """Project 3D centerline (ego frame) to 2D pixel coordinates.

    OpenLaneV2 (Argoverse2) stores centerlines in ego vehicle frame.
    Camera extrinsics give the ego-to-camera transform:
        pts_cam = R @ pts_ego + T

    Args:
        centerline_3d: (N, 3) points in ego/vehicle frame.
        K: (3, 3) camera intrinsic matrix.
        R: (3, 3) rotation (ego -> camera).
        T: (3,) translation (ego -> camera).

    Returns:
        (M, 2) pixel coordinates for points with z > 0, or None if < 2 valid.
    """
    pts_cam = (R @ centerline_3d.T).T + T  # (N, 3)

    # Filter points behind camera
    valid = pts_cam[:, 2] > 0.5
    if valid.sum() < 2:
        return None

    pts_valid = pts_cam[valid]
    pts_pixel = (K @ pts_valid.T).T  # (N, 3)
    pts_pixel = pts_pixel[:, :2] / pts_pixel[:, 2:3]  # (N, 2)

    return pts_pixel


class OpenLaneV2Extractor:
    """Extract and process lane geometries from OpenLaneV2 pickle files."""

    # Typical Argoverse2 image dimensions
    DEFAULT_IMAGE_W = 1550
    DEFAULT_IMAGE_H = 2048

    def __init__(
        self,
        data_root: str,
        camera: str = "ring_front_center",
        polyline_k: int = 16,
        image_w: int = DEFAULT_IMAGE_W,
        image_h: int = DEFAULT_IMAGE_H,
    ):
        self.data_root = Path(data_root)
        self.camera = camera
        self.polyline_k = polyline_k
        self.image_w = image_w
        self.image_h = image_h

    def _load_pkl(self, split: str = "train") -> dict:
        """Load the appropriate pickle file for the given split."""
        # Try common naming patterns
        candidates = [
            f"data_dict_subset_A_{split}_lanesegnet.pkl",
            f"data_dict_subset_A_{split}.pkl",
            f"data_dict_sample_ls.pkl",
        ]
        for name in candidates:
            path = self.data_root / name
            if path.exists():
                logger.info(f"Loading OpenLaneV2 from {path}")
                with open(path, "rb") as f:
                    return pickle.load(f)

        raise FileNotFoundError(
            f"No OpenLaneV2 pickle found in {self.data_root}. "
            f"Tried: {candidates}"
        )

    def extract(
        self,
        split: str = "train",
        max_lanes: Optional[int] = None,
    ) -> np.ndarray:
        """Extract normalized, resampled lane geometries.

        Args:
            split: Dataset split ("train", "val", "test").
            max_lanes: Cap on number of lanes to extract (None = all).

        Returns:
            (N, K, 2) array of normalized [0,1] lane geometries.
        """
        data = self._load_pkl(split)
        geometries = []

        for entry_key, entry in data.items():
            if self.camera not in entry.get("sensor", {}):
                continue

            cam = entry["sensor"][self.camera]
            K = np.array(cam["intrinsic"]["K"], dtype=np.float64)
            R = np.array(cam["extrinsic"]["rotation"], dtype=np.float64)
            T = np.array(cam["extrinsic"]["translation"], dtype=np.float64)

            lane_segments = entry.get("annotation", {}).get("lane_segment", [])

            for seg in lane_segments:
                centerline = np.array(seg["centerline"], dtype=np.float64)
                if len(centerline) < 2:
                    continue

                pts_pixel = _project_to_pixel(centerline, K, R, T)
                if pts_pixel is None:
                    continue

                # Filter to within image bounds
                in_bounds = (
                    (pts_pixel[:, 0] >= 0)
                    & (pts_pixel[:, 0] <= self.image_w)
                    & (pts_pixel[:, 1] >= 0)
                    & (pts_pixel[:, 1] <= self.image_h)
                )
                if in_bounds.sum() < 2:
                    continue

                pts_visible = pts_pixel[in_bounds]

                # Normalize to [0, 1]
                pts_norm = pts_visible.copy()
                pts_norm[:, 0] /= self.image_w
                pts_norm[:, 1] /= self.image_h

                # Resample to K points
                resampled = _resample_polyline(pts_norm, self.polyline_k)

                # Validate range
                if resampled.min() < -0.1 or resampled.max() > 1.1:
                    continue

                resampled = np.clip(resampled, 0.0, 1.0)
                geometries.append(resampled)

                if max_lanes and len(geometries) >= max_lanes:
                    break

            if max_lanes and len(geometries) >= max_lanes:
                break

        result = np.array(geometries, dtype=np.float32)
        logger.info(
            f"Extracted {len(result)} lane geometries from OpenLaneV2 "
            f"({split}, camera={self.camera})"
        )
        return result


def extract_openlane_geometries(
    data_root: str,
    split: str = "train",
    max_lanes: Optional[int] = None,
    polyline_k: int = 16,
) -> np.ndarray:
    """Convenience function to extract OpenLaneV2 geometries.

    Args:
        data_root: Path to OpenLane-V2 data directory.
        split: Dataset split.
        max_lanes: Maximum number of lanes to extract.
        polyline_k: Number of waypoints per lane.

    Returns:
        (N, K, 2) normalized lane geometries.
    """
    extractor = OpenLaneV2Extractor(
        data_root=data_root,
        polyline_k=polyline_k,
    )
    return extractor.extract(split=split, max_lanes=max_lanes)
