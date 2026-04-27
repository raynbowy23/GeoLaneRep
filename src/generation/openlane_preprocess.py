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
        scene_radius: float = 60.0,
        max_curvature_var: float = 0.001,
    ) -> np.ndarray:
        """Extract BEV lane geometries using ego-frame XY coordinates directly.

        Uses centerline[:, :2] (top-down XY) instead of projecting through
        camera intrinsics/extrinsics. This matches the geometry distribution
        of overhead intersection cameras far better than perspective projection.

        Args:
            split: Dataset split ("train", "val", "test").
            max_lanes: Cap on number of lanes to extract (None = all).
            scene_radius: Normalization radius in meters (lanes within
                ±scene_radius of ego are mapped to [0, 1]).

        Returns:
            (N, K, 2) array of normalized [0,1] lane geometries.
        """
        data = self._load_pkl(split)
        geometries = []

        for entry_key, entry in data.items():
            lane_segments = entry.get("annotation", {}).get("lane_segment", [])

            for seg in lane_segments:
                centerline = np.array(seg["centerline"], dtype=np.float64)
                if len(centerline) < 2:
                    continue

                # Use BEV XY directly — drop Z (ground plane)
                pts_bev = centerline[:, :2]

                # Normalize to [0, 1] using fixed scene radius
                pts_norm = pts_bev / (2 * scene_radius) + 0.5

                # Keep lanes mostly within bounds
                in_bounds = (
                    (pts_norm[:, 0] >= -0.1) & (pts_norm[:, 0] <= 1.1)
                    & (pts_norm[:, 1] >= -0.1) & (pts_norm[:, 1] <= 1.1)
                )
                if in_bounds.sum() < 2:
                    continue

                pts_valid = pts_norm[in_bounds]

                # Resample to K points
                resampled = _resample_polyline(pts_valid, self.polyline_k)
                resampled = np.clip(resampled, 0.0, 1.0)

                # Filter out curved/winding lanes (intersections, ramps, connectors)
                diffs = np.diff(resampled, axis=0)
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])
                angle_changes = np.abs(np.diff(angles))
                # Wrap to [-pi, pi]
                angle_changes = np.where(
                    angle_changes > np.pi, 2 * np.pi - angle_changes, angle_changes
                )
                if np.var(angle_changes) > max_curvature_var:
                    continue

                geometries.append(resampled)

                if max_lanes and len(geometries) >= max_lanes:
                    break

            if max_lanes and len(geometries) >= max_lanes:
                break

        result = np.array(geometries, dtype=np.float32)
        logger.info(
            f"Extracted {len(result)} BEV lane geometries from OpenLaneV2 ({split})"
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
