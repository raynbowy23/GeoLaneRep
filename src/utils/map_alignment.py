"""Map alignment: estimate 2D similarity transform from SUMO lanes to tracklets.

Aligns SUMO lane geometry to observed tracklet positions using an
ICP-like procedure that estimates rotation, scale, and translation.
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class MapAlignment:
    """Estimate 2D similarity transform aligning SUMO lanes to trajectories.

    The transform T is a 3x3 matrix encoding:
        - rotation angle theta
        - uniform scale s
        - translation (tx, ty)

    Applied as: [x', y', 1]^T = T @ [x, y, 1]^T
    """

    def __init__(self):
        self.T = np.eye(3)

    def estimate(
        self,
        sumo_points: np.ndarray,
        tracklet_points: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-4,
        max_corresp_dist: float = 50.0,
        rigid: bool = True,
    ) -> np.ndarray:
        """ICP-like alignment: minimize sum of point-to-nearest distances.

        Args:
            sumo_points: (S, 2) SUMO lane points in meter-space.
            tracklet_points: (N, 2) tracklet centroids in meter-space.
            max_iter: maximum ICP iterations.
            tol: convergence tolerance on mean distance change.
            max_corresp_dist: maximum correspondence distance (meters).
            rigid: if True, use rigid transform (rotation + translation only,
                   no scale). Default True because SUMO and tracklets are
                   already in the same meter-space; only GPS offset needs
                   correction.

        Returns:
            T: (3, 3) transform matrix.
        """
        if len(sumo_points) < 3 or len(tracklet_points) < 3:
            logger.warning("Too few points for alignment, returning identity")
            return self.T.copy()

        src = sumo_points.copy()
        tree = cKDTree(tracklet_points)
        prev_mean_dist = float('inf')

        for iteration in range(max_iter):
            # 1. Find nearest tracklet for each SUMO point
            dists, indices = tree.query(src)

            # Filter by max correspondence distance
            valid = dists < max_corresp_dist
            if valid.sum() < 3:
                logger.debug(
                    f"ICP iter {iteration}: too few correspondences "
                    f"({valid.sum()}), stopping"
                )
                break

            src_valid = src[valid]
            dst_valid = tracklet_points[indices[valid]]

            # 2. Estimate transform from correspondences
            T_step = self._estimate_similarity(src_valid, dst_valid, rigid=rigid)

            # 3. Apply transform to source points
            src = self.apply_with_transform(sumo_points, T_step @ self.T)

            # Update cumulative transform
            self.T = T_step @ self.T

            # 4. Check convergence
            mean_dist = dists[valid].mean()
            if abs(prev_mean_dist - mean_dist) < tol:
                logger.debug(
                    f"ICP converged at iter {iteration}, "
                    f"mean_dist={mean_dist:.4f}m"
                )
                break
            prev_mean_dist = mean_dist

        logger.info(
            f"MapAlignment: {iteration + 1} iterations, "
            f"final mean_dist={prev_mean_dist:.4f}m, "
            f"transform: scale={self.get_scale():.4f}, "
            f"rotation={np.degrees(self.get_rotation()):.2f}deg, "
            f"translation={self.get_translation()}"
        )
        return self.T.copy()

    @staticmethod
    def _estimate_similarity(
        src: np.ndarray, dst: np.ndarray, rigid: bool = True
    ) -> np.ndarray:
        """Estimate 2D transform (Procrustes) from point correspondences.

        Args:
            src: (K, 2) source points.
            dst: (K, 2) destination points.
            rigid: if True, fix scale=1.0 (rotation + translation only).

        Returns:
            T: (3, 3) transform matrix.
        """
        # Compute centroids
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        # Center points
        src_c = src - src_mean
        dst_c = dst - dst_mean

        # Compute cross-covariance matrix
        H = src_c.T @ dst_c  # (2, 2)

        # SVD
        U, S_vals, Vt = np.linalg.svd(H)

        # Rotation (handle reflection)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1, np.sign(d)])
        R = Vt.T @ sign_matrix @ U.T

        # Scale
        if rigid:
            scale = 1.0
        else:
            src_var = (src_c ** 2).sum()
            if src_var < 1e-10:
                scale = 1.0
            else:
                scale = (sign_matrix @ np.diag(S_vals)).trace() / src_var

        # Translation
        t = dst_mean - scale * R @ src_mean

        # Build 3x3 transform
        T = np.eye(3)
        T[:2, :2] = scale * R
        T[:2, 2] = t

        return T

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply learned transform to points.

        Args:
            points: (K, 2) points to transform.

        Returns:
            (K, 2) transformed points.
        """
        return self.apply_with_transform(points, self.T)

    @staticmethod
    def apply_with_transform(
        points: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """Apply a 3x3 transform to 2D points.

        Args:
            points: (K, 2) points.
            T: (3, 3) transform matrix.

        Returns:
            (K, 2) transformed points.
        """
        pts_h = np.hstack([points, np.ones((len(points), 1))])
        transformed = (T @ pts_h.T).T
        return transformed[:, :2]

    def get_rotation(self) -> float:
        """Extract rotation angle (radians) from the transform."""
        return np.arctan2(self.T[1, 0], self.T[0, 0])

    def get_scale(self) -> float:
        """Extract scale from the transform."""
        return np.sqrt(self.T[0, 0] ** 2 + self.T[1, 0] ** 2)

    def get_translation(self) -> np.ndarray:
        """Extract translation vector from the transform."""
        return self.T[:2, 2]
