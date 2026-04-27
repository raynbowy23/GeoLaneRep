import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    query_idx: int
    retrieved_indices: np.ndarray   # (k,)
    similarities: np.ndarray        # (k,)
    retrieved_geometries: np.ndarray # (k, K, 2)
    warm_start: np.ndarray          # (K, 2)


class LaneRetrievalIndex:
    """Cosine-similarity index over lane embeddings.

    Stores embeddings, geometries, and lane metadata for retrieval.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        geometries: np.ndarray,
        lane_keys: List[str],
        cameras: Optional[List[str]] = None,
        roles: Optional[np.ndarray] = None,
    ):
        """
        Args:
            embeddings: (N, D) behavioral embeddings.
            geometries: (N, K, 2) normalized waypoints.
            lane_keys: (N,) lane identifier strings.
            cameras: (N,) camera names (optional, for cross-camera filtering).
            roles: (N, R) structural role vectors (optional, for spec filtering).
        """
        self.embeddings = embeddings
        self.geometries = geometries
        self.lane_keys = lane_keys
        self.cameras = cameras
        self.roles = roles
        self.N, self.D = embeddings.shape
        self.K = geometries.shape[1]

        # Pre-normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings_norm = embeddings / np.maximum(norms, 1e-8)

        logger.info(
            f"LaneRetrievalIndex: {self.N} lanes, embed_dim={self.D}, "
            f"polyline_k={self.K}"
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        exclude_same_camera: Optional[str] = None,
        exclude_indices: Optional[set] = None,
    ) -> RetrievalResult:
        """Retrieve k nearest lanes by cosine similarity.

        Args:
            query_embedding: (D,) embedding vector.
            k: Number of neighbors to retrieve.
            exclude_same_camera: If set, exclude lanes from this camera.
            exclude_indices: Set of indices to exclude (e.g., the query itself).

        Returns:
            RetrievalResult with top-k matches and warm-start geometry.
        """
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        sims = self.embeddings_norm @ q_norm  # (N,)

        # Mask exclusions
        if exclude_same_camera and self.cameras:
            for i, cam in enumerate(self.cameras):
                if cam == exclude_same_camera:
                    sims[i] = -2.0
        if exclude_indices:
            for i in exclude_indices:
                sims[i] = -2.0

        top_k = np.argsort(sims)[-k:][::-1]
        top_sims = sims[top_k]
        top_geoms = self.geometries[top_k]

        warm = self._interpolate(top_geoms, top_sims)

        return RetrievalResult(
            query_idx=-1,
            retrieved_indices=top_k,
            similarities=top_sims,
            retrieved_geometries=top_geoms,
            warm_start=warm,
        )

    def retrieve_by_index(self, idx: int, k: int = 3,
                          cross_camera: bool = True) -> RetrievalResult:
        """Retrieve k nearest lanes for a lane already in the index.

        Args:
            idx: Index of the query lane.
            k: Number of neighbors (excluding self).
            cross_camera: If True, exclude same-camera lanes.
        """
        exclude_cam = self.cameras[idx] if cross_camera and self.cameras else None
        result = self.retrieve(
            self.embeddings[idx], k=k,
            exclude_same_camera=exclude_cam,
            exclude_indices={idx},
        )
        result.query_idx = idx
        return result

    def retrieve_all(self, k: int = 3,
                     cross_camera: bool = True) -> List[RetrievalResult]:
        """Retrieve for every lane in the index."""
        return [self.retrieve_by_index(i, k, cross_camera) for i in range(self.N)]

    def retrieve_by_role(
        self,
        role_filter: dict,
        k: int = 3,
    ) -> RetrievalResult:
        """Retrieve lanes matching a structural role filter.

        Args:
            role_filter: Dict with optional keys: 'is_leftmost', 'is_rightmost',
                'has_successor', 'lateral_rank' (with tolerance 0.2).
            k: Number of results.

        Returns:
            RetrievalResult from the centroid of matching lanes.
        """
        if self.roles is None:
            logger.warning("No roles stored in index, returning global centroid")
            centroid = self.embeddings.mean(axis=0)
            return self.retrieve(centroid, k=k)

        mask = np.ones(self.N, dtype=bool)
        # roles columns: [lateral_rank, is_leftmost, is_rightmost, has_successor, group_size]
        if "is_leftmost" in role_filter:
            mask &= self.roles[:, 1].astype(bool) == role_filter["is_leftmost"]
        if "is_rightmost" in role_filter:
            mask &= self.roles[:, 2].astype(bool) == role_filter["is_rightmost"]
        if "has_successor" in role_filter:
            mask &= self.roles[:, 3].astype(bool) == role_filter["has_successor"]
        if "lateral_rank" in role_filter:
            mask &= np.abs(self.roles[:, 0] - role_filter["lateral_rank"]) < 0.2

        if mask.sum() == 0:
            logger.warning("No lanes match role filter, using global centroid")
            centroid = self.embeddings.mean(axis=0)
        else:
            centroid = self.embeddings[mask].mean(axis=0)

        return self.retrieve(centroid, k=k)

    @staticmethod
    def _interpolate(geometries: np.ndarray, similarities: np.ndarray) -> np.ndarray:
        """Similarity-weighted interpolation of retrieved geometries.

        Args:
            geometries: (k, K, 2) retrieved lane waypoints.
            similarities: (k,) cosine similarities (may be negative).

        Returns:
            (K, 2) warm-start geometry.
        """
        # Shift similarities to positive range, then normalize
        weights = np.maximum(similarities, 0.0)
        total = weights.sum()
        if total < 1e-8:
            # Fallback: uniform weights
            weights = np.ones(len(similarities)) / len(similarities)
        else:
            weights = weights / total

        return np.sum(geometries * weights[:, None, None], axis=0)


def build_retrieval_index(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    use_embeddings: bool = True,
) -> LaneRetrievalIndex:
    """Build a retrieval index from a trained encoder and dataset.

    Args:
        model: Trained LaneEncoder (in eval mode).
        dataset: LaneDataset instance.
        device: torch device.
        use_embeddings: If True, use 128-dim embeddings. If False, use 64-dim
            L2-normalized projections.

    Returns:
        LaneRetrievalIndex ready for queries.
    """
    from torch.utils.data import DataLoader
    from src.data.lane_dataset import collate_fn

    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=collate_fn,
    )

    all_embeddings = []
    all_geometries = []
    all_keys = []
    all_cameras = []
    all_roles = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            stats_input = torch.cat([
                batch["traj_stats"].to(device),
                batch["roles"].to(device),
            ], dim=-1)
            if model.use_cross_lane_attention and "group_ids" in batch:
                output = model.forward_grouped(
                    geometry=batch["geometry"].to(device),
                    traj_polylines=batch["traj_polylines"].to(device),
                    traj_mask=batch["traj_mask"].to(device),
                    traj_stats=stats_input,
                    group_ids=batch["group_ids"].to(device),
                )
            else:
                output = model(
                    geometry=batch["geometry"].to(device),
                    traj_polylines=batch["traj_polylines"].to(device),
                    traj_mask=batch["traj_mask"].to(device),
                    traj_stats=stats_input,
                )

            key = "embedding" if use_embeddings else "projection"
            all_embeddings.append(output[key].cpu().numpy())
            all_geometries.append(batch["geometry"].numpy())
            all_keys.extend(batch["lane_keys"])
            all_cameras.extend(batch["cameras"])
            if "roles" in batch:
                all_roles.append(batch["roles"].numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    geometries = np.concatenate(all_geometries, axis=0)
    roles = np.concatenate(all_roles, axis=0) if all_roles else None

    return LaneRetrievalIndex(
        embeddings, geometries, all_keys, all_cameras, roles=roles
    )
