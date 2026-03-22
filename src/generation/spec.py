"""Lane specification resolution for directed generation (Pipeline B).

Resolves user-specified lane constraints (e.g., "rightmost lane in US12_Park
group 0") to a target embedding and spatial context for the diffusion model.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BehaviorPrefix:
    """Partial behavioral pattern for prefix-conditioned generation.

    Specify any subset of trajectory statistics. Unset fields (None) are
    unconstrained and will be filled by averaging matching lanes.

    Fields correspond to the 4-dim traj_stats vector:
        mean_speed, mean_curvature, mean_lateral_offset, trajectory_count
    """

    mean_speed: Optional[float] = None
    mean_curvature: Optional[float] = None
    mean_lateral_offset: Optional[float] = None
    trajectory_count: Optional[float] = None

    def to_mask_and_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mask, values) arrays of length 4.

        mask[i] is True if the field is specified.
        values[i] is the target value (0 if unset).
        """
        fields = [
            self.mean_speed, self.mean_curvature,
            self.mean_lateral_offset, self.trajectory_count,
        ]
        mask = np.array([f is not None for f in fields], dtype=bool)
        values = np.array([f if f is not None else 0.0 for f in fields],
                          dtype=np.float32)
        return mask, values


@dataclass
class LaneSpecification:
    """User constraints for directed lane generation."""

    lateral_rank: Optional[float] = None
    is_leftmost: Optional[bool] = None
    is_rightmost: Optional[bool] = None
    has_successor: Optional[bool] = None
    camera: Optional[str] = None
    group_id: Optional[int] = None
    behavior: Optional[BehaviorPrefix] = None

    # Role replacement: generate a lane at an existing lane's position
    # but with a different role.  e.g. replace_cls=5, replace_with="rightmost"
    # means "replace cls=5 (merge) with a rightmost-style lane."
    replace_cls: Optional[int] = None
    replace_with: Optional[str] = None  # "rightmost", "leftmost", "merge"

    @classmethod
    def rightmost(cls, camera: str = None, group_id: int = None) -> "LaneSpecification":
        return cls(is_rightmost=True, camera=camera, group_id=group_id)

    @classmethod
    def leftmost(cls, camera: str = None, group_id: int = None) -> "LaneSpecification":
        return cls(is_leftmost=True, camera=camera, group_id=group_id)

    @classmethod
    def merge_lane(cls, camera: str = None, group_id: int = None) -> "LaneSpecification":
        return cls(has_successor=True, camera=camera, group_id=group_id)

    @classmethod
    def replace_role(
        cls,
        replace_cls: int,
        replace_with: str,
        camera: str = None,
        group_id: int = None,
    ) -> "LaneSpecification":
        """Replace an existing lane's role with a different one.

        Generates a lane at the existing lane's position but conditioned
        on the target role's embedding.

        Args:
            replace_cls: cls_id of the lane to replace.
            replace_with: Target role ("rightmost", "leftmost", "merge").
            camera: Camera name.
            group_id: Group ID.

        Example:
            # Replace cls=5 (merge lane) with a rightmost lane
            spec = LaneSpecification.replace_role(5, "rightmost",
                                                   camera="US12_Park", group_id=1)
        """
        role_map = {
            "rightmost": dict(is_rightmost=True),
            "leftmost": dict(is_leftmost=True),
            "merge": dict(has_successor=True),
        }
        kwargs = role_map.get(replace_with, {})
        return cls(
            camera=camera,
            group_id=group_id,
            replace_cls=replace_cls,
            replace_with=replace_with,
            **kwargs,
        )

    @classmethod
    def from_behavior(
        cls,
        camera: str = None,
        group_id: int = None,
        mean_speed: float = None,
        mean_curvature: float = None,
        mean_lateral_offset: float = None,
        trajectory_count: float = None,
        **role_kwargs,
    ) -> "LaneSpecification":
        """Create a spec from partial behavioral pattern + location.

        Example:
            # "High-speed, low-curvature rightmost lane in US12_Park group 0"
            spec = LaneSpecification.from_behavior(
                camera="US12_Park", group_id=0,
                mean_speed=0.8, mean_curvature=0.01,
                is_rightmost=True,
            )
        """
        behavior = BehaviorPrefix(
            mean_speed=mean_speed,
            mean_curvature=mean_curvature,
            mean_lateral_offset=mean_lateral_offset,
            trajectory_count=trajectory_count,
        )
        return cls(
            camera=camera,
            group_id=group_id,
            behavior=behavior,
            **role_kwargs,
        )

    @classmethod
    def from_natural_language(
        cls, description: str, camera: str = None, group_id: int = None
    ) -> "LaneSpecification":
        """Parse simple natural language lane descriptions."""
        desc = description.lower().strip()
        spec = cls(camera=camera, group_id=group_id)

        if "rightmost" in desc or "right" in desc:
            spec.is_rightmost = True
        elif "leftmost" in desc or "left" in desc:
            spec.is_leftmost = True

        if "merge" in desc or "exit" in desc:
            spec.has_successor = True

        # Parse behavioral hints
        behavior = BehaviorPrefix()
        has_behavior = False
        if "fast" in desc or "high speed" in desc or "highway" in desc:
            behavior.mean_speed = 0.8
            has_behavior = True
        elif "slow" in desc or "low speed" in desc:
            behavior.mean_speed = 0.2
            has_behavior = True
        if "curvy" in desc or "high curvature" in desc or "winding" in desc:
            behavior.mean_curvature = 0.05
            has_behavior = True
        elif "straight" in desc or "low curvature" in desc:
            behavior.mean_curvature = 0.005
            has_behavior = True
        if has_behavior:
            spec.behavior = behavior

        return spec


@dataclass
class SpatialContext:
    """Spatial context of the target road section for directed generation."""

    group_geometries: np.ndarray    # (G, K, 2) existing lanes in the group
    group_heading: float            # mean direction in radians
    anchor_geometry: np.ndarray     # (K, 2) nearest existing lane
    anchor_side: str                # "left" or "right"
    median_lane_spacing: float      # estimated inter-lane distance
    lateral_perp: Optional[np.ndarray] = None  # (2,) perpendicular direction (increasing lateral rank)

    # Relational context (optional, for relational diffusion model)
    neighbor_geometry: Optional[np.ndarray] = None  # (K, 2) lane to merge into / diverge from
    relationship_type: Optional[str] = None         # "successor", "adjacent", or None
    merge_point: Optional[float] = None             # [0,1] where topology change occurs
    offset: Optional[float] = None                  # starting lateral distance from neighbor


class SpecEmbeddingResolver:
    """Resolve a LaneSpecification to a target embedding and spatial context.

    Uses the retrieval index to find lanes matching the spec constraints,
    averages their embeddings, and extracts spatial context from the dataset.
    """

    def __init__(self, index, dataset=None):
        """
        Args:
            index: LaneRetrievalIndex with embeddings, geometries, lane_keys.
            dataset: LaneDataset instance (for spatial context extraction).
        """
        self.index = index
        self.dataset = dataset

        # Extract role information from dataset samples
        self._roles = None
        self._cameras = None
        self._group_ids = None
        if dataset is not None:
            self._extract_metadata()

    def _extract_metadata(self):
        """Extract role vectors, traj stats, and metadata from dataset samples."""
        N = len(self.dataset.samples)
        roles_list = []
        traj_stats_list = []
        cameras = []
        group_ids = []

        for sample in self.dataset.samples:
            role = sample.role
            roles_list.append([
                role.lateral_rank,
                float(role.is_leftmost),
                float(role.is_rightmost),
                float(role.has_successor),
                role.group_size,
            ])
            traj_stats_list.append(sample.traj_stats)
            cameras.append(sample.camera)
            group_ids.append(sample.group_id)

        self._roles = np.array(roles_list, dtype=np.float32)
        self._traj_stats = np.array(traj_stats_list, dtype=np.float32)
        self._cameras = cameras
        self._group_ids = group_ids

    def resolve(
        self, spec: LaneSpecification
    ) -> Tuple[np.ndarray, Optional[SpatialContext]]:
        """Resolve spec to target embedding and optional spatial context.

        When a BehaviorPrefix is provided, matching lanes are weighted by
        behavioral similarity so the target embedding is biased toward the
        requested pattern.

        Args:
            spec: Lane specification with constraints.

        Returns:
            (target_embedding, spatial_context) tuple.
            target_embedding: (D,) averaged embedding of matching lanes.
            spatial_context: SpatialContext if camera/group specified, else None.
        """
        matching_indices = self._filter_by_spec(spec)

        if len(matching_indices) == 0:
            logger.warning(
                f"No lanes match spec {spec}, using global mean embedding"
            )
            target_embedding = self.index.embeddings.mean(axis=0)
        elif spec.behavior is not None and self._traj_stats is not None:
            target_embedding = self._behavior_weighted_embedding(
                matching_indices, spec.behavior,
            )
            logger.info(
                f"Spec resolved: {len(matching_indices)} matching lanes "
                f"(behavior-weighted)"
            )
        else:
            target_embedding = self.index.embeddings[matching_indices].mean(axis=0)
            logger.info(
                f"Spec resolved: {len(matching_indices)} matching lanes"
            )

        spatial_context = None
        if spec.camera is not None and spec.group_id is not None:
            spatial_context = self._extract_spatial_context(
                spec.camera, spec.group_id, spec
            )

        return target_embedding, spatial_context

    def _behavior_weighted_embedding(
        self, indices: np.ndarray, behavior: "BehaviorPrefix",
    ) -> np.ndarray:
        """Compute behavior-weighted average of embeddings at indices.

        Lanes whose traj_stats are closer to the requested behavioral prefix
        get higher weight via a Gaussian kernel on the specified dimensions.
        """
        mask, values = behavior.to_mask_and_values()
        if not mask.any():
            return self.index.embeddings[indices].mean(axis=0)

        stats = self._traj_stats[indices]  # (M, 4)
        # Compute per-dimension distance only on specified fields
        diffs = stats[:, mask] - values[mask]  # (M, n_specified)
        # Normalize by per-dimension std to handle different scales
        stds = stats[:, mask].std(axis=0)
        stds = np.maximum(stds, 1e-8)
        diffs_norm = diffs / stds
        distances = np.linalg.norm(diffs_norm, axis=1)  # (M,)

        # Gaussian kernel weights (sigma=1 in normalized space)
        weights = np.exp(-0.5 * distances ** 2)
        total = weights.sum()
        if total < 1e-8:
            weights = np.ones(len(indices)) / len(indices)
        else:
            weights = weights / total

        return (self.index.embeddings[indices] * weights[:, None]).sum(axis=0)

    def _filter_by_spec(self, spec: LaneSpecification) -> np.ndarray:
        """Filter index lanes by spec constraints (role + behavior)."""
        if self._roles is None:
            return np.arange(self.index.N)

        mask = np.ones(len(self._roles), dtype=bool)

        if spec.is_leftmost is not None:
            mask &= self._roles[:, 1].astype(bool) == spec.is_leftmost

        if spec.is_rightmost is not None:
            mask &= self._roles[:, 2].astype(bool) == spec.is_rightmost

        if spec.has_successor is not None:
            mask &= self._roles[:, 3].astype(bool) == spec.has_successor

        if spec.lateral_rank is not None:
            mask &= np.abs(self._roles[:, 0] - spec.lateral_rank) < 0.2

        return np.where(mask)[0]

    def _extract_spatial_context(
        self, camera: str, group_id: int, spec: LaneSpecification
    ) -> Optional[SpatialContext]:
        """Extract spatial context for a specific camera/group."""
        if self._cameras is None:
            return None

        # Find all lanes in this group
        group_mask = [
            i for i in range(len(self._cameras))
            if self._cameras[i] == camera and self._group_ids[i] == group_id
        ]

        if len(group_mask) == 0:
            logger.warning(f"No lanes found in {camera} group {group_id}")
            return None

        group_geoms = self.index.geometries[group_mask]  # (G, K, 2)

        # Compute group tangent and perpendicular using the SAME convention
        # as _compute_lane_roles in lane_dataset.py:
        #   tangent = mean of per-lane (end - start), normalized
        #   fold to upper half-plane for consistency
        #   perp = 90° CCW rotation of tangent
        # Lateral rank increases in the perp direction.
        tangents = []
        for g in group_geoms:
            t = g[-1] - g[0]
            norm = np.linalg.norm(t)
            if norm > 1e-8:
                tangents.append(t / norm)
        if tangents:
            mean_tangent = np.mean(tangents, axis=0)
            mean_tangent /= np.linalg.norm(mean_tangent) + 1e-8
            # Fold to upper half-plane (same as dataset)
            if mean_tangent[1] < 0 or (mean_tangent[1] == 0 and mean_tangent[0] < 0):
                mean_tangent = -mean_tangent
        else:
            mean_tangent = np.array([1.0, 0.0])

        perp = np.array([-mean_tangent[1], mean_tangent[0]])
        heading = float(np.arctan2(mean_tangent[1], mean_tangent[0]))

        # Find anchor lane using UNFOLDED perpendicular for visual consistency.
        # The dataset's lateral_rank uses a folded tangent, which flips
        # left/right for one traffic direction. Instead, compute fresh
        # lateral positions from the unfolded mean tangent.
        unfolded_tangent = np.mean(tangents, axis=0) if tangents else np.array([1.0, 0.0])
        unfolded_tangent = unfolded_tangent / (np.linalg.norm(unfolded_tangent) + 1e-8)
        # Perpendicular: 90° CCW rotation (no folding)
        unfolded_perp = np.array([-unfolded_tangent[1], unfolded_tangent[0]])

        group_centroids = group_geoms.mean(axis=1)  # (G, 2)
        visual_lateral = group_centroids @ unfolded_perp  # project onto unfolded perp

        # Role replacement: anchor is the specific lane being replaced
        if spec.replace_cls is not None:
            # Find the lane with matching cls_id in this group
            replaced = False
            group_samples = [
                s for s in self.dataset.samples
                if s.camera == camera and s.group_id == group_id
            ] if self.dataset else []
            for gi, gm_idx in enumerate(group_mask):
                sample = self.dataset.samples[gm_idx] if self.dataset else None
                if sample and sample.cls_id == spec.replace_cls:
                    anchor_idx = gi
                    lat = visual_lateral[gi]
                    median_lat = np.median(visual_lateral)
                    anchor_side = "right" if lat >= median_lat else "left"
                    replaced = True
                    logger.info(
                        f"Role replacement: cls={spec.replace_cls} → {spec.replace_with}"
                    )
                    break
            if not replaced:
                logger.warning(
                    f"replace_cls={spec.replace_cls} not found in group, "
                    f"falling back to default anchor selection"
                )
                anchor_idx = np.argmax(visual_lateral)
                anchor_side = "right"
        elif spec.has_successor:
            successor_flags = self._roles[group_mask, 3].astype(bool)
            if successor_flags.any():
                candidates = np.where(successor_flags)[0]
                anchor_idx = candidates[np.argmax(visual_lateral[candidates])]
                anchor_side = "right"
            else:
                anchor_idx = np.argmax(visual_lateral)
                anchor_side = "right"
        elif spec.is_rightmost or (spec.lateral_rank is not None and spec.lateral_rank > 0.5):
            # Use role annotations to find the actual rightmost lane
            rightmost_flags = self._roles[group_mask, 2].astype(bool)
            if rightmost_flags.any():
                anchor_idx = int(np.where(rightmost_flags)[0][0])
            else:
                anchor_idx = int(np.argmax(visual_lateral))
            anchor_side = "right"
        else:
            # Use role annotations to find the actual leftmost lane
            leftmost_flags = self._roles[group_mask, 1].astype(bool)
            if leftmost_flags.any():
                anchor_idx = int(np.where(leftmost_flags)[0][0])
            else:
                anchor_idx = int(np.argmin(visual_lateral))
            anchor_side = "left"

        anchor_geom = group_geoms[anchor_idx]

        # Estimate median inter-lane spacing using the same perp direction
        if len(group_geoms) > 1:
            centroids = group_geoms.mean(axis=1)  # (G, 2)
            lateral_positions = centroids @ perp
            sorted_pos = np.sort(lateral_positions)
            spacings = np.diff(sorted_pos)
            median_spacing = float(np.median(spacings)) if len(spacings) > 0 else 0.01
        else:
            median_spacing = 0.01

        # --- Populate relational context ---
        # Terminology:
        #   anchor  = existing lane closest to the spec (used for warm-start shape)
        #   neighbor = the lane the GENERATED lane relates to spatially
        #
        # The relational model generates in the neighbor's canonical frame.
        # The generated lane's position is learned from training pairs where
        # the target had a real offset from the neighbor.
        #
        # For rightmost:  neighbor = current rightmost lane (anchor)
        #                 offset = +median_lane_spacing (one lane further right)
        # For leftmost:   neighbor = current leftmost lane (anchor)
        #                 offset = -median_lane_spacing (one lane further left)
        # For merge:      neighbor = the lane being merged INTO (inward neighbor)
        #                 offset = computed from actual geometry

        neighbor_geom = anchor_geom
        neighbor_idx = anchor_idx
        relationship_type = "adjacent"
        explicit_offset = None  # None = compute from geometry

        if spec.replace_cls is not None:
            # Role replacement: neighbor is the closest lane to the replaced lane.
            # The generated lane stays at the replaced lane's position.
            # Offset = actual lateral distance from neighbor to the replaced lane,
            # so the relational model generates at the correct position.
            anchor_lateral = visual_lateral[anchor_idx]
            other_indices = [i for i in range(len(group_geoms)) if i != anchor_idx]
            if other_indices:
                other_laterals = visual_lateral[other_indices]
                closest = other_indices[
                    np.argmin(np.abs(other_laterals - anchor_lateral))
                ]
                neighbor_idx = closest
                neighbor_geom = group_geoms[closest]
                # Signed lateral distance from neighbor to replaced lane
                explicit_offset = float(anchor_lateral - visual_lateral[closest])
            else:
                explicit_offset = 0.0
            relationship_type = "adjacent"

        elif spec.has_successor:
            # Merge lane: neighbor is the non-successor lane closest to
            # the anchor (the lane the merge lane converges into).
            non_successor_mask = ~self._roles[group_mask, 3].astype(bool)
            if non_successor_mask.any():
                non_succ_indices = np.where(non_successor_mask)[0]
                anchor_lateral = visual_lateral[anchor_idx]
                lateral_dists = np.abs(visual_lateral[non_succ_indices] - anchor_lateral)
                neighbor_idx = non_succ_indices[np.argmin(lateral_dists)]
                neighbor_geom = group_geoms[neighbor_idx]
            relationship_type = "successor"

        elif spec.is_rightmost:
            # Generate one lane further right than the current rightmost.
            # Neighbor = current rightmost (the anchor).
            # Offset = +median_lane_spacing so model places it further out.
            neighbor_geom = anchor_geom
            neighbor_idx = anchor_idx
            relationship_type = "adjacent"
            explicit_offset = median_spacing

        elif spec.is_leftmost:
            # Generate one lane further left than the current leftmost.
            # Neighbor = current leftmost (the anchor).
            # Offset = -median_lane_spacing so model places it further out.
            neighbor_geom = anchor_geom
            neighbor_idx = anchor_idx
            relationship_type = "adjacent"
            explicit_offset = -median_spacing

        # Compute merge_point: where along the anchor it is closest
        # to the neighbor (for merge lanes this is the convergence point)
        K = anchor_geom.shape[0]
        if neighbor_idx != anchor_idx:
            min_dists = np.zeros(K)
            for k in range(K):
                dists_k = np.linalg.norm(neighbor_geom - anchor_geom[k], axis=1)
                min_dists[k] = dists_k.min()
            merge_point = float(np.argmin(min_dists) / max(K - 1, 1))
        else:
            # Anchor == neighbor: no meaningful merge point
            merge_point = 1.0  # end of lane (parallel adjacency)

        # Compute offset: lateral distance between generated lane and neighbor.
        # Must use unfolded_perp to match the training convention in
        # relational_pairs.py (which also uses unfolded group_perp).
        if explicit_offset is not None:
            offset = explicit_offset
        elif neighbor_idx != anchor_idx:
            anchor_start = anchor_geom[0]
            nb_dists = np.linalg.norm(neighbor_geom - anchor_start, axis=1)
            nearest_nb_pt = neighbor_geom[np.argmin(nb_dists)]
            offset = float(np.dot(anchor_start - nearest_nb_pt, unfolded_perp))
        else:
            # Anchor == neighbor: use median spacing as fallback
            offset = median_spacing if anchor_side == "right" else -median_spacing

        return SpatialContext(
            group_geometries=group_geoms,
            group_heading=heading,
            anchor_geometry=anchor_geom,
            anchor_side=anchor_side,
            median_lane_spacing=median_spacing,
            lateral_perp=perp,
            neighbor_geometry=neighbor_geom,
            relationship_type=relationship_type,
            merge_point=merge_point,
            offset=offset,
        )
