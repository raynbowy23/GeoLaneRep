"""Directed lane geometry generator (Pipeline B).

Given a user specification (e.g., "rightmost lane in US12_Park group 0"),
generates novel lane geometry that fits the target road section.

The diffusion model operates in canonical space (centered, aligned, unit-length).
Generated shapes are denormalized back to image space using the target group's
heading, scale, and centroid position.

Pipeline (baseline):
    spec → resolve to target embedding + spatial context
         → build warm-start from anchor lane (in anchor's canonical space)
         → diffuse conditioned on target embedding (in canonical space)
         → denormalize to image space using anchor's pose
         → post-process and score candidates

Pipeline (relational):
    spec → resolve to target embedding + spatial context
         → build warm-start from anchor lane (in NEIGHBOR's canonical space)
         → diffuse conditioned on embedding + relational context
         → denormalize to image space using NEIGHBOR's pose
         → generated lane appears at learned offset from neighbor
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.generation.augment import to_canonical, from_canonical
from src.generation.spec import LaneSpecification, SpatialContext, SpecEmbeddingResolver

logger = logging.getLogger(__name__)


def _segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects p3-p4 using cross products."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


@dataclass
class GenerationResult:
    """Result of directed lane generation."""

    candidates: np.ndarray       # (n, K, 2) generated lane geometries (image space)
    scores: np.ndarray           # (n,) cosine similarity to target embedding
    best: np.ndarray             # (K, 2) best candidate (image space)
    target_embedding: np.ndarray # (D,) resolved target embedding
    spatial_context: Optional[SpatialContext]
    spec: LaneSpecification


class DirectedLaneGenerator:
    """Orchestrates Pipeline B: spec → target embedding → diffusion → candidates.

    The diffusion model generates in canonical space (curvature patterns only).
    Results are placed in image space using the target group's spatial context.

    Args:
        resolver: SpecEmbeddingResolver for spec → embedding resolution.
        trainer: LaneDiffusionTrainer with trained denoiser.
        encoder: Trained LaneEncoder for re-encoding generated candidates.
        dataset: LaneDataset for re-encoding support.
        device: Torch device.
        relational_trainer: RelationalDiffusionTrainer (optional).
    """

    def __init__(self, resolver, trainer, encoder=None, dataset=None, device="cpu",
                 relational_trainer=None):
        self.resolver = resolver
        self.trainer = trainer
        self.relational_trainer = relational_trainer
        self.encoder = encoder
        self.dataset = dataset
        self.device = torch.device(device) if isinstance(device, str) else device

    def generate(
        self,
        spec: LaneSpecification,
        n_candidates: int = 5,
        warm_start_t: int = 50,
    ) -> GenerationResult:
        """Generate lane candidates matching a specification (baseline model).

        Uses the anchor lane's pose for denormalization — generated lane
        lands at the anchor's position.  This is the baseline behavior;
        for spatially-aware placement use generate_relational().

        Args:
            spec: Lane specification with constraints.
            n_candidates: Number of diverse candidates to generate.
            warm_start_t: Diffusion timestep for warm-start (0=clean, T=full noise).

        Returns:
            GenerationResult with scored candidates in image space.
        """
        # 1. Resolve spec
        target_embedding, spatial_context = self.resolver.resolve(spec)

        # 2. Get placement parameters from anchor
        target_centroid, target_angle, target_scale = self._get_placement(
            spatial_context
        )

        # 3. Build warm-start in canonical space from anchor lane
        canonical_warm_start = self._build_canonical_warm_start(spatial_context)

        # 4. Run diffusion in canonical space
        cond = torch.tensor(target_embedding, dtype=torch.float32)

        sample_kwargs = dict(n_samples=n_candidates)
        if canonical_warm_start is not None:
            ws = torch.tensor(
                canonical_warm_start.flatten(), dtype=torch.float32
            )
            sample_kwargs["warm_start"] = ws
            sample_kwargs["warm_start_t"] = warm_start_t

        generated = self.trainer.sample(cond, **sample_kwargs)

        K = self.resolver.index.K
        canonical_candidates = generated.cpu().numpy().reshape(n_candidates, K, 2)

        # 5. Denormalize from canonical to image space
        candidates = np.zeros_like(canonical_candidates)
        for i in range(n_candidates):
            candidates[i] = from_canonical(
                canonical_candidates[i],
                target_centroid, target_angle, target_scale,
            )

        # 5b. Apply lateral offset so generated lane doesn't overlap anchor.
        #     The diffusion model generates curvature in canonical space;
        #     placement is determined by anchor pose + lateral shift.
        if spatial_context is not None:
            candidates = self._apply_lateral_offset(candidates, spec, spatial_context)

        # 6. Post-process
        candidates = self._post_process(candidates)

        # 7. Score candidates
        scores = self._score_candidates(
            candidates, target_embedding, spatial_context, spec=spec,
        )

        best_idx = np.argmax(scores)

        return GenerationResult(
            candidates=candidates,
            scores=scores,
            best=candidates[best_idx],
            target_embedding=target_embedding,
            spatial_context=spatial_context,
            spec=spec,
        )

    def generate_with_embedding(
        self,
        embedding: np.ndarray,
        reference_geom: np.ndarray,
        n_candidates: int = 10,
        warm_start_t: int = 50,
        override_centroid: Optional[np.ndarray] = None,
    ) -> GenerationResult:
        """Generate using a pre-computed embedding and reference geometry.

        Same pipeline as generate() but bypasses spec resolution.
        Uses reference_geom for warm-start and angle/scale.

        Args:
            embedding: (D,) conditioning embedding.
            reference_geom: (K, 2) reference lane geometry in image space.
            n_candidates: Number of candidates.
            warm_start_t: Diffusion warm-start timestep.
            override_centroid: (2,) if set, place the generated lane at this
                centroid instead of the reference geometry's centroid.

        Returns:
            GenerationResult with scored candidates.
        """
        # Placement from reference geometry
        _, centroid, angle, scale = to_canonical(reference_geom)
        if override_centroid is not None:
            centroid = override_centroid

        # Warm-start from reference in canonical space
        canonical_ws, _, _, _ = to_canonical(reference_geom)
        canonical_ws = canonical_ws.astype(np.float32)

        # Run diffusion
        cond = torch.tensor(embedding, dtype=torch.float32)
        ws = torch.tensor(canonical_ws.flatten(), dtype=torch.float32)
        generated = self.trainer.sample(
            cond, n_samples=n_candidates,
            warm_start=ws, warm_start_t=warm_start_t,
        )

        K = reference_geom.shape[0]
        canonical_candidates = generated.cpu().numpy().reshape(n_candidates, K, 2)

        # Denormalize
        candidates = np.zeros_like(canonical_candidates)
        for i in range(n_candidates):
            candidates[i] = from_canonical(
                canonical_candidates[i], centroid, angle, scale,
            )

        # Post-process + score
        candidates = self._post_process(candidates)
        scores = self._score_candidates(candidates, embedding)
        best_idx = np.argmax(scores)

        return GenerationResult(
            candidates=candidates,
            scores=scores,
            best=candidates[best_idx],
            target_embedding=embedding,
            spatial_context=None,
            spec=None,
        )

    def generate_relational(
        self,
        spec: LaneSpecification,
        n_candidates: int = 5,
        warm_start_t: int = 50,
    ) -> GenerationResult:
        """Generate lane candidates with relational context (merge/diverge).

        Uses the NEIGHBOR's canonical frame as the shared reference:
        - The model generates the target in the neighbor's frame, where
          the spatial offset is part of the learned distribution.
        - Denormalization uses the neighbor's pose (centroid, angle, scale),
          so the generated lane appears at the correct position in image
          space without any post-hoc offset.

        Falls back to baseline generate() when no relational trainer or
        no relational context is available.

        Args:
            spec: Lane specification with constraints.
            n_candidates: Number of candidates to generate.
            warm_start_t: Diffusion timestep for warm-start.

        Returns:
            GenerationResult with scored candidates in image space.
        """
        if self.relational_trainer is None:
            logger.info("No relational trainer, falling back to baseline")
            return self.generate(spec, n_candidates, warm_start_t)

        # 1. Resolve spec
        target_embedding, spatial_context = self.resolver.resolve(spec)

        # Check if relational context is available
        has_relation = (
            spatial_context is not None
            and spatial_context.neighbor_geometry is not None
        )

        if not has_relation:
            logger.info("No relational context in spatial_context, falling back")
            return self.generate(spec, n_candidates, warm_start_t)

        logger.info(
            f"Using relational generation: relationship={spatial_context.relationship_type}, "
            f"merge_point={spatial_context.merge_point:.2f}, offset={spatial_context.offset:.4f}"
        )

        # 2. Get NEIGHBOR's canonical pose for denormalization
        #    This is the key difference from baseline: we denormalize using
        #    the neighbor's frame, not the anchor/target's frame.
        neighbor_img = spatial_context.neighbor_geometry  # (K, 2)
        _, nb_centroid, nb_angle, nb_scale = to_canonical(neighbor_img)

        # 3. Build warm-start: express anchor in NEIGHBOR's canonical frame
        #    (so the warm-start is in the same frame the model was trained in)
        canonical_warm_start = None
        if spatial_context.anchor_geometry is not None:
            anchor_img = spatial_context.anchor_geometry
            # Apply neighbor's transform to anchor
            anchor_centered = anchor_img - nb_centroid
            c, s = np.cos(-nb_angle), np.sin(-nb_angle)
            R = np.array([[c, -s], [s, c]])
            anchor_rotated = anchor_centered @ R.T
            canonical_warm_start = (anchor_rotated / nb_scale).astype(np.float32)

            # When anchor == neighbor (rightmost/leftmost specs), the
            # warm-start lands at origin. Shift it by the offset along
            # the canonical y-axis so it starts at the target position.
            #
            # When anchor != neighbor (merge specs), the warm-start
            # already has the natural offset encoded — do NOT shift.
            anchor_is_neighbor = np.allclose(
                anchor_img, neighbor_img, atol=1e-6,
            )
            if anchor_is_neighbor:
                offset_val = spatial_context.offset or 0.0
                canonical_offset = offset_val / nb_scale if nb_scale > 1e-8 else offset_val
                canonical_warm_start[:, 1] += canonical_offset

        # 4. Neighbor in its own canonical frame (centered, aligned, unit-length)
        neighbor_canonical, _, _, _ = to_canonical(neighbor_img)

        # 5. Prepare tensors
        cond = torch.tensor(target_embedding, dtype=torch.float32)
        nb_geom = torch.tensor(
            neighbor_canonical.flatten(), dtype=torch.float32,
        )
        merge_pt = torch.tensor(
            [spatial_context.merge_point or 0.5], dtype=torch.float32,
        )
        offset_t = torch.tensor(
            [spatial_context.offset or 0.0], dtype=torch.float32,
        )
        has_rel = torch.tensor([1.0], dtype=torch.float32)

        # 6. Run relational diffusion
        sample_kwargs = dict(
            n_samples=n_candidates,
        )
        if canonical_warm_start is not None:
            ws = torch.tensor(
                canonical_warm_start.flatten(), dtype=torch.float32,
            )
            sample_kwargs["warm_start"] = ws
            sample_kwargs["warm_start_t"] = warm_start_t

        generated = self.relational_trainer.sample(
            cond, nb_geom, merge_pt, offset_t, has_rel,
            **sample_kwargs,
        )

        K = self.resolver.index.K
        canonical_candidates = generated.cpu().numpy().reshape(n_candidates, K, 2)

        # 7. Denormalize from neighbor's canonical frame to image space
        #    from_canonical(x, centroid, angle, scale) reverses
        #    the centering/rotation/scaling, placing the generated lane
        #    at its learned offset from the neighbor in image space.
        candidates = np.zeros_like(canonical_candidates)
        for i in range(n_candidates):
            candidates[i] = from_canonical(
                canonical_candidates[i],
                nb_centroid, nb_angle, nb_scale,
            )

        # 8. Post-process and score
        candidates = self._post_process(candidates)
        scores = self._score_candidates(
            candidates, target_embedding, spatial_context, spec=spec,
        )
        best_idx = np.argmax(scores)

        return GenerationResult(
            candidates=candidates,
            scores=scores,
            best=candidates[best_idx],
            target_embedding=target_embedding,
            spatial_context=spatial_context,
            spec=spec,
        )

    def _get_placement(self, spatial_context: Optional[SpatialContext]):
        """Get target centroid, angle, scale for denormalization.

        Uses the anchor lane's pose. The angle must match what was used
        for canonicalization (anchor's own angle) to avoid distortion.
        """
        if spatial_context is None:
            # Fallback: use mean of all index geometries
            mean_geom = self.resolver.index.geometries.mean(axis=0)
            _, centroid, angle, scale = to_canonical(mean_geom)
            return centroid, angle, scale

        anchor = spatial_context.anchor_geometry
        _, centroid, angle, scale = to_canonical(anchor)

        return centroid, angle, scale

    def _apply_lateral_offset(
        self,
        candidates: np.ndarray,
        spec: LaneSpecification,
        spatial_context: SpatialContext,
    ) -> np.ndarray:
        """Offset generated candidates laterally so they don't overlap the anchor.

        Baseline generation uses the anchor's pose for denormalization, so the
        generated lane lands on top of the anchor. This shifts it by one lane
        width in the appropriate direction:
          - rightmost: +1 lane width along lateral_perp (further right)
          - leftmost:  -1 lane width along lateral_perp (further left)
          - merge:     tapered offset (full at start, zero at end) to show
                       convergence toward the neighboring lane
          - replace:   no offset (stays at replaced lane's position)
        """
        if spec.replace_cls is not None:
            return candidates  # role replacement: stay at replaced lane's position

        perp = spatial_context.lateral_perp
        spacing = spatial_context.median_lane_spacing
        if perp is None or spacing < 1e-6:
            return candidates

        if spec.is_rightmost:
            offset = perp * spacing
            candidates = candidates + offset[None, None, :]
        elif spec.is_leftmost:
            offset = -perp * spacing
            candidates = candidates + offset[None, None, :]
        elif spec.has_successor:
            # Merge lane: taper from full offset at start to zero at end
            # so the generated lane converges toward the group
            K = candidates.shape[1]
            taper = np.linspace(1.0, 0.0, K)[:, None]  # (K, 1)
            offset = perp * spacing * taper  # (K, 2)
            candidates = candidates + offset[None, :, :]

        return candidates

    def _build_canonical_warm_start(
        self, spatial_context: Optional[SpatialContext]
    ) -> Optional[np.ndarray]:
        """Build warm-start by converting the anchor lane to canonical space.

        The anchor lane is the existing lane closest to the requested spec.
        In canonical space, the model adjusts its curvature based on the
        conditioning embedding.
        """
        if spatial_context is None:
            return None

        anchor = spatial_context.anchor_geometry  # (K, 2) in image space
        canonical, _, _, _ = to_canonical(anchor)
        return canonical.astype(np.float32)

    def _post_process(
        self, candidates: np.ndarray, n_smooth_passes: int = 2,
    ) -> np.ndarray:
        """Post-process generated candidates: clip, multi-pass smooth, filter jagged.

        Args:
            candidates: (n, K, 2) generated geometries.
            n_smooth_passes: Number of 3-point moving average passes.

        Returns:
            (n, K, 2) post-processed candidates.
        """
        # Clip to valid range
        candidates = np.clip(candidates, 0.0, 1.0)

        # Multi-pass smoothing (moving average with window=3)
        for _ in range(n_smooth_passes):
            for i in range(len(candidates)):
                geom = candidates[i]
                smoothed = geom.copy()
                for j in range(1, len(geom) - 1):
                    smoothed[j] = (geom[j - 1] + geom[j] + geom[j + 1]) / 3.0
                candidates[i] = smoothed

        return candidates

    def _score_candidates(
        self,
        candidates: np.ndarray,
        target_embedding: np.ndarray,
        spatial_context: Optional[SpatialContext] = None,
        spec: Optional[LaneSpecification] = None,
    ) -> np.ndarray:
        """Score candidates using spec-aware geometric quality metrics.

        Scoring adapts to the lane type being generated:
          - Rightmost/leftmost: smooth + heading-aligned with group direction
          - Merge: smooth + convergence quality (starts offset, ends near neighbor)
          - Default: smooth + inter-candidate consensus

        Weights:
          - Smoothness (0.2): curvature variance + self-intersection penalty
          - Spec-aware quality (0.8): heading alignment OR convergence quality

        NOTE on offset-aware scoring:
          Coherence (distance to group centroid) and chamfer (distance to nearest
          real lane) are deliberately NOT used. The lateral offset applied in
          _apply_lateral_offset means the generated lane is intentionally one
          lane width away from existing lanes — penalizing that distance would
          bias toward candidates that collapse back onto existing geometry.

        Future evaluation metrics (G3 figure, not candidate selection):
          - IoU with ground truth lanelet polygons
          - Topology validation (predecessor/successor connectivity)
          - Downstream planning success rate (SUMO integration)
          See docs/experiments.md for full evaluation plan.
        """
        n = len(candidates)

        # --- 1. Smoothness score (0.2 weight) ---
        # Combines curvature variance and self-intersection penalty.
        smooth_scores = np.zeros(n)
        for i, geom in enumerate(candidates):
            diffs = np.diff(geom, axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            if lengths.sum() < 1e-8:
                continue
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            angle_changes = np.abs(np.diff(angles))
            curvature_score = 1.0 / (1.0 + np.var(angle_changes))

            # Self-intersection penalty: check if any segments cross
            intersection_penalty = 1.0
            K = len(geom)
            for a in range(K - 1):
                for b in range(a + 2, K - 1):
                    if _segments_intersect(geom[a], geom[a+1], geom[b], geom[b+1]):
                        intersection_penalty = 0.3
                        break
                if intersection_penalty < 1.0:
                    break

            smooth_scores[i] = curvature_score * intersection_penalty

        # --- 2. Spec-aware quality score (0.8 weight) ---
        spec_scores = np.ones(n) * 0.5  # neutral default

        if spatial_context is not None and spec is not None:
            if spec.has_successor:
                # Merge lane: convergence quality
                # Good merge = starts offset from neighbor, ends close to neighbor
                spec_scores = self._score_merge_convergence(
                    candidates, spatial_context,
                )
            elif spec.is_rightmost or spec.is_leftmost:
                # Adjacent lane: heading alignment with group direction
                spec_scores = self._score_heading_alignment(
                    candidates, spatial_context,
                )
            else:
                # Generic: inter-candidate consensus (agreement = quality)
                spec_scores = self._score_consensus(candidates)
        elif spatial_context is not None:
            # No spec info: fall back to heading alignment
            spec_scores = self._score_heading_alignment(
                candidates, spatial_context,
            )

        scores = 0.2 * smooth_scores + 0.8 * spec_scores
        return scores

    @staticmethod
    def _score_heading_alignment(
        candidates: np.ndarray, spatial_context: SpatialContext,
    ) -> np.ndarray:
        """Score by how well candidate heading matches the group direction.

        A rightmost/leftmost lane should be parallel to existing traffic flow.
        Handles anti-parallel lanes (opposite traffic direction) by checking
        both the heading and its 180° rotation.
        """
        group_heading = spatial_context.group_heading
        n = len(candidates)
        scores = np.zeros(n)
        for i, geom in enumerate(candidates):
            direction = geom[-1] - geom[0]
            cand_heading = np.arctan2(direction[1], direction[0])
            # Angular difference, wrapped to [-pi, pi]
            diff = abs(cand_heading - group_heading)
            diff = min(diff, 2 * np.pi - diff)
            # Also check anti-parallel (180° rotated) — lanes can flow either way
            diff_anti = abs(cand_heading - group_heading + np.pi)
            diff_anti = min(diff_anti, 2 * np.pi - diff_anti)
            diff = min(diff, diff_anti)
            # Convert to score: 0 diff = 1.0, pi/2 diff = 0.0
            scores[i] = max(0.0, 1.0 - diff / (np.pi / 2))
        return scores

    @staticmethod
    def _score_merge_convergence(
        candidates: np.ndarray, spatial_context: SpatialContext,
    ) -> np.ndarray:
        """Score merge lane candidates by convergence quality.

        A good merge lane:
          - Starts offset from the neighbor (diverged at the beginning)
          - Ends close to the neighbor (converged at the end)
          - Has monotonically decreasing distance to neighbor along its length
        """
        neighbor = spatial_context.neighbor_geometry
        if neighbor is None:
            return np.ones(len(candidates)) * 0.5

        n = len(candidates)
        scores = np.zeros(n)
        for i, geom in enumerate(candidates):
            K = len(geom)
            # Per-point distance to nearest neighbor point
            dists = np.array([
                np.min(np.linalg.norm(neighbor - geom[k], axis=1))
                for k in range(K)
            ])

            # Convergence: start should be far, end should be close
            start_dist = np.mean(dists[:3])   # first few points
            end_dist = np.mean(dists[-3:])    # last few points
            convergence_ratio = start_dist / (end_dist + 1e-8)
            # Ratio > 1 means converging (good), < 1 means diverging (bad)
            convergence_score = min(1.0, convergence_ratio / 3.0)

            # Monotonicity: distance should generally decrease
            # Count fraction of steps where distance decreases
            decreasing = np.sum(np.diff(dists) < 0) / max(len(dists) - 1, 1)

            scores[i] = 0.6 * convergence_score + 0.4 * decreasing

        return scores

    @staticmethod
    def _score_consensus(candidates: np.ndarray) -> np.ndarray:
        """Score by agreement with other candidates (consensus = quality).

        Candidates that are close to the mean of all candidates score higher.
        Outliers score lower.
        """
        n = len(candidates)
        if n <= 1:
            return np.ones(n)
        mean_shape = candidates.mean(axis=0)  # (K, 2)
        dists = np.array([
            np.mean(np.linalg.norm(c - mean_shape, axis=1))
            for c in candidates
        ])
        max_dist = dists.max() + 1e-8
        return 1.0 - dists / max_dist
