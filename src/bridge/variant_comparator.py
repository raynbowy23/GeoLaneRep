"""Compare lane variants in embedding space after re-encoding.

Provides quantitative metrics for evaluating lane edits by comparing
re-encoded SUMO embeddings against real camera-observed embeddings.

This is the final step of the synthesis loop:
    edit → simulate → extract → re-encode → COMPARE
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LaneComparisonResult:
    """Comparison metrics for a single lane."""

    lane_id: str
    cosine_similarity: float     # [-1, 1] similarity to matched real lane
    l2_distance: float           # Euclidean distance in embedding space
    los_match: bool              # Whether LOS grade matches
    los_real: str                # Real lane LOS
    los_simulated: str           # Simulated lane LOS
    speed_delta_mph: float       # Speed difference (simulated - real)
    density_delta: float         # Density difference


@dataclass
class VariantComparisonReport:
    """Full comparison report for a lane editing variant."""

    variant_name: str
    lane_results: List[LaneComparisonResult]
    mean_cosine_similarity: float
    mean_l2_distance: float
    los_accuracy: float          # Fraction of lanes with matching LOS
    mean_speed_delta: float
    mean_density_delta: float


class VariantComparator:
    """Compare lane edit variants against real observations.

    Args:
        bridge: EncoderTrafficBridge for computing HCM metrics from embeddings.
    """

    def __init__(self, bridge=None):
        self.bridge = bridge

    def compare(
        self,
        real_embeddings: Dict[str, np.ndarray],
        simulated_embeddings: Dict[str, np.ndarray],
        real_metrics: Optional[Dict[str, "TrafficMetrics"]] = None,
        simulated_metrics: Optional[Dict[str, "TrafficMetrics"]] = None,
        variant_name: str = "default",
    ) -> VariantComparisonReport:
        """Compare real vs simulated lane embeddings.

        Args:
            real_embeddings: Dict mapping lane_id -> (D,) real embeddings.
            simulated_embeddings: Dict mapping lane_id -> (D,) re-encoded embeddings.
            real_metrics: Optional traffic metrics for real lanes.
            simulated_metrics: Optional traffic metrics for simulated lanes.
            variant_name: Label for this variant.

        Returns:
            VariantComparisonReport with per-lane and aggregate metrics.
        """
        # Find common lanes
        common_lanes = set(real_embeddings.keys()) & set(simulated_embeddings.keys())
        if not common_lanes:
            logger.warning("No common lanes between real and simulated embeddings")
            return VariantComparisonReport(
                variant_name=variant_name,
                lane_results=[],
                mean_cosine_similarity=0.0,
                mean_l2_distance=0.0,
                los_accuracy=0.0,
                mean_speed_delta=0.0,
                mean_density_delta=0.0,
            )

        results = []
        for lid in sorted(common_lanes):
            real_emb = real_embeddings[lid]
            sim_emb = simulated_embeddings[lid]

            # Cosine similarity
            dot = np.dot(real_emb, sim_emb)
            norm_r = np.linalg.norm(real_emb) + 1e-8
            norm_s = np.linalg.norm(sim_emb) + 1e-8
            cosine_sim = float(dot / (norm_r * norm_s))

            # L2 distance
            l2_dist = float(np.linalg.norm(real_emb - sim_emb))

            # Traffic metrics comparison
            los_real = "?"
            los_sim = "?"
            speed_delta = 0.0
            density_delta = 0.0
            los_match = False

            if real_metrics and lid in real_metrics:
                los_real = real_metrics[lid].los
            if simulated_metrics and lid in simulated_metrics:
                los_sim = simulated_metrics[lid].los
            if los_real != "?" and los_sim != "?":
                los_match = los_real == los_sim

            if real_metrics and simulated_metrics:
                if lid in real_metrics and lid in simulated_metrics:
                    speed_delta = (
                        simulated_metrics[lid].speed_mph
                        - real_metrics[lid].speed_mph
                    )
                    density_delta = (
                        simulated_metrics[lid].density_veh_mi_ln
                        - real_metrics[lid].density_veh_mi_ln
                    )

            results.append(LaneComparisonResult(
                lane_id=lid,
                cosine_similarity=cosine_sim,
                l2_distance=l2_dist,
                los_match=los_match,
                los_real=los_real,
                los_simulated=los_sim,
                speed_delta_mph=speed_delta,
                density_delta=density_delta,
            ))

        # Aggregate
        cos_sims = [r.cosine_similarity for r in results]
        l2_dists = [r.l2_distance for r in results]
        los_matches = [r.los_match for r in results]
        speed_deltas = [r.speed_delta_mph for r in results]
        density_deltas = [r.density_delta for r in results]

        report = VariantComparisonReport(
            variant_name=variant_name,
            lane_results=results,
            mean_cosine_similarity=float(np.mean(cos_sims)),
            mean_l2_distance=float(np.mean(l2_dists)),
            los_accuracy=float(np.mean(los_matches)) if los_matches else 0.0,
            mean_speed_delta=float(np.mean(speed_deltas)),
            mean_density_delta=float(np.mean(density_deltas)),
        )

        logger.info(
            f"Variant '{variant_name}': {len(results)} lanes compared | "
            f"cos_sim={report.mean_cosine_similarity:.3f} | "
            f"L2={report.mean_l2_distance:.3f} | "
            f"LOS_acc={report.los_accuracy:.1%}"
        )
        return report

    def compare_multiple_variants(
        self,
        real_embeddings: Dict[str, np.ndarray],
        variants: Dict[str, Dict[str, np.ndarray]],
        real_metrics: Optional[Dict[str, "TrafficMetrics"]] = None,
        variant_metrics: Optional[Dict[str, Dict[str, "TrafficMetrics"]]] = None,
    ) -> List[VariantComparisonReport]:
        """Compare multiple variants against the same real baseline.

        Args:
            real_embeddings: Real lane embeddings.
            variants: Dict mapping variant_name -> {lane_id: embedding}.
            real_metrics: Real traffic metrics.
            variant_metrics: Dict mapping variant_name -> {lane_id: TrafficMetrics}.

        Returns:
            List of VariantComparisonReport, sorted by mean_cosine_similarity (desc).
        """
        reports = []
        for name, sim_embs in variants.items():
            sim_metrics = None
            if variant_metrics and name in variant_metrics:
                sim_metrics = variant_metrics[name]

            report = self.compare(
                real_embeddings, sim_embs,
                real_metrics, sim_metrics,
                variant_name=name,
            )
            reports.append(report)

        # Sort by cosine similarity (higher = more similar to real)
        reports.sort(key=lambda r: r.mean_cosine_similarity, reverse=True)
        return reports

    def compare_nearest(
        self,
        real_embeddings: Dict[str, np.ndarray],
        simulated_embeddings: Dict[str, np.ndarray],
        variant_name: str = "default",
    ) -> VariantComparisonReport:
        """Compare by nearest-neighbor matching in embedding space.

        For each real lane, finds the most similar simulated lane by cosine
        similarity. Use when real and simulated lane IDs don't correspond
        (e.g., camera-observed vs SUMO edge IDs).

        Returns:
            VariantComparisonReport with per-lane results using NN matching.
        """
        if not real_embeddings or not simulated_embeddings:
            logger.warning("Empty embeddings for nearest-neighbor comparison")
            return VariantComparisonReport(
                variant_name=variant_name,
                lane_results=[], mean_cosine_similarity=0.0,
                mean_l2_distance=0.0, los_accuracy=0.0,
                mean_speed_delta=0.0, mean_density_delta=0.0,
            )

        real_ids = sorted(real_embeddings.keys())
        sim_ids = sorted(simulated_embeddings.keys())
        real_mat = np.stack([real_embeddings[k] for k in real_ids])
        sim_mat = np.stack([simulated_embeddings[k] for k in sim_ids])

        # Normalize for cosine similarity
        real_norm = real_mat / (np.linalg.norm(real_mat, axis=1, keepdims=True) + 1e-8)
        sim_norm = sim_mat / (np.linalg.norm(sim_mat, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity matrix: (n_real, n_sim)
        cos_matrix = real_norm @ sim_norm.T

        results = []
        for i, rid in enumerate(real_ids):
            best_j = int(np.argmax(cos_matrix[i]))
            best_sim_id = sim_ids[best_j]
            cos_sim = float(cos_matrix[i, best_j])
            l2_dist = float(np.linalg.norm(real_mat[i] - sim_mat[best_j]))

            results.append(LaneComparisonResult(
                lane_id=f"{rid}↔{best_sim_id}",
                cosine_similarity=cos_sim,
                l2_distance=l2_dist,
                los_match=False,
                los_real="?", los_simulated="?",
                speed_delta_mph=0.0, density_delta=0.0,
            ))

        cos_sims = [r.cosine_similarity for r in results]
        l2_dists = [r.l2_distance for r in results]

        report = VariantComparisonReport(
            variant_name=variant_name,
            lane_results=results,
            mean_cosine_similarity=float(np.mean(cos_sims)),
            mean_l2_distance=float(np.mean(l2_dists)),
            los_accuracy=0.0,
            mean_speed_delta=0.0,
            mean_density_delta=0.0,
        )

        logger.info(
            f"Variant '{variant_name}' (NN match): {len(results)} lanes | "
            f"cos_sim={report.mean_cosine_similarity:.3f} | "
            f"L2={report.mean_l2_distance:.3f}"
        )
        return report

    @staticmethod
    def format_report(report: VariantComparisonReport) -> str:
        """Format a report as a human-readable string."""
        lines = [
            f"=== Variant: {report.variant_name} ===",
            f"  Lanes compared: {len(report.lane_results)}",
            f"  Mean cosine similarity: {report.mean_cosine_similarity:.4f}",
            f"  Mean L2 distance: {report.mean_l2_distance:.4f}",
            f"  LOS accuracy: {report.los_accuracy:.1%}",
            f"  Mean speed delta: {report.mean_speed_delta:+.1f} mph",
            f"  Mean density delta: {report.mean_density_delta:+.1f} veh/mi/ln",
            "",
            "  Per-lane results:",
        ]
        for r in report.lane_results:
            lines.append(
                f"    {r.lane_id}: cos={r.cosine_similarity:.3f} "
                f"L2={r.l2_distance:.3f} "
                f"LOS {r.los_real}→{r.los_simulated} "
                f"{'✓' if r.los_match else '✗'}"
            )
        return "\n".join(lines)
