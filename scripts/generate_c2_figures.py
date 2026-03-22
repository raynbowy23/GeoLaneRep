#!/usr/bin/env python3
"""C2 Figures: Re-encoding Fidelity Visualization.

Generates paper figures from pre-computed C2 synthesis loop results:
  - C2a: Per-camera cosine similarity bar chart with aggregate reference
  - C2b: Per-lane similarity distribution (box plot per camera)
  - C2c: Embedding space comparison (real vs re-encoded, 2D projection)

Reads from: results/c2_synthesis_loop/

Usage:
    python scripts/generate_c2_figures.py
    python scripts/generate_c2_figures.py --figures C2a C2b C2c
    python scripts/generate_c2_figures.py --results-dir results/c2_synthesis_loop
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Camera display colors
_CAMERA_COLORS = {
    "I43_Keefe": "#e6194b",
    "I43_Walnut": "#3cb44b",
    "US12_CountyAB": "#4363d8",
    "US12_Greenway": "#f58231",
    "US12_JohnNolen": "#911eb4",
    "US12_Monona": "#42d4f4",
    "US12_Park": "#f032e6",
    "US12_Stoughton": "#bfef45",
    "US12_Todd": "#fabed4",
    "US12_Whitney": "#469990",
    "US12_Yahara": "#dcbeff",
}


def _load_results(results_dir: Path):
    """Load C2 experiment results."""
    summary_path = results_dir / "c2_summary.json"
    results_path = results_dir / "c2_results.json"

    if not summary_path.exists():
        logger.error(f"No C2 summary found at {summary_path}")
        logger.error("Run 'make c2-experiment-all' first.")
        return None, None

    with open(summary_path) as f:
        summary = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    return summary, results


# ---------------------------------------------------------------------------
# C2a: Per-camera cosine similarity bar chart
# ---------------------------------------------------------------------------

def figure_c2a(summary, results, output_dir: Path):
    """Bar chart of re-encoding cosine similarity per camera.

    Shows how faithfully each camera's behavioral embeddings are preserved
    through the SUMO simulation loop. Aggregate mean shown as dashed line.
    """
    cameras = [r["camera"] for r in results]
    cos_sims = [r["mean_cosine_similarity"] for r in results]
    n_lanes = [r["n_matched_lanes"] for r in results]

    agg = summary["aggregate"]
    mean_cos = agg["mean_cosine_similarity"]
    std_cos = agg["std_cosine_similarity"]

    # Sort by cosine similarity
    order = np.argsort(cos_sims)[::-1]
    cameras = [cameras[i] for i in order]
    cos_sims = [cos_sims[i] for i in order]
    n_lanes = [n_lanes[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [_CAMERA_COLORS.get(c, "#888888") for c in cameras]
    bars = ax.barh(range(len(cameras)), cos_sims, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)

    # Aggregate reference line
    ax.axvline(mean_cos, color="black", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Mean: {mean_cos:.3f} ± {std_cos:.3f}")

    # Fidelity regions
    ax.axvspan(0.5, 1.0, alpha=0.05, color="green")
    ax.axvspan(0.3, 0.5, alpha=0.05, color="cyan")
    ax.axvspan(0.0, 0.3, alpha=0.05, color="orange")

    ax.set_yticks(range(len(cameras)))
    ax.set_yticklabels([f"{c} ({n})" for c, n in zip(cameras, n_lanes)], fontsize=9)
    ax.set_xlabel("Cosine Similarity (real ↔ re-encoded)", fontsize=11)
    ax.set_title("C2: Re-encoding Fidelity per Camera", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(cos_sims) * 1.15)
    ax.invert_yaxis()

    # Value labels on bars
    for i, (v, n) in enumerate(zip(cos_sims, n_lanes)):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Annotation
    ax.text(0.98, 0.02,
            f"{agg['total_matched_lanes']} lanes across {summary['n_cameras']} cameras\n"
            f"SUMO → FCD → encoder → cosine similarity",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    path = output_dir / "C2a_reencoding_fidelity.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved C2a to {path}")


# ---------------------------------------------------------------------------
# C2b: Per-lane similarity distribution (box plot)
# ---------------------------------------------------------------------------

def figure_c2b(summary, results, output_dir: Path):
    """Box plot of per-lane cosine similarities grouped by camera.

    Shows the distribution of lane-level re-encoding fidelity, revealing
    which lanes are well-captured vs poorly-captured by SUMO simulation.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort cameras by mean cosine similarity
    results_sorted = sorted(results, key=lambda r: r["mean_cosine_similarity"], reverse=True)

    box_data = []
    labels = []
    colors_list = []

    for r in results_sorted:
        per_lane = r.get("per_lane", [])
        if not per_lane:
            continue
        sims = [l["cosine_similarity"] for l in per_lane]
        box_data.append(sims)
        labels.append(r["camera"])
        colors_list.append(_CAMERA_COLORS.get(r["camera"], "#888888"))

    bp = ax.boxplot(
        box_data, vert=True, patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (data, color) in enumerate(zip(box_data, colors_list)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter(
            [i + 1 + j for j in jitter], data,
            c=color, s=20, alpha=0.6, edgecolors="white", linewidths=0.3,
            zorder=3,
        )

    # Aggregate reference
    agg = summary["aggregate"]
    ax.axhline(agg["mean_cosine_similarity"], color="black", linestyle="--",
               linewidth=1.2, alpha=0.6,
               label=f"Aggregate mean: {agg['mean_cosine_similarity']:.3f}")

    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cosine Similarity (real ↔ re-encoded)", fontsize=11)
    ax.set_title("C2: Per-Lane Re-encoding Fidelity Distribution", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "C2b_lane_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved C2b to {path}")


# ---------------------------------------------------------------------------
# C2c: L2 distance vs cosine similarity scatter
# ---------------------------------------------------------------------------

def figure_c2c(summary, results, output_dir: Path):
    """Scatter: per-lane cosine similarity vs L2 distance, colored by camera.

    Shows the relationship between the two embedding space metrics and
    highlights the domain gap structure across cameras.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        per_lane = r.get("per_lane", [])
        if not per_lane:
            continue
        cam = r["camera"]
        color = _CAMERA_COLORS.get(cam, "#888888")
        cos_vals = [l["cosine_similarity"] for l in per_lane]
        l2_vals = [l["l2_distance"] for l in per_lane]
        ax.scatter(cos_vals, l2_vals, c=color, label=cam, s=40, alpha=0.7,
                   edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("L2 Distance", fontsize=11)
    ax.set_title("C2: Embedding Space Metrics per Lane", fontsize=13, fontweight="bold")

    # Expected: negative correlation (high cos_sim → low L2)
    ax.legend(
        loc="upper right", fontsize=7, ncol=2,
        framealpha=0.8, markerscale=0.8,
    )
    ax.grid(alpha=0.3)

    # Annotate aggregate
    agg = summary["aggregate"]
    ax.axvline(agg["mean_cosine_similarity"], color="gray", linestyle=":", alpha=0.5)
    ax.axhline(agg["mean_l2_distance"], color="gray", linestyle=":", alpha=0.5)
    ax.text(
        agg["mean_cosine_similarity"] + 0.01, agg["mean_l2_distance"] + 0.3,
        f"({agg['mean_cosine_similarity']:.3f}, {agg['mean_l2_distance']:.1f})",
        fontsize=8, color="gray",
    )

    fig.tight_layout()
    path = output_dir / "C2c_embedding_metrics.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved C2c to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate C2 re-encoding fidelity figures")
    parser.add_argument("--results-dir", default="results/c2_synthesis_loop",
                        help="Directory with C2 experiment results")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as results-dir)")
    parser.add_argument("--figures", nargs="+", default=["C2a", "C2b", "C2c"],
                        help="Which figures to generate")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, results = _load_results(results_dir)
    if summary is None:
        return

    logger.info(f"Loaded C2 results: {summary['n_cameras']} cameras, "
                f"{summary['aggregate']['total_matched_lanes']} lanes")

    figure_map = {
        "C2a": figure_c2a,
        "C2b": figure_c2b,
        "C2c": figure_c2c,
    }

    for fig_name in args.figures:
        if fig_name in figure_map:
            logger.info(f"Generating {fig_name}...")
            figure_map[fig_name](summary, results, output_dir)
        else:
            logger.warning(f"Unknown figure: {fig_name}")

    logger.info(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
