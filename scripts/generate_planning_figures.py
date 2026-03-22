#!/usr/bin/env python3
"""Planning Evaluation Figures (D2-series).

Generates paper figures from pre-computed planning evaluation results:
  - D2a: Per-camera traffic impact summary (ΔSpeed, ΔDensity bar chart)
  - D2b: LOS distribution before/after (grouped bar chart)
  - D2c: Generation score vs embedding shift scatter

Reads from: results/planning_eval/

Usage:
    python scripts/generate_planning_figures.py
    python scripts/generate_planning_figures.py --figures D2a D2b D2c
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

LOS_GRADES = ["A", "B", "C", "D", "E", "F"]
LOS_COLORS = {
    "A": "#2ecc71", "B": "#27ae60", "C": "#f1c40f",
    "D": "#e67e22", "E": "#e74c3c", "F": "#c0392b",
}


def _load_results(results_dir: Path):
    """Load planning evaluation results."""
    summary_path = results_dir / "planning_summary.json"
    results_path = results_dir / "planning_results.json"

    if not summary_path.exists():
        logger.error(f"No planning summary at {summary_path}")
        logger.error("Run 'make planning-eval-all' first.")
        return None, None

    with open(summary_path) as f:
        summary = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    return summary, results


# ---------------------------------------------------------------------------
# D2a: Traffic impact per camera (ΔSpeed + ΔDensity grouped bars)
# ---------------------------------------------------------------------------

def figure_d2a(summary, results, output_dir: Path):
    """Grouped bar chart: ΔSpeed and ΔDensity per camera with variant labels."""
    cameras = [r["camera"] for r in results]
    delta_speeds = [r["deltas"]["delta_speed_mph"] for r in results]
    delta_densities = [r["deltas"]["delta_density_veh_mi_ln"] for r in results]
    variants = [r["variant"]["name"] for r in results]
    gen_scores = [r["variant"].get("generation_score") for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(cameras))
    colors = [_CAMERA_COLORS.get(c, "#888888") for c in cameras]

    # Speed deltas
    bars1 = ax1.bar(x, delta_speeds, color=colors, edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("ΔSpeed (mph)", fontsize=11)
    ax1.set_title("Planning Evaluation: Traffic Impact of Generated Lane Variants",
                   fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Annotate with variant type
    for i, (ds, v) in enumerate(zip(delta_speeds, variants)):
        ax1.text(i, ds + (0.02 if ds >= 0 else -0.02), v.replace("add_", "+"),
                 ha="center", va="bottom" if ds >= 0 else "top",
                 fontsize=7, rotation=45, color="gray")

    # Density deltas (inverted: negative = improvement)
    bars2 = ax2.bar(x, delta_densities, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("ΔDensity (veh/mi/ln)", fontsize=11)
    ax2.set_xlabel("Camera", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate with gen scores
    for i, (dd, gs) in enumerate(zip(delta_densities, gen_scores)):
        if gs is not None:
            ax2.text(i, dd + (0.02 if dd >= 0 else -0.02),
                     f"g={gs:.2f}", ha="center",
                     va="bottom" if dd >= 0 else "top",
                     fontsize=7, color="gray")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cameras, rotation=45, ha="right", fontsize=9)

    # Aggregate annotation
    agg = summary["aggregate"]
    fig.text(0.99, 0.01,
             f"Aggregate: ΔSpeed={agg['mean_delta_speed_mph']:+.2f} mph, "
             f"ΔDensity={agg['mean_delta_density']:+.2f} veh/mi/ln | "
             f"{agg['speed_improved_count']}/{summary['n_cameras']} cameras improved speed, "
             f"{agg['density_improved_count']}/{summary['n_cameras']} improved density",
             ha="right", va="bottom", fontsize=8, color="gray", style="italic",
             transform=fig.transFigure)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = output_dir / "D2a_traffic_impact.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved D2a to {path}")


# ---------------------------------------------------------------------------
# D2b: LOS distribution before/after
# ---------------------------------------------------------------------------

def figure_d2b(summary, results, output_dir: Path):
    """Stacked bar chart: LOS distribution before and after lane addition."""
    cameras = [r["camera"] for r in results]
    n = len(cameras)

    fig, ax = plt.subplots(figsize=(12, 5))

    bar_width = 0.35
    x = np.arange(n)

    # Build LOS counts matrices
    for side, offset, label_suffix in [
        ("baseline", -bar_width/2, "Before"),
        ("modified", bar_width/2, "After"),
    ]:
        bottom = np.zeros(n)
        for grade in LOS_GRADES:
            counts = []
            for r in results:
                los_dist = r[side]["los_distribution"]
                counts.append(los_dist.get(grade, 0))
            counts = np.array(counts, dtype=float)
            if counts.sum() == 0:
                continue
            ax.bar(x + offset, counts, bar_width, bottom=bottom,
                   color=LOS_COLORS[grade], edgecolor="white", linewidth=0.3,
                   label=f"LOS {grade} ({label_suffix})" if offset < 0 else f"_LOS {grade} ({label_suffix})")
            bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels(cameras, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Lanes", fontsize=11)
    ax.set_title("LOS Distribution: Before vs After Lane Addition",
                 fontsize=13, fontweight="bold")

    # Add Before/After labels
    ax.text(x[0] - bar_width/2, ax.get_ylim()[1] * 0.95, "Before",
            ha="center", fontsize=8, color="gray")
    ax.text(x[0] + bar_width/2, ax.get_ylim()[1] * 0.95, "After",
            ha="center", fontsize=8, color="gray")

    # Simplified legend (just LOS grades)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=LOS_COLORS[g], label=f"LOS {g}") for g in LOS_GRADES
               if any(r["baseline"]["los_distribution"].get(g, 0) +
                      r["modified"]["los_distribution"].get(g, 0) > 0 for r in results)]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "D2b_los_distribution.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved D2b to {path}")


# ---------------------------------------------------------------------------
# D2c: Generation score vs embedding shift
# ---------------------------------------------------------------------------

def figure_d2c(summary, results, output_dir: Path):
    """Scatter: generation score vs re-encoding cosine similarity per camera."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        cam = r["camera"]
        gs = r["variant"].get("generation_score")
        emb = r.get("embedding_shift", {}).get("mean_cosine_similarity")
        if gs is None or emb is None:
            continue

        color = _CAMERA_COLORS.get(cam, "#888888")
        ax.scatter(gs, emb, c=color, s=100, edgecolors="white", linewidths=0.5,
                   zorder=3)
        ax.annotate(cam.replace("US12_", "").replace("I43_", ""),
                    (gs, emb), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="gray")

    ax.set_xlabel("Generation Score (diffusion quality)", fontsize=11)
    ax.set_ylabel("Re-encoding Cosine Similarity", fontsize=11)
    ax.set_title("Generation Quality vs Loop Closure Fidelity",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # Add aggregate reference
    agg = summary.get("aggregate", {})
    emb_shift = agg.get("mean_embedding_shift")
    if emb_shift:
        ax.axhline(emb_shift, color="gray", linestyle=":", alpha=0.5,
                   label=f"Mean embed shift: {emb_shift:.3f}")
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = output_dir / "D2c_gen_vs_embedding.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved D2c to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate planning evaluation figures")
    parser.add_argument("--results-dir", default="results/planning_eval")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figures", nargs="+", default=["D2a", "D2b", "D2c"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, results = _load_results(results_dir)
    if summary is None:
        return

    logger.info(f"Loaded planning results: {summary['n_cameras']} cameras")

    figure_map = {
        "D2a": figure_d2a,
        "D2b": figure_d2b,
        "D2c": figure_d2c,
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
