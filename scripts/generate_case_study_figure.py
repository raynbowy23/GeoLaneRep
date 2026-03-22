#!/usr/bin/env python3
"""Case Study Figures: DT-relevant downstream use demonstration.

Two case studies from the planning evaluation:
  - US12_Yahara: "Win-win" — both speed and density improve
  - US12_Whitney: Congested corridor — largest density relief, LOS B present

Each case study shows:
  Left:  Before/after traffic metrics comparison
  Right: Per-lane density redistribution

Usage:
    python scripts/generate_case_study_figure.py
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOS_COLORS = {
    "A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22",
    "D": "#e74c3c", "E": "#c0392b", "F": "#8e44ad",
}


def load_camera_result(results_dir: Path, camera: str) -> dict:
    path = results_dir / f"{camera}.json"
    with open(path) as f:
        return json.load(f)


def plot_case_study(ax_metrics, ax_lanes, result, camera_label):
    """Plot a single case study across two axes."""
    baseline = result["baseline"]
    modified = result["modified"]
    deltas = result["deltas"]
    variant = result["variant"]

    # -- Left: Before/After grouped bars for speed, density, flow --
    metrics = ["Speed\n(mph)", "Density\n(veh/mi/ln)", "Flow\n(veh/hr)"]
    before_vals = [
        baseline["mean_speed_mph"],
        baseline["mean_density_veh_mi_ln"],
        baseline["mean_flow_veh_hr"] / 10,  # scale for visual
    ]
    after_vals = [
        modified["mean_speed_mph"],
        modified["mean_density_veh_mi_ln"],
        modified["mean_flow_veh_hr"] / 10,
    ]

    x = np.arange(len(metrics))
    w = 0.3
    bars_before = ax_metrics.bar(x - w/2, before_vals, w, label="Before",
                                  color="#5b9bd5", edgecolor="white", linewidth=0.5)
    bars_after = ax_metrics.bar(x + w/2, after_vals, w, label="After",
                                 color="#ed7d31", edgecolor="white", linewidth=0.5)

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metrics, fontsize=9)
    ax_metrics.set_ylabel("Value", fontsize=9)
    ax_metrics.legend(fontsize=8, loc="upper right")
    ax_metrics.grid(axis="y", alpha=0.3)

    # Delta annotations
    delta_speed = deltas["delta_speed_mph"]
    delta_density = deltas["delta_density_veh_mi_ln"]
    speed_sign = "+" if delta_speed >= 0 else ""
    density_sign = "+" if delta_density >= 0 else ""

    # LOS summary
    los_before = baseline["los_distribution"]
    los_after = modified["los_distribution"]
    los_before_str = ", ".join(f"{k}:{v}" for k, v in sorted(los_before.items()))
    los_after_str = ", ".join(f"{k}:{v}" for k, v in sorted(los_after.items()))

    title_color = "#2ecc71" if delta_density < 0 else "#e74c3c"
    ax_metrics.set_title(
        f"{camera_label}\n"
        f"Variant: {variant['name']} (gen={variant['generation_score']:.3f})\n"
        f"Speed: {speed_sign}{delta_speed:.2f} mph | "
        f"Density: {density_sign}{delta_density:.2f} veh/mi/ln\n"
        f"LOS: [{los_before_str}] -> [{los_after_str}]",
        fontsize=9, fontweight="bold", loc="left",
        color="#333333",
    )

    # -- Right: Per-lane density redistribution --
    before_lanes = baseline["per_lane"]
    after_lanes = modified["per_lane"]

    # Show before lanes
    before_ids = [l["lane_id"].split("_")[-1] for l in before_lanes]
    before_densities = [l["density_veh_mi_ln"] for l in before_lanes]
    before_los = [l["los"] for l in before_lanes]

    after_ids = [l["lane_id"].split("_")[-1] for l in after_lanes]
    after_densities = [l["density_veh_mi_ln"] for l in after_lanes]
    after_los = [l["los"] for l in after_lanes]

    # Combine for grouped bar
    n_before = len(before_lanes)
    n_after = len(after_lanes)
    max_n = max(n_before, n_after)

    y_before = np.arange(n_before)
    y_after = np.arange(n_after)

    h = 0.35
    # Before bars
    b_colors = [LOS_COLORS.get(l, "#808080") for l in before_los]
    ax_lanes.barh(y_before + h/2, before_densities, h, color=b_colors,
                   edgecolor="white", linewidth=0.3, alpha=0.7, label="Before")

    # After bars
    a_colors = [LOS_COLORS.get(l, "#808080") for l in after_los]
    ax_lanes.barh(y_after[:n_after] - h/2, after_densities, h, color=a_colors,
                   edgecolor="white", linewidth=0.3, alpha=0.9, label="After")

    # Labels
    all_ids = before_ids if n_before >= n_after else after_ids
    ax_lanes.set_yticks(range(max_n))
    ax_lanes.set_yticklabels([f"lane_{i}" for i in range(max_n)], fontsize=7)
    ax_lanes.set_xlabel("Density (veh/mi/ln)", fontsize=9)
    ax_lanes.set_title("Per-Lane Density Redistribution", fontsize=9)
    ax_lanes.invert_yaxis()
    ax_lanes.grid(axis="x", alpha=0.3)

    # Highlight new lane
    if n_after > n_before:
        ax_lanes.annotate(
            "NEW", (after_densities[-1], n_after - 1 - h/2),
            fontsize=7, fontweight="bold", color="#e74c3c",
            xytext=(10, 0), textcoords="offset points",
        )

    # Legend for LOS colors
    handles = [mpatches.Patch(facecolor=LOS_COLORS[g], label=f"LOS {g}")
               for g in ["A", "B"] if any(
                   l["los"] == g for l in before_lanes + after_lanes
               )]
    if handles:
        ax_lanes.legend(handles=handles, fontsize=7, loc="lower right")


def main():
    results_dir = Path("results/planning_eval")
    output_dir = Path("results/case_studies")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Case study cameras
    cases = [
        ("US12_Yahara", "Case Study 1: US12_Yahara — Win-Win Scenario"),
        ("US12_Whitney", "Case Study 2: US12_Whitney — Congested Corridor Relief"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, (camera, label) in enumerate(cases):
        result = load_camera_result(results_dir, camera)
        plot_case_study(axes[row, 0], axes[row, 1], result, label)

    fig.suptitle(
        "GeoLane Planning Evaluation: Case Studies\n"
        "Encoder -> Generator -> SUMO -> Re-encode -> Compare",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = output_dir / "case_studies_planning.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved case studies to {path}")

    # Also save a text summary
    summary_path = output_dir / "case_study_notes.md"
    notes = []
    for camera, label in cases:
        r = load_camera_result(results_dir, camera)
        d = r["deltas"]
        v = r["variant"]
        emb = r.get("embedding_shift", {})
        notes.append(f"""## {label}

- **Camera**: {camera}
- **Variant**: {v['name']} (generated by {v['generated_by']}, score={v['generation_score']:.3f})
- **Target edge**: {r['target_edge']}
- **Baseline**: {r['baseline']['n_lanes']} lanes, speed={r['baseline']['mean_speed_mph']:.1f} mph, density={r['baseline']['mean_density_veh_mi_ln']:.1f} veh/mi/ln
- **Modified**: {r['modified']['n_lanes']} lanes, speed={r['modified']['mean_speed_mph']:.1f} mph, density={r['modified']['mean_density_veh_mi_ln']:.1f} veh/mi/ln
- **Impact**: speed {'improved' if d['speed_improved'] else 'decreased'} by {abs(d['delta_speed_mph']):.2f} mph, density {'improved' if d['density_improved'] else 'increased'} by {abs(d['delta_density_veh_mi_ln']):.2f} veh/mi/ln
- **LOS**: {r['baseline']['los_distribution']} -> {r['modified']['los_distribution']}
- **Embedding shift**: cos_sim={emb.get('mean_cosine_similarity', 'N/A')}, L2={emb.get('mean_l2_distance', 'N/A')}
- **Narrative**: {"Both speed and density improved — the generated leftmost lane redistributed traffic effectively." if camera == "US12_Yahara" else "Largest density reduction across all cameras. The only corridor with LOS B baseline. Generated rightmost lane absorbed off-ramp traffic."}
""")

    summary_path.write_text("\n".join(notes))
    logger.info(f"Saved case study notes to {summary_path}")


if __name__ == "__main__":
    main()
