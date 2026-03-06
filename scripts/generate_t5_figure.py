#!/usr/bin/env python3
"""Generate T5 ablation figure: joint vs two-stage comparison.

Loads training history JSON files from each variant and produces:
  - T5a: Training curves (anomaly accuracy over epochs) for all three variants
  - T5b: Contrastive quality (pos_sim, neg_sim) over epochs for joint variants
  - T5c: Summary table printed to console and saved as CSV

Requires that training has been run for:
  (a) Two-stage:        results/temporal_encoder/history.json
  (b) Joint scratch:    results/joint_encoder/history.json        (no warm-start)
  (c) Joint warm-start: results/joint_encoder_warm/history.json   (warm-start)

Usage:
    python scripts/generate_t5_figure.py
    python scripts/generate_t5_figure.py --output-dir results/figures/t5
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

# Default paths for each variant
DEFAULT_PATHS = {
    "(a) Two-stage (frozen)": "results/temporal_encoder/history.json",
    "(b) Joint (scratch)": "results/joint_encoder/history.json",
    "(c) Joint (warm-start)": "results/joint_encoder_warm/history.json",
}


def _load_history(path: str) -> dict:
    """Load training history JSON, return empty dict if missing."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"History not found: {p}")
        return {}
    with open(p) as f:
        return json.load(f)


def _epochs_to_threshold(values: list, threshold: float = 0.8) -> str:
    """Find first epoch where values >= threshold."""
    for i, v in enumerate(values):
        if v >= threshold:
            return str(i + 1)
    return ">{}".format(len(values))


def figure_t5a(histories: dict, output_dir: Path):
    """Training curves: anomaly accuracy over epochs for all variants."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"(a) Two-stage (frozen)": "#2196F3",
              "(b) Joint (scratch)": "#FF9800",
              "(c) Joint (warm-start)": "#4CAF50"}
    linestyles = {"(a) Two-stage (frozen)": "--",
                  "(b) Joint (scratch)": "-",
                  "(c) Joint (warm-start)": "-"}

    for label, hist in histories.items():
        # Two-stage uses train/accuracy, joint uses train/anomaly_accuracy
        acc = hist.get("train/anomaly_accuracy", hist.get("train/accuracy", []))
        if not acc:
            logger.warning(f"No accuracy data for {label}")
            continue
        epochs = np.arange(1, len(acc) + 1)
        ax.plot(epochs, acc, label=label,
                color=colors.get(label, "gray"),
                linestyle=linestyles.get(label, "-"),
                linewidth=2, alpha=0.85)

    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5, label="80% threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Anomaly Detection Accuracy")
    ax.set_title("T5a: Anomaly Accuracy — Two-Stage vs Joint Training")
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "t5a_accuracy_curves.pdf"
    fig.savefig(str(path), dpi=300)
    fig.savefig(str(path.with_suffix(".png")), dpi=150)
    plt.close(fig)
    logger.info(f"Saved T5a to {path}")


def figure_t5b(histories: dict, output_dir: Path):
    """Contrastive quality: pos_sim and neg_sim over epochs (joint variants only)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"(b) Joint (scratch)": "#FF9800",
              "(c) Joint (warm-start)": "#4CAF50"}

    for label, hist in histories.items():
        if "Two-stage" in label:
            continue  # No contrastive metrics for two-stage
        pos = hist.get("train/mean_pos_sim", [])
        neg = hist.get("train/mean_neg_sim", [])
        if not pos:
            continue
        epochs = np.arange(1, len(pos) + 1)

        c = colors.get(label, "gray")
        ax1.plot(epochs, pos, label=f"{label}", color=c, linewidth=2)
        ax2.plot(epochs, neg, label=f"{label}", color=c, linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Positive Similarity")
    ax1.set_title("Positive Pair Similarity")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Negative Similarity")
    ax2.set_title("Negative Pair Similarity")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("T5b: Contrastive Quality — Joint Training Variants", fontsize=13)
    fig.tight_layout()

    path = output_dir / "t5b_contrastive_quality.pdf"
    fig.savefig(str(path), dpi=300)
    fig.savefig(str(path.with_suffix(".png")), dpi=150)
    plt.close(fig)
    logger.info(f"Saved T5b to {path}")


def summary_table(histories: dict, output_dir: Path):
    """Print and save T5 summary table as CSV."""
    rows = []
    header = [
        "Variant", "anomaly_acc", "mean_pos_sim", "mean_neg_sim",
        "pos-neg_gap", "epochs_to_80%",
    ]

    for label, hist in histories.items():
        acc = hist.get("train/anomaly_accuracy", hist.get("train/accuracy", []))
        pos = hist.get("train/mean_pos_sim", [])
        neg = hist.get("train/mean_neg_sim", [])

        final_acc = f"{acc[-1]:.3f}" if acc else "—"
        final_pos = f"{pos[-1]:.3f}" if pos else "—"
        final_neg = f"{neg[-1]:.3f}" if neg else "—"

        if pos and neg:
            gap = f"{pos[-1] - neg[-1]:.3f}"
        else:
            gap = "—"

        e80 = _epochs_to_threshold(acc, 0.8) if acc else "—"

        rows.append([label, final_acc, final_pos, final_neg, gap, e80])

    # Print
    col_widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print("\n" + "=" * 70)
    print("T5 Ablation Summary")
    print("=" * 70)
    print(fmt.format(*header))
    print("-" * sum(col_widths) + "-" * (len(header) - 1) * 2)
    for row in rows:
        print(fmt.format(*row))
    print()

    # Save CSV
    csv_path = output_dir / "t5_summary.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    logger.info(f"Saved summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate T5 ablation figure")
    parser.add_argument(
        "--output-dir", default="results/figures/t5",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--two-stage", default=DEFAULT_PATHS["(a) Two-stage (frozen)"],
        help="Path to two-stage history.json",
    )
    parser.add_argument(
        "--joint-scratch", default=DEFAULT_PATHS["(b) Joint (scratch)"],
        help="Path to joint (scratch) history.json",
    )
    parser.add_argument(
        "--joint-warm", default=DEFAULT_PATHS["(c) Joint (warm-start)"],
        help="Path to joint (warm-start) history.json",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load histories
    histories = {}
    for label, path in [
        ("(a) Two-stage (frozen)", args.two_stage),
        ("(b) Joint (scratch)", args.joint_scratch),
        ("(c) Joint (warm-start)", args.joint_warm),
    ]:
        hist = _load_history(path)
        if hist:
            histories[label] = hist

    if not histories:
        logger.error("No training histories found. Run training first.")
        return

    n_found = len(histories)
    logger.info(f"Found {n_found}/3 training histories")

    figure_t5a(histories, output_dir)
    figure_t5b(histories, output_dir)
    summary_table(histories, output_dir)

    logger.info(f"All T5 figures saved to {output_dir}")


if __name__ == "__main__":
    main()
