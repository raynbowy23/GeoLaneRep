#!/usr/bin/env python3
"""Generate the GeoLane framework overview figure.

This is the thesis-defining conceptual figure showing the full GeoLane
contribution: multi-modal lane observations -> contrastive lane encoder ->
digital twin downstream uses, with a closed synthesis loop.

No data loading required -- pure programmatic drawing.

Usage:
    python scripts/generate_framework_figure.py

Outputs:
    results/framework_figure.png
    results/framework_figure.pdf
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines
import numpy as np

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG_COLOR = "#FFFFFF"
INPUT_COLOR = "#D6E6F0"       # muted light blue
ENCODER_COLOR = "#A8D5BA"     # muted green (hero block)
ENCODER_BORDER = "#3A7D5C"    # darker green border
DT_COLOR = "#E8DFF0"          # muted lavender
LOOP_COLOR = "#FFF3CD"        # muted warm yellow
ARROW_COLOR = "#555555"
TEXT_COLOR = "#222222"
SUBTITLE_COLOR = "#555555"

FONT_TITLE = {"fontsize": 16, "fontweight": "bold", "color": TEXT_COLOR,
               "fontfamily": "serif"}
FONT_BLOCK_TITLE = {"fontsize": 12, "fontweight": "bold", "color": TEXT_COLOR,
                     "fontfamily": "serif"}
FONT_BODY = {"fontsize": 9, "color": TEXT_COLOR, "fontfamily": "serif"}
FONT_SMALL = {"fontsize": 8, "color": SUBTITLE_COLOR, "fontfamily": "serif"}
FONT_BULLET = {"fontsize": 8.5, "color": TEXT_COLOR, "fontfamily": "serif"}
FONT_ARROW = {"fontsize": 8, "color": ARROW_COLOR, "fontfamily": "serif",
              "fontstyle": "italic"}


def _rounded_box(ax, xy, width, height, color, edgecolor=None, linewidth=1.5,
                 zorder=2, alpha=1.0):
    """Draw a rounded rectangle and return the patch."""
    ec = edgecolor or color
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor=ec,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, xy_start, xy_end, label=None, connectionstyle="arc3,rad=0",
           color=ARROW_COLOR, linewidth=1.8):
    """Draw a fancy arrow between two points, optionally labelled."""
    arrow = FancyArrowPatch(
        xy_start, xy_end,
        arrowstyle="-|>",
        connectionstyle=connectionstyle,
        color=color,
        linewidth=linewidth,
        mutation_scale=14,
        zorder=5,
    )
    ax.add_patch(arrow)
    if label:
        mx = (xy_start[0] + xy_end[0]) / 2
        my = (xy_start[1] + xy_end[1]) / 2
        ax.text(mx, my + 0.025, label, ha="center", va="bottom", **FONT_ARROW)


def _draw_icon_trajectories(ax, cx, cy, size=0.04):
    """Tiny wavy-line icon representing camera trajectories."""
    xs = np.linspace(cx - size, cx + size, 40)
    for offset in [-0.008, 0.0, 0.008]:
        ys = cy + offset + 0.004 * np.sin(8 * np.pi * (xs - cx) / (2 * size))
        ax.plot(xs, ys, color="#3B7DD8", linewidth=1.0, zorder=6)


def _draw_icon_lanes(ax, cx, cy, size=0.04):
    """Tiny parallel-lines icon representing lane geometry."""
    for offset in [-0.01, 0.0, 0.01]:
        ax.plot([cx - size, cx + size], [cy + offset, cy + offset],
                color="#3B7DD8", linewidth=1.0, zorder=6)


def _draw_icon_graph(ax, cx, cy, size=0.025):
    """Tiny graph-node icon representing lane connectivity."""
    nodes = [(cx - size, cy), (cx + size, cy),
             (cx, cy + size * 0.9), (cx, cy - size * 0.9)]
    # edges
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ax.plot([nodes[i][0], nodes[j][0]],
                    [nodes[i][1], nodes[j][1]],
                    color="#888888", linewidth=0.6, zorder=6)
    for (nx, ny) in nodes:
        ax.plot(nx, ny, "o", color="#3B7DD8", markersize=3, zorder=7)


def _draw_dt_sub_box(ax, x, y, w, h, title, subtitle, color="#D8CCE8"):
    """Small sub-box inside the DT-uses block."""
    _rounded_box(ax, (x, y), w, h, color=color, edgecolor="#9B86B3",
                 linewidth=1.0, zorder=3)
    ax.text(x + w / 2, y + h * 0.62, title,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=TEXT_COLOR, fontfamily="serif", zorder=4)
    ax.text(x + w / 2, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=7, color=SUBTITLE_COLOR,
            fontfamily="serif", zorder=4)


def generate_framework_figure():
    """Create and save the GeoLane framework overview figure."""

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.text(0.50, 0.96,
            "GeoLane: Behavioral Geometry Layer for Digital Twins",
            ha="center", va="top", **FONT_TITLE)

    # ==================================================================
    # Block 1 -- INPUT  (left)
    # ==================================================================
    bx1, by1, bw1, bh1 = 0.03, 0.22, 0.22, 0.62
    _rounded_box(ax, (bx1, by1), bw1, bh1, INPUT_COLOR,
                 edgecolor="#8FBBD9", linewidth=1.5)

    ax.text(bx1 + bw1 / 2, by1 + bh1 - 0.04,
            "Multi-modal Lane\nObservations", ha="center", va="top",
            **FONT_BLOCK_TITLE)

    # Sub-items with icons
    items_y0 = by1 + bh1 - 0.18
    item_gap = 0.155
    icon_x = bx1 + 0.045
    text_x = bx1 + 0.09

    # Camera trajectories
    _draw_icon_trajectories(ax, icon_x, items_y0)
    ax.text(text_x, items_y0, "Camera trajectories",
            va="center", **FONT_BODY)

    # Lane geometry
    _draw_icon_lanes(ax, icon_x, items_y0 - item_gap)
    ax.text(text_x, items_y0 - item_gap, "Lane geometry",
            va="center", **FONT_BODY)

    # Lane connectivity / roles
    _draw_icon_graph(ax, icon_x, items_y0 - 2 * item_gap)
    ax.text(text_x, items_y0 - 2 * item_gap, "Lane connectivity / roles",
            va="center", **FONT_BODY)

    # ==================================================================
    # Block 2 -- GEOLANE ENCODER  (center, highlighted)
    # ==================================================================
    bx2, by2, bw2, bh2 = 0.32, 0.17, 0.28, 0.72
    _rounded_box(ax, (bx2, by2), bw2, bh2, ENCODER_COLOR,
                 edgecolor=ENCODER_BORDER, linewidth=2.5)

    # Inner glow effect -- slightly smaller, lighter box
    _rounded_box(ax, (bx2 + 0.008, by2 + 0.008),
                 bw2 - 0.016, bh2 - 0.016,
                 color="#C5E8D5", edgecolor="none", linewidth=0, zorder=2.5,
                 alpha=0.55)

    ax.text(bx2 + bw2 / 2, by2 + bh2 - 0.04,
            "GeoLane Encoder", ha="center", va="top",
            fontsize=14, fontweight="bold", color="#2A5E3F",
            fontfamily="serif", zorder=4)

    # Sub-title
    ax.text(bx2 + bw2 / 2, by2 + bh2 - 0.12,
            "Contrastive Lane Encoder", ha="center", va="top",
            fontsize=10, fontweight="bold", color=TEXT_COLOR,
            fontfamily="serif", zorder=4)

    ax.text(bx2 + bw2 / 2, by2 + bh2 - 0.19,
            "128-dim behavioral embedding", ha="center", va="top",
            fontsize=9, color=SUBTITLE_COLOR, fontfamily="serif",
            fontstyle="italic", zorder=4)

    # Key property bullets
    bullet_x = bx2 + 0.03
    bullet_y0 = by2 + bh2 - 0.30
    bullet_gap = 0.065
    properties = [
        "Captures role, not just shape",
        "Generalizes across cameras",
        "Zero-shot transfer",
    ]
    for i, prop in enumerate(properties):
        y = bullet_y0 - i * bullet_gap
        # bullet marker
        ax.plot(bullet_x, y, "s", color=ENCODER_BORDER, markersize=4,
                zorder=4)
        ax.text(bullet_x + 0.02, y, prop, va="center", **FONT_BULLET,
                zorder=4)

    # Star badge -- "main contribution"
    badge_y = by2 + 0.02
    ax.text(bx2 + bw2 / 2, badge_y,
            "Main Contribution",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=ENCODER_BORDER,
            fontfamily="serif", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F5E9",
                      edgecolor=ENCODER_BORDER, linewidth=1.0),
            zorder=6)

    # ==================================================================
    # Block 3 -- DT USES  (right)
    # ==================================================================
    bx3, by3, bw3, bh3 = 0.68, 0.22, 0.28, 0.62
    _rounded_box(ax, (bx3, by3), bw3, bh3, DT_COLOR,
                 edgecolor="#B0A0C0", linewidth=1.5)

    ax.text(bx3 + bw3 / 2, by3 + bh3 - 0.04,
            "Digital Twin Applications", ha="center", va="top",
            **FONT_BLOCK_TITLE)

    # 4 sub-boxes (2 x 2 grid)
    sub_w, sub_h = 0.115, 0.11
    pad_x, pad_y = 0.015, 0.02
    grid_x0 = bx3 + (bw3 - 2 * sub_w - pad_x) / 2
    grid_y0 = by3 + 0.05

    dt_items = [
        ("Lane\nComparison", "cross-site similarity"),
        ("Variant\nGeneration", "diffusion-based"),
        ("Anomaly\nDetection", "temporal monitoring"),
        ("Synthesis\nLoop", "SUMO re-encode"),
    ]
    for idx, (title, sub) in enumerate(dt_items):
        col = idx % 2
        row = 1 - idx // 2   # top row first
        x = grid_x0 + col * (sub_w + pad_x)
        y = grid_y0 + row * (sub_h + pad_y)
        _draw_dt_sub_box(ax, x, y, sub_w, sub_h, title, sub)

    # ==================================================================
    # Arrows between blocks
    # ==================================================================
    # Input -> Encoder
    _arrow(ax, (bx1 + bw1, by1 + bh1 / 2),
           (bx2, by2 + bh2 / 2),
           label="features")

    # Encoder -> DT Uses
    _arrow(ax, (bx2 + bw2, by2 + bh2 / 2),
           (bx3, by3 + bh3 / 2),
           label="embeddings")

    # ==================================================================
    # Closed-loop arrow (below main blocks)
    # ==================================================================
    loop_y = 0.10
    loop_x0 = bx2 + bw2 / 2 - 0.02
    loop_x1 = bx3 + bw3 / 2 + 0.02

    # Background ribbon for the loop
    _rounded_box(ax, (loop_x0 - 0.04, loop_y - 0.045),
                 loop_x1 - loop_x0 + 0.08, 0.09,
                 LOOP_COLOR, edgecolor="#D4C27A", linewidth=1.0, zorder=1)

    loop_labels = ["encode", "generate", "simulate", "re-encode", "compare"]
    n = len(loop_labels)
    xs = np.linspace(loop_x0, loop_x1, n)
    for i, lbl in enumerate(loop_labels):
        ax.text(xs[i], loop_y, lbl, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#7A6B20",
                fontfamily="serif", zorder=4)
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.025, loop_y),
                        xytext=(xs[i] + 0.025, loop_y),
                        arrowprops=dict(arrowstyle="-|>", color="#A08C30",
                                        linewidth=1.2),
                        zorder=4)

    # Return arrow (compare -> encode) -- curved below
    ax.annotate("", xy=(xs[0], loop_y - 0.03),
                xytext=(xs[-1], loop_y - 0.03),
                arrowprops=dict(arrowstyle="-|>", color="#A08C30",
                                linewidth=1.2,
                                connectionstyle="arc3,rad=0.35"),
                zorder=4)

    ax.text((xs[0] + xs[-1]) / 2, loop_y - 0.065, "closed synthesis loop",
            ha="center", va="top", fontsize=7, color="#7A6B20",
            fontfamily="serif", fontstyle="italic", zorder=4)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "framework_figure.png", dpi=200,
                bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig(out_dir / "framework_figure.pdf",
                bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)

    print(f"Saved: {out_dir / 'framework_figure.png'}")
    print(f"Saved: {out_dir / 'framework_figure.pdf'}")


if __name__ == "__main__":
    generate_framework_figure()
