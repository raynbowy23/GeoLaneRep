#!/usr/bin/env python3
"""Generate detailed architecture diagrams for each module.

Produces research-quality programmatic figures showing internal architecture
of each module with exact dimensions, layer types, and data flow.

Figures:
    A1 — Lane Encoder: PolylineEncoder internals, fusion, cross-lane attention, heads
    A2 — Temporal Encoder: frozen encoder per window, GRU sequence, anomaly head
    A3 — FiLM Diffusion Generator: denoiser with FiLM layers, DDPM reverse process
    A4 — Full System Pipeline: E→T→M→G→C→D with data flow
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np


# ── Color palette ──────────────────────────────────────────────────────────

COLORS = {
    # Module colors
    "geometry":   "#4A90D9",   # blue
    "trajectory": "#E67E22",   # orange
    "stats":      "#27AE60",   # green
    "fusion":     "#8E44AD",   # purple
    "attention":  "#E74C3C",   # red
    "output":     "#2C3E50",   # dark gray
    "temporal":   "#F39C12",   # gold
    "diffusion":  "#1ABC9C",   # teal
    "film":       "#E74C3C",   # red
    "bridge":     "#3498DB",   # light blue
    "twin":       "#95A5A6",   # gray
    "bg":         "#FAFAFA",   # near white
    "box_bg":     "#FFFFFF",   # white
    "text":       "#2C3E50",   # dark
    "dim":        "#7F8C8D",   # muted
    "highlight":  "#F1C40F",   # yellow accent
}


def _draw_box(ax, x, y, w, h, label, color, fontsize=8, alpha=0.9,
              sublabel=None, bold=False, edgecolor=None, textcolor="white"):
    """Draw a rounded rectangle with centered label."""
    ec = edgecolor or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=ec,
        alpha=alpha, linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2 + (0.015 if sublabel else 0),
        label, ha="center", va="center",
        fontsize=fontsize, fontweight=weight, color=textcolor,
        zorder=4,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.025,
            sublabel, ha="center", va="center",
            fontsize=fontsize - 1.5, color=textcolor, alpha=0.85,
            zorder=4,
        )
    return box


def _draw_arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>",
                lw=1.2, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, connectionstyle=connectionstyle,
        zorder=5, mutation_scale=12,
    )
    ax.add_patch(arrow)
    return arrow


def _dim_label(ax, x, y, text, fontsize=6.5, color=None):
    """Draw a dimension/shape annotation."""
    c = color or COLORS["dim"]
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=c, fontstyle="italic", zorder=4)


# ══════════════════════════════════════════════════════════════════════════
# A1 — Lane Encoder Architecture
# ══════════════════════════════════════════════════════════════════════════

def figure_a1(output_dir: Path):
    """Detailed Lane Encoder architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    # Title
    ax.text(0.5, 1.0, "Lane Encoder Architecture", ha="center", va="top",
            fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(0.5, 0.965, "Multi-modal fusion with optional cross-lane attention",
            ha="center", va="top", fontsize=9, color=COLORS["dim"])

    # ── Input boxes (bottom) ──
    y_input = 0.05
    h_input = 0.055

    # Geometry input
    _draw_box(ax, 0.02, y_input, 0.18, h_input,
              "Lane Geometry", COLORS["geometry"], fontsize=8, bold=True,
              sublabel="(B, K=16, 2)")

    # Trajectory input
    _draw_box(ax, 0.28, y_input, 0.18, h_input,
              "Trajectories", COLORS["trajectory"], fontsize=8, bold=True,
              sublabel="(B, T, K=16, 2)")

    # Trajectory mask
    _draw_box(ax, 0.28, y_input - 0.0, 0.18, h_input,
              "Trajectories", COLORS["trajectory"], fontsize=8, bold=True,
              sublabel="(B, T, K=16, 2)")

    # Stats + roles input
    _draw_box(ax, 0.54, y_input, 0.18, h_input,
              "Stats + Roles", COLORS["stats"], fontsize=8, bold=True,
              sublabel="(B, 9)")

    # ── Encoder blocks (layer 1) ──
    y_enc = 0.18
    h_enc = 0.12

    # Geometry PolylineEncoder
    bx = 0.0
    _draw_box(ax, bx, y_enc, 0.22, h_enc,
              "PolylineEncoder", COLORS["geometry"], fontsize=8, bold=True)
    # Internal details
    ax.text(bx + 0.11, y_enc + 0.085, "Linear(2→64) + SinPosEnc",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.06, "TransformerEncoder ×2",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.035, "  4 heads, FFN=256, GELU",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.013, "Mean Pool → (B, 64)",
            ha="center", fontsize=6, color=COLORS["highlight"])

    # Trajectory PolylineEncoder
    bx = 0.26
    _draw_box(ax, bx, y_enc, 0.22, h_enc,
              "PolylineEncoder", COLORS["trajectory"], fontsize=8, bold=True)
    ax.text(bx + 0.11, y_enc + 0.085, "Linear(2→64) + SinPosEnc",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.06, "TransformerEncoder ×2",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.035, "  4 heads, FFN=256, GELU",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.013, "Per-traj → Mean Pool → (B, 64)",
            ha="center", fontsize=6, color=COLORS["highlight"])

    # Stats MLP
    bx = 0.52
    _draw_box(ax, bx, y_enc, 0.22, h_enc,
              "Stats MLP", COLORS["stats"], fontsize=8, bold=True)
    ax.text(bx + 0.11, y_enc + 0.075, "Linear(9→64)",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.05, "GELU",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.025, "Linear(64→64)",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(bx + 0.11, y_enc + 0.005, "→ (B, 64)",
            ha="center", fontsize=6, color=COLORS["highlight"])

    # Arrows: inputs → encoders
    _draw_arrow(ax, 0.11, y_input + h_input, 0.11, y_enc, COLORS["geometry"])
    _draw_arrow(ax, 0.37, y_input + h_input, 0.37, y_enc, COLORS["trajectory"])
    _draw_arrow(ax, 0.63, y_input + h_input, 0.63, y_enc, COLORS["stats"])

    # ── Geometry dropout annotation ──
    ax.text(0.0, y_enc + h_enc + 0.015,
            "⊗ Geometry dropout (p=0.2)",
            fontsize=6.5, color=COLORS["attention"], fontstyle="italic")

    # ── BatchNorm layer ──
    y_bn = 0.345
    h_bn = 0.035
    for i, (bx, col, lbl) in enumerate([
        (0.03, COLORS["geometry"], "BN1d(64)"),
        (0.29, COLORS["trajectory"], "BN1d(64)"),
        (0.55, COLORS["stats"], "BN1d(64)"),
    ]):
        _draw_box(ax, bx, y_bn, 0.16, h_bn, lbl, col, fontsize=7, alpha=0.7)

    # Arrows: encoders → batchnorm
    _draw_arrow(ax, 0.11, y_enc + h_enc, 0.11, y_bn, COLORS["geometry"])
    _draw_arrow(ax, 0.37, y_enc + h_enc, 0.37, y_bn, COLORS["trajectory"])
    _draw_arrow(ax, 0.63, y_enc + h_enc, 0.63, y_bn, COLORS["stats"])

    # ── Concatenation ──
    y_cat = 0.42
    _draw_box(ax, 0.15, y_cat, 0.44, 0.035,
              "Concatenate → (B, 192)", COLORS["fusion"], fontsize=8,
              alpha=0.85)

    # Arrows: batchnorms → concat
    _draw_arrow(ax, 0.11, y_bn + h_bn, 0.25, y_cat, COLORS["geometry"],
                connectionstyle="arc3,rad=-0.15")
    _draw_arrow(ax, 0.37, y_bn + h_bn, 0.37, y_cat, COLORS["trajectory"])
    _draw_arrow(ax, 0.63, y_bn + h_bn, 0.49, y_cat, COLORS["stats"],
                connectionstyle="arc3,rad=0.15")

    # ── Fusion MLP ──
    y_fus = 0.49
    h_fus = 0.08
    _draw_box(ax, 0.15, y_fus, 0.44, h_fus,
              "Fusion MLP", COLORS["fusion"], fontsize=9, bold=True)
    ax.text(0.37, y_fus + 0.055, "Linear(192→256) → GELU → Dropout(0.1) → Linear(256→128)",
            ha="center", fontsize=6, color="white", alpha=0.9)
    ax.text(0.37, y_fus + 0.015, "→ embedding (B, 128)",
            ha="center", fontsize=6.5, color=COLORS["highlight"])

    _draw_arrow(ax, 0.37, y_cat + 0.035, 0.37, y_fus, COLORS["fusion"])

    # ── Cross-lane attention (optional, right side) ──
    y_cla = 0.46
    h_cla = 0.14
    bx_cla = 0.78
    _draw_box(ax, bx_cla, y_cla, 0.21, h_cla,
              "CrossLane Attn", COLORS["attention"], fontsize=8, bold=True,
              edgecolor="#C0392B")
    ax.text(bx_cla + 0.105, y_cla + h_cla - 0.02, "(optional)",
            ha="center", fontsize=6, color="white", fontstyle="italic")
    ax.text(bx_cla + 0.105, y_cla + 0.085,
            "Pack by group_id → (G, L, 128)",
            ha="center", fontsize=5.5, color="white")
    ax.text(bx_cla + 0.105, y_cla + 0.065,
            "Q,K,V projections (128→128)",
            ha="center", fontsize=5.5, color="white")
    ax.text(bx_cla + 0.105, y_cla + 0.045,
            "4-head self-attention",
            ha="center", fontsize=5.5, color="white")
    ax.text(bx_cla + 0.105, y_cla + 0.025,
            "+ Pairwise relative bias:",
            ha="center", fontsize=5.5, color="white")
    ax.text(bx_cla + 0.105, y_cla + 0.008,
            "Δlateral, Δspeed, ρ_density",
            ha="center", fontsize=5.5, color=COLORS["highlight"])

    # Arrow: fusion → cross-lane attention
    _draw_arrow(ax, 0.59, y_fus + h_fus / 2, bx_cla, y_cla + h_cla / 2,
                COLORS["attention"], connectionstyle="arc3,rad=0.1")

    # ── Output heads (top) ──
    y_head = 0.68
    h_head = 0.08

    # Projection head
    _draw_box(ax, 0.02, y_head, 0.2, h_head,
              "Projection Head", COLORS["output"], fontsize=7.5, bold=True)
    ax.text(0.12, y_head + 0.045, "Linear(128→128)→BN→ReLU",
            ha="center", fontsize=5.5, color="white")
    ax.text(0.12, y_head + 0.02, "→Linear(128→64)→BN→L2norm",
            ha="center", fontsize=5.5, color="white")
    ax.text(0.12, y_head + 0.001, "→ z (B, 64)",
            ha="center", fontsize=5.5, color=COLORS["highlight"])

    # Rank head
    _draw_box(ax, 0.25, y_head, 0.17, h_head,
              "Rank Head", COLORS["output"], fontsize=7.5, bold=True)
    ax.text(0.335, y_head + 0.04, "Linear(128→32)→ReLU",
            ha="center", fontsize=5.5, color="white")
    ax.text(0.335, y_head + 0.02, "→Linear(32→1)",
            ha="center", fontsize=5.5, color="white")
    ax.text(0.335, y_head + 0.001, "→ lateral rank",
            ha="center", fontsize=5.5, color=COLORS["highlight"])

    # Edge head
    _draw_box(ax, 0.45, y_head, 0.14, h_head,
              "Edge Head", COLORS["output"], fontsize=7.5, bold=True)
    ax.text(0.52, y_head + 0.035, "Linear(128→2)",
            ha="center", fontsize=5.5, color="white")
    ax.text(0.52, y_head + 0.001, "→ is_left, is_right",
            ha="center", fontsize=5.5, color=COLORS["highlight"])

    # Size head
    _draw_box(ax, 0.62, y_head, 0.14, h_head,
              "Size Head", COLORS["output"], fontsize=7.5, bold=True)
    ax.text(0.69, y_head + 0.035, "Linear(128→1)",
            ha="center", fontsize=5.5, color="white")
    ax.text(0.69, y_head + 0.001, "→ group size",
            ha="center", fontsize=5.5, color=COLORS["highlight"])

    # Arrows: fusion → heads
    y_fus_top = y_fus + h_fus
    _draw_arrow(ax, 0.25, y_fus_top, 0.12, y_head, COLORS["output"],
                connectionstyle="arc3,rad=-0.1")
    _draw_arrow(ax, 0.30, y_fus_top, 0.335, y_head, COLORS["output"])
    _draw_arrow(ax, 0.40, y_fus_top, 0.52, y_head, COLORS["output"],
                connectionstyle="arc3,rad=0.08")
    _draw_arrow(ax, 0.45, y_fus_top, 0.69, y_head, COLORS["output"],
                connectionstyle="arc3,rad=0.12")

    # ── Loss annotations (top) ──
    y_loss = 0.82
    ax.text(0.12, y_loss, "InfoNCE\nContrastive Loss",
            ha="center", va="bottom", fontsize=7, color=COLORS["fusion"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["fusion"], alpha=0.9))
    ax.text(0.42, y_loss, "MSE / BCE\nRegression Loss",
            ha="center", va="bottom", fontsize=7, color=COLORS["output"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["output"], alpha=0.9))

    _draw_arrow(ax, 0.12, y_head + h_head, 0.12, y_loss - 0.01, COLORS["fusion"],
                style="-|>", lw=1.0)
    _draw_arrow(ax, 0.42, y_head + h_head, 0.42, y_loss - 0.01, COLORS["output"],
                style="-|>", lw=1.0)

    # ── Dimension summary box ──
    summary = (
        "Dimensions:  K=16 points  |  d_model=64  |  embed_dim=128  |  proj_dim=64\n"
        "Transformer: 2 layers × 4 heads × FFN=256  |  Stats: 4 traj + 5 role = 9 dim"
    )
    ax.text(0.5, -0.03, summary, ha="center", va="top", fontsize=7,
            color=COLORS["dim"], fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#DDD", alpha=0.8))

    plt.tight_layout()
    path = output_dir / "A1_lane_encoder_architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  A1 saved to {path}")


# ══════════════════════════════════════════════════════════════════════════
# A2 — Temporal Encoder Architecture
# ══════════════════════════════════════════════════════════════════════════

def figure_a2(output_dir: Path):
    """Detailed Temporal Encoder architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(0.5, 1.0, "Temporal Encoder Architecture", ha="center", va="top",
            fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(0.5, 0.965, "Frozen lane encoder per window → GRU → anomaly detection",
            ha="center", va="top", fontsize=9, color=COLORS["dim"])

    # ── Windows at bottom ──
    n_windows = 5
    w_win = 0.13
    h_win = 0.10
    y_win = 0.05
    gap = 0.04
    total_w = n_windows * w_win + (n_windows - 1) * gap
    x_start = 0.5 - total_w / 2

    window_labels = ["w₀", "w₁", "w₂", "w₃", "..."]
    window_sublabels = [
        "traj + stats",
        "traj + stats",
        "traj + stats",
        "traj + stats",
        "",
    ]

    for i in range(n_windows):
        x = x_start + i * (w_win + gap)
        col = COLORS["trajectory"] if i < 4 else COLORS["dim"]
        _draw_box(ax, x, y_win, w_win, h_win,
                  f"Window {window_labels[i]}", col, fontsize=7.5,
                  sublabel=window_sublabels[i] if i < 4 else None)

    # ── Static geometry ──
    _draw_box(ax, 0.0, y_win + h_win + 0.06, 0.12, 0.05,
              "Geometry", COLORS["geometry"], fontsize=7.5, bold=True,
              sublabel="(B, K, 2)")
    ax.text(0.12, y_win + h_win + 0.09, "  (shared across windows)",
            fontsize=6, color=COLORS["dim"], fontstyle="italic")

    # ── Frozen encoder boxes ──
    y_enc = 0.25
    h_enc = 0.10
    for i in range(n_windows):
        x = x_start + i * (w_win + gap)
        if i < 4:
            _draw_box(ax, x, y_enc, w_win, h_enc,
                      "Frozen", COLORS["geometry"], fontsize=7.5,
                      sublabel="LaneEncoder", alpha=0.7)
            # Snowflake for frozen
            ax.text(x + w_win - 0.01, y_enc + h_enc - 0.01, "❄",
                    fontsize=8, ha="right", va="top", zorder=5)
            # Arrow: window → encoder
            _draw_arrow(ax, x + w_win / 2, y_win + h_win,
                        x + w_win / 2, y_enc, COLORS["trajectory"])
        else:
            ax.text(x + w_win / 2, y_enc + h_enc / 2, "⋯",
                    ha="center", fontsize=18, color=COLORS["dim"])

    # Arrow: geometry → encoders
    for i in range(4):
        x = x_start + i * (w_win + gap) + w_win / 2
        _draw_arrow(ax, 0.06, y_win + h_win + 0.06,
                    x, y_enc + h_enc, COLORS["geometry"],
                    connectionstyle=f"arc3,rad={-0.15 + i * 0.08}", lw=0.8)

    # ── Embedding sequence ──
    y_emb = 0.42
    h_emb = 0.04
    for i in range(n_windows):
        x = x_start + i * (w_win + gap)
        if i < 4:
            _draw_box(ax, x + 0.01, y_emb, w_win - 0.02, h_emb,
                      f"e_{window_labels[i]}", COLORS["fusion"], fontsize=7,
                      alpha=0.8, textcolor="white")
            _draw_arrow(ax, x + w_win / 2, y_enc + h_enc,
                        x + w_win / 2, y_emb, COLORS["fusion"])
        else:
            ax.text(x + w_win / 2, y_emb + h_emb / 2, "⋯",
                    ha="center", fontsize=14, color=COLORS["dim"])

    _dim_label(ax, 0.92, y_emb + h_emb / 2, "(B, W, 128)")

    # Forward fill annotation
    ax.text(x_start - 0.02, y_emb + h_emb + 0.015,
            "Forward-fill invalid windows",
            fontsize=6, color=COLORS["dim"], fontstyle="italic")

    # ── GRU ──
    y_gru = 0.53
    h_gru = 0.10
    gru_w = total_w + 0.02
    gru_x = x_start - 0.01
    _draw_box(ax, gru_x, y_gru, gru_w, h_gru,
              "GRU", COLORS["temporal"], fontsize=11, bold=True)
    ax.text(gru_x + gru_w / 2, y_gru + 0.025,
            "input=128, hidden=128, 1 layer, batch_first=True",
            ha="center", fontsize=6.5, color="white")

    # Arrow: embeddings → GRU
    _draw_arrow(ax, 0.5, y_emb + h_emb, 0.5, y_gru, COLORS["temporal"])

    # ── GRU hidden states ──
    y_hid = 0.70
    h_hid = 0.04
    for i in range(n_windows):
        x = x_start + i * (w_win + gap)
        if i < 4:
            _draw_box(ax, x + 0.01, y_hid, w_win - 0.02, h_hid,
                      f"h_{window_labels[i]}", COLORS["temporal"], fontsize=7,
                      alpha=0.8, textcolor="white")

    _draw_arrow(ax, 0.5, y_gru + h_gru, 0.5, y_hid, COLORS["temporal"])
    _dim_label(ax, 0.92, y_hid + h_hid / 2, "(B, W, 128)")

    # ── Anomaly head ──
    y_anom = 0.80
    h_anom = 0.08
    anom_w = 0.30
    anom_x = 0.5 - anom_w / 2
    _draw_box(ax, anom_x, y_anom, anom_w, h_anom,
              "Anomaly Head", COLORS["attention"], fontsize=9, bold=True)
    ax.text(0.5, y_anom + 0.025,
            "Linear(128→64)→ReLU→Dropout→Linear(64→1)",
            ha="center", fontsize=6, color="white")

    _draw_arrow(ax, 0.5, y_hid + h_hid, 0.5, y_anom, COLORS["attention"])

    # ── Output ──
    y_out = 0.92
    ax.text(0.5, y_out, "anomaly_scores (B, W)",
            ha="center", fontsize=9, fontweight="bold", color=COLORS["attention"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["attention"], alpha=0.9))
    _draw_arrow(ax, 0.5, y_anom + h_anom, 0.5, y_out - 0.015, COLORS["attention"])

    # ── Training note ──
    ax.text(0.02, -0.03,
            "Training: BCE loss on anomaly labels  |  Frozen encoder weights (❄)  |  "
            "Window sizes: 5/10/15/30 sec",
            fontsize=7, color=COLORS["dim"], fontstyle="italic")

    plt.tight_layout()
    path = output_dir / "A2_temporal_encoder_architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  A2 saved to {path}")


# ══════════════════════════════════════════════════════════════════════════
# A3 — FiLM Diffusion Generator Architecture
# ══════════════════════════════════════════════════════════════════════════

def figure_a3(output_dir: Path):
    """Detailed FiLM-conditioned diffusion architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(0.5, 1.0, "FiLM-Conditioned Diffusion Generator", ha="center", va="top",
            fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(0.5, 0.965,
            "DDPM with behavioral embedding conditioning via Feature-wise Linear Modulation",
            ha="center", va="top", fontsize=9, color=COLORS["dim"])

    # ── Left panel: Denoiser architecture ──
    panel_left = 0.02
    panel_w = 0.55

    # Inputs at bottom
    y_inp = 0.06
    # x_t input
    _draw_box(ax, panel_left, y_inp, 0.15, 0.05,
              "x_t", COLORS["diffusion"], fontsize=8, bold=True,
              sublabel="(B, 32)")
    _dim_label(ax, panel_left + 0.075, y_inp - 0.015, "noisy geometry\nK×2 flattened",
               fontsize=5.5)

    # t input
    _draw_box(ax, panel_left + 0.18, y_inp, 0.12, 0.05,
              "t", COLORS["temporal"], fontsize=8, bold=True,
              sublabel="(B,)")

    # Sinusoidal embedding
    _draw_box(ax, panel_left + 0.18, y_inp + 0.07, 0.12, 0.04,
              "Sin/Cos Emb", COLORS["temporal"], fontsize=6.5,
              sublabel="→ (B, 64)", alpha=0.8)
    _draw_arrow(ax, panel_left + 0.24, y_inp + 0.05,
                panel_left + 0.24, y_inp + 0.07, COLORS["temporal"])

    # Conditioning embedding (right side)
    _draw_box(ax, panel_left + 0.40, y_inp, 0.15, 0.05,
              "cond", COLORS["fusion"], fontsize=8, bold=True,
              sublabel="(B, 128)")
    _dim_label(ax, panel_left + 0.475, y_inp - 0.015,
               "behavioral\nembedding", fontsize=5.5)

    # Concat x_t + t_emb
    y_cat = 0.17
    _draw_box(ax, panel_left + 0.02, y_cat, 0.28, 0.035,
              "Concat [x_t, t_emb] → (B, 96)", COLORS["output"], fontsize=7,
              alpha=0.8)
    _draw_arrow(ax, panel_left + 0.075, y_inp + 0.05,
                panel_left + 0.12, y_cat, COLORS["diffusion"])
    _draw_arrow(ax, panel_left + 0.24, y_inp + 0.11,
                panel_left + 0.20, y_cat, COLORS["temporal"])

    # ── FiLM layers ──
    film_layers = [
        ("FiLM Layer 1", "96 → 256", 0.24),
        ("FiLM Layer 2", "256 → 256", 0.41),
        ("FiLM Layer 3", "256 → 128", 0.58),
    ]

    y_film_start = y_cat + 0.06
    film_h = 0.11
    film_w = 0.35
    film_x = panel_left + 0.02

    for i, (name, dims, y_f) in enumerate(film_layers):
        _draw_box(ax, film_x, y_f, film_w, film_h,
                  name, COLORS["film"], fontsize=9, bold=True)

        # Internal structure
        in_d, out_d = dims.split(" → ")
        ax.text(film_x + film_w / 2, y_f + film_h - 0.02,
                f"Linear({in_d}→{out_d})",
                ha="center", fontsize=6, color="white")
        ax.text(film_x + film_w / 2, y_f + film_h - 0.04,
                f"LayerNorm({out_d})",
                ha="center", fontsize=6, color="white")
        ax.text(film_x + film_w / 2, y_f + film_h - 0.06,
                "h × (1 + γ) + β",
                ha="center", fontsize=7, color=COLORS["highlight"], fontweight="bold")
        ax.text(film_x + film_w / 2, y_f + film_h - 0.08,
                "GELU activation",
                ha="center", fontsize=6, color="white")

        # Conditioning arrow (from right)
        cond_x = panel_left + 0.475
        _draw_arrow(ax, cond_x, y_f + film_h / 2,
                    film_x + film_w, y_f + film_h / 2,
                    COLORS["fusion"], connectionstyle="arc3,rad=0.0", lw=1.0)

        # γ and β annotation
        ax.text(film_x + film_w + 0.02, y_f + film_h / 2 + 0.015,
                f"γ = Linear(128→{out_d})", fontsize=5.5, color=COLORS["fusion"])
        ax.text(film_x + film_w + 0.02, y_f + film_h / 2 - 0.01,
                f"β = Linear(128→{out_d})", fontsize=5.5, color=COLORS["fusion"])

        # Arrow between layers
        if i > 0:
            prev_y = film_layers[i - 1][2] + film_h
            _draw_arrow(ax, film_x + film_w / 2, prev_y,
                        film_x + film_w / 2, y_f, COLORS["film"])

    # Arrow: concat → first FiLM
    _draw_arrow(ax, film_x + film_w / 2, y_cat + 0.035,
                film_x + film_w / 2, film_layers[0][2], COLORS["film"])

    # Cond arrow vertical line
    cond_x = panel_left + 0.475
    _draw_arrow(ax, cond_x, y_inp + 0.05, cond_x, 0.65, COLORS["fusion"],
                style="-", lw=1.5)

    # ── Output layer ──
    y_out = 0.73
    h_out = 0.05
    _draw_box(ax, film_x, y_out, film_w, h_out,
              "Output Linear(128→32)", COLORS["output"], fontsize=8, bold=True)
    _draw_arrow(ax, film_x + film_w / 2, film_layers[-1][2] + film_h,
                film_x + film_w / 2, y_out, COLORS["output"])

    # Predicted noise
    y_pred = 0.82
    ax.text(film_x + film_w / 2, y_pred, "ε̂  (predicted noise)",
            ha="center", fontsize=10, fontweight="bold", color=COLORS["diffusion"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLORS["diffusion"], alpha=0.9))
    _draw_arrow(ax, film_x + film_w / 2, y_out + h_out,
                film_x + film_w / 2, y_pred - 0.015, COLORS["diffusion"])

    # ── Right panel: DDPM schedule ──
    rx = 0.62
    rw = 0.38

    # Schedule box
    _draw_box(ax, rx, 0.72, rw - 0.02, 0.15,
              "DDPM Schedule", COLORS["diffusion"], fontsize=9, bold=True,
              edgecolor=COLORS["diffusion"])
    ax.text(rx + (rw - 0.02) / 2, 0.83,
            "T = 100 steps", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.805,
            "β: 1e-4 → 0.02 (linear)", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.78,
            "α̅ = ∏αₜ  (cumulative)", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.755,
            "q(xₜ|x₀) = √α̅ₜ x₀ + √(1-α̅ₜ) ε",
            ha="center", fontsize=6.5, color=COLORS["highlight"])
    ax.text(rx + (rw - 0.02) / 2, 0.735,
            "Reverse: xₜ₋₁ = (xₜ - β/√(1-α̅) ε̂)/√α + σz",
            ha="center", fontsize=6, color="white")

    # Warm start box
    _draw_box(ax, rx, 0.55, rw - 0.02, 0.13,
              "Warm Start", COLORS["temporal"], fontsize=9, bold=True)
    ax.text(rx + (rw - 0.02) / 2, 0.655,
            "Start from anchor lane", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.63,
            "at t = warm_start_t (default 50)",
            ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.605,
            "Preserves coarse shape,", ha="center", fontsize=6.5, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.58,
            "diffusion adjusts curvature",
            ha="center", fontsize=6.5, color=COLORS["highlight"])

    # Post-processing box
    _draw_box(ax, rx, 0.38, rw - 0.02, 0.13,
              "Post-Processing", COLORS["stats"], fontsize=9, bold=True)
    ax.text(rx + (rw - 0.02) / 2, 0.49,
            "1. Clip to [0, 1]", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.465,
            "2. Multi-pass smoothing (2×)", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.44,
            "   3-point moving average", ha="center", fontsize=6.5, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.415,
            "3. Smoothness filter (θ=1.0)", ha="center", fontsize=7, color="white")
    ax.text(rx + (rw - 0.02) / 2, 0.395,
            "→ candidates (n, K, 2)", ha="center", fontsize=6.5,
            color=COLORS["highlight"])

    # Pipeline flow box
    _draw_box(ax, rx, 0.08, rw - 0.02, 0.26,
              "Generation Pipeline", COLORS["output"], fontsize=9, bold=True)
    steps = [
        "1. Resolve spec → target embedding",
        "2. Get anchor lane → canonical space",
        "3. Build warm-start from anchor",
        "4. Run DDPM reverse (conditioned)",
        "5. Denormalize: canonical → image",
        "6. Post-process + smooth",
        "7. Score candidates (cosine sim)",
        "8. Return best + all candidates",
    ]
    for j, step in enumerate(steps):
        ax.text(rx + 0.02, 0.30 - j * 0.025, step,
                fontsize=6.5, color="white")

    # Summary
    ax.text(0.5, -0.03,
            "Geometry dim: K×2 = 32  |  Cond dim: 128  |  Hidden: 256  |  "
            "T=100  |  LR=1e-4  |  Optimizer: AdamW",
            ha="center", fontsize=7, color=COLORS["dim"], fontstyle="italic")

    plt.tight_layout()
    path = output_dir / "A3_diffusion_architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  A3 saved to {path}")


# ══════════════════════════════════════════════════════════════════════════
# A4 — Full System Pipeline
# ══════════════════════════════════════════════════════════════════════════

def figure_a4(output_dir: Path):
    """Full system pipeline with all modules."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(0.5, 1.0, "GeoLane System Architecture", ha="center", va="top",
            fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(0.5, 0.96, "End-to-end pipeline: observation → encoding → generation → digital twin",
            ha="center", va="top", fontsize=9, color=COLORS["dim"])

    # Module positions (left to right flow)
    modules = [
        # (x, y, w, h, name, color, description, key_detail)
        (0.0, 0.40, 0.13, 0.22,
         "E\nLane\nEncoder", COLORS["geometry"],
         "Geometry + Traj + Stats\n→ 128-dim embedding",
         "PolylineEnc ×2\nFusion MLP\nContrastive loss"),

        (0.16, 0.40, 0.13, 0.22,
         "T\nTemporal\nEncoder", COLORS["temporal"],
         "Window sequence\n→ anomaly scores",
         "Frozen E + GRU\nAnomaly head\nBCE loss"),

        (0.32, 0.40, 0.13, 0.22,
         "M\nBehavioral\nTaxonomy", COLORS["stats"],
         "Embedding clustering\n→ scene types",
         "K-means on E\nSpeed/density/rank\nCross-camera"),

        (0.48, 0.40, 0.13, 0.22,
         "G\nLane\nGenerator", COLORS["diffusion"],
         "Spec → FiLM diffusion\n→ lane geometry",
         "DDPM T=100\n3× FiLM layers\nWarm-start"),

        (0.64, 0.40, 0.13, 0.22,
         "C\nTraffic\nBridge", COLORS["bridge"],
         "Embedding → HCM\nspeed/density/LOS",
         "Affine calibration\nSUMO validation\nLOS A–F"),

        (0.80, 0.40, 0.13, 0.22,
         "D\nDigital\nTwin", COLORS["output"],
         "JSON export\n→ Rust twin",
         "lane_state.json\ntopology.json\nmodifications.json"),
    ]

    for x, y, w, h, name, color, desc, detail in modules:
        # Main box
        _draw_box(ax, x, y, w, h, "", color, fontsize=1, alpha=0.92)

        # Module name (top part)
        ax.text(x + w / 2, y + h - 0.03, name,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color="white", linespacing=1.1)

        # Description (middle)
        ax.text(x + w / 2, y + 0.08, desc,
                ha="center", va="center", fontsize=5.5,
                color="white", alpha=0.9, linespacing=1.3)

        # Detail (bottom)
        ax.text(x + w / 2, y + 0.02, detail,
                ha="center", va="bottom", fontsize=5,
                color=COLORS["highlight"], alpha=0.85, linespacing=1.2)

    # ── Flow arrows between modules ──
    for i in range(len(modules) - 1):
        x1 = modules[i][0] + modules[i][2]
        x2 = modules[i + 1][0]
        y_mid = modules[i][1] + modules[i][3] / 2
        _draw_arrow(ax, x1, y_mid, x2, y_mid, "#555", lw=2.0)

    # ── Data flow labels ──
    flow_labels = [
        (0.145, 0.52, "embeddings\n(B, 128)"),
        (0.305, 0.52, "h_seq\nanom. scores"),
        (0.465, 0.52, "cluster IDs\nspec labels"),
        (0.625, 0.52, "generated\ngeometry"),
        (0.785, 0.52, "speed/density\nLOS grade"),
    ]
    for x, y, lbl in flow_labels:
        ax.text(x, y, lbl, ha="center", va="bottom", fontsize=5.5,
                color=COLORS["dim"], fontstyle="italic", linespacing=1.2)

    # ── Input data (bottom) ──
    y_data = 0.15
    data_items = [
        (0.0, "Camera\nFrames", COLORS["geometry"]),
        (0.12, "Lane\nAnnotations", COLORS["geometry"]),
        (0.24, "Vehicle\nTrajectories", COLORS["trajectory"]),
        (0.36, "Role\nDescriptors", COLORS["stats"]),
    ]
    for x, lbl, col in data_items:
        _draw_box(ax, x, y_data, 0.10, 0.08, lbl, col, fontsize=6.5, alpha=0.7)

    # Arrows from data to encoder
    for x, _, _ in data_items:
        _draw_arrow(ax, x + 0.05, y_data + 0.08,
                    modules[0][0] + modules[0][2] / 2, modules[0][1],
                    COLORS["geometry"], lw=0.8,
                    connectionstyle="arc3,rad=0.1")

    # ── Ephemeral event path (top) ──
    y_event = 0.78
    # Event annotation
    ax.text(0.35, y_event + 0.12,
            "Ephemeral Event (e.g., incident, construction, weather)",
            ha="center", fontsize=8, fontweight="bold", color=COLORS["attention"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDE8E8",
                      edgecolor=COLORS["attention"], alpha=0.9))

    # Arrows from event down to T and G
    _draw_arrow(ax, 0.22, y_event + 0.10,
                modules[1][0] + modules[1][2] / 2, modules[1][1] + modules[1][3],
                COLORS["attention"], lw=1.5, connectionstyle="arc3,rad=-0.1")
    ax.text(0.18, y_event + 0.02, "behavioral\nshift", fontsize=6,
            color=COLORS["attention"], ha="center")

    _draw_arrow(ax, 0.48, y_event + 0.10,
                modules[3][0] + modules[3][2] / 2, modules[3][1] + modules[3][3],
                COLORS["attention"], lw=1.5, connectionstyle="arc3,rad=0.1")
    ax.text(0.52, y_event + 0.02, "trigger\ngeneration", fontsize=6,
            color=COLORS["attention"], ha="center")

    # ── Feedback loop ──
    # G feeds back to E (re-encoding for scoring)
    ax.annotate("", xy=(modules[0][0] + modules[0][2] / 2, modules[0][1]),
                xytext=(modules[3][0] + modules[3][2] / 2, modules[3][1]),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["dim"],
                                lw=1.0, linestyle="dashed",
                                connectionstyle="arc3,rad=0.5"))
    ax.text(0.24, 0.30, "re-encode candidates\n(optional scoring)",
            fontsize=5.5, color=COLORS["dim"], ha="center", fontstyle="italic")

    # ── Digital Twin output (right) ──
    _draw_box(ax, 0.88, 0.20, 0.12, 0.12,
              "geolane_twin", COLORS["twin"], fontsize=7, bold=True,
              sublabel="(Rust)")
    ax.text(0.94, 0.16, "EncoderBridge polls JSON\n→ LaneModification stack\n→ Real-time rendering",
            ha="center", fontsize=5.5, color=COLORS["dim"], linespacing=1.3)
    _draw_arrow(ax, modules[-1][0] + modules[-1][2] / 2, modules[-1][1],
                0.94, 0.32, COLORS["twin"], lw=1.5)

    plt.tight_layout()
    path = output_dir / "A4_system_pipeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  A4 saved to {path}")


# ══════════════════════════════════════════════════════════════════════════
# A5 — Embedding Connectivity: How Encoder Outputs Feed Downstream Modules
# ══════════════════════════════════════════════════════════════════════════

def figure_a5(output_dir: Path):
    """Detailed embedding connectivity diagram.

    Shows exactly which encoder outputs flow into each downstream module,
    distinguishing the two training regimes:
      - Joint (end-to-end): Encoder + GRU + anomaly head + contrastive loss
      - Two-stage (frozen): Frozen encoder + GRU + anomaly head only

    And the non-end-to-end gap: Encoder → (pre-computed embeddings) → Diffusion Generator.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])

    ax.text(0.5, 1.02, "Embedding Connectivity: Encoder → Downstream Modules",
            ha="center", va="top", fontsize=16, fontweight="bold",
            color=COLORS["text"])
    ax.text(0.5, 0.985,
            "Joint encoder trains E+T end-to-end; Generator uses pre-computed embeddings (not end-to-end)",
            ha="center", va="top", fontsize=9, color=COLORS["dim"])

    # ════════════════════════════════════════════════════════════════════
    # Central encoder block (left-center)
    # ════════════════════════════════════════════════════════════════════
    enc_x, enc_y, enc_w, enc_h = 0.0, 0.35, 0.22, 0.32
    _draw_box(ax, enc_x, enc_y, enc_w, enc_h,
              "", COLORS["geometry"], fontsize=1, alpha=0.95)
    ax.text(enc_x + enc_w / 2, enc_y + enc_h - 0.02,
            "Lane Encoder (E)", ha="center", fontsize=11,
            fontweight="bold", color="white")
    ax.text(enc_x + enc_w / 2, enc_y + enc_h - 0.045,
            "TRAINABLE in joint mode", ha="center", fontsize=6.5,
            color=COLORS["highlight"], fontweight="bold")

    # Encoder outputs (stacked within the box)
    output_items = [
        ("embedding", "(B, 128)", 0.235, COLORS["fusion"]),
        ("projection", "(B, 64)", 0.195, COLORS["output"]),
        ("pred_rank", "(B,)", 0.160, COLORS["temporal"]),
        ("pred_edge", "(B, 2)", 0.128, COLORS["stats"]),
        ("pred_size", "(B,)", 0.098, COLORS["attention"]),
        ("traj_stats*", "(B, 4)", 0.065, COLORS["dim"]),
    ]
    for label, shape, dy, col in output_items:
        y_item = enc_y + dy
        _draw_box(ax, enc_x + 0.015, y_item, 0.19, 0.025,
                  f"{label}  {shape}", col, fontsize=6.5, alpha=0.85)

    ax.text(enc_x + enc_w / 2, enc_y + 0.035,
            "*pass-through input", fontsize=5, color="white",
            ha="center", fontstyle="italic", alpha=0.7)

    # Encoder inputs (small boxes below)
    y_inp = 0.20
    inp_items = [
        ("Geometry\n(B,16,2)", COLORS["geometry"], 0.0),
        ("Trajectories\n(B,T,16,2)", COLORS["trajectory"], 0.08),
        ("Stats+Roles\n(B,9)", COLORS["stats"], 0.16),
    ]
    for label, col, dx in inp_items:
        _draw_box(ax, enc_x + dx, y_inp, 0.07, 0.06, label, col,
                  fontsize=5.5, alpha=0.7)
        _draw_arrow(ax, enc_x + dx + 0.035, y_inp + 0.06,
                    enc_x + enc_w / 2, enc_y, col, lw=0.8,
                    connectionstyle="arc3,rad=0.05")

    # ════════════════════════════════════════════════════════════════════
    # Downstream modules (right side, arranged vertically)
    # ════════════════════════════════════════════════════════════════════

    mod_x = 0.56
    mod_w = 0.42
    mod_h = 0.13

    # ── Module T: Temporal Encoder (JOINT — end-to-end) ──
    y_temp = 0.82
    # Highlight box for joint training
    _draw_box(ax, mod_x - 0.015, y_temp - 0.015, mod_w + 0.03, mod_h + 0.05,
              "", COLORS["highlight"], fontsize=1, alpha=0.25,
              edgecolor=COLORS["highlight"])
    ax.text(mod_x + mod_w / 2, y_temp + mod_h + 0.025,
            "END-TO-END (Joint Training)", ha="center",
            fontsize=8, fontweight="bold", color="#B8860B",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF8DC",
                      edgecolor=COLORS["highlight"], alpha=0.95))
    _draw_box(ax, mod_x, y_temp, mod_w, mod_h,
              "", COLORS["temporal"], fontsize=1, alpha=0.92)
    ax.text(mod_x + mod_w / 2, y_temp + mod_h - 0.015,
            "T — Temporal Encoder (GRU + Anomaly Head)", ha="center",
            fontsize=9, fontweight="bold", color="white")

    steps_t = [
        ("Joint mode:", "E._encode_per_lane() TRAINABLE — gradients flow back!", 0.075),
        ("Two-stage:", "E frozen (❄) — only GRU + anomaly head learn", 0.045),
        ("Both modes:", "[e_0,...,e_W] → GRU(128→128) → MLP(128→64→1)", 0.015),
    ]
    for title, detail, dy in steps_t:
        ax.text(mod_x + 0.015, y_temp + dy,
                title, fontsize=6.5, fontweight="bold", color="white")
        ax.text(mod_x + 0.015 + len(title) * 0.0045, y_temp + dy,
                f"  {detail}", fontsize=5.5, color=COLORS["highlight"])

    # ── Module G: Generator (NOT end-to-end) ──
    y_gen = 0.63
    # Gray box to indicate the break
    _draw_box(ax, mod_x - 0.015, y_gen - 0.015, mod_w + 0.03, mod_h + 0.05,
              "", "#E0E0E0", fontsize=1, alpha=0.3,
              edgecolor="#999")
    ax.text(mod_x + mod_w / 2, y_gen + mod_h + 0.025,
            "NOT END-TO-END (pre-computed embeddings)", ha="center",
            fontsize=8, fontweight="bold", color="#666",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5",
                      edgecolor="#999", alpha=0.95))
    _draw_box(ax, mod_x, y_gen, mod_w, mod_h,
              "", COLORS["diffusion"], fontsize=1, alpha=0.92)
    ax.text(mod_x + mod_w / 2, y_gen + mod_h - 0.015,
            "G — Lane Generator (FiLM Diffusion)", ha="center",
            fontsize=9, fontweight="bold", color="white")

    steps_g = [
        ("1. Retrieval Index:", "store pre-computed (embedding, geometry) pairs", 0.075),
        ("2. Spec → target_emb:", "avg matching embeddings (frozen, no gradient)", 0.045),
        ("3. FiLM Conditioning:", "target_emb(128) → γ,β per layer (3× FiLM)", 0.015),
    ]
    for title, detail, dy in steps_g:
        ax.text(mod_x + 0.015, y_gen + dy,
                title, fontsize=6.5, fontweight="bold", color="white")
        ax.text(mod_x + 0.015 + len(title) * 0.0045, y_gen + dy,
                f"  {detail}", fontsize=5.5, color=COLORS["highlight"])

    # ── Module M: Behavioral Taxonomy ──
    y_tax = 0.46
    _draw_box(ax, mod_x, y_tax, mod_w, mod_h,
              "", COLORS["stats"], fontsize=1, alpha=0.92)
    ax.text(mod_x + mod_w / 2, y_tax + mod_h - 0.015,
            "M — Behavioral Taxonomy", ha="center",
            fontsize=9, fontweight="bold", color="white")

    steps_m = [
        ("1. Cluster:", "K-means on embedding (128-dim) → cluster_id", 0.075),
        ("2. Statistics:", "per-cluster mean speed/density/rank from traj_stats", 0.045),
        ("3. Cross-camera:", "cosine sim on embeddings → retrieval", 0.015),
    ]
    for title, detail, dy in steps_m:
        ax.text(mod_x + 0.015, y_tax + dy,
                title, fontsize=6.5, fontweight="bold", color="white")
        ax.text(mod_x + 0.015 + len(title) * 0.0045, y_tax + dy,
                f"  {detail}", fontsize=5.5, color=COLORS["highlight"])

    # ── Module C: Traffic Bridge ──
    y_bridge = 0.29
    _draw_box(ax, mod_x, y_bridge, mod_w, mod_h,
              "", COLORS["bridge"], fontsize=1, alpha=0.92)
    ax.text(mod_x + mod_w / 2, y_bridge + mod_h - 0.015,
            "C — Traffic Bridge (HCM)", ha="center",
            fontsize=9, fontweight="bold", color="white")

    steps_c = [
        ("1. Features:", "traj_stats(4) + pred_rank(1) + pred_edge(2) + emb_stats(3)", 0.075),
        ("2. Calibrate:", "affine transform or TrafficTranslatorMLP(10→64→64)", 0.045),
        ("3. Output:", "speed_mph, density_veh/mi/ln, flow, LOS A–F", 0.015),
    ]
    for title, detail, dy in steps_c:
        ax.text(mod_x + 0.015, y_bridge + dy,
                title, fontsize=6.5, fontweight="bold", color="white")
        ax.text(mod_x + 0.015 + len(title) * 0.0045, y_bridge + dy,
                f"  {detail}", fontsize=5.5, color=COLORS["highlight"])

    # ── Module D: Digital Twin ──
    y_dt = 0.12
    _draw_box(ax, mod_x, y_dt, mod_w, mod_h,
              "", COLORS["output"], fontsize=1, alpha=0.92)
    ax.text(mod_x + mod_w / 2, y_dt + mod_h - 0.015,
            "D — Digital Twin Export", ha="center",
            fontsize=9, fontweight="bold", color="white")

    steps_d = [
        ("1. lane_state.json:", "speed, density, LOS, anomaly_score, emb stats", 0.075),
        ("2. topology.json:", "generated geometries in SUMO coords", 0.045),
        ("3. modifications.json:", "CloseLane/SetSpeed from anomaly + LOS", 0.015),
    ]
    for title, detail, dy in steps_d:
        ax.text(mod_x + 0.015, y_dt + dy,
                title, fontsize=6.5, fontweight="bold", color="white")
        ax.text(mod_x + 0.015 + len(title) * 0.0045, y_dt + dy,
                f"  {detail}", fontsize=5.5, color=COLORS["highlight"])

    # ════════════════════════════════════════════════════════════════════
    # Connection arrows
    # ════════════════════════════════════════════════════════════════════

    label_x = 0.33

    emb_y = enc_y + 0.235 + 0.0125  # center of embedding output

    # -- THICK SOLID: Encoder ↔ Temporal (end-to-end, gradient flows both ways) --
    _draw_arrow(ax, enc_x + enc_w, emb_y + 0.02,
                mod_x, y_temp + mod_h / 2,
                COLORS["fusion"], lw=3.0,
                connectionstyle="arc3,rad=-0.12")
    # Backward gradient arrow (dashed, same path reversed)
    ax.annotate(
        "", xy=(enc_x + enc_w, emb_y + 0.04),
        xytext=(mod_x, y_temp + mod_h / 2 + 0.02),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["attention"],
                        lw=1.8, linestyle="dashed",
                        connectionstyle="arc3,rad=0.12"))
    ax.text(label_x - 0.02, 0.80,
            "shared weights (trainable)",
            fontsize=7, fontweight="bold", color=COLORS["fusion"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=COLORS["fusion"], alpha=0.9))
    ax.text(label_x - 0.02, 0.775,
            "← ∇(BCE + InfoNCE) backprop",
            fontsize=6, color=COLORS["attention"], fontweight="bold")

    # Loss annotations for joint training
    ax.text(0.03, 0.77,
            "Joint Loss:\nα·BCE(anomaly)\n+ β·InfoNCE(contrastive)\n+ role regression",
            fontsize=6, color=COLORS["attention"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDE8E8",
                      edgecolor=COLORS["attention"], alpha=0.85),
            linespacing=1.4)

    # -- DASHED: Encoder → Generator (pre-computed, NO gradient) --
    _draw_arrow(ax, enc_x + enc_w, emb_y,
                mod_x, y_gen + mod_h / 2,
                "#888888", lw=2.0,
                connectionstyle="arc3,rad=-0.05")
    # X mark to show the break
    break_x, break_y = 0.40, 0.69
    ax.text(break_x, break_y, "✕", fontsize=14, fontweight="bold",
            color=COLORS["attention"], ha="center", va="center", zorder=6)
    ax.text(label_x, 0.66,
            "pre-computed embeddings",
            fontsize=7, fontweight="bold", color="#666",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#999", alpha=0.9))
    ax.text(label_x, 0.64,
            "no gradient — diffusion loss ≠ encoder update",
            fontsize=5.5, color=COLORS["attention"], fontstyle="italic")

    # -- embedding → Taxonomy --
    _draw_arrow(ax, enc_x + enc_w, emb_y - 0.02,
                mod_x, y_tax + mod_h / 2,
                COLORS["fusion"], lw=1.8,
                connectionstyle="arc3,rad=0.05")
    ax.text(label_x, 0.53, "embedding (128)",
            fontsize=6.5, fontweight="bold", color=COLORS["fusion"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=COLORS["fusion"], alpha=0.85))

    # -- traj_stats + heads → Bridge --
    _draw_arrow(ax, enc_x + enc_w, enc_y + enc_h / 2 - 0.07,
                mod_x, y_bridge + mod_h / 2,
                COLORS["bridge"], lw=1.8,
                connectionstyle="arc3,rad=0.10")
    ax.text(label_x - 0.03, 0.38,
            "traj_stats + heads + emb",
            fontsize=6.5, fontweight="bold", color=COLORS["bridge"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=COLORS["bridge"], alpha=0.85))

    # -- Inter-module flows on right side --
    # Bridge → DT
    _draw_arrow(ax, mod_x + mod_w / 2, y_bridge,
                mod_x + mod_w / 2, y_dt + mod_h,
                COLORS["output"], lw=1.3)
    # Generator → DT
    _draw_arrow(ax, mod_x + mod_w - 0.03, y_gen,
                mod_x + mod_w - 0.03, y_dt + mod_h,
                COLORS["diffusion"], lw=1.0,
                connectionstyle="arc3,rad=0.12")
    # Temporal → DT
    _draw_arrow(ax, mod_x + 0.03, y_temp,
                mod_x + 0.03, y_dt + mod_h,
                COLORS["temporal"], lw=1.0,
                connectionstyle="arc3,rad=-0.12")

    # Right-side labels
    ax.text(mod_x + mod_w + 0.005, 0.50,
            "generated\ntopology", fontsize=5, color=COLORS["diffusion"],
            fontstyle="italic", rotation=-90)
    ax.text(mod_x - 0.04, 0.50,
            "anomaly\nscores", fontsize=5, color=COLORS["temporal"],
            fontstyle="italic")

    # ════════════════════════════════════════════════════════════════════
    # Optional re-encoding feedback
    # ════════════════════════════════════════════════════════════════════
    ax.annotate(
        "", xy=(enc_x + enc_w / 2, enc_y + enc_h),
        xytext=(mod_x + mod_w / 4, y_gen),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["dim"],
                        lw=1.0, linestyle=(0, (3, 5)),
                        connectionstyle="arc3,rad=-0.35"))
    ax.text(0.18, 0.73,
            "re-encode candidates\n(optional scoring only,\nno training signal)",
            fontsize=5.5, color=COLORS["dim"], ha="center", fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=COLORS["dim"], alpha=0.7))

    # ════════════════════════════════════════════════════════════════════
    # Training regime comparison (bottom panel)
    # ════════════════════════════════════════════════════════════════════
    comp_y = 0.0
    comp_h = 0.08

    # Two-stage box
    _draw_box(ax, 0.02, comp_y, 0.45, comp_h,
              "", COLORS["geometry"], fontsize=1, alpha=0.15,
              edgecolor=COLORS["geometry"])
    ax.text(0.245, comp_y + comp_h - 0.015,
            "Two-Stage Training (T5 row a)", ha="center",
            fontsize=8, fontweight="bold", color=COLORS["geometry"])
    ax.text(0.245, comp_y + 0.035,
            "1. Train E alone (contrastive) → freeze E", ha="center",
            fontsize=6.5, color=COLORS["text"])
    ax.text(0.245, comp_y + 0.015,
            "2. Train GRU + anomaly head with frozen E → BCE only",
            ha="center", fontsize=6.5, color=COLORS["text"])

    # Joint box
    _draw_box(ax, 0.52, comp_y, 0.48, comp_h,
              "", COLORS["highlight"], fontsize=1, alpha=0.15,
              edgecolor=COLORS["highlight"])
    ax.text(0.76, comp_y + comp_h - 0.015,
            "Joint Training (T5 rows b,c)  ← your current setup",
            ha="center", fontsize=8, fontweight="bold", color="#B8860B")
    ax.text(0.76, comp_y + 0.035,
            "Single model: E + GRU + anomaly + contrastive heads — all trainable",
            ha="center", fontsize=6.5, color=COLORS["text"])
    ax.text(0.76, comp_y + 0.015,
            "Loss = α·BCE + β·InfoNCE + role regression → backprop through E",
            ha="center", fontsize=6.5, color=COLORS["text"])

    # Summary note
    ax.text(0.5, -0.055,
            "Key insight: E↔T is end-to-end (joint loss shapes encoder).  "
            "E→G is NOT end-to-end (diffusion trains on frozen embeddings — "
            "encoder doesn't learn what makes good generation conditioning).",
            ha="center", fontsize=7, color=COLORS["dim"], fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#DDD", alpha=0.8))

    plt.tight_layout()
    path = output_dir / "A5_embedding_connectivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  A5 saved to {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate architecture diagrams")
    parser.add_argument("--output-dir", type=str, default="results/architecture_figures")
    parser.add_argument("--figures", nargs="*", default=["A1", "A2", "A3", "A4", "A5"],
                        help="Which figures to generate (A1 A2 A3 A4 A5)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_map = {
        "A1": figure_a1,
        "A2": figure_a2,
        "A3": figure_a3,
        "A4": figure_a4,
        "A5": figure_a5,
    }

    for fig_name in args.figures:
        fig_name = fig_name.upper()
        if fig_name in figure_map:
            print(f"Generating {fig_name}...")
            figure_map[fig_name](output_dir)
        else:
            print(f"Unknown figure: {fig_name}")


if __name__ == "__main__":
    main()
