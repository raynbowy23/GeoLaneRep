# GeoLaneRep Architecture

System-level view of how the pieces fit together. Implementation details for individual modules live in [`src_overview.md`](src_overview.md); training and generation theory are stored in [`problem_method.md`](problem_method.md).

## Pipeline

```
                         ┌──────────────────────────────────────────────┐
                         │              roadside camera feed             │
                         └──────────────────┬───────────────────────────┘
                                            │
   ┌────────────────────────────────────────▼─────────────────────────────────────┐
   │                              Preprocessing                                    │
   │                                                                               │
   │  scripts/extract_video.py   YOLOv11n + tracker → trajectory.csv per camera    │
   │  scripts/run_assignment.py  geometric tracklet → lane assignment              │
   │                                                                               │
   │  outputs:  results/preprocess/{cam}/trajectory.csv                            │
   │            results/preprocessing/lane_assignment/{cam}/lane_assignments.csv   │
   └────────────────────────────────────────┬─────────────────────────────────────┘
                                            │
   ┌────────────────────────────────────────▼─────────────────────────────────────┐
   │                                  Encoder                                      │
   │                                                                               │
   │  src/data/lane_dataset.py             per-window LaneSample construction      │
   │  src/data/temporal_dataset.py         windowed sequences for the temporal head│
   │  src/models/lane_encoder.py           geom + traj + descriptor → 128-dim z    │
   │  src/models/cross_lane_attention.py   group-aware attention with rel. bias   │
   │  src/models/joint_encoder.py          GRU + anomaly head over per-window z   │
   │  src/training/contrastive.py          Stage 1: contrastive + role regression │
   │  src/training/joint_trainer.py        Stage 2: contrastive + temporal joint  │
   │                                                                               │
   │  entrypoints:  scripts/train.py --mode {contrastive,joint}                    │
   │  outputs:      results/lane_contrastive/checkpoints/best.pt                   │
   │                results/joint_encoder/checkpoints/best.pt                      │
   └────────────────────────────────────────┬─────────────────────────────────────┘
                                            │
   ┌──────────────────────┬─────────────────┴──────────────────┬───────────────────┐
   │   Cross-camera       │       Temporal anomaly              │   Generation      │
   │   matching           │       detection                     │                   │
   │                      │                                     │                   │
   │  src/training/       │  src/models/joint_encoder.py        │  src/generation/  │
   │   zero_shot_eval.py  │   (GRU + anomaly head)              │   diffusion.py    │
   │  src/zero_shot_      │  src/training/temporal_trainer.py   │   directed.py     │
   │   lanes.py           │   (synthetic anomaly injection)     │   relational_*.py │
   │                      │                                     │   trajectory_     │
   │  scripts/            │  scripts/generate.py temporal       │    gen.py         │
   │   eval_baseline.py   │                                     │                   │
   │                      │                                     │  scripts/         │
   │  outputs: per-method │  outputs: T1-T4 figures             │   train_          │
   │   results/zero_shot/ │   in results/joint_encoder/figures/ │    generation.py  │
   │                      │                                     │   generate_       │
   │                      │                                     │    trajectory_    │
   │                      │                                     │    figures.py     │
   └──────────────────────┴─────────────────────────────────────┴───────────────────┘
```

## Stage 1 — Preprocessing

`extract_video.py` runs YOLOv11n on each 511 Wisconsin camera video, applies persistent tracking, and writes `trajectory.csv` (one row per frame×track) plus `last_frame.npy`. `run_assignment.py` then assigns each tracklet to a hand-annotated lane by mean point-to-polyline distance under a 60-pixel threshold; rejected tracks are flagged out-of-bounds. Output is per-camera `lane_assignments.csv` plus a top-level `summary.csv`.

The dataset feeding the encoder is built lazily inside `src/data/lane_dataset.py` from these CSVs and the per-camera `annotation.json`. Static lane geometry is arc-length-resampled to *K*=16 points; trajectories are organized into temporal windows (default 5-min) and resampled to the same *K*=16 polyline length.

## Stage 2 — Encoder

The encoder fuses three streams into a 128-dim per-window embedding:

| Stream    | Input                                  | Module                                   | Output |
|-----------|----------------------------------------|------------------------------------------|--------|
| Geometry  | (K, 2) lane polyline                   | Linear→Transformer→MeanPool→BN          | ℝ⁶⁴   |
| Trajectory| (Nₜ, K, 2) tracklets + validity mask   | Linear→Transformer→Masked MeanPool→BN   | ℝ⁶⁴   |
| Descriptor| 4 traj stats ‖ 5 role flags = ℝ⁹       | MLP(9→64→64) + BN                       | ℝ⁶⁴   |

Concatenation (ℝ¹⁹²) → `Linear(192→256) → GELU → dropout → Linear(256→128)` → per-window embedding `z_{i,w}`. A static per-lane embedding is the mean of `z_{i,w}` across valid windows.

When cross-lane attention is enabled, lanes sharing a `group_id` are batched together and `cross_lane_attention.py` adds a multi-head self-attention pass with a per-head bias projected from three pairwise relative features (Δlateral, Δspeed, density ratio). The result is a group-aware embedding `z̃_i` used by the projection head; pre-attention `z_i` is what the role regression heads see.

**Geometry dropout** (p=0.2) zeros the geometry embedding during training so the encoder learns to operate from trajectories + descriptors alone — this is what makes zero-shot transfer to a new camera (without annotation) work.

## Training objectives

Stage 1 (`scripts/train.py --mode contrastive`):

```
L_E(e) = w_ctr(e) · L_ctr  +  w_role(e) · L_role
```

with weights scheduled in three phases:

| Phase                | w_ctr | w_role | Purpose                                |
|----------------------|------:|-------:|----------------------------------------|
| 0%–30%   structural  |   0.3 |    2.0 | role heads establish lane structure    |
| 30%–70%  balanced    |   1.0 |    1.0 | both objectives refine jointly         |
| 70%–100% alignment   |   2.0 |    0.5 | contrastive sharpens cross-camera fit  |

Positive pairs for InfoNCE are mined structurally **across cameras** (Δrank < 0.15 + identical edge flags + role cosine ≥ 0.8). The role loss combines lateral-rank, edge-flag (BCE w/ pos_weight=3), and group-size heads.

Stage 2 (`scripts/train.py --mode joint`):

```
L = α · L_temp  +  β · L_E
```

The encoder remains trainable; gradients flow from both the InfoNCE+role loss and a validity-weighted BCE over per-window anomaly logits supervised by synthetic injections (speed compression, trajectory dropout, lateral offset).

## Stage 3 — Downstream tasks

All three downstream tasks reuse the encoder weights; only generation adds a separately-trained head.

**Cross-camera matching.** `zero_shot_eval.py` runs leave-one-camera-out: encode every other camera as the reference bank, encode held-out queries with `geometry_dropout` engaged, return the cosine-nearest reference. The matched reference's structural attributes (rank, edge flags) transfer to the query as zero-shot labels.

**Anomaly detection.** `joint_encoder.py` exposes a GRU + MLP head over the per-window embedding sequence. At inference, sigmoid logits cross a Youden's-J-tuned threshold; the GRU keeps preceding-window context, so the decision at window *w* sees the lane's behavioral history.

**Behavior-conditioned generation.** A FiLM-conditioned DDPM (T_diff=100) warm-starts from a canonicalized anchor geometry rather than pure noise.  Each denoiser layer applies γ, β projections of the conditioning embedding.  Two variants share the same training pipeline:

- `src/generation/diffusion.py` — independent generation
- `src/generation/relational_diffusion.py` — adds neighbor geometry + merge-point context

Both are trained by `scripts/train_generation.py`; the figure-producing side uses `scripts/generate_trajectory_figures.py`, which anchors generation on real trajectories rather than sampling from the canonical prior alone.

## Repo layout

```
scripts/                  CLI entry points (train, eval, figures, preprocessing)
  train.py                  Stage 1 + Stage 2 trainer (--mode {contrastive,joint})
  train_generation.py       diffusion + relational diffusion trainer
  eval_baseline.py          unified comparison-table runner (Table 1)
  generate.py               figure-set dispatcher
  generate_*.py             per-figure-family producers
  extract_video.py          video → trajectory.csv
  run_assignment.py         tracklet → lane assignment
  visualize_*.py            on-image overlays

src/
  data/                   datasets (lane, temporal-windowed)
  models/                 encoder, GRU temporal head, joint encoder, attention
  training/               loss orchestration (contrastive, joint, temporal trainer)
  generation/             diffusion, relational diffusion, retrieval, specs
  zero_shot_lanes.py      pseudo-lane discovery + zero-shot prediction
  lane_assignment.py      geometric assignment helpers

configs/                  YAML configs
results/                  pipeline outputs (gitignored)
docs/                     this directory
Makefile                  canonical workflow
```

## Design choices worth flagging

1. **Stage-1 encoder receives only `traj_stats(4)`, never roles.** Earlier versions concatenated roles into the descriptor stream, which made the role-regression heads trivially solvable by passing through their own input. Roles are now supervision-only; the encoder must infer rank from trajectory behavior.

2. **Regression heads attach to pre-attention `z_i`, not post-attention `z̃_i`.** Cross-lane attention mixes within-group context, which would contaminate per-lane supervision. The projection head (used for contrastive alignment, where group context helps) does see `z̃_i`.

3. **Three-phase loss schedule.** Joint optimization of contrastive and role losses from epoch 0 caused contrastive gradients to fight role supervision. The schedule lets role establish lane structure first, then progressively shifts weight to cross-camera alignment.

4. **Geometry dropout, not augmentation.** Zero-shot transfer is achieved by training the encoder to operate without geometry on a fraction of batches, not by augmenting geometry with noise.

5. **Trajectory-grounded generation.** The diffusion model is trained on canonical lane geometries, but at inference the generator anchors on real vehicle trajectories. This avoids drift from the canonical prior and produces lanes physically located where vehicles actually drive.
