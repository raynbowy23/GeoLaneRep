# Reproducing the Paper

Maps every paper figure and table to the command that produces it. Numbers and discussion live in the paper itself; this file is the "how to run it" index.

## Prerequisites

```bash
uv sync # install dependencies (Python ≥3.13)
# Datasets must be available at ./dataset/ (16 cameras of 511 Wisconsin video
# + per-camera annotation.json + last_frame.npy). Not bundled with the repo.
```

## End-to-end pipeline

If starting from scratch (raw camera videos), run in order:

```bash
make extract                                              # video → trajectory.csv per camera
make assign                                               # geometric tracklet→lane assignment
make assign-viz                                           # (optional) sanity-check lane assignments
make train CONFIG=configs/lane_contrastive.yaml           # Stage 1 contrastive encoder
make train-joint CHECKPOINT=results/lane_contrastive/checkpoints/best.pt   # Stage 2 joint
make train-generation-relational-scratch                  # diffusion + relational diffusion
```

Once trained checkpoints exist, every figure/table below regenerates from them.

## Paper §5.1 — Setup

| Item | Source / how reproduced |
|---|---|
| Dataset stats (16 cams / 132 lanes / 38 groups / 104,415 trajectories) | derived during `make extract` + `make assign`; see `results/preprocessing/lane_assignment/summary.csv` |
| Preprocessing illustration (`preprocessing.png`) | manually composited from `results/preprocessing/lanegroup_viz/` and `results/preprocessing/tracklet_check/` |
| Conceptual figures (`geolanerep_pipeline.png`, `geolanerep_encoder.png`, `geolanerep_downstream.png`) | hand-drawn / not produced by this repo |

## Paper §5.2.1 — Comparison Across Models (Table 1)

```bash
make comparison-table
```

Outputs:

- `results/baseline/comparison_table.{json,csv,pdf,png}` — Table 1
- `results/zero_shot/{joint_encoder,lane_contrastive,lane_contrastive_no_crossattn}.json` — per-method LOCO breakdown

Runs `traj-stats`, `stats-oracle`, `per-camera-sup`, two-stage frozen, geometry-only ablation, trajectory-only ablation, contrastive-only encoder, joint encoder, One-Class SVM, LSTM (traj-stats) — every row in Table 1.

## Paper §5.2.2 — Training Behavior (Figs T5a / T5b)

```bash
make t5-ablation # trains joint-from-scratch and joint-warm-start variants
make t5-figures # T5a anomaly-accuracy curves + T5b contrastive quality
```

Outputs `results/joint_encoder/figures/T5a_accuracy_curves.png`, `T5b_contrastive_quality.png`, `T5_summary.csv`.

## Paper §5.3 — Representation Quality (Figs E2 / E4)

```bash
make encoder-figures
```

Outputs:

- `results/joint_encoder/figures/E2_lateral_rank.png` — paper Fig. `lateral_rank.png`
- `results/joint_encoder/figures/E4_embedding_similarity.png` — paper Fig. `embedding_similarity.png`

## Paper §5.4 — Anomaly Detection (Figs T1, T2, T3)

```bash
make temporal-figures-joint # T1 + T2 + T3 + T4 from joint checkpoint
```

Outputs under `results/joint_encoder/figures/`:

- `T1_window_size_comparison.png` — §5.4.2 window sweep
- `T2_anomaly_timeline.png` — §5.4.3 lane-role timelines
- `T3_roc_confusion.png` — §5.4.1 AUROC + confusion matrix
- `T4_embedding_delta_heatmap.png` — supplementary (not in paper)

## Paper §5.5 — Behavior-Conditioned Generation

### §5.5.1 Qualitative comparison (Fig. `US12_Yahara_g0_baseline_vs_relational.png`)

```bash
make behavior-comparison \
    GEN_CAMERA=US12_Yahara GEN_GROUP=0 \
    REL_DIFFUSION_CHECKPOINT=results/generation/figures/relational_diffusion_model.pt
```

Output: `results/generation/trajectory/US12_Yahara_g0_baseline_vs_relational.png`.

## Baselines (Table 1, manual mode)

`make comparison-table` runs all of these together. To run individually:

```bash
python scripts/eval_baseline.py --method traj-stats
python scripts/eval_baseline.py --method stats-oracle
python scripts/eval_baseline.py --method per-camera-sup
python scripts/eval_baseline.py --method svm
python scripts/eval_baseline.py --method lstm
python scripts/eval_baseline.py --method encoder \
    --checkpoint results/joint_encoder/checkpoints/best.pt
```

## Other documentation

- `docs/architecture.md` — pipeline / system-level overview
- `docs/scripts_overview.md` — what each `scripts/*.py` does
- `docs/src_overview.md` — what each `src/*.py` module does
- `docs/problem_method.md` — markdown port of paper §3–§4 (problem setup + method)
- `docs/future_improvements.md` — open items from the paper's limitations
