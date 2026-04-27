# Scripts Overview

One-stop reference for every script under `scripts/`. Each entry lists what the script does and representative usage. The scripts themselves now carry no module-level docstring — this file is the single source of truth.

Categories:

- [Entry points](#entry-points)
- [Training](#training)
- [Evaluation](#evaluation)
- [Figures](#figures)
- [Preprocessing & data prep](#preprocessing--data-prep)
- [Visualization](#visualization)

---

## Entry points

### `generate.py` — figure-set dispatcher

Thin CLI that routes `python scripts/generate.py <set> [args]` to one of the category-specific figure scripts. Per-set `--help` forwards to that script's own argparse.

Available sets (on this checkout):

- `encoder` → `generate_encoder_figures.py` (E2/E4)
- `temporal` → `generate_temporal_figures.py` (T1–T5)
- `taxonomy` → `generate_taxonomy_figures.py` (M1–M3)
- `trajectory` → `generate_trajectory_figures.py` (trajectory-grounded lanes)

```
python scripts/generate.py                          # usage + available sets
python scripts/generate.py encoder --checkpoint results/joint_encoder/checkpoints/best.pt
python scripts/generate.py temporal --figures T5
python scripts/generate.py trajectory --mode compare --camera US12_Yahara ...
```

---

## Training

### `train.py` — lane encoder (contrastive or joint)

Trains the lane encoder in one of two modes, sharing CLI plumbing.

- `--mode contrastive` — Stage 1: contrastive encoder only
- `--mode joint` — Stage 2: contrastive + temporal, optionally warm-started

```
# Stage 1
python scripts/train.py --mode contrastive --config configs/lane_contrastive.yaml
python scripts/train.py --mode contrastive --config configs/lane_contrastive.yaml --held-out I43_Keefe

# Stage 2
python scripts/train.py --mode joint --config configs/lane_contrastive.yaml
python scripts/train.py --mode joint --config configs/lane_contrastive.yaml \
    --encoder-checkpoint results/lane_contrastive/checkpoints/best.pt
```

### `train_generation.py` — diffusion models

Trains the baseline FiLM-conditioned diffusion model and optionally the relational diffusion model. Both checkpoints feed `generate_trajectory_figures.py`, which does the actual figure production grounded on real trajectories. Produces:

- `results/generation/figures/diffusion_model.pt`
- `results/generation/figures/relational_diffusion_model.pt` (if `--train-relational`)

```
# Baseline only, no OpenLaneV2 pretrain
python scripts/train_generation.py \
    --checkpoint results/joint_encoder/checkpoints/best.pt

# Baseline + relational, no pretrain (best-performing setup)
python scripts/train_generation.py \
    --checkpoint results/joint_encoder/checkpoints/best.pt \
    --train-relational

# With OpenLaneV2 Stage-1 pretraining
python scripts/train_generation.py \
    --checkpoint results/joint_encoder/checkpoints/best.pt \
    --train-relational \
    --openlane-data results/generation/openlane_canonical.npz \
    --pretrain-epochs 500
```

---

## Evaluation

### `eval_baseline.py` — unified comparison

Runs all baselines, ablations, and encoder methods through leave-one-camera-out and writes the comparison table. Also emits per-method `results/zero_shot/{method}.json` for the encoder checkpoints, so the comparison table and the per-method LOCO records come out of one invocation.

Methods:

- `traj-stats` — 5 hand-crafted trajectory stats (no roles, no encoder)
- `stats-oracle` — `traj_stats(4)` + `roles(5)` = 9-dim (uses ground-truth roles)
- `per-camera-sup` — per-camera supervised classifier (not zero-shot)
- `no-cross-attn` — encoder trained without cross-lane attention
- `encoder` — full encoder (contrastive-only or joint checkpoint)

```
python scripts/eval_baseline.py --method traj-stats
python scripts/eval_baseline.py --method per-camera-sup
python scripts/eval_baseline.py --method no-cross-attn \
    --checkpoint results/lane_contrastive_no_crossattn/checkpoints/best.pt
python scripts/eval_baseline.py --method encoder \
    --checkpoint results/joint_encoder/checkpoints/best.pt
python scripts/eval_baseline.py --all    # run every method in one pass
```

---

## Figures

All figure scripts are normally invoked via `scripts/generate.py <set>`; the direct commands below also work.

### `generate_encoder_figures.py` — E-series (encoder evaluation)

- **E2** — cross-camera lateral rank alignment: matched vs ground truth scatter
- **E4** — cross-camera behavioral alignment: similarity boxplot + heatmap

```
python scripts/generate_encoder_figures.py \
    --checkpoint results/joint_encoder/checkpoints/best.pt

# Or from a pre-saved cross-camera results JSON (no checkpoint needed)
python scripts/generate_encoder_figures.py \
    --results-json results/joint_encoder/cross_camera_results.json
```

### `generate_temporal_figures.py` — T-series (temporal + training ablation)

- **T1** — window size comparison: anomaly detection at 5/15/30-sec windows
- **T2** — anomaly score timeline with injected incident overlay
- **T3** — embedding trajectory (UMAP): single lane traced through an incident
- **T4** — embedding delta heatmap `‖e(t) − e(t−1)‖` per lane per window
- **T5** — joint scratch vs joint warm-start ablation (training curves only; reads history JSONs, does not require `--checkpoint`)

Legacy aliases: `2a=T3(UMAP)`, `2b=T2(timeline)`, `2c=T4(heatmap)`.

```
# T1–T4 (joint checkpoint)
python scripts/generate_temporal_figures.py \
    --config configs/lane_contrastive.yaml \
    --checkpoint results/joint_encoder/checkpoints/best.pt \
    --joint

# T5 ablation (no checkpoint needed)
python scripts/generate_temporal_figures.py --figures T5 \
    --joint-scratch results/joint_encoder/history.json \
    --joint-warm   results/joint_encoder_warm/history.json
```

### `generate_taxonomy_figures.py` — M-series (behavioral taxonomy)

- **M1** — UMAP of all lane embeddings colored by discovered cluster label
- **M2** — cluster behavioral statistics table (mean speed/density/lat_rank → HCM type)
- **M3** — cross-camera retrieval: query lane → top-3 similar from other cameras

```
python scripts/generate_taxonomy_figures.py \
    --checkpoint results/lane_contrastive/checkpoints/best.pt

# Joint checkpoint
python scripts/generate_taxonomy_figures.py \
    --checkpoint results/joint_encoder/checkpoints/best.pt
```

### `generate_trajectory_figures.py` — trajectory-grounded lane generation

Two modes, both anchored on real vehicle trajectories:

- `--mode lanes` — rightmost / leftmost / merge variants for a camera group,
  optionally with diffusion shape. Without `--use-diffusion`: pure trajectory
  output. With `--use-diffusion`: trajectory placement + diffusion candidates.
- `--mode compare` — same trajectory anchor, different behavioral specs
  (e.g. high-speed through lane vs slow merge lane). Demonstrates that
  behavior-conditioned diffusion produces visually distinct shapes at the same
  position.

```
# Pure trajectory output (no diffusion)
python scripts/generate_trajectory_figures.py --mode lanes --camera US12_Yahara

# Hybrid: trajectory placement + diffusion shape
python scripts/generate_trajectory_figures.py --mode lanes \
    --camera US12_Yahara --use-diffusion \
    --checkpoint results/joint_encoder/checkpoints/best.pt \
    --diffusion-checkpoint results/generation/figures/diffusion_model.pt

# Behavior comparison (requires diffusion)
python scripts/generate_trajectory_figures.py --mode compare \
    --camera US12_Yahara --group-id 0 \
    --checkpoint results/joint_encoder/checkpoints/best.pt \
    --diffusion-checkpoint results/generation/figures/diffusion_model.pt \
    --relational-checkpoint results/generation/figures/relational_diffusion_model.pt
```

## Preprocessing & data prep

### `extract_video.py` — trajectories from traffic-camera video

Extracts trajectory data from 511 camera videos. Supports parallel extraction across multiple cameras.

Outputs per camera (default: `results/preprocess/{camera}/`):

- `trajectory.csv` — vehicle trajectories
- `last_frame.npy` — last processed frame
- `collect_cars.npy` — vehicle detections
- `collect_det_dots_including_truck.npy` — extended detections

```
# Single video
python scripts/extract_video.py \
    --video dataset/511video/camera_001.mp4 \
    --camera camera_001 \
    --output results/preprocess

# All cameras from a list
python scripts/extract_video.py \
    --list dataset/camera_location_list.txt \
    --video-dir dataset/511video \
    --output results/preprocess \
    --parallel 4
```

### `run_assignment.py` — geometric lane assignment

Assigns tracklets to lanes via geometric proximity. Writes per-camera `lane_assignments.csv` under `results/preprocessing/lane_assignment/{camera}/`.

```
python scripts/run_assignment.py --config configs/lanelet_core.yaml
python scripts/run_assignment.py --config configs/lanelet_core.yaml --camera US12_Greenway
```

---

## Visualization

### `visualize_assignment.py` — lane-assignment overlays

Renders tracklets colored by assigned lane on the camera frame. Reads from `results/preprocessing/lane_assignment/` and writes PNGs under its `viz/` subdir.

```
python scripts/visualize_assignment.py --config configs/lanelet_core.yaml
python scripts/visualize_assignment.py --config configs/lanelet_core.yaml --camera US12_Greenway
```

### `visualize_contrastive.py` — contrastive embedding visualizations

Produces visualization PNGs:

- `embedding_space.png` — t-SNE colored by lateral rank
- `matching_grid.png` — top matched (query, ref) lane pairs
- `similarity_heatmap.png` — cosine similarity matrix by camera

Modes:

- `--held-out CAM` — single held-out camera visualizations
- `--all-cameras` — leave-one-out loop, one matching grid per camera
- *(neither)* — global embedding space + heatmap only

```
python scripts/visualize_contrastive.py --checkpoint best.pt
python scripts/visualize_contrastive.py --checkpoint best.pt --held-out I43_Keefe
python scripts/visualize_contrastive.py --checkpoint best.pt --all-cameras
```
