# src/ Module Overview

Module-purpose docstrings extracted from every `src/*.py` file.  The source files no longer carry these docstrings at the top — this file is the single source of truth.


## Contents

- [`src/ (root)`](#src-root)
- [`src/data/`](#srcdata)
- [`src/generation/`](#srcgeneration)
- [`src/models/`](#srcmodels)
- [`src/training/`](#srctraining)
- [`src/utils/`](#srcutils)

---

## `src/ (root)`

### `src/lane_assignment.py`

Geometric lane assignment: assign tracklets to annotated lanes.

No ML training required. Each tracklet is assigned to the annotation lane with the smallest perpendicular projection distance (in pixel space).

Pipeline:
  trajectory.csv
    → per track: project onto every annotation lane (all groups)
    → assign to nearest lane (or reject if too far)

---

### `src/zero_shot_lanes.py`

Zero-shot lane property prediction from contrastive encoder.

Given a camera with annotation geometry and trajectory data, builds pseudo-lanes from annotation waypoints (for lane boundaries) and predicts lane properties via the trained encoder's regression heads.

Pipeline:
    annotation.json + trajectory.csv
        → annotation-based lane building (trajectory assignment)
        → encode trajectory-only (geometry=zeros)
        → regression heads → predicted lateral_rank, edge flags, lane_count
        → cosine match to reference bank (diagnostic only)

---

## `src/data/`

### `src/data/annotation_loader.py`

Load hand-drawn annotation JSON files and convert to normalized [0,1] lane geometries.

---

### `src/data/lane_dataset.py`

Lane-level dataset for contrastive representation learning.

Each sample represents one annotated lane: its geometry (waypoints), assigned trajectories, aggregate statistics, and a structural role descriptor used for positive pair mining.

---

### `src/data/temporal_dataset.py`

Temporal lane dataset for time-windowed trajectory encoding.

Extends the lane dataset concept by preserving trajectory timestamps and splitting them into overlapping time windows. Each sample provides windowed trajectory data for a single lane, enabling temporal change detection.

Reuses geometry helpers and role computation from lane_dataset.py.

---

## `src/generation/`

### `src/generation/__init__.py`

Lane geometry generation from behavioral embeddings.

---

### `src/generation/augment.py`

Geometry augmentation and canonical space transforms for diffusion training.

Canonical space: each lane is centered at origin, rotated so start→end aligns with the positive x-axis, and scaled to unit arc length. This strips away camera-specific position/orientation/scale, leaving only the curvature pattern.  Both OpenLaneV2 and annotated lanes look identical in this space.

---

### `src/generation/diffusion.py`

Conditional diffusion model for lane geometry generation.

DDPM with FiLM conditioning on behavioral embeddings. Generates (K, 2) lane geometries conditioned on a target behavioral embedding.

Architecture:
    x_t (K*2=32) + t_emb (64) → FiLM-conditioned MLP → predicted noise ε

The model is small by design: lane geometries are low-dimensional (32 values), so 100 diffusion steps and a 3-layer MLP are sufficient.

---

### `src/generation/directed.py`

Directed lane geometry generator (Pipeline B).

Given a user specification (e.g., "rightmost lane in US12_Park group 0"), generates novel lane geometry that fits the target road section.

The diffusion model operates in canonical space (centered, aligned, unit-length). Generated shapes are denormalized back to image space using the target group's heading, scale, and centroid position.

Pipeline (baseline):
    spec → resolve to target embedding + spatial context
         → build warm-start from anchor lane (in anchor's canonical space)
         → diffuse conditioned on target embedding (in canonical space)
         → denormalize to image space using anchor's pose
         → post-process and score candidates

Pipeline (relational):
    spec → resolve to target embedding + spatial context
         → build warm-start from anchor lane (in NEIGHBOR's canonical space)
         → diffuse conditioned on embedding + relational context
         → denormalize to image space using NEIGHBOR's pose
         → generated lane appears at learned offset from neighbor

---

### `src/generation/openlane_preprocess.py`

Extract lane centerline geometries from OpenLaneV2 for diffusion pretraining.

Reads the OpenLaneV2 pickle dataset, projects 3D centerlines to 2D pixel coordinates using camera intrinsics/extrinsics, normalizes to [0,1], and resamples to K=16 points for compatibility with geolane_encoder format.

Usage:
    from src.generation.openlane_preprocess import extract_openlane_geometries
    geometries = extract_openlane_geometries(
        "/path/to/OpenLane-V2",
        split="train",
        max_lanes=10000,
    )
    # geometries.shape == (N, 16, 2)

---

### `src/generation/relational_diffusion.py`

Relational conditioning diffusion model for lane geometry generation.

Extends the baseline FiLM-conditioned DDPM with relational context:
  - neighbor_geom: geometry of the lane being merged into / diverged from
  - merge_point: where along the lane the topology change occurs [0,1]
  - offset: starting lateral distance from the neighbor

Architecture:
    x_t (32) + t_emb (64) → concat (96)
    cond (128) + rel_emb (64) → cond_aug (192)
    3× FiLM layers conditioned on cond_aug → predicted noise ε

The relational encoder is a small MLP that encodes (neighbor_geom, merge_point, offset, has_relation) → 64-dim vector, concatenated with the behavioral embedding for FiLM conditioning.

Coexists with LaneDenoiser for A/B comparison.

---

### `src/generation/relational_pairs.py`

Build relational training pairs for the relational diffusion model.

Extracts (target_lane, neighbor_lane, merge_point, offset) pairs from annotation relationships (successor, adjacent) and converts them to a shared canonical frame for diffusion training.

Key design: both target and neighbor are expressed in the NEIGHBOR's canonical frame.  The neighbor is centered at origin and aligned with the x-axis (standard canonical form), while the target retains its natural spatial offset.  This means the diffusion model learns to generate geometry at the correct lateral/longitudinal position relative to the neighbor — no post-hoc offset hack needed.

At inference the generated geometry is denormalized using the neighbor's pose (centroid, angle, scale) so it lands at the right location in image space automatically.

---

### `src/generation/retrieval.py`

Embedding-based lane geometry retrieval and warm-start interpolation.

Given a behavioral embedding e_target, retrieve the k nearest lanes by cosine similarity and produce a warm-start geometry via weighted interpolation.

---

### `src/generation/spec.py`

Lane specification resolution for directed generation (Pipeline B).

Resolves user-specified lane constraints (e.g., "rightmost lane in US12_Park group 0") to a target embedding and spatial context for the diffusion model.

---

### `src/generation/trajectory_gen.py`

Trajectory-based lane generation.

Generates plausible new lane geometries directly from trajectory data — no diffusion model, no canonical space, no camera calibration required.

The approach:
  1. Extract pseudo-lane centerlines from trajectory clusters.
  2. Estimate lane direction (tangent) and perpendicular from cluster layout.
  3. Estimate inter-lane spacing from cluster centroids.
  4. Generate new lane by offsetting the outermost cluster (rightmost/leftmost)
     or by synthesizing a converging path (merge).

All coordinates are in normalized [0, 1] image space.

---

## `src/models/`

### `src/models/cross_lane_attention.py`

Cross-lane self-attention within lane groups.

Lanes within the same group attend to each other, enabling the model to learn group-relative properties (e.g. "fastest lane = passing lane") from trajectory behavior rather than absolute statistics.

Pairwise relative features (additive attention bias):
- lateral_offset_diff: traj_stats[i,2] - traj_stats[j,2]
- speed_diff: traj_stats[i,0] - traj_stats[j,0]
- density_ratio: traj_stats[i,3] / (traj_stats[j,3] + eps)

---

### `src/models/joint_encoder.py`

Joint lane encoder for simultaneous contrastive + temporal training.

Combines a trainable LaneEncoder with a GRU temporal head and anomaly detector. Unlike LaneTemporalEncoder (which freezes the encoder), this model backpropagates through the encoder from both losses:

  - Contrastive path: per-window projections -> InfoNCE per window -> average
  - Temporal path: GRU over per-window embeddings -> anomaly head -> BCE

Architecture:
    geometry (static)          <- shared across all windows
    traj_polylines(w)          <- per window
    traj_stats(w) + roles      <- per window
            |
      LaneEncoder (trainable)
            |
      e_i(w)  <-  per-window embedding (B, W, 128)
            |                    |
      GRU over windows      proj_head per window
            |                    |
      anomaly_head           InfoNCE per window (averaged)
            |
      BCE loss

---

### `src/models/lane_encoder.py`

Lane encoder for contrastive representation learning.

Fuses annotation geometry, trajectory behavior, and aggregate statistics into a single lane embedding, then projects to a contrastive space.

Architecture:
    annotation waypoints  ->  PolylineEncoder  --+
                                                  +-> fusion MLP -> embedding -> projection head
    assigned trajectories ->  PolylineEncoder  --+
                                                  |
    traj_stats            ->  stats MLP        --+

Optional cross-lane attention (gated by use_cross_lane_attention):
    Per-lane embeddings packed by group_id
        -> MultiheadSelfAttention with pairwise relative feature bias
        -> Unpacked back to flat batch

Roles (lateral_rank, edge flags, group_size) are concatenated with traj_stats as encoder input (stats_dim=9 = 4 traj_stats + 5 role descriptor). They are also used in the contrastive loss (positive pair mining) and as regression targets (auxiliary supervision).

---

### `src/models/polyline_encoder.py`

Per-tracklet polyline encoder: local K-point polyline -> embedding.

---

### `src/models/temporal_encoder.py`

Temporal lane encoder for time-series anomaly detection.

Wraps a frozen LaneEncoder, encodes each time window independently, then feeds the sequence of embeddings through a GRU to capture temporal dynamics. An anomaly head predicts per-window anomaly scores.

Architecture:
    For each window w:
        e_w = frozen_lane_encoder._encode_per_lane(geometry, traj_w, mask_w, stats_w)
    [e_0, ..., e_{W-1}] -> GRU -> h_seq (B, W, embed_dim)
    h_seq -> anomaly_head -> anomaly_scores (B, W)

---

## `src/training/`

### `src/training/contrastive.py`

Contrastive training for lane representation learning.

InfoNCE loss with structural positive mining. Positives are lanes with similar structural roles across different cameras. Negatives are all other lanes in the batch (including same-camera lanes with different roles).

---

### `src/training/joint_trainer.py`

Joint training loop for contrastive + temporal lane encoder.

Trains both objectives simultaneously through a single trainable encoder:
  - Contrastive (InfoNCE + role regression): shapes encoder to produce structurally meaningful lane embeddings.
  - Temporal (BCE on synthetic anomalies): shapes encoder + GRU to detect per-window behavioral changes.

Total loss = alpha * temporal_loss + beta * contrastive_loss

---

### `src/training/temporal_trainer.py`

Training loop for temporal lane encoder with synthetic anomaly injection.

Trains the GRU + anomaly head on top of a frozen LaneEncoder. Anomalies are synthetically injected at training time: speed drops, count drops, and lateral shifts. The model learns to detect these temporal changes via BCE loss.

---

### `src/training/zero_shot_eval.py`

Cross-camera lane alignment evaluation for contrastive lane embeddings.

Given a trained LaneEncoder:
1. Encode all training-camera lanes (full-info: geometry + traj + roles) -> reference bank
2. On held-out camera: encode lanes (full-info) -> query set
3. Match via cosine similarity
4. Evaluate cross-camera alignment quality

All inputs (geometry from annotation, roles from OSM, trajectories from tracking) are freely available at deployment — this tests cross-camera generalization, not zero-shot inference.

---

## `src/utils/`

### `src/utils/contrastive_viz.py`

Visualization utilities for contrastive lane embeddings.

---

### `src/utils/visualization.py`

Visualization constants and helpers.

---
