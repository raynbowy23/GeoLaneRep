# Contrastive Lane Encoder — Training & Evaluation
# =================================================

# Optional local-only targets (e.g. GeoLane-Twin experiment pipeline).
# Missing is fine — the `-` prefix suppresses the error.
-include .twin/Makefile

.PHONY: help train viz-contrastive viz-contrastive-all encoder-figures assign assign-cam assign-viz annotate train-joint t5-ablation t5-figures temporal-figures-joint viz-contrastive-joint extract extract-cam taxonomy-figures train-generation-scratch train-generation-relational-scratch train-generation-relational comparison-table all-figures clean

PYTHON     ?= python
export QT_QPA_PLATFORM ?= offscreen
export MPLBACKEND      ?= Agg
CONFIG     ?= configs/lane_contrastive.yaml
ANNOT_DIR  ?= ../graph_geolane_annotator
CHECKPOINT ?=
CAMERA     ?=
HELD_OUT   ?=
VIDEO_DIR  ?= ./dataset/511video
CAMERA_LIST ?= ./dataset/camera_location_list.txt
PREPROCESS_DIR ?= ../dataset/preprocess
PARALLEL   ?= 4
DETECTION_PERIOD ?= 3600

help:
	@echo ""
	@echo "\033[34mContrastive Lane Encoder\033[0m"
	@echo "========================="
	@echo ""
	@echo "Training:"
	@echo "  make train                         Train contrastive lane encoder"
	@echo "  make train HELD_OUT=I43_Keefe      Train with held-out camera"
	@echo ""
	@echo "Evaluation:"
	@echo "  make comparison-table              Run all methods + LOCO, write per-method zero-shot JSONs + comparison table"
	@echo "  make encoder-figures               Generate E2/E3/E4 evaluation figures"
	@echo ""
	@echo "Visualization:"
	@echo "  make viz-contrastive               Visualize embeddings/matching/heatmap"
	@echo "  make viz-contrastive-all           Leave-one-out viz for all cameras"
	@echo ""
	@echo "Lane Assignment (geometric):"
	@echo "  make assign                        Assign tracklets to lanes (all cameras)"
	@echo "  make assign-cam CAMERA=X           Assign for one camera"
	@echo "  make assign-viz                    Visualize lane assignments"
	@echo ""
	@echo "Temporal Encoder:"
	@echo "  make train-joint                   Train joint contrastive+temporal encoder"
	@echo "  make t5-ablation                   Run T5 ablation: joint scratch vs joint warm-start"
	@echo "  make t5-figures                    Generate T5 comparison figures from training results"
	@echo "  make temporal-figures-joint        Generate T1-T4 figures (joint checkpoint)"
	@echo "  make viz-contrastive-joint         Visualize contrastive embeddings (joint checkpoint)"
	@echo ""
	@echo "Behavioral Taxonomy (Module 3):"
	@echo "  make taxonomy-figures              Generate M1/M2/M3 taxonomy figures"
	@echo ""
	@echo "Generation Training (checkpoints fed into trajectory figures):"
	@echo "  make train-generation-scratch            Baseline diffusion only, no pretraining"
	@echo "  make train-generation-relational-scratch Baseline + relational, no pretraining (best)"
	@echo "  make train-generation-relational         Baseline + relational with OpenLaneV2 pretrain"
	@echo ""
	@echo "Preprocessing:"
	@echo "  make extract                       Extract trajectories from all camera videos"
	@echo "  make extract-cam CAMERA=X          Extract trajectories from single camera"
	@echo ""
	@echo "  make all-figures                   Generate all paper figures (E2-E6, T5, M1-M3)"
	@echo ""
	@echo "Other:"
	@echo "  make annotate                      Open the lanelet annotator tool"
	@echo "  make clean                         Remove __pycache__ directories"
	@echo ""

# ── Training ──────────────────────────────────────────────

train:
	$(PYTHON) scripts/train.py --mode contrastive --config $(CONFIG) $(if $(HELD_OUT),--held-out $(HELD_OUT),)

# ── Evaluation ───────────────────────────────────────────

encoder-figures:
	$(PYTHON) scripts/generate.py encoder $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) --output-dir results/joint_encoder/figures

# ── Visualization ────────────────────────────────────────

viz-contrastive:
	$(PYTHON) scripts/visualize_contrastive.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) $(if $(CONFIG),--config $(CONFIG),) $(if $(HELD_OUT),--held-out $(HELD_OUT),)

viz-contrastive-all:
	$(PYTHON) scripts/visualize_contrastive.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) $(if $(CONFIG),--config $(CONFIG),) --all-cameras

# ── Lane Assignment (geometric) ──────────────────────────

assign:
	$(PYTHON) scripts/run_assignment.py --config $(CONFIG)

assign-cam:
	$(PYTHON) scripts/run_assignment.py --config $(CONFIG) --camera $(CAMERA)

assign-viz:
	$(PYTHON) scripts/visualize_assignment.py --config $(CONFIG)

# ── Annotation ───────────────────────────────────────────

annotate:
	@echo "\033[34mOpening Lanelet Annotator...\033[0m"
	cd $(ANNOT_DIR) && uv run python main.py

# ── Temporal Encoder ────────────────────────────────────

train-joint:
	$(PYTHON) scripts/train.py --mode joint --config $(CONFIG) \
		$(if $(CHECKPOINT),--encoder-checkpoint $(CHECKPOINT),) \
		$(if $(HELD_OUT),--held-out $(HELD_OUT),)

# ── T5 Ablation ───────────────────────────────────────
# Compares joint variants: (a) joint from scratch, (b) joint warm-start.
# Results saved under results/joint_encoder/ and results/joint_encoder_warm/.

t5-ablation:
	@echo "\033[34m[T5] (a) Joint from scratch\033[0m"
	$(PYTHON) scripts/train.py --mode joint --config $(CONFIG) \
		--held-out I43_Keefe I43_Walnut
	@echo "\033[34m[T5] (b) Joint warm-start\033[0m"
	$(PYTHON) scripts/train.py --mode joint --config $(CONFIG) \
		--encoder-checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/lane_contrastive/checkpoints/best.pt) \
		--held-out I43_Keefe I43_Walnut
	@echo "\033[32m[T5] Both variants complete. Compare results/ directories.\033[0m"

t5-figures:
	$(PYTHON) scripts/generate.py temporal --figures T5 \
		--output-dir results/joint_encoder/figures \
		--joint-scratch results/joint_encoder/history.json \
		--joint-warm results/joint_encoder_warm/history.json

temporal-figures-joint:
	$(PYTHON) scripts/generate.py temporal --config $(CONFIG) \
		--checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/joint_encoder/checkpoints/best.pt) \
		--joint --output-dir results/joint_encoder/figures \
		--figures T1 T2 T3 T4

viz-contrastive-joint:
	$(PYTHON) scripts/visualize_contrastive.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(HELD_OUT),--held-out $(HELD_OUT),) \
		--output-dir results/joint_encoder/visualizations

# ── Behavioral Taxonomy (Module 3) ──────────────────────

taxonomy-figures:
	$(PYTHON) scripts/generate.py taxonomy \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),)

# ── Trajectory-Based Lane Generation ─────────────────────────

trajectory-gen:
	$(PYTHON) scripts/generate.py trajectory --mode lanes \
		--config $(CONFIG) \
		--camera $(if $(GEN_CAMERA),$(GEN_CAMERA),US12_Yahara) \
		--group-id $(if $(GEN_GROUP),$(GEN_GROUP),0) \
		--output-dir results/generation/trajectory \
		$(if $(SPECS),--specs $(SPECS),)

# Hybrid: trajectory placement + baseline diffusion shape
# Requires CHECKPOINT (encoder) and DIFFUSION_CHECKPOINT.
trajectory-gen-hybrid:
	$(PYTHON) scripts/generate.py trajectory --mode lanes \
		--config $(CONFIG) \
		--camera $(if $(GEN_CAMERA),$(GEN_CAMERA),US12_Yahara) \
		--group-id $(if $(GEN_GROUP),$(GEN_GROUP),0) \
		--output-dir results/generation/trajectory \
		--use-diffusion \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(DIFFUSION_CHECKPOINT),--diffusion-checkpoint $(DIFFUSION_CHECKPOINT),) \
		$(if $(SPECS),--specs $(SPECS),)

# Behavioral conditioning comparison: same position, different lane types
behavior-comparison:
	$(PYTHON) scripts/generate.py trajectory --mode compare \
		--config $(CONFIG) \
		--camera $(if $(GEN_CAMERA),$(GEN_CAMERA),US12_Yahara) \
		--group-id $(if $(GEN_GROUP),$(GEN_GROUP),0) \
		--checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/joint_encoder/checkpoints/best.pt) \
		--diffusion-checkpoint $(if $(DIFFUSION_CHECKPOINT),$(DIFFUSION_CHECKPOINT),results/generation/figures/diffusion_model.pt) \
		$(if $(REL_DIFFUSION_CHECKPOINT),--relational-checkpoint $(REL_DIFFUSION_CHECKPOINT),) \
		--output-dir results/generation/trajectory

# Hybrid: trajectory placement + relational diffusion (scene-aware, best option)
# Requires CHECKPOINT, DIFFUSION_CHECKPOINT, and REL_DIFFUSION_CHECKPOINT.
trajectory-gen-relational:
	$(PYTHON) scripts/generate.py trajectory --mode lanes \
		--config $(CONFIG) \
		--camera $(if $(GEN_CAMERA),$(GEN_CAMERA),US12_Yahara) \
		--group-id $(if $(GEN_GROUP),$(GEN_GROUP),0) \
		--output-dir results/generation/trajectory \
		--use-diffusion \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),) \
		$(if $(DIFFUSION_CHECKPOINT),--diffusion-checkpoint $(DIFFUSION_CHECKPOINT),) \
		$(if $(REL_DIFFUSION_CHECKPOINT),--relational-checkpoint $(REL_DIFFUSION_CHECKPOINT),) \
		$(if $(SPECS),--specs $(SPECS),)

# ── Generation Training (diffusion + optional relational) ─────────
# The figure-producing side of generation lives in generate_trajectory_figures.py
# (trajectory-grounded). These targets only train the diffusion models.

DIFFUSION_EPOCHS         ?= 1000
DIFFUSION_CHECKPOINT     ?= results/generation/figures/diffusion_model.pt
REL_DIFFUSION_EPOCHS     ?= 1000
REL_DIFFUSION_CHECKPOINT ?= results/generation/figures/relational_diffusion_model.pt
NO_RELATION_RATIO        ?= 0.3
AUGMENT_FACTOR           ?= 10
OPENLANE_DATA            ?= results/generation/openlane_canonical.npz
PRETRAIN_EPOCHS          ?= 500

# Baseline diffusion only — no OpenLaneV2 pretraining.
train-generation-scratch:
	$(PYTHON) scripts/train_generation.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--diffusion-epochs $(DIFFUSION_EPOCHS)

# Baseline + relational, no pretraining (best-performing setup per project memory).
train-generation-relational-scratch:
	$(PYTHON) scripts/train_generation.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--diffusion-epochs $(DIFFUSION_EPOCHS) \
		--train-relational --relational-epochs $(REL_DIFFUSION_EPOCHS) \
		--no-relation-ratio $(NO_RELATION_RATIO) \
		--augment-factor $(AUGMENT_FACTOR)

# Baseline + relational with OpenLaneV2 Stage-1 pretraining.
train-generation-relational:
	$(PYTHON) scripts/train_generation.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--diffusion-epochs $(DIFFUSION_EPOCHS) \
		--openlane-data $(OPENLANE_DATA) --pretrain-epochs $(PRETRAIN_EPOCHS) \
		--train-relational --relational-epochs $(REL_DIFFUSION_EPOCHS) \
		--no-relation-ratio $(NO_RELATION_RATIO) \
		--augment-factor $(AUGMENT_FACTOR)

DT_EXPORT_DIR        ?= results/dt_state
DT_CAMERA            ?= US12_Monona
SUMO_ROOT            ?=

# ── Preprocessing (video → trajectories) ────────────────

extract:
	$(PYTHON) scripts/extract_video.py \
		--list $(CAMERA_LIST) \
		--video-dir $(VIDEO_DIR) \
		--output $(PREPROCESS_DIR) \
		--detection-period $(DETECTION_PERIOD) \
		--parallel $(PARALLEL)

extract-cam:
	$(PYTHON) scripts/extract_video.py \
		--video $(VIDEO_DIR)/$(CAMERA).mp4 \
		--camera $(CAMERA) \
		--output $(PREPROCESS_DIR) \
		--detection-period $(DETECTION_PERIOD)

# ── Comparison & Baselines ──────────────────────────────

comparison-table:
	$(PYTHON) scripts/eval_baseline.py --all \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

all-figures: encoder-figures t5-figures taxonomy-figures
	@echo "\033[32mAll figures generated.\033[0m"

# ── Cleanup ──────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32mCleaned\033[0m"
