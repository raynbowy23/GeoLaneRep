# Contrastive Lane Encoder — Training & Evaluation
# =================================================

.PHONY: help train eval-zero-shot viz-contrastive viz-contrastive-all encoder-figures assign assign-cam assign-viz annotate train-temporal temporal-figures train-joint t5-ablation t5-figures clean

PYTHON     ?= python
CONFIG     ?= configs/lane_contrastive.yaml
ANNOT_DIR  ?= ../graph_geolane_annotator
CHECKPOINT ?=
CAMERA     ?=
HELD_OUT   ?=

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
	@echo "  make eval-zero-shot                Leave-one-out zero-shot eval"
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
	@echo "  make train-temporal                Train temporal encoder (anomaly detection)"
	@echo "  make train-joint                   Train joint contrastive+temporal encoder"
	@echo "  make t5-ablation                   Run T5 ablation: two-stage vs joint vs joint+warm-start"
	@echo "  make t5-figures                    Generate T5 comparison figures from training results"
	@echo "  make temporal-figures              Generate figures 2a/2b/2c"
	@echo ""
	@echo "Other:"
	@echo "  make annotate                      Open the lanelet annotator tool"
	@echo "  make clean                         Remove __pycache__ directories"
	@echo ""

# ── Training ──────────────────────────────────────────────

train:
	$(PYTHON) scripts/train_contrastive.py --config $(CONFIG) $(if $(HELD_OUT),--held-out $(HELD_OUT),)

# ── Evaluation ───────────────────────────────────────────

eval-zero-shot:
	$(PYTHON) scripts/eval_zero_shot.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG),--config $(CONFIG),)

encoder-figures:
	$(PYTHON) scripts/generate_encoder_figures.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt)

# ── Visualization ────────────────────────────────────────

viz-contrastive:
	$(PYTHON) scripts/visualize_contrastive.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG),--config $(CONFIG),) $(if $(HELD_OUT),--held-out $(HELD_OUT),)

viz-contrastive-all:
	$(PYTHON) scripts/visualize_contrastive.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG),--config $(CONFIG),) --all-cameras

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

train-temporal:
	$(PYTHON) scripts/train_temporal.py --config $(CONFIG) \
		--encoder-checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/lane_contrastive/checkpoints/best.pt)

train-joint:
	$(PYTHON) scripts/train_joint.py --config $(CONFIG) \
		$(if $(CHECKPOINT),--encoder-checkpoint $(CHECKPOINT),) \
		$(if $(HELD_OUT),--held-out $(HELD_OUT),)

# ── T5 Ablation ───────────────────────────────────────
# Runs all three variants: (a) two-stage, (b) joint from scratch, (c) joint warm-start
# Results saved to results/temporal_encoder/, results/joint_encoder/ respectively

t5-ablation:
	@echo "\033[34m[T5] (a) Two-stage: frozen encoder + GRU\033[0m"
	$(PYTHON) scripts/train_temporal.py --config $(CONFIG) \
		--encoder-checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/lane_contrastive/checkpoints/best.pt)
	@echo "\033[34m[T5] (b) Joint from scratch\033[0m"
	$(PYTHON) scripts/train_joint.py --config $(CONFIG) \
		--held-out I43_Keefe I43_Walnut
	@echo "\033[34m[T5] (c) Joint warm-start\033[0m"
	$(PYTHON) scripts/train_joint.py --config $(CONFIG) \
		--encoder-checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/lane_contrastive/checkpoints/best.pt) \
		--held-out I43_Keefe I43_Walnut
	@echo "\033[32m[T5] All three variants complete. Compare results/ directories.\033[0m"

t5-figures:
	$(PYTHON) scripts/generate_t5_figure.py \
		--two-stage results/temporal_encoder/history.json \
		--joint-scratch results/joint_encoder/history.json \
		--joint-warm results/joint_encoder_warm/history.json

temporal-figures:
	$(PYTHON) scripts/generate_temporal_figures.py --config $(CONFIG) \
		--checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/temporal_encoder/checkpoints/best.pt) \
		--encoder-checkpoint results/lane_contrastive/checkpoints/best.pt

# ── Cleanup ──────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32mCleaned\033[0m"
