# Contrastive Lane Encoder — Training & Evaluation
# =================================================

.PHONY: help train eval-zero-shot viz-contrastive viz-contrastive-all encoder-figures assign assign-cam assign-viz annotate clean

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

# ── Cleanup ──────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32mCleaned\033[0m"
