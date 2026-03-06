# Lanelet Discovery v4 — Training & Inference
# =============================================

.PHONY: help train train-core cache cache-core cache-cam clean-cache clean-cache-core infer infer-core annotate lanegroup-viz assign assign-cam assign-viz lanelet-graph lanelet-graph-cam lanelet-graph-viz train-contrastive eval-zero-shot viz-contrastive viz-contrastive-all zero-shot-predict clean

PYTHON     ?= python
CONFIG     ?= configs/lanelet.yaml
CONFIG_CORE?= configs/lanelet_core.yaml
CONFIG_CTR ?= configs/lane_contrastive.yaml
ANNOT_DIR  ?= ../graph_geolane_annotator
CHECKPOINT ?=
CAMERA     ?=
SPLIT      ?= test
HELD_OUT   ?=

help:
	@echo ""
	@echo "\033[34mLanelet Discovery v4\033[0m"
	@echo "====================="
	@echo ""
	@echo "Training:"
	@echo "  make train             Train with lanelet.yaml"
	@echo "  make train-core        Train with lanelet_core.yaml"
	@echo ""
	@echo "Cache (preprocess graphs):"
	@echo "  make cache             Build graph cache (lanelet.yaml)"
	@echo "  make cache-core        Build graph cache (lanelet_core.yaml)"
	@echo "  make cache-cam CAMERA=US12_Greenway  Build+visualize one camera (SPLIT=test)"
	@echo ""
	@echo "Inference:"
	@echo "  make infer             Run inference (set CHECKPOINT=path)"
	@echo "  make infer-core        Run inference with core config"
	@echo ""
	@echo "Lane Assignment (geometric):"
	@echo "  make assign            Assign tracklets to lanes (all cameras)"
	@echo "  make assign-cam CAMERA=US12_Greenway  Assign for one camera"
	@echo "  make assign-viz        Visualize lane assignments"
	@echo ""
	@echo "Lanelet Graph (data-driven):"
	@echo "  make lanelet-graph     Build lanelet graphs from assignments (all cameras)"
	@echo "  make lanelet-graph-cam CAMERA=X  Build for one camera"
	@echo "  make lanelet-graph-viz Visualize lanelet graph vs annotation"
	@echo ""
	@echo "Visualization:"
	@echo "  make lanegroup-viz     Visualize lane groups per camera"
	@echo ""
	@echo "Annotation:"
	@echo "  make annotate          Open the lanelet annotator tool"
	@echo ""
	@echo "Contrastive Lane Learning:"
	@echo "  make train-contrastive           Train contrastive lane encoder"
	@echo "  make train-contrastive HELD_OUT=I43_Keefe  Train with held-out camera"
	@echo "  make eval-zero-shot CHECKPOINT=path  Leave-one-out zero-shot eval"
	@echo "  make viz-contrastive CHECKPOINT=path  Visualize embeddings/matching/heatmap"
	@echo "  make viz-contrastive-all CHECKPOINT=path  Leave-one-out viz for all cameras"
	@echo "  make zero-shot-predict CAMERA=X  Zero-shot lane detection on new camera"
	@echo ""
	@echo "  make clean-cache       Remove lanelet graph cache"
	@echo "  make clean-cache-core  Remove lanelet_core graph cache"
	@echo "  make clean             Remove __pycache__ directories"
	@echo ""

# ── Training ──────────────────────────────────────────────

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

train-core:
	$(PYTHON) scripts/train.py --config $(CONFIG_CORE)

# ── Graph cache ───────────────────────────────────────────

cache:
	$(PYTHON) scripts/build_cache.py --config $(CONFIG) --visualize

cache-core:
	$(PYTHON) scripts/build_cache.py --config $(CONFIG_CORE) --visualize

cache-cam:
	$(PYTHON) scripts/build_cache.py --config $(CONFIG_CORE) --camera $(CAMERA) --split $(SPLIT) --visualize

clean-cache:
	rm -rf results/lanelet/graph_cache/
	@echo "\033[32mCleaned lanelet graph cache\033[0m"

clean-cache-core:
	rm -rf results/lanelet_core/graph_cache/
	@echo "\033[32mCleaned lanelet_core graph cache\033[0m"

# ── Inference ─────────────────────────────────────────────

infer:
	$(PYTHON) scripts/inference.py --config $(CONFIG) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

infer-core:
	$(PYTHON) scripts/inference.py --config $(CONFIG_CORE) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

# ── Lane Assignment (geometric) ───────────────────────────

assign:
	$(PYTHON) scripts/run_assignment.py --config $(CONFIG_CORE)

assign-cam:
	$(PYTHON) scripts/run_assignment.py --config $(CONFIG_CORE) --camera $(CAMERA)

assign-viz:
	$(PYTHON) scripts/visualize_assignment.py --config $(CONFIG_CORE)

# ── Lanelet Graph (data-driven) ──────────────────────────

lanelet-graph:
	$(PYTHON) scripts/build_lanelet_graph.py --config $(CONFIG_CORE)

lanelet-graph-cam:
	$(PYTHON) scripts/build_lanelet_graph.py --config $(CONFIG_CORE) --camera $(CAMERA)

lanelet-graph-viz:
	$(PYTHON) scripts/visualize_lanelet_graph.py --config $(CONFIG_CORE)

# ── Contrastive Lane Learning ─────────────────────────────

train-contrastive:
	$(PYTHON) scripts/train_contrastive.py --config $(CONFIG_CTR) $(if $(HELD_OUT),--held-out $(HELD_OUT),)

eval-zero-shot:
	$(PYTHON) scripts/eval_zero_shot.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG_CTR),--config $(CONFIG_CTR),)

viz-contrastive:
	$(PYTHON) scripts/visualize_contrastive.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG_CTR),--config $(CONFIG_CTR),) $(if $(HELD_OUT),--held-out $(HELD_OUT),)

viz-contrastive-all:
	$(PYTHON) scripts/visualize_contrastive.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG_CTR),--config $(CONFIG_CTR),) --all-cameras

zero-shot-predict:
	$(PYTHON) scripts/run_zero_shot.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/lane_contrastive/checkpoints/best.pt) $(if $(CONFIG_CTR),--config $(CONFIG_CTR),) --camera $(CAMERA)

# ── Visualization ─────────────────────────────────────────

lanegroup-viz:
	$(PYTHON) scripts/visualize_lane_groups.py --config $(CONFIG_CORE)

# ── Annotation ────────────────────────────────────────────

annotate:
	@echo "\033[34mOpening Lanelet Annotator...\033[0m"
	cd $(ANNOT_DIR) && uv run python main.py

# ── Cleanup ───────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32mCleaned\033[0m"
