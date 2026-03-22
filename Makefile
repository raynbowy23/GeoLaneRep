# Contrastive Lane Encoder — Training & Evaluation
# =================================================

.PHONY: help train eval-zero-shot viz-contrastive viz-contrastive-all encoder-figures assign assign-cam assign-viz annotate train-temporal temporal-figures train-joint t5-ablation t5-figures temporal-figures-joint viz-contrastive-joint extract extract-cam taxonomy-figures generation-figures generation-figures-scratch generation-directed generation-locations generation-relational generation-relational-scratch generation-relational-directed generation-compare bridge-figures dt-figures dt-figures-d3 dt-live dt-synthesis-loop dt-precompute dt-precompute-variants dt-demo comparison-table c2-experiment-variants-all all-figures clean

PYTHON     ?= python
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
	@echo "  make temporal-figures              Generate T1-T4 figures (two-stage)"
	@echo "  make temporal-figures-joint        Generate T1-T4 figures (joint checkpoint)"
	@echo "  make viz-contrastive-joint         Visualize contrastive embeddings (joint checkpoint)"
	@echo ""
	@echo "Behavioral Taxonomy (Module 3):"
	@echo "  make taxonomy-figures              Generate M1/M2/M3 taxonomy figures"
	@echo ""
	@echo "Lane Generation (Module 3c):"
	@echo "  make generation-figures            G1-G3: all generation figures"
	@echo "  make generation-directed           G2-G3 from saved diffusion checkpoint"
	@echo "  make generation-locations          List available camera/group locations"
	@echo ""
	@echo "  Directed generation options:"
	@echo "    LANE_TYPE=rightmost|leftmost|merge  Lane type(s) to generate"
	@echo "    SPEC='fast straight rightmost lane' Natural language specification"
	@echo "    GEN_CAMERA=US12_Park GEN_GROUP=0    Target location"
	@echo "    MEAN_SPEED=0.8 MEAN_CURVATURE=0.01  Behavioral prefix"
	@echo "    N_CANDIDATES=10                     Number of candidates"
	@echo "    REPLACE_ROLE='5:rightmost'           Replace lane role (cls_id:role)"
	@echo "    REMOVE_LANE='5'                      Remove lane by cls_id"
	@echo "    TARGET_LANES=3                       Regenerate group with N lanes"
	@echo ""
	@echo "Relational Generation (Module 3d):"
	@echo "  make generation-relational         Train relational diffusion + generate figures"
	@echo "  make generation-relational-directed Relational directed generation from checkpoint"
	@echo "  make generation-compare            Side-by-side baseline vs relational comparison"
	@echo ""
	@echo "Preprocessing:"
	@echo "  make extract                       Extract trajectories from all camera videos"
	@echo "  make extract-cam CAMERA=X          Extract trajectories from single camera"
	@echo ""
	@echo "Digital Twin Integration (Module 6):"
	@echo "  make dt-precompute                 Pre-compute lane state for twin (D1)"
	@echo "  make dt-precompute-variants        Pre-compute + generate variants (D2)"
	@echo "  make dt-demo                       Full demo: pre-compute + launch instructions"
	@echo "  make dt-figures                    Generate D1/D2/D4 paper figures"
	@echo "  make dt-figures-d3                 D3: edit-type failure taxonomy"
	@echo "  make dt-synthesis-loop             D4: synthesis loop (behavioral metrics)"
	@echo "    DT_CAMERA=US12_Monona              Target camera for DT demo"
	@echo "    DT_EXPORT_DIR=shared/dt_state      Shared directory for twin"
	@echo ""
	@echo "CrossTraffic Bridge (Module 4):"
	@echo "  make bridge-figures                Generate C1/C2/C3 bridge figures"
	@echo "  make bridge-figures CALIBRATION=configs/camera_calibration.yaml"
	@echo "  make bridge-figures --validate     Also run CrossTraffic validator"
	@echo ""
	@echo "Comparison & Baselines:"
	@echo "  make comparison-table              Run all baselines + encoder, print table"
	@echo "  make all-figures                   Generate all paper figures (E2-E6, T5, M1-M3)"
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
	$(PYTHON) scripts/eval_zero_shot.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) $(if $(CONFIG),--config $(CONFIG),)

encoder-figures:
	$(PYTHON) scripts/generate_encoder_figures.py $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) --output-dir results/joint_encoder/figures

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
		--output-dir results/joint_encoder/figures \
		--two-stage results/temporal_encoder/history.json \
		--joint-scratch results/joint_encoder/history.json \
		--joint-warm results/joint_encoder_warm/history.json

temporal-figures:
	$(PYTHON) scripts/generate_temporal_figures.py --config $(CONFIG) \
		--checkpoint $(if $(CHECKPOINT),$(CHECKPOINT),results/temporal_encoder/checkpoints/best.pt) \
		--encoder-checkpoint results/lane_contrastive/checkpoints/best.pt \
		--figures T1 T2 T3 T4

temporal-figures-joint:
	$(PYTHON) scripts/generate_temporal_figures.py --config $(CONFIG) \
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
	$(PYTHON) scripts/generate_taxonomy_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),)

# ── Lane Generation (Module 4) ────────────────────────────

DIFFUSION_EPOCHS     ?= 1000
DIFFUSION_CHECKPOINT ?= results/generation/figures/diffusion_model.pt
LANE_TYPE            ?=
SPEC                 ?=
GEN_CAMERA           ?=
GEN_GROUP            ?=
MEAN_SPEED           ?=
MEAN_CURVATURE       ?=
N_CANDIDATES         ?= 5
REPLACE_ROLE         ?=
REMOVE_LANE          ?=
TARGET_LANES         ?=

# List available camera/group locations for directed generation
generation-locations:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--list-locations

OPENLANE_DATA        ?= results/generation/openlane_canonical.npz
PRETRAIN_EPOCHS      ?= 500

# G1-G3: two-stage training (OpenLaneV2 pretrain + geolane fine-tune) + all figures
generation-figures:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--train-diffusion --diffusion-epochs $(DIFFUSION_EPOCHS) \
		--openlane-data $(OPENLANE_DATA) --pretrain-epochs $(PRETRAIN_EPOCHS) \
		--figures G1 G2 G3

# G1-G3: train diffusion WITHOUT OpenLaneV2 pretraining (geolane data only)
generation-figures-scratch:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--train-diffusion --diffusion-epochs $(DIFFUSION_EPOCHS) \
		--figures G1 G2 G3

# G2-G3 only: load saved diffusion checkpoint for directed generation + quality metrics
# Examples:
#   make generation-directed                                       # default: rightmost/leftmost/merge
#   make generation-directed LANE_TYPE=rightmost GEN_CAMERA=US12_Park GEN_GROUP=0
#   make generation-directed SPEC="fast straight rightmost lane"
#   make generation-directed LANE_TYPE=merge MEAN_SPEED=0.8 MEAN_CURVATURE=0.01
#   make generation-directed REPLACE_ROLE="5:rightmost"            # replace cls=5 with rightmost role
#   make generation-directed REMOVE_LANE="5"                       # remove cls=5 from group
#   make generation-directed TARGET_LANES=3                        # regenerate with 3 lanes
generation-directed:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--n-candidates $(N_CANDIDATES) \
		$(if $(GEN_CAMERA),--camera $(GEN_CAMERA),) \
		$(if $(GEN_GROUP),--group-id $(GEN_GROUP),) \
		$(if $(LANE_TYPE),--lane-type $(LANE_TYPE),) \
		$(if $(SPEC),--spec "$(SPEC)",) \
		$(if $(MEAN_SPEED),--mean-speed $(MEAN_SPEED),) \
		$(if $(MEAN_CURVATURE),--mean-curvature $(MEAN_CURVATURE),) \
		$(if $(MEAN_LATERAL_OFFSET),--mean-lateral-offset $(MEAN_LATERAL_OFFSET),) \
		$(if $(REPLACE_ROLE),--replace-role $(REPLACE_ROLE),) \
		$(if $(REMOVE_LANE),--remove-lane $(REMOVE_LANE),) \
		$(if $(TARGET_LANES),--target-lanes $(TARGET_LANES),) \
		--figures G2 G3

# ── Relational Generation (Module 3d) ────────────────────

REL_DIFFUSION_EPOCHS     ?= 1000
REL_DIFFUSION_CHECKPOINT ?= results/generation/figures/relational_diffusion_model.pt
NO_RELATION_RATIO        ?= 0.3
AUGMENT_FACTOR           ?= 10

# Train relational diffusion with two-stage pretraining (default)
generation-relational:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--train-diffusion --diffusion-epochs $(DIFFUSION_EPOCHS) \
		--openlane-data $(OPENLANE_DATA) --pretrain-epochs $(PRETRAIN_EPOCHS) \
		--train-relational --relational-epochs $(REL_DIFFUSION_EPOCHS) \
		--no-relation-ratio $(NO_RELATION_RATIO) \
		--augment-factor $(AUGMENT_FACTOR) \
		--compare-relational \
		$(if $(GEN_CAMERA),--camera $(GEN_CAMERA),) \
		$(if $(GEN_GROUP),--group-id $(GEN_GROUP),) \
		--figures G1 G2 G3

# Train relational diffusion WITHOUT pretraining (original approach)
generation-relational-scratch:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--train-diffusion --diffusion-epochs $(DIFFUSION_EPOCHS) \
		--train-relational --relational-epochs $(REL_DIFFUSION_EPOCHS) \
		--no-relation-ratio $(NO_RELATION_RATIO) \
		--augment-factor $(AUGMENT_FACTOR) \
		--compare-relational \
		$(if $(GEN_CAMERA),--camera $(GEN_CAMERA),) \
		$(if $(GEN_GROUP),--group-id $(GEN_GROUP),) \
		--figures G1 G2 G3

# Relational directed generation from saved checkpoint
# Examples:
#   make generation-relational-directed LANE_TYPE=merge GEN_CAMERA=US12_Park GEN_GROUP=0
#   make generation-relational-directed SPEC="merge lane merging into rightmost"
#   make generation-relational-directed REPLACE_ROLE="5:rightmost"
generation-relational-directed:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--relational-checkpoint $(REL_DIFFUSION_CHECKPOINT) \
		--n-candidates $(N_CANDIDATES) \
		$(if $(GEN_CAMERA),--camera $(GEN_CAMERA),) \
		$(if $(GEN_GROUP),--group-id $(GEN_GROUP),) \
		$(if $(LANE_TYPE),--lane-type $(LANE_TYPE),) \
		$(if $(SPEC),--spec "$(SPEC)",) \
		$(if $(MEAN_SPEED),--mean-speed $(MEAN_SPEED),) \
		$(if $(MEAN_CURVATURE),--mean-curvature $(MEAN_CURVATURE),) \
		$(if $(MEAN_LATERAL_OFFSET),--mean-lateral-offset $(MEAN_LATERAL_OFFSET),) \
		$(if $(REPLACE_ROLE),--replace-role $(REPLACE_ROLE),) \
		$(if $(REMOVE_LANE),--remove-lane $(REMOVE_LANE),) \
		$(if $(TARGET_LANES),--target-lanes $(TARGET_LANES),) \
		--relational \
		--figures G2 G3

# Side-by-side comparison: baseline vs relational diffusion
generation-compare:
	$(PYTHON) scripts/generate_generation_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--relational-checkpoint $(REL_DIFFUSION_CHECKPOINT) \
		--n-candidates $(N_CANDIDATES) \
		$(if $(GEN_CAMERA),--camera $(GEN_CAMERA),) \
		$(if $(GEN_GROUP),--group-id $(GEN_GROUP),) \
		$(if $(REPLACE_ROLE),--replace-role $(REPLACE_ROLE),) \
		$(if $(REMOVE_LANE),--remove-lane $(REMOVE_LANE),) \
		$(if $(TARGET_LANES),--target-lanes $(TARGET_LANES),) \
		--compare-relational \
		--figures G2 G3

CALIBRATION          ?=
DT_EXPORT_DIR        ?= shared/dt_state
DT_CAMERA            ?= US12_Monona
LANE_MAPPING         ?=
CALIBRATION_POINTS   ?=
SUMO_ROOT            ?=

# ── Digital Twin Integration (Module 6) ───────────────────

# Pre-compute lane state for twin (D1)
# Writes lane_state.json + timing.json to shared/dt_state/
# Twin can be launched separately: cd ../geolane_twin && cargo run
dt-precompute:
	$(PYTHON) scripts/precompute_dt_state.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--camera $(DT_CAMERA) \
		--output-dir $(DT_EXPORT_DIR) \
		--extract-frame dataset/511video/$(DT_CAMERA).mp4

# Pre-compute lane state + generate variant proposals (D2)
# Writes lane_state.json + topology.json + variant_results.json + modifications.json
dt-precompute-variants:
	$(PYTHON) scripts/precompute_dt_state.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--camera $(DT_CAMERA) \
		--output-dir $(DT_EXPORT_DIR) \
		--extract-frame dataset/511video/$(DT_CAMERA).mp4 \
		--generate-variants \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT)

# Full D-series demo: pre-compute + launch twin (if cargo available)
# Run encoder pre-compute, then launch twin pointing at shared/dt_state/
dt-demo: dt-precompute-variants
	@echo ""
	@echo "\033[32mEncoder state written to $(DT_EXPORT_DIR)/\033[0m"
	@echo "\033[34mLaunch the twin separately:\033[0m"
	@echo "  cd ../geolane_twin && cargo run -- --encoder-bridge $(DT_EXPORT_DIR)"
	@echo ""

# Generate D-series paper figures (D1-D4)
dt-figures:
	$(PYTHON) scripts/generate_dt_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--figures D1 D2 D4

# D3: Edit-Type Failure Taxonomy (independent vs relational)
dt-figures-d3:
	$(PYTHON) scripts/generate_dt_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(DIFFUSION_CHECKPOINT),--diffusion-checkpoint $(DIFFUSION_CHECKPOINT),--diffusion-checkpoint results/diffusion/checkpoints/best.pt) \
		$(if $(RELATIONAL_CHECKPOINT),--relational-checkpoint $(RELATIONAL_CHECKPOINT),) \
		--figures D3

# D2 with live export to twin (legacy — use dt-precompute instead)
dt-live:
	$(PYTHON) scripts/generate_dt_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		--figures D2 --export-dir $(DT_EXPORT_DIR) --live \
		$(if $(LANE_MAPPING),--lane-mapping $(LANE_MAPPING),) \
		$(if $(CALIBRATION_POINTS),--calibration-points $(CALIBRATION_POINTS),)

# D4 synthesis loop figure (behavioral metrics comparison)
dt-synthesis-loop:
	$(PYTHON) scripts/generate_dt_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--figures D4

# ── C2: Behavioral Geometry Synthesis Loop ────────────────

# C2 figures: re-encoding fidelity visualization (from pre-computed results)
c2-figures:
	$(PYTHON) scripts/generate_c2_figures.py \
		--results-dir results/c2_synthesis_loop \
		--figures C2a C2b C2c

# C2 experiment: single camera re-encoding loop
c2-experiment:
	$(PYTHON) scripts/run_c2_experiment.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--camera $(DT_CAMERA)

# C2 experiment: all cameras
c2-experiment-all:
	$(PYTHON) scripts/run_c2_experiment.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--all-cameras

# C2 experiment: with variant generation (single camera)
c2-experiment-variants:
	$(PYTHON) scripts/run_c2_experiment.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--camera $(DT_CAMERA) \
		--generate-variants \
		--diffusion-checkpoint results/generation/figures/diffusion_model.pt

# C2 experiment: all cameras with relational generation
c2-experiment-relational-all:
	$(PYTHON) scripts/run_c2_experiment.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--all-cameras \
		--generate-variants \
		--diffusion-checkpoint results/generation/figures/diffusion_model.pt \
		--relational-checkpoint results/generation/figures/relational_diffusion_model.pt

# C2 experiment: all cameras with calibration + variant generation
c2-experiment-variants-all:
	$(PYTHON) scripts/run_c2_experiment.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--all-cameras \
		--generate-variants \
		--diffusion-checkpoint results/generation/figures/diffusion_model.pt

# ── Planning Evaluation ────────────────────────────────────
# Full pipeline: encoder → generator → SUMO → re-encode → compare
# Use higher flow rate to create congestion (LOS C/D) so lane addition shows impact
PLAN_FLOW_RATE ?= 3600

# Planning eval: single camera with generation pipeline
planning-eval:
	$(PYTHON) scripts/run_planning_eval.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--camera $(DT_CAMERA) \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--flow-rate $(PLAN_FLOW_RATE)

# Planning eval: all cameras with generation pipeline
planning-eval-all:
	$(PYTHON) scripts/run_planning_eval.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--all-cameras \
		--diffusion-checkpoint $(DIFFUSION_CHECKPOINT) \
		--flow-rate $(PLAN_FLOW_RATE)

# Planning eval: without generation (position-only, for testing)
planning-eval-quick:
	$(PYTHON) scripts/run_planning_eval.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(SUMO_ROOT),--sumo-root $(SUMO_ROOT),) \
		--camera $(DT_CAMERA) \
		--flow-rate $(PLAN_FLOW_RATE) \
		--no-generate

# Planning evaluation figures (D2 series)
planning-figures:
	$(PYTHON) scripts/generate_planning_figures.py \
		--results-dir results/planning_eval

# ── CrossTraffic Bridge (Module 4) ────────────────────────

bridge-figures:
	$(PYTHON) scripts/generate_bridge_figures.py \
		$(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),--checkpoint results/joint_encoder/checkpoints/best.pt) \
		$(if $(CONFIG),--config $(CONFIG),) \
		$(if $(CALIBRATION),--calibration $(CALIBRATION),) \
		--figures C1 C2 C3

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

all-figures: encoder-figures t5-figures taxonomy-figures generation-figures bridge-figures dt-figures
	@echo "\033[32mAll figures generated.\033[0m"

# ── Cleanup ──────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "\033[32mCleaned\033[0m"
