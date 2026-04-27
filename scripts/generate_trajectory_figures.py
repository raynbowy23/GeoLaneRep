#!/usr/bin/env python3

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPEC_COLORS = {
    "rightmost": "#e41a1c",
    "leftmost":  "#377eb8",
    "merge":     "#ff7f00",
}

LANE_COLORS = ["#1f78b4", "#33a02c", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3"]

BEHAVIORS = [
    dict(
        label="High-speed through lane",
        spec_kwargs=dict(mean_speed=0.85, mean_curvature=0.005, is_rightmost=True),
        color="#e41a1c",
        linestyle="-",
    ),
    dict(
        label="Slow merge lane",
        spec_kwargs=dict(mean_speed=0.25, mean_curvature=0.04, has_successor=True),
        color="#ff7f00",
        linestyle="-",
    ),
]


def _load_diffusion_generator(args, config):
    """Load encoder + diffusion model and return a DirectedLaneGenerator.

    Returns None on failure so callers in --mode lanes can fall back gracefully.
    --mode compare treats None as a fatal error.
    """
    try:
        from src.data.lane_dataset import LaneDataset
        from src.generation.retrieval import build_retrieval_index
        from src.generation.spec import SpecEmbeddingResolver
        from src.generation.directed import DirectedLaneGenerator
        from src.training.zero_shot_eval import load_trained_encoder
        from src.generation.diffusion import LaneDenoiser, DDPMSchedule, LaneDiffusionTrainer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config.get("model", {})
        train_cfg = config.get("contrastive_training", {})

        model, _ = load_trained_encoder(args.checkpoint, device)
        dataset = LaneDataset(
            config=config,
            polyline_k=model_cfg.get("polyline_k", 16),
            max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
            role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
        )
        index = build_retrieval_index(model, dataset, device, use_embeddings=True)

        K = index.K
        cond_dim = index.D
        denoiser = LaneDenoiser(geom_dim=K * 2, t_dim=64, cond_dim=cond_dim, hidden_dim=256)
        schedule = DDPMSchedule(T=100)
        trainer = LaneDiffusionTrainer(denoiser, schedule, device=str(device))

        ckpt_path = args.diffusion_checkpoint
        if ckpt_path is None:
            ckpt_path = str(Path(args.checkpoint).parent / "diffusion_model.pt")
        trainer.load(ckpt_path)
        logger.info(f"Loaded diffusion model from {ckpt_path}")

        rel_trainer = None
        rel_ckpt = getattr(args, "relational_checkpoint", None)
        if rel_ckpt:
            from src.generation.relational_diffusion import (
                RelationalLaneDenoiser, RelationalDiffusionTrainer,
            )
            rel_denoiser = RelationalLaneDenoiser(
                geom_dim=K * 2, t_dim=64, cond_dim=cond_dim,
                rel_dim=64, hidden_dim=256,
            )
            rel_trainer = RelationalDiffusionTrainer(rel_denoiser, schedule, device=str(device))
            rel_trainer.load(rel_ckpt)
            logger.info(f"Loaded relational diffusion model from {rel_ckpt}")

        resolver = SpecEmbeddingResolver(index, dataset)
        return DirectedLaneGenerator(
            resolver, trainer, encoder=model, dataset=dataset,
            device=str(device),
            relational_trainer=rel_trainer,
        )

    except Exception as e:
        logger.error(f"Failed to load diffusion generator: {e}")
        return None


def _load_frame_and_lanes(args, config):
    """Load camera frame (RGB) and build pseudo-lanes for the target group.

    Returns (frame_rgb, frame_shape, pseudo_lanes) or (None, None, None) on failure.
    """
    from src.data.annotation_loader import load_annotation_json
    from src.zero_shot_lanes import build_lanes_from_annotation

    annot_dir = Path(config["data"].get("annotation_dir", "../dataset/preprocess"))
    cam_dir = annot_dir / args.camera
    image_w = config["data"].get("image_width", 1920)
    image_h = config["data"].get("image_height", 1080)

    traj_path = cam_dir / "trajectory.csv"
    if not traj_path.exists():
        logger.error(f"No trajectory.csv at {traj_path}")
        return None, None, None
    traj_df = pl.read_csv(str(traj_path))
    logger.info(f"Loaded {len(traj_df)} trajectory points")

    frame = None
    for fname in ["last_frame.npy", "last_frame.png"]:
        fp = cam_dir / fname
        if fp.exists():
            frame = np.load(str(fp)) if fname.endswith(".npy") else cv2.imread(str(fp))
            break
    if frame is None:
        frame = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        logger.warning("No frame found, using black background")
    if frame.ndim == 3 and frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame
    frame_shape = (frame.shape[0], frame.shape[1])

    annot_path = cam_dir / "annotation.json"
    if not annot_path.exists():
        logger.error(f"No annotation.json at {annot_path}")
        return None, None, None
    annotation = load_annotation_json(annot_path)

    annotation_filtered = dict(annotation)
    if "lane_groups" in annotation:
        filtered = [g for g in annotation["lane_groups"]
                    if g.get("group_id", 0) == args.group_id]
        annotation_filtered = {**annotation, "lane_groups": filtered or annotation["lane_groups"]}
        if not filtered:
            logger.warning(f"Group {args.group_id} not found, using all groups")

    pseudo_lanes = build_lanes_from_annotation(
        annotation_filtered, traj_df, frame_shape, args.camera, config,
    )
    if not pseudo_lanes:
        logger.error("No pseudo-lanes built — check trajectory/annotation data")
        return None, None, None
    logger.info(f"Built {len(pseudo_lanes)} pseudo-lanes")
    return frame_rgb, frame_shape, pseudo_lanes


def _run_hybrid_diffusion(traj_result, traj_anchor, generator, spec_name, args):
    """Run diffusion generation using trajectory context. Returns GenerationResult or None."""
    from src.generation.spec import LaneSpecification
    try:
        spec = LaneSpecification(
            is_rightmost=(spec_name == "rightmost"),
            is_leftmost=(spec_name == "leftmost"),
            has_successor=(spec_name == "merge"),
            camera=args.camera,
            group_id=args.group_id,
        )

        existing = traj_result.existing_lanes
        if spec_name == "leftmost":
            neighbor = existing[0]
        else:
            neighbor = existing[-1]

        if generator.relational_trainer is not None and len(existing) >= 1:
            logger.info(f"[{spec_name}] Using relational diffusion with trajectory neighbor")
            return generator.generate_with_trajectory_context(
                trajectory_anchor=traj_anchor,
                neighbor_centerline=neighbor,
                spec=spec,
                n_candidates=args.n_candidates,
            )
        else:
            return generator.generate_with_trajectory_anchor(
                trajectory_anchor=traj_anchor,
                spec=spec,
                n_candidates=args.n_candidates,
            )
    except Exception as e:
        logger.warning(f"Hybrid diffusion failed for {spec_name}: {e}")
        return None


def _draw_candidates_col(ax, frame_rgb, traj_result, diff_result, traj_anchor,
                         spec_name, color):
    ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.45)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")

    for cl in traj_result.existing_lanes:
        ax.plot(cl[:, 0], cl[:, 1], color="#888888", linewidth=2, alpha=0.5)

    ax.plot(traj_anchor[:, 0], traj_anchor[:, 1], color="#444444", linewidth=1.5,
            linestyle="--", alpha=0.7, label="anchor", zorder=7)

    if diff_result is not None:
        n = len(diff_result.candidates)
        for cand in diff_result.candidates:
            ax.plot(cand[:, 0], cand[:, 1], color=color, linewidth=1.5,
                    alpha=0.35, zorder=8)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title(f"Generated lane candidates ({n})", fontsize=10)
    else:
        ax.plot(traj_anchor[:, 0], traj_anchor[:, 1], color=color, linewidth=3,
                label=f"{spec_name} (trajectory fallback)", zorder=10)
        ax.set_title("Lane generation (fallback)", fontsize=10)

    ax.set_xlabel("x (norm)"); ax.set_ylabel("y (norm)")


def _draw_best_col(ax, frame_rgb, traj_result, diff_result, traj_anchor,
                   spec_name, color):
    ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.45)
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")

    for cl in traj_result.existing_lanes:
        ax.plot(cl[:, 0], cl[:, 1], color="#888888", linewidth=2, alpha=0.5)

    gen = diff_result.best if diff_result is not None else traj_anchor
    ax.plot(gen[:, 0], gen[:, 1], color=color, linewidth=4,
            label=f"{spec_name} (generated)", zorder=10)
    ax.scatter(gen[0, 0], gen[0, 1], c="green", s=80, zorder=11)
    ax.scatter(gen[-1, 0], gen[-1, 1], c=color, s=80, zorder=11)
    ax.set_title(f"Generated lane\n({spec_name})", fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("x (norm)"); ax.set_ylabel("y (norm)")


def _log_embedding_difference(generator, args):
    """Log how different the two behavioral embeddings are (compare mode)."""
    from src.generation.spec import LaneSpecification
    embeddings = []
    for behavior in BEHAVIORS:
        spec = LaneSpecification.from_behavior(
            camera=args.camera, group_id=args.group_id, **behavior["spec_kwargs"],
        )
        try:
            emb, _ = generator.resolver.resolve(spec)
            embeddings.append(emb)
            logger.info(f"  [{behavior['label']}] embedding resolved")
        except Exception as e:
            logger.warning(f"  [{behavior['label']}] resolution failed: {e}")

    if len(embeddings) == 2:
        e1, e2 = embeddings
        cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
        l2 = np.linalg.norm(e1 - e2)
        logger.info(
            f"Embedding distance between behaviors: "
            f"cosine_sim={cos_sim:.3f}, L2={l2:.4f}"
        )
        if cos_sim > 0.99:
            logger.warning(
                "Embeddings are nearly identical — behavioral conditioning may not "
                "have enough signal to produce visually distinct outputs."
            )


def _run_lanes_mode(args, config):
    from src.generation.trajectory_gen import generate

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_diffusion and not args.checkpoint:
        logger.error("--use-diffusion requires --checkpoint")
        return

    frame_rgb, _, pseudo_lanes = _load_frame_and_lanes(args, config)
    if pseudo_lanes is None:
        return

    diffusion_generator = None
    if args.use_diffusion:
        diffusion_generator = _load_diffusion_generator(args, config)
        if diffusion_generator is None:
            logger.warning("Falling back to pure trajectory generation")

    n_specs = len(args.specs)
    n_cols = 3 if diffusion_generator is not None else 2
    fig_w = 6 * n_cols
    fig, axes = plt.subplots(n_specs, n_cols, figsize=(fig_w, 5 * n_specs), squeeze=False)
    mode_label = "Lane Generation by Diffusion" if diffusion_generator else "Trajectory-Based Lane Generation"
    fig.suptitle(
        f"{mode_label}\n{args.camera} group {args.group_id}",
        fontsize=14, fontweight="bold",
    )

    for row, spec_name in enumerate(args.specs):
        traj_result = generate(pseudo_lanes, spec=spec_name, k=args.k)
        if traj_result is None:
            logger.warning(f"Trajectory generation failed for spec={spec_name}")
            continue

        color = SPEC_COLORS[spec_name]

        # Col 0: existing lane centerlines
        ax = axes[row, 0]
        ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for li, cl in enumerate(traj_result.existing_lanes):
            lc = LANE_COLORS[li % len(LANE_COLORS)]
            ax.plot(cl[:, 0], cl[:, 1], color=lc, linewidth=2.5, alpha=0.9,
                    label=f"lane {li}")
        ax.set_title(f"Existing lanes ({len(pseudo_lanes)} clusters)", fontsize=10)
        ax.set_xlabel("x (norm)"); ax.set_ylabel("y (norm)")

        traj_gen = traj_result.generated
        logger.info(
            f"[{spec_name}] trajectory spacing={traj_result.spacing:.4f}, "
            f"perp={traj_result.perp.round(3)}, "
            f"traj_gen [{traj_gen.min():.3f}, {traj_gen.max():.3f}]"
        )

        if diffusion_generator is None:
            # Col 1: pure trajectory output
            ax = axes[row, 1]
            ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.45)
            ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
            for cl in traj_result.existing_lanes:
                ax.plot(cl[:, 0], cl[:, 1], color="#888888", linewidth=2, alpha=0.6)
            ax.plot(traj_gen[:, 0], traj_gen[:, 1], color=color, linewidth=4,
                    label=f"{spec_name} (trajectory)", zorder=10)
            ax.scatter(traj_gen[0, 0], traj_gen[0, 1], c="green", s=80, zorder=11)
            ax.scatter(traj_gen[-1, 0], traj_gen[-1, 1], c=color, s=80, zorder=11)
            ax.legend(fontsize=9, loc="lower right")
            ax.set_title({"rightmost": "New rightmost lane",
                          "leftmost": "New leftmost lane",
                          "merge": "Merge lane"}[spec_name], fontsize=10)
            ax.set_xlabel("x (norm)"); ax.set_ylabel("y (norm)")
        else:
            diff_result = _run_hybrid_diffusion(
                traj_result, traj_gen, diffusion_generator, spec_name, args,
            )
            _draw_candidates_col(axes[row, 1], frame_rgb, traj_result, diff_result,
                                 traj_gen, spec_name, color)
            _draw_best_col(axes[row, 2], frame_rgb, traj_result, diff_result,
                           traj_gen, spec_name, color)

    plt.tight_layout()
    out_path = output_dir / f"{args.camera}_g{args.group_id}_trajectory_gen.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved to {out_path}")


def _run_compare_mode(args, config):
    from src.generation.trajectory_gen import generate as traj_generate
    from src.generation.spec import LaneSpecification

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_rgb, _, pseudo_lanes = _load_frame_and_lanes(args, config)
    if pseudo_lanes is None:
        return

    generator = _load_diffusion_generator(args, config)
    if generator is None:
        logger.error("compare mode requires a working diffusion generator — aborting.")
        return

    traj_result = traj_generate(pseudo_lanes, spec="rightmost", k=args.k)
    if traj_result is None:
        logger.error("Trajectory generation failed")
        return

    traj_anchor = traj_result.generated
    neighbor = traj_result.existing_lanes[-1]

    n_rows = len(BEHAVIORS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)
    fig.suptitle(
        f"Behavioral Conditioning: Same Position, Different Lane Types\n"
        f"{args.camera} group {args.group_id}",
        fontsize=14, fontweight="bold",
    )
    col_titles = [
        "Existing lanes (context)",
        "Trajectory anchor\n(same position for both)",
        "Diffusion output\n(conditioned on behavior)",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for row, behavior in enumerate(BEHAVIORS):
        color = behavior["color"]
        label = behavior["label"]
        spec = LaneSpecification.from_behavior(
            camera=args.camera, group_id=args.group_id, **behavior["spec_kwargs"],
        )

        # Col 0: existing lanes
        ax = axes[row, 0]
        ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for li, cl in enumerate(traj_result.existing_lanes):
            ax.plot(cl[:, 0], cl[:, 1], color=LANE_COLORS[li % len(LANE_COLORS)],
                    linewidth=2.5, alpha=0.9)
        ax.set_ylabel(label, fontsize=10, fontweight="bold", labelpad=8)
        ax.set_xlabel("x (norm)")

        # Col 1: trajectory anchor (fixed)
        ax = axes[row, 1]
        ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.45)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for cl in traj_result.existing_lanes:
            ax.plot(cl[:, 0], cl[:, 1], color="#888888", linewidth=2, alpha=0.5)
        ax.plot(traj_anchor[:, 0], traj_anchor[:, 1], color="#333333", linewidth=3,
                label="trajectory anchor", zorder=10)
        ax.scatter(traj_anchor[0, 0], traj_anchor[0, 1], c="green", s=80, zorder=11)
        ax.scatter(traj_anchor[-1, 0], traj_anchor[-1, 1], c="#333333", s=80, zorder=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlabel("x (norm)")

        # Col 2: diffusion output under this behavior
        ax = axes[row, 2]
        ax.imshow(frame_rgb, extent=[0, 1, 1, 0], alpha=0.45)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
        for cl in traj_result.existing_lanes:
            ax.plot(cl[:, 0], cl[:, 1], color="#888888", linewidth=2, alpha=0.5)

        try:
            result = generator.generate_with_trajectory_context(
                trajectory_anchor=traj_anchor,
                neighbor_centerline=neighbor,
                spec=spec,
                n_candidates=args.n_candidates,
            )
            gen = result.best
            for cand in result.candidates:
                ax.plot(cand[:, 0], cand[:, 1], color=color, linewidth=1,
                        alpha=0.2, linestyle="--", zorder=8)
            ax.plot(gen[:, 0], gen[:, 1], color=color, linewidth=4,
                    label=f"{label} (best)", zorder=10)
            ax.scatter(gen[0, 0], gen[0, 1], c="green", s=80, zorder=11)
            ax.scatter(gen[-1, 0], gen[-1, 1], c=color, s=80, zorder=11)
            logger.info(f"[{label}] embedding norm: {np.linalg.norm(result.target_embedding):.3f}")
        except Exception as e:
            logger.warning(f"Generation failed for {label}: {e}")
            ax.plot(traj_anchor[:, 0], traj_anchor[:, 1], color=color, linewidth=3,
                    label=f"{label} (fallback)", zorder=10)

        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlabel("x (norm)")

    plt.tight_layout()
    out_path = output_dir / f"{args.camera}_g{args.group_id}_behavior_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved to {out_path}")

    _log_embedding_difference(generator, args)


def main():
    parser = argparse.ArgumentParser(description="Trajectory-grounded lane generation figures")
    parser.add_argument("--mode", choices=["lanes", "compare"], required=True,
                        help="lanes: per-spec variants for a group; compare: same position, different behaviors")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--config", default="configs/lane_contrastive.yaml")
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--output-dir", default="results/generation/trajectory")
    parser.add_argument("--k", type=int, default=32, help="Waypoints per lane")
    parser.add_argument("--n-candidates", type=int, default=5,
                        help="Number of diffusion candidates")

    # lanes-mode options
    parser.add_argument("--specs", nargs="+",
                        default=["rightmost", "leftmost", "merge"],
                        choices=["rightmost", "leftmost", "merge"],
                        help="(lanes mode) which spec(s) to generate")
    parser.add_argument("--use-diffusion", action="store_true",
                        help="(lanes mode) trajectory placement + diffusion shape")

    # checkpoint options (required for compare, optional for lanes with --use-diffusion)
    parser.add_argument("--checkpoint", default=None,
                        help="Encoder checkpoint (.pt)")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="Diffusion model checkpoint (.pt)")
    parser.add_argument("--relational-checkpoint", default=None,
                        help="Relational diffusion checkpoint (.pt); enables scene-aware generation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.mode == "compare":
        if not args.checkpoint or not args.diffusion_checkpoint:
            parser.error("--mode compare requires --checkpoint and --diffusion-checkpoint")
        _run_compare_mode(args, config)
    else:
        _run_lanes_mode(args, config)


if __name__ == "__main__":
    main()
