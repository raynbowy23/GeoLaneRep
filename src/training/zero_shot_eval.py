"""Zero-shot lane assignment evaluation for contrastive lane embeddings.

Given a trained LaneEncoder:
1. Encode all training-camera lanes (with geometry) -> reference bank
2. On held-out camera: encode lanes using trajectory-only (geometry dropped)
3. Match via cosine similarity
4. Compare to geometric baseline (LaneAssigner)
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.lane_dataset import LaneDataset, LaneSample, collate_fn
from src.models.lane_encoder import LaneEncoder

logger = logging.getLogger(__name__)


def load_trained_encoder(checkpoint_path: str, device: torch.device) -> Tuple[LaneEncoder, dict]:
    """Load a trained LaneEncoder from checkpoint.

    Supports both standalone contrastive checkpoints and joint training
    checkpoints (extracts lane_encoder_state_dict automatically).

    Returns:
        (model, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model_cfg = config.get("model", {})

    # Joint encoder trains without cross-lane attention
    is_joint = "lane_encoder_state_dict" in ckpt
    use_cross_lane = False if is_joint else model_cfg.get("use_cross_lane_attention", False)

    model = LaneEncoder(
        polyline_k=model_cfg.get("polyline_k", 16),
        d_model=model_cfg.get("polyline_encoder_dim", 64),
        embed_dim=model_cfg.get("embed_dim", 128),
        proj_dim=model_cfg.get("proj_dim", 64),
        polyline_mode=model_cfg.get("polyline_encoder", "transformer"),
        polyline_layers=model_cfg.get("polyline_encoder_layers", 2),
        polyline_heads=model_cfg.get("polyline_encoder_heads", 4),
        stats_dim=model_cfg.get("stats_dim", 9),
        geometry_dropout=0.0,  # no dropout at eval
        dropout=0.0,
        use_cross_lane_attention=use_cross_lane,
        cross_lane_heads=model_cfg.get("cross_lane_heads", 4),
        rel_feat_dim=model_cfg.get("rel_feat_dim", 3),
    ).to(device)

    if is_joint:
        model.load_state_dict(ckpt["lane_encoder_state_dict"])
        logger.info("Loaded LaneEncoder from joint checkpoint (lane_encoder_state_dict)")
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model, config


@torch.no_grad()
def encode_lanes(
    model: LaneEncoder,
    loader: DataLoader,
    device: torch.device,
    drop_geometry: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Encode all lanes in a dataloader.

    Args:
        model: Trained LaneEncoder.
        loader: DataLoader yielding batches.
        device: torch device.
        drop_geometry: If True, zero out geometry (trajectory-only encoding).

    Returns:
        (projections, roles, lane_keys) — all concatenated across batches.
    """
    all_proj = []
    all_roles = []
    all_keys = []

    for batch in loader:
        stats_input = torch.cat([batch["traj_stats"], batch["roles"]], dim=-1).to(device)
        if model.use_cross_lane_attention and "group_ids" in batch:
            output = model.forward_grouped(
                geometry=batch["geometry"].to(device),
                traj_polylines=batch["traj_polylines"].to(device),
                traj_mask=batch["traj_mask"].to(device),
                traj_stats=stats_input,
                group_ids=batch["group_ids"].to(device),
                drop_geometry=drop_geometry,
            )
        else:
            output = model(
                geometry=batch["geometry"].to(device),
                traj_polylines=batch["traj_polylines"].to(device),
                traj_mask=batch["traj_mask"].to(device),
                traj_stats=stats_input,
                drop_geometry=drop_geometry,
            )
        all_proj.append(output["projection"].cpu())
        all_roles.append(batch["roles"])
        all_keys.extend(batch["lane_keys"])

    projections = torch.cat(all_proj, dim=0)
    roles = torch.cat(all_roles, dim=0)
    return projections, roles, all_keys


def evaluate_zero_shot(
    model: LaneEncoder,
    dataset: LaneDataset,
    held_out_camera: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Run zero-shot evaluation for a single held-out camera.

    Steps:
    1. Encode training lanes (with geometry) -> reference bank
    2. Encode held-out lanes (trajectory-only) -> query set
    3. Match queries to references via cosine similarity
    4. Evaluate: lateral rank difference, edge flag accuracy, role similarity

    Returns:
        Dict of evaluation metrics.
    """
    held_out_indices = dataset.get_camera_indices(held_out_camera)
    if not held_out_indices:
        logger.warning(f"No samples for {held_out_camera}")
        return {}

    held_out_set = set(held_out_indices)
    train_indices = [i for i in range(len(dataset)) if i not in held_out_set]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        Subset(dataset, held_out_indices),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Encode
    ref_proj, ref_roles, ref_keys = encode_lanes(
        model, train_loader, device, drop_geometry=False
    )
    query_proj, query_roles, query_keys = encode_lanes(
        model, eval_loader, device, drop_geometry=True
    )

    # Match: cosine similarity (projections already L2-normalized)
    sim_matrix = torch.mm(query_proj, ref_proj.t())  # (N_query, N_ref)
    best_match_idx = sim_matrix.argmax(dim=1)
    best_match_sim = sim_matrix.gather(1, best_match_idx.unsqueeze(1)).squeeze(1)

    matched_ref_roles = ref_roles[best_match_idx]

    # Metrics
    # 1. Lateral rank difference
    lat_rank_diff = (query_roles[:, 0] - matched_ref_roles[:, 0]).abs()

    # 2. Edge flag accuracy (leftmost + rightmost)
    leftmost_match = (query_roles[:, 1] == matched_ref_roles[:, 1]).float()
    rightmost_match = (query_roles[:, 2] == matched_ref_roles[:, 2]).float()
    edge_accuracy = (leftmost_match + rightmost_match) / 2.0

    # 3. Successor flag accuracy
    successor_match = (query_roles[:, 3] == matched_ref_roles[:, 3]).float()

    # 4. Overall role similarity (cosine between role vectors)
    role_sim = torch.nn.functional.cosine_similarity(query_roles, matched_ref_roles, dim=1)

    # Per-query results
    results_per_lane = []
    for i in range(len(query_keys)):
        results_per_lane.append({
            "query_lane": query_keys[i],
            "matched_ref": ref_keys[best_match_idx[i].item()],
            "cosine_sim": best_match_sim[i].item(),
            "lat_rank_diff": lat_rank_diff[i].item(),
            "query_lat_rank": query_roles[i, 0].item(),
            "matched_lat_rank": matched_ref_roles[i, 0].item(),
            "query_is_leftmost": bool(query_roles[i, 1].item()),
            "query_is_rightmost": bool(query_roles[i, 2].item()),
            "matched_is_leftmost": bool(matched_ref_roles[i, 1].item()),
            "matched_is_rightmost": bool(matched_ref_roles[i, 2].item()),
            "role_sim": role_sim[i].item(),
        })

    return {
        "held_out_camera": held_out_camera,
        "n_query_lanes": len(query_keys),
        "n_ref_lanes": len(ref_keys),
        "mean_match_sim": best_match_sim.mean().item(),
        "mean_lat_rank_diff": lat_rank_diff.mean().item(),
        "edge_flag_accuracy": edge_accuracy.mean().item(),
        "successor_flag_accuracy": successor_match.mean().item(),
        "mean_role_similarity": role_sim.mean().item(),
        "per_lane": results_per_lane,
    }


def leave_one_camera_out_eval(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> Dict[str, dict]:
    """Run leave-one-camera-out zero-shot evaluation.

    For each camera, hold it out, encode training lanes with geometry,
    encode held-out lanes trajectory-only, and evaluate matching quality.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        config_path: Optional config override. If None, uses config from checkpoint.

    Returns:
        Dict mapping camera_name -> evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_config = load_trained_encoder(checkpoint_path, device)

    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = ckpt_config

    model_cfg = config.get("model", {})
    train_cfg = config.get("contrastive_training", {})

    dataset = LaneDataset(
        config=config,
        polyline_k=model_cfg.get("polyline_k", 16),
        max_traj_per_lane=train_cfg.get("max_traj_per_lane", 50),
        role_similarity_threshold=train_cfg.get("role_similarity_threshold", 0.8),
    )

    all_results = {}
    for camera in dataset.cameras:
        logger.info(f"Evaluating held-out camera: {camera}")
        metrics = evaluate_zero_shot(
            model, dataset, camera, device,
            batch_size=train_cfg.get("batch_size", 32),
        )
        if metrics:
            all_results[camera] = metrics
            logger.info(
                f"  {camera}: match_sim={metrics['mean_match_sim']:.3f} "
                f"lat_diff={metrics['mean_lat_rank_diff']:.3f} "
                f"edge_acc={metrics['edge_flag_accuracy']:.3f} "
                f"role_sim={metrics['mean_role_similarity']:.3f}"
            )

    # Aggregate
    if all_results:
        agg = defaultdict(list)
        for cam_metrics in all_results.values():
            for k in ["mean_match_sim", "mean_lat_rank_diff", "edge_flag_accuracy",
                       "successor_flag_accuracy", "mean_role_similarity"]:
                agg[k].append(cam_metrics[k])

        logger.info("=== Leave-One-Camera-Out Aggregate ===")
        for k, vals in agg.items():
            logger.info(f"  {k}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    return all_results
