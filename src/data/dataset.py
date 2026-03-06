"""TrackletDataset: loads trajectories, builds tracklet graphs, returns PyG Data.

Imports DataManager and OSMConnection from graph_geolane (v1) for data loading.
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import yaml
from shapely.geometry import LineString, MultiPoint, Point
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.data.tracklet_graph import build_tracklet_graph, split_trajectories_by_direction
from src.data.density_contours import assign_vehicles_to_lane_groups

logger = logging.getLogger(__name__)


def _merge_consecutive_sumo_lanes(
    gps_lane_geom: Dict[str, np.ndarray],
    lane_headings: Dict[str, float],
    sumo_net_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Merge consecutive SUMO lane segments into single logical lanes.

    SUMO splits a physical road lane at junction nodes, producing IDs like
    ``176724154#0_0`` → ``176724154#1_0`` for what is actually one lane.
    This function parses ``<connection dir="s">`` elements to find
    straight-through 1:1 links and chains them into merged geometries.

    Only merges when the connection is strictly 1:1 (one straight successor,
    one straight predecessor).  Diverges (1:many) and merges (many:1) are
    topology changes and are left alone.

    Args:
        gps_lane_geom: lane_id → (N, 2+) GPS coordinates.
        lane_headings: lane_id → heading in radians.
        sumo_net_path: path to SUMO ``osm.net.xml``.

    Returns:
        Tuple of (merged_gps_lane_geom, merged_lane_headings).
    """
    import xml.etree.ElementTree as ET

    net_path = Path(sumo_net_path)
    if not net_path.exists():
        logger.warning(f"SUMO net file not found: {sumo_net_path}, skipping lane merge")
        return gps_lane_geom, lane_headings

    existing_lanes = set(gps_lane_geom.keys())
    if not existing_lanes:
        return gps_lane_geom, lane_headings

    # --- 1. Parse <connection dir="s"> elements ---
    # Build raw successor / predecessor multi-maps first, then filter to 1:1.
    successor_multi: Dict[str, List[str]] = {}   # from_lane → [to_lanes]
    predecessor_multi: Dict[str, List[str]] = {}  # to_lane → [from_lanes]

    try:
        for _, elem in ET.iterparse(str(net_path), events=("end",)):
            if elem.tag != "connection":
                continue
            if elem.get("dir") != "s":
                elem.clear()
                continue
            from_edge = elem.get("from", "")
            to_edge = elem.get("to", "")
            from_lane_idx = elem.get("fromLane", "")
            to_lane_idx = elem.get("toLane", "")
            elem.clear()

            # Skip internal junction edges (start with ":")
            if from_edge.startswith(":") or to_edge.startswith(":"):
                continue

            from_lane = f"{from_edge}_{from_lane_idx}"
            to_lane = f"{to_edge}_{to_lane_idx}"

            # Only consider lanes that exist in our geometry dict
            if from_lane not in existing_lanes or to_lane not in existing_lanes:
                continue

            successor_multi.setdefault(from_lane, []).append(to_lane)
            predecessor_multi.setdefault(to_lane, []).append(from_lane)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse SUMO net.xml: {e}, skipping lane merge")
        return gps_lane_geom, lane_headings

    # --- 2. Filter to strict 1:1 connections ---
    successor: Dict[str, str] = {}
    predecessor: Dict[str, str] = {}
    for from_lane, to_lanes in successor_multi.items():
        if len(to_lanes) != 1:
            continue
        to_lane = to_lanes[0]
        if len(predecessor_multi.get(to_lane, [])) != 1:
            continue
        successor[from_lane] = to_lane
        predecessor[to_lane] = from_lane

    if not successor:
        logger.info("No consecutive SUMO lanes to merge")
        return gps_lane_geom, lane_headings

    # --- 3. Build chains from roots (lanes with no predecessor in the 1:1 map) ---
    chains: List[List[str]] = []
    visited = set()
    for lane in existing_lanes:
        if lane in visited:
            continue
        if lane in predecessor:
            continue  # not a root — skip, will be visited from its root
        # Walk forward
        chain = [lane]
        visited.add(lane)
        cur = lane
        while cur in successor:
            nxt = successor[cur]
            if nxt in visited:
                break  # cycle guard
            chain.append(nxt)
            visited.add(nxt)
            cur = nxt
        if len(chain) > 1:
            chains.append(chain)

    if not chains:
        logger.info("No consecutive SUMO lanes to merge")
        return gps_lane_geom, lane_headings

    # --- 4. Merge geometries ---
    merged_geom: Dict[str, np.ndarray] = {}
    merged_headings: Dict[str, float] = {}
    consumed = set()  # lanes absorbed into a chain

    for chain in chains:
        merged_key = chain[0]  # use first segment ID as canonical key
        segments = []
        for lid in chain:
            pts = np.array(gps_lane_geom[lid], dtype=np.float64)
            if pts.ndim != 2 or len(pts) < 2:
                continue
            segments.append(pts)
            consumed.add(lid)

        if len(segments) < 2:
            continue

        # Concatenate, deduplicating junction-overlap points.
        # The last point of segment A ≈ first point of segment B at the junction.
        all_pts = [segments[0]]
        for seg in segments[1:]:
            prev_end = all_pts[-1][-1, :2]
            seg_start = seg[0, :2]
            dist = np.linalg.norm(prev_end - seg_start)
            if dist < 1.0:  # ~1 meter overlap tolerance
                all_pts.append(seg[1:])  # skip duplicate first point
            else:
                all_pts.append(seg)

        concat = np.vstack(all_pts)
        merged_geom[merged_key] = concat

        # Recompute heading from full merged geometry
        dx = concat[-1, 0] - concat[0, 0]
        dy = concat[-1, 1] - concat[0, 1]
        merged_headings[merged_key] = float(np.arctan2(dy, dx))

    # --- 5. Build final dictionaries: merged chains + untouched singletons ---
    final_geom = {}
    final_headings = {}

    # Add merged chains
    for key, pts in merged_geom.items():
        final_geom[key] = pts
        final_headings[key] = merged_headings[key]

    # Add lanes not consumed by any chain
    for lid in existing_lanes:
        if lid not in consumed:
            final_geom[lid] = gps_lane_geom[lid]
            if lid in lane_headings:
                final_headings[lid] = lane_headings[lid]

    n_chains = len(chains)
    n_consumed = len(consumed)
    n_before = len(existing_lanes)
    n_after = len(final_geom)
    logger.info(
        f"Merged {n_before} SUMO lanes into {n_after} logical lanes "
        f"({n_chains} chains, {n_consumed} segments merged)"
    )

    return final_geom, final_headings


class TrackletDataset(Dataset):
    """Per-camera, per-time-window tracklet graph dataset.

    Each sample is a PyG Data graph where:
      - Nodes = tracklets (short vehicle trajectory segments).
      - Edges = spatial/temporal proximity.
      - Targets = next-displacement (always), SUMO lane labels (supervised only).

    Long trajectories are chunked into time windows (default 60s) to keep
    graph size tractable for GPU training.
    """

    def __init__(
        self,
        processed_data: Dict,
        config: dict,
        split: str = "train",
    ):
        """
        Args:
            processed_data: Dict[camera_loc -> ProcessedData] from v1 DataManager.
            config: full YAML config dict.
            split: "train" | "val" | "test".
        """
        self.config = config
        self.split = split
        self.samples: List[Data] = []

        data_cfg = config.get("data", {})

        # Try loading cached graphs
        cache_path = self._cache_path(config, split)
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached graphs: {cache_path}")
            self.samples = torch.load(cache_path, weights_only=False)
            logger.info(f"TrackletDataset [{split}]: {len(self.samples)} samples (from cache)")
            return
        train_ratio = data_cfg.get("train_ratio", 0.70)
        val_ratio = data_cfg.get("val_ratio", 0.15)
        holdout_cameras = set(data_cfg.get("holdout_cameras", []))
        val_cameras = set(data_cfg.get("val_cameras", []))
        split_by_location = data_cfg.get("split_by_location", False)
        use_sumo = data_cfg.get("use_sumo_targets", False)
        use_annotation = data_cfg.get("use_annotation_gt", False)
        annotation_dir = Path(data_cfg.get("annotation_dir", "../dataset/preprocess"))
        window_sec = data_cfg.get("time_window", 60.0)
        window_stride = data_cfg.get("time_window_stride", 30.0)

        # Load annotation JSON per camera (if enabled)
        annotation_data = {}
        if use_annotation:
            from src.data.annotation_loader import load_annotation_json
            for camera_loc in processed_data:
                annot_path = annotation_dir / camera_loc / "annotation.json"
                if annot_path.exists():
                    try:
                        annotation_data[camera_loc] = load_annotation_json(annot_path)
                        n_groups = len(annotation_data[camera_loc]["lane_groups"])
                        logger.info(f"{camera_loc}: loaded annotation ({n_groups} groups)")
                    except Exception as e:
                        logger.warning(f"{camera_loc}: failed to load annotation: {e}")
                else:
                    logger.warning(f"{camera_loc}: no annotation.json at {annot_path}")

        split_by_direction = data_cfg.get("split_by_direction", False)
        use_lane_groups = data_cfg.get("use_lane_groups", False)

        # Load cached lane coordinate labels (from LaneCoordNet preprocessing)
        lane_coord_cache = {}
        use_lane_coord = data_cfg.get("use_lane_coord_labels", True)
        preprocess_dir = Path(data_cfg.get("preprocess_path", "./results/preprocess"))
        if not use_lane_coord:
            logger.info("Skipping LaneCoordNet labels (use_lane_coord_labels: false)")
        else:
            for camera_loc in processed_data:
                if split_by_direction:
                    # Per-direction label files: lane_coords_dir{gid}.npz
                    dir_labels = {}
                    for p in sorted(preprocess_dir.joinpath(camera_loc).glob("lane_coords_dir*.npz")):
                        gid = int(p.stem.replace("lane_coords_dir", ""))
                        loaded = np.load(p, allow_pickle=True)
                        if "vehicle_labels" not in loaded:
                            logger.warning(f"{camera_loc}: {p.name} missing 'vehicle_labels' key, skipping")
                            continue
                        dir_labels[gid] = loaded["vehicle_labels"].item()
                    if dir_labels:
                        lane_coord_cache[camera_loc] = dir_labels
                        total = sum(len(v) for v in dir_labels.values())
                        logger.info(f"{camera_loc}: loaded {total} per-direction lane labels from Phi ({len(dir_labels)} groups)")
                    else:
                        # Fall back to single-file labels
                        cache_path = preprocess_dir / camera_loc / "lane_coords.npz"
                        if cache_path.exists():
                            loaded = np.load(cache_path, allow_pickle=True)
                            if "vehicle_labels" not in loaded:
                                logger.warning(f"{camera_loc}: lane_coords.npz missing 'vehicle_labels' key, skipping")
                            else:
                                lane_coord_cache[camera_loc] = loaded["vehicle_labels"].item()
                                logger.info(f"{camera_loc}: loaded {len(lane_coord_cache[camera_loc])} vehicle lane labels from Phi (single file)")
                else:
                    cache_path = preprocess_dir / camera_loc / "lane_coords.npz"
                    if cache_path.exists():
                        loaded = np.load(cache_path, allow_pickle=True)
                        if "vehicle_labels" not in loaded:
                            logger.warning(f"{camera_loc}: lane_coords.npz missing 'vehicle_labels' key, skipping")
                        else:
                            lane_coord_cache[camera_loc] = loaded["vehicle_labels"].item()
                            logger.info(f"{camera_loc}: loaded {len(lane_coord_cache[camera_loc])} vehicle lane labels from Phi")

        test_ratio = data_cfg.get("test_ratio", 0.15)

        for camera_loc, pdata in processed_data.items():
            # --- Camera-level routing ---
            # Holdout cameras: test-only (legacy; empty list in new config)
            if camera_loc in holdout_cameras:
                if split != "test":
                    continue
            elif split_by_location and split == "val":
                # Val split: only val_cameras contribute
                if camera_loc not in val_cameras:
                    continue
            elif split_by_location and split == "train":
                # Train split: val_cameras do NOT contribute to train
                if camera_loc in val_cameras:
                    continue
            # Test split: ALL cameras contribute (temporal holdout)

            traj_full = pdata.trajectories  # FULL trajectory (all time)
            frame = pdata.frame
            if traj_full is None or len(traj_full) == 0:
                continue

            # Time-based split — filter AFTER saving full traj for lane groups
            traj = traj_full
            if "time" in traj.columns:
                t_min = traj["time"].min()
                t_max = traj["time"].max()
                t_range = t_max - t_min
                if split == "test":
                    # Test: last test_ratio of time from ALL cameras
                    t_test_start = t_min + t_range * (1.0 - test_ratio)
                    traj = traj.filter(pl.col("time") >= t_test_start)
                elif split == "train":
                    # Train: first (1 - test_ratio) of time
                    t_train_end = t_min + t_range * (1.0 - test_ratio)
                    traj = traj.filter(pl.col("time") < t_train_end)
                elif split == "val":
                    # Val (location-based): first (1 - test_ratio) of time
                    # (excludes test window so no temporal leak)
                    t_val_end = t_min + t_range * (1.0 - test_ratio)
                    traj = traj.filter(pl.col("time") < t_val_end)

            if len(traj) < 10:
                continue

            H, W = frame.shape[:2] if frame is not None else (720, 1280)

            fixed_lane_width = data_cfg.get("fixed_lane_width", data_cfg.get("fixed_lane_width_m", 3.5))
            image_wh = (W, H)  # use actual frame dimensions for normalization

            # Extract homography only if SUMO targets are used (not needed for annotation mode)
            metadata = getattr(pdata, "metadata", {}) or {}
            pixel_hom = metadata.get("pixel_hom", None) if use_sumo else None
            cam_ref_gps = None
            if pixel_hom is not None:
                traj_pts = traj_full.select(["x", "y"]).to_numpy().astype(np.float64)
                from src.utils.homography_scale import pixels_to_local_meters, gps_to_local_meters
                _, cam_ref_gps = pixels_to_local_meters(pixel_hom, traj_pts)

            # Chunk into time windows for tractable graph sizes
            # Pass traj_full for lane group detection (road geometry is split-independent)
            cam_lane_labels = lane_coord_cache.get(camera_loc)
            graphs = self._build_windowed_graphs(
                traj, (H, W), config, window_sec, window_stride,
                cam_lane_labels, split_by_direction, use_lane_groups,
                camera_loc=camera_loc, pixel_hom=pixel_hom,
                ref_gps=cam_ref_gps,
                traj_full=traj_full,
                image_wh=image_wh,
            )

            # Pre-compute ICP alignment ONCE per direction using ALL time
            # windows' tracklets. This gives a stable, consistent transform
            # instead of each window getting a slightly different alignment.
            sumo_alignment_cache = {}  # heading_float -> np.ndarray (3,3)
            if use_sumo and data_cfg.get("align_sumo_to_tracklets", True):
                from collections import defaultdict as _defaultdict
                from src.utils.map_alignment import MapAlignment
                dir_centroids = _defaultdict(list)
                for g in graphs:
                    h = getattr(g, "lane_group_heading", None)
                    if h is not None:
                        dir_centroids[float(h)].append(g.centroids.numpy())
                gps_lane_geom_raw = (getattr(pdata, "metadata", {}) or {}).get("gps_lane_geom", {})
                lane_headings_raw = (getattr(pdata, "metadata", {}) or {}).get("lane_headings", {})
                for heading_val, cent_list in dir_centroids.items():
                    all_cents = np.concatenate(cent_list, axis=0)
                    if len(all_cents) < 5 or cam_ref_gps is None:
                        continue
                    # Same heading filter as _prepare_sumo_lanes_meters
                    filt_geom = {}
                    for lid, pts in gps_lane_geom_raw.items():
                        lh = lane_headings_raw.get(lid)
                        if lh is not None and np.cos(heading_val - lh) < 0.5:
                            continue
                        filt_geom[lid] = pts
                    if not filt_geom:
                        continue
                    # GPS -> meters
                    sumo_m_pts = []
                    for lid in sorted(filt_geom.keys()):
                        pts = np.array(filt_geom[lid], dtype=np.float64)
                        if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
                            continue
                        sumo_m_pts.append(gps_to_local_meters(pts[:, :2], cam_ref_gps))
                    if not sumo_m_pts:
                        continue
                    all_sumo = np.concatenate(sumo_m_pts, axis=0)
                    if len(all_sumo) >= 3:
                        aligner = MapAlignment()
                        aligner.estimate(all_sumo, all_cents, max_corresp_dist=50.0)
                        sumo_alignment_cache[heading_val] = aligner.T.copy()

            # Pre-compute annotation matching for all lane groups in this camera
            annot_match = {}  # graph index -> list of matched annotation group_ids
            if use_annotation and camera_loc in annotation_data:
                from src.data.annotation_loader import (
                    match_annotation_group,
                    annotation_lanes_to_normalized,
                    get_annotation_relationships,
                )
                annot = annotation_data[camera_loc]

                # Match per unique lane group (not per window) to avoid
                # consuming annotation groups across time windows
                lg_heading_map = {}  # lane_group_id -> heading
                lg_graph_indices = {}  # lane_group_id -> [graph indices]
                for gi, graph in enumerate(graphs):
                    gh = getattr(graph, "lane_group_heading", None)
                    lg_id = getattr(graph, "lane_group_id", None)
                    if gh is not None and lg_id is not None:
                        lg_heading_map[lg_id] = gh
                        lg_graph_indices.setdefault(lg_id, []).append(gi)

                # Step 1: Greedy 1:1 matching across unique lane groups
                lg_candidates = []
                for lg_id, gh in lg_heading_map.items():
                    best_cos = -2.0
                    for lg in annot["lane_groups"]:
                        heading_rad = math.radians(lg.get("heading_deg", 0.0))
                        diff = gh - heading_rad
                        diff_flip = gh - (heading_rad + math.pi)
                        cs = max(math.cos(diff), math.cos(diff_flip))
                        if cs > best_cos:
                            best_cos = cs
                    lg_candidates.append((lg_id, gh, best_cos))

                lg_candidates.sort(key=lambda x: -x[2])
                claimed_groups = set()
                lg_annot_groups = {}  # lane_group_id -> [group_ids]
                for lg_id, gh, _ in lg_candidates:
                    matched_gid = match_annotation_group(gh, annot, excluded_group_ids=claimed_groups)
                    if matched_gid is not None:
                        claimed_groups.add(matched_gid)
                        lg_annot_groups[lg_id] = [matched_gid]
                        logger.info(
                            f"  {camera_loc} lg={lg_id} heading={math.degrees(gh):.1f}deg "
                            f"-> annot group={matched_gid} ({len(lg_graph_indices[lg_id])} windows)"
                        )

                # Step 2: Assign remaining unmatched annotation groups to closest lane group
                all_annot_gids = {lg["group_id"] for lg in annot["lane_groups"]}
                unclaimed = all_annot_gids - claimed_groups
                # Proximity threshold: max distance (normalized) between
                # annotation lane waypoints and nearest tracklet centroid
                proximity_thresh = data_cfg.get("unclaimed_annot_proximity", 0.15)

                for ugid in unclaimed:
                    # Get annotation lane waypoints in normalized [0,1]
                    unclaimed_lanes = annotation_lanes_to_normalized(
                        annot, ugid, image_wh=image_wh)
                    if not unclaimed_lanes:
                        continue
                    unclaimed_pts = np.concatenate(list(unclaimed_lanes.values()), axis=0)

                    # Find closest detected lane group with tracklets nearby
                    best_lg = None
                    best_min_dist = float("inf")
                    for lg_id in lg_heading_map:
                        # Get tracklet centroids for this lane group (use first window)
                        gi0 = lg_graph_indices[lg_id][0]
                        centroids = graphs[gi0].centroids.numpy()  # (N, 2) in normalized [0,1] before sd rotation
                        # Compute min distance from any annotation point to any tracklet
                        from scipy.spatial.distance import cdist
                        dists = cdist(unclaimed_pts, centroids)
                        min_dist = float(dists.min()) if dists.size > 0 else float("inf")
                        if min_dist < best_min_dist:
                            best_min_dist = min_dist
                            best_lg = lg_id

                    if best_lg is not None and best_min_dist < proximity_thresh:
                        lg_annot_groups.setdefault(best_lg, []).append(ugid)
                        logger.info(
                            f"  {camera_loc} annot group={ugid} (unclaimed) "
                            f"-> assigned to lg={best_lg} (min_dist={best_min_dist:.3f})"
                        )
                    else:
                        logger.info(
                            f"  {camera_loc} annot group={ugid} (unclaimed) "
                            f"-> skipped (min_dist={best_min_dist:.3f} > {proximity_thresh})"
                        )

                # Populate annot_match: graph index -> list of annotation group_ids
                for lg_id, gids in lg_annot_groups.items():
                    for gi in lg_graph_indices[lg_id]:
                        annot_match[gi] = gids

            for gi, graph in enumerate(graphs):
                graph.camera_loc = camera_loc

                # Fixed lane width in meters (constant, no pixel conversion needed)
                graph.lane_width_m = torch.tensor(fixed_lane_width, dtype=torch.float32)

                if gi in annot_match:
                    # --- Annotation GT branch ---
                    matched_gids = annot_match[gi]  # list of annotation group_ids
                    # Merge lanes from all matched annotation groups
                    annot_lanes = {}
                    primary_gid = matched_gids[0]
                    for gid in matched_gids:
                        lanes = annotation_lanes_to_normalized(
                            annot, gid, image_wh=image_wh)
                        annot_lanes.update(lanes)

                    if annot_lanes:
                        graph.gt_labels = torch.full((graph.num_nodes,), -1, dtype=torch.long)

                        # Build target polylines from annotation lanes
                        target_poly, target_mask = self._extract_target_polylines(
                            pdata, config, (H, W), graph=graph,
                            prepared_lanes=annot_lanes)

                        # Build GT lanelet graph with explicit annotation relationships
                        self._build_gt_lanelet_graph_from_annotation(
                            graph, annot, primary_gid, annot_lanes, config)
                        gt_n = graph.gt_lanelet_positions.shape[0] if graph.gt_lanelet_positions is not None else 0
                        logger.info(f"    {camera_loc} lg={getattr(graph, 'lane_group_id', '?')}: GT built {gt_n} wps from {len(annot_lanes)} lanes (groups={matched_gids})")
                    else:
                        graph.gt_labels = torch.full((graph.num_nodes,), -1, dtype=torch.long)
                        target_poly, target_mask = None, None
                        self._set_empty_lanelet_gt(graph)

                elif use_annotation:
                    # Lane group didn't match any annotation group
                    graph.gt_labels = torch.full((graph.num_nodes,), -1, dtype=torch.long)
                    target_poly, target_mask = None, None
                    self._set_empty_lanelet_gt(graph)

                elif use_sumo:
                    # --- SUMO GT branch ---
                    # Look up precomputed alignment for this direction
                    gh = getattr(graph, "lane_group_heading", None)
                    precomputed_T = sumo_alignment_cache.get(float(gh)) if gh is not None else None

                    # Prepare SUMO lanes ONCE (GPS->meters, clip, evidence filter)
                    # and share across all three consumers
                    prepared_lanes = self._prepare_sumo_lanes_meters(
                        graph, pdata, config, alignment_transform=precomputed_T)

                    gt_labels = self._match_to_sumo(graph, pdata, config,
                                                    prepared_lanes=prepared_lanes)
                    graph.gt_labels = gt_labels
                    # Extract GT polyline targets for polyline decoder loss
                    target_poly, target_mask = self._extract_target_polylines(
                        pdata, config, (H, W), graph=graph,
                        prepared_lanes=prepared_lanes,
                    )
                else:
                    graph.gt_labels = torch.full((graph.num_nodes,), -1, dtype=torch.long)
                    target_poly, target_mask = None, None

                # Always set target_polylines so PyG Batch collation sees uniform attributes
                model_cfg = config.get("model", {})
                data_cfg = config.get("data", {})
                if data_cfg.get("use_lane_groups", False):
                    n_slots = model_cfg.get("num_slots_per_group", model_cfg.get("num_slots", 8))
                else:
                    n_slots = model_cfg.get("num_slots", 8)
                K = model_cfg.get("lane_shape_num_points", 10)
                if target_poly is not None:
                    graph.target_polylines = target_poly
                    graph.target_valid_mask = target_mask
                else:
                    graph.target_polylines = torch.zeros(n_slots, K, 2, dtype=torch.float32)
                    graph.target_valid_mask = torch.zeros(n_slots, dtype=torch.bool)

                # Build GT lanelet graph (v3: waypoint-level supervision)
                # (annotation branch already handled above)
                if use_sumo and not (use_annotation and camera_loc in annotation_data):
                    self._build_gt_lanelet_graph(graph, pdata, config,
                                                 prepared_lanes=prepared_lanes)
                elif not use_sumo and not (use_annotation and camera_loc in annotation_data):
                    self._set_empty_lanelet_gt(graph)

                # Skip graphs with no GT in annotation mode — no supervision signal
                if use_annotation and hasattr(graph, 'gt_lanelet_positions'):
                    if graph.gt_lanelet_positions is None or graph.gt_lanelet_positions.shape[0] == 0:
                        continue

                # Rotate to (s,d) frame for rotation invariance
                if hasattr(graph, 'lane_group_heading') and graph.lane_group_heading is not None:
                    graph = self._rotate_to_sd_frame(graph)

                self.samples.append(graph)

        logger.info(f"TrackletDataset [{split}]: {len(self.samples)} samples from {len(processed_data)} cameras.")

        # Save to cache for next run
        if cache_path and self.samples:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.samples, cache_path)
            logger.info(f"Saved graph cache: {cache_path}")

    @staticmethod
    def _cache_path(config: dict, split: str) -> Optional[Path]:
        """Compute cache file path from data config hash.

        Returns None if caching is disabled (data.cache_graphs: false).
        Cache key covers all config keys that affect graph construction.
        """
        data_cfg = config.get("data", {})
        if not data_cfg.get("cache_graphs", True):
            return None

        # Keys that determine graph structure
        cache_keys = {
            "tracklet_length", "tracklet_min_points", "tracklet_polyline_k", "max_neighbors_per_node",
            "edge_radius", "edge_temporal_max", "edge_tangent_threshold",
            "time_window", "time_window_stride",
            "train_ratio", "val_ratio", "test_ratio",
            "holdout_cameras", "split_by_direction", "direction_min_gap_deg",
            "use_sumo_targets", "pseudo_gt_min_gap_factor",
            "use_lane_coord_labels", "detection_period",
            "use_density_contours",
            "use_lane_groups",
            "fixed_lane_width_m",
            "lane_group_method",
            "hull_buffer_px",
            "dbscan_min_samples",
            "min_vehicles_per_group",
            "density_sigma_along",
            "density_sigma_across",
            "refine_contours",
            "refine_buffer_px",
            "lane_group_n_iters",
            "lane_group_hidden",
            "lane_group_k_max",
            "lane_group_edge_radius",
            "use_lateral_ordering",
            "lateral_min_overlap_frames",
            "scale_penalty_sigma",
            "community_resolution",
            "sumo_match_radius_m",
            "clip_sumo_lanes",
            "clip_sumo_margin_m",
            "min_lane_evidence",
            "merge_consecutive_sumo_lanes",
            "split_by_location", "val_cameras",
            "use_annotation_gt", "annotation_dir",
        }
        key_dict = {k: data_cfg.get(k) for k in sorted(cache_keys) if data_cfg.get(k) is not None}
        key_dict["_cache_version"] = 12  # v12: 1:1 annotation group matching with 180° flip
        key_dict["split"] = split
        # Include camera list so adding/removing cameras invalidates cache
        cam_list_path = data_cfg.get("camera_locations", "")
        if cam_list_path and Path(cam_list_path).exists():
            cams = sorted(l.strip() for l in Path(cam_list_path).read_text().splitlines() if l.strip())
            key_dict["_cameras"] = cams
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        h = hashlib.sha256(key_str.encode()).hexdigest()[:12]

        exp_cfg = config.get("experiment", {})
        save_dir = Path(exp_cfg.get("saving_path", "./results"))
        exp_name = exp_cfg.get("experiment_name", "lane_repr")
        return save_dir / exp_name / "graph_cache" / f"{split}_{h}.pt"

    def _build_windowed_graphs(
        self,
        traj: pl.DataFrame,
        frame_shape: Tuple[int, int],
        config: dict,
        window_sec: float,
        stride_sec: float,
        lane_coord_labels=None,
        split_by_direction: bool = False,
        use_lane_groups: bool = False,
        camera_loc: str = "",
        pixel_hom=None,
        ref_gps=None,
        traj_full: pl.DataFrame = None,
        image_wh: Optional[Tuple[int, int]] = None,
    ) -> List[Data]:
        """Split trajectory into overlapping time windows and build a graph per window.

        Routing priority:
        1. use_lane_groups — one graph per lane group per window (spatial + directional)
        2. split_by_direction — one graph per heading group per window (directional only)
        3. default — single graph per window
        """
        if use_lane_groups:
            return self._build_windowed_graphs_lane_groups(
                traj, frame_shape, config, window_sec, stride_sec, lane_coord_labels,
                camera_loc=camera_loc, pixel_hom=pixel_hom, ref_gps=ref_gps,
                traj_full=traj_full,
                image_wh=image_wh,
            )
        if split_by_direction:
            return self._build_windowed_graphs_split(
                traj, frame_shape, config, window_sec, stride_sec, lane_coord_labels,
                pixel_hom=pixel_hom, ref_gps=ref_gps,
                image_wh=image_wh,
            )
        return self._build_windowed_graphs_single(
            traj, frame_shape, config, window_sec, stride_sec, lane_coord_labels,
            pixel_hom=pixel_hom, ref_gps=ref_gps,
            image_wh=image_wh,
        )

    def _build_windowed_graphs_single(
        self,
        traj: pl.DataFrame,
        frame_shape: Tuple[int, int],
        config: dict,
        window_sec: float,
        stride_sec: float,
        lane_coord_labels: Optional[Dict[int, int]] = None,
        single_direction: bool = False,
        pixel_hom=None,
        ref_gps=None,
        skip_density_contours: bool = False,
        image_wh: Optional[Tuple[int, int]] = None,
    ) -> List[Data]:
        """Build windowed graphs for a single trajectory set (one direction or all)."""
        graphs = []

        if "time" not in traj.columns:
            graph = build_tracklet_graph(
                traj, frame_shape, config,
                lane_coord_labels=lane_coord_labels,
                single_direction=single_direction,
                pixel_hom=pixel_hom,
                ref_gps=ref_gps,
                skip_density_contours=skip_density_contours,
                image_wh=image_wh,
            )
            if graph is not None:
                graphs.append(graph)
            return graphs

        t_min = traj["time"].min()
        t_max = traj["time"].max()
        t_range = t_max - t_min

        if t_range <= window_sec:
            graph = build_tracklet_graph(
                traj, frame_shape, config,
                lane_coord_labels=lane_coord_labels,
                single_direction=single_direction,
                pixel_hom=pixel_hom,
                ref_gps=ref_gps,
                skip_density_contours=skip_density_contours,
                image_wh=image_wh,
            )
            if graph is not None:
                graphs.append(graph)
            return graphs

        t_start = t_min
        n_windows = max(1, int((t_max - t_min - window_sec) / stride_sec) + 1)
        w_idx = 0
        while t_start < t_max:
            t_end = t_start + window_sec
            window_traj = traj.filter(
                (pl.col("time") >= t_start) & (pl.col("time") < t_end)
            )
            if len(window_traj) >= 10:
                graph = build_tracklet_graph(
                    window_traj, frame_shape, config,
                    lane_coord_labels=lane_coord_labels,
                    single_direction=single_direction,
                    pixel_hom=pixel_hom,
                    ref_gps=ref_gps,
                    skip_density_contours=skip_density_contours,
                    image_wh=image_wh,
                )
                if graph is not None:
                    graphs.append(graph)
            w_idx += 1
            if w_idx % 50 == 0:
                logger.debug(f"    Window {w_idx}/{n_windows} ({len(graphs)} graphs so far)")
            t_start += stride_sec

        return graphs

    def _build_windowed_graphs_lane_groups(
        self,
        traj: pl.DataFrame,
        frame_shape: Tuple[int, int],
        config: dict,
        window_sec: float,
        stride_sec: float,
        lane_coord_labels=None,
        camera_loc: str = "",
        pixel_hom=None,
        ref_gps=None,
        traj_full: pl.DataFrame = None,
        image_wh: Optional[Tuple[int, int]] = None,
    ) -> List[Data]:
        """Partition trajectories by lane group, then build windowed graphs per group.

        Each lane group is a spatially + directionally coherent road segment
        produced by detect_lane_groups(). This subsumes both use_density_contours
        and split_by_direction.

        Lane group detection and vehicle assignment use the FULL (unsplit)
        trajectory so that the cached vid→gid mapping covers all vehicles
        across all splits. The split-filtered ``traj`` is used only for
        the actual graph construction within each lane group.
        """
        data_cfg = config.get("data", config)
        method = data_cfg.get("lane_group_method", "dbscan")

        # Build method-specific kwargs
        lane_group_kwargs = dict(
            min_track_points=data_cfg.get("tracklet_min_points", 5),
            min_gap_deg=data_cfg.get("direction_min_gap_deg", 45.0),
            min_vehicles_per_group=data_cfg.get("min_vehicles_per_group", 3),
        )
        if method == "dbscan":
            lane_group_kwargs.update(
                hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
                dbscan_min_samples=data_cfg.get("dbscan_min_samples", 3),
            )
            if "dbscan_eps" in data_cfg:
                lane_group_kwargs["dbscan_eps"] = data_cfg["dbscan_eps"]
        elif method == "learned":
            # Resolve cache directory for this camera's lane_groups.npz
            preprocess_dir = Path(data_cfg.get("preprocess_path", "./results/preprocess"))
            cam_cache_dir = preprocess_dir / camera_loc if camera_loc else preprocess_dir
            lane_group_kwargs.update(
                hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
                cache_dir=str(cam_cache_dir),
            )
        elif method == "community":
            lane_group_kwargs.update(
                hull_buffer_px=data_cfg.get("hull_buffer_px", 20.0),
                community_resolution=data_cfg.get("community_resolution", 1.0),
                scale_sigma=data_cfg.get("scale_penalty_sigma", 0.5),
                edge_radius=data_cfg.get("lane_group_edge_radius", 150.0),
            )
        else:  # density
            lane_group_kwargs.update(
                tracklet_length=data_cfg.get("tracklet_length", 15),
                sigma_along=data_cfg.get("density_sigma_along", 25.0),
                sigma_across=data_cfg.get("density_sigma_across", 6.0),
                neck_ratio=data_cfg.get("neck_ratio", 0.4),
                refine_contours=data_cfg.get("refine_contours", True),
                refine_buffer_px=data_cfg.get("refine_buffer_px", 25.0),
            )

        # --- Lane group disk cache ---
        # Cache contour geometry AND per-vehicle group assignments.
        # Contours are split-independent; vehicle assignments are cached for
        # the full trajectory and filtered to the current split via a column
        # join, avoiding repeated _compute_track_stats() + rasterization.
        from src.data.density_contours import detect_lane_groups
        preprocess_dir = Path(data_cfg.get("preprocess_path", "./results/preprocess"))
        cache_dir = preprocess_dir / camera_loc if camera_loc else preprocess_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Config hash for cache invalidation
        cache_key_data = json.dumps(
            {"method": method, **{k: v for k, v in sorted(lane_group_kwargs.items())
                                   if not isinstance(v, (np.ndarray,))}},
            sort_keys=True, default=str,
        )
        config_hash = hashlib.sha256(cache_key_data.encode()).hexdigest()[:16]
        cache_path = cache_dir / f"lane_groups_{method}.npz"

        headings = None
        traj_labeled = None  # traj with 'lane_group' column added

        # Use FULL (unsplit) trajectory for lane group detection & vehicle assignment.
        # Road geometry is split-independent; the cached vid->gid mapping must cover
        # ALL vehicles so that every split can match its vehicle IDs correctly.
        traj_for_detection = traj_full if traj_full is not None else traj

        if cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=True)
                if str(cached.get("config_hash", "")) == config_hash:
                    n_contours = int(cached["n_contours"])
                    headings = {i: float(cached[f"heading_{i}"]) for i in range(n_contours)}

                    # Try loading cached vehicle -> group mapping
                    if "vid_arr" in cached and "gid_arr" in cached:
                        vid_arr = cached["vid_arr"]
                        gid_arr = cached["gid_arr"]
                        # Match dtype of id column in traj for join compatibility
                        id_dtype = traj["id"].dtype
                        vid_to_gid = pl.DataFrame({
                            "id": pl.Series(vid_arr).cast(id_dtype),
                            "lane_group": gid_arr.astype(np.int32),
                        })
                        # Join onto current split's traj (fast integer join)
                        traj_labeled = traj.join(vid_to_gid, on="id", how="left")
                        # Vehicles not in the mapping (new or outside contours) get -1
                        traj_labeled = traj_labeled.with_columns(
                            pl.col("lane_group").fill_null(-1)
                        )
                        n_matched = traj_labeled.filter(pl.col("lane_group") >= 0)["id"].n_unique()
                        n_total = traj["id"].n_unique()
                        logger.info(
                            f"Loaded lane group cache with vehicle mapping: "
                            f"{cache_path} ({n_contours} groups, "
                            f"{len(vid_arr)} cached vids, "
                            f"{n_matched}/{n_total} matched in split, "
                            f"hash={config_hash})"
                        )
                        # If <50% of split vehicles matched, cache was likely built
                        # from a different split — invalidate and rebuild from full traj
                        if n_matched < n_total * 0.5 and traj_full is not None:
                            logger.warning(
                                f"Lane group cache has poor vehicle coverage "
                                f"({n_matched}/{n_total}), rebuilding from full trajectory"
                            )
                            traj_labeled = None  # force rebuild below
                    else:
                        # Legacy cache without vehicle mapping — fall through
                        logger.info(
                            f"Loaded lane group contours from cache (no vehicle mapping): "
                            f"{cache_path} ({n_contours} contours, hash={config_hash})"
                        )
                else:
                    logger.info("Lane group cache stale (config changed), re-detecting")
            except Exception as e:
                logger.warning(f"Failed to load lane group cache: {e}")

        if traj_labeled is None:
            # Either no cache, stale cache, or legacy cache without vid mapping.
            # Run detection on FULL trajectory for split-independent results.
            contours_loaded = None
            if headings is not None and cache_path.exists():
                # Legacy cache had contour geometry but no vid mapping — reuse it
                try:
                    legacy = np.load(cache_path, allow_pickle=True)
                    n_c = int(legacy["n_contours"])
                    contours_loaded = [legacy[f"contour_{i}"] for i in range(n_c)]
                except Exception:
                    contours_loaded = None

            if contours_loaded is not None:
                contours, group_headings = contours_loaded, headings
                track_stats = None
            else:
                contours, group_headings, track_stats = detect_lane_groups(
                    traj_for_detection, frame_shape, method=method, **lane_group_kwargs,
                )

            # Run assignment on FULL trajectory
            precomputed = (contours, group_headings, track_stats) if track_stats is not None else (contours, group_headings)
            groups = assign_vehicles_to_lane_groups(
                traj_for_detection, frame_shape, method=method,
                precomputed=precomputed,
                **lane_group_kwargs,
            )

            # Build vid -> gid mapping from assignment results
            vid_gid_pairs = []
            headings = {}
            for group_traj_g, heading_g, gid_g in groups:
                group_vids = group_traj_g["id"].unique().to_numpy()
                for vid in group_vids:
                    vid_gid_pairs.append((vid, gid_g))
                headings[gid_g] = heading_g

            if vid_gid_pairs:
                vids_all, gids_all = zip(*vid_gid_pairs)
                vid_to_gid = pl.DataFrame({
                    "id": np.array(vids_all),
                    "lane_group": np.array(gids_all, dtype=np.int32),
                })
                # Label the SPLIT-FILTERED trajectory (not full) for graph building
                traj_labeled = traj.join(vid_to_gid, on="id", how="left")
                traj_labeled = traj_labeled.with_columns(
                    pl.col("lane_group").fill_null(-1)
                )
            else:
                traj_labeled = traj.with_columns(pl.lit(-1).alias("lane_group"))

            # Save cache with contour geometry + vehicle mapping (from full traj)
            try:
                save_dict = {"config_hash": config_hash, "n_contours": len(contours)}
                for i, cnt in enumerate(contours):
                    save_dict[f"contour_{i}"] = cnt
                    save_dict[f"heading_{i}"] = headings.get(i, group_headings.get(i, 0.0))
                if vid_gid_pairs:
                    save_dict["vid_arr"] = np.array(vids_all)
                    save_dict["gid_arr"] = np.array(gids_all, dtype=np.int32)
                np.savez(cache_path, **save_dict)
                logger.info(f"Saved lane group cache with vehicle mapping: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save lane group cache: {e}")

        # Build windowed graphs per group from labeled DataFrame
        unique_groups = sorted(
            g for g in traj_labeled["lane_group"].unique().to_list() if g >= 0
        )
        all_graphs = []
        for gid in unique_groups:
            heading = headings.get(gid, 0.0)
            group_traj = traj_labeled.filter(pl.col("lane_group") == gid).drop("lane_group")
            if len(group_traj) < 10:
                continue

            group_labels = self._filter_labels_for_group(lane_coord_labels, group_traj, gid)

            graphs = self._build_windowed_graphs_single(
                group_traj, frame_shape, config,
                window_sec, stride_sec,
                lane_coord_labels=group_labels,
                single_direction=True,
                pixel_hom=pixel_hom,
                ref_gps=ref_gps,
                skip_density_contours=True,
                image_wh=image_wh,
            )
            for g in graphs:
                g.lane_group_id = gid
                g.lane_group_heading = heading
            all_graphs.extend(graphs)
            logger.debug(
                f"  Group {gid} (heading={np.degrees(heading):.0f}deg, "
                f"{len(group_traj)} pts): {len(graphs)} windowed graphs"
            )

        logger.info(
            f"{camera_loc}: {len(unique_groups)} lane groups -> "
            f"{len(all_graphs)} windowed graphs"
        )
        return all_graphs

    def _build_windowed_graphs_split(
        self,
        traj: pl.DataFrame,
        frame_shape: Tuple[int, int],
        config: dict,
        window_sec: float,
        stride_sec: float,
        lane_coord_labels=None,
        pixel_hom=None,
        ref_gps=None,
        image_wh: Optional[Tuple[int, int]] = None,
    ) -> List[Data]:
        """Split trajectories by direction, then build windowed graphs per group."""
        min_gap = config.get("data", config).get("direction_min_gap_deg", 45.0)
        groups = split_trajectories_by_direction(traj, min_gap_deg=min_gap)
        all_graphs = []

        for group_traj, heading, gid in groups:
            # Filter lane_coord_labels to vehicles in this direction group
            group_labels = self._filter_labels_for_group(lane_coord_labels, group_traj, gid)

            graphs = self._build_windowed_graphs_single(
                group_traj, frame_shape, config,
                window_sec, stride_sec,
                lane_coord_labels=group_labels,
                single_direction=True,
                pixel_hom=pixel_hom,
                ref_gps=ref_gps,
                image_wh=image_wh,
            )
            for g in graphs:
                g.direction_group = gid
                g.direction_heading = heading
            all_graphs.extend(graphs)

        return all_graphs

    @staticmethod
    def _filter_labels_for_group(
        lane_coord_labels,
        group_traj: pl.DataFrame,
        gid: int,
    ) -> Optional[Dict[int, int]]:
        """Filter lane_coord_labels to vehicles present in a direction group."""
        if lane_coord_labels is None:
            return None

        # Per-direction dict: {gid: {vid: label}}
        if isinstance(lane_coord_labels, dict) and all(isinstance(v, dict) for v in lane_coord_labels.values()):
            return lane_coord_labels.get(gid)

        # Single flat dict: filter to vehicles in this group
        group_vids = set(group_traj["id"].unique().to_list())
        return {vid: lbl for vid, lbl in lane_coord_labels.items() if vid in group_vids}

    @staticmethod
    def _clip_lane_to_observed(lane_pts_m: np.ndarray, centroids_m: np.ndarray,
                               margin: float = 10.0) -> Optional[np.ndarray]:
        """Clip a SUMO lane polyline to the observed tracklet extent.

        Projects the tracklet bounding box onto the lane and extracts the
        sub-segment that overlaps with the observation area (plus a margin).

        Args:
            lane_pts_m: (L, 2) lane points in local meters.
            centroids_m: (N, 2) tracklet centroids in local meters.
            margin: buffer in meters beyond tracklet extent.
        Returns:
            clipped: (K, 2) clipped lane points, or None if no overlap.
        """
        if len(lane_pts_m) < 2 or len(centroids_m) < 2:
            return None

        lane_line = LineString(lane_pts_m)
        if lane_line.length < 1.0:
            return None

        traj_mp = MultiPoint(centroids_m)
        minx, miny, maxx, maxy = traj_mp.bounds

        # Project bounding box corners onto lane to find overlap range
        corners = [
            Point(minx, miny), Point(maxx, miny),
            Point(minx, maxy), Point(maxx, maxy),
        ]
        projs = [lane_line.project(c) for c in corners]
        start = max(0.0, min(projs) - margin)
        end = min(lane_line.length, max(projs) + margin)

        if end - start < 5.0:  # too short segment
            return None

        # Extract clipped sub-segment by walking along the original coords
        # and interpolating at start/end positions
        clipped_coords = []

        # Add interpolated start point
        p_start = lane_line.interpolate(start)
        clipped_coords.append([p_start.x, p_start.y])

        # Add original interior points within the [start, end] range
        cum_dist = 0.0
        for i in range(len(lane_pts_m) - 1):
            seg_len = np.linalg.norm(lane_pts_m[i + 1] - lane_pts_m[i])
            next_dist = cum_dist + seg_len
            # Include point i+1 if it falls within [start, end]
            if next_dist > start and next_dist < end:
                clipped_coords.append(lane_pts_m[i + 1].tolist())
            cum_dist = next_dist

        # Add interpolated end point
        p_end = lane_line.interpolate(end)
        clipped_coords.append([p_end.x, p_end.y])

        if len(clipped_coords) < 2:
            return None

        return np.array(clipped_coords, dtype=np.float64)

    def _prepare_sumo_lanes_meters(
        self, graph: Data, pdata, config: dict,
        alignment_transform: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Convert GPS SUMO lanes to meters, align, clip, and filter — once per graph.

        Returns dict of lane_id -> (L, 2) meter-space points, or None.
        Result is shared by _match_to_sumo, _extract_target_polylines,
        and _build_gt_lanelet_graph to avoid triple redundant computation.

        Args:
            alignment_transform: optional precomputed (3,3) ICP transform.
                When provided, reuses this instead of running per-window ICP.
                This ensures consistent GT positions across time windows.
        """
        from scipy.spatial import cKDTree

        data_cfg = config.get("data", {})
        clip_lanes = data_cfg.get("clip_sumo_lanes", False)
        clip_margin = data_cfg.get("clip_sumo_margin_m", 10.0)
        min_evidence = data_cfg.get("min_lane_evidence", 5)
        match_radius_m = data_cfg.get("sumo_match_radius_m", 5.0)

        metadata = getattr(pdata, "metadata", {}) or {}
        gps_lane_geom = metadata.get("gps_lane_geom", None)
        lane_headings = metadata.get("lane_headings", {})

        if gps_lane_geom is None or len(gps_lane_geom) == 0:
            return None

        # Per-group heading filter: keep only SAME-direction lanes.
        # cos(Δθ) > 0 means Δθ < 90°, filtering out opposite direction
        # (cos ≈ -1) while keeping same-direction lanes even if slightly angled.
        graph_heading = getattr(graph, "lane_group_heading", None)
        if graph_heading is not None and lane_headings:
            filtered = {}
            for lid, pts in gps_lane_geom.items():
                lh = lane_headings.get(lid)
                if lh is not None and np.cos(graph_heading - lh) < 0.0:
                    continue
                filtered[lid] = pts
            if not filtered:
                # If filter removes everything, skip filtering entirely
                logger.warning(
                    f"Heading filter removed ALL {len(gps_lane_geom)} lanes "
                    f"(graph_heading={np.degrees(graph_heading):.0f}°). "
                    f"Keeping all lanes."
                )
            else:
                n_dropped = len(gps_lane_geom) - len(filtered)
                if n_dropped > 0:
                    logger.debug(
                        f"Heading filter: kept {len(filtered)}/{len(gps_lane_geom)} lanes "
                        f"(graph_heading={np.degrees(graph_heading):.0f}°)"
                    )
                gps_lane_geom = filtered

        if not gps_lane_geom:
            return None

        ref_gps = graph.ref_gps.numpy() if hasattr(graph, "ref_gps") and graph.ref_gps is not None else None
        if ref_gps is None:
            return None

        centroids = graph.centroids.numpy()  # (N, 2)
        if len(centroids) < 2:
            return None

        # Convert all GPS lanes to local meters, optionally clipping
        from src.utils.homography_scale import gps_to_local_meters
        lane_meter_geoms = {}
        for lid in sorted(gps_lane_geom.keys()):
            pts = np.array(gps_lane_geom[lid], dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
                continue
            lane_m = gps_to_local_meters(pts[:, :2], ref_gps)
            lane_meter_geoms[lid] = lane_m

        if not lane_meter_geoms:
            return None

        # Map alignment: correct systematic GPS offset between SUMO and tracklets.
        # Uses precomputed transform (from all time windows) for consistency,
        # or falls back to per-window ICP if no precomputed transform provided.
        if data_cfg.get("align_sumo_to_tracklets", True):
            if alignment_transform is not None:
                # Reuse precomputed alignment (consistent across time windows)
                from src.utils.map_alignment import MapAlignment
                lane_meter_geoms = {
                    lid: MapAlignment.apply_with_transform(pts, alignment_transform)
                    for lid, pts in lane_meter_geoms.items()
                }
            elif len(centroids) >= 5:
                # Fallback: per-window ICP
                from src.utils.map_alignment import MapAlignment
                all_sumo_pts = np.concatenate(list(lane_meter_geoms.values()), axis=0)
                if len(all_sumo_pts) >= 3:
                    aligner = MapAlignment()
                    aligner.estimate(all_sumo_pts, centroids, max_corresp_dist=50.0)
                    lane_meter_geoms = {
                        lid: aligner.apply(pts)
                        for lid, pts in lane_meter_geoms.items()
                    }

        # Clip lanes to observed region
        if clip_lanes:
            clipped_geoms = {}
            for lid, lane_m in lane_meter_geoms.items():
                clipped = self._clip_lane_to_observed(lane_m, centroids, margin=clip_margin)
                if clipped is not None:
                    clipped_geoms[lid] = clipped
            lane_meter_geoms = clipped_geoms

        if not lane_meter_geoms:
            return None

        # Filter by minimum tracklet evidence using bulk KDTree query
        if clip_lanes and min_evidence > 0 and lane_meter_geoms:
            all_lane_pts = []
            all_lane_lid = []
            for lid, lm in lane_meter_geoms.items():
                all_lane_pts.append(lm)
                all_lane_lid.extend([lid] * len(lm))
            all_pts = np.vstack(all_lane_pts)
            tree = cKDTree(all_pts)
            # Bulk query: nearest lane point for each centroid
            dists, indices = tree.query(centroids, k=1)
            close_mask = dists < match_radius_m
            # Count evidence per lane using vectorized operations
            lane_evidence = {}
            for lid in lane_meter_geoms:
                lane_evidence[lid] = 0
            close_indices = indices[close_mask]
            for idx in close_indices:
                lane_evidence[all_lane_lid[idx]] += 1
            # Remove lanes with insufficient evidence
            n_before = len(lane_meter_geoms)
            for lid in list(lane_meter_geoms.keys()):
                if lane_evidence.get(lid, 0) < min_evidence:
                    del lane_meter_geoms[lid]
            n_removed = n_before - len(lane_meter_geoms)
            if n_removed > 0:
                logger.debug(
                    f"Evidence filter: removed {n_removed}/{n_before} lanes "
                    f"(min_evidence={min_evidence}, match_radius={match_radius_m}m)"
                )

        if not lane_meter_geoms:
            logger.warning(
                f"All SUMO lanes filtered out! "
                f"(heading_filter + evidence_filter removed everything)"
            )
            return None

        cam_loc = getattr(graph, "camera_loc", "?")
        lg_id = getattr(graph, "lane_group_id", "?")
        lane_ids = sorted(lane_meter_geoms.keys())
        lane_lengths = {lid: f"{len(pts)}pts" for lid, pts in lane_meter_geoms.items()}
        logger.info(
            f"Loaded {len(lane_ids)} SUMO lanes for {cam_loc} lg={lg_id}: "
            f"{', '.join(f'{lid} ({lane_lengths[lid]})' for lid in lane_ids)}"
        )

        return lane_meter_geoms

    def _match_to_sumo(self, graph: Data, pdata, config: dict,
                       prepared_lanes: Optional[Dict[str, np.ndarray]] = None) -> torch.Tensor:
        """Spatial matching of tracklet centroids to SUMO lane geometries (meter-space).

        Returns (N,) integer lane labels (-1 = unmatched).
        Uses pre-computed prepared_lanes when available (from _prepare_sumo_lanes_meters)
        to avoid redundant GPS conversion, clipping, and evidence filtering.
        """
        from scipy.spatial import cKDTree

        data_cfg = config.get("data", {})
        match_radius_m = data_cfg.get("sumo_match_radius_m", 5.0)
        centroids = graph.centroids.numpy()  # (N, 2) meter-space

        # Use pre-computed lanes or fall back to full computation
        lane_meter_geoms = prepared_lanes
        if lane_meter_geoms is None:
            lane_meter_geoms = self._prepare_sumo_lanes_meters(graph, pdata, config)

        if not lane_meter_geoms:
            return torch.full((graph.num_nodes,), -1, dtype=torch.long)

        # Map string lane IDs -> consecutive integers
        final_lane_ids = sorted(lane_meter_geoms.keys())
        lane_id_to_idx = {lid: idx for idx, lid in enumerate(final_lane_ids)}

        all_lane_pts = []
        all_lane_idx = []
        for lid in final_lane_ids:
            lane_m = lane_meter_geoms[lid]
            all_lane_pts.append(lane_m)
            all_lane_idx.extend([lane_id_to_idx[lid]] * len(lane_m))

        all_pts = np.vstack(all_lane_pts)
        all_lane_idx = np.array(all_lane_idx, dtype=np.int64)
        tree = cKDTree(all_pts)
        dists, indices = tree.query(centroids, k=1)

        # Vectorized label assignment
        labels = np.full(len(centroids), -1, dtype=np.int64)
        close_mask = dists < match_radius_m
        labels[close_mask] = all_lane_idx[indices[close_mask]]

        n_matched = int(close_mask.sum())
        n_lanes = len(final_lane_ids)
        metadata = getattr(pdata, "metadata", {}) or {}
        n_total = len(metadata.get("gps_lane_geom", {}))
        match_frac = n_matched / max(len(centroids), 1)
        clip_str = f" (clipped from {n_total})" if data_cfg.get("clip_sumo_lanes", False) and n_total != n_lanes else ""
        # Show min distance for debugging coordinate mismatch
        min_dist_str = f", min_dist={dists.min():.4f}" if len(dists) > 0 else ""
        cent_range = f", cent=[{centroids.min():.3f},{centroids.max():.3f}]" if len(centroids) > 0 else ""
        lane_range = f", lane=[{all_pts.min():.3f},{all_pts.max():.3f}]" if len(all_pts) > 0 else ""
        logger.info(f"_match_to_sumo: {n_matched}/{len(centroids)} ({match_frac:.0%}) matched to {n_lanes} lanes{clip_str}, "
                     f"radius={match_radius_m:.4f}{min_dist_str}{cent_range}{lane_range}")

        return torch.from_numpy(labels)

    @staticmethod
    def _extract_target_polylines(
        pdata,
        config: dict,
        frame_shape: Tuple[int, int],
        graph: Optional[Data] = None,
        prepared_lanes: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract SUMO lane geometries as meter-space polyline targets.

        Uses pre-computed prepared_lanes when available (from _prepare_sumo_lanes_meters)
        to avoid redundant GPS conversion, clipping, and evidence filtering.

        Returns:
            target_polylines: (max_lanes, K, 2) in local meters, or None.
            target_valid_mask: (max_lanes,) boolean, or None.
        """
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        if data_cfg.get("use_lane_groups", False):
            num_slots = model_cfg.get("num_slots_per_group", model_cfg.get("num_slots", 8))
        else:
            num_slots = model_cfg.get("num_slots", 8)
        K = model_cfg.get("lane_shape_num_points", 10)

        # Use pre-computed lanes (already GPS-converted, clipped, evidence-filtered)
        lane_meter_geoms = prepared_lanes
        if lane_meter_geoms is None:
            # Fall back to full computation (backward compat)
            metadata = getattr(pdata, "metadata", {}) or {}
            gps_lane_geom = metadata.get("gps_lane_geom", None)
            if gps_lane_geom is None or len(gps_lane_geom) == 0:
                return None, None
            ref_gps = None
            if graph is not None and hasattr(graph, "ref_gps") and graph.ref_gps is not None:
                ref_gps = graph.ref_gps.numpy()
            if ref_gps is None:
                return None, None

            clip_lanes = data_cfg.get("clip_sumo_lanes", False)
            clip_margin = data_cfg.get("clip_sumo_margin_m", 10.0)
            min_evidence = data_cfg.get("min_lane_evidence", 5)
            match_radius_m = data_cfg.get("sumo_match_radius_m", 5.0)
            centroids_m = graph.centroids.numpy() if graph is not None and hasattr(graph, "centroids") else None

            lane_headings = metadata.get("lane_headings", {})
            graph_heading = getattr(graph, "lane_group_heading", None) if graph is not None else None
            working_geom = gps_lane_geom
            if graph_heading is not None and lane_headings:
                working_geom = {}
                for lid, pts in gps_lane_geom.items():
                    lh = lane_headings.get(lid)
                    if lh is not None and np.cos(graph_heading - lh) < 0.0:
                        continue
                    working_geom[lid] = pts

            from src.utils.homography_scale import gps_to_local_meters as _gps_to_local_meters
            lane_meter_geoms = {}
            for lane_id, pts in working_geom.items():
                pts = np.array(pts, dtype=np.float64) if not isinstance(pts, np.ndarray) else pts.astype(np.float64)
                if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
                    continue
                pts_m = _gps_to_local_meters(pts[:, :2], ref_gps)
                if clip_lanes and centroids_m is not None and len(centroids_m) >= 2:
                    clipped = TrackletDataset._clip_lane_to_observed(pts_m, centroids_m, margin=clip_margin)
                    if clipped is None:
                        continue
                    if min_evidence > 0:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(clipped)
                        n_nearby = (tree.query(centroids_m, k=1)[0] < match_radius_m).sum()
                        if n_nearby < min_evidence:
                            continue
                    pts_m = clipped
                lane_meter_geoms[lane_id] = pts_m

        if not lane_meter_geoms:
            return None, None

        lane_polys = []
        for lane_id in sorted(lane_meter_geoms.keys()):
            resampled = TrackletDataset._resample_polyline(lane_meter_geoms[lane_id], K)
            lane_polys.append(resampled)

        polylines = torch.zeros(num_slots, K, 2, dtype=torch.float32)
        mask = torch.zeros(num_slots, dtype=torch.bool)
        for i, poly in enumerate(lane_polys[:num_slots]):
            polylines[i] = torch.from_numpy(poly)
            mask[i] = True

        return polylines, mask

    @staticmethod
    def _resample_polyline(pts: np.ndarray, k: int) -> np.ndarray:
        """Resample a polyline to exactly k evenly-spaced points."""
        diffs = np.diff(pts, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        if total_len < 1e-8:
            return np.tile(pts[0], (k, 1)).astype(np.float32)
        target_dists = np.linspace(0, total_len, k)
        resampled = np.zeros((k, 2), dtype=np.float64)
        for i, d in enumerate(target_dists):
            idx = np.searchsorted(cum_len, d, side="right") - 1
            idx = min(idx, len(pts) - 2)
            seg_len = seg_lens[idx]
            if seg_len < 1e-8:
                resampled[i] = pts[idx]
            else:
                t = (d - cum_len[idx]) / seg_len
                resampled[i] = pts[idx] * (1 - t) + pts[idx + 1] * t
        return resampled.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        graph = self.samples[idx]
        if self.split == "train" and self.config.get("data", {}).get("augmentation", False):
            graph = self._augment(graph)
        return graph

    def _augment(self, graph: Data) -> Data:
        """Apply data augmentation to a training graph.

        - Lateral jitter: shift centroids/polylines perpendicular to heading (±1.5m)
        - Heading noise: rotate tangents by ±3°
        - Tracklet dropout: randomly drop 20% of tracklets
        """
        import copy
        graph = copy.copy(graph)  # shallow copy — tensors shared until modified

        N = graph.num_nodes
        rng = np.random.default_rng()

        # 1. Tracklet dropout: drop 20% of nodes
        dropout_rate = 0.2
        keep_mask = rng.random(N) > dropout_rate
        if keep_mask.sum() < 5:  # safety: keep at least 5 nodes
            return graph
        keep_idx = np.where(keep_mask)[0]
        keep_t = torch.tensor(keep_idx, dtype=torch.long)

        # Remap edges
        old_to_new = -np.ones(N, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(len(keep_idx))

        edge_index = graph.edge_index.numpy()
        src, dst = edge_index[0], edge_index[1]
        edge_keep = keep_mask[src] & keep_mask[dst]
        new_src = old_to_new[src[edge_keep]]
        new_dst = old_to_new[dst[edge_keep]]

        graph.x = graph.x[keep_t]
        graph.centroids = graph.centroids[keep_t].clone()
        graph.tangents = graph.tangents[keep_t].clone()
        graph.polylines = graph.polylines[keep_t].clone()
        graph.edge_index = torch.from_numpy(np.stack([new_src, new_dst], axis=0).astype(np.int64))
        if graph.edge_attr is not None and graph.edge_attr.shape[0] > 0:
            graph.edge_attr = graph.edge_attr[torch.tensor(np.where(edge_keep)[0], dtype=torch.long)]
        if hasattr(graph, "gt_labels") and graph.gt_labels is not None:
            graph.gt_labels = graph.gt_labels[keep_t]
        if hasattr(graph, "pseudo_labels") and graph.pseudo_labels is not None:
            graph.pseudo_labels = graph.pseudo_labels[keep_t]
        if hasattr(graph, "pixel_centroids") and graph.pixel_centroids is not None:
            graph.pixel_centroids = graph.pixel_centroids[keep_t]
        if hasattr(graph, "pixel_tangents") and graph.pixel_tangents is not None:
            graph.pixel_tangents = graph.pixel_tangents[keep_t]
        if hasattr(graph, "track_ids") and graph.track_ids is not None:
            graph.track_ids = graph.track_ids[keep_t]
        N = len(keep_idx)

        # 2. Lateral jitter: shift perpendicular to heading
        lateral_std = 1.0  # meters
        lateral_shift = rng.normal(0, lateral_std, size=N).astype(np.float32)
        tangents_np = graph.tangents.numpy()
        perp = np.stack([-tangents_np[:, 1], tangents_np[:, 0]], axis=1)  # perpendicular
        offset = perp * lateral_shift[:, None]
        graph.centroids = graph.centroids + torch.from_numpy(offset)
        # Also shift polylines (global_points if present)
        graph.polylines = graph.polylines + torch.from_numpy(offset[:, None, :])

        # 3. Heading noise: rotate tangents by ±3°
        angle_noise = rng.normal(0, np.radians(3.0), size=N).astype(np.float32)
        cos_a = np.cos(angle_noise)
        sin_a = np.sin(angle_noise)
        tx, ty = tangents_np[:, 0], tangents_np[:, 1]
        new_tx = cos_a * tx - sin_a * ty
        new_ty = sin_a * tx + cos_a * ty
        graph.tangents = torch.tensor(np.stack([new_tx, new_ty], axis=1), dtype=torch.float32)

        return graph

    def _build_gt_lanelet_graph(
        self,
        graph: Data,
        pdata,
        config: dict,
        prepared_lanes: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Build ground-truth lanelet graph from SUMO lane polylines.

        Uses pre-computed prepared_lanes when available (from _prepare_sumo_lanes_meters)
        to avoid redundant GPS conversion, clipping, and evidence filtering.

        Edge types: 0=no_edge, 1=successor, 2=merge, 3=diverge, 4=adjacent

        Stores results directly on the graph Data object.
        """
        from scipy.spatial import cKDTree

        data_cfg = config.get("data", {})
        P_fixed = data_cfg.get("lanelet_gt_waypoints_per_lane", 0)  # 0 = adaptive
        waypoint_spacing_m = data_cfg.get("lanelet_gt_waypoint_spacing", data_cfg.get("lanelet_gt_waypoint_spacing_m", 15.0))
        adj_thresh_m = data_cfg.get("lanelet_adjacent_threshold", data_cfg.get("lanelet_adjacent_threshold_m", 5.0))

        # Use pre-computed lanes or fall back to full computation
        lane_meter_geoms = prepared_lanes
        if lane_meter_geoms is None:
            lane_meter_geoms = self._prepare_sumo_lanes_meters(graph, pdata, config)

        # Handle map alignment (rare, opt-in feature — must recompute from GPS)
        metadata = getattr(pdata, "metadata", {}) or {}
        if data_cfg.get("use_map_alignment", False) and lane_meter_geoms:
            ref_gps = graph.ref_gps.cpu().numpy() if hasattr(graph, "ref_gps") and graph.ref_gps is not None else None
            centroids_m = graph.centroids.cpu().numpy() if hasattr(graph, "centroids") else None
            gps_lane_geom = metadata.get("gps_lane_geom", {})
            if ref_gps is not None and centroids_m is not None and len(centroids_m) >= 5:
                from src.utils.map_alignment import MapAlignment
                all_sumo_m = list(lane_meter_geoms.values())
                if all_sumo_m:
                    sumo_pts_m = np.concatenate(all_sumo_m, axis=0)
                    aligner = MapAlignment()
                    aligner.estimate(sumo_pts_m, centroids_m)
                    lane_meter_geoms = {lid: aligner.apply(pts) for lid, pts in lane_meter_geoms.items()}

        if not lane_meter_geoms:
            self._set_empty_lanelet_gt(graph)
            return

        all_positions = []
        all_tangents = []
        all_lane_ids = []
        lane_id_map = {}
        lane_node_ranges = {}

        for lane_key in sorted(lane_meter_geoms.keys()):
            lane_m = lane_meter_geoms[lane_key]

            # Determine waypoint count: adaptive based on lane length, or fixed
            if P_fixed > 0:
                P = P_fixed
            else:
                # Adaptive: one waypoint per spacing_m, min 2, max 10
                lane_length = float(np.sum(np.linalg.norm(np.diff(lane_m, axis=0), axis=1)))
                P = max(2, min(10, int(round(lane_length / waypoint_spacing_m)) + 1))

            # Resample to P waypoints
            waypoints = self._resample_polyline(lane_m, P)

            # Compute tangents at each waypoint (vectorized)
            tangents_wp = np.zeros_like(waypoints)
            if len(waypoints) >= 2:
                # Interior points: central difference
                tangents_wp[1:-1] = waypoints[2:] - waypoints[:-2]
                # Endpoints: forward/backward difference
                tangents_wp[0] = waypoints[1] - waypoints[0]
                tangents_wp[-1] = waypoints[-1] - waypoints[-2]
                norms = np.linalg.norm(tangents_wp, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                tangents_wp = tangents_wp / norms

            lid = len(lane_id_map)
            lane_id_map[lane_key] = lid

            start_idx = sum(len(p) for p in all_positions)
            all_positions.append(waypoints)
            all_tangents.append(tangents_wp)
            all_lane_ids.extend([lid] * len(waypoints))
            lane_node_ranges[lane_key] = (start_idx, start_idx + len(waypoints))

        if not all_positions:
            self._set_empty_lanelet_gt(graph)
            return

        positions = np.concatenate(all_positions, axis=0)  # (G, 2)
        tangents = np.concatenate(all_tangents, axis=0)  # (G, 2)
        lane_ids = np.array(all_lane_ids)  # (G,)
        G = len(positions)

        # Build edges
        edge_src = []
        edge_dst = []
        edge_types = []

        # Type 1: Successor edges (consecutive waypoints on same lane)
        for lane_key, (s, e) in lane_node_ranges.items():
            for i in range(s, e - 1):
                edge_src.append(i)
                edge_dst.append(i + 1)
                edge_types.append(1)  # successor
                edge_src.append(i + 1)
                edge_dst.append(i)
                edge_types.append(1)  # successor (reverse)

        # Type 4: Adjacent edges — use cKDTree.query_pairs instead of O(L²×P²) loops
        if G > 1:
            pos_tree = cKDTree(positions)
            close_pairs = pos_tree.query_pairs(adj_thresh_m, output_type='ndarray')
            if len(close_pairs) > 0:
                for ia, ib in close_pairs:
                    # Only cross-lane pairs (not same lane)
                    if lane_ids[ia] == lane_ids[ib]:
                        continue
                    # Check heading similarity (adjacent lanes should be parallel)
                    cos_sim = np.dot(tangents[ia], tangents[ib])
                    if cos_sim > 0.7:
                        edge_src.append(ia)
                        edge_dst.append(ib)
                        edge_types.append(4)  # adjacent
                        edge_src.append(ib)
                        edge_dst.append(ia)
                        edge_types.append(4)  # adjacent

        # Store on graph
        graph.gt_lanelet_positions = torch.tensor(positions, dtype=torch.float32)
        graph.gt_lanelet_tangents = torch.tensor(tangents, dtype=torch.float32)
        graph.gt_lanelet_lane_ids = torch.tensor(lane_ids, dtype=torch.long)

        if edge_src:
            graph.gt_lanelet_edge_index = torch.tensor(
                [edge_src, edge_dst], dtype=torch.long
            )
            graph.gt_lanelet_edge_types = torch.tensor(edge_types, dtype=torch.long)

            # Build dense SUMO adjacency type matrix (G, G) for topology prior
            sumo_adj = torch.zeros(G, G, dtype=torch.long)
            for s, d, t in zip(edge_src, edge_dst, edge_types):
                sumo_adj[s, d] = t
            graph.sumo_adj_types = sumo_adj
        else:
            graph.gt_lanelet_edge_index = torch.zeros(2, 0, dtype=torch.long)
            graph.gt_lanelet_edge_types = torch.zeros(0, dtype=torch.long)
            graph.sumo_adj_types = torch.zeros(G, G, dtype=torch.long)

    @staticmethod
    def _rotate_to_sd_frame(graph: Data) -> Data:
        """Rotate graph into lane-group-relative (s,d) frame.

        After rotation:
        - s-axis (+x) = along-traffic direction
        - d-axis (+y) = cross-traffic (left-to-right)
        - All tangents point roughly along +x
        - All positions are zero-centered then rotated
        """
        heading = float(graph.lane_group_heading)  # radians, global frame
        cos_h, sin_h = math.cos(-heading), math.sin(-heading)
        R = torch.tensor([[cos_h, -sin_h], [sin_h, cos_h]], dtype=torch.float32)

        # 1. Rotate centroids (zero-center first for numerical stability)
        origin = graph.centroids.mean(dim=0)  # (2,)
        centered = graph.centroids - origin    # (N, 2)
        graph.centroids = (centered @ R.T)     # (N, 2) in (s,d) frame

        # 2. Rotate tangents -> should now point along +s
        graph.tangents = (graph.tangents @ R.T)  # (N, 2)

        # 3. Update node features: tangent_cos (idx 3), tangent_sin (idx 4)
        graph.x[:, 3] = graph.tangents[:, 0]  # cos in (s,d) frame
        graph.x[:, 4] = graph.tangents[:, 1]  # sin in (s,d) frame

        # 4. Rotate GT lanelet positions & tangents (if present)
        if hasattr(graph, 'gt_lanelet_positions') and graph.gt_lanelet_positions is not None:
            if graph.gt_lanelet_positions.shape[0] > 0:
                gt_centered = graph.gt_lanelet_positions - origin
                graph.gt_lanelet_positions = (gt_centered @ R.T)
        if hasattr(graph, 'gt_lanelet_tangents') and graph.gt_lanelet_tangents is not None:
            if graph.gt_lanelet_tangents.shape[0] > 0:
                graph.gt_lanelet_tangents = (graph.gt_lanelet_tangents @ R.T)

        # 5. Rotate global_points (used by slot attention)
        if hasattr(graph, 'global_points') and graph.global_points is not None:
            if graph.global_points.shape[0] > 0:
                gp_centered = graph.global_points - origin
                graph.global_points = (gp_centered @ R.T)

        # 6. Store inverse transform metadata for visualization
        graph.sd_origin = origin          # (2,) center in global meters
        graph.sd_heading = torch.tensor(heading, dtype=torch.float32)  # scalar

        # 7. Set lane_group_heading to 0 -- model now always sees +x as traffic dir
        graph.lane_group_heading = 0.0

        return graph

    def _build_gt_lanelet_graph_from_annotation(
        self,
        graph: Data,
        annotation: dict,
        group_id: int,
        annot_lanes: Dict[str, np.ndarray],
        config: dict,
    ) -> None:
        """Build GT lanelet graph using annotation lanes and explicit relationships.

        Unlike _build_gt_lanelet_graph which infers adjacency from geometry,
        this uses the annotator's explicit successor/adjacent relationships.

        Edge types: 0=no_edge, 1=successor, 2=merge, 3=diverge, 4=adjacent
        """
        from src.data.annotation_loader import get_annotation_relationships

        data_cfg = config.get("data", {})
        P_fixed = data_cfg.get("lanelet_gt_waypoints_per_lane", 0)
        waypoint_spacing_m = data_cfg.get("lanelet_gt_waypoint_spacing", data_cfg.get("lanelet_gt_waypoint_spacing_m", 15.0))

        if not annot_lanes:
            self._set_empty_lanelet_gt(graph)
            return

        all_positions = []
        all_tangents = []
        all_lane_ids = []
        lane_id_map = {}  # lane_key -> int id
        lane_node_ranges = {}  # lane_key -> (start, end)

        for lane_key in sorted(annot_lanes.keys()):
            lane_m = annot_lanes[lane_key]

            if P_fixed > 0:
                P = P_fixed
            else:
                lane_length = float(np.sum(np.linalg.norm(np.diff(lane_m, axis=0), axis=1)))
                P = max(2, min(10, int(round(lane_length / waypoint_spacing_m)) + 1))

            waypoints = self._resample_polyline(lane_m, P)

            tangents_wp = np.zeros_like(waypoints)
            if len(waypoints) >= 2:
                tangents_wp[1:-1] = waypoints[2:] - waypoints[:-2]
                tangents_wp[0] = waypoints[1] - waypoints[0]
                tangents_wp[-1] = waypoints[-1] - waypoints[-2]
                norms = np.linalg.norm(tangents_wp, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                tangents_wp = tangents_wp / norms

            lid = len(lane_id_map)
            lane_id_map[lane_key] = lid

            start_idx = sum(len(p) for p in all_positions)
            all_positions.append(waypoints)
            all_tangents.append(tangents_wp)
            all_lane_ids.extend([lid] * len(waypoints))
            lane_node_ranges[lane_key] = (start_idx, start_idx + len(waypoints))

        if not all_positions:
            self._set_empty_lanelet_gt(graph)
            return

        positions = np.concatenate(all_positions, axis=0)
        tangents = np.concatenate(all_tangents, axis=0)
        lane_ids = np.array(all_lane_ids)
        G = len(positions)

        # Build edges from annotation relationships
        edge_src = []
        edge_dst = []
        edge_types = []

        # Type 1: Successor edges within each lane (consecutive waypoints)
        for lane_key, (s, e) in lane_node_ranges.items():
            for i in range(s, e - 1):
                edge_src.append(i)
                edge_dst.append(i + 1)
                edge_types.append(1)  # successor
                edge_src.append(i + 1)
                edge_dst.append(i)
                edge_types.append(1)  # successor (reverse)

        # Cross-lane edges from explicit annotation relationships
        relationships = get_annotation_relationships(annotation, group_id)
        for rel in relationships:
            from_key = f"annot_{rel['from_group']}_{rel['from_cls']}"
            to_key = f"annot_{rel['to_group']}_{rel['to_cls']}"

            if from_key not in lane_node_ranges or to_key not in lane_node_ranges:
                continue

            from_s, from_e = lane_node_ranges[from_key]
            to_s, to_e = lane_node_ranges[to_key]

            if rel["type"] == "successor":
                # Connect last waypoint of from_lane to first waypoint of to_lane
                edge_src.append(from_e - 1)
                edge_dst.append(to_s)
                edge_types.append(1)  # successor
                edge_src.append(to_s)
                edge_dst.append(from_e - 1)
                edge_types.append(1)  # successor (reverse)

            elif rel["type"] == "adjacent":
                # Connect corresponding waypoints between adjacent lanes
                from_len = from_e - from_s
                to_len = to_e - to_s
                n_pairs = min(from_len, to_len)
                for k in range(n_pairs):
                    # Map indices proportionally if lengths differ
                    fi = from_s + int(k * from_len / n_pairs)
                    ti = to_s + int(k * to_len / n_pairs)
                    edge_src.append(fi)
                    edge_dst.append(ti)
                    edge_types.append(4)  # adjacent
                    edge_src.append(ti)
                    edge_dst.append(fi)
                    edge_types.append(4)  # adjacent

        # Store on graph
        graph.gt_lanelet_positions = torch.tensor(positions, dtype=torch.float32)
        graph.gt_lanelet_tangents = torch.tensor(tangents, dtype=torch.float32)
        graph.gt_lanelet_lane_ids = torch.tensor(lane_ids, dtype=torch.long)

        if edge_src:
            graph.gt_lanelet_edge_index = torch.tensor(
                [edge_src, edge_dst], dtype=torch.long)
            graph.gt_lanelet_edge_types = torch.tensor(edge_types, dtype=torch.long)

            sumo_adj = torch.zeros(G, G, dtype=torch.long)
            for s, d, t in zip(edge_src, edge_dst, edge_types):
                sumo_adj[s, d] = t
            graph.sumo_adj_types = sumo_adj
        else:
            graph.gt_lanelet_edge_index = torch.zeros(2, 0, dtype=torch.long)
            graph.gt_lanelet_edge_types = torch.zeros(0, dtype=torch.long)
            graph.sumo_adj_types = torch.zeros(G, G, dtype=torch.long)

    @staticmethod
    def _set_empty_lanelet_gt(graph: Data):
        """Set empty GT lanelet tensors on graph for uniform batching."""
        graph.gt_lanelet_positions = torch.zeros(0, 2, dtype=torch.float32)
        graph.gt_lanelet_tangents = torch.zeros(0, 2, dtype=torch.float32)
        graph.gt_lanelet_lane_ids = torch.zeros(0, dtype=torch.long)
        graph.gt_lanelet_edge_index = torch.zeros(2, 0, dtype=torch.long)
        graph.gt_lanelet_edge_types = torch.zeros(0, dtype=torch.long)
        graph.sumo_adj_types = torch.zeros(0, 0, dtype=torch.long)

    def collate_fn(batch: List[Data]) -> Data:
        """Collate a batch of PyG Data objects using PyG's Batch.

        This handles variable-size graphs properly.
        """
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
