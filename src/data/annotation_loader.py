"""Load hand-drawn annotation JSON files and convert to normalized [0,1] lane geometries."""

import json
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_annotation_json(path) -> dict:
    """Parse annotation JSON and validate required keys.

    Args:
        path: Path to annotation.json file.

    Returns:
        Parsed annotation dict with keys: camera, image, lane_groups.

    Raises:
        ValueError: If required keys are missing.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    required = {"camera", "lane_groups"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Annotation {path} missing keys: {missing}")

    for lg in data["lane_groups"]:
        if "group_id" not in lg or "lanes" not in lg:
            raise ValueError(f"Annotation {path}: lane_group missing group_id or lanes")

    return data


def match_annotation_group(
    graph_heading_rad: float,
    annotation: dict,
    excluded_group_ids: Optional[set] = None,
) -> Optional[int]:
    """Find the annotation group whose heading best matches the graph heading.

    Compares graph lane_group_heading (radians) to each annotation group's
    heading_deg (converted to radians) using cosine similarity.
    Handles 180-degree ambiguity between density detector and annotator conventions.

    Args:
        graph_heading_rad: Graph heading in radians.
        annotation: Parsed annotation dict.
        excluded_group_ids: Set of annotation group_ids already claimed by
            other lane groups (for 1:1 matching).

    Returns:
        Best-matching group_id (always returns one if unclaimed groups remain).
    """
    if excluded_group_ids is None:
        excluded_group_ids = set()

    best_group = None
    best_cos = -2.0

    graph_deg = math.degrees(graph_heading_rad)
    matches = []
    for lg in annotation["lane_groups"]:
        gid = lg["group_id"]
        heading_deg = lg.get("heading_deg", 0.0)
        heading_rad = math.radians(heading_deg)
        # Check both normal and 180°-flipped headings to handle
        # convention differences between density detector and annotator
        diff = graph_heading_rad - heading_rad
        diff_flip = graph_heading_rad - (heading_rad + math.pi)
        cos_sim = max(math.cos(diff), math.cos(diff_flip))
        matches.append((gid, heading_deg, cos_sim))
        if cos_sim > best_cos and gid not in excluded_group_ids:
            best_cos = cos_sim
            best_group = gid

    logger.debug(
        f"match_annotation_group: graph={graph_deg:.1f}deg -> "
        + ", ".join(f"G{gid}({hdeg:.1f}deg cos={cs:.3f})" for gid, hdeg, cs in matches)
        + f" excluded={excluded_group_ids}"
        + f" => best=G{best_group} cos={best_cos:.3f}"
    )

    # Always return best available match — greedy 1:1 sorting in the caller
    # ensures optimal pairing. No threshold needed since every detected
    # lane group should get an annotation group if one is available.
    return best_group


def annotation_lanes_to_normalized(
    annotation: dict,
    group_id: int,
    image_wh: Tuple[int, int] = (1280, 720),
) -> Dict[str, np.ndarray]:
    """Convert annotation pixel waypoints to normalized [0,1] lane geometries.

    Extracts lanes for the matching annotation group and normalizes pixel
    coordinates by image dimensions.

    Args:
        annotation: Parsed annotation dict.
        group_id: Annotation group_id to extract.
        image_wh: (width, height) of the camera frame for normalization.

    Returns:
        Dict mapping "annot_{gid}_{cls}" -> (N, 2) normalized [0,1] coordinates.
    """
    group = None
    for lg in annotation["lane_groups"]:
        if lg["group_id"] == group_id:
            group = lg
            break

    if group is None:
        logger.warning(f"Annotation group {group_id} not found")
        return {}

    result = {}
    gid = group["group_id"]
    wh = np.array(image_wh, dtype=np.float64)

    for lane in group["lanes"]:
        cls_id = lane["cls_id"]
        waypoints = lane.get("waypoints", [])
        if len(waypoints) < 2:
            continue

        pts_px = np.array([[wp["x"], wp["y"]] for wp in waypoints], dtype=np.float64)
        pts_norm = pts_px / wh

        lane_key = f"annot_{gid}_{cls_id}"
        result[lane_key] = pts_norm

    return result


def get_group_lanes(
    annotation: dict,
    group_id: int,
    image_wh: Tuple[int, int] = (1920, 1080),
) -> list:
    """Get lane dicts with numpy waypoints for a specific annotation group.

    Args:
        annotation: Parsed annotation dict.
        group_id: Annotation group_id.
        image_wh: (width, height) for normalization.

    Returns:
        List of dicts with keys: cls_id, waypoints (N,2 normalized), color.
    """
    for lg in annotation["lane_groups"]:
        if lg["group_id"] == group_id:
            wh = np.array(image_wh, dtype=np.float64)
            lanes = []
            for lane in lg["lanes"]:
                wps = lane.get("waypoints", [])
                if len(wps) < 2:
                    continue
                pts = np.array([[wp["x"], wp["y"]] for wp in wps], dtype=np.float64)
                pts_norm = pts / wh
                lanes.append({
                    "cls_id": lane["cls_id"],
                    "waypoints": pts_norm,
                    "color": lane.get("color", [255, 255, 255]),
                })
            return lanes
    return []


def get_annotation_relationships(
    annotation: dict,
    group_id: int,
) -> list:
    """Extract relationships for a specific annotation group.

    Args:
        annotation: Parsed annotation dict.
        group_id: Annotation group_id.

    Returns:
        List of relationship dicts with keys: type, from_cls, to_cls.
    """
    for lg in annotation["lane_groups"]:
        if lg["group_id"] == group_id:
            return lg.get("relationships", [])
    return []
