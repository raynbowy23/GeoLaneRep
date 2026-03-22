"""Modify SUMO networks to inject generated lane variants.

Uses netconvert's plain XML pipeline to properly rebuild connections
and internal junctions after adding a lane:

    1. netconvert --sumo-net-file original.net.xml --plain-output-prefix tmp
    2. Modify the edge definition (add lane / increase numLanes)
    3. netconvert --node-files ... --edge-files ... -o modified.net.xml

This ensures valid junctions, connections, and signal timing.
"""

import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LANE_WIDTH = 3.2


@dataclass
class LaneAddition:
    """Specification for adding a lane to a SUMO network.

    Encoder-derived attributes control SUMO lane behavior:
        speed: from encoder's observed mean_speed
        width: narrower for merge/diverge, standard for mainline
        allow_change_left: lane change permission (merge lanes allow)
        allow_change_right: lane change permission
        disallow_classes: vehicle class restrictions
        is_merge: if True, lane has convergence behavior
    """

    edge_id: str
    position: str = "rightmost"  # "rightmost" or "leftmost"
    speed: float = 24.59  # m/s (~55 mph)
    width: float = DEFAULT_LANE_WIDTH
    allow_change_left: bool = True
    allow_change_right: bool = True
    disallow_classes: Optional[List[str]] = None  # e.g., ["truck"] for passing lanes
    is_merge: bool = False


def add_lane_to_network(
    net_path: Path,
    addition: LaneAddition,
    output_path: Optional[Path] = None,
) -> Path:
    """Add a lane to a SUMO network using netconvert plain XML pipeline.

    1. Export to plain XML (nod, edg, con, tll)
    2. Modify edge file to increase numLanes
    3. Rebuild with netconvert

    Args:
        net_path: Path to original .net.xml.
        addition: Lane addition specification.
        output_path: Where to write modified network. If None, uses a temp file.

    Returns:
        Path to the modified network file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prefix = tmpdir / "plain"

        # Step 1: Export to plain XML
        cmd_export = [
            "netconvert",
            "--sumo-net-file", str(net_path),
            "--plain-output-prefix", str(prefix),
        ]
        result = subprocess.run(cmd_export, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"netconvert export failed: {result.stderr[:300]}")

        edg_file = Path(f"{prefix}.edg.xml")
        nod_file = Path(f"{prefix}.nod.xml")
        con_file = Path(f"{prefix}.con.xml")
        tll_file = Path(f"{prefix}.tll.xml")
        typ_file = Path(f"{prefix}.typ.xml")

        if not edg_file.exists():
            raise RuntimeError(f"Edge file not generated: {edg_file}")

        # Step 2: Modify edge file — increase numLanes for target edge
        tree = ET.parse(str(edg_file))
        root = tree.getroot()

        target_edge = None
        for edge in root.findall("edge"):
            if edge.get("id") == addition.edge_id:
                target_edge = edge
                break

        if target_edge is None:
            raise ValueError(f"Edge '{addition.edge_id}' not found in edge file")

        old_num_lanes = int(target_edge.get("numLanes", "1"))
        new_num_lanes = old_num_lanes + 1
        target_edge.set("numLanes", str(new_num_lanes))

        # Optionally widen spread to accommodate new lane
        spread = target_edge.get("spreadType", "right")
        # Keep spread as-is; netconvert handles geometry

        tree.write(str(edg_file), xml_declaration=True, encoding="utf-8")
        logger.info(
            f"Modified edge '{addition.edge_id}': "
            f"{old_num_lanes} → {new_num_lanes} lanes"
        )

        # Step 3: Rebuild network with netconvert
        if output_path is None:
            import os
            fd, tmp = tempfile.mkstemp(suffix=".net.xml")
            output_path = Path(tmp)
            os.close(fd)

        cmd_build = [
            "netconvert",
            "--node-files", str(nod_file),
            "--edge-files", str(edg_file),
        ]
        if con_file.exists():
            cmd_build.extend(["--connection-files", str(con_file)])
        if tll_file.exists():
            cmd_build.extend(["--tllogic-files", str(tll_file)])
        if typ_file.exists():
            cmd_build.extend(["--type-files", str(typ_file)])

        cmd_build.extend([
            "-o", str(output_path),
            "--no-warnings", "true",
        ])

        result = subprocess.run(cmd_build, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"netconvert rebuild failed: {result.stderr[:300]}")

    # Step 4: Apply encoder-derived attributes to the new lane
    # The new lane is the highest-index lane on the edge
    new_lane_id = f"{addition.edge_id}_{new_num_lanes - 1}"
    _apply_lane_attributes(output_path, new_lane_id, addition)

    logger.info(
        f"Built modified network: {output_path} "
        f"(edge '{addition.edge_id}': {old_num_lanes}→{new_num_lanes} lanes, "
        f"speed={addition.speed:.1f}m/s, width={addition.width:.1f}m"
        f"{', merge' if addition.is_merge else ''}"
        f"{', disallow=' + ','.join(addition.disallow_classes) if addition.disallow_classes else ''})"
    )
    return output_path


def _apply_lane_attributes(
    net_path: Path, lane_id: str, addition: LaneAddition,
) -> None:
    """Apply encoder-derived attributes to a lane in compiled net.xml.

    Sets speed, width, lane change permissions, and vehicle class restrictions
    based on what the encoder learned about the target lane type.
    """
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    for edge in root.findall("edge"):
        for lane in edge.findall("lane"):
            if lane.get("id") == lane_id:
                # Speed from encoder observation
                lane.set("speed", f"{addition.speed:.2f}")

                # Width (narrower for merge/diverge)
                lane.set("width", f"{addition.width:.2f}")

                # Lane change permissions
                if not addition.allow_change_left:
                    lane.set("changeLeft", "authority")  # only emergency
                if not addition.allow_change_right:
                    lane.set("changeRight", "authority")

                # Vehicle class restrictions
                if addition.disallow_classes:
                    existing = lane.get("disallow", "")
                    classes = set(existing.split()) if existing else set()
                    classes.update(addition.disallow_classes)
                    lane.set("disallow", " ".join(sorted(classes)))

                logger.info(
                    f"Applied attributes to '{lane_id}': "
                    f"speed={addition.speed:.1f}, width={addition.width:.1f}"
                )

                tree.write(str(net_path), xml_declaration=True, encoding="utf-8")
                return

    logger.warning(f"Lane '{lane_id}' not found for attribute application")


def replace_lane_shape(
    net_path: Path,
    lane_id: str,
    geometry_sumo: np.ndarray,
    output_path: Optional[Path] = None,
) -> Path:
    """Replace a lane's shape in a compiled SUMO .net.xml with explicit geometry.

    This directly edits the net.xml to overwrite the lane's shape attribute
    with the provided coordinates. Use after add_lane_to_network to inject
    diffusion-generated geometry.

    Args:
        net_path: Path to .net.xml file.
        lane_id: SUMO lane ID (e.g., "38249173_5").
        geometry_sumo: (N, 2) array of SUMO meter coordinates.
        output_path: Where to write. If None, modifies in place.

    Returns:
        Path to the modified network file.
    """
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    found = False
    for edge in root.findall("edge"):
        for lane in edge.findall("lane"):
            if lane.get("id") == lane_id:
                shape_str = " ".join(
                    f"{x:.2f},{y:.2f}" for x, y in geometry_sumo
                )
                lane.set("shape", shape_str)
                # Also update length
                diffs = np.diff(geometry_sumo, axis=0)
                length = float(np.sum(np.linalg.norm(diffs, axis=1)))
                lane.set("length", f"{length:.2f}")
                found = True
                break
        if found:
            break

    if not found:
        raise ValueError(f"Lane '{lane_id}' not found in {net_path}")

    out = output_path or net_path
    tree.write(str(out), xml_declaration=True, encoding="utf-8")
    logger.info(f"Replaced shape for lane '{lane_id}' ({len(geometry_sumo)} points)")
    return out


def pixel_to_sumo(
    geometry_pixel: np.ndarray,
    reference_lane_geoms: dict,
    edge_id: str,
) -> np.ndarray:
    """Convert pixel-space [0,1] lane geometry to SUMO meter coordinates.

    Uses the bounding box of existing lanes on the same edge as the
    coordinate frame. The generated lane (in [0,1] pixel space) is mapped
    to the SUMO meter bounding box of the target edge.

    Args:
        geometry_pixel: (K, 2) normalized [0,1] lane geometry.
        reference_lane_geoms: Dict[lane_id, (N, 2)] from _parse_net_lanes.
        edge_id: Target edge ID.

    Returns:
        (K, 2) array in SUMO meter coordinates.
    """
    # Collect all points from lanes on this edge
    edge_points = []
    for lid, geom in reference_lane_geoms.items():
        if lid.rsplit("_", 1)[0] == edge_id:
            edge_points.append(geom)

    if not edge_points:
        raise ValueError(f"No lanes found for edge '{edge_id}'")

    all_pts = np.concatenate(edge_points, axis=0)
    sumo_min = all_pts.min(axis=0)
    sumo_max = all_pts.max(axis=0)
    sumo_range = sumo_max - sumo_min
    sumo_range = np.maximum(sumo_range, 1e-6)  # avoid div by zero

    # Map [0,1] → [sumo_min, sumo_max]
    geometry_sumo = geometry_pixel * sumo_range + sumo_min
    return geometry_sumo


def get_mainline_edges(net_path: Path) -> List[str]:
    """Get all mainline edge IDs from a SUMO network."""
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    edges = []
    for edge in root.findall("edge"):
        eid = edge.get("id", "")
        if eid.startswith(":"):
            continue
        etype = edge.get("type", "")
        if etype in ("highway.motorway", "highway.trunk"):
            edges.append(eid)
    return edges
