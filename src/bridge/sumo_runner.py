"""Run SUMO simulations on existing per-camera networks to extract ground-truth
traffic metrics (speed, density, LOS) for bridge validation.

Each camera location has a pre-built SUMO network in the graph_geolane dataset.
This module:
1. Runs SUMO in batch mode with edge-level output collection
2. Parses the output to extract per-lane mean speed, density, and flow
3. Maps SUMO lane IDs to encoder lane keys via geometric matching

The resulting metrics serve as independent ground truth for the C-series
bridge figures, replacing the circular traj_stats-based reference.
"""

import logging
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# SUMO network root in graph_geolane
DEFAULT_SUMO_ROOT = Path(__file__).resolve().parent.parent.parent.parent / (
    "graph_geolane/dataset/sumo"
)

# HCM LOS thresholds (density in pc/mi/ln) — same as traffic_translator.py
LOS_THRESHOLDS = {"A": 11.0, "B": 18.0, "C": 26.0, "D": 35.0, "E": 45.0}

# Road types to include (match OSMConnection.ALLOWED_ROAD_TYPES)
ALLOWED_ROAD_TYPES = {"highway.motorway", "highway.trunk"}


@dataclass
class SUMOLaneMetrics:
    """Per-lane metrics extracted from SUMO simulation."""

    lane_id: str          # SUMO lane ID (e.g., "6783546#0_0")
    edge_id: str          # SUMO edge ID
    speed_m_s: float      # mean speed in m/s
    density_veh_km: float  # vehicles per km
    flow_veh_hr: float    # vehicles per hour
    occupancy: float      # lane occupancy [0, 1]
    geometry: np.ndarray  # (N, 2) lane shape in SUMO coordinates

    @property
    def speed_mph(self) -> float:
        return self.speed_m_s * 2.237

    @property
    def density_veh_mi_ln(self) -> float:
        return self.density_veh_km * 1.609

    @property
    def los(self) -> str:
        d = self.density_veh_mi_ln
        for grade, threshold in LOS_THRESHOLDS.items():
            if d <= threshold:
                return grade
        return "F"


def _parse_net_lanes(net_path: Path) -> Dict[str, np.ndarray]:
    """Parse lane geometries from a SUMO .net.xml file.

    Returns:
        Dict mapping lane_id -> (N, 2) array of SUMO coordinates.
        Only includes lanes on allowed road types.
    """
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    lane_geoms = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if edge_id.startswith(":"):
            continue
        edge_type = edge.get("type", "")
        if edge_type not in ALLOWED_ROAD_TYPES:
            continue

        for lane in edge.findall("lane"):
            lane_id = lane.get("id")
            shape_str = lane.get("shape", "")
            if not shape_str:
                continue
            points = []
            for p in shape_str.split():
                coords = p.split(",")
                points.append((float(coords[0]), float(coords[1])))
            lane_geoms[lane_id] = np.array(points, dtype=np.float64)

    return lane_geoms


def _generate_calibrated_demand(
    net_path: Path, output_file: Path, sim_duration: int = 3600,
    flow_rate: int = 1200,
) -> bool:
    """Generate calibrated flow demand for mainline edges.

    Creates vehicle flows between connected edge pairs using
    departLane="free" to spread traffic across lanes naturally.
    Only creates flows between edges that share a junction node
    (i.e., physically connected), avoiding unreachable OD pairs.

    Args:
        net_path: Path to SUMO network file.
        output_file: Path to write routes XML.
        sim_duration: Simulation duration in seconds.
        flow_rate: Vehicle flow rate in veh/hr per direction.

    Returns:
        True if demand was generated successfully.
    """
    tree = ET.parse(str(net_path))
    root = tree.getroot()

    # Collect mainline edges with their from/to nodes
    edge_info = {}  # edge_id -> (from_node, to_node)
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if edge_id.startswith(":"):
            continue
        edge_type = edge.get("type", "")
        if edge_type in ALLOWED_ROAD_TYPES:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            edge_info[edge_id] = (from_node, to_node)

    if not edge_info:
        return False

    # Find connected edge pairs: edge A's to_node == edge B's from_node
    # Build chains for longer routes
    # For simplicity, create flows between fringe edges (sources/sinks)
    # A fringe edge has no predecessor (from_node not in any to_node set)
    # or no successor (to_node not in any from_node set)
    to_nodes = {v[1] for v in edge_info.values()}
    from_nodes = {v[0] for v in edge_info.values()}

    source_edges = [e for e, (f, t) in edge_info.items() if f not in to_nodes]
    sink_edges = [e for e, (f, t) in edge_info.items() if t not in from_nodes]

    # If no clear fringe, use all edges
    if not source_edges:
        source_edges = list(edge_info.keys())
    if not sink_edges:
        sink_edges = list(edge_info.keys())

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
        'length="5" maxSpeed="30.0" speedFactor="normc(1.0,0.1,0.2,2.0)"/>',
    ]

    flow_id = 0
    # Distribute flow_rate across OD pairs
    n_pairs = max(len(source_edges) * len(sink_edges), 1)
    per_pair_rate = max(flow_rate // n_pairs, 1)
    period = max(1, int(3600 / per_pair_rate))

    for from_edge in source_edges:
        for to_edge in sink_edges:
            if from_edge == to_edge:
                continue
            lines.append(
                f'    <flow id="f{flow_id}" type="car" '
                f'begin="0" end="{sim_duration}" '
                f'from="{from_edge}" to="{to_edge}" '
                f'period="{period}" '
                f'departLane="free" departSpeed="max"/>'
            )
            flow_id += 1

    lines.append("</routes>")
    output_file.write_text("\n".join(lines))
    logger.info(
        f"Generated {flow_id} flows, period={period}s "
        f"({len(source_edges)} sources, {len(sink_edges)} sinks)"
    )
    return True


def run_sumo_simulation(
    camera: str,
    sumo_root: Path = None,
    sim_duration: int = 3600,
    aggregation_period: int = 300,
    flow_rate: int = 1200,
    collect_fcd: bool = False,
    fcd_output_path: Optional[Path] = None,
) -> List[SUMOLaneMetrics]:
    """Run SUMO simulation for a camera and extract per-lane metrics.

    Args:
        camera: Camera name (e.g., "US12_Park").
        sumo_root: Root directory containing per-camera SUMO networks.
        sim_duration: Simulation duration in seconds.
        aggregation_period: Aggregation interval for edge output (seconds).
        flow_rate: Vehicle flow rate per edge in veh/hr.
        collect_fcd: If True, also collect FCD (Floating Car Data) output
            for trajectory extraction.
        fcd_output_path: Where to save FCD output. If None and collect_fcd
            is True, saves to {sumo_root}/{camera}/fcd_output.xml.

    Returns:
        List of SUMOLaneMetrics, one per mainline lane.
    """
    if sumo_root is None:
        sumo_root = DEFAULT_SUMO_ROOT

    camera_dir = sumo_root / camera
    net_file = camera_dir / "osm.net.xml"
    net_file_gz = camera_dir / "osm.net.xml.gz"

    # Use uncompressed if available, else compressed
    if net_file.exists():
        net_path = net_file
    elif net_file_gz.exists():
        net_path = net_file_gz
    else:
        logger.warning(f"No SUMO network found for {camera} at {camera_dir}")
        return []

    # Parse lane geometries from network
    if net_file.exists():
        lane_geoms = _parse_net_lanes(net_file)
    else:
        import gzip
        import shutil
        with tempfile.NamedTemporaryFile(suffix=".net.xml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with gzip.open(net_file_gz, "rb") as f_in:
            with open(tmp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        lane_geoms = _parse_net_lanes(tmp_path)
        tmp_path.unlink()

    # Create edge output config
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        edge_output = tmpdir / "edge_output.xml"
        demand_file = tmpdir / "demand.rou.xml"

        # Generate demand using randomTrips with calibrated period.
        # period = 3600 / flow_rate (seconds between departures).
        # departLane="free" spreads vehicles across lanes naturally,
        # avoiding merge bottlenecks that cause bimodal A/F patterns.
        period = max(0.5, 3600.0 / flow_rate)
        random_trips = Path("/usr/share/sumo/tools/randomTrips.py")
        if random_trips.exists():
            trips_file = tmpdir / "trips.xml"
            routes_file = tmpdir / "routes.rou.xml"
            cmd_trips = [
                "python", str(random_trips),
                "-n", str(net_path),
                "-o", str(trips_file),
                "-r", str(routes_file),
                "-e", str(sim_duration),
                "-p", str(period),
                "--fringe-factor", "5",
                "--allow-fringe",
                "-t", 'departLane="free" departSpeed="max"',
                "--validate",
            ]
            try:
                trip_result = subprocess.run(
                    cmd_trips, capture_output=True, text=True, timeout=60,
                )
                if trip_result.returncode == 0 and routes_file.exists():
                    demand_file = routes_file
                else:
                    logger.warning(
                        f"randomTrips failed for {camera}: "
                        f"{trip_result.stderr[:300]}"
                    )
                    demand_file = None
            except Exception as e:
                logger.warning(f"Trip generation failed for {camera}: {e}")
                demand_file = None
        else:
            logger.warning("randomTrips.py not found")
            demand_file = None

        if demand_file is None:
            logger.warning(f"No demand generated for {camera}")
            return []

        # Build additional file for lane-level data collection
        additional_file = tmpdir / "additional.xml"
        additional_file.write_text(
            f'<additional>\n'
            f'  <laneData id="lane_metrics" '
            f'file="{edge_output}" '
            f'freq="{aggregation_period}" '
            f'excludeEmpty="true"/>\n'
            f'</additional>\n'
        )

        # FCD output for trajectory extraction
        fcd_file = None
        if collect_fcd:
            if fcd_output_path is not None:
                fcd_file = Path(fcd_output_path)
            else:
                fcd_file = camera_dir / "fcd_output.xml"

        # Build SUMO command
        cmd = [
            "sumo",
            "-n", str(net_path),
            "-a", str(additional_file),
            "--end", str(sim_duration),
            "--no-step-log", "true",
            "--duration-log.statistics", "true",
            "--ignore-route-errors", "true",
        ]
        if demand_file is not None and demand_file.exists():
            cmd.extend(["-r", str(demand_file)])
        if fcd_file is not None:
            cmd.extend(["--fcd-output", str(fcd_file)])

        logger.info(
            f"Running SUMO for {camera} ({sim_duration}s, "
            f"flow_rate={flow_rate}, period={period:.2f}s)..."
        )
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                logger.warning(
                    f"SUMO returned {result.returncode} for {camera}: "
                    f"{result.stderr[:500]}"
                )
                # Continue anyway — output may still be valid
        except subprocess.TimeoutExpired:
            logger.error(f"SUMO timed out for {camera}")
            return []
        except FileNotFoundError:
            logger.error("SUMO not found. Install with: apt install sumo")
            return []

        # Parse edge output
        if not edge_output.exists():
            logger.warning(f"No edge output generated for {camera}")
            return []

        return _parse_edge_output(edge_output, lane_geoms)


def _parse_edge_output(
    output_path: Path, lane_geoms: Dict[str, np.ndarray],
) -> List[SUMOLaneMetrics]:
    """Parse SUMO edgeData output XML into per-lane metrics.

    Aggregates across all time intervals to get mean values.
    """
    tree = ET.parse(str(output_path))
    root = tree.getroot()

    # Collect per-lane stats across intervals
    lane_stats: Dict[str, List[dict]] = {}

    for interval in root.findall("interval"):
        for edge in interval.findall("edge"):
            edge_id = edge.get("id", "")
            if edge_id.startswith(":"):
                continue

            # edgeData may have per-lane or per-edge granularity
            # Check for lane children first
            lanes = edge.findall("lane")
            if lanes:
                for lane_elem in lanes:
                    lane_id = lane_elem.get("id")
                    stats = _extract_lane_stats(lane_elem, edge_id, lane_id)
                    if stats:
                        lane_stats.setdefault(lane_id, []).append(stats)
            else:
                # Edge-level data: distribute to all lanes of this edge
                speed = _safe_float(edge.get("speed", "-1"))
                density = _safe_float(edge.get("density", "0"))
                # sampledSeconds gives total vehicle-seconds on the edge
                sampled = _safe_float(edge.get("sampledSeconds", "0"))

                if speed < 0 or sampled == 0:
                    continue

                # Find lanes belonging to this edge
                edge_lanes = [
                    lid for lid in lane_geoms if lid.rsplit("_", 1)[0] == edge_id
                ]
                n_lanes = max(len(edge_lanes), 1)

                for lid in edge_lanes:
                    stats = {
                        "speed": speed,
                        "density": density / n_lanes,
                        "sampled": sampled / n_lanes,
                    }
                    lane_stats.setdefault(lid, []).append(stats)

    # Aggregate and build results
    results = []
    for lane_id, intervals in lane_stats.items():
        if not intervals:
            continue

        # Only include lanes on allowed road types (present in lane_geoms)
        if lane_id not in lane_geoms:
            continue

        speeds = [s["speed"] for s in intervals if s["speed"] > 0]
        densities = [s["density"] for s in intervals]
        sampled = [s["sampled"] for s in intervals]

        if not speeds:
            continue

        mean_speed = np.mean(speeds)  # m/s
        mean_density = np.mean(densities)  # veh/km
        total_sampled = np.sum(sampled)

        # Flow = speed * density (fundamental relation)
        flow = mean_speed * 3.6 * mean_density  # veh/hr (speed in km/h * density in veh/km)

        # Occupancy approximation from density
        # Assume average vehicle length 5m
        occupancy = min(mean_density * 0.005, 1.0)

        edge_id = lane_id.rsplit("_", 1)[0]

        results.append(SUMOLaneMetrics(
            lane_id=lane_id,
            edge_id=edge_id,
            speed_m_s=mean_speed,
            density_veh_km=mean_density,
            flow_veh_hr=flow,
            occupancy=occupancy,
            geometry=lane_geoms.get(lane_id, np.zeros((2, 2))),
        ))

    logger.info(
        f"Extracted metrics for {len(results)} mainline lanes "
        f"from {len(lane_stats)} total"
    )
    return results


def _extract_lane_stats(
    lane_elem: ET.Element, edge_id: str, lane_id: str,
) -> Optional[dict]:
    """Extract speed/density from a lane XML element."""
    speed = _safe_float(lane_elem.get("speed", "-1"))
    density = _safe_float(lane_elem.get("density", "0"))
    sampled = _safe_float(lane_elem.get("sampledSeconds", "0"))

    if speed < 0 or sampled == 0:
        return None

    return {"speed": speed, "density": density, "sampled": sampled}


def _safe_float(s: str) -> float:
    """Parse float, return -1 on failure."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return -1.0


def run_all_cameras(
    sumo_root: Path = None,
    cameras: List[str] = None,
    sim_duration: int = 3600,
    flow_rate: int = 1200,
    collect_fcd: bool = False,
) -> Dict[str, List[SUMOLaneMetrics]]:
    """Run SUMO for all cameras and collect metrics.

    Args:
        sumo_root: Root directory with per-camera SUMO networks.
        cameras: List of camera names. If None, discovers from sumo_root.
        sim_duration: Simulation duration in seconds.
        flow_rate: Vehicle flow rate per edge in veh/hr.
        collect_fcd: If True, also collect FCD output for trajectory extraction.

    Returns:
        Dict mapping camera name -> list of SUMOLaneMetrics.
    """
    if sumo_root is None:
        sumo_root = DEFAULT_SUMO_ROOT

    if cameras is None:
        cameras = sorted([
            d.name for d in sumo_root.iterdir()
            if d.is_dir() and (d / "osm.net.xml").exists()
        ])

    all_metrics = {}
    for cam in cameras:
        metrics = run_sumo_simulation(
            cam, sumo_root, sim_duration, flow_rate=flow_rate,
            collect_fcd=collect_fcd,
        )
        if metrics:
            all_metrics[cam] = metrics
            logger.info(
                f"  {cam}: {len(metrics)} lanes, "
                f"mean speed {np.mean([m.speed_mph for m in metrics]):.1f} mph"
            )
        else:
            logger.warning(f"  {cam}: no metrics (simulation may have failed)")

    return all_metrics


def match_sumo_to_encoder_lanes(
    sumo_metrics: Dict[str, List[SUMOLaneMetrics]],
    dataset,
    distance_threshold: float = 50.0,
) -> Dict[str, SUMOLaneMetrics]:
    """Match SUMO lanes to encoder dataset lanes by geometric proximity.

    Uses centroid distance between SUMO lane geometry and encoder lane
    geometry (both in their respective coordinate spaces) to find the
    closest match. Since both are derived from the same road network,
    the matching is based on relative position within each camera's
    lane group.

    Args:
        sumo_metrics: Per-camera SUMO metrics from run_all_cameras.
        dataset: LaneDataset with samples.
        distance_threshold: Max normalized distance for matching.

    Returns:
        Dict mapping lane_key -> SUMOLaneMetrics for matched lanes.
    """
    matched = {}

    for sample in dataset.samples:
        cam = sample.camera
        if cam not in sumo_metrics:
            continue

        cam_sumo = sumo_metrics[cam]
        if not cam_sumo:
            continue

        # Strategy: match by lateral rank ordering within each edge/group.
        # Encoder lanes in a group are ordered by lateral_rank.
        # SUMO lanes in an edge are ordered by lane index (rightmost = _0).

        # Group encoder lanes by group_id
        group_key = (cam, sample.group_id)

        # Group SUMO lanes by edge
        sumo_by_edge: Dict[str, List[SUMOLaneMetrics]] = {}
        for sm in cam_sumo:
            sumo_by_edge.setdefault(sm.edge_id, []).append(sm)

        # For each SUMO edge, sort lanes by index
        for edge_id, edge_lanes in sumo_by_edge.items():
            edge_lanes.sort(key=lambda m: int(m.lane_id.rsplit("_", 1)[1]))

        # Find encoder lanes in this group
        group_samples = [
            s for s in dataset.samples
            if s.camera == cam and s.group_id == sample.group_id
        ]
        group_samples.sort(key=lambda s: s.role.lateral_rank)

        # Match to the SUMO edge with the most similar lane count
        best_edge = None
        best_count_diff = float("inf")
        for edge_id, edge_lanes in sumo_by_edge.items():
            diff = abs(len(edge_lanes) - len(group_samples))
            if diff < best_count_diff:
                best_count_diff = diff
                best_edge = edge_id

        if best_edge is None:
            continue

        edge_lanes = sumo_by_edge[best_edge]

        # Match by position in sorted order
        for i, gs in enumerate(group_samples):
            if gs.lane_key == sample.lane_key:
                # Map to corresponding SUMO lane by relative position
                sumo_idx = int(i * len(edge_lanes) / len(group_samples))
                sumo_idx = min(sumo_idx, len(edge_lanes) - 1)
                matched[sample.lane_key] = edge_lanes[sumo_idx]
                break

    logger.info(
        f"Matched {len(matched)}/{len(dataset.samples)} encoder lanes "
        f"to SUMO lanes"
    )
    return matched
