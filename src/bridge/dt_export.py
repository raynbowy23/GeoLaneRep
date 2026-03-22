"""Export encoder state to JSON for geolane_twin digital twin integration.

The twin (Rust) watches a shared directory for JSON updates. This module
writes three types of files:

1. lane_state.json   — per-lane behavioral state (embeddings, LOS, anomaly)
2. topology.json     — generated lane geometries in SUMO coordinate space
3. modifications.json — lane modifications (close/speed/add) derived from
                        behavioral shifts detected by the temporal encoder

The twin's EncoderBridge reads these files and converts them into
LaneModification commands on its PendingModifications stack.

Usage:
    exporter = DigitalTwinExporter(
        output_dir="shared/dt_state",
        calibration=SumoCalibration(reference_points=[...]),
    )
    exporter.write_lane_state(metrics, anomaly_scores, embeddings)
    exporter.write_topology(generated_lanes, camera, group_id)
    exporter.write_modifications(behavioral_diff)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate calibration (pixel → SUMO)
# ---------------------------------------------------------------------------

@dataclass
class SumoCalibration:
    """Affine transform from pixel coordinates to SUMO network coordinates.

    Mirrors geolane_twin's SumoCalibration: 2×3 affine matrix fitted from
    reference point pairs (pixel_x, pixel_y) → (sumo_x, sumo_y).
    """

    reference_points: List[Tuple[float, float, float, float]] = field(
        default_factory=list
    )
    # 2×3 affine: [[a, b, tx], [c, d, ty]]
    transform: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.transform is None and len(self.reference_points) >= 3:
            self._fit_affine()
        elif self.transform is None:
            # Identity fallback (pixel = SUMO)
            self.transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def _fit_affine(self):
        """Least-squares affine fit from reference points."""
        pts = self.reference_points
        n = len(pts)
        A = np.zeros((2 * n, 6))
        b = np.zeros(2 * n)
        for i, (px, py, sx, sy) in enumerate(pts):
            A[2 * i] = [px, py, 1, 0, 0, 0]
            A[2 * i + 1] = [0, 0, 0, px, py, 1]
            b[2 * i] = sx
            b[2 * i + 1] = sy
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.transform = params.reshape(2, 3)

    def pixel_to_sumo(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates to SUMO coordinates."""
        p = np.array([px, py, 1.0])
        result = self.transform @ p
        return float(result[0]), float(result[1])

    def transform_polyline(
        self, points: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Convert (N, 2) pixel polyline to SUMO coordinates."""
        ones = np.ones((len(points), 1))
        augmented = np.hstack([points, ones])  # (N, 3)
        result = (self.transform @ augmented.T).T  # (N, 2)
        return [(float(r[0]), float(r[1])) for r in result]


# ---------------------------------------------------------------------------
# JSON schemas matching geolane_twin expectations
# ---------------------------------------------------------------------------

@dataclass
class LaneStateEntry:
    """Per-lane behavioral state for the twin."""

    lane_key: str
    camera: str
    group_id: int
    # Traffic metrics
    speed_mph: float
    density_veh_mi_ln: float
    flow_veh_hr_ln: float
    los: str
    vc_ratio: float
    # Behavioral state
    anomaly_score: float = 0.0
    behavioral_cluster: int = -1
    # Embedding (truncated for JSON size)
    embedding_norm: float = 0.0
    embedding_mean: float = 0.0
    # SUMO mapping
    sumo_edge_id: Optional[str] = None
    sumo_lane_id: Optional[str] = None


@dataclass
class TopologyEntry:
    """Generated lane geometry in SUMO coordinate space."""

    lane_key: str
    camera: str
    group_id: int
    lane_type: str  # "rightmost", "leftmost", "merge"
    # Polyline in SUMO coordinates
    shape: List[Tuple[float, float]] = field(default_factory=list)
    width: float = 3.2
    speed_limit: float = 24.6  # m/s (~55 mph)
    # Generation metadata
    confidence: float = 0.0
    source: str = "encoder_generation"


@dataclass
class ModificationEntry:
    """Lane modification command for the twin's PendingModifications stack.

    Maps to geolane_twin's LaneModification enum variants:
    - CloseLane { lane_id, reason }
    - SetSpeed { lane_id, old_speed, new_speed }
    - AddLane { edge_id, new_lane }
    - OpenLane { lane_id }
    """

    action: str  # "close", "set_speed", "add", "open"
    lane_id: Optional[str] = None
    edge_id: Optional[str] = None
    reason: str = ""
    old_speed: float = 0.0
    new_speed: float = 0.0
    # For AddLane: polyline shape in SUMO coordinates
    shape: List[Tuple[float, float]] = field(default_factory=list)
    width: float = 3.2
    speed_limit: float = 24.6


# ---------------------------------------------------------------------------
# Behavioral diff → modifications
# ---------------------------------------------------------------------------

def compute_modifications(
    current_state: List[LaneStateEntry],
    previous_state: Optional[List[LaneStateEntry]] = None,
    anomaly_threshold: float = 0.7,
    speed_drop_ratio: float = 0.3,
) -> List[ModificationEntry]:
    """Derive lane modifications from behavioral state changes.

    Detects:
    - Anomaly score spike → CloseLane (incident detected)
    - Speed drop > 70% → SetSpeed (congestion)
    - LOS degradation F → CloseLane (capacity failure)

    Args:
        current_state: Current per-lane behavioral state.
        previous_state: Previous state for diff (optional).
        anomaly_threshold: Anomaly score above which to close lane.
        speed_drop_ratio: Speed drop ratio triggering speed change.

    Returns:
        List of modification commands.
    """
    modifications = []
    prev_map = {}
    if previous_state:
        prev_map = {s.lane_key: s for s in previous_state}

    for lane in current_state:
        if lane.sumo_lane_id is None:
            continue

        # Anomaly-based closure
        if lane.anomaly_score > anomaly_threshold:
            modifications.append(ModificationEntry(
                action="close",
                lane_id=lane.sumo_lane_id,
                reason=f"Anomaly detected (score={lane.anomaly_score:.2f})",
            ))
            continue

        # Speed drop detection
        prev = prev_map.get(lane.lane_key)
        if prev and prev.speed_mph > 5.0:
            ratio = lane.speed_mph / prev.speed_mph
            if ratio < speed_drop_ratio:
                new_speed_ms = max(lane.speed_mph * 0.447, 2.0)  # mph → m/s
                old_speed_ms = prev.speed_mph * 0.447
                modifications.append(ModificationEntry(
                    action="set_speed",
                    lane_id=lane.sumo_lane_id,
                    old_speed=old_speed_ms,
                    new_speed=new_speed_ms,
                    reason=f"Speed drop {prev.speed_mph:.0f}→{lane.speed_mph:.0f} mph",
                ))

        # LOS F → close
        if lane.los == "F" and (prev is None or prev.los != "F"):
            modifications.append(ModificationEntry(
                action="close",
                lane_id=lane.sumo_lane_id,
                reason=f"LOS F (density={lane.density_veh_mi_ln:.1f})",
            ))

    return modifications


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class DigitalTwinExporter:
    """Write encoder state to JSON files for the twin to consume.

    The twin watches output_dir for file changes. Each write updates
    a generation counter so the twin can detect staleness.

    Args:
        output_dir: Shared directory path.
        calibration: Pixel → SUMO coordinate transform.
        lane_mapping: Optional dict mapping lane_key → sumo_lane_id.
    """

    def __init__(
        self,
        output_dir: str = "shared/dt_state",
        calibration: Optional[SumoCalibration] = None,
        lane_mapping: Optional[Dict[str, str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calibration = calibration or SumoCalibration()
        self.lane_mapping = lane_mapping or {}
        self._generation = 0

    def _write_json(self, filename: str, data: dict):
        """Write JSON with metadata header."""
        self._generation += 1
        payload = {
            "version": "1.0",
            "timestamp": time.time(),
            "generation": self._generation,
            "source": "geolane_encoder",
            "data": data,
        }
        path = self.output_dir / filename
        # Atomic write: write to temp then rename
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.rename(path)
        logger.debug(f"Wrote {path} (gen={self._generation})")

    def write_lane_state(
        self,
        metrics,
        anomaly_scores: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        cluster_labels: Optional[np.ndarray] = None,
    ):
        """Write per-lane behavioral state.

        Args:
            metrics: List of TrafficMetrics from bridge.
            anomaly_scores: (N,) temporal anomaly scores.
            embeddings: (N, D) lane embeddings.
            cluster_labels: (N,) behavioral cluster assignments.
        """
        entries = []
        for i, m in enumerate(metrics):
            entry = LaneStateEntry(
                lane_key=m.lane_key,
                camera=m.camera,
                group_id=getattr(m, "group_id", 0),
                speed_mph=m.speed_mph,
                density_veh_mi_ln=m.density_veh_mi_ln,
                flow_veh_hr_ln=m.flow_veh_hr_ln,
                los=m.los,
                vc_ratio=m.vc_ratio,
                sumo_edge_id=self._get_edge_id(m.lane_key),
                sumo_lane_id=self.lane_mapping.get(m.lane_key),
            )
            if anomaly_scores is not None and i < len(anomaly_scores):
                entry.anomaly_score = float(anomaly_scores[i])
            if embeddings is not None and i < len(embeddings):
                entry.embedding_norm = float(np.linalg.norm(embeddings[i]))
                entry.embedding_mean = float(embeddings[i].mean())
            if cluster_labels is not None and i < len(cluster_labels):
                entry.behavioral_cluster = int(cluster_labels[i])
            entries.append(entry)

        self._write_json("lane_state.json", {
            "lanes": [asdict(e) for e in entries],
            "n_lanes": len(entries),
        })
        logger.info(f"Exported {len(entries)} lane states")

    def write_topology(
        self,
        generated_lanes: List[dict],
        camera: str,
        group_id: int,
    ):
        """Write generated lane geometries in SUMO coordinates.

        Args:
            generated_lanes: List of dicts with 'points' (N,2), 'lane_type',
                'confidence', etc. from the generation pipeline.
            camera: Camera name.
            group_id: Lane group ID.
        """
        entries = []
        for lane in generated_lanes:
            points = lane.get("points", np.zeros((0, 2)))
            if isinstance(points, np.ndarray) and len(points) > 0:
                shape = self.calibration.transform_polyline(points)
            else:
                shape = []

            entries.append(TopologyEntry(
                lane_key=f"{camera}_g{group_id}_{lane.get('lane_type', 'unknown')}",
                camera=camera,
                group_id=group_id,
                lane_type=lane.get("lane_type", "unknown"),
                shape=shape,
                width=lane.get("width", 3.2),
                speed_limit=lane.get("speed_limit", 24.6),
                confidence=lane.get("confidence", 0.0),
            ))

        self._write_json("topology.json", {
            "lanes": [asdict(e) for e in entries],
            "camera": camera,
            "group_id": group_id,
        })
        logger.info(f"Exported {len(entries)} generated lanes for {camera}/g{group_id}")

    def write_modifications(
        self,
        current_state: List[LaneStateEntry],
        previous_state: Optional[List[LaneStateEntry]] = None,
        anomaly_threshold: float = 0.7,
    ):
        """Compute and write behavioral-diff modifications.

        Args:
            current_state: Current lane states.
            previous_state: Previous lane states for diff.
            anomaly_threshold: Threshold for anomaly-based closures.
        """
        mods = compute_modifications(
            current_state, previous_state,
            anomaly_threshold=anomaly_threshold,
        )
        self._write_json("modifications.json", {
            "modifications": [asdict(m) for m in mods],
            "n_modifications": len(mods),
        })
        if mods:
            logger.info(
                f"Exported {len(mods)} modifications: "
                + ", ".join(f"{m.action}({m.lane_id})" for m in mods)
            )

    def write_full_update(
        self,
        metrics,
        anomaly_scores=None,
        embeddings=None,
        cluster_labels=None,
        generated_lanes=None,
        camera: str = "",
        group_id: int = 0,
        previous_state=None,
    ):
        """Write all three files in one call (atomic update).

        Convenience method for the D2 ephemeral event demo.
        """
        self.write_lane_state(metrics, anomaly_scores, embeddings, cluster_labels)

        if generated_lanes:
            self.write_topology(generated_lanes, camera, group_id)

        # Build state entries for modification diff
        entries = []
        for i, m in enumerate(metrics):
            entry = LaneStateEntry(
                lane_key=m.lane_key,
                camera=m.camera,
                group_id=getattr(m, "group_id", 0),
                speed_mph=m.speed_mph,
                density_veh_mi_ln=m.density_veh_mi_ln,
                flow_veh_hr_ln=m.flow_veh_hr_ln,
                los=m.los,
                vc_ratio=m.vc_ratio,
                sumo_lane_id=self.lane_mapping.get(m.lane_key),
            )
            if anomaly_scores is not None and i < len(anomaly_scores):
                entry.anomaly_score = float(anomaly_scores[i])
            entries.append(entry)

        self.write_modifications(entries, previous_state)

    def _get_edge_id(self, lane_key: str) -> Optional[str]:
        """Derive SUMO edge ID from lane_key if mapping exists."""
        sumo_lane = self.lane_mapping.get(lane_key)
        if sumo_lane and "_" in sumo_lane:
            # SUMO convention: edge_id = lane_id without trailing _index
            return "_".join(sumo_lane.rsplit("_", 1)[:-1])
        return None
