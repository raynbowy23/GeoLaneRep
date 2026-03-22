"""Translate lane encoder outputs to HCM traffic engineering parameters.

The encoder produces normalized behavioral statistics (mean_speed, curvature,
lateral_offset, trajectory_count) in pixel/frame space. This module:

1. Calibrates them to real-world units (mph, veh/mi/ln) using per-camera
   calibration factors or learned affine transforms.
2. Computes derived HCM parameters (flow, density, LOS).
3. Formats output for the CrossTraffic transportations-validator.

Two calibration modes:
  - Manual: per-camera pixel_per_meter + fps values (if available)
  - Learned: lightweight MLP trained on (encoder_stats → ground_truth_metrics) pairs
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """Per-camera calibration from pixel space to real-world units."""

    camera: str
    pixels_per_meter: float = 10.0      # image scale
    fps: float = 30.0                    # video frame rate
    lane_length_m: float = 100.0         # approximate visible lane length
    speed_limit_mph: float = 55.0        # posted speed limit
    num_lanes: int = 2                   # number of lanes in group
    density_scale: float = 0.0           # if >0, use this instead of capacity-based estimate


# HCM LOS thresholds for basic freeway segments (density in pc/mi/ln)
LOS_THRESHOLDS = {
    "A": 11.0,
    "B": 18.0,
    "C": 26.0,
    "D": 35.0,
    "E": 45.0,
    # F: density > 45 or V/C > 1.0
}

# Typical freeway capacity (pc/hr/ln)
TYPICAL_CAPACITY = 2200.0


# ---------------------------------------------------------------------------
# Affine calibration (manual)
# ---------------------------------------------------------------------------

def calibrate_speed(
    mean_speed_norm: float,
    calibration: CameraCalibration,
) -> float:
    """Convert normalized mean_speed to mph.

    mean_speed is mean displacement (pixels/frame) between consecutive
    trajectory points, normalized by image width.

    Args:
        mean_speed_norm: Normalized speed from traj_stats[0].
        calibration: Camera calibration parameters.

    Returns:
        Estimated speed in mph.
    """
    image_width_px = 1920  # typical
    # pixels/frame → meters/second → mph
    px_per_frame = mean_speed_norm * image_width_px
    m_per_sec = (px_per_frame * calibration.fps) / calibration.pixels_per_meter
    mph = m_per_sec * 2.237
    return mph


def calibrate_density(
    traj_count_norm: float,
    calibration: CameraCalibration,
    observation_period_sec: float = 60.0,
) -> float:
    """Convert normalized trajectory count to density (veh/mi/ln).

    traj_count_norm is the trajectory count divided by the max count
    across all lanes in the dataset. We estimate density using:
        density = (count / observation_time) / (lane_length * num_lanes)
        scaled to veh/mi/ln.

    Args:
        traj_count_norm: Normalized count from traj_stats[3].
        calibration: Camera calibration parameters.
        observation_period_sec: Duration over which trajectories were counted.

    Returns:
        Estimated density in veh/mi/ln.
    """
    # If a calibrated density scale is provided, use it directly
    if calibration.density_scale > 0:
        return traj_count_norm * calibration.density_scale
    # Fallback: capacity-based heuristic
    max_density_estimate = TYPICAL_CAPACITY / calibration.speed_limit_mph
    return traj_count_norm * max_density_estimate


def compute_flow(speed_mph: float, density_veh_mi_ln: float) -> float:
    """Fundamental traffic equation: flow = speed * density."""
    return speed_mph * density_veh_mi_ln


def classify_los(density: float, vc_ratio: float = None) -> str:
    """Classify Level of Service from density (HCM basic freeway).

    Args:
        density: Density in pc/mi/ln.
        vc_ratio: Volume-to-capacity ratio (optional, for LOS F check).

    Returns:
        LOS grade as string: "A" through "F".
    """
    if vc_ratio is not None and vc_ratio > 1.0:
        return "F"

    for grade, threshold in LOS_THRESHOLDS.items():
        if density <= threshold:
            return grade
    return "F"


# ---------------------------------------------------------------------------
# Learned calibration MLP
# ---------------------------------------------------------------------------

class TrafficTranslatorMLP(nn.Module):
    """Learned mapping from encoder features to calibrated HCM metrics.

    Input features (per lane):
        - traj_stats (4): mean_speed, curvature, lateral_offset, count_norm
        - role predictions (3): pred_rank, pred_edge[0], pred_edge[1]
        - embedding stats (3): embedding norm, embedding mean, embedding std

    Output (per lane):
        - speed_mph (1): calibrated speed
        - density (1): calibrated density in veh/mi/ln
        - los_logits (6): LOS classification logits (A-F)
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.speed_head = nn.Linear(hidden_dim, 1)
        self.density_head = nn.Linear(hidden_dim, 1)
        self.los_head = nn.Linear(hidden_dim, 6)  # A, B, C, D, E, F

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) per-lane features.

        Returns:
            Dict with speed, density, los_logits.
        """
        h = self.net(x)
        return {
            "speed_mph": self.speed_head(h).squeeze(-1),
            "density": torch.relu(self.density_head(h)).squeeze(-1),
            "los_logits": self.los_head(h),
        }


# ---------------------------------------------------------------------------
# Bridge: encoder → HCM metrics → CrossTraffic validator format
# ---------------------------------------------------------------------------

@dataclass
class TrafficMetrics:
    """HCM traffic metrics for a single lane."""

    lane_key: str
    camera: str
    speed_mph: float
    density_veh_mi_ln: float
    flow_veh_hr_ln: float
    los: str
    vc_ratio: float
    capacity: float = TYPICAL_CAPACITY

    def to_validator_dict(self) -> dict:
        """Format for CrossTraffic transportations-validator JSONExtractor."""
        return {
            "speed": round(self.speed_mph, 1),
            "density": round(self.density_veh_mi_ln, 1),
            "volume": round(self.flow_veh_hr_ln, 0),
            "los": self.los,
            "capacity": round(self.capacity, 0),
            "lane_key": self.lane_key,
            "camera": self.camera,
        }


class EncoderTrafficBridge:
    """Bridge from lane encoder to traffic engineering metrics.

    Supports two modes:
    - Manual calibration: uses per-camera CameraCalibration
    - Learned calibration: uses a trained TrafficTranslatorMLP

    Usage:
        bridge = EncoderTrafficBridge(calibrations={
            "US12_Park": CameraCalibration("US12_Park", pixels_per_meter=12.0),
        })
        metrics = bridge.translate(encoder_output, dataset_sample)
    """

    def __init__(
        self,
        calibrations: Optional[Dict[str, CameraCalibration]] = None,
        translator_mlp: Optional[TrafficTranslatorMLP] = None,
        default_calibration: Optional[CameraCalibration] = None,
    ):
        self.calibrations = calibrations or {}
        self.translator_mlp = translator_mlp
        self.default_calibration = default_calibration or CameraCalibration("default")

    def translate(
        self,
        traj_stats: np.ndarray,
        camera: str,
        lane_key: str,
        pred_rank: float = 0.5,
        pred_edge: np.ndarray = None,
        embedding: np.ndarray = None,
    ) -> TrafficMetrics:
        """Translate encoder outputs to traffic metrics.

        Args:
            traj_stats: (4,) [mean_speed, curvature, lateral_offset, count_norm]
            camera: Camera name for calibration lookup.
            lane_key: Lane identifier.
            pred_rank: Predicted lateral rank [0, 1].
            pred_edge: (2,) predicted edge logits.
            embedding: (D,) lane embedding (optional, for MLP mode).

        Returns:
            TrafficMetrics with calibrated values.
        """
        if self.translator_mlp is not None and embedding is not None:
            return self._translate_learned(
                traj_stats, camera, lane_key, pred_rank, pred_edge, embedding,
            )
        return self._translate_manual(traj_stats, camera, lane_key)

    def _translate_manual(
        self,
        traj_stats: np.ndarray,
        camera: str,
        lane_key: str,
    ) -> TrafficMetrics:
        """Manual calibration using affine transforms."""
        cal = self.calibrations.get(camera, self.default_calibration)

        speed = calibrate_speed(traj_stats[0], cal)
        density = calibrate_density(traj_stats[3], cal)
        flow = compute_flow(speed, density)
        vc_ratio = flow / cal.speed_limit_mph if cal.speed_limit_mph > 0 else 0.0
        # Use density-based V/C ratio
        vc_ratio = density / (TYPICAL_CAPACITY / speed) if speed > 0 else 0.0
        los = classify_los(density, vc_ratio)

        return TrafficMetrics(
            lane_key=lane_key,
            camera=camera,
            speed_mph=speed,
            density_veh_mi_ln=density,
            flow_veh_hr_ln=flow,
            los=los,
            vc_ratio=vc_ratio,
        )

    def _translate_learned(
        self,
        traj_stats: np.ndarray,
        camera: str,
        lane_key: str,
        pred_rank: float,
        pred_edge: np.ndarray,
        embedding: np.ndarray,
    ) -> TrafficMetrics:
        """Learned calibration via MLP."""
        if pred_edge is None:
            pred_edge = np.array([0.0, 0.0])

        # Build feature vector
        emb_stats = np.array([
            np.linalg.norm(embedding),
            embedding.mean(),
            embedding.std(),
        ])
        features = np.concatenate([
            traj_stats,              # 4
            [pred_rank],             # 1
            pred_edge[:2],           # 2
            emb_stats,               # 3
        ]).astype(np.float32)

        x = torch.tensor(features).unsqueeze(0)
        self.translator_mlp.eval()
        with torch.no_grad():
            out = self.translator_mlp(x)

        speed = float(out["speed_mph"][0])
        density = float(out["density"][0])
        flow = compute_flow(max(speed, 0.1), max(density, 0.0))
        los_idx = int(out["los_logits"][0].argmax())
        los_grades = ["A", "B", "C", "D", "E", "F"]
        los = los_grades[los_idx]
        vc_ratio = flow / TYPICAL_CAPACITY

        return TrafficMetrics(
            lane_key=lane_key,
            camera=camera,
            speed_mph=speed,
            density_veh_mi_ln=density,
            flow_veh_hr_ln=flow,
            los=los,
            vc_ratio=vc_ratio,
        )

    def translate_batch(
        self,
        dataset,
        encoder_output: dict = None,
    ) -> List[TrafficMetrics]:
        """Translate all lanes in a dataset to traffic metrics.

        Args:
            dataset: LaneDataset with samples.
            encoder_output: Dict with 'embedding', 'pred_rank', 'pred_edge'
                tensors. If None, uses manual calibration from traj_stats only.

        Returns:
            List of TrafficMetrics, one per lane.
        """
        results = []
        for i, sample in enumerate(dataset.samples):
            kwargs = {
                "traj_stats": sample.traj_stats,
                "camera": sample.camera,
                "lane_key": sample.lane_key,
            }
            if encoder_output is not None:
                kwargs["pred_rank"] = float(encoder_output["pred_rank"][i])
                kwargs["pred_edge"] = encoder_output["pred_edge"][i].numpy()
                kwargs["embedding"] = encoder_output["embedding"][i].numpy()

            results.append(self.translate(**kwargs))
        return results
