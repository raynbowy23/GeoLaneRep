#!/usr/bin/env python
"""Evaluate zero-shot lane matching with leave-one-camera-out.

Usage:
    python scripts/eval_zero_shot.py --checkpoint results/lane_contrastive/checkpoints/best.pt
    python scripts/eval_zero_shot.py --checkpoint best.pt --config configs/lane_contrastive.yaml
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot lane evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Optional config override")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    from src.training.zero_shot_eval import leave_one_camera_out_eval

    results = leave_one_camera_out_eval(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    # Save results
    output_path = args.output
    if output_path is None:
        ckpt_dir = Path(args.checkpoint).parent.parent
        output_path = str(ckpt_dir / "zero_shot_results.json")

    # Strip per-lane details for JSON serialization
    serializable = {}
    for cam, metrics in results.items():
        serializable[cam] = {k: v for k, v in metrics.items() if k != "per_lane"}

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
