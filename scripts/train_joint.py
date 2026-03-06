#!/usr/bin/env python
"""Train joint contrastive + temporal lane encoder.

Usage:
    python scripts/train_joint.py \
        --config configs/lane_contrastive.yaml \
        --encoder-checkpoint results/lane_contrastive/checkpoints/best.pt  # optional warm-start
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Train joint contrastive + temporal encoder")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--encoder-checkpoint",
        default=None,
        help="Optional path to pre-trained LaneEncoder checkpoint for warm-start",
    )
    parser.add_argument(
        "--held-out",
        nargs="+",
        default=None,
        help="Camera(s) to hold out from training",
    )
    args = parser.parse_args()

    from src.training.joint_trainer import JointTrainer

    trainer = JointTrainer(args.config, args.encoder_checkpoint)
    trainer.run(held_out_cameras=args.held_out)


if __name__ == "__main__":
    main()
