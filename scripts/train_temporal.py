#!/usr/bin/env python
"""Train temporal lane encoder for anomaly detection.

Usage:
    python scripts/train_temporal.py \
        --config configs/lane_contrastive.yaml \
        --encoder-checkpoint results/lane_contrastive/checkpoints/best.pt
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
    parser = argparse.ArgumentParser(description="Train temporal lane encoder")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--encoder-checkpoint",
        required=True,
        help="Path to pre-trained LaneEncoder checkpoint",
    )
    args = parser.parse_args()

    from src.training.temporal_trainer import TemporalTrainer

    trainer = TemporalTrainer(args.config, args.encoder_checkpoint)
    trainer.run()


if __name__ == "__main__":
    main()
