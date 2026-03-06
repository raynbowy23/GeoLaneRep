#!/usr/bin/env python
"""Train contrastive lane encoder.

Usage:
    python scripts/train_contrastive.py --config configs/lane_contrastive.yaml
    python scripts/train_contrastive.py --config configs/lane_contrastive.yaml --held-out I43_Keefe
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
    parser = argparse.ArgumentParser(description="Train contrastive lane encoder")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument(
        "--held-out",
        nargs="+",
        default=None,
        help="Camera(s) to hold out for zero-shot eval (overrides config)",
    )
    args = parser.parse_args()

    from src.training.contrastive import ContrastiveTrainer

    trainer = ContrastiveTrainer(args.config)

    # Determine held-out cameras (CLI overrides config)
    held_out = args.held_out
    if held_out is None:
        cfg_val = trainer.config.get("contrastive_training", {}).get(
            "held_out_camera", None
        )
        if isinstance(cfg_val, list):
            held_out = cfg_val
        elif isinstance(cfg_val, str):
            held_out = [cfg_val]

    trainer.run(held_out_cameras=held_out)


if __name__ == "__main__":
    main()
