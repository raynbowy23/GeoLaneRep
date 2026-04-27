#!/usr/bin/env python3

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
    parser = argparse.ArgumentParser(description="Train lane encoder (contrastive or joint)")
    parser.add_argument("--mode", choices=["contrastive", "joint"], required=True,
                        help="contrastive = stage 1 encoder only; joint = contrastive + temporal")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--held-out", nargs="+", default=None,
                        help="Camera(s) to hold out (overrides config for contrastive mode)")
    parser.add_argument("--encoder-checkpoint", default=None,
                        help="(joint mode only) optional warm-start from a trained encoder")
    args = parser.parse_args()

    if args.mode == "contrastive":
        from src.training.contrastive import ContrastiveTrainer
        trainer = ContrastiveTrainer(args.config)

        held_out = args.held_out
        if held_out is None:
            cfg_val = trainer.config.get("contrastive_training", {}).get("held_out_camera", None)
            if isinstance(cfg_val, list):
                held_out = cfg_val
            elif isinstance(cfg_val, str):
                held_out = [cfg_val]
        trainer.run(held_out_cameras=held_out)
    else:  # joint
        from src.training.joint_trainer import JointTrainer
        trainer = JointTrainer(args.config, args.encoder_checkpoint)
        trainer.run(held_out_cameras=args.held_out)


if __name__ == "__main__":
    main()
