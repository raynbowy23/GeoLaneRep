"""Entry point for v3 Lanelet Discovery training."""

import sys
import argparse
import logging
from pathlib import Path

# Add project roots to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_ROOT = PROJECT_ROOT.parent / "graph_geolane"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(V1_ROOT))
sys.path.insert(0, str(V1_ROOT / "src"))

from src.training.trainer import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Lanelet Discovery Training (v3)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    logger.info(f"Config: {config_path}")
    pipeline = TrainingPipeline(str(config_path))
    pipeline.run()


if __name__ == "__main__":
    main()
