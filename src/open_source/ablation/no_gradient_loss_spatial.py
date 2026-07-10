import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from open_source.ablation.common import configure_common_parser, train_pure_mse_ablation
from open_source.unet_training import import_project_config


def parse_args():
    config = import_project_config()
    parser = argparse.ArgumentParser(description="Train the spatial PCR-Net ablation without the gradient guidance loss.")
    configure_common_parser(parser, config)
    parser.set_defaults(generalization="station")
    parser.add_argument("--tbase-dir", default=config.TBASE_ADVANCE_DIR)
    return parser.parse_args(), config


def main():
    args, config = parse_args()
    train_pure_mse_ablation(args, config, "no-gradient-loss-spatial")


if __name__ == "__main__":
    main()
