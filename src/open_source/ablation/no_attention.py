import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from open_source.ablation.common import configure_common_parser, train_guided_ablation
from open_source.unet_training import import_project_config


def parse_args():
    config = import_project_config()
    parser = argparse.ArgumentParser(description="Train the PCR-Net ablation without attention and SFT.")
    configure_common_parser(parser, config)
    parser.set_defaults(generalization="year", use_attention=False, use_sft=False)
    parser.add_argument("--tbase-dir", default=config.TBASE_ADVANCE_DIR)
    parser.add_argument("--guidance-dir", default=config.CB_BASE)
    return parser.parse_args(), config


def main():
    args, config = parse_args()
    train_guided_ablation(args, config, "no-attention")


if __name__ == "__main__":
    main()
