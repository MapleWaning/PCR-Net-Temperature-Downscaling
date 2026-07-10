import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from open_source.unet_training import add_common_args, fit_baseline_unet, import_project_config


def parse_args():
    config = import_project_config()
    parser = argparse.ArgumentParser(description="Train the Basic U-Net baseline.")
    add_common_args(parser, config, str(Path("open_source") / "base_line" / "U-Net" / "runs"))
    parser.add_argument("--tbase-dir", default=config.GFS_DIR)
    return parser.parse_args(), config


def main():
    args, config = parse_args()
    fit_baseline_unet(args, config)


if __name__ == "__main__":
    main()
