import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from open_source.unet_training import add_common_args, add_model_switch_args, fit_pcr_net, import_project_config


def parse_args():
    config = import_project_config()
    parser = argparse.ArgumentParser(description="Train PCR-Net with a ResNet U-Net backbone.")
    add_common_args(parser, config, str(Path("open_source") / "PCR-Net" / "runs"))
    add_model_switch_args(parser)
    parser.add_argument("--tbase-dir", default=config.TBASE_ADVANCE_DIR)
    parser.add_argument("--guidance-dir", default=config.CB_SPATIAL_BASE)
    return parser.parse_args(), config


def main():
    args, config = parse_args()
    fit_pcr_net(args, config, run_label="pcr-net")


if __name__ == "__main__":
    main()
