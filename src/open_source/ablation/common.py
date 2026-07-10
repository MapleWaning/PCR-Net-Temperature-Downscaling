from pathlib import Path

from open_source.unet_training import (
    add_common_args,
    add_model_switch_args,
    fit_pcr_net,
    fit_resunet_pure_mse,
    import_project_config,
)


ABLATION_OUTPUT_DIR = str(Path("open_source") / "ablation" / "runs")


def configure_common_parser(parser, config):
    add_common_args(parser, config, ABLATION_OUTPUT_DIR)
    add_model_switch_args(parser)


def train_guided_ablation(args, config, run_label):
    return fit_pcr_net(args, config, run_label=run_label)


def train_pure_mse_ablation(args, config, run_label):
    return fit_resunet_pure_mse(args, config, run_label=run_label)
