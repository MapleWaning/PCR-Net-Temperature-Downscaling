import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    config = importlib.import_module("config")
    parser = argparse.ArgumentParser(description="Evaluate neural downscaling models with RMSE, MAE, MBE, and R2.")
    parser.add_argument("--dataset-type", choices=["pcr-net", "baseline"], default="pcr-net")
    parser.add_argument("--model-type", choices=["resattunet", "basic-unet"], default="resattunet")
    parser.add_argument("--generalization", choices=["year", "station"], default="year")
    parser.add_argument("--model-path", default=getattr(config, "FINAL_MODEL_PATH", getattr(config, "MODEL_PATH", None)))
    parser.add_argument("--tbase-dir", default=None)
    parser.add_argument("--guidance-dir", default=None)
    parser.add_argument("--static-path", default=config.STATIC_PATH)
    parser.add_argument("--truth-path", default=config.TRUTH_PATH)
    parser.add_argument("--test-meta-csv", default=config.TEST_META_CSV)
    parser.add_argument("--test-sample-csv", default=None)
    parser.add_argument("--test-years", type=comma_ints, default=config.TEST_YEAR)
    parser.add_argument("--batch-size", type=int, default=config.TEST_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--input-channels", type=int, default=config.TEST_INPUT_CHANNELS)
    parser.add_argument("--output-channels", type=int, default=config.TEST_OUTPUT_CHANNELS)
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-csv", default=None)
    parser.add_argument("--print-interval", type=int, default=50)
    return finalize_args(parser.parse_args(), config), config


def finalize_args(args, config):
    if args.tbase_dir is None:
        args.tbase_dir = config.TBASE_ADVANCE_DIR if args.dataset_type == "pcr-net" else config.GFS_DIR
    if args.guidance_dir is None:
        args.guidance_dir = getattr(config, "CB_SPATIAL_BASE", getattr(config, "CB_BASE", None))
    return args


def comma_ints(value):
    if isinstance(value, list):
        return value
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_dataset(args):
    from open_source.unet_datasets import (
        ClimateDownscaleDataset_Baseline,
        ClimateDownscaleDataset_Baseline_V2,
        ClimateDownscaleDataset_V3,
        ClimateDownscaleDataset_V4,
    )
    from open_source.unet_training import prepare_baseline_batch, prepare_guided_batch

    if args.dataset_type == "pcr-net":
        if args.generalization == "year":
            dataset = ClimateDownscaleDataset_V3(
                tbase_dir=args.tbase_dir,
                rf_base_dir=args.guidance_dir,
                static_npy_path=args.static_path,
                truth_npy_path=args.truth_path,
                years=args.test_years,
                split="test",
                sample_index_csv=args.test_sample_csv,
            )
        else:
            dataset = ClimateDownscaleDataset_V4(
                tbase_dir=args.tbase_dir,
                rf_base_dir=args.guidance_dir,
                static_npy_path=args.static_path,
                truth_npy_path=args.truth_path,
                target_station_ids=read_station_ids(args.test_meta_csv),
                split="test",
                sample_index_csv=args.test_sample_csv,
            )
        return dataset, prepare_guided_batch

    if args.generalization == "year":
        dataset = ClimateDownscaleDataset_Baseline(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            years=args.test_years,
            split="test",
            sample_index_csv=args.test_sample_csv,
        )
    else:
        dataset = ClimateDownscaleDataset_Baseline_V2(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            target_station_ids=read_station_ids(args.test_meta_csv),
            split="test",
            sample_index_csv=args.test_sample_csv,
        )
    return dataset, prepare_baseline_batch


def build_model(args):
    from open_source.unet_models import BasicUNet, ResAttUNet

    if args.model_type == "basic-unet":
        return BasicUNet(in_channels=args.input_channels, out_channels=args.output_channels)
    return ResAttUNet(
        in_channels=args.input_channels,
        out_channels=args.output_channels,
        use_attention=args.use_attention,
        use_sft=args.use_sft,
        pretrained_backbone=args.pretrained_backbone,
    )


def read_station_ids(path):
    import pandas as pd

    return pd.read_csv(path)["Station_ID"].tolist()


def load_checkpoint(model, model_path, device):
    import torch

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint was not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        state_dict = normalize_checkpoint_keys(state_dict)
        model.load_state_dict(state_dict)


def normalize_checkpoint_keys(state_dict):
    normalized = {}
    replacements = (
        (".att.", ".attention."),
        (".W_g.", ".w_g."),
        (".W_x.", ".w_x."),
        (".W_dem.", ".w_dem."),
        (".PriorNet.", ".prior_net."),
        (".SFT_scale_conv.", ".scale."),
        (".SFT_shift_conv.", ".shift."),
    )
    for key, value in state_dict.items():
        new_key = key.removeprefix("module.")
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        normalized[new_key] = value
    return normalized


def main():
    args, _ = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from open_source.test.metrics import evaluate_model, print_metric_table, write_metric_csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, prepare_batch = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(args)
    load_checkpoint(model, args.model_path, device)

    print(f"Dataset: {args.dataset_type} | Generalization: {args.generalization}")
    print(f"Samples: {len(dataset)} | Model: {args.model_path}")
    results = evaluate_model(
        model=model,
        test_loader=loader,
        device=device,
        prepare_batch_fn=prepare_batch,
        output_channels=args.output_channels,
        print_interval=args.print_interval,
    )
    print_metric_table(results)
    if args.save_csv:
        write_metric_csv(args.save_csv, results)
        print(f"Saved metrics to: {args.save_csv}")


if __name__ == "__main__":
    main()
