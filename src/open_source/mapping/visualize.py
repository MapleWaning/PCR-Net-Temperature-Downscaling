import argparse
import importlib
import os
import random
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args():
    config = importlib.import_module("config")
    parser = argparse.ArgumentParser(description="Render n sample-level refinement maps.")
    parser.add_argument("--dataset-type", choices=["pcr-net", "baseline"], default="pcr-net")
    parser.add_argument("--model-type", choices=["resattunet", "basic-unet"], default="resattunet")
    parser.add_argument("--generalization", choices=["year", "station"], default="year")
    parser.add_argument("--n-samples", type=int, default=getattr(config, "PICNUM", 5))
    parser.add_argument("--sample-indices", type=comma_ints, default=None)
    parser.add_argument("--station-id", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--seed", type=int, default=getattr(config, "seed", 42))
    parser.add_argument("--model-path", default=getattr(config, "FINAL_MODEL_PATH", getattr(config, "MODEL_PATH", None)))
    parser.add_argument("--tbase-dir", default=None)
    parser.add_argument("--guidance-dir", default=None)
    parser.add_argument("--static-path", default=config.STATIC_PATH)
    parser.add_argument("--truth-path", default=config.TRUTH_PATH)
    parser.add_argument("--test-meta-csv", default=config.TEST_META_CSV)
    parser.add_argument("--test-sample-csv", default=None)
    parser.add_argument("--test-years", type=comma_ints, default=config.TEST_YEAR)
    parser.add_argument("--input-channels", type=int, default=config.TEST_INPUT_CHANNELS)
    parser.add_argument("--output-channels", type=int, default=config.TEST_OUTPUT_CHANNELS)
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", default=str(Path("open_source") / "mapping" / "outputs"))
    parser.add_argument("--label", default=getattr(config, "EXPERIMENT", getattr(config, "LABEL", "model")))
    parser.add_argument("--diff-range", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=300)
    return finalize_args(parser.parse_args(), config), config


def finalize_args(args, config):
    if args.tbase_dir is None:
        args.tbase_dir = config.TBASE_ADVANCE_DIR if args.dataset_type == "pcr-net" else config.GFS_DIR
    if args.guidance_dir is None:
        args.guidance_dir = getattr(config, "CB_BASE", getattr(config, "CB_SPATIAL_BASE", None))
    return args


def comma_ints(value):
    if value is None or isinstance(value, list):
        return value
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_dataset(args):
    from open_source.unet_datasets import (
        ClimateDownscaleDataset_Baseline,
        ClimateDownscaleDataset_Baseline_V2,
        ClimateDownscaleDataset_V3,
        ClimateDownscaleDataset_V4,
    )

    if args.dataset_type == "pcr-net":
        if args.generalization == "year":
            return ClimateDownscaleDataset_V3(
                tbase_dir=args.tbase_dir,
                rf_base_dir=args.guidance_dir,
                static_npy_path=args.static_path,
                truth_npy_path=args.truth_path,
                years=args.test_years,
                split="test",
                sample_index_csv=args.test_sample_csv,
            )
        return ClimateDownscaleDataset_V4(
            tbase_dir=args.tbase_dir,
            rf_base_dir=args.guidance_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            target_station_ids=read_station_ids(args.test_meta_csv),
            split="test",
            sample_index_csv=args.test_sample_csv,
        )

    if args.generalization == "year":
        return ClimateDownscaleDataset_Baseline(
            tbase_dir=args.tbase_dir,
            static_npy_path=args.static_path,
            truth_npy_path=args.truth_path,
            years=args.test_years,
            split="test",
            sample_index_csv=args.test_sample_csv,
        )
    return ClimateDownscaleDataset_Baseline_V2(
        tbase_dir=args.tbase_dir,
        static_npy_path=args.static_path,
        truth_npy_path=args.truth_path,
        target_station_ids=read_station_ids(args.test_meta_csv),
        split="test",
        sample_index_csv=args.test_sample_csv,
    )


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


def select_sample_indices(dataset, args):
    if args.sample_indices is not None:
        return [validate_index(idx, len(dataset)) for idx in args.sample_indices[: args.n_samples]]

    matches = find_matching_indices(dataset, args.station_id, args.date)
    if matches:
        return matches[: args.n_samples]
    if args.station_id or args.date:
        raise ValueError("No samples matched the requested station/date filters.")

    n_samples = min(args.n_samples, len(dataset))
    rng = random.Random(args.seed)
    return rng.sample(range(len(dataset)), n_samples)


def find_matching_indices(dataset, station_id=None, date=None):
    if station_id is None and date is None:
        return []
    matches = []
    import pandas as pd

    target_date = pd.Timestamp(date).strftime("%Y-%m-%d") if date else None
    iterator = dataset.index_map.items() if isinstance(dataset.index_map, dict) else enumerate(dataset.index_map)
    for idx, info in iterator:
        if station_id is not None and str(info.get("station_id")) != str(station_id):
            continue
        if target_date is not None and sample_date(info) != target_date:
            continue
        matches.append(idx)
    return matches


def validate_index(idx, dataset_size):
    if idx < 0 or idx >= dataset_size:
        raise IndexError(f"Sample index {idx} is outside dataset range [0, {dataset_size - 1}]")
    return idx


def visualize_sample(dataset, model, sample_idx, sample_number, output_dir, device, label, diff_range, dpi):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    info = dataset.index_map[sample_idx]
    station_id = info.get("station_id", "unknown")
    date_string = sample_date(info)
    data, target, mask = dataset[sample_idx]

    base = data["base"].unsqueeze(0).to(device)
    dem = data["dem"].unsqueeze(0).to(device)
    alb = data["alb"].unsqueeze(0).to(device)
    time_feat = data["time"].unsqueeze(0).to(device)
    lu = data["lu"].unsqueeze(0).to(device)
    lu_onehot = F.one_hot(lu, num_classes=10).permute(0, 3, 1, 2).float()
    inputs = torch.cat([base, dem, alb, time_feat, lu_onehot], dim=1)

    model.eval()
    with torch.no_grad():
        output = model(inputs)

    dem_m = denormalize_dem(dem[0, 0]).detach().cpu()
    base_temp = denormalize_temp(base[0, 0]).detach().cpu()
    pred_temp = denormalize_temp(output[0, 0]).detach().cpu()
    correction = pred_temp - base_temp

    height, width = target.shape[1], target.shape[2]
    station_row, station_col = height // 2, width // 2
    station_temp = None
    if mask[0, station_row, station_col] > 0:
        station_temp = denormalize_temp(target[0, station_row, station_col]).item()
        base_at_station = base_temp[station_row, station_col].item()
        pred_at_station = pred_temp[station_row, station_col].item()

    temp_min = min(base_temp.min().item(), pred_temp.min().item())
    temp_max = max(base_temp.max().item(), pred_temp.max().item())
    if temp_max - temp_min < 0.1:
        temp_min -= 0.5
        temp_max += 0.5

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    im0 = axes[0].imshow(dem_m, cmap="terrain")
    axes[0].set_title(f"Terrain Elevation (DEM)\nRange: {dem_m.min():.0f} to {dem_m.max():.0f} m", fontsize=12)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Elevation (m)")

    im1 = axes[1].imshow(base_temp, cmap="Spectral_r", vmin=temp_min, vmax=temp_max)
    title = f"Input Base Temperature\nMean: {base_temp.mean():.1f} deg C"
    if station_temp is not None:
        title += f"\nStation base: {base_at_station:.2f} deg C"
    axes[1].set_title(title, fontsize=12)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Temperature (deg C)")

    im2 = axes[2].imshow(pred_temp, cmap="Spectral_r", vmin=temp_min, vmax=temp_max)
    title = f"Model Refined Temperature\nMean: {pred_temp.mean():.1f} deg C"
    if station_temp is not None:
        title += f"\nStation pred: {pred_at_station:.2f} deg C | target: {station_temp:.2f} deg C"
    axes[2].set_title(title, fontsize=12)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Temperature (deg C)")

    im3 = axes[3].imshow(correction, cmap="bwr", vmin=-diff_range, vmax=diff_range)
    axes[3].set_title(
        f"Correction (Prediction - Base)\nRange: {correction.min():.2f} to {correction.max():.2f} deg C",
        fontsize=12,
    )
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label="Correction (deg C)")

    marker_colors = ("red", "red", "red", "black")
    for ax, color in zip(axes, marker_colors):
        ax.scatter(station_col, station_row, c=color, s=100, marker="x", linewidths=3)
        ax.set_xticks([])
        ax.set_yticks([])
    if station_temp is not None:
        axes[0].text(
            station_col,
            station_row + 8,
            f"Target: {station_temp:.1f} deg C",
            color="red",
            fontsize=10,
            ha="center",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    fig.suptitle(f"{label} refinement | Station: {station_id} | Date: {date_string} | Sample: {sample_idx}", fontsize=16)
    fig.tight_layout()
    output_path = output_dir / make_output_name(label, sample_number, sample_idx, station_id, date_string)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def sample_date(info):
    import pandas as pd

    date_value = info.get("date", info.get("date_obj"))
    if date_value is not None:
        return pd.Timestamp(date_value).strftime("%Y-%m-%d")
    year = int(info["year"])
    day_idx = int(info.get("day_idx", info.get("day_idx_in_year", 0)))
    return (pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_idx)).strftime("%Y-%m-%d")


def denormalize_temp(tensor):
    from open_source.unet_datasets import TEMP_MEAN, TEMP_STD

    return tensor * TEMP_STD + TEMP_MEAN


def denormalize_dem(tensor):
    from open_source.unet_datasets import DEM_MEAN, DEM_STD

    return tensor * DEM_STD + DEM_MEAN


def make_output_name(label, sample_number, sample_idx, station_id, date_string):
    safe_label = sanitize(label)
    safe_station = sanitize(str(station_id))
    return f"{safe_label}_{sample_number:03d}_idx{sample_idx}_station{safe_station}_{date_string}.png"


def sanitize(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    return value.strip("-") or "sample"


def main():
    args, _ = parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args)
    model = build_model(args).to(device)
    load_checkpoint(model, args.model_path, device)
    indices = select_sample_indices(dataset, args)

    print(f"Rendering {len(indices)} samples to {output_dir}")
    for number, sample_idx in enumerate(indices, start=1):
        output_path = visualize_sample(
            dataset=dataset,
            model=model,
            sample_idx=sample_idx,
            sample_number=number,
            output_dir=output_dir,
            device=device,
            label=args.label,
            diff_range=args.diff_range,
            dpi=args.dpi,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
