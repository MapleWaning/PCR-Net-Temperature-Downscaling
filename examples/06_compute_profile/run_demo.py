from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pcr_downscaling.demo import (  # noqa: E402
    command_line,
    completed_process_summary,
    ensure_demo_dataset,
    missing_modules,
    write_json,
)
from pcr_downscaling.adapters.layout import DataLayout  # noqa: E402
from pcr_downscaling.demo_cases import DATASET_CHOICES, infer_years, resolve_packaged_or_demo1_case  # noqa: E402


@dataclass(frozen=True)
class ProfileCase:
    name: str
    root: Path
    layout: DataLayout
    years: list[int]
    all_sample_csv: Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int_list(value: str) -> list[int]:
    parsed = [positive_int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return parsed


def parse_channel_mults(value: str) -> tuple[int, ...]:
    return tuple(positive_int(part.strip()) for part in value.split(",") if part.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple inference compute profile for U-Net, PCR-Net, and Diffusion.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="smoke_case")
    parser.add_argument("--data-root", default=None, help="Override the prepared demo dataset root.")
    parser.add_argument("--num-samples", type=positive_int, default=16)
    parser.add_argument("--diffusion-steps", type=positive_int_list, default=positive_int_list("4,10,25,50,100"))
    parser.add_argument("--diffusion-base-channels", type=positive_int, default=64)
    parser.add_argument("--diffusion-channel-mults", type=parse_channel_mults, default=parse_channel_mults("1,2,4"))
    parser.add_argument("--diffusion-num-res-blocks", type=positive_int, default=2)
    parser.add_argument("--diffusion-dropout", type=float, default=0.0)
    parser.add_argument("--num-train-timesteps", type=positive_int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--macs-to-flops-factor", type=float, default=2.0)
    parser.add_argument("--amp-memory", action="store_true", help="Use AMP only for CUDA peak-memory profiling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_profile_case(root: Path, dataset: str, data_root: str | None) -> ProfileCase:
    if data_root:
        case_root = Path(data_root).expanduser().resolve()
        name = Path(data_root).name
        if not case_root.exists():
            raise FileNotFoundError(f"Demo dataset was not found: {case_root}")
    else:
        case_root, name = resolve_packaged_or_demo1_case(root, dataset)

    manifest_path = case_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        years = [int(year) for year in manifest.get("years", infer_years(case_root))]
    else:
        years = infer_years(case_root)

    return ProfileCase(
        name=name,
        root=case_root,
        layout=DataLayout.from_root(case_root),
        years=years,
        all_sample_csv=case_root / "model_inputs" / "splits" / "all_selected_samples.csv",
    )


def select_device(torch_module, requested: str):
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def build_input_dataset(case, tbase_dir: Path):
    from open_source.unet_datasets import ClimateDownscaleDataset_Baseline

    layout = case.layout
    return ClimateDownscaleDataset_Baseline(
        tbase_dir=tbase_dir,
        static_npy_path=layout.model_inputs / "static.npy",
        truth_npy_path=layout.model_inputs / "truth.npy",
        years=case.years,
        split="profile",
        sample_index_csv=case.all_sample_csv,
    )


def model_input_from_dataset_item(raw_item, torch_module, functional, device):
    sample, _, _ = raw_item
    base = sample["base"].unsqueeze(0).to(device)
    dem = sample["dem"].unsqueeze(0).to(device)
    albedo = sample["alb"].unsqueeze(0).to(device)
    time_feat = sample["time"].unsqueeze(0).to(device)
    land_use = sample["lu"].unsqueeze(0).to(device)
    land_use_onehot = functional.one_hot(land_use.long(), num_classes=10).permute(0, 3, 1, 2).float()
    inputs = torch_module.cat([base, dem, albedo, time_feat, land_use_onehot], dim=1)
    if tuple(inputs.shape[1:]) != (20, 128, 128):
        raise ValueError(f"Unexpected model input shape: {tuple(inputs.shape)}")
    return inputs


def collect_inputs(dataset, sample_count: int, torch_module, functional, device):
    return [model_input_from_dataset_item(dataset[index], torch_module, functional, device) for index in range(sample_count)]


def run_profile(args, case, output_root: Path) -> dict[str, object]:
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from open_source.unet_models import BasicUNet, ResAttUNet
    from pcr_downscaling.compute_profile import (
        DiffusionForwardWrapper,
        DiffusionUNet,
        SimpleDDIMScheduler,
        compute_flops_forward,
        count_parameters,
        peak_memory_diffusion_sampling,
        peak_memory_single_forward,
        state_dict_size_mb,
    )

    torch.manual_seed(args.seed)
    device = select_device(torch, args.device)
    output_root.mkdir(parents=True, exist_ok=True)
    layout = case.layout

    unet_dataset = build_input_dataset(case, layout.standard / "era5")
    pcr_dataset = build_input_dataset(case, layout.physical / "t_base")
    sample_count = min(args.num_samples, len(unet_dataset), len(pcr_dataset))
    if sample_count <= 0:
        raise ValueError("No samples are available for compute profiling.")

    unet_inputs = collect_inputs(unet_dataset, sample_count, torch, F, device)
    pcr_inputs = collect_inputs(pcr_dataset, sample_count, torch, F, device)
    x_unet_example = unet_inputs[0]
    x_pcr_example = pcr_inputs[0]
    target_channels = 3

    unet_model = BasicUNet(in_channels=20, out_channels=target_channels).to(device).eval()
    pcr_model = ResAttUNet(
        in_channels=20,
        out_channels=target_channels,
        use_attention=True,
        use_sft=True,
        pretrained_backbone=False,
    ).to(device).eval()
    diffusion_model = DiffusionUNet(
        in_channels=20 + target_channels,
        out_channels=target_channels,
        base_channels=args.diffusion_base_channels,
        channel_mults=args.diffusion_channel_mults,
        num_res_blocks=args.diffusion_num_res_blocks,
        dropout=args.diffusion_dropout,
    ).to(device).eval()

    y_t_example = torch.randn(
        x_pcr_example.shape[0],
        target_channels,
        x_pcr_example.shape[-2],
        x_pcr_example.shape[-1],
        device=device,
        dtype=x_pcr_example.dtype,
    )
    diffusion_input_example = torch.cat([y_t_example, x_pcr_example], dim=1)

    unet_params = count_parameters(unet_model)
    pcr_params = count_parameters(pcr_model)
    diffusion_params = count_parameters(diffusion_model)

    unet_size_mb = state_dict_size_mb(unet_model, output_root, "unet")
    pcr_size_mb = state_dict_size_mb(pcr_model, output_root, "pcr_net")
    diffusion_size_mb = state_dict_size_mb(diffusion_model, output_root, "diffusion")

    unet_flops_g = compute_flops_forward(unet_model, (x_unet_example,), device, args.macs_to_flops_factor)
    pcr_flops_g = compute_flops_forward(pcr_model, (x_pcr_example,), device, args.macs_to_flops_factor)
    diffusion_wrapper = DiffusionForwardWrapper(diffusion_model, timestep=args.num_train_timesteps // 2).to(device)
    diffusion_flops_g = compute_flops_forward(
        diffusion_wrapper,
        (diffusion_input_example,),
        device,
        args.macs_to_flops_factor,
    )

    unet_peak_mb = peak_memory_single_forward(unet_model, unet_inputs, device, use_amp=args.amp_memory)
    pcr_peak_mb = peak_memory_single_forward(pcr_model, pcr_inputs, device, use_amp=args.amp_memory)
    scheduler = SimpleDDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    diffusion_peak_by_steps = {
        steps: peak_memory_diffusion_sampling(
            diffusion_model,
            pcr_inputs,
            scheduler,
            steps,
            device,
            target_channels=target_channels,
            use_amp=args.amp_memory,
        )
        for steps in args.diffusion_steps
    }

    rows = [
        {
            "Model": "U-Net baseline",
            "Input shape": "[1, 20, 128, 128]",
            "Output shape": "[1, 3, 128, 128]",
            "Params": unet_params,
            "Params(M)": unet_params / 1e6,
            "FLOPs / forward(G)": unet_flops_g,
            "Steps / NFE": 1,
            "Effective FLOPs(G)": unet_flops_g,
            "Model size(MB)": unet_size_mb,
            "Peak memory(MB)": unet_peak_mb,
        },
        {
            "Model": "PCR-Net",
            "Input shape": "[1, 20, 128, 128]",
            "Output shape": "[1, 3, 128, 128]",
            "Params": pcr_params,
            "Params(M)": pcr_params / 1e6,
            "FLOPs / forward(G)": pcr_flops_g,
            "Steps / NFE": 1,
            "Effective FLOPs(G)": pcr_flops_g,
            "Model size(MB)": pcr_size_mb,
            "Peak memory(MB)": pcr_peak_mb,
        },
    ]
    for steps in args.diffusion_steps:
        rows.append(
            {
                "Model": f"Diffusion-style U-Net-{steps}",
                "Input shape": "[1, 23, 128, 128]",
                "Output shape": "[1, 3, 128, 128]",
                "Params": diffusion_params,
                "Params(M)": diffusion_params / 1e6,
                "FLOPs / forward(G)": diffusion_flops_g,
                "Steps / NFE": steps,
                "Effective FLOPs(G)": diffusion_flops_g * steps,
                "Model size(MB)": diffusion_size_mb,
                "Peak memory(MB)": diffusion_peak_by_steps[steps],
            }
        )

    csv_path = output_root / "complexity_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "amp_memory": args.amp_memory,
        "requested_samples": args.num_samples,
        "available_samples": {
            "unet": len(unet_dataset),
            "pcr_net": len(pcr_dataset),
        },
        "used_samples": sample_count,
        "sample_note": "sample count is capped by the selected demo dataset",
        "diffusion_config": {
            "in_channels": 23,
            "out_channels": target_channels,
            "base_channels": args.diffusion_base_channels,
            "channel_mults": list(args.diffusion_channel_mults),
            "num_res_blocks": args.diffusion_num_res_blocks,
            "dropout": args.diffusion_dropout,
            "num_train_timesteps": args.num_train_timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "sampling_steps": args.diffusion_steps,
        },
        "profile": {
            "macs_to_flops_factor": args.macs_to_flops_factor,
            "peak_memory_available": device.type == "cuda",
            "csv": str(csv_path),
            "rows": rows,
        },
    }


def main():
    args = parse_args()
    requested_dataset = args.dataset
    selected_dataset, dataset_artifact = ensure_demo_dataset(ROOT, args.dataset, args.data_root)
    case = resolve_profile_case(ROOT, selected_dataset, args.data_root)
    output_root = ROOT / "outputs" / "demos" / "06_compute_profile" / case.name
    missing = missing_modules(["torch", "torchvision", "h5py", "numpy", "pandas", "thop"])

    summary = {
        "demo": "06_compute_profile",
        "status": "pending",
        "requested_dataset": requested_dataset,
        "dataset": case.name,
        "artifact_downloads": {
            "dataset": completed_process_summary(dataset_artifact),
        },
        "missing_modules": missing,
        "input_data_root": str(case.layout.root),
        "inputs": {
            "unet_tbase_dir": str(case.layout.standard / "era5"),
            "pcr_tbase_dir": str(case.layout.physical / "t_base"),
            "static": str(case.layout.model_inputs / "static.npy"),
            "truth": str(case.layout.model_inputs / "truth.npy"),
            "profile_sample_csv": str(case.all_sample_csv),
        },
        "run_command": command_line(["python", *sys.argv]),
    }

    if missing:
        summary["status"] = "missing-dependencies"
        output = write_json(output_root / "summary.json", summary)
        print(f"Demo 06 status: {summary['status']}")
        print(f"Summary: {output}")
        raise SystemExit(1)

    try:
        summary.update(run_profile(args, case, output_root))
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"

    output = write_json(output_root / "summary.json", summary)
    print(f"Demo 06 status: {summary['status']}")
    print(f"Summary: {output}")
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
