from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pcr_downscaling.demo import (  # noqa: E402
    command_line,
    completed_process_summary,
    ensure_demo_dataset,
    missing_modules,
    run_python,
    write_json,
)
from pcr_downscaling.demo_cases import add_demo_case_args, resolve_demo_case  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the core PCR-Net ablation training matrix.")
    add_demo_case_args(parser)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def guidance_dir_for(case, layout):
    spatial_guidance = layout.catboost_inference / "catboost_lst_spatial"
    if case.split_mode == "spatial" and spatial_guidance.exists():
        return spatial_guidance
    return layout.catboost_inference / "catboost_lst"


def common_training_args(case, layout, output_root: Path, run_name: str, epochs: int) -> list[str]:
    return [
        "--generalization",
        case.generalization,
        "--epochs",
        str(epochs),
        "--batch-size",
        "16",
        "--num-workers",
        "0",
        "--no-pretrained-backbone",
        "--tbase-dir",
        str(layout.physical / "t_base"),
        "--train-years",
        case.train_years_arg,
        "--val-years",
        case.val_years_arg,
        "--train-meta-csv",
        str(case.train_meta_csv),
        "--val-meta-csv",
        str(case.val_meta_csv),
        "--train-sample-csv",
        str(case.train_sample_csv),
        "--val-sample-csv",
        str(case.val_sample_csv),
        "--static-path",
        str(layout.model_inputs / "static.npy"),
        "--truth-path",
        str(layout.model_inputs / "truth.npy"),
        "--output-dir",
        str(output_root / "runs"),
        "--run-name",
        run_name,
    ]


def ablation_matrix(case, layout, output_root: Path, epochs: int) -> list[dict[str, object]]:
    guidance_dir = guidance_dir_for(case, layout)
    no_attention_command = [
        "src/open_source/ablation/no_attention.py",
        *common_training_args(
            case,
            layout,
            output_root,
            f"demo-no-attention-{case.name}-{case.split_mode}",
            epochs,
        ),
        "--guidance-dir",
        str(guidance_dir),
    ]
    no_gradient_loss_command = [
        "src/open_source/ablation/no_gradient_loss.py",
        *common_training_args(
            case,
            layout,
            output_root,
            f"demo-no-gradient-loss-{case.name}-{case.split_mode}",
            epochs,
        ),
    ]
    common_required = {
        "static": layout.model_inputs / "static.npy",
        "truth": layout.model_inputs / "truth.npy",
        "tbase": layout.physical / "t_base",
    }
    return [
        {
            "name": "no_attention",
            "description": "PCR-Net without attention and SFT, still using CatBoost guidance.",
            "command": no_attention_command,
            "required": {**common_required, "guidance": guidance_dir},
        },
        {
            "name": "no_gradient_loss",
            "description": "PCR-Net architecture trained with pure MSE, without gradient guidance loss.",
            "command": no_gradient_loss_command,
            "required": common_required,
        },
    ]


def run_ablation(root: Path, item: dict[str, object], missing: list[str], smoke_only: bool) -> dict[str, object]:
    required = item["required"]
    exists = {name: path.exists() for name, path in required.items()}
    missing_inputs = [name for name, present in exists.items() if not present]
    result = {
        "name": item["name"],
        "description": item["description"],
        "required": {name: str(path) for name, path in required.items()},
        "exists": exists,
        "missing_inputs": missing_inputs,
        "command": command_line(["python", *item["command"]]),
        "status": "smoke-only",
        "run_output": None,
    }
    if smoke_only:
        return result
    if missing:
        result["status"] = "missing-dependencies"
        return result
    if not all(exists.values()):
        result["status"] = "missing-inputs"
        if "guidance" in missing_inputs:
            result["run_output"] = "Missing CatBoost inference. Run Demo 02 on this dataset first."
        return result

    completed = run_python(root, item["command"])
    result["run_output"] = completed.stdout
    result["status"] = "ok" if completed.returncode == 0 else "failed"
    return result


def summarize_status(results: list[dict[str, object]], smoke_only: bool) -> str:
    if smoke_only:
        return "smoke-only"
    statuses = [result["status"] for result in results]
    if all(status == "ok" for status in statuses):
        return "ok"
    if any(status == "failed" for status in statuses):
        return "failed"
    return "failed"


def main():
    args = parse_args()
    requested_dataset = args.dataset
    selected_dataset, dataset_artifact = ensure_demo_dataset(ROOT, args.dataset, args.data_root)
    case = resolve_demo_case(ROOT, selected_dataset, args.data_root, args.split_mode, args.version)
    layout = case.layout
    missing = missing_modules(["torch", "torchvision", "h5py", "pandas"])
    output_root = ROOT / "outputs" / "demos" / "04_ablation_training" / case.name / case.split_mode
    epochs = 3 if case.name == "mini_case" else 1
    matrix = ablation_matrix(case, layout, output_root, epochs)
    results = [run_ablation(ROOT, item, missing, args.smoke_only) for item in matrix]
    status = summarize_status(results, args.smoke_only)

    summary = {
        "demo": "04_ablation_training",
        "status": status,
        "requested_dataset": requested_dataset,
        "dataset": case.name,
        "split_mode": case.split_mode,
        "smoke_only": args.smoke_only,
        "artifact_downloads": {
            "dataset": completed_process_summary(dataset_artifact),
        },
        "missing_modules": missing,
        "input_data_root": str(layout.root),
        "epochs": epochs,
        "sample_csvs": {
            "train": str(case.train_sample_csv),
            "val": str(case.val_sample_csv),
        },
        "matrix": results,
    }
    output = write_json(output_root / "summary.json", summary)
    print(f"Demo 04 status: {status}")
    print(f"Summary: {output}")
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
