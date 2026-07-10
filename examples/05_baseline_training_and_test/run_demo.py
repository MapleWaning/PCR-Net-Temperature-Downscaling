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
    parser = argparse.ArgumentParser(description="Run baseline RF, no-LST CatBoost, and Basic U-Net demos.")
    add_demo_case_args(parser)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def newest_checkpoint(run_root: Path) -> Path | None:
    candidates = list(run_root.glob("*/best_model.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_features_command(target: str, output_dir: Path, case) -> list[str]:
    layout = case.layout
    command = [
        "src/open_source/data_prepare/build_features.py",
        "--target",
        target,
        "--split",
        case.feature_split,
        "--output-dir",
        str(output_dir),
        "--years",
        case.years_arg,
        "--tbase-dir",
        str(layout.standard / "era5"),
        "--static-npy-path",
        str(layout.model_inputs / "static.npy"),
        "--truth-npy-path",
        str(layout.model_inputs / "truth.npy"),
    ]
    if case.feature_split == "station":
        command.extend(
            [
                "--train-meta-csv",
                str(case.train_meta_csv),
                "--val-meta-csv",
                str(case.val_meta_csv),
                "--test-meta-csv",
                str(case.test_meta_csv),
            ]
        )
    else:
        command.extend(
            [
                "--train-years",
                case.train_years_arg,
                "--val-years",
                case.val_years_arg,
                "--test-years",
                case.test_years_arg,
            ]
        )
    return command


def main():
    args = parse_args()
    requested_dataset = args.dataset
    selected_dataset, dataset_artifact = ensure_demo_dataset(ROOT, args.dataset, args.data_root)
    case = resolve_demo_case(ROOT, selected_dataset, args.data_root, args.split_mode, args.version)
    layout = case.layout
    output_root = ROOT / "outputs" / "demos" / "05_baseline_training_and_test" / case.name / case.split_mode
    cb_parquet = output_root / "parquet" / "baseline_cb" / case.feature_split
    rf_parquet = output_root / "parquet" / "baseline_rf" / case.feature_split
    cb_model = output_root / "models" / "catboost_no_lst.cbm"
    rf_model = output_root / "models" / "rf_no_lst.joblib"
    unet_run_root = output_root / "runs" / "baseline_unet"
    unet_metrics_csv = output_root / "baseline_unet_metrics.csv"

    missing = missing_modules(["torch", "torchvision", "h5py", "pandas", "catboost", "pyarrow", "sklearn", "joblib"])

    cb_build_command = build_features_command("baseline_cb", cb_parquet, case)
    rf_build_command = build_features_command("baseline_rf", rf_parquet, case)
    cb_train_command = [
        "src/open_source/base_line/CB/train.py",
        "--task-type",
        "CPU",
        "--iterations",
        "80",
        "--learning-rate",
        "0.08",
        "--depth",
        "4",
        "--early-stopping-rounds",
        "20",
        "--train-parquet",
        str(cb_parquet / "train.parquet"),
        "--val-parquet",
        str(cb_parquet / "val.parquet"),
        "--test-parquet",
        str(cb_parquet / "test.parquet"),
        "--model-save-path",
        str(cb_model),
    ]
    rf_train_command = [
        "src/open_source/base_line/RF/train.py",
        "--train-parquet",
        str(rf_parquet / "train.parquet"),
        "--val-parquet",
        str(rf_parquet / "val.parquet"),
        "--test-parquet",
        str(rf_parquet / "test.parquet"),
        "--model-save-path",
        str(rf_model),
        "--n-estimators",
        "20",
        "--max-depth",
        "6",
        "--n-jobs",
        "1",
        "--verbose",
        "0",
    ]
    unet_train_command = [
        "src/open_source/base_line/U-Net/train.py",
        "--generalization",
        case.generalization,
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--num-workers",
        "0",
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
        "--tbase-dir",
        str(layout.standard / "era5"),
        "--output-dir",
        str(unet_run_root),
        "--run-name",
        "demo-baseline-unet",
    ]
    unet_eval_command = [
        "src/open_source/test/evaluate.py",
        "--dataset-type",
        "baseline",
        "--model-type",
        "basic-unet",
        "--generalization",
        case.generalization,
        "--test-years",
        case.test_years_arg,
        "--test-meta-csv",
        str(case.test_meta_csv),
        "--test-sample-csv",
        str(case.test_sample_csv),
        "--batch-size",
        "16",
        "--num-workers",
        "0",
        "--static-path",
        str(layout.model_inputs / "static.npy"),
        "--truth-path",
        str(layout.model_inputs / "truth.npy"),
        "--tbase-dir",
        str(layout.standard / "era5"),
        "--save-csv",
        str(unet_metrics_csv),
    ]

    outputs = {
        "cb_build": None,
        "cb_train": None,
        "rf_build": None,
        "rf_train": None,
        "unet_train": None,
        "unet_eval": None,
    }
    status = "smoke-only"
    unet_checkpoint = None

    if not missing and not args.smoke_only:
        steps = [
            ("cb_build", cb_build_command),
            ("cb_train", cb_train_command),
            ("rf_build", rf_build_command),
            ("rf_train", rf_train_command),
            ("unet_train", unet_train_command),
        ]
        status = "ok"
        for name, command in steps:
            result = run_python(ROOT, command)
            outputs[name] = result.stdout
            if result.returncode != 0:
                status = "failed"
                break

        if status == "ok":
            unet_checkpoint = newest_checkpoint(unet_run_root)
            if unet_checkpoint:
                unet_eval_command.extend(["--model-path", str(unet_checkpoint)])
                result = run_python(ROOT, unet_eval_command)
                outputs["unet_eval"] = result.stdout
                status = "ok" if result.returncode == 0 else "failed"
            else:
                status = "failed"
                outputs["unet_eval"] = "No best_model.pth was produced."

    summary = {
        "demo": "05_baseline_training_and_test",
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
        "sample_csvs": {
            "train": str(case.train_sample_csv),
            "val": str(case.val_sample_csv),
            "test": str(case.test_sample_csv),
        },
        "years": {
            "all": case.years,
            "train": case.train_years,
            "val": case.val_years,
            "test": case.test_years,
        },
        "input_contract": {
            "era5": [str(layout.standard / "era5" / f"era5_t2m_{year}.h5") for year in case.years],
            "static": str(layout.model_inputs / "static.npy"),
            "truth": str(layout.model_inputs / "truth.npy"),
        },
        "forbidden_inputs": {
            "lst_used": False,
            "catboost_guidance_used": False,
        },
        "commands": {
            "cb_build": command_line(["python", *cb_build_command]),
            "cb_train": command_line(["python", *cb_train_command]),
            "rf_build": command_line(["python", *rf_build_command]),
            "rf_train": command_line(["python", *rf_train_command]),
            "unet_train": command_line(["python", *unet_train_command]),
            "unet_eval": command_line(["python", *unet_eval_command]),
        },
        "model_outputs": {
            "catboost_no_lst": str(cb_model),
            "rf_no_lst": str(rf_model),
            "baseline_unet_checkpoint": str(unet_checkpoint) if unet_checkpoint else None,
            "baseline_unet_metrics": str(unet_metrics_csv),
        },
        "outputs": outputs,
    }
    summary_path = write_json(output_root / "summary.json", summary)
    print(f"Demo 05 status: {status}")
    print(f"Summary: {summary_path}")
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
