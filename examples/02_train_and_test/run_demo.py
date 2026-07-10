from __future__ import annotations

import argparse
import csv
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
from pcr_downscaling.demo_cases import add_demo_case_args, is_demo1_model_ready_case, resolve_demo_case  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run PCR-Net training and testing on a prepared demo dataset.")
    add_demo_case_args(parser)
    parser.add_argument("--smoke-only", action="store_true", help="Only write command wiring without executing training.")
    return parser.parse_args()


def newest_checkpoint(run_root: Path) -> Path | None:
    candidates = list(run_root.glob("*/best_model.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def newest_history(run_root: Path) -> Path | None:
    candidates = list(run_root.glob("*/training_history.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def strictly_decreases(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    return all(curr < prev for prev, curr in zip(values, values[1:]))


def loss_trend(history_path: Path | None) -> dict[str, object]:
    if history_path is None or not history_path.exists():
        return {"checked": False, "history_csv": str(history_path) if history_path else None}
    with history_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]
    return {
        "checked": True,
        "history_csv": str(history_path),
        "epochs": len(rows),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_decreases_each_epoch": strictly_decreases(train_losses),
        "val_decreases_each_epoch": strictly_decreases(val_losses),
    }


def guidance_output_dir(case, output_root: Path) -> Path:
    if is_demo1_model_ready_case(case):
        folder = "catboost_lst_spatial" if case.split_mode == "spatial" else "catboost_lst"
        return case.layout.catboost_inference / folder
    return output_root / "catboost_inference" / "catboost_lst"


def main():
    args = parse_args()
    requested_dataset = args.dataset
    selected_dataset, dataset_artifact = ensure_demo_dataset(ROOT, args.dataset, args.data_root)
    case = resolve_demo_case(ROOT, selected_dataset, args.data_root, args.split_mode, args.version)
    layout = case.layout
    missing = missing_modules(["torch", "torchvision", "h5py", "pandas", "catboost", "pyarrow", "sklearn"])
    output_root = ROOT / "outputs" / "demos" / "02_train_and_test" / case.name / case.split_mode
    run_root = output_root / "runs"
    metrics_csv = output_root / "metrics.csv"
    parquet_dir = output_root / "parquet" / "catboost_lst" / case.feature_split
    catboost_model = output_root / "catboost" / "catboost_lst.cbm"
    catboost_output_dir = guidance_output_dir(case, output_root)
    expected_guidance = [catboost_output_dir / f"cb_t2m_{year}.h5" for year in case.years]
    pcr_epochs = 3 if case.name == "mini_case" else 1

    build_features_command = [
        "src/open_source/data_prepare/build_features.py",
        "--target",
        "catboost_lst",
        "--split",
        case.feature_split,
        "--output-dir",
        str(parquet_dir),
        "--years",
        case.years_arg,
        "--tbase-dir",
        str(layout.physical / "t_base"),
        "--lst-dir",
        str(layout.standard / "lst"),
        "--static-npy-path",
        str(layout.model_inputs / "static.npy"),
        "--truth-npy-path",
        str(layout.model_inputs / "truth.npy"),
    ]
    if case.feature_split == "station":
        build_features_command.extend(
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
        build_features_command.extend(
            [
                "--train-years",
                case.train_years_arg,
                "--val-years",
                case.val_years_arg,
                "--test-years",
                case.test_years_arg,
            ]
        )
    catboost_train_command = [
        "src/open_source/CatBoost/train.py",
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
        str(parquet_dir / "train.parquet"),
        "--val-parquet",
        str(parquet_dir / "val.parquet"),
        "--test-parquet",
        str(parquet_dir / "test.parquet"),
        "--model-save-path",
        str(catboost_model),
    ]
    catboost_inference_command = [
        "src/open_source/CatBoost/inference.py",
        "--model-path",
        str(catboost_model),
        "--output-dir",
        str(catboost_output_dir),
        "--years",
        case.years_arg,
        "--station-csv",
        str(layout.standard / "metadata" / "high-quality-meta.csv"),
        "--static-npy-path",
        str(layout.model_inputs / "static.npy"),
        "--tbase-dir",
        str(layout.physical / "t_base"),
        "--lst-dir",
        str(layout.standard / "lst"),
        "--sample-csv",
        str(case.all_sample_csv),
    ]
    pcr_train_command = [
        "src/open_source/PCR-Net/train.py",
        "--generalization",
        case.generalization,
        "--epochs",
        str(pcr_epochs),
        "--batch-size",
        "16",
        "--num-workers",
        "0",
        "--no-pretrained-backbone",
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
        str(layout.physical / "t_base"),
        "--guidance-dir",
        str(catboost_output_dir),
        "--output-dir",
        str(run_root),
        "--run-name",
        f"demo-pcr-net-{case.name}-{case.split_mode}",
    ]
    eval_command = [
        "src/open_source/test/evaluate.py",
        "--dataset-type",
        "pcr-net",
        "--model-type",
        "resattunet",
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
        str(layout.physical / "t_base"),
        "--guidance-dir",
        str(catboost_output_dir),
        "--save-csv",
        str(metrics_csv),
    ]

    build_features_output = None
    catboost_train_output = None
    catboost_inference_output = None
    pcr_train_output = None
    eval_output = None
    checkpoint = None
    trend = {"checked": False}
    status = "smoke-only"
    if not missing and not args.smoke_only:
        build = run_python(ROOT, build_features_command)
        build_features_output = build.stdout
        if build.returncode != 0:
            status = "failed"
        else:
            cb_train = run_python(ROOT, catboost_train_command)
            catboost_train_output = cb_train.stdout
            if cb_train.returncode != 0:
                status = "failed"
            else:
                cb_infer = run_python(ROOT, catboost_inference_command)
                catboost_inference_output = cb_infer.stdout
                if cb_infer.returncode != 0 or not all(path.exists() for path in expected_guidance):
                    status = "failed"
                else:
                    pcr_train = run_python(ROOT, pcr_train_command)
                    pcr_train_output = pcr_train.stdout
                    status = "trained" if pcr_train.returncode == 0 else "failed"
        if status == "trained":
            checkpoint = newest_checkpoint(run_root)
            trend = loss_trend(newest_history(run_root))
            if checkpoint:
                eval_command.extend(["--model-path", str(checkpoint)])
                evaluation = run_python(ROOT, eval_command)
                eval_output = evaluation.stdout
                status = "ok" if evaluation.returncode == 0 else "failed"
            else:
                status = "failed"
                eval_output = "No best_model.pth was produced."

    summary = {
        "demo": "02_train_and_test",
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
            "all": str(case.all_sample_csv),
        },
        "years": {
            "all": case.years,
            "train": case.train_years,
            "val": case.val_years,
            "test": case.test_years,
        },
        "pcr_epochs": pcr_epochs,
        "loss_trend": trend,
        "input_contract": {
            "metadata": str(layout.standard / "metadata" / "high-quality-meta.csv"),
            "weather": str(layout.standard / "gsod" / "weather_data.npy"),
            "era5": [str(layout.standard / "era5" / f"era5_t2m_{year}.h5") for year in case.years],
            "lst": [str(layout.standard / "lst" / f"lst_{year}.h5") for year in case.years],
            "static": str(layout.model_inputs / "static.npy"),
            "truth": str(layout.model_inputs / "truth.npy"),
            "tbase": [str(layout.physical / "t_base" / f"t_base_advanced_{year}.h5") for year in case.years],
        },
        "input_exists": {
            "metadata": (layout.standard / "metadata" / "high-quality-meta.csv").exists(),
            "weather": (layout.standard / "gsod" / "weather_data.npy").exists(),
            "era5": all((layout.standard / "era5" / f"era5_t2m_{year}.h5").exists() for year in case.years),
            "lst": all((layout.standard / "lst" / f"lst_{year}.h5").exists() for year in case.years),
            "static": (layout.model_inputs / "static.npy").exists(),
            "truth": (layout.model_inputs / "truth.npy").exists(),
            "tbase": all((layout.physical / "t_base" / f"t_base_advanced_{year}.h5").exists() for year in case.years),
        },
        "model_stage_outputs": {
            "catboost_model": str(catboost_model),
            "catboost_guidance": [str(path) for path in expected_guidance],
            "pcr_checkpoint": str(checkpoint) if checkpoint else None,
            "metrics_csv": str(metrics_csv),
        },
        "build_features_command": command_line(["python", *build_features_command]),
        "catboost_train_command": command_line(["python", *catboost_train_command]),
        "catboost_inference_command": command_line(["python", *catboost_inference_command]),
        "pcr_train_command": command_line(["python", *pcr_train_command]),
        "eval_command": command_line(["python", *eval_command]),
        "catboost_model": str(catboost_model),
        "catboost_guidance": [str(path) for path in expected_guidance],
        "checkpoint": str(checkpoint) if checkpoint else None,
        "metrics_csv": str(metrics_csv),
        "build_features_output": build_features_output,
        "catboost_train_output": catboost_train_output,
        "catboost_inference_output": catboost_inference_output,
        "pcr_train_output": pcr_train_output,
        "eval_output": eval_output,
    }
    output = write_json(output_root / "summary.json", summary)
    print(f"Demo 02 status: {status}")
    print(f"Summary: {output}")
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
