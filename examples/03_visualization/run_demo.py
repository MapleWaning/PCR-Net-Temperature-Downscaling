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
    ensure_pretrained_checkpoint,
    missing_modules,
    run_python,
    write_json,
)
from pcr_downscaling.demo_cases import add_demo_case_args, resolve_demo_case  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run visualization smoke demo or launch the real visualizer.")
    add_demo_case_args(parser)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_dataset = args.dataset
    selected_dataset, dataset_artifact = ensure_demo_dataset(ROOT, args.dataset, args.data_root)
    case = resolve_demo_case(ROOT, selected_dataset, args.data_root, args.split_mode, args.version)
    layout = case.layout
    output_dir = ROOT / "outputs" / "demos" / "03_visualization" / case.name / case.split_mode
    pretrained_artifact = ensure_pretrained_checkpoint(ROOT, case.split_mode, args.model_path)
    model_path, model_source = select_model_path(args.model_path, case.name, case.split_mode)
    demo2_guidance = (
        ROOT
        / "outputs"
        / "demos"
        / "02_train_and_test"
        / case.name
        / case.split_mode
        / "catboost_inference"
        / "catboost_lst"
    )
    if demo2_guidance.exists():
        guidance_dir = demo2_guidance
        guidance_source = "demo2"
    elif case.split_mode == "spatial" and (layout.catboost_inference / "catboost_lst_spatial").exists():
        guidance_dir = layout.catboost_inference / "catboost_lst_spatial"
        guidance_source = "prepared-spatial"
    else:
        guidance_dir = layout.catboost_inference / "catboost_lst"
        guidance_source = "prepared"
    visual_meta_csv = layout.model_inputs / "splits" / "all_selected_meta.csv"
    if not visual_meta_csv.exists():
        visual_meta_csv = case.test_meta_csv
    target_sample = select_target_sample(case.all_sample_csv)
    missing = missing_modules(["torch", "torchvision", "matplotlib", "h5py", "pandas"])
    command = [
        "src/open_source/mapping/visualize.py",
        "--n-samples",
        "1",
        "--model-path",
        str(model_path),
        "--dataset-type",
        "pcr-net",
        "--model-type",
        "resattunet",
        "--generalization",
        case.generalization,
        "--test-years",
        "2023",
        "--test-meta-csv",
        str(visual_meta_csv),
        "--test-sample-csv",
        str(case.all_sample_csv),
        "--station-id",
        target_sample["station_id"],
        "--date",
        target_sample["date"],
        "--static-path",
        str(layout.model_inputs / "static.npy"),
        "--truth-path",
        str(layout.model_inputs / "truth.npy"),
        "--tbase-dir",
        str(layout.physical / "t_base"),
        "--guidance-dir",
        str(guidance_dir),
        "--output-dir",
        str(output_dir / "maps"),
        "--label",
        f"demo-pcr-net-{case.name}-{case.split_mode}",
        "--dpi",
        "120",
    ]

    status = "ready" if not missing and model_path.exists() and guidance_dir.exists() else "failed"
    real_output = None
    if status == "ready":
        result = run_python(ROOT, command)
        status = "ok" if result.returncode == 0 else "failed"
        real_output = result.stdout
    else:
        real_output = (
            "Missing dependencies, model checkpoint, or CatBoost guidance. "
            "Demo 03 requires pretrained/Demo 02 weights and CatBoost guidance from prepared data or Demo 02."
        )

    summary = {
        "demo": "03_visualization",
        "status": status,
        "requested_dataset": requested_dataset,
        "dataset": case.name,
        "split_mode": case.split_mode,
        "execute_requested": args.execute,
        "artifact_downloads": {
            "dataset": completed_process_summary(dataset_artifact),
            "pretrained": completed_process_summary(pretrained_artifact),
        },
        "missing_modules": missing,
        "input_data_root": str(layout.root),
        "target_sample": target_sample,
        "model_path": str(model_path),
        "model_source": model_source,
        "model_exists": model_path.exists(),
        "guidance_dir": str(guidance_dir),
        "guidance_source": guidance_source,
        "guidance_exists": guidance_dir.exists(),
        "real_command": command_line(["python", *command]),
        "real_output": real_output,
    }
    output = write_json(output_dir / "summary.json", summary)
    print(f"Demo 03 status: {status}")
    print(f"Summary: {output}")
    if status == "failed":
        raise SystemExit(1)


def select_target_sample(sample_csv: Path) -> dict[str, str]:
    preferred = {"station_id": "36982099999", "date": "2023-07-20"}
    if not sample_csv.exists():
        return preferred

    with sample_csv.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if str(row.get("Station_ID")) == preferred["station_id"] and row.get("target_date") == preferred["date"]:
            return preferred
    if rows:
        first = rows[0]
        return {
            "station_id": str(first["Station_ID"]),
            "date": str(first.get("target_date") or first.get("date")),
        }
    return preferred


def select_model_path(explicit_model_path: str | None, dataset: str, split_mode: str) -> tuple[Path, str]:
    if explicit_model_path:
        path = Path(explicit_model_path)
        return path, "explicit"

    pretrained = pretrained_checkpoint(split_mode)
    if pretrained.exists():
        return pretrained, "pretrained"

    demo_checkpoint = find_demo_checkpoint(dataset, split_mode)
    if demo_checkpoint is not None:
        return demo_checkpoint, "demo2"

    return pretrained, "missing"


def pretrained_checkpoint(split_mode: str) -> Path:
    name = "pcr-time.pth" if split_mode == "temporal" else "pcr-spatial.pth"
    return ROOT / "assets" / "pretrained" / "pcr_net" / name


def find_demo_checkpoint(dataset: str, split_mode: str) -> Path | None:
    run_root = ROOT / "outputs" / "demos" / "02_train_and_test" / dataset / split_mode / "runs"
    candidates = list(run_root.glob("*/best_model.pth"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return None


if __name__ == "__main__":
    main()
