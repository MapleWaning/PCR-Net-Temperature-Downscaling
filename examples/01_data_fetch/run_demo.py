from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pcr_downscaling.demo import run_python, write_json  # noqa: E402
from pcr_downscaling.fixtures import create_minimal_model_fixture  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal data fetch/preprocess demo.")
    parser.add_argument("--proxy", default=None, help="Optional HTTP/HTTPS proxy, for example http://127.0.0.1:7890")
    return parser.parse_args()


def prepare_fixture(base: Path) -> None:
    metadata_path = base / "processed" / "standard" / "metadata" / "high-quality-meta.csv"
    raw_path = base / "raw" / "gsod" / "station_data_merged" / "DEMO00001_2020_2020.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "Station_ID": "DEMO00001",
                "Station_Name": "Demo Station",
                "Latitude": 43.25,
                "Longitude": 76.90,
            }
        ]
    ).to_csv(metadata_path, index=False)

    pd.DataFrame(
        [
            {"DATE": "2020-01-01", "TEMP": 32.0, "MAX": 40.0, "MIN": 25.0},
            {"DATE": "2020-01-02", "TEMP": 34.0, "MAX": 42.0, "MIN": 28.0},
        ]
    ).to_csv(raw_path, index=False)


def main():
    args = parse_args()
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    base = ROOT / "outputs" / "demos" / "01_data_fetch"
    prepare_fixture(base)
    config = ROOT / "configs" / "demos" / "data_fetch_minimal.yaml"

    plan = run_python(ROOT, ["scripts/run_pipeline.py", "plan", "--config", str(config), "--modules", "gsod"])
    run = run_python(ROOT, ["scripts/run_pipeline.py", "run", "--config", str(config), "--modules", "gsod"])
    model_ready = create_minimal_model_fixture(
        base / "model_ready",
        years=(2008, 2009, 2010),
        include_synthetic_guidance=False,
    )

    summary = {
        "demo": "01_data_fetch",
        "status": "ok" if plan.returncode == 0 and run.returncode == 0 else "failed",
        "sample_count": 2,
        "sample_limit": 100,
        "config": str(config),
        "outputs": {
            "metadata": str(base / "processed" / "standard" / "metadata" / "high-quality-meta.csv"),
            "weather": str(base / "processed" / "standard" / "gsod" / "weather_data.npy"),
            "manifest": str(base / "processed" / "standard" / "reports" / "manifest.json"),
        },
        "model_ready_root": str(base / "model_ready"),
        "model_ready_outputs": {
            "manifest": str(model_ready["manifest"]),
            "static": str(model_ready["static"]),
            "truth": str(model_ready["truth"]),
            "era5_raw": [str(path) for path in model_ready["era5_files"]],
            "lst": [str(path) for path in model_ready["lst_files"]],
            "tbase": [str(path) for path in model_ready["tbase_files"]],
            "splits": str(base / "model_ready" / "model_inputs" / "splits"),
            "replacement_report": str(base / "model_ready" / "model_inputs" / "reports" / "replacement_report.csv"),
        },
        "model_ready_notes": {
            "catboost_guidance": "Not produced by Demo 01. Demo 02 trains CatBoost and writes catboost_inference into this model_ready dataset.",
        },
        "plan_output": plan.stdout,
        "run_output": run.stdout,
    }
    output = write_json(base / "summary.json", summary)
    print(f"Demo 01 status: {summary['status']}")
    print(f"Summary: {output}")
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
