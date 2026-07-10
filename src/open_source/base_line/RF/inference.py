import argparse
from pathlib import Path

import joblib

from open_source.spatial_inference import import_project_config, run_full_domain_inference


def parse_years(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    config = import_project_config()
    parser = argparse.ArgumentParser(description="Run full-domain inference for the no-LST Random Forest baseline.")
    parser.add_argument("--model-path", default=str(Path("open_source") / "base_line" / "RF" / "rf_no_lst.joblib"))
    parser.add_argument("--output-dir", default=getattr(config, "PURE_RF_BASE"))
    parser.add_argument("--years", default=",".join(map(str, getattr(config, "FULL_YEAR"))))
    parser.add_argument("--station-csv", default=getattr(config, "STATION_META"))
    parser.add_argument("--static-npy-path", default=getattr(config, "STATIC_PATH"))
    parser.add_argument("--tbase-dir", default=getattr(config, "GFS_DIR"))
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load(args.model_path)
    run_full_domain_inference(
        model=model,
        model_name="RandomForest-No-LST",
        output_prefix="rf_t2m",
        include_lst=False,
        years=parse_years(args.years),
        station_csv=args.station_csv,
        static_npy_path=args.static_npy_path,
        tbase_dir=args.tbase_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
