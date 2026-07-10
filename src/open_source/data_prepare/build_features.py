import argparse
import importlib
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from open_source.data_prepare.schema import (
    FEATURE_SETS,
    FEATURES_NO_LST,
    FEATURES_WITH_LST,
    TARGET_COLUMNS,
)


TARGETS_WITH_LST = {"catboost_lst"}
TARGETS_NO_LST = {"baseline_cb", "baseline_rf"}
DEFAULT_FILE_NAMES = {
    "train": "train.parquet",
    "val": "val.parquet",
    "test": "test.parquet",
}


def import_project_config():
    try:
        return importlib.import_module("config")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import config.py. Run from the project root or pass all paths explicitly."
        ) from exc


def include_lst_for_target(target):
    if target in TARGETS_WITH_LST:
        return True
    if target in TARGETS_NO_LST:
        return False
    raise ValueError(f"Unsupported target: {target}")


def resolve_tbase_path(tbase_dir, year):
    candidates = [
        f"t_base_advanced_{year}.h5",
        f"era5_t2m_{year}.h5",
        f"t_base_{year}.h5",
    ]
    for name in candidates:
        path = Path(tbase_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No T_base file found for {year} in {tbase_dir}")


def resolve_static_features(static_data, station_id):
    for key in (station_id, str(station_id)):
        if key in static_data:
            return static_data[key]
    try:
        int_key = int(station_id)
    except (TypeError, ValueError):
        int_key = None
    if int_key is not None and int_key in static_data:
        return static_data[int_key]
    return None


def aggregate_base_values(tbase_values):
    return np.array(
        [
            np.mean(tbase_values),
            np.max(tbase_values),
            np.min(tbase_values),
        ],
        dtype=np.float32,
    )


def build_row(
    station_id,
    year,
    day_of_year,
    base_values,
    static_features,
    month_idx,
    year_idx,
    truth_value,
    lst_values=None,
):
    doy_sin = np.sin(2 * np.pi * day_of_year / 365.0)
    doy_cos = np.cos(2 * np.pi * day_of_year / 365.0)
    dem = static_features["dem"]
    land_use = static_features["land_use"][year_idx, 64, 64]
    albedo = static_features["albedo"][year_idx, month_idx, 64, 64]

    row = {
        "Station_ID": str(station_id),
        "Year": int(year),
        "DOY": int(day_of_year),
        "era5_mean": float(base_values[0]),
        "era5_max": float(base_values[1]),
        "era5_min": float(base_values[2]),
        "dem_ch1": float(dem[0, 64, 64]),
        "dem_ch2": float(dem[1, 64, 64]),
        "dem_ch3": float(dem[2, 64, 64]),
        "dem_ch4": float(dem[3, 64, 64]),
        "land_use": int(land_use),
        "albedo": float(albedo),
        "doy_sin": float(doy_sin),
        "doy_cos": float(doy_cos),
        "Residual_Mean": float(truth_value[0] - base_values[0]),
        "Residual_Max": float(truth_value[1] - base_values[1]),
        "Residual_Min": float(truth_value[2] - base_values[2]),
    }

    if lst_values is not None:
        # LST columns are inserted at the same positions used by the CatBoost LST model.
        row["lst_mean"] = float(lst_values[0])
        row["lst_max"] = float(lst_values[1])
        row["lst_min"] = float(lst_values[2])

    return row


def ordered_dataframe(rows, include_lst):
    feature_cols = FEATURES_WITH_LST if include_lst else FEATURES_NO_LST
    return pd.DataFrame(rows, columns=["Station_ID", "Year", "DOY"] + feature_cols + TARGET_COLUMNS)


def write_parquet(rows, output_path, include_lst):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = ordered_dataframe(rows, include_lst)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")


def station_index_map(static_data, meta_csv_path):
    meta_df = pd.read_csv(meta_csv_path)
    target_stations = set(meta_df["Station_ID"].astype(str).tolist())
    all_station_ids = list(static_data.keys())
    return {
        str(station_id): idx
        for idx, station_id in enumerate(all_station_ids)
        if str(station_id) in target_stations
    }


def extract_station_rows(
    meta_csv_path,
    tbase_dir,
    static_npy_path,
    truth_npy_path,
    years,
    include_lst,
    lst_dir=None,
    global_start_date=pd.Timestamp("2008-01-01"),
    static_start_year=2008,
):
    static_data = np.load(static_npy_path, allow_pickle=True).item()
    truth_data = np.load(truth_npy_path)
    station_map = station_index_map(static_data, meta_csv_path)

    if not station_map:
        raise ValueError(f"No stations from {meta_csv_path} matched the static data.")

    rows = []
    for year in years:
        print(f"Processing station split year: {year}")
        tbase_path = resolve_tbase_path(tbase_dir, year)
        lst_path = Path(lst_dir) / f"lst_{year}.h5" if include_lst and lst_dir else None
        if include_lst and (lst_path is None or not lst_path.exists()):
            print(f"Skipping {year}: missing LST file")
            continue

        with h5py.File(tbase_path, "r") as f_base:
            f_lst = h5py.File(lst_path, "r") if include_lst else None
            try:
                date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
                total_days = min(len(date_range), f_base["data"].shape[0])

                for day in tqdm(range(total_days), desc=f"Days in {year}"):
                    current_date = date_range[day]
                    global_day_idx = (current_date - global_start_date).days
                    month_idx = current_date.month - 1
                    day_of_year = current_date.dayofyear

                    for station_id, station_idx in station_map.items():
                        truth_value = truth_data[station_idx, global_day_idx]
                        if np.isnan(truth_value).any() or (truth_value < -100).any() or (truth_value > 100).any():
                            continue

                        tbase_center = f_base["data"][day, station_idx, :, 64, 64]
                        base_values = aggregate_base_values(tbase_center)

                        lst_values = None
                        if include_lst:
                            lst_values = f_lst["data"][day, station_idx, :, 64, 64]
                            if np.isnan(lst_values).any() or lst_values[0] < -100:
                                continue

                        static_features = resolve_static_features(static_data, station_id)
                        if static_features is None:
                            continue

                        year_idx = min(
                            max(0, year - static_start_year),
                            len(static_features["albedo"]) - 1,
                        )
                        rows.append(
                            build_row(
                                station_id,
                                year,
                                day_of_year,
                                base_values,
                                static_features,
                                month_idx,
                                year_idx,
                                truth_value,
                                lst_values,
                            )
                        )
            finally:
                if f_lst is not None:
                    f_lst.close()

    return rows


def build_station_split_parquet(
    meta_csv_path,
    tbase_dir,
    static_npy_path,
    truth_npy_path,
    output_parquet,
    years,
    include_lst,
    lst_dir=None,
    global_start_date=pd.Timestamp("2008-01-01"),
    static_start_year=2008,
):
    rows = extract_station_rows(
        meta_csv_path=meta_csv_path,
        tbase_dir=tbase_dir,
        static_npy_path=static_npy_path,
        truth_npy_path=truth_npy_path,
        years=years,
        include_lst=include_lst,
        lst_dir=lst_dir,
        global_start_date=global_start_date,
        static_start_year=static_start_year,
    )
    write_parquet(rows, output_parquet, include_lst)


def prepare_station_splits(
    train_meta_csv,
    val_meta_csv,
    test_meta_csv,
    tbase_dir,
    static_npy_path,
    truth_npy_path,
    output_dir,
    years,
    include_lst,
    lst_dir=None,
):
    output_dir = Path(output_dir)
    split_to_meta = {
        "train": train_meta_csv,
        "val": val_meta_csv,
        "test": test_meta_csv,
    }
    for split_name, meta_path in split_to_meta.items():
        build_station_split_parquet(
            meta_csv_path=meta_path,
            tbase_dir=tbase_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            output_parquet=output_dir / DEFAULT_FILE_NAMES[split_name],
            years=years,
            include_lst=include_lst,
            lst_dir=lst_dir,
        )


def year_split_name(year, train_years, val_years, test_years):
    if year in train_years:
        return "train"
    if year in val_years:
        return "val"
    if year in test_years:
        return "test"
    return None


def prepare_year_splits(
    tbase_dir,
    static_npy_path,
    truth_npy_path,
    output_dir,
    years,
    train_years,
    val_years,
    test_years,
    include_lst,
    lst_dir=None,
    global_start_date=pd.Timestamp("2008-01-01"),
    static_start_year=2008,
):
    static_data = np.load(static_npy_path, allow_pickle=True).item()
    truth_data = np.load(truth_npy_path)
    station_ids = list(static_data.keys())
    rows_by_split = {"train": [], "val": [], "test": []}

    for year in years:
        split_name = year_split_name(year, set(train_years), set(val_years), set(test_years))
        if split_name is None:
            continue

        print(f"Processing year split year: {year} -> {split_name}")
        tbase_path = resolve_tbase_path(tbase_dir, year)
        lst_path = Path(lst_dir) / f"lst_{year}.h5" if include_lst and lst_dir else None
        if include_lst and (lst_path is None or not lst_path.exists()):
            print(f"Skipping {year}: missing LST file")
            continue

        with h5py.File(tbase_path, "r") as f_base:
            f_lst = h5py.File(lst_path, "r") if include_lst else None
            try:
                date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
                total_days = min(len(date_range), f_base["data"].shape[0])

                for day in tqdm(range(total_days), desc=f"Days in {year}"):
                    current_date = date_range[day]
                    global_day_idx = (current_date - global_start_date).days
                    month_idx = current_date.month - 1
                    day_of_year = current_date.dayofyear

                    for station_idx, station_id in enumerate(station_ids):
                        truth_value = truth_data[station_idx, global_day_idx]
                        if np.isnan(truth_value).any():
                            continue

                        tbase_center = f_base["data"][day, station_idx, :, 64, 64]
                        base_values = aggregate_base_values(tbase_center)

                        lst_values = None
                        if include_lst:
                            lst_values = f_lst["data"][day, station_idx, :, 64, 64]
                            if np.isnan(lst_values).any() or lst_values[0] < -100:
                                continue

                        static_features = resolve_static_features(static_data, station_id)
                        if static_features is None:
                            continue

                        year_idx = min(
                            max(0, year - static_start_year),
                            len(static_features["albedo"]) - 1,
                        )
                        rows_by_split[split_name].append(
                            build_row(
                                station_id,
                                year,
                                day_of_year,
                                base_values,
                                static_features,
                                month_idx,
                                year_idx,
                                truth_value,
                                lst_values,
                            )
                        )
            finally:
                if f_lst is not None:
                    f_lst.close()

    output_dir = Path(output_dir)
    for split_name, rows in rows_by_split.items():
        write_parquet(rows, output_dir / DEFAULT_FILE_NAMES[split_name], include_lst)


def default_paths_from_config(target, split):
    config = import_project_config()
    include_lst = include_lst_for_target(target)
    output_dir = Path("open_source") / "data_prepare" / target / split
    return {
        "years": getattr(config, "FULL_YEAR"),
        "train_years": getattr(config, "TRAIN_YEAR"),
        "val_years": getattr(config, "VERIFY_YEAR"),
        "test_years": getattr(config, "TEST_YEAR"),
        "train_meta_csv": getattr(config, "TRAIN_META_CSV"),
        "val_meta_csv": getattr(config, "VAL_META_CSV"),
        "test_meta_csv": getattr(config, "TEST_META_CSV"),
        "tbase_dir": getattr(config, "TBASE_ADVANCE_DIR") if include_lst else getattr(config, "GFS_DIR"),
        "lst_dir": getattr(config, "LST_PATH") if include_lst else None,
        "static_npy_path": getattr(config, "STATIC_PATH"),
        "truth_npy_path": getattr(config, "TRUTH_PATH"),
        "output_dir": output_dir,
    }


def parse_years(value):
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Build parquet feature datasets.")
    parser.add_argument("--target", choices=sorted(FEATURE_SETS), required=True)
    parser.add_argument("--split", choices=["station", "year"], required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--years")
    parser.add_argument("--train-years")
    parser.add_argument("--val-years")
    parser.add_argument("--test-years")
    parser.add_argument("--train-meta-csv")
    parser.add_argument("--val-meta-csv")
    parser.add_argument("--test-meta-csv")
    parser.add_argument("--tbase-dir")
    parser.add_argument("--lst-dir")
    parser.add_argument("--static-npy-path")
    parser.add_argument("--truth-npy-path")
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = default_paths_from_config(args.target, args.split)
    include_lst = include_lst_for_target(args.target)

    years = parse_years(args.years) or defaults["years"]
    output_dir = args.output_dir or defaults["output_dir"]
    tbase_dir = args.tbase_dir or defaults["tbase_dir"]
    lst_dir = args.lst_dir or defaults["lst_dir"]
    static_npy_path = args.static_npy_path or defaults["static_npy_path"]
    truth_npy_path = args.truth_npy_path or defaults["truth_npy_path"]

    if args.split == "station":
        prepare_station_splits(
            train_meta_csv=args.train_meta_csv or defaults["train_meta_csv"],
            val_meta_csv=args.val_meta_csv or defaults["val_meta_csv"],
            test_meta_csv=args.test_meta_csv or defaults["test_meta_csv"],
            tbase_dir=tbase_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            output_dir=output_dir,
            years=years,
            include_lst=include_lst,
            lst_dir=lst_dir,
        )
    else:
        prepare_year_splits(
            tbase_dir=tbase_dir,
            static_npy_path=static_npy_path,
            truth_npy_path=truth_npy_path,
            output_dir=output_dir,
            years=years,
            train_years=parse_years(args.train_years) or defaults["train_years"],
            val_years=parse_years(args.val_years) or defaults["val_years"],
            test_years=parse_years(args.test_years) or defaults["test_years"],
            include_lst=include_lst,
            lst_dir=lst_dir,
        )


if __name__ == "__main__":
    main()
