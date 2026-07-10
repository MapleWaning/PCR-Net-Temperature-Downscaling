import importlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from open_source.data_prepare.build_features import resolve_static_features, resolve_tbase_path
from open_source.data_prepare.schema import FEATURES_NO_LST, FEATURES_WITH_LST


def import_project_config():
    try:
        return importlib.import_module("config")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import config.py. Run from the project root or pass all paths explicitly."
        ) from exc


def aggregate_base_image(tbase_raw):
    return np.stack(
        [
            np.mean(tbase_raw, axis=0),
            np.max(tbase_raw, axis=0),
            np.min(tbase_raw, axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def build_feature_frame(base_image, static_features, year_idx, month_idx, day_of_year, include_lst, lst_image=None):
    height, width = base_image.shape[1:]
    n_pixels = height * width
    dem = static_features["dem"]
    land_use = static_features["land_use"][year_idx]
    albedo = static_features["albedo"][year_idx, month_idx]
    doy_sin = np.full(n_pixels, np.sin(2 * np.pi * day_of_year / 365.0), dtype=np.float32)
    doy_cos = np.full(n_pixels, np.cos(2 * np.pi * day_of_year / 365.0), dtype=np.float32)

    arrays = [
        base_image[0].reshape(-1),
        base_image[1].reshape(-1),
        base_image[2].reshape(-1),
    ]
    if include_lst:
        arrays.extend(
            [
                lst_image[0].reshape(-1),
                lst_image[1].reshape(-1),
                lst_image[2].reshape(-1),
            ]
        )
    arrays.extend(
        [
            dem[0].reshape(-1),
            dem[1].reshape(-1),
            dem[2].reshape(-1),
            dem[3].reshape(-1),
            land_use.reshape(-1),
            albedo.reshape(-1),
            doy_sin,
            doy_cos,
        ]
    )

    feature_columns = FEATURES_WITH_LST if include_lst else FEATURES_NO_LST
    frame = pd.DataFrame(np.column_stack(arrays), columns=feature_columns)
    frame["land_use"] = frame["land_use"].astype(int)
    return frame


def run_full_domain_inference(
    model,
    model_name,
    output_prefix,
    include_lst,
    years,
    station_csv,
    static_npy_path,
    tbase_dir,
    output_dir,
    lst_dir=None,
    static_start_year=2008,
    sample_csv=None,
):
    print(f"Running {model_name} full-domain inference...")
    static_data = np.load(static_npy_path, allow_pickle=True).item()
    meta_df = pd.read_csv(station_csv)
    station_ids = meta_df["Station_ID"].astype(str).values
    sample_lookup = build_sample_lookup(sample_csv) if sample_csv else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        print(f"Processing year: {year}")
        tbase_path = resolve_tbase_path(tbase_dir, year)
        lst_path = Path(lst_dir) / f"lst_{year}.h5" if include_lst and lst_dir else None
        if include_lst and (lst_path is None or not lst_path.exists()):
            print(f"Skipping {year}: missing LST file")
            continue

        output_path = output_dir / f"{output_prefix}_{year}.h5"
        with h5py.File(tbase_path, "r") as f_base:
            f_lst = h5py.File(lst_path, "r") if include_lst else None
            try:
                total_days = f_base["data"].shape[0]
                _, _, _, height, width = f_base["data"].shape

                with h5py.File(output_path, "w") as f_out:
                    output = f_out.create_dataset(
                        "data",
                        shape=(total_days, len(station_ids), 3, height, width),
                        dtype="float32",
                        chunks=(1, 1, 3, height, width),
                        compression="lzf",
                        fillvalue=np.nan,
                    )

                    if sample_lookup is None:
                        day_station_pairs = [
                            (day, station_idx)
                            for day in range(total_days)
                            for station_idx in range(len(station_ids))
                        ]
                    else:
                        day_station_pairs = [
                            (day_idx, station_idx)
                            for day_idx, station_idx in sample_lookup.get(year, [])
                            if day_idx < total_days and station_idx < len(station_ids)
                        ]

                    for day, station_idx in tqdm(day_station_pairs, desc=f"Year {year}"):
                        current_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day)
                        month_idx = current_date.month - 1
                        day_of_year = current_date.dayofyear

                        station_id = station_ids[station_idx]
                        static_features = resolve_static_features(static_data, station_id)
                        if static_features is None:
                            continue

                        year_idx = min(
                            max(0, year - static_start_year),
                            len(static_features["albedo"]) - 1,
                        )

                        # The model predicts residuals, which are added back to the ERA5 aggregate.
                        base_image = aggregate_base_image(f_base["data"][day, station_idx])
                        lst_image = f_lst["data"][day, station_idx] if include_lst else None
                        features = build_feature_frame(
                            base_image,
                            static_features,
                            year_idx,
                            month_idx,
                            day_of_year,
                            include_lst,
                            lst_image,
                        )
                        residuals = model.predict(features)
                        residual_map = residuals.T.reshape(3, height, width)
                        output[day, station_idx] = base_image + residual_map
            finally:
                if f_lst is not None:
                    f_lst.close()

        print(f"Saved {output_path}")


def build_sample_lookup(sample_csv):
    df = pd.read_csv(sample_csv)
    lookup = {}
    for _, row in df.iterrows():
        date = pd.Timestamp(row.get("target_date", row.get("date")))
        station_idx = int(row["output_station_index"])
        day_idx = int(row.get("target_day_idx", date.dayofyear - 1))
        lookup.setdefault(int(date.year), set()).add((day_idx, station_idx))
    return {year: sorted(items) for year, items in lookup.items()}
