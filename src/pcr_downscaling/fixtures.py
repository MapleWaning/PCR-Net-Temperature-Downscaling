from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


GLOBAL_START_DATE = pd.Timestamp("2008-01-01")
SAMPLE_MONTH_DAYS = ((1, 15), (4, 15), (7, 20), (11, 10))


def create_minimal_model_fixture(
    root: str | Path,
    year: int = 2008,
    include_synthetic_guidance: bool = False,
    years: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Path | list[Path]]:
    base = Path(root).resolve()
    standard = base / "standard"
    physical = base / "physical"
    guidance = base / "catboost_inference" / "catboost_lst"
    model_inputs = base / "model_inputs"

    years = sorted({int(item) for item in (years or [year])})
    last_year = max(years)
    station_id = "DEMO00001"
    total_global_days = (pd.Timestamp(f"{last_year}-12-31") - GLOBAL_START_DATE).days + 1
    size = 128

    metadata = pd.DataFrame(
        [
            {
                "Station_ID": station_id,
                "Station_Name": "Demo Station",
                "Latitude": 43.25,
                "Longitude": 76.90,
            }
        ]
    )
    metadata_path = standard / "metadata" / "high-quality-meta.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)

    sample_rows = build_sample_rows(station_id, years)
    write_split_files(base, metadata, sample_rows, years)

    weather = np.full((1, total_global_days, 3), np.nan, dtype=np.float32)
    weather_path = standard / "gsod" / "weather_data.npy"
    weather_path.parent.mkdir(parents=True, exist_ok=True)

    static_path = model_inputs / "static.npy"
    truth_path = model_inputs / "truth.npy"
    fixture_manifest_path = base / "fixture_manifest.json"
    for path in (static_path, truth_path, weather_path, fixture_manifest_path, base / "manifest.json"):
        path.parent.mkdir(parents=True, exist_ok=True)

    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, size), np.linspace(-1.0, 1.0, size), indexing="ij")
    elevation = (1200.0 + 400.0 * np.exp(-2.0 * (xx**2 + yy**2))).astype(np.float32)
    slope = np.full((size, size), 8.0, dtype=np.float32)
    aspect_sin = np.sin(np.pi * xx).astype(np.float32)
    aspect_cos = np.cos(np.pi * yy).astype(np.float32)
    dem = np.stack([elevation, slope, aspect_sin, aspect_cos], axis=0)
    static_years = max(1, last_year - GLOBAL_START_DATE.year + 1)
    albedo = np.full((static_years, 12, size, size), 0.22, dtype=np.float32)
    land_use = np.full((static_years, size, size), 2, dtype=np.uint8)
    static = {station_id: {"dem": dem, "albedo": albedo, "land_use": land_use}}
    np.save(static_path, static)

    truth = np.full((1, total_global_days, 3), np.nan, dtype=np.float32)
    era5_paths = []
    lst_paths = []
    tbase_paths = []
    guidance_paths = []
    spatial = (0.8 * xx - 0.5 * yy).astype(np.float32)

    for current_year in years:
        days = len(pd.date_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31", freq="D"))
        era5_path = standard / "era5" / f"era5_t2m_{current_year}.h5"
        lst_path = standard / "lst" / f"lst_{current_year}.h5"
        tbase_path = physical / "t_base" / f"t_base_advanced_{current_year}.h5"
        guidance_path = guidance / f"cb_t2m_{current_year}.h5"
        for path in (era5_path, lst_path, tbase_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        if include_synthetic_guidance:
            guidance_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(era5_path, "w") as f_era5, h5py.File(lst_path, "w") as f_lst, h5py.File(tbase_path, "w") as f_base:
            era5_ds = f_era5.create_dataset(
                "data",
                shape=(days, 1, 4, size, size),
                dtype="float32",
                chunks=(1, 1, 4, size, size),
                compression="lzf",
                fillvalue=np.nan,
            )
            tbase_ds = f_base.create_dataset(
                "data",
                shape=(days, 1, 4, size, size),
                dtype="float32",
                chunks=(1, 1, 4, size, size),
                compression="lzf",
                fillvalue=np.nan,
            )
            lst_ds = f_lst.create_dataset(
                "data",
                shape=(days, 1, 3, size, size),
                dtype="float32",
                chunks=(1, 1, 3, size, size),
                compression="lzf",
                fillvalue=np.nan,
            )
            guide_ds = None
            if include_synthetic_guidance:
                f_guide = h5py.File(guidance_path, "w")
                guide_ds = f_guide.create_dataset(
                    "data",
                    shape=(days, 1, 3, size, size),
                    dtype="float32",
                    chunks=(1, 1, 3, size, size),
                    compression="lzf",
                    fillvalue=np.nan,
                )
            else:
                f_guide = None
            try:
                for sample_date in sample_dates_for_year(current_year):
                    day = sample_date.dayofyear - 1
                    global_idx = (sample_date - GLOBAL_START_DATE).days
                    seasonal = np.float32(8.0 + 12.0 * np.sin(2.0 * np.pi * day / days))
                    raw = np.stack(
                        [
                            seasonal + spatial - 1.5,
                            seasonal + spatial,
                            seasonal + spatial + 1.5,
                            seasonal + spatial + 0.5,
                        ],
                        axis=0,
                    ).astype(np.float32)
                    era5_ds[day, 0] = raw
                    advanced = raw + np.float32(0.15 * np.cos(2.0 * np.pi * day / days))
                    tbase_ds[day, 0] = advanced
                    base_3ch = np.stack([advanced.mean(axis=0), advanced.max(axis=0), advanced.min(axis=0)], axis=0)
                    lst = (base_3ch + np.float32(0.4 * np.cos(2.0 * np.pi * day / days))).astype(np.float32)
                    lst_ds[day, 0] = lst
                    correction = np.float32(0.25 * np.sin(2.0 * np.pi * day / days))
                    target = (base_3ch + correction + np.float32(0.05 * (lst - base_3ch))).astype(np.float32)
                    if guide_ds is not None:
                        guide_ds[day, 0] = target
                    truth[0, global_idx] = target[:, size // 2, size // 2]
                    weather[0, global_idx] = target[:, size // 2, size // 2]
            finally:
                if f_guide is not None:
                    f_guide.close()

        era5_paths.append(era5_path)
        lst_paths.append(lst_path)
        tbase_paths.append(tbase_path)
        if include_synthetic_guidance:
            guidance_paths.append(guidance_path)

    np.save(weather_path, weather)
    np.save(truth_path, truth)
    write_manifest_files(
        base=base,
        years=years,
        sample_rows=sample_rows,
        truth_shape=truth.shape,
        static_station_count=len(static),
        include_synthetic_guidance=include_synthetic_guidance,
    )

    guidance_note = (
        "Synthetic guidance tensor with CatBoost-compatible filename; not generated by a trained CatBoost model."
        if include_synthetic_guidance
        else "Not generated by Demo 01. Demo 02 trains CatBoost and writes this product into catboost_inference/."
    )
    fixture_manifest_path.write_text(
        "{\n"
        '  "era5": "Synthetic ERA5-like base tensor for baseline model demos.",\n'
        '  "lst": "Synthetic LST-like tensor for the LST CatBoost demo.",\n'
        '  "tbase": "Synthetic advanced physical base tensor for PCR-Net demos.",\n'
        f'  "catboost_guidance": "{guidance_note}"\n'
        "}\n",
        encoding="utf-8",
    )

    outputs: dict[str, Path | list[Path]] = {
        "root": base,
        "metadata": metadata_path,
        "static": static_path,
        "truth": truth_path,
        "era5": era5_paths[0],
        "era5_files": era5_paths,
        "lst": lst_paths[0],
        "lst_files": lst_paths,
        "tbase": tbase_paths[0],
        "tbase_files": tbase_paths,
        "weather": weather_path,
        "manifest": base / "manifest.json",
        "fixture_manifest": fixture_manifest_path,
    }
    if include_synthetic_guidance:
        outputs["guidance"] = guidance_paths[0]
        outputs["guidance_files"] = guidance_paths
    return outputs


def sample_dates_for_year(year: int) -> list[pd.Timestamp]:
    return [pd.Timestamp(year=year, month=month, day=day) for month, day in SAMPLE_MONTH_DAYS]


def build_sample_rows(station_id: str, years: list[int]) -> list[dict[str, object]]:
    rows = []
    for current_year in years:
        for sample_date in sample_dates_for_year(current_year):
            day_idx = sample_date.dayofyear - 1
            global_day_idx = (sample_date - GLOBAL_START_DATE).days
            rows.append(
                {
                    "sample_id": f"{station_id}_{sample_date.date().isoformat()}",
                    "Station_ID": station_id,
                    "output_station_index": 0,
                    "source_station_index": 0,
                    "year": current_year,
                    "target_date": sample_date.date().isoformat(),
                    "source_date": sample_date.date().isoformat(),
                    "target_day_idx": day_idx,
                    "source_day_idx": day_idx,
                    "target_global_day_idx": global_day_idx,
                    "source_global_day_idx": global_day_idx,
                    "substituted": False,
                }
            )
    return rows


def temporal_splits(years: list[int]) -> dict[str, list[int]]:
    if len(years) >= 3:
        return {"train": [years[0]], "val": [years[1]], "test": years[2:]}
    if len(years) == 2:
        return {"train": [years[0]], "val": [years[1]], "test": [years[1]]}
    return {"train": years, "val": years, "test": years}


def write_split_files(base: Path, metadata: pd.DataFrame, sample_rows: list[dict[str, object]], years: list[int]) -> None:
    split_dir = base / "model_inputs" / "splits"
    report_dir = base / "model_inputs" / "reports"
    split_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "train_meta.csv",
        "val_meta.csv",
        "test_meta.csv",
        "all_selected_meta.csv",
        "spatial_train_meta.csv",
        "spatial_val_meta.csv",
        "spatial_test_meta.csv",
    ):
        metadata.to_csv(split_dir / name, index=False)

    pd.DataFrame(sample_rows).to_csv(split_dir / "all_selected_samples.csv", index=False)
    year_splits = temporal_splits(years)
    for split_name, split_years in year_splits.items():
        rows = [row for row in sample_rows if int(row["year"]) in split_years]
        write_split_sample_csv(split_dir / f"temporal_{split_name}_samples.csv", rows, "temporal", split_name)

    for split_name in ("train", "val", "test"):
        write_split_sample_csv(split_dir / f"spatial_{split_name}_samples.csv", sample_rows, "spatial", split_name)

    write_json(split_dir / "temporal_years.json", year_splits)
    write_json(
        split_dir / "station_selection.json",
        {
            "selected_stations_in_output_order": metadata["Station_ID"].astype(str).tolist(),
            "demo1_samples": [[row["Station_ID"], row["target_date"]] for row in sample_rows],
            "spatial_split_stations": {
                "train": metadata["Station_ID"].astype(str).tolist(),
                "val": metadata["Station_ID"].astype(str).tolist(),
                "test": metadata["Station_ID"].astype(str).tolist(),
            },
            "note": "Demo 01 fallback uses one station for all spatial splits so every demo can run from the same layout.",
        },
    )
    pd.DataFrame(columns=list(sample_rows[0].keys())).to_csv(report_dir / "replacement_report.csv", index=False)


def write_split_sample_csv(path: Path, rows: list[dict[str, object]], strategy: str, split: str) -> None:
    payload = [dict({"split_strategy": strategy, "split": split}, **row) for row in rows]
    pd.DataFrame(payload).to_csv(path, index=False)


def write_manifest_files(
    base: Path,
    years: list[int],
    sample_rows: list[dict[str, object]],
    truth_shape: tuple[int, ...],
    static_station_count: int,
    include_synthetic_guidance: bool,
) -> None:
    write_json(
        base / "manifest.json",
        {
            "dataset": "demo1_model_ready",
            "sample_universe_count": len(sample_rows),
            "samples": [[row["Station_ID"], row["target_date"]] for row in sample_rows],
            "years": years,
            "replacement_count": 0,
            "layout_root": str(base),
            "source": "Demo 01 generated model-ready dataset",
            "catboost_guidance": (
                "synthetic fixture guidance"
                if include_synthetic_guidance
                else "not generated by Demo 01; Demo 02 writes trained CatBoost inference into this dataset"
            ),
        },
    )
    h5_shapes = {
        "era5": [[len(pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")), 1, 4, 128, 128] for year in years],
        "lst": [[len(pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")), 1, 3, 128, 128] for year in years],
        "tbase": [[len(pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")), 1, 4, 128, 128] for year in years],
    }
    if include_synthetic_guidance:
        h5_shapes["catboost_lst"] = [
            [len(pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")), 1, 3, 128, 128]
            for year in years
        ]
    write_json(
        base / "validation_summary.json",
        {
            "metadata_rows": 1,
            "sample_universe_count": len(sample_rows),
            "truth_shape": list(truth_shape),
            "static_station_count": static_station_count,
            "replacement_count": 0,
            "h5_products": h5_shapes,
        },
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
