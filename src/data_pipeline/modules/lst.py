from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm

from data_pipeline.common.dates import days_in_year
from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.station import normalize_station_id, read_metadata
from data_pipeline.modules.base import ModuleResult


class LstModule:
    name = "lst"
    channels = 3

    def plan(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv) if cfg.metadata_csv.exists() else pd.DataFrame()
        tif_dir = cfg.path(cfg.get("manual_inputs.lst_tif_dir"), cfg.raw_root / "lst" / "modis_exports")
        rows = []
        for _, station in metadata.iterrows():
            sid = normalize_station_id(station["Station_ID"])
            for year in range(cfg.start_year, cfg.end_year + 1):
                path = tif_dir / self._file_name(cfg, year, sid)
                rows.append({"Station_ID": sid, "Year": year, "file": str(path), "exists": path.exists()})

        report = cfg.output_root / "reports" / "lst_manual_tasks.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report, index=False)
        return ModuleResult(self.name, reports=[report], details={"missing_count": sum(not row["exists"] for row in rows)})

    def run(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv)
        tif_dir = cfg.path(cfg.get("manual_inputs.lst_tif_dir"), cfg.raw_root / "lst" / "modis_exports")
        output_dir = ensure_dir(cfg.path(cfg.get("lst.output_dir"), cfg.output_root / "lst"))
        patch = load_patch_data(cfg.path(cfg.get("lst.patch_csv"), "") if cfg.get("lst.patch_csv") else None)
        station_ids = [normalize_station_id(value) for value in metadata["Station_ID"]]
        written: list[Path] = []

        for year in range(cfg.start_year, cfg.end_year + 1):
            output = output_dir / f"lst_{year}.h5"
            full_shape = (days_in_year(year), len(station_ids), self.channels, cfg.image_size, cfg.image_size)
            with h5py.File(output, "w") as handle:
                dataset = handle.create_dataset(
                    "data",
                    shape=full_shape,
                    dtype="float32",
                    chunks=(1, 1, self.channels, cfg.image_size, cfg.image_size),
                    compression="lzf",
                )

                for station_idx, sid in enumerate(tqdm(station_ids, desc=f"LST {year}")):
                    tif_path = tif_dir / self._file_name(cfg, year, sid)
                    if not tif_path.exists():
                        dataset[:, station_idx] = np.zeros((full_shape[0], self.channels, cfg.image_size, cfg.image_size), dtype=np.float32)
                        continue
                    dataset[:, station_idx] = process_station_year_tif(tif_path, year, cfg.image_size, self.channels, patch.get((sid, year)))
            written.append(output)

        return ModuleResult(self.name, outputs=written, details={"station_count": len(station_ids)})

    def _file_name(self, cfg, year: int, station_id: str) -> str:
        pattern = str(cfg.get("lst.file_pattern", "{year}_Station_{station_id}.tif"))
        return pattern.format(year=year, station_id=station_id)


def load_patch_data(path: Path | None) -> dict[tuple[str, int], np.ndarray]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    patches: dict[tuple[str, int], np.ndarray] = {}
    for _, row in frame.iterrows():
        sid = normalize_station_id(row["Station_ID"])
        year = int(float(row["Year"]))
        patches[(sid, year)] = np.array([row["TEMP"], row["MAX"], row["MIN"]], dtype=np.float32)
    return patches


def process_station_year_tif(path: Path, year: int, target_size: int, channels: int, patch_values: np.ndarray | None) -> np.ndarray:
    import rasterio

    expected_days = days_in_year(year)
    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)

    raw_height, raw_width = raw.shape[1], raw.shape[2]
    actual_days = raw.shape[0] // channels
    reshaped = raw[: actual_days * channels].reshape(actual_days, channels, raw_height, raw_width)

    if actual_days == expected_days - 1:
        last_day = reshaped[-1].copy()
        if patch_values is not None:
            last_day[:, raw_height // 2, raw_width // 2] = patch_values
        reshaped = np.vstack([reshaped, last_day[np.newaxis, ...]])
    elif actual_days > expected_days:
        reshaped = reshaped[:expected_days]
    elif actual_days < expected_days:
        pad = np.zeros((expected_days - actual_days, channels, raw_height, raw_width), dtype=np.float32)
        reshaped = np.vstack([reshaped, pad])

    filled = fill_time_nan(reshaped)
    zoom_h = target_size / raw_height
    zoom_w = target_size / raw_width
    return zoom(filled, (1, 1, zoom_h, zoom_w), order=1).astype(np.float32)


def fill_time_nan(data: np.ndarray) -> np.ndarray:
    if not np.isnan(data).any():
        return data.astype(np.float32)
    days, channels, height, width = data.shape
    flat = data.reshape(days, -1)
    frame = pd.DataFrame(flat)
    frame = frame.interpolate(method="linear", limit_direction="both", axis=0)
    filled = frame.fillna(0.0).to_numpy(dtype=np.float32)
    return filled.reshape(days, channels, height, width)
