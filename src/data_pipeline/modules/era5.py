from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from data_pipeline.common.dates import days_in_month, days_in_year
from data_pipeline.common.hdf5 import h5_exists_with_shape
from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.raster import station_aeqd_grids
from data_pipeline.common.station import read_metadata
from data_pipeline.modules.base import ModuleResult


class Era5Module:
    name = "era5"

    def plan(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv) if cfg.metadata_csv.exists() else None
        outputs = [self._year_path(cfg, year) for year in range(cfg.start_year, cfg.end_year + 1)]
        details = {"missing_count": len([path for path in outputs if not path.exists()])}
        if metadata is not None:
            details["station_count"] = len(metadata)
        return ModuleResult(self.name, outputs=outputs, details=details)

    def run(self, context) -> ModuleResult:
        import cdsapi

        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv)
        output_dir = ensure_dir(self._output_dir(cfg))
        temp_dir = ensure_dir(self._temp_dir(cfg))
        grid_lats, grid_lons = station_aeqd_grids(metadata, cfg.image_size, cfg.target_res_m)
        client = cdsapi.Client()
        written: list[Path] = []

        for year in range(cfg.start_year, cfg.end_year + 1):
            h5_path = output_dir / f"era5_t2m_{year}.h5"
            expected_shape = (days_in_year(year), len(metadata), 4, cfg.image_size, cfg.image_size)
            if h5_exists_with_shape(h5_path, expected_shape):
                written.append(h5_path)
                continue
            if h5_path.exists():
                h5_path.unlink()

            for month in range(1, 13):
                nc_path = temp_dir / f"era5_{year}_{month:02d}.nc"
                if not verify_netcdf(nc_path):
                    self._download_month(client, cfg, year, month, nc_path)
                self._append_month(nc_path, h5_path, grid_lats, grid_lons, len(metadata), cfg.image_size)

            clean_era5_h5(h5_path)
            written.append(h5_path)

        return ModuleResult(self.name, outputs=written, details={"station_count": len(metadata)})

    def _output_dir(self, cfg) -> Path:
        return cfg.path(cfg.get("era5.output_dir"), cfg.output_root / "era5")

    def _temp_dir(self, cfg) -> Path:
        return cfg.path(cfg.get("era5.temp_dir"), cfg.temp_root / "era5")

    def _year_path(self, cfg, year: int) -> Path:
        return self._output_dir(cfg) / f"era5_t2m_{year}.h5"

    def _download_month(self, client, cfg, year: int, month: int, save_path: Path) -> None:
        days = [str(day).zfill(2) for day in range(1, days_in_month(year, month) + 1)]
        times = list(cfg.get("era5.download_times", ["00:00", "06:00", "12:00", "18:00"]))
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": str(year),
                "month": str(month).zfill(2),
                "day": days,
                "time": times,
                "format": "netcdf",
            },
            str(save_path),
        )

    def _append_month(self, nc_path: Path, h5_path: Path, grid_lats, grid_lons, n_stations: int, image_size: int) -> None:
        with xr.open_dataset(nc_path) as ds:
            ds = normalize_era5_dataset(ds)
            variable = find_temperature_variable(ds)
            data = ds[variable].interp(latitude=grid_lats, longitude=grid_lons, method="linear").values.astype(np.float32)

        if data.shape[0] % 4 != 0:
            raise ValueError(f"ERA5 month has incomplete 4-time daily groups: {nc_path}")
        n_days = data.shape[0] // 4
        data = data.reshape(n_days, 4, n_stations, image_size, image_size).transpose(0, 2, 1, 3, 4)
        data -= 273.15

        with h5py.File(h5_path, "a") as handle:
            if "data" not in handle:
                handle.create_dataset(
                    "data",
                    data=data,
                    maxshape=(None, n_stations, 4, image_size, image_size),
                    chunks=(1, n_stations, 4, image_size, image_size),
                    dtype="float32",
                    compression="lzf",
                )
            else:
                dataset = handle["data"]
                dataset.resize(dataset.shape[0] + n_days, axis=0)
                dataset[-n_days:] = data


def verify_netcdf(path: str | Path) -> bool:
    candidate = Path(path)
    if not candidate.exists() or candidate.stat().st_size == 0:
        return False
    try:
        with xr.open_dataset(candidate) as ds:
            return "time" in ds.dims or "valid_time" in ds.dims
    except Exception:  # noqa: BLE001
        return False


def normalize_era5_dataset(ds):
    if "time" not in ds.dims and "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    return ds


def find_temperature_variable(ds) -> str:
    for candidate in ("t2m", "2t", "var167", "temperature"):
        if candidate in ds.data_vars:
            return candidate
    raise ValueError(f"Could not find ERA5 2m temperature variable. Available: {list(ds.data_vars)}")


def clean_era5_h5(path: str | Path) -> None:
    with h5py.File(path, "r+") as handle:
        data = handle["data"][:]
        if np.isfinite(data).all():
            return
        days, stations, channels, height, width = data.shape
        for station_idx in range(stations):
            station_data = data[:, station_idx]
            if np.isfinite(station_data).all():
                continue
            station_data = station_data.astype(np.float32, copy=True)
            station_data[~np.isfinite(station_data)] = np.nan
            flat = station_data.reshape(days, -1)
            frame = pd.DataFrame(flat)
            frame = frame.interpolate(method="linear", limit_direction="both", axis=0)
            frame = frame.bfill().ffill().fillna(0.0)
            data[:, station_idx] = frame.to_numpy(dtype=np.float32).reshape(days, channels, height, width)
        handle["data"][:] = data.astype(np.float32)
