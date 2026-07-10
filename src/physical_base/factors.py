from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.station import read_metadata
from physical_base.config import PhysicalConfig
from physical_base.physics.grid import station_aeqd_grids
from physical_base.physics.lapse_rate import calculate_gamma_monthly, find_variable, normalize_longitude
from physical_base.physics.topography import calculate_topographic_factors


FACTOR_FILES = {
    "H_DEM_100m": "H_DEM_100m.h5",
    "H_ERA5_avg": "H_ERA5_avg.h5",
    "Slope_100m": "Slope_100m.h5",
    "Aspect_100m": "Aspect_100m.h5",
    "Basin_Mask_100m": "Basin_Mask_100m.h5",
    "Gamma_Monthly_100m": "Gamma_Monthly_100m.h5",
}


def run_factors(config: PhysicalConfig) -> list[Path]:
    metadata = read_metadata(config.metadata_csv)
    h_dem = np.load(config.dem_elevation).astype(np.float32)
    expected = (len(metadata), config.image_size, config.image_size)
    if tuple(h_dem.shape) != expected:
        raise ValueError(f"DEM elevation shape {h_dem.shape} does not match expected {expected}")

    factor_dir = ensure_dir(config.factor_dir)
    grid_lats, grid_lons = station_aeqd_grids(metadata, config.image_size, config.target_res_m)

    slope, aspect, basin = calculate_topographic_factors(
        h_dem,
        config.target_res_m,
        int(config.get("physics.tpi_window_size", 11)),
        float(config.get("physics.basin_tpi_threshold_m", -15)),
        float(config.get("physics.basin_elevation_limit_m", 2500)),
    )
    h_era5 = interpolate_surface_height(config.surface_geopotential, grid_lats, grid_lons, float(config.get("physics.gravity", 9.80665)))
    gamma = calculate_gamma_monthly(
        config.pressure_monthly,
        grid_lats,
        grid_lons,
        float(config.get("physics.gravity", 9.80665)),
        config.climatology_years,
        [int(level) for level in config.get("era5_pressure.pressure_levels", [500, 700, 850])],
    )

    outputs = [
        save_factor(factor_dir / FACTOR_FILES["H_DEM_100m"], h_dem),
        save_factor(factor_dir / FACTOR_FILES["H_ERA5_avg"], h_era5),
        save_factor(factor_dir / FACTOR_FILES["Slope_100m"], slope),
        save_factor(factor_dir / FACTOR_FILES["Aspect_100m"], aspect),
        save_factor(factor_dir / FACTOR_FILES["Basin_Mask_100m"], basin.astype(bool), dtype=bool),
        save_factor(factor_dir / FACTOR_FILES["Gamma_Monthly_100m"], gamma),
    ]
    return outputs


def interpolate_surface_height(surface_nc_path: Path, grid_lats, grid_lons, gravity: float) -> np.ndarray:
    with xr.open_dataset(surface_nc_path) as ds:
        ds = normalize_longitude(ds)
        z_var = find_variable(ds, ("z", "geopotential"))
        data = ds[z_var]
        for dim in ("time", "valid_time"):
            if dim in data.dims:
                data = data.isel({dim: 0})
        interpolated = data.interp(latitude=grid_lats, longitude=grid_lons, method="linear").values
    return (interpolated / gravity).astype(np.float32)


def save_factor(path: Path, data: np.ndarray, dtype=np.float32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data.astype(dtype), compression="lzf")
    return path


def factor_paths(config: PhysicalConfig) -> dict[str, Path]:
    return {name: config.factor_dir / filename for name, filename in FACTOR_FILES.items()}


def load_factors(config: PhysicalConfig) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for name, path in factor_paths(config).items():
        with h5py.File(path, "r") as handle:
            matrices[name] = handle["data"][:]
    return matrices
