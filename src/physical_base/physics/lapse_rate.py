from __future__ import annotations

import numpy as np
import xarray as xr


def calculate_gamma_monthly(
    pressure_nc_path,
    grid_lats,
    grid_lons,
    gravity: float,
    climatology_years: list[int],
    pressure_levels: list[int],
) -> np.ndarray:
    with xr.open_dataset(pressure_nc_path) as ds:
        ds = normalize_longitude(ds)
        time_var = "time" if "time" in ds.coords else "valid_time"
        if time_var in ds.coords and climatology_years:
            mask = ds[time_var].dt.year.isin(climatology_years)
            ds = ds.where(mask, drop=True)

        level_var = find_level_coord(ds)
        if level_var is None:
            raise ValueError("Pressure-level ERA5 data must include a pressure level coordinate")
        if level_var is not None and pressure_levels:
            ds = select_pressure_levels(ds, level_var, pressure_levels)

        z_var = find_variable(ds, ("z", "geopotential"))
        t_var = find_variable(ds, ("t", "temperature"))
        climatology = ds.groupby(f"{time_var}.month").mean(dim=time_var)

        z_da = climatology[z_var].transpose("month", level_var, "latitude", "longitude")
        t_da = climatology[t_var].transpose("month", level_var, "latitude", "longitude")
        z = z_da.values / gravity
        temperature = t_da.values

    # The regression axis is pressure level: gamma = dT / dz.
    n_levels = z.shape[1]
    sum_z = np.sum(z, axis=1)
    sum_t = np.sum(temperature, axis=1)
    sum_zt = np.sum(z * temperature, axis=1)
    sum_zz = np.sum(z * z, axis=1)
    denominator = n_levels * sum_zz - sum_z**2
    gamma = np.divide(n_levels * sum_zt - sum_z * sum_t, denominator, out=np.zeros_like(sum_t, dtype=np.float32), where=denominator != 0)

    gamma_da = xr.DataArray(
        gamma,
        coords=[climatology.month, climatology.latitude, climatology.longitude],
        dims=["month", "latitude", "longitude"],
    )
    return gamma_da.interp(latitude=grid_lats, longitude=grid_lons, method="linear").values.astype(np.float32)


def normalize_longitude(ds):
    if "longitude" in ds.coords and float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby("longitude")
    return ds


def find_variable(ds, names: tuple[str, ...]) -> str:
    for name in names:
        if name in ds.data_vars:
            return name
    raise ValueError(f"Missing variable, expected one of {names}. Available: {list(ds.data_vars)}")


def find_level_coord(ds) -> str | None:
    for name in ("pressure_level", "level", "isobaricInhPa"):
        if name in ds.coords or name in ds.dims:
            return name
    return None


def select_pressure_levels(ds, level_var: str, pressure_levels: list[int]):
    levels = ds[level_var].values
    try:
        return ds.sel({level_var: pressure_levels})
    except Exception:  # noqa: BLE001
        string_levels = [str(level) for level in pressure_levels]
        if all(str(level) in {str(value) for value in levels} for level in pressure_levels):
            return ds.sel({level_var: string_levels})
    return ds
