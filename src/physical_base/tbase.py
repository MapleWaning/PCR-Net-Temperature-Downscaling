from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_pipeline.common.dates import year_date_range
from data_pipeline.common.paths import ensure_dir
from physical_base.config import PhysicalConfig
from physical_base.factors import load_factors
from physical_base.physics.forcing import monthly_total_forcing


def run_tbase(config: PhysicalConfig) -> list[Path]:
    factors = load_factors(config)
    h_dem = factors["H_DEM_100m"].astype(np.float32)
    h_era5 = factors["H_ERA5_avg"].astype(np.float32)
    slope = factors["Slope_100m"].astype(np.float32)
    aspect = factors["Aspect_100m"].astype(np.float32)
    basin = factors["Basin_Mask_100m"].astype(bool)
    gamma = factors["Gamma_Monthly_100m"].astype(np.float32)

    delta_h = h_dem - h_era5
    inversion_months = {int(value) for value in config.get("physics.inversion_months", [12, 1, 2])}
    inversion_top_m = float(config.get("physics.inversion_top_m", 900))
    gamma_inversion = float(config.get("physics.gamma_inversion", 0.010))
    beta_radiation = float(config.get("physics.beta_radiation", 1.5))
    output_dir = ensure_dir(config.tbase_dir)

    outputs: list[Path] = []
    for year in range(config.start_year, config.end_year + 1):
        src_path = config.era5_temp_dir / f"era5_t2m_{year}.h5"
        if not src_path.exists():
            continue
        output_path = output_dir / f"t_base_advanced_{year}.h5"
        write_tbase_year(
            src_path,
            output_path,
            year,
            delta_h,
            slope,
            aspect,
            basin,
            gamma,
            h_dem,
            inversion_months,
            inversion_top_m,
            gamma_inversion,
            beta_radiation,
        )
        outputs.append(output_path)
    return outputs


def write_tbase_year(
    src_path: Path,
    output_path: Path,
    year: int,
    delta_h: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    basin: np.ndarray,
    gamma: np.ndarray,
    h_dem: np.ndarray,
    inversion_months: set[int],
    inversion_top_m: float,
    gamma_inversion: float,
    beta_radiation: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, "r") as src, h5py.File(output_path, "w") as dst:
        src_dataset = src["data"]
        dst_dataset = dst.create_dataset(
            "data",
            shape=src_dataset.shape,
            dtype="float32",
            chunks=src_dataset.chunks,
            compression="lzf",
        )
        dates = year_date_range(year, src_dataset.shape[0])
        cached_month = None
        cached_forcing = None

        for day_idx in tqdm(range(src_dataset.shape[0]), desc=f"Physical tbase {year}"):
            month = int(dates[day_idx].month)
            if month != cached_month:
                forcing = monthly_total_forcing(
                    month,
                    delta_h,
                    slope,
                    aspect,
                    basin,
                    gamma,
                    h_dem,
                    inversion_months,
                    inversion_top_m,
                    gamma_inversion,
                    beta_radiation,
                )
                cached_forcing = forcing[np.newaxis, :, np.newaxis, :, :].astype(np.float32)
                cached_month = month
            dst_dataset[day_idx : day_idx + 1] = src_dataset[day_idx : day_idx + 1] + cached_forcing
