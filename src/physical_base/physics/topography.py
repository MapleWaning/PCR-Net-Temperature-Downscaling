from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def calculate_topographic_factors(
    dem: np.ndarray,
    target_res_m: float,
    tpi_window_size: int,
    basin_tpi_threshold_m: float,
    basin_elevation_limit_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = np.zeros_like(dem, dtype=np.float32)
    aspect = np.zeros_like(dem, dtype=np.float32)
    basin_mask = np.zeros_like(dem, dtype=bool)

    for idx in range(dem.shape[0]):
        station_dem = dem[idx].astype(np.float32)
        dy, dx = np.gradient(station_dem, target_res_m, target_res_m)
        slope[idx] = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect[idx] = np.pi - np.arctan2(-dy, dx)

        macro_dem = uniform_filter(station_dem, size=tpi_window_size)
        tpi = station_dem - macro_dem
        basin_mask[idx] = (tpi < basin_tpi_threshold_m) & (station_dem < basin_elevation_limit_m)

    return slope, aspect, basin_mask
