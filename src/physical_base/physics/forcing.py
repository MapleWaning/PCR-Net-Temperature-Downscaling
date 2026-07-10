from __future__ import annotations

import numpy as np


def radiation_forcing(slope: np.ndarray, aspect: np.ndarray, beta_radiation: float) -> np.ndarray:
    trf_index = np.sin(slope) * np.cos(aspect - np.pi)
    return beta_radiation * trf_index


def monthly_total_forcing(
    month: int,
    delta_h: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    basin_mask: np.ndarray,
    gamma_monthly: np.ndarray,
    h_dem: np.ndarray,
    inversion_months: set[int],
    inversion_top_m: float,
    gamma_inversion: float,
    beta_radiation: float,
) -> np.ndarray:
    gamma = gamma_monthly[month - 1]
    normal_lapse = gamma * delta_h

    if month in inversion_months:
        inv_condition = basin_mask & (h_dem < inversion_top_m)
        lapse = np.where(inv_condition, gamma_inversion * delta_h, normal_lapse)
    else:
        lapse = normal_lapse

    return lapse + radiation_forcing(slope, aspect, beta_radiation)
