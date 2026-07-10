from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from data_pipeline.common.dates import days_in_year
from data_pipeline.common.station import read_metadata
from data_pipeline.common.validation import ValidationRecord, validate_shape, write_validation_report
from physical_base.config import PhysicalConfig
from physical_base.factors import factor_paths


def validate_physical_outputs(config: PhysicalConfig) -> list[ValidationRecord]:
    metadata = read_metadata(config.metadata_csv)
    n = len(metadata)
    size = config.image_size
    records: list[ValidationRecord] = []

    expected_factors = {
        "H_DEM_100m": (n, size, size),
        "H_ERA5_avg": (n, size, size),
        "Slope_100m": (n, size, size),
        "Aspect_100m": (n, size, size),
        "Basin_Mask_100m": (n, size, size),
        "Gamma_Monthly_100m": (12, n, size, size),
    }
    for name, path in factor_paths(config).items():
        shape = h5_shape(path)
        if shape is None:
            records.append(ValidationRecord("physical", name, "error", f"missing or unreadable: {path}"))
        else:
            records.append(validate_shape(shape, expected_factors[name], "physical", name))

    for year in range(config.start_year, config.end_year + 1):
        path = config.tbase_dir / f"t_base_advanced_{year}.h5"
        source = config.era5_temp_dir / f"era5_t2m_{year}.h5"
        source_shape = h5_shape(source)
        shape = h5_shape(path)
        if source_shape is None:
            records.append(ValidationRecord("physical", str(source), "warning", "ERA5 source missing"))
        elif shape is None:
            records.append(ValidationRecord("physical", str(path), "warning", "tbase output missing"))
        else:
            records.append(validate_shape(shape, source_shape, "physical", str(path)))
            if shape[0] != days_in_year(year):
                records.append(ValidationRecord("physical", str(path), "warning", f"days={shape[0]}, calendar_days={days_in_year(year)}"))

    write_validation_report(records, config.report_dir / "physical_validation_report.csv")
    return records


def h5_shape(path: str | Path) -> tuple[int, ...] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        with h5py.File(candidate, "r") as handle:
            return tuple(handle["data"].shape)
    except Exception:  # noqa: BLE001
        return None


def nan_inf_summary(path: str | Path) -> dict[str, int] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    with h5py.File(candidate, "r") as handle:
        data = handle["data"][:]
    if not np.issubdtype(data.dtype, np.floating):
        return {"nan_count": 0, "inf_count": 0}
    return {"nan_count": int(np.isnan(data).sum()), "inf_count": int(np.isinf(data).sum())}
