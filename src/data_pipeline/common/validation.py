from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ValidationRecord:
    module: str
    item: str
    status: str
    detail: str


def expected_spatial_shape(n_stations: int, image_size: int = 128) -> tuple[int, int, int]:
    return (n_stations, image_size, image_size)


def validate_shape(actual: tuple[int, ...], expected: tuple[int, ...], module: str, item: str) -> ValidationRecord:
    if tuple(actual) == tuple(expected):
        return ValidationRecord(module, item, "ok", f"shape={actual}")
    return ValidationRecord(module, item, "error", f"shape={actual}, expected={expected}")


def finite_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "dtype": str(array.dtype),
    }


def write_validation_report(records: list[ValidationRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(output, index=False)
    return output
