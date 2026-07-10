from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def write_h5_dataset(path: str | Path, data: np.ndarray, dataset: str = "data", **attrs: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset(dataset, data=data, compression="lzf")
        for key, value in attrs.items():
            handle.attrs[key] = value
    return output


def h5_shape(path: str | Path, dataset: str = "data") -> tuple[int, ...]:
    with h5py.File(path, "r") as handle:
        return tuple(handle[dataset].shape)


def h5_exists_with_shape(path: str | Path, expected: tuple[int, ...], dataset: str = "data") -> bool:
    candidate = Path(path)
    if not candidate.exists():
        return False
    try:
        return h5_shape(candidate, dataset) == tuple(expected)
    except Exception:  # noqa: BLE001
        return False
