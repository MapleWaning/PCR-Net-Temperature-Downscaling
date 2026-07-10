from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_for_file(path: str | Path) -> tuple[int, ...] | None:
    candidate = Path(path)
    if candidate.suffix == ".npy":
        return tuple(np.load(candidate, mmap_mode="r").shape)
    if candidate.suffix in {".h5", ".hdf5"}:
        with h5py.File(candidate, "r") as handle:
            return tuple(handle["data"].shape) if "data" in handle else None
    return None


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return output
