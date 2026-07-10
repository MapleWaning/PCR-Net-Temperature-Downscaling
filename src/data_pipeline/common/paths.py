from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def existing_file(path: str | Path) -> Path | None:
    candidate = Path(path)
    return candidate if candidate.exists() and candidate.is_file() else None
