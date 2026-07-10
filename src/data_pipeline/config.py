from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).expanduser().resolve()
        default_path = config_path.parent / "default.yaml"
        data = _load_yaml(default_path) if default_path.exists() and default_path != config_path else {}
        data = _deep_merge(data, _load_yaml(config_path))
        return cls(raw=data, config_path=config_path)

    def section(self, name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        value = self.raw.get(name, default or {})
        return value if isinstance(value, dict) else {}

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def project_path(self, key: str, fallback: str) -> Path:
        return self.path(self.get(f"project.{key}", fallback))

    def path(self, value: str | Path | None, default: str | None = None) -> Path:
        raw_value = value if value not in (None, "") else default
        if raw_value is None:
            raise ValueError("Path value is required")
        path = Path(str(raw_value)).expanduser()
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        return path

    @property
    def output_root(self) -> Path:
        return self.project_path("output_root", "./processed")

    @property
    def raw_root(self) -> Path:
        return self.project_path("raw_root", "./raw")

    @property
    def temp_root(self) -> Path:
        return self.project_path("temp_root", "./tmp")

    @property
    def metadata_csv(self) -> Path:
        return self.path(self.get("station.metadata_csv"), self.output_root / "metadata" / "high-quality-meta.csv")

    @property
    def start_year(self) -> int:
        return int(self.get("time.start_year", 2008))

    @property
    def end_year(self) -> int:
        return int(self.get("time.end_year", 2024))

    @property
    def start_date(self) -> str:
        return str(self.get("time.start_date", f"{self.start_year}-01-01"))

    @property
    def end_date(self) -> str:
        return str(self.get("time.end_date", f"{self.end_year}-12-31"))

    @property
    def image_size(self) -> int:
        return int(self.get("grid.image_size", 128))

    @property
    def target_res_m(self) -> float:
        return float(self.get("grid.target_res_m", 100))

    @property
    def enabled_modules(self) -> list[str]:
        modules = self.section("modules")
        return [name for name, enabled in modules.items() if bool(enabled)]

    def module_path(self, module: str, key: str, default: Path | str) -> Path:
        return self.path(self.get(f"{module}.{key}"), default)
