from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PhysicalConfig:
    raw: dict[str, Any]
    config_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "PhysicalConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data, config_path)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def path(self, value: str | Path | None, default: str | Path | None = None) -> Path:
        raw_value = value if value not in (None, "") else default
        if raw_value is None:
            raise ValueError("Path value is required")
        path = Path(str(raw_value)).expanduser()
        if not path.is_absolute():
            path = (self.config_path.parent / path).resolve()
        return path

    @property
    def metadata_csv(self) -> Path:
        return self.path(self.get("metadata_csv"))

    @property
    def dem_elevation(self) -> Path:
        return self.path(self.get("inputs.dem_elevation"))

    @property
    def era5_temp_dir(self) -> Path:
        return self.path(self.get("inputs.era5_temp_dir"))

    @property
    def surface_geopotential(self) -> Path:
        return self.path(self.get("physical_raw.era5_surface_geopotential"))

    @property
    def pressure_monthly(self) -> Path:
        return self.path(self.get("physical_raw.era5_pressure_monthly"))

    @property
    def factor_dir(self) -> Path:
        return self.path(self.get("outputs.factor_dir"))

    @property
    def tbase_dir(self) -> Path:
        return self.path(self.get("outputs.tbase_dir"))

    @property
    def report_dir(self) -> Path:
        return self.path(self.get("outputs.report_dir"))

    @property
    def start_year(self) -> int:
        return int(self.get("time.start_year", 2008))

    @property
    def end_year(self) -> int:
        return int(self.get("time.end_year", 2024))

    @property
    def climatology_years(self) -> list[int]:
        years = self.get("time.climatology_years")
        if years:
            return [int(year) for year in years]
        return list(range(self.start_year, self.end_year + 1))

    @property
    def image_size(self) -> int:
        return int(self.get("grid.image_size", 128))

    @property
    def target_res_m(self) -> float:
        return float(self.get("grid.target_res_m", 100))
