from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from data_pipeline.outputs.manifest import file_sha256, shape_for_file
from physical_base.config import PhysicalConfig
from physical_base.factors import factor_paths, run_factors
from physical_base.tbase import run_tbase
from physical_base.validation import h5_shape, validate_physical_outputs


def plan(config: PhysicalConfig) -> Path:
    rows = [
        {"name": "metadata_csv", "path": str(config.metadata_csv), "exists": config.metadata_csv.exists()},
        {"name": "dem_elevation", "path": str(config.dem_elevation), "exists": config.dem_elevation.exists()},
        {"name": "era5_surface_geopotential", "path": str(config.surface_geopotential), "exists": config.surface_geopotential.exists()},
        {"name": "era5_pressure_monthly", "path": str(config.pressure_monthly), "exists": config.pressure_monthly.exists()},
    ]
    for year in range(config.start_year, config.end_year + 1):
        path = config.era5_temp_dir / f"era5_t2m_{year}.h5"
        rows.append({"name": f"era5_t2m_{year}", "path": str(path), "exists": path.exists(), "shape": h5_shape(path)})

    import pandas as pd

    output = config.report_dir / "physical_plan.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def run_factors_stage(config: PhysicalConfig) -> list[Path]:
    outputs = run_factors(config)
    write_physical_reports(config)
    return outputs


def run_tbase_stage(config: PhysicalConfig) -> list[Path]:
    outputs = run_tbase(config)
    write_physical_reports(config)
    return outputs


def run_all(config: PhysicalConfig) -> list[Path]:
    outputs = []
    outputs.extend(run_factors(config))
    outputs.extend(run_tbase(config))
    write_physical_reports(config)
    return outputs


def write_physical_reports(config: PhysicalConfig) -> Path:
    validate_physical_outputs(config)
    files = []
    for path in factor_paths(config).values():
        if path.exists():
            files.append(file_entry("factor", path))
    for year in range(config.start_year, config.end_year + 1):
        path = config.tbase_dir / f"t_base_advanced_{year}.h5"
        if path.exists():
            files.append(file_entry("tbase", path))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_csv": str(config.metadata_csv),
        "metadata_sha256": file_sha256(config.metadata_csv) if config.metadata_csv.exists() else None,
        "climatology_years": config.climatology_years,
        "pressure_levels": config.get("era5_pressure.pressure_levels", [500, 700, 850]),
        "physics": config.get("physics", {}),
        "inputs": {
            "dem_elevation": str(config.dem_elevation),
            "era5_temp_dir": str(config.era5_temp_dir),
            "surface_geopotential": str(config.surface_geopotential),
            "pressure_monthly": str(config.pressure_monthly),
        },
        "files": files,
        "validation_report": str(config.report_dir / "physical_validation_report.csv"),
    }
    output = config.report_dir / "physical_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return output


def file_entry(kind: str, path: Path) -> dict[str, object]:
    shape = shape_for_file(path)
    return {
        "kind": kind,
        "path": str(path),
        "shape": list(shape) if shape is not None else None,
        "sha256": file_sha256(path),
    }
