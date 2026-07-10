from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from data_pipeline.common.dates import daily_range, days_in_year
from data_pipeline.common.logging import get_logger
from data_pipeline.common.station import read_metadata
from data_pipeline.common.validation import ValidationRecord, validate_shape, write_validation_report
from data_pipeline.config import PipelineConfig
from data_pipeline.modules.base import ModuleResult
from data_pipeline.outputs.manifest import file_sha256, shape_for_file, write_manifest
from data_pipeline.registry import get_modules


@dataclass
class DataContext:
    config: PipelineConfig
    logger_name: str = "data_pipeline"

    @property
    def logger(self):
        return get_logger(self.logger_name)

    def metadata(self):
        return read_metadata(self.config.metadata_csv)


def plan_pipeline(config: PipelineConfig, module_names: list[str] | None = None) -> list[ModuleResult]:
    context = DataContext(config)
    return [module.plan(context) for module in get_modules(module_names or config.enabled_modules)]


def run_pipeline(config: PipelineConfig, module_names: list[str] | None = None) -> list[ModuleResult]:
    context = DataContext(config)
    results: list[ModuleResult] = []
    for module in get_modules(module_names or config.enabled_modules):
        context.logger.info("Running module: %s", module.name)
        results.append(module.run(context))
    write_reports(config, results)
    return results


def write_reports(config: PipelineConfig, results: list[ModuleResult]) -> None:
    records = validate_results(config, results)
    report_path = config.output_root / "reports" / "validation_report.csv"
    write_validation_report(records, report_path)

    metadata = read_metadata(config.metadata_csv) if config.metadata_csv.exists() else None
    files = []
    for result in results:
        for path in [*result.outputs, *result.reports]:
            if not Path(path).exists():
                continue
            shape = shape_for_file(path)
            files.append(
                {
                    "module": result.name,
                    "path": str(path),
                    "shape": list(shape) if shape is not None else None,
                    "sha256": file_sha256(path),
                }
            )

    payload = {
        "project": config.get("project.name", "unnamed"),
        "metadata_csv": str(config.metadata_csv),
        "metadata_sha256": file_sha256(config.metadata_csv) if config.metadata_csv.exists() else None,
        "station_count": int(len(metadata)) if metadata is not None else None,
        "station_id_sha256": station_id_hash(metadata) if metadata is not None else None,
        "time": {"start_date": config.start_date, "end_date": config.end_date, "start_year": config.start_year, "end_year": config.end_year},
        "grid": {"image_size": config.image_size, "target_res_m": config.target_res_m},
        "files": files,
        "validation_report": str(report_path),
    }
    write_manifest(config.output_root / "reports" / "manifest.json", payload)


def station_id_hash(metadata) -> str:
    import hashlib

    digest = hashlib.sha256()
    for station_id in metadata["Station_ID"].astype(str):
        digest.update(station_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_results(config: PipelineConfig, results: Iterable[ModuleResult]) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    metadata = read_metadata(config.metadata_csv) if config.metadata_csv.exists() else None
    if metadata is None:
        return [ValidationRecord("pipeline", "metadata", "error", f"missing: {config.metadata_csv}")]

    n = len(metadata)
    total_days = len(daily_range(config.start_date, config.end_date))
    for result in results:
        for path in result.outputs:
            expected = expected_shape(config, result.name, Path(path), n, total_days)
            actual = safe_shape(path)
            if actual is None:
                records.append(ValidationRecord(result.name, str(path), "error", "missing or unreadable"))
            elif expected is not None:
                records.append(validate_shape(actual, expected, result.name, str(path)))
            else:
                records.append(ValidationRecord(result.name, str(path), "ok", f"shape={actual}"))
    return records


def safe_shape(path: str | Path) -> tuple[int, ...] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    try:
        if candidate.suffix == ".npy":
            return tuple(np.load(candidate, mmap_mode="r").shape)
        if candidate.suffix in {".h5", ".hdf5"}:
            with h5py.File(candidate, "r") as handle:
                return tuple(handle["data"].shape)
    except Exception:  # noqa: BLE001
        return None
    return None


def expected_shape(config: PipelineConfig, module: str, path: Path, n: int, total_days: int) -> tuple[int, ...] | None:
    size = config.image_size
    if module == "gsod" and path.suffix == ".npy":
        return (n, total_days, 3)
    if module == "albedo":
        return (n, size, size)
    if module == "dem":
        return (n, size, size)
    if module == "cci":
        return (n, size, size)
    year = year_from_name(path.name)
    if module == "era5" and year is not None:
        return (days_in_year(year), n, 4, size, size)
    if module == "lst" and year is not None:
        return (days_in_year(year), n, 3, size, size)
    return None


def year_from_name(name: str) -> int | None:
    match = re.search(r"(20\d{2})", name)
    return int(match.group(1)) if match else None
