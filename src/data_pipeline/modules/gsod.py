from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from data_pipeline.common.dates import daily_range, days_in_year
from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.station import build_noaa_id, read_metadata, standardize_metadata, write_metadata
from data_pipeline.modules.base import ModuleResult


class GsodModule:
    name = "gsod"
    feature_cols = ["TEMP", "MAX", "MIN"]
    missing_markers = {9999.9: np.nan, 99.99: np.nan, 999.9: np.nan}

    def plan(self, context) -> ModuleResult:
        cfg = context.config
        outputs = [cfg.metadata_csv, self._weather_path(cfg)]
        details = {"metadata_exists": cfg.metadata_csv.exists(), "station_count": None}
        if cfg.metadata_csv.exists():
            details["station_count"] = int(len(read_metadata(cfg.metadata_csv)))
        return ModuleResult(self.name, outputs=outputs, details=details)

    def run(self, context) -> ModuleResult:
        cfg = context.config
        raw_dir = ensure_dir(self._download_dir(cfg))
        gsod_dir = ensure_dir(cfg.output_root / "gsod")

        if not cfg.metadata_csv.exists():
            candidates = self._filter_station_history(cfg)
            completeness = self._download_station_data(cfg, candidates, raw_dir)
            metadata = self._select_high_quality_metadata(cfg, candidates, completeness)
            write_metadata(metadata, cfg.metadata_csv)
        else:
            metadata = read_metadata(cfg.metadata_csv)
            completeness = pd.DataFrame()

        data, nan_report = self._build_weather_tensor(cfg, metadata, raw_dir)
        weather_path = self._weather_path(cfg)
        np.save(weather_path, data)

        nan_report_path = gsod_dir / "station_nan_report.csv"
        nan_report.to_csv(nan_report_path, index=False)

        completeness_path = gsod_dir / "completeness_report.csv"
        if not completeness.empty:
            completeness.to_csv(completeness_path, index=False)
        elif not completeness_path.exists():
            pd.DataFrame().to_csv(completeness_path, index=False)

        return ModuleResult(
            self.name,
            outputs=[cfg.metadata_csv, weather_path],
            reports=[nan_report_path, completeness_path],
            details={"shape": tuple(data.shape), "features": self.feature_cols},
        )

    def _download_dir(self, cfg) -> Path:
        return cfg.path(cfg.get("gsod.download_dir"), cfg.raw_root / "gsod" / "station_data_merged")

    def _weather_path(self, cfg) -> Path:
        return cfg.output_root / "gsod" / "weather_data.npy"

    def _filter_station_history(self, cfg) -> pd.DataFrame:
        history_source = cfg.get("station.history_csv", "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv")
        df = pd.read_csv(history_source, dtype={"USAF": str, "WBAN": str}, low_memory=False)
        region = cfg.section("station").get("region", {})
        coverage = cfg.section("station").get("coverage", {})

        selected = df[
            (df["LAT"].astype(float) >= float(region.get("lat_min", -90))) &
            (df["LAT"].astype(float) <= float(region.get("lat_max", 90))) &
            (df["LON"].astype(float) >= float(region.get("lon_min", -180))) &
            (df["LON"].astype(float) <= float(region.get("lon_max", 180))) &
            (df["BEGIN"].astype(float) <= float(coverage.get("begin_before", 0))) &
            (df["END"].astype(float) >= float(coverage.get("end_after", 99999999)))
        ].copy()

        selected["Station_ID"] = [build_noaa_id(u, w) for u, w in zip(selected["USAF"], selected["WBAN"])]
        selected["Station_Name"] = selected.get("STATION NAME", "UNKNOWN")
        selected["Latitude"] = selected["LAT"].astype(float)
        selected["Longitude"] = selected["LON"].astype(float)
        selected = selected.sort_values("Longitude").reset_index(drop=True)

        candidate_path = cfg.output_root / "metadata" / "candidate-stations.csv"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(candidate_path, index=False)
        return standardize_metadata(selected)

    def _download_station_data(self, cfg, stations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
        tasks = [(row["Station_ID"], row["Station_Name"]) for _, row in stations.iterrows()]
        max_workers = int(cfg.get("gsod.max_workers", 8))
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._download_single_station, cfg, sid, name, output_dir): sid for sid, name in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), unit="station", desc="GSOD"):
                results.append(future.result())

        return pd.DataFrame(results)

    def _download_single_station(self, cfg, station_id: str, station_name: str, output_dir: Path) -> dict[str, Any]:
        stats: dict[str, Any] = {"Station_ID": station_id, "Station_Name": station_name}
        yearly_frames: list[pd.DataFrame] = []
        total_expected = 0
        total_missing = 0

        for year in range(cfg.start_year, cfg.end_year + 1):
            expected = days_in_year(year)
            total_expected += expected
            missing = expected
            url = f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station_id}.csv"

            try:
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    frame = pd.read_csv(StringIO(response.content.decode("utf-8")), quotechar='"')
                    if "DATE" in frame.columns:
                        frame["DATE"] = pd.to_datetime(frame["DATE"])
                    yearly_frames.append(frame)
                    missing = max(0, expected - len(frame))
            except Exception:  # noqa: BLE001
                missing = expected

            stats[f"Missing_{year}"] = missing
            total_missing += missing

        stats["Total_Missing_Rate_Percent"] = round((total_missing / total_expected) * 100, 2) if total_expected else 100.0

        if yearly_frames:
            merged = pd.concat(yearly_frames, ignore_index=True)
            if "DATE" in merged.columns:
                merged = merged.sort_values("DATE")
            merged.to_csv(output_dir / f"{station_id}_{cfg.start_year}_{cfg.end_year}.csv", index=False)

        return stats

    def _select_high_quality_metadata(self, cfg, candidates: pd.DataFrame, completeness: pd.DataFrame) -> pd.DataFrame:
        max_missing = float(cfg.get("station.quality.max_missing_rate", 0.05)) * 100.0
        if completeness.empty:
            return candidates
        keep_ids = set(
            completeness.loc[
                completeness["Total_Missing_Rate_Percent"].astype(float) <= max_missing,
                "Station_ID",
            ].astype(str)
        )
        return candidates[candidates["Station_ID"].astype(str).isin(keep_ids)].reset_index(drop=True)

    def _build_weather_tensor(self, cfg, metadata: pd.DataFrame, data_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
        target_dates = daily_range(cfg.start_date, cfg.end_date)
        data = np.full((len(metadata), len(target_dates), len(self.feature_cols)), np.nan, dtype=np.float32)
        report_rows: list[dict[str, Any]] = []

        for station_idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="GSOD tensor"):
            sid = str(row["Station_ID"])
            station_row: dict[str, Any] = {"Station_ID": sid, "Station_Name": row["Station_Name"]}
            source = self._find_station_csv(data_dir, sid, cfg)
            if source is None:
                station_row["Total_NaN"] = len(target_dates)
                report_rows.append(station_row)
                continue

            frame = pd.read_csv(source)
            if "DATE" not in frame.columns:
                station_row["Total_NaN"] = len(target_dates)
                report_rows.append(station_row)
                continue

            frame["DATE"] = pd.to_datetime(frame["DATE"])
            frame = frame.set_index("DATE")
            frame = frame[~frame.index.duplicated(keep="first")].reindex(target_dates)

            temp_series = None
            station_features = []
            for col in self.feature_cols:
                if col in frame.columns:
                    series = pd.to_numeric(frame[col], errors="coerce").replace(self.missing_markers)
                    series = (series - 32.0) * 5.0 / 9.0
                    filled = smart_fill_series(series, int(cfg.get("gsod.gap_threshold_days", 5)))
                    station_features.append(filled.to_numpy(dtype=np.float32))
                    if col == "TEMP":
                        temp_series = filled
                else:
                    station_features.append(np.full(len(target_dates), np.nan, dtype=np.float32))

            data[station_idx] = np.stack(station_features, axis=1)
            if temp_series is not None:
                yearly_nans = temp_series.isna().groupby(temp_series.index.year).sum()
                total_nan = 0
                for year in range(cfg.start_year, cfg.end_year + 1):
                    count = int(yearly_nans.get(year, 0))
                    station_row[str(year)] = count
                    total_nan += count
                station_row["Total_NaN"] = total_nan
            report_rows.append(station_row)

        return data, pd.DataFrame(report_rows)

    def _find_station_csv(self, data_dir: Path, sid: str, cfg) -> Path | None:
        candidates = [
            data_dir / f"{sid}_{cfg.start_year}_{cfg.end_year}.csv",
            data_dir / f"{sid}.csv",
            data_dir / f"{sid}_2008_2024.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None


def smart_fill_series(series: pd.Series, gap_threshold: int = 5) -> pd.Series:
    if series.isna().all():
        return series
    is_nan = series.isna()
    groups = is_nan.ne(is_nan.shift()).cumsum()
    group_sizes = groups.map(groups.value_counts())
    interpolated = series.interpolate(method="linear", limit_direction="both")
    interpolated.loc[is_nan & (group_sizes >= gap_threshold)] = np.nan
    return interpolated
