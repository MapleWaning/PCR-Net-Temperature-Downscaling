from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.raster import cci_tile_name, list_tifs
from data_pipeline.common.station import read_metadata
from data_pipeline.modules.base import ModuleResult


CCI_LUT = np.zeros(256, dtype=np.uint8)
CCI_LUT[10:13] = 1
CCI_LUT[50:100] = 2
CCI_LUT[130:141] = 3
CCI_LUT[120:123] = 4
CCI_LUT[150:154] = 4
CCI_LUT[180] = 5
CCI_LUT[210] = 6
CCI_LUT[190] = 7
CCI_LUT[200:203] = 8
CCI_LUT[220] = 9
CCI_LUT[20] = 10


class CciModule:
    name = "cci"

    def plan(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv) if cfg.metadata_csv.exists() else pd.DataFrame()
        tif_dir = cfg.path(cfg.get("manual_inputs.cci_tif_dir"), cfg.raw_root / "cci" / "GLC_FCS30D_30m")
        tiles = sorted(required_cci_tiles(metadata, float(cfg.get("cci.buffer_degree", 0.5))))
        rows = []
        for tile in tiles:
            filename = f"GLC_FCS30D_20002022_{tile}_Annual.tif"
            path = tif_dir / filename
            rows.append({"tile": tile, "file": str(path), "exists": path.exists()})

        report = cfg.output_root / "reports" / "cci_manual_download_manifest.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report, index=False)
        return ModuleResult(self.name, reports=[report], details={"tile_count": len(rows), "missing_count": sum(not r["exists"] for r in rows)})

    def run(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv)
        tif_dir = cfg.path(cfg.get("manual_inputs.cci_tif_dir"), cfg.raw_root / "cci" / "GLC_FCS30D_30m")
        output_dir = ensure_dir(cfg.path(cfg.get("cci.output_dir"), cfg.output_root / "cci"))

        tif_index = build_tif_index(tif_dir)
        n_years = cfg.end_year - cfg.start_year + 1
        data = np.zeros((len(metadata), n_years, cfg.image_size, cfg.image_size), dtype=np.uint8)

        for station_idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="CCI extract"):
            tif_path = find_covering_tif(float(row["Longitude"]), float(row["Latitude"]), tif_index)
            if tif_path is None:
                continue
            station_cube = extract_station_cci(tif_path, float(row["Longitude"]), float(row["Latitude"]), cfg.start_year, cfg.end_year, cfg.image_size, cfg.target_res_m)
            data[station_idx] = station_cube

        forward_fill_time(data)
        outputs = []
        for year_idx, year in enumerate(range(cfg.start_year, cfg.end_year + 1)):
            year_data = fill_categorical_holes(data[:, year_idx])
            output = output_dir / f"GLC_FCS30D_{year}.npy"
            np.save(output, year_data)
            outputs.append(output)

        return ModuleResult(self.name, outputs=outputs, details={"station_count": len(metadata)})


def required_cci_tiles(metadata: pd.DataFrame, buffer_degree: float) -> set[str]:
    tiles: set[str] = set()
    for _, row in metadata.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        points = [
            (lat, lon),
            (lat - buffer_degree, lon),
            (lat + buffer_degree, lon),
            (lat, lon - buffer_degree),
            (lat, lon + buffer_degree),
        ]
        for point_lat, point_lon in points:
            tiles.add(cci_tile_name(point_lat, point_lon))
    return tiles


def build_tif_index(tif_dir: Path) -> list[dict[str, object]]:
    import rasterio
    from rasterio.warp import transform_bounds

    index: list[dict[str, object]] = []
    for path in list_tifs(tif_dir):
        try:
            with rasterio.open(path) as src:
                if src.crs and src.crs.to_epsg() != 4326:
                    bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                else:
                    b = src.bounds
                    bounds = (b.left, b.bottom, b.right, b.top)
                index.append({"path": path, "bounds": bounds})
        except Exception:  # noqa: BLE001
            continue
    return index


def find_covering_tif(lon: float, lat: float, tif_index: list[dict[str, object]]) -> Path | None:
    for item in tif_index:
        left, bottom, right, top = item["bounds"]
        if left <= lon <= right and bottom <= lat <= top:
            return Path(item["path"])
    return None


def extract_station_cci(tif_path: Path, lon: float, lat: float, start_year: int, end_year: int, image_size: int, target_res_m: float) -> np.ndarray:
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    n_years = end_year - start_year + 1
    station_cube = np.zeros((n_years, image_size, image_size), dtype=np.uint8)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)
    half_width_m = image_size * target_res_m / 2.0

    with rasterio.open(tif_path) as src:
        is_multiband = src.count > 10
        with WarpedVRT(src, crs="EPSG:3857", resampling=Resampling.mode) as vrt:
            window = rasterio.windows.from_bounds(
                center_x - half_width_m,
                center_y - half_width_m,
                center_x + half_width_m,
                center_y + half_width_m,
                transform=vrt.transform,
            )
            for year_idx, year in enumerate(range(start_year, end_year + 1)):
                band_id = 1 if not is_multiband else (year - 2000 + 1 if year <= 2022 else 23)
                raw = vrt.read(
                    band_id,
                    window=window,
                    out_shape=(image_size, image_size),
                    resampling=Resampling.mode,
                    fill_value=0,
                )
                raw_uint8 = np.clip(np.asarray(raw), 0, 255).astype(np.uint8)
                station_cube[year_idx] = CCI_LUT[raw_uint8]
    return station_cube


def forward_fill_time(data: np.ndarray) -> None:
    for year_idx in range(1, data.shape[1]):
        current = data[:, year_idx]
        previous = data[:, year_idx - 1]
        mask = (current == 0) & (previous != 0)
        current[mask] = previous[mask]
        data[:, year_idx] = current


def fill_categorical_holes(batch: np.ndarray) -> np.ndarray:
    filled = batch.copy()
    for idx in range(batch.shape[0]):
        image = batch[idx]
        valid = image != 0
        if np.all(valid) or not np.any(valid):
            continue
        nearest = distance_transform_edt(1 - valid, return_distances=False, return_indices=True)
        filled[idx] = image[nearest[0], nearest[1]]
    return filled
