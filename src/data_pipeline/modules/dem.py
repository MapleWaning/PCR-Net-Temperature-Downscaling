from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.raster import aeqd_wkt, srtm_tile
from data_pipeline.common.station import read_metadata
from data_pipeline.modules.base import ModuleResult


class DemModule:
    name = "dem"

    def plan(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv) if cfg.metadata_csv.exists() else pd.DataFrame()
        raw_dir = cfg.path(cfg.get("dem.raw_dir"), cfg.raw_root / "dem" / "cgiar_srtm")
        tiles = sorted(required_srtm_tiles(metadata, float(cfg.get("dem.buffer_degree", 0.5))))
        rows = []
        for col, row in tiles:
            filename = f"srtm_{col:02d}_{row:02d}.zip"
            path = raw_dir / filename
            rows.append({"tile_col": col, "tile_row": row, "file": str(path), "exists": path.exists()})
        report = cfg.output_root / "reports" / "dem_required_tiles.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report, index=False)
        return ModuleResult(self.name, reports=[report], details={"tile_count": len(rows), "missing_count": sum(not r["exists"] for r in rows)})

    def run(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv)
        raw_dir = ensure_dir(cfg.path(cfg.get("dem.raw_dir"), cfg.raw_root / "dem" / "cgiar_srtm"))
        output_dir = ensure_dir(cfg.path(cfg.get("dem.output_dir"), cfg.output_root / "dem"))
        mosaic_path = cfg.path(cfg.get("dem.mosaic_path"), cfg.raw_root / "dem" / "full_region_dem.tif")

        tiles = sorted(required_srtm_tiles(metadata, float(cfg.get("dem.buffer_degree", 0.5))))
        for tile in tiles:
            ensure_srtm_zip(raw_dir, tile)

        if not mosaic_path.exists():
            unzip_all(raw_dir)
            stitch_tifs(raw_dir, mosaic_path)

        arrays = crop_dem_for_stations(mosaic_path, metadata, cfg.image_size, cfg.target_res_m)
        outputs = []
        for name, array in arrays.items():
            path = output_dir / f"{name}.npy"
            np.save(path, array.astype(np.float32))
            outputs.append(path)

        return ModuleResult(self.name, outputs=outputs, details={name: tuple(array.shape) for name, array in arrays.items()})


def required_srtm_tiles(metadata: pd.DataFrame, buffer_degree: float) -> set[tuple[int, int]]:
    tiles: set[tuple[int, int]] = set()
    for _, row in metadata.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        corners = [
            (lat - buffer_degree, lon - buffer_degree),
            (lat - buffer_degree, lon + buffer_degree),
            (lat + buffer_degree, lon - buffer_degree),
            (lat + buffer_degree, lon + buffer_degree),
        ]
        for corner_lat, corner_lon in corners:
            tile = srtm_tile(corner_lat, corner_lon)
            if tile is not None:
                tiles.add(tile)
    return tiles


def ensure_srtm_zip(raw_dir: Path, tile: tuple[int, int]) -> Path:
    col, row = tile
    filename = f"srtm_{col:02d}_{row:02d}.zip"
    path = raw_dir / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{filename}"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {response.status_code}")
    with path.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fh.write(chunk)
    return path


def unzip_all(raw_dir: Path) -> None:
    for path in raw_dir.glob("*.zip"):
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if all((raw_dir / name).exists() for name in names):
                continue
            archive.extractall(raw_dir)


def stitch_tifs(raw_dir: Path, output_path: Path) -> None:
    import rasterio
    from rasterio.merge import merge

    tif_files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.TIF")))
    if not tif_files:
        raise FileNotFoundError(f"No SRTM tif files found in {raw_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = [rasterio.open(path) for path in tif_files]
    try:
        mosaic, transform = merge(sources)
        meta = sources[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "crs": sources[0].crs,
                "compress": "lzw",
            }
        )
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic)
    finally:
        for src in sources:
            src.close()


def crop_dem_for_stations(mosaic_path: Path, metadata: pd.DataFrame, image_size: int, target_res_m: float) -> dict[str, np.ndarray]:
    from osgeo import gdal

    buffer = 2
    process_size = image_size + 2 * buffer
    n_stations = len(metadata)
    elevation = np.full((n_stations, image_size, image_size), np.nan, dtype=np.float32)
    slope = np.full_like(elevation, np.nan)
    aspect_sin = np.zeros_like(elevation)
    aspect_cos = np.ones_like(elevation)

    for idx, row in tqdm(metadata.iterrows(), total=n_stations, desc="DEM crop"):
        try:
            e, s, sin_a, cos_a = crop_one_station(mosaic_path, float(row["Latitude"]), float(row["Longitude"]), process_size, image_size, buffer, target_res_m)
            elevation[idx] = e
            slope[idx] = s
            aspect_sin[idx] = sin_a
            aspect_cos[idx] = cos_a
        except Exception:  # noqa: BLE001
            continue

    return {
        "dem_elevation": np.nan_to_num(elevation, nan=0.0),
        "dem_slope": np.nan_to_num(slope, nan=0.0),
        "dem_aspect_sin": np.nan_to_num(aspect_sin, nan=0.0),
        "dem_aspect_cos": np.nan_to_num(aspect_cos, nan=1.0),
    }


def crop_one_station(mosaic_path: Path, lat: float, lon: float, process_size: int, image_size: int, buffer: int, target_res_m: float):
    from osgeo import gdal

    half_dist = process_size * target_res_m / 2.0
    warp_options = gdal.WarpOptions(
        format="VRT",
        dstSRS=aeqd_wkt(lat, lon),
        outputBounds=[-half_dist, -half_dist, half_dist, half_dist],
        width=process_size,
        height=process_size,
        resampleAlg=gdal.GRA_Bilinear,
    )
    elev_ds = gdal.Warp("", str(mosaic_path), options=warp_options)
    if elev_ds is None:
        raise RuntimeError("GDAL Warp failed")

    elev = elev_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    slope_ds = gdal.DEMProcessing("", elev_ds, "slope", options=gdal.DEMProcessingOptions(format="MEM", computeEdges=True))
    aspect_ds = gdal.DEMProcessing("", elev_ds, "aspect", options=gdal.DEMProcessingOptions(format="MEM", computeEdges=True))
    slope = slope_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    aspect = aspect_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    start = buffer
    end = buffer + image_size
    aspect_rad = np.radians(aspect[start:end, start:end])
    return (
        elev[start:end, start:end],
        slope[start:end, start:end],
        np.sin(aspect_rad),
        np.cos(aspect_rad),
    )
