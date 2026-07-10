from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def station_aeqd_grids(stations_df, image_size: int, target_res_m: float):
    import xarray as xr
    from pyproj import CRS, Transformer

    half_size_m = image_size * target_res_m / 2.0
    x = np.linspace(-half_size_m, half_size_m, image_size)
    y = np.linspace(-half_size_m, half_size_m, image_size)
    xx, yy = np.meshgrid(x, y)

    all_lats = np.zeros((len(stations_df), image_size, image_size), dtype=np.float32)
    all_lons = np.zeros_like(all_lats)
    crs_wgs84 = CRS.from_epsg(4326)

    for idx, row in stations_df.reset_index(drop=True).iterrows():
        lat_center = float(row["Latitude"])
        lon_center = float(row["Longitude"])
        local = CRS.from_string(
            f"+proj=aeqd +lat_0={lat_center} +lon_0={lon_center} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        transformer = Transformer.from_crs(local, crs_wgs84, always_xy=True)
        lon_grid, lat_grid = transformer.transform(xx, yy)
        all_lats[idx] = lat_grid
        all_lons[idx] = lon_grid

    return (
        xr.DataArray(all_lats, dims=("station", "y", "x")),
        xr.DataArray(all_lons, dims=("station", "y", "x")),
    )


def aeqd_wkt(lat: float, lon: float) -> str:
    return (
        'PROJCS["Local_AEQD",'
        'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
        'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
        'PROJECTION["Azimuthal_Equidistant"],'
        'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
        f'PARAMETER["Central_Meridian",{lon}],PARAMETER["Latitude_Of_Origin",{lat}],'
        'UNIT["Meter",1.0]]'
    )


def srtm_tile(lat: float, lon: float) -> tuple[int, int] | None:
    if not -60 <= lat <= 60:
        return None
    col = int(math.floor((lon + 180) / 5)) + 1
    row = int(math.floor((60 - lat) / 5)) + 1
    return col, row


def cci_tile_name(lat: float, lon: float) -> str:
    tile_lat = math.floor(lat / 5) * 5
    tile_lon = math.floor(lon / 5) * 5
    lon_prefix = "E" if tile_lon >= 0 else "W"
    lat_prefix = "N" if tile_lat >= 0 else "S"
    return f"{lon_prefix}{int(abs(tile_lon))}{lat_prefix}{int(abs(tile_lat))}"


def list_tifs(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
