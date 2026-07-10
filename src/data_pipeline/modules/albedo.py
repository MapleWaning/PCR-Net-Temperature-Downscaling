from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_pipeline.common.gee import initialize_gee
from data_pipeline.common.paths import ensure_dir
from data_pipeline.common.retry import retry_call
from data_pipeline.common.station import read_metadata
from data_pipeline.modules.base import ModuleResult


class AlbedoModule:
    name = "albedo"
    fill_value = -9999.0

    def plan(self, context) -> ModuleResult:
        cfg = context.config
        output_dir = self._output_dir(cfg)
        outputs = [
            output_dir / f"Albedo_100m_{year}_{month:02d}.npy"
            for year in range(cfg.start_year, cfg.end_year + 1)
            for month in range(1, 13)
        ]
        missing = [path for path in outputs if not path.exists()]
        return ModuleResult(self.name, outputs=outputs, details={"missing_count": len(missing)})

    def run(self, context) -> ModuleResult:
        cfg = context.config
        metadata = read_metadata(cfg.metadata_csv)
        output_dir = ensure_dir(self._output_dir(cfg))
        ee = initialize_gee(str(cfg.get("gee.project_id", "")) or None)

        written: list[Path] = []
        for year in range(cfg.start_year, cfg.end_year + 1):
            for month in range(1, 13):
                output_path = output_dir / f"Albedo_100m_{year}_{month:02d}.npy"
                expected_shape = (len(metadata), cfg.image_size, cfg.image_size)
                if output_path.exists():
                    try:
                        if tuple(np.load(output_path, mmap_mode="r").shape) == expected_shape:
                            written.append(output_path)
                            continue
                    except Exception:  # noqa: BLE001
                        pass
                batch = self._download_month_batch(ee, cfg, metadata, year, month)
                np.save(output_path, batch)
                written.append(output_path)

        return ModuleResult(self.name, outputs=written, details={"station_count": len(metadata)})

    def _output_dir(self, cfg) -> Path:
        return cfg.path(cfg.get("albedo.output_dir"), cfg.output_root / "albedo")

    def _download_month_batch(self, ee, cfg, metadata, year: int, month: int) -> np.ndarray:
        import geemap

        n_stations = len(metadata)
        batch = np.full((n_stations, cfg.image_size, cfg.image_size), self.fill_value, dtype=np.float32)
        half_size_m = cfg.image_size * cfg.target_res_m / 2.0
        max_retries = int(cfg.get("albedo.max_retries", 5))

        for idx, row in tqdm(metadata.iterrows(), total=n_stations, desc=f"Albedo {year}-{month:02d}"):
            lon = float(row["Longitude"])
            lat = float(row["Latitude"])

            def load_array():
                point = ee.Geometry.Point([lon, lat])
                roi = point.transform("EPSG:3857", 1).buffer(half_size_m).bounds()
                image = monthly_landsat_albedo(ee, year, month, roi, cfg.target_res_m)
                return geemap.ee_to_numpy(image, region=roi)

            try:
                arr = retry_call(load_array, attempts=max_retries)
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                h = min(arr.shape[0], cfg.image_size)
                w = min(arr.shape[1], cfg.image_size)
                batch[idx, :h, :w] = arr[:h, :w]
            except Exception:  # noqa: BLE001
                continue

        return batch


def preprocess_landsat(ee, image, sensor: str):
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 3).eq(0)
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 1).eq(0))
    )
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)

    if sensor in {"L8", "L9"}:
        blue = optical.select("SR_B2").rename("blue")
        red = optical.select("SR_B4").rename("red")
        nir = optical.select("SR_B5").rename("nir")
        swir1 = optical.select("SR_B6").rename("swir1")
        swir2 = optical.select("SR_B7").rename("swir2")
    else:
        blue = optical.select("SR_B1").rename("blue")
        red = optical.select("SR_B3").rename("red")
        nir = optical.select("SR_B4").rename("nir")
        swir1 = optical.select("SR_B5").rename("swir1")
        swir2 = optical.select("SR_B7").rename("swir2")

    albedo = (
        blue.multiply(0.356)
        .add(red.multiply(0.130))
        .add(nir.multiply(0.373))
        .add(swir1.multiply(0.085))
        .add(swir2.multiply(0.072))
        .subtract(0.0018)
        .rename("albedo")
    )
    return image.addBands(albedo.clamp(0, 1)).updateMask(mask).select("albedo").copyProperties(image, ["system:time_start"])


def landsat_collection(ee, start_date, end_date, bounds):
    def collection(product_id: str, sensor: str):
        return (
            ee.ImageCollection(product_id)
            .filterDate(start_date, end_date)
            .filterBounds(bounds)
            .map(lambda image: preprocess_landsat(ee, image, sensor))
        )

    collections = [
        collection("LANDSAT/LC09/C02/T1_L2", "L9"),
        collection("LANDSAT/LC08/C02/T1_L2", "L8"),
        collection("LANDSAT/LE07/C02/T1_L2", "L7"),
        collection("LANDSAT/LT05/C02/T1_L2", "L5"),
    ]
    merged = collections[0]
    for item in collections[1:]:
        merged = merged.merge(item)
    return merged


def monthly_landsat_albedo(ee, year: int, month: int, region, target_scale: float):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    current = landsat_collection(ee, start, end, region).median()
    fallback = landsat_collection(ee, start.advance(-1, "month"), end.advance(1, "month"), region).median()
    yearly = landsat_collection(ee, ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31), region).median()

    return (
        ee.ImageCollection([yearly, fallback, current])
        .mosaic()
        .unmask(AlbedoModule.fill_value)
        .toFloat()
        .reproject(crs="EPSG:3857", scale=target_scale)
    )
