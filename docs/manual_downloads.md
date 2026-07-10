# Manual Downloads and Third-Party Data

The complete third-party source datasets are not redistributed in this repository. They remain available from their respective providers and are subject to the providers' current access procedures, licences, terms of use, and attribution requirements. The repository distributes only a minimal fixture, a reduced demo subset, pretrained checkpoints, and derived model-ready samples required to demonstrate the public workflows.

## Automatically Downloaded Release Artifacts

The repository downloader handles:

```bash
python scripts/download_release_artifact.py mini_case
python scripts/download_release_artifact.py pcr_time
python scripts/download_release_artifact.py pcr_spatial
```

These are repository Release artifacts, not original third-party source products.

## GSOD Station Observations

- Provider: NOAA/NCEI.
- Product: Global Surface Summary of the Day (GSOD).
- Access: public HTTPS through repository code.
- Authentication: no API key is normally required.
- Study period: 2008-2024 for the full research setting.
- Local outputs: `standard/gsod/weather_data.npy` and station metadata under `standard/metadata/`.

The pipeline uses station metadata and daily observations to build mean, maximum, and minimum daily temperature labels.

## ERA5 2 m Air Temperature

- Provider: ECMWF/Copernicus Climate Data Store.
- CDS dataset: `reanalysis-era5-single-levels`.
- Variable: `2m_temperature`.
- Times requested by the repository default: `00:00`, `06:00`, `12:00`, `18:00`.
- Source format: NetCDF.
- Processed output: `standard/era5/era5_t2m_YYYY.h5`.
- HDF5 key: `data`.
- Processed units: Celsius.

Users need a CDS account, accepted product terms, a personal API token, and `cdsapi` configured locally.

## ERA5 Physical Inputs

The physical-baseline workflow expects local NetCDF inputs:

```text
physical_raw.era5_surface_geopotential
physical_raw.era5_pressure_monthly
```

The checked physical configuration uses:

- Surface geopotential dataset: `reanalysis-era5-single-levels`.
- Surface variable: `geopotential`.
- Example static request: year 2014, month 01, day 01, time 00:00.
- Pressure-level monthly dataset: `reanalysis-era5-pressure-levels-monthly-means`.
- Pressure-level variables: `geopotential`, `temperature`.
- Pressure levels: 500, 700, 850 hPa.
- Product type: `monthly_averaged_reanalysis`.
- Months: 01-12.
- Time: `00:00`.
- Climatology years in the checked configuration: 2008, 2009, 2010, 2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021, 2024.

These inputs are used to derive physical factors such as coarse geopotential height and monthly lapse-rate climatology. Private CDS credentials must never be committed.

## SRTM DEM

- Source: CGIAR SRTM zip files.
- Current repository implementation: direct CGIAR SRTM 5 x 5 degree zip download URLs.
- Expected raw directory: configured by `dem.raw_dir`, defaulting to `raw/dem/cgiar_srtm`.
- Filename pattern: `srtm_<col>_<row>.zip`.
- Processed outputs:
  - `standard/dem/dem_elevation.npy`
  - `standard/dem/dem_slope.npy`
  - `standard/dem/dem_aspect_sin.npy`
  - `standard/dem/dem_aspect_cos.npy`

The current repository implementation does not use Google Earth Engine for SRTM.

## MODIS LST

The research LST preparation uses Google Earth Engine task submission followed by Google Drive download.

- Terra collection: `MODIS/061/MOD11A1`.
- Aqua collection: `MODIS/061/MYD11A1`.
- Bands used: `LST_Day_1km`, `LST_Night_1km`.
- Export scale: 1000 m.
- Export CRS: `EPSG:4326`.
- Station buffer: 8000 m radius, producing approximately 16 x 16 pixels before repository upsampling.
- Export destination: Google Drive folder such as `LST_Dataset_2008_2024`.
- Repository input directory after download: configured by `manual_inputs.lst_tif_dir`, defaulting to `raw/lst/modis_exports`.
- Filename pattern expected by the repository: `{year}_Station_{station_id}.tif`.

Daily processing logic used in the reference GEE task:

- Terra and Aqua images are filtered by day.
- Day observations are averaged across Terra/Aqua when available.
- Night observations are averaged across Terra/Aqua when available.
- Daily maximum uses the maximum of Terra/Aqua daytime LST.
- Daily minimum uses the minimum of Terra/Aqua nighttime LST.
- Monthly climatology from the merged Terra/Aqua collection fills masked observations.
- MODIS scale factor `0.02` is applied and Kelvin values are converted to Celsius by subtracting `273.15`.
- Exported daily channels are TEMP, MAX, and MIN.

The repository `lst` module then reads the downloaded GeoTIFFs, reshapes them into daily 3-channel tensors, fills missing days, interpolates missing values over time, upsamples to 128 x 128, and writes `standard/lst/lst_YYYY.h5`.

## Landsat-Derived Albedo

- Access: Google Earth Engine.
- Collections:
  - `LANDSAT/LT05/C02/T1_L2`
  - `LANDSAT/LE07/C02/T1_L2`
  - `LANDSAT/LC08/C02/T1_L2`
  - `LANDSAT/LC09/C02/T1_L2`
- Optical scale: multiply by `0.0000275` and add `-0.2`.
- QA mask: excludes cloud shadow, cloud, and dilated cloud bits from `QA_PIXEL`.
- Formula:

```text
albedo = 0.356 * blue
       + 0.130 * red
       + 0.373 * nir
       + 0.085 * swir1
       + 0.072 * swir2
       - 0.0018
```

- Monthly composite: current-month median, with previous/next-month and yearly fallback mosaics.
- Fill value: `-9999.0`.
- Processed output pattern: `standard/albedo/Albedo_100m_YYYY_MM.npy`.

Albedo is a study-derived variable, not an official USGS Landsat albedo product.

## GLC_FCS30D Land Cover

- Provider: AIR-CAS/CASEarth.
- Product: GLC_FCS30D global 30 m fine-classification land-cover dynamics product.
- Access: manual download from the official product page.
- Expected raw directory: configured by `manual_inputs.cci_tif_dir`, defaulting to `raw/cci/GLC_FCS30D_30m`.
- Filename pattern: `GLC_FCS30D_20002022_<tile>_Annual.tif`.
- Processed output pattern: `standard/cci/GLC_FCS30D_YYYY.npy`.

The current code remaps source classes to 10 model classes with a lookup table. For years after 2022, the code reads band 23, effectively reusing the 2022 layer for 2023-2024.

Full research tile names confirmed from the 118-station preparation directory:

```text
GLC_FCS30D_20002022_E60N30_Annual.tif
GLC_FCS30D_20002022_E60N35_Annual.tif
GLC_FCS30D_20002022_E60N40_Annual.tif
GLC_FCS30D_20002022_E60N45_Annual.tif
GLC_FCS30D_20002022_E60N50_Annual.tif
GLC_FCS30D_20002022_E65N35_Annual.tif
GLC_FCS30D_20002022_E65N40_Annual.tif
GLC_FCS30D_20002022_E65N45_Annual.tif
GLC_FCS30D_20002022_E65N50_Annual.tif
GLC_FCS30D_20002022_E70N30_Annual.tif
GLC_FCS30D_20002022_E70N35_Annual.tif
GLC_FCS30D_20002022_E70N40_Annual.tif
GLC_FCS30D_20002022_E70N45_Annual.tif
GLC_FCS30D_20002022_E70N50_Annual.tif
GLC_FCS30D_20002022_E75N30_Annual.tif
GLC_FCS30D_20002022_E75N35_Annual.tif
GLC_FCS30D_20002022_E75N40_Annual.tif
GLC_FCS30D_20002022_E75N45_Annual.tif
GLC_FCS30D_20002022_E75N50_Annual.tif
GLC_FCS30D_20002022_E80N35_Annual.tif
GLC_FCS30D_20002022_E80N40_Annual.tif
GLC_FCS30D_20002022_E80N45_Annual.tif
GLC_FCS30D_20002022_E80N50_Annual.tif
GLC_FCS30D_20002022_E85N40_Annual.tif
GLC_FCS30D_20002022_E85N45_Annual.tif
GLC_FCS30D_20002022_E85N50_Annual.tif
GLC_FCS30D_20002022_E90N30_Annual.tif
GLC_FCS30D_20002022_E90N35_Annual.tif
GLC_FCS30D_20002022_E90N40_Annual.tif
GLC_FCS30D_20002022_E90N45_Annual.tif
GLC_FCS30D_20002022_E90N50_Annual.tif
GLC_FCS30D_20002022_E95N30_Annual.tif
GLC_FCS30D_20002022_E95N35_Annual.tif
GLC_FCS30D_20002022_E95N40_Annual.tif
GLC_FCS30D_20002022_E95N45_Annual.tif
GLC_FCS30D_20002022_E95N50_Annual.tif
```

Use the pipeline plan command to generate a case-specific manifest:

```bash
python scripts/run_pipeline.py plan --config configs/data/default.yaml --modules cci
```

## Validation

Use plan commands before full processing:

```bash
python scripts/run_pipeline.py plan --config configs/data/default.yaml --modules gsod,era5,dem,albedo,lst,cci
python scripts/run_physical.py plan --config configs/physical/physical.yaml
```

Replace config paths with your local data directories before running full-data workflows.
