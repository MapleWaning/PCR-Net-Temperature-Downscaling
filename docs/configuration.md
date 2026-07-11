# Configuration Reference

This page summarizes the important configuration files. Demo settings are intentionally reduced and should not be treated as manuscript settings.

## `configs/data/default.yaml`

| Field | Meaning |
|---|---|
| `project.raw_root` | Local directory for downloaded or manually supplied source files. |
| `project.output_root` | Processed output directory. |
| `project.temp_root` | Temporary download/processing directory. |
| `station.metadata_csv` | Station metadata CSV. |
| `station.history_csv` | NOAA ISD station-history source. |
| `station.region` | Latitude/longitude bounds for station selection. |
| `station.coverage` | Required station-history coverage window. |
| `station.quality.max_missing_rate` | Maximum allowed missing-rate threshold for station observations. |
| `time.start_year`, `time.end_year` | Study years. |
| `grid.target_res_m` | Target grid spacing in meters. |
| `grid.image_size` | Patch size in pixels. |
| `modules.*` | Enable or disable pipeline modules. |
| `gee.project_id` | User's Earth Engine project ID. |
| `era5.download_times` | ERA5 single-level daily times. |
| `lst.file_pattern` | Expected station-year LST GeoTIFF naming pattern. |
| `manual_inputs.*` | Directories for manually acquired products. |

## `configs/physical/physical.yaml`

| Field | Meaning |
|---|---|
| `metadata_csv` | Station metadata used by the physical pipeline. |
| `inputs.dem_elevation` | Processed DEM elevation array. |
| `inputs.era5_temp_dir` | ERA5 temperature HDF5 directory. |
| `physical_raw.era5_surface_geopotential` | Local ERA5 surface geopotential NetCDF. |
| `physical_raw.era5_pressure_monthly` | Local ERA5 pressure-level monthly NetCDF. |
| `outputs.factor_dir` | Physical factor HDF5 directory. |
| `outputs.tbase_dir` | Advanced temperature-base output directory. |
| `time.climatology_years` | Years used to build monthly lapse-rate climatology. |
| `era5_pressure.pressure_levels` | Pressure levels used for lapse-rate estimation. |
| `physics.*` | Constants and thresholds for topography and inversion corrections. |

Some checked-in example configs contain local absolute paths from development. Treat them as templates and replace paths with your own local directories.

## `configs/model/default.yaml`

| Field | Meaning |
|---|---|
| `data_root` | Root for prepared model data. |
| `standard_root` | Standard data products. |
| `physical_root` | Physical products. |
| `catboost_inference_root` | Tree-model guidance maps. |
| `model_inputs_root` | Static features, labels, and split CSVs. |
| `years.*` | Reduced demo year settings. |
| `training.*` | Reduced demo training parameters. |

## PCR-Net Training Loss Parameters

The lower-level PCR-Net training entrypoint also reads defaults from `config.py`:

| CLI option | Default symbol | Current value | Meaning |
|---|---|---:|---|
| `--lambda-grad` | `LAMBDA_GRAD` | `0.1` | Weight for the CatBoost teacher-guidance gradient loss term. |
| `--alpha-terrain` | `ALPHA_TERRAIN` | `0.1` | Weight for terrain-amplified station supervision based on DEM slope. |

These values are repository demo/default values. They document the existence of
the two adjustable parameters and should not be treated as optimized or final
paper-associated hyperparameters.

## `configs/artifacts/artifact_manifest.json`

Defines Release tag, asset names, sizes, SHA256 hashes, download paths, and installed runtime paths. The downloader script reads this manifest directly.

## Demo CLI Configuration

Demos 2-5 share `--dataset`, `--data-root`, `--split-mode`, and `--version`. Demo 6 uses `--dataset` and benchmark-specific options instead of temporal/spatial version selection.
