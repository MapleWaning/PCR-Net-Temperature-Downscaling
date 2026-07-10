# Manual Inputs

Some datasets are intentionally handled as prepared local inputs.

## CCI / GLC_FCS30D

Run:

```bash
python -m data_pipeline.cli plan --config configs/example_case.yaml --modules cci
```

The plan writes `processed/reports/cci_manual_download_manifest.csv`. Download every listed `GLC_FCS30D_20002022_{tile}_Annual.tif` file and place it in `manual_inputs.cci_tif_dir`.

## MODIS LST

Run:

```bash
python -m data_pipeline.cli plan --config configs/example_case.yaml --modules lst
```

The plan writes `processed/reports/lst_manual_tasks.csv`. The first migrated version expects one GeoTIFF per station and year in `manual_inputs.lst_tif_dir`, using the configured `lst.file_pattern`.

Default pattern:

```text
{year}_Station_{station_id}.tif
```

## ERA5 Physical Inputs

The physical workflow expects:

- `physical_raw.era5_surface_geopotential`
- `physical_raw.era5_pressure_monthly`

Use `python -m physical_base.cli plan --config configs/physical.yaml` to check whether these files and the DATA-stage inputs are present.
