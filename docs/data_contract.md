# Data Contract

The repository separates generated products by producer and by training responsibility.

```text
data/processed/
  standard/
    metadata/high-quality-meta.csv
    gsod/weather_data.npy
    albedo/Albedo_100m_YYYY_MM.npy
    era5/era5_t2m_YYYY.h5
    lst/lst_YYYY.h5
    dem/dem_elevation.npy
    dem/dem_slope.npy
    dem/dem_aspect_sin.npy
    dem/dem_aspect_cos.npy
    cci/GLC_FCS30D_YYYY.npy
    reports/
  physical/
    factors/*.h5
    t_base/t_base_advanced_YYYY.h5
    reports/
  catboost_inference/
    catboost_lst/cb_t2m_YYYY.h5
    baseline_cb/cb_t2m_YYYY.h5
    baseline_rf/rf_t2m_YYYY.h5
  model_inputs/
    static.npy
    truth.npy
    splits/*.csv
    parquet/
```

PCR-Net training consumes:

1. Standard data from `data/processed/standard`.
2. Advanced physical base from `data/processed/physical`.
3. Tree-model guidance maps from `data/processed/catboost_inference`.

`model_inputs` may duplicate or reorganize paths for compatibility with existing entry points. It must not change the processing, training, testing, or inference logic.

## Station Order

`metadata/high-quality-meta.csv` is the only authority for station order. All first station dimensions must match its row order. Code must derive station counts from metadata rows.

## Demo Data

Prepared demo data is reserved for `assets/demo_data`. It is distinct from downloaded products in `data/processed`, but should keep the same inner contract so either root can train:

```bash
PCR_DATA_ROOT=data/processed
PCR_DATA_ROOT=assets/demo_data/mini_case
```
