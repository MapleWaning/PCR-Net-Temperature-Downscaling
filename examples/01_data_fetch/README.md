# Demo 01: Minimal Data Fetch/Preprocess

This demo creates one local station fixture and runs the real `data_pipeline` GSOD path for two days.
It also writes a model-ready demo dataset with the same internal layout used by the prepared demo datasets.

```bash
python examples/01_data_fetch/run_demo.py
```

It writes outputs under `outputs/demos/01_data_fetch/`.

Model-ready outputs are written under:

```text
outputs/demos/01_data_fetch/model_ready/
  manifest.json
  validation_summary.json
  model_inputs/static.npy
  model_inputs/truth.npy
  model_inputs/splits/*.csv
  model_inputs/splits/temporal_years.json
  model_inputs/reports/replacement_report.csv
  standard/era5/era5_t2m_2008.h5
  standard/era5/era5_t2m_2009.h5
  standard/era5/era5_t2m_2010.h5
  standard/lst/lst_2008.h5
  standard/lst/lst_2009.h5
  standard/lst/lst_2010.h5
  physical/t_base/t_base_advanced_2008.h5
  physical/t_base/t_base_advanced_2009.h5
  physical/t_base/t_base_advanced_2010.h5
  fixture_manifest.json
```

Demo 01 does not write `catboost_inference`. Demo 02 trains the LST CatBoost model
and writes `catboost_inference` into this dataset when the model demos use Demo 01 fallback data.

For network-backed experiments, pass a proxy explicitly:

```bash
python examples/01_data_fetch/run_demo.py --proxy http://127.0.0.1:7890
```
