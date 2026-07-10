# PCR-Net Temperature Downscaling

This repository packages the open-source release workflow for an academic temperature downscaling model. It contains three connected parts:

- `data_pipeline`: downloads or prepares station-scale source data and writes standard tensors.
- `physical_base`: builds physical prior factors and advanced temperature base tensors.
- `open_source`: trains CatBoost baselines, PCR-Net, U-Net baselines, ablations, evaluation, and visualization.

The model logic is preserved from the migrated modules. Repository-level files provide packaging, paths, demos, and documentation.

## Data Layout

Local generated data is not tracked by git.

```text
data/
  raw/                         downloaded or manually supplied source files
  interim/                     temporary files
  processed/
    standard/                  data_pipeline outputs
    physical/                  physical_base outputs
    catboost_inference/        CatBoost/RF guidance maps
    model_inputs/              adapter outputs consumed by training code
```

PCR-Net training uses three upstream product groups:

```text
data/processed/standard/
data/processed/physical/
data/processed/catboost_inference/
```

`data/processed/model_inputs/` is only a compatibility layer for existing training entry points.

## External Assets

Pretrained PCR-Net weights are managed through GitHub Releases, not committed to this repository:

```text
assets/pretrained/pcr_net/
```

Prepared demo data will live under:

```text
assets/demo_data/
```

`smoke_case/` is a tiny repository-managed dataset for quick checks. `mini_case/` is a Release-managed 288-sample dataset and should be downloaded into the same directory when available. If both prepared cases are unavailable, model demos can fall back to Demo 01 output at `outputs/demos/01_data_fetch/model_ready/`.

## Quick Smoke Demos

From the repository root:

```bash
python examples/01_data_fetch/run_demo.py
python examples/02_train_and_test/run_demo.py
python examples/03_visualization/run_demo.py
python examples/04_ablation_training/run_demo.py
python examples/05_baseline_training_and_test/run_demo.py
```

The first demo runs a real minimal GSOD preprocessing path using a tiny local fixture and writes a model-ready fallback layout. The model demos use `assets/demo_data/smoke_case` by default, can switch to the Release-managed `mini_case`, and fall back to Demo 01 model-ready data only when both prepared cases are unavailable.

## Important Contracts

- Metadata row order is the single source of truth for station order.
- Station count must be derived from metadata, not hard-coded.
- ERA5 outputs are already Celsius.
- `physical_base` reads standard data products and does not reprocess GSOD, DEM, or ERA5.
- CatBoost/RF inference outputs provide PCR-Net guidance maps.
- Existing model entries keep the `open_source.*` package name for compatibility.

See `docs/data_contract.md` and `docs/model_workflows.md` for more detail.
