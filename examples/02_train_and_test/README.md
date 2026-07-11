# Demo 02: CatBoost Teacher and PCR-Net Training/Testing

## Purpose

Run the main model-side workflow on a prepared demo dataset.

## What This Demo Does

1. Builds LST CatBoost parquet features.
2. Trains the LST CatBoost residual model.
3. Runs sample-limited CatBoost inference to write teacher maps.
4. Trains PCR-Net with CatBoost guidance.
5. Evaluates the produced PCR-Net checkpoint.

## What This Demo Does Not Do

It does not run full manuscript training schedules. Mini-case metrics are workflow-validation results and are not expected to match the manuscript metrics.

## Requirements

Model dependencies: PyTorch, TorchVision, CatBoost, PyArrow, pandas, h5py, and scikit-learn.

## Input Data

Default: `assets/demo_data/smoke_case/`.

Recommended public dataset: `assets/demo_data/mini_case/`.

Fallback: `outputs/demos/01_data_fetch/model_ready/` when prepared datasets are unavailable.

## Main Command

```bash
python examples/02_train_and_test/run_demo.py --dataset mini_case
```

## Important CLI Options

- `--dataset smoke_case|mini_case`
- `--data-root <prepared-data-root>`
- `--version time|spatial`
- `--split-mode temporal|spatial`
- `--smoke-only`

## PCR-Net Loss Parameters

Demo 2 trains PCR-Net through `src/open_source/PCR-Net/train.py`. The underlying
PCR-Net training entrypoint includes two tunable loss parameters:

- `--lambda-grad`: weight for the CatBoost teacher-guidance gradient loss term.
- `--alpha-terrain`: weight used to amplify station supervision in complex
  terrain based on the DEM slope channel.

The repository defaults are currently:

```text
LAMBDA_GRAD = 0.1
ALPHA_TERRAIN = 0.1
```

These values are demo/default settings that show the two parameters exist in
the public training code. They should not be interpreted as optimal values or as
the final manuscript hyperparameter setting. The Demo 2 wrapper uses these
defaults unless the lower-level PCR-Net training command is run with explicit
overrides.

## Processing Steps

The demo writes features, CatBoost models, teacher maps, PCR-Net runs, metrics, and a summary under:

```text
outputs/demos/02_train_and_test/<case>/<split>/
```

## Expected Outputs

```text
summary.json
parquet/catboost_lst/<split>/*.parquet
catboost/catboost_lst.cbm
catboost_inference/catboost_lst/cb_t2m_YYYY.h5
runs/*/best_model.pth
metrics.csv
```

For Demo 1 fallback data, CatBoost teacher maps may be written into the fallback data root.

## Runtime and Hardware

GPU is recommended for PCR-Net training, but the demo uses reduced epochs and small batches. No runtime estimate is documented.

## Relationship to the Manuscript

This is the closest public demo to the main PCR-Net workflow. Full manuscript metrics require the complete research dataset and full settings.

## Limitations

The demo uses reduced training parameters. It may reuse or overwrite files in its output directory.

## Troubleshooting

If CatBoost has too few samples, use `--dataset mini_case`. If GPU memory is limited, keep the demo batch-size defaults or run on CPU.
