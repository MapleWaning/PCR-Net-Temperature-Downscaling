# Demo 05: Baseline Training and Testing

## Purpose

Run baseline model workflows on a prepared demo dataset.

## What This Demo Does

It trains/tests:

- Random Forest without LST.
- CatBoost without LST.
- Basic U-Net with MSE loss.

## What This Demo Does Not Do

It does not use LST or CatBoost teacher maps. It does not implement every possible manuscript baseline if a baseline is not listed above.

## Requirements

PyTorch, TorchVision, CatBoost, scikit-learn, joblib, PyArrow, pandas, and h5py.

## Input Data

Baseline inputs use ERA5-style HDF5 files:

```text
assets/demo_data/<case>/standard/era5/era5_t2m_YYYY.h5
```

## Main Command

```bash
python examples/05_baseline_training_and_test/run_demo.py --dataset mini_case
```

## Important CLI Options

- `--dataset smoke_case|mini_case`
- `--version time|spatial`
- `--data-root <prepared-data-root>`
- `--smoke-only`

## Processing Steps

The demo builds no-LST feature tables, trains RF and CatBoost baselines, trains Basic U-Net, and evaluates the Basic U-Net checkpoint.

## Expected Outputs

```text
outputs/demos/05_baseline_training_and_test/<case>/<split>/summary.json
outputs/demos/05_baseline_training_and_test/<case>/<split>/models/
outputs/demos/05_baseline_training_and_test/<case>/<split>/runs/
outputs/demos/05_baseline_training_and_test/<case>/<split>/baseline_unet_metrics.csv
```

## Runtime and Hardware

GPU is recommended for U-Net training. RF and CatBoost demo commands use reduced settings. No runtime estimate is documented.

## Relationship to the Manuscript

The demo checks baseline code paths. Manuscript baseline metrics require the full research dataset and full settings.

## Limitations

Fairness conditions in the demo are reduced for quick workflow validation. Baselines do not use teacher maps.

## Troubleshooting

If RF or CatBoost training has too few samples, use `--dataset mini_case`.
