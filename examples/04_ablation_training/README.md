# Demo 04: Ablation Experiments

## Purpose

Run the core PCR-Net ablation matrix on a prepared demo dataset.

## What This Demo Does

It trains:

- `no_attention`: PCR-Net without attention and SFT, still using CatBoost guidance.
- `no_gradient_loss`: PCR-Net architecture trained with pure MSE, without gradient guidance loss.

## What This Demo Does Not Do

It does not retrain CatBoost. It does not run the full manuscript ablation schedule.

## Requirements

PyTorch, TorchVision, h5py, and pandas.

## Input Data

Prepared data from `smoke_case`, `mini_case`, or a compatible `--data-root`. The `no_attention` variant requires CatBoost guidance maps.

## Main Command

```bash
python examples/04_ablation_training/run_demo.py --dataset mini_case
```

## Important CLI Options

- `--dataset smoke_case|mini_case`
- `--version time|spatial`
- `--data-root <prepared-data-root>`
- `--smoke-only`

## Processing Steps

The demo builds an ablation matrix and executes each variant when inputs and dependencies exist.

## Expected Outputs

```text
outputs/demos/04_ablation_training/<case>/<split>/summary.json
outputs/demos/04_ablation_training/<case>/<split>/runs/
```

## Runtime and Hardware

GPU is recommended for training. No runtime estimate is documented.

## Relationship to the Manuscript

Ablation trends on the reduced mini-case dataset may differ from those obtained using the complete research dataset.

## Limitations

Only the implemented matrix is run by this demo. Other ablation scripts may exist under `src/open_source/ablation/` but are not part of this public demo matrix unless listed above.

## Troubleshooting

If guidance is missing, run Demo 2 first for the same dataset or use the prepared `mini_case` data.
