# Demo 03: Pretrained Inference and Visualization

## Purpose

Render sample-level PCR-Net prediction maps from prepared data and a checkpoint.

## What This Demo Does

- Selects a temporal or spatial PCR-Net checkpoint.
- Builds the visualization dataset.
- Renders one sample-level figure.
- Writes a JSON summary.

## What This Demo Does Not Do

It does not retrain PCR-Net.

## Requirements

PyTorch, TorchVision, Matplotlib, h5py, and pandas.

## Input Data

Prepared dataset plus CatBoost guidance. The selected pretrained checkpoint is:

- `assets/pretrained/pcr_net/pcr-time.pth` for temporal/time.
- `assets/pretrained/pcr_net/pcr-spatial.pth` for spatial.

If no Release checkpoint exists, the demo falls back to a matching Demo 2 checkpoint.

## Main Command

```bash
python examples/03_visualization/run_demo.py --dataset mini_case --version time
```

## Important CLI Options

- `--dataset smoke_case|mini_case`
- `--version time|spatial`
- `--model-path <checkpoint>`
- `--data-root <prepared-data-root>`

## Processing Steps

The demo selects a target sample, loads a model, runs inference, and calls the real visualizer.

## Expected Outputs

```text
outputs/demos/03_visualization/<case>/<split>/summary.json
outputs/demos/03_visualization/<case>/<split>/maps/
```

## Runtime and Hardware

GPU is recommended but not required for a single sample. No runtime estimate is documented.

## Relationship to the Manuscript

Figures validate the public visualization workflow. They are not final manuscript figures.

## Limitations

The default target sample is station `36982099999` on `2023-07-20` when available; otherwise the first available sample is used.

## Troubleshooting

If no checkpoint exists, download `pcr_time` or `pcr_spatial`, or pass `--model-path`.
