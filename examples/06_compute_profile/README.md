# Demo 06: Computational Profile

## Purpose

Compare simple inference compute characteristics for Basic U-Net, PCR-Net, and a diffusion-style U-Net.

## What This Demo Does

- Builds sample input tensors from a prepared dataset.
- Initializes Basic U-Net, PCR-Net, and diffusion-style U-Net structures.
- Counts trainable parameters.
- Estimates serialized model state-dict size.
- Uses THOP to compute FLOPs per network evaluation.
- Measures CUDA peak memory when CUDA is available.
- Writes a CSV benchmark table and JSON summary.

## What This Demo Does Not Do

It does not train the diffusion-style model. It does not report manuscript benchmark values directly; users should inspect the generated CSV.

## Requirements

PyTorch, TorchVision, h5py, pandas, and THOP.

## Input Data

`smoke_case` or `mini_case`. The default sample count is capped by the available sample pool. `smoke_case` has four samples.

## Main Command

```bash
python examples/06_compute_profile/run_demo.py --dataset mini_case
```

## Important CLI Options

- `--dataset smoke_case|mini_case`
- `--data-root <prepared-data-root>`
- `--num-samples`
- `--diffusion-steps`
- `--diffusion-base-channels`
- `--diffusion-num-res-blocks`
- `--amp-memory`
- `--device auto|cpu|cuda`

## Processing Steps

1. Load model-ready sample tensors.
2. Build 20-channel neural inputs.
3. Build 23-channel diffusion inputs by concatenating `y_t` and the 20-channel condition.
4. Profile one network evaluation.
5. Multiply diffusion FLOPs by number of function evaluations (NFE).

Effective FLOPs = FLOPs per network evaluation x number of function evaluations (NFE).

## Expected Outputs

```text
outputs/demos/06_compute_profile/<case>/summary.json
outputs/demos/06_compute_profile/<case>/complexity_results.csv
```

## Runtime and Hardware

FLOPs and parameter counts are theoretical. Peak memory depends on GPU, driver, PyTorch, CUDA, precision, and system conditions. CPU runs report no CUDA peak-memory value.

## Relationship to the Manuscript

This demo reproduces the public benchmarking procedure, not a fixed manuscript table. Use the generated CSV for the current environment.

## Limitations

Latency is not currently written by this demo. Diffusion-style models can be benchmarked structurally without trained weights.

## Troubleshooting

If THOP is missing, install the model dependencies. If CUDA is unavailable, run with `--device cpu` or let `--device auto` select CPU.
