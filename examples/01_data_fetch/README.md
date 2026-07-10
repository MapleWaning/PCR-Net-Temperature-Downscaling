# Demo 01: Data Fetch and Preprocessing

## Purpose

Validate the preprocessing interface and write a model-ready fallback layout.

## What This Demo Does

- Creates a tiny local GSOD fixture.
- Runs the real `data_pipeline` GSOD path for two days.
- Writes synthetic model-ready tensors with the same inner layout used by prepared demo datasets.

## What This Demo Does Not Do

Demo 1 validates the preprocessing interface and standard sample format. It is not a complete automatic downloader for every third-party dataset used in the manuscript.

## Requirements

Core Python dependencies plus pandas, NumPy, h5py, and PyYAML.

## Input Data

The demo creates its own tiny fixture under `outputs/demos/01_data_fetch/`.

## Main Command

```bash
python examples/01_data_fetch/run_demo.py
```

Optional proxy:

```bash
python examples/01_data_fetch/run_demo.py --proxy http://127.0.0.1:7890
```

## Important CLI Options

- `--proxy`: sets `HTTP_PROXY` and `HTTPS_PROXY` for network-backed checks.

## Processing Steps

1. Write one local station metadata fixture.
2. Write two daily station observation rows.
3. Run `scripts/run_pipeline.py plan` for GSOD.
4. Run `scripts/run_pipeline.py run` for GSOD.
5. Create a model-ready fixture for 2008, 2009, and 2010.

## Expected Outputs

```text
outputs/demos/01_data_fetch/summary.json
outputs/demos/01_data_fetch/model_ready/
```

The model-ready directory contains `static.npy`, `truth.npy`, split CSVs, ERA5-style HDF5 files, LST HDF5 files, and physical-base HDF5 files.

## Runtime and Hardware

No GPU is required. No runtime estimate is documented.

## Relationship to the Manuscript

This demo checks data interfaces only. It is not a manuscript data reconstruction.

## Limitations

Demo 1 does not write `catboost_inference`. Run Demo 2 if a fallback dataset needs CatBoost teacher maps.

## Troubleshooting

If downstream demos cannot find CatBoost guidance in Demo 1 fallback data, run Demo 2 with `--data-root outputs/demos/01_data_fetch/model_ready`.
