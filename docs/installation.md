# Installation

This guide describes the minimal environment needed to run the public PCR-Net demos and data utilities.

## Tested Baseline

- Python: 3.10 or newer.
- PyTorch: 2.10.0 was used during local validation.
- Operating systems: the current repository has been exercised on Windows. Linux/macOS users should expect path and geospatial-library differences and should report issues.

GPU execution is recommended for neural model training and inference. CPU execution is supported for small demos but will be slower. Install the PyTorch build that matches your CUDA driver if you use a GPU.

## Conda Installation

From the repository root:

```bash
conda env create -f environment.yml
conda activate pcrnet
```

If your platform cannot solve GDAL or PyTorch from the listed channels, install the geospatial stack and PyTorch following their official platform-specific instructions, then install the remaining packages from `requirements.txt`.

## Pip Installation

Create and activate a Python environment first, then run:

```bash
python -m pip install -r requirements.txt
```

The pip route is convenient for model-only demos. Geospatial packages such as GDAL and rasterio may require platform-specific wheels or Conda packages.

## Release Artifacts

Release-managed demo data and pretrained checkpoints are distributed with the
stable submission snapshot
[`v0.9.0`](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0).
After installing the environment, follow [quick_start.md](quick_start.md) for
the artifact download commands.

## Verify the Environment

Run:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import h5py, pandas, catboost, thop; print('core imports ok')"
```

For data-preparation workflows:

```bash
python -c "import rasterio, xarray, ee, geemap, cdsapi; print('data imports ok')"
```

## Minimal Post-Install Test

```bash
python examples/06_compute_profile/run_demo.py --dataset smoke_case --num-samples 1 --diffusion-steps 4
```

This command uses the tracked `smoke_case` dataset and does not require downloading Release artifacts.

## Common Installation Problems

**PyTorch cannot see CUDA.** Install the PyTorch build matching your CUDA driver. CPU mode is still usable for small checks.

**GDAL or rasterio fails to install with pip.** Prefer Conda packages from `conda-forge`.

**CatBoost GPU errors.** The public demos configure CatBoost CPU execution where needed. Full experiments may require additional GPU setup.

**Earth Engine authentication fails.** Run an interactive authentication flow and initialize Earth Engine with your own Google Cloud project before using GEE-backed data workflows.
