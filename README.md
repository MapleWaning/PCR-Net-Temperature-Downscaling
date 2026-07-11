# PCR-Net Temperature Downscaling

PCR-Net is a physics-corrected and knowledge-distilled workflow for downscaling coarse 2 m air-temperature fields to 100 m sample patches. The repository combines a physically corrected temperature baseline, an LST-guided CatBoost teacher map, and a terrain-aware PCR-Net refinement model. High-resolution land-surface temperature (LST) is used to train the teacher model; PCR-Net inference uses the physical baseline and static/time covariates and does not require LST at inference time.

## Repository Status

The current public version is the stable submission snapshot [v0.9.0](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0). This release is intended for manuscript submission and public review. It is not yet the final paper-associated software release, and no specific future release tag has been predetermined.

## Main Capabilities

- Data acquisition and preprocessing interfaces for station observations, ERA5, DEM, land cover, albedo, and LST inputs.
- Physical-baseline generation for advanced temperature-base tensors.
- LST-enhanced CatBoost teacher training and teacher-map generation.
- PCR-Net training, testing, pretrained inference, and visualization.
- Ablation experiments for model components and loss terms.
- Baseline experiments for Random Forest, CatBoost without LST, and Basic U-Net.
- Computational profiling for Basic U-Net, PCR-Net, and a diffusion-style U-Net.

## Repository Structure

```text
configs/        Configuration examples and artifact manifest.
docs/           User, installation, data, reproducibility, and workflow guides.
examples/       Six runnable public demos.
scripts/        Artifact download and pipeline wrappers.
src/            Data pipeline, physical baseline, PCR-Net, baselines, and helpers.
assets/         Small tracked smoke data plus Release-managed demo data and weights.
data/           Local raw/interim/processed data; not tracked by Git.
outputs/        Generated demo outputs; not tracked by Git.
runs/           Training checkpoints and logs; not tracked by Git.
```

## Installation Summary

The recommended route is Conda:

```bash
conda env create -f environment.yml
conda activate pcrnet
```

A pip-oriented alternative is also provided:

```bash
python -m pip install -r requirements.txt
```

See [docs/installation.md](docs/installation.md) for CPU/GPU notes, PyTorch compatibility, and installation checks.

## Download Demo Artifacts

Release artifacts are installed with:

```bash
python scripts/download_release_artifact.py mini_case
python scripts/download_release_artifact.py pcr_time
python scripts/download_release_artifact.py pcr_spatial
```

The installed runtime paths are:

```text
assets/demo_data/mini_case/
assets/pretrained/pcr_net/pcr-time.pth
assets/pretrained/pcr_net/pcr-spatial.pth
```

The downloader uses [configs/artifacts/artifact_manifest.json](configs/artifacts/artifact_manifest.json) for asset names, sizes, SHA256 hashes, and installation paths. See [docs/data_and_artifacts.md](docs/data_and_artifacts.md) for the artifact table and manual fallback instructions.

## Quick Start

From the repository root:

```bash
python scripts/download_release_artifact.py mini_case
python scripts/download_release_artifact.py pcr_time
python examples/03_visualization/run_demo.py --dataset mini_case --version time
```

The visualization summary is written to:

```text
outputs/demos/03_visualization/mini_case/temporal/summary.json
```

Figures are written under:

```text
outputs/demos/03_visualization/mini_case/temporal/maps/
```

For the full first-run path, see [docs/quick_start.md](docs/quick_start.md).

## Demo Index

| Demo | Purpose | Main command | Required data/artifacts |
|---|---|---|---|
| [Demo 1](examples/01_data_fetch/README.md) | Minimal preprocessing and model-ready layout check. | `python examples/01_data_fetch/run_demo.py` | Built-in fixture only. |
| [Demo 2](examples/02_train_and_test/README.md) | CatBoost teacher plus PCR-Net training/testing. | `python examples/02_train_and_test/run_demo.py --dataset mini_case` | `mini_case`; downloads automatically when requested. |
| [Demo 3](examples/03_visualization/README.md) | Pretrained inference and visualization. | `python examples/03_visualization/run_demo.py --dataset mini_case --version time` | `mini_case` and matching PCR-Net checkpoint. |
| [Demo 4](examples/04_ablation_training/README.md) | Core ablation training matrix. | `python examples/04_ablation_training/run_demo.py --dataset mini_case` | `mini_case`; CatBoost guidance included or generated. |
| [Demo 5](examples/05_baseline_training_and_test/README.md) | Baseline model training/testing. | `python examples/05_baseline_training_and_test/run_demo.py --dataset mini_case` | `mini_case`. |
| [Demo 6](examples/06_compute_profile/README.md) | Computational profile for neural models. | `python examples/06_compute_profile/run_demo.py --dataset mini_case` | `mini_case` or `smoke_case`; no trained diffusion weights required. |

See [examples/README.md](examples/README.md) for dependencies among demos and output locations.

## Reproducibility Boundary

This repository supports three reproducibility levels:

- Software smoke tests with the tracked minimal fixture.
- Public demo workflow reproduction with the Release `mini_case` dataset and Release checkpoints.
- Manuscript numerical reproduction with the complete 118-station, 2008-2024 research dataset and full experimental settings.

The `mini_case` dataset is a reduced subset for workflow validation. Its metrics are not expected to match manuscript metrics. Exact manuscript-result reproduction requires third-party source data, full preprocessing, full training schedules, and the final paper-associated configuration set. See [docs/reproducibility.md](docs/reproducibility.md).

## Computing Requirements

The codebase targets Python 3.10 or newer. Local validation used PyTorch `2.10.0`. The model demos require PyTorch, TorchVision, CatBoost, scikit-learn, pandas, NumPy, h5py, PyArrow, Matplotlib, and THOP for Demo 6. Data preparation workflows additionally use geospatial and remote-data packages such as rasterio, GDAL, xarray, pyproj, Earth Engine, geemap, and cdsapi.

GPU execution is recommended for neural training and inference, but the demos are configured with small batch sizes and can run on CPU with reduced speed. Demo 6 peak-memory values are hardware, driver, CUDA, PyTorch, precision, and system-condition dependent; theoretical FLOPs are more stable across hardware.

## Data Availability and Restrictions

The Release provides the reduced `mini_case` dataset and two PCR-Net checkpoints. The complete third-party source datasets are not redistributed in this repository. They remain subject to their providers' access procedures, licences, terms of use, and attribution requirements. Manual acquisition instructions are documented in [docs/manual_downloads.md](docs/manual_downloads.md).

## License

Repository code is released under the [MIT License](LICENSE). Third-party data and derived artifacts may be subject to separate provider terms and do not inherit the repository software license.

## Citation

Use [CITATION.cff](CITATION.cff) to cite this software repository. The manuscript has not been assigned a public DOI in this repository documentation.

## Support

Use GitHub Issues for bug reports, unclear documentation, installation problems, or reproducibility questions.
