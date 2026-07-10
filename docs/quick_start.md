# Quick Start

This is the shortest path for a first-time user who wants to verify the public repository workflows.

## 1. Clone and Enter the Repository

```bash
git clone https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling.git
cd PCR-Net-Temperature-Downscaling
```

## 2. Create the Environment

```bash
conda env create -f environment.yml
conda activate pcrnet
```

See [installation.md](installation.md) if your platform needs a custom PyTorch or GDAL installation.

## 3. Download Public Artifacts

```bash
python scripts/download_release_artifact.py mini_case
python scripts/download_release_artifact.py pcr_time
```

This installs:

```text
assets/demo_data/mini_case/
assets/pretrained/pcr_net/pcr-time.pth
```

## 4. Run a Recommended First Demo

```bash
python examples/03_visualization/run_demo.py --dataset mini_case --version time
```

Expected summary:

```text
outputs/demos/03_visualization/mini_case/temporal/summary.json
```

Expected figures:

```text
outputs/demos/03_visualization/mini_case/temporal/maps/
```

## 5. Run the Full Demo Index

After the quick visualization check, follow [examples/README.md](../examples/README.md) for all six demos.

The public demos validate interfaces and workflow behavior. They are not expected to reproduce the manuscript metrics.
