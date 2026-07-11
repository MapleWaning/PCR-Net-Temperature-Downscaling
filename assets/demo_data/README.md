# Prepared Demo Data

This directory stores model-ready demo datasets. It is not a replacement for
the complete research dataset.

## Datasets

| Dataset | Source | Purpose | Notes |
|---|---|---|---|
| `smoke_case/` | Tracked in Git | Fast interface checks | Four samples only. |
| `mini_case/` | GitHub Release artifact | Public demo workflow validation | Reduced 288-sample real subset. |

Both datasets keep the same inner layout as a prepared `data/processed` tree so
demos can switch between them with `--dataset smoke_case` or
`--dataset mini_case`.

## Download

Install the Release-managed mini-case with:

```bash
python scripts/download_release_artifact.py mini_case
```

The active artifact release is the stable submission snapshot
[`v0.9.0`](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0).
The downloader verifies size and SHA256 values from
`configs/artifacts/artifact_manifest.json`.

## Expected Layout

```text
assets/demo_data/
  smoke_case/
    manifest.json
    validation_summary.json
    standard/
    physical/
    catboost_inference/
    model_inputs/
  mini_case/
    manifest.json
    validation_summary.json
    standard/
    physical/
    catboost_inference/
    model_inputs/
```

The `mini_case` archive itself may also remain in this directory after download.
It is ignored by Git.

## Reproducibility Scope

`smoke_case` and `mini_case` are for software and workflow validation. Metrics
computed from these datasets are not manuscript metrics.
