# Configuration Files

This directory contains templates and manifests used by the data pipeline,
physical baseline, demos, model wrappers, and Release artifact downloader.

## Directory Map

```text
configs/
  artifacts/   Release artifact manifest.
  data/        Data-pipeline templates.
  demos/       Demo-specific reduced configuration.
  model/       Model/demo training defaults.
  physical/    Physical-baseline templates.
```

## Important Files

| File | Purpose |
|---|---|
| `configs/artifacts/artifact_manifest.json` | Release tag, asset names, sizes, SHA256 hashes, download paths, and runtime install paths. |
| `configs/data/default.yaml` | Full data-pipeline template. |
| `configs/data/example_case.yaml` | Example data-pipeline case. |
| `configs/demos/data_fetch_minimal.yaml` | Minimal Demo 1 fixture-generation settings. |
| `configs/model/default.yaml` | Reduced model/demo settings. |
| `configs/physical/physical.yaml` | Physical-baseline template. |

Some configuration templates may contain development example paths. Replace
them with local directories before running full data preparation. Public docs
use repository-relative paths for installed artifacts and demo outputs.

## Release Manifest

The artifact downloader reads `configs/artifacts/artifact_manifest.json`
directly:

```bash
python scripts/download_release_artifact.py mini_case
python scripts/download_release_artifact.py pcr_time
python scripts/download_release_artifact.py pcr_spatial
python scripts/download_release_artifact.py all
```

The current manifest points to the stable submission snapshot
[`v0.9.0`](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0).
This release is intended for manuscript submission and public review. It is not
yet the final paper-associated software release.

## Editing Guidance

Keep credentials, tokens, local secrets, and private provider keys out of
configuration files committed to Git. Use local copies or environment-specific
overrides for private paths and credentials.
