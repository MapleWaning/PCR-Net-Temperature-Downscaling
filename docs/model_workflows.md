# Model Workflows

This page is retained for compatibility with earlier documentation links. The
current workflow overview is in `docs/model_and_workflow.md`.

## Main Dependency Chain

```text
standard data products
  -> physical baseline
  -> CatBoost teacher guidance
  -> PCR-Net or baseline neural training
  -> evaluation, visualization, and demo outputs
```

## Source Entrypoints

The migrated model code remains under `src/open_source/`. The public demos call
these modules through repository helper code, but the lower-level scripts remain
available for advanced users:

```bash
python src/open_source/data_prepare/build_features.py --target catboost_lst --split year
python src/open_source/CatBoost/train.py --task-type CPU
python src/open_source/CatBoost/inference.py
python src/open_source/PCR-Net/train.py --generalization year --no-pretrained-backbone
python src/open_source/test/evaluate.py --model-path assets/pretrained/pcr_net/pcr-time.pth
python src/open_source/mapping/visualize.py --model-path assets/pretrained/pcr_net/pcr-time.pth
python src/open_source/ablation/no_attention.py --generalization year --no-pretrained-backbone
```

Use `PYTHONPATH=src` when running source scripts directly from the repository
root.

## Release Weights

Current Release-managed checkpoint aliases are:

```text
assets/pretrained/pcr_net/pcr-time.pth
assets/pretrained/pcr_net/pcr-spatial.pth
```

Download them with:

```bash
python scripts/download_release_artifact.py pcr_time
python scripts/download_release_artifact.py pcr_spatial
```

These checkpoints are for public workflow validation and pretrained demo
inference. They are distributed with the stable submission snapshot
[`v0.9.0`](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0).
