# Model Workflows

## Main Training Dependency Chain

```text
data_pipeline
  -> data/processed/standard
physical_base
  -> data/processed/physical
CatBoost or RF inference
  -> data/processed/catboost_inference
PCR-Net training
  -> runs/
evaluation and visualization
  -> outputs/
```

## Existing Entrypoints

The migrated model code remains under `src/open_source` to preserve existing imports.

Typical entries:

```bash
python src/open_source/data_prepare/build_features.py --target catboost_lst --split year
python src/open_source/CatBoost/train.py --task-type CPU
python src/open_source/CatBoost/inference.py
python src/open_source/PCR-Net/train.py --generalization year --no-pretrained-backbone
python src/open_source/test/evaluate.py --model-path assets/pretrained/pcr_net/final_model.pth
python src/open_source/mapping/visualize.py --model-path assets/pretrained/pcr_net/final_model.pth
python src/open_source/ablation/no_attention.py --generalization year --no-pretrained-backbone
```

Use `PYTHONPATH=src` when running scripts directly from source.

## Release Weights

Full PCR-Net checkpoints are release artifacts. They should be downloaded into:

```text
assets/pretrained/pcr_net/
  best_model.pth
  final_model.pth
  metadata.json
```

The repository keeps this directory documented but does not commit large weight files.
