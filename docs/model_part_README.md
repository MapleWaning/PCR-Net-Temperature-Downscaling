# Model Source Notes

This legacy page is retained for users who followed earlier repository notes. The current user-facing model overview is in `docs/model_and_workflow.md`, and public demo usage is in `examples/README.md`.

## Current Model Scope

The migrated model code under `src/open_source/` covers:

- LST-guided CatBoost residual training and inference.
- Random Forest and no-LST CatBoost baselines.
- Basic U-Net baseline training.
- PCR-Net training with `ResAttUNet`, attention, SFT, and guided loss terms.
- Core ablation entries.
- Shared neural evaluation and sample-level visualization.

## Source Layout

```text
src/open_source/
  data_prepare/        Tabular feature building.
  CatBoost/            LST-guided CatBoost teacher.
  base_line/
    CB/                No-LST CatBoost baseline.
    RF/                No-LST Random Forest baseline.
    U-Net/             Basic U-Net baseline.
  PCR-Net/             PCR-Net training entrypoint.
  ablation/            PCR-Net ablation entrypoints.
  test/                Shared neural evaluation.
  mapping/             Sample-level visualization.
  unet_datasets.py     Neural dataset adapters.
  unet_models.py       `BasicUNet` and `ResAttUNet`.
  unet_losses.py       Baseline and guided losses.
  unet_training.py     Shared training loops.
```

Hyphenated directories such as `PCR-Net` and `U-Net` are script directories, not Python dotted-import package names.

## Typical Advanced Commands

```bash
python src/open_source/data_prepare/build_features.py --target catboost_lst --split year
python src/open_source/CatBoost/train.py --task-type CPU
python src/open_source/CatBoost/inference.py
python src/open_source/PCR-Net/train.py --generalization year --no-pretrained-backbone
python src/open_source/test/evaluate.py --model-path assets/pretrained/pcr_net/pcr-time.pth
python src/open_source/mapping/visualize.py --model-path assets/pretrained/pcr_net/pcr-time.pth
python src/open_source/ablation/no_attention.py --generalization year --no-pretrained-backbone
```

For most users, the `examples/` demos are the recommended entrypoints because they set paths, reduced demo schedules, artifact checks, and fallback behavior.

## Model Input Contract

The neural input has 20 channels:

```text
physical base (3) + DEM/topography (4) + albedo (1) + time features (2) + land-use one-hot (10)
```

PCR-Net training additionally consumes CatBoost teacher guidance maps for the guided loss. Baseline U-Net training does not use guidance maps.

## Evaluation

The shared neural evaluator reports RMSE, MAE, MBE, and R2 over valid pixels after converting normalized predictions and targets back to Celsius.
