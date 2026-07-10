# Open-Source Model Modules

This directory contains the migrated model-side code used by the public demos. Most users should start with the wrappers under `examples/`, which configure paths, reduced schedules, artifact downloads, and fallback behavior.

## Contents

```text
data_prepare/        Tabular feature building for teacher and baseline models.
CatBoost/            LST-guided CatBoost teacher training and inference.
base_line/CB/        No-LST CatBoost baseline.
base_line/RF/        No-LST Random Forest baseline.
base_line/U-Net/     Basic U-Net baseline.
PCR-Net/             PCR-Net training entrypoint.
ablation/            Core ablation entrypoints.
test/                Shared neural evaluation.
mapping/             Sample-level visualization.
unet_datasets.py     Neural dataset adapters.
unet_models.py       `BasicUNet` and `ResAttUNet`.
unet_losses.py       Baseline and guided losses.
unet_training.py     Shared training loops.
```

## Notes

- Hyphenated directories such as `PCR-Net` and `U-Net` are script directories, not Python dotted-import package names.
- The public PCR-Net checkpoints are stored outside Git and installed as `assets/pretrained/pcr_net/pcr-time.pth` and `assets/pretrained/pcr_net/pcr-spatial.pth`.
- Use `PYTHONPATH=src` when running these scripts directly from a source checkout.
- For public workflow validation, prefer the demo commands in `examples/README.md`.
