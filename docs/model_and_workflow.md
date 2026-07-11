# Model and Workflow

The PCR-Net workflow is:

```text
Physical baseline
-> LST-guided CatBoost teacher manifold
-> Terrain-aware, knowledge-distilled PCR-Net
-> 100 m temperature output
```

## Physical Baseline

The physical-base stage uses standard data products and physical factors to generate `t_base_advanced_YYYY.h5`. These tensors preserve a daily, station, channel, and patch layout compatible with the model datasets.

## CatBoost Teacher

The CatBoost teacher is trained with features derived from the physical base, static covariates, time information, and LST inputs. It writes teacher maps such as:

```text
catboost_inference/catboost_lst/cb_t2m_YYYY.h5
```

LST is used in teacher training and teacher-map generation. PCR-Net inference does not require high-resolution LST.

## PCR-Net Student

PCR-Net uses `ResAttUNet` with:

- residual prediction over the normalized physical temperature base;
- terrain-aware attention gates;
- spatial feature transform (SFT);
- a hybrid refinement loss with gradient-guidance terms during training.

The hybrid refinement loss exposes two tunable parameters in the public training
entrypoint:

- `--lambda-grad` controls the weight of the CatBoost teacher-guidance gradient
  loss term.
- `--alpha-terrain` controls the terrain-weighted station supervision term based
  on the DEM slope channel.

The current repository defaults are `LAMBDA_GRAD = 0.1` and
`ALPHA_TERRAIN = 0.1` in `config.py`. These values are included to demonstrate
that both parameters are part of the training interface; they are not claimed to
be optimal settings.

The neural input has 20 channels:

```text
physical base (3) + DEM/topography (4) + albedo (1) + time (2) + land-use one-hot (10)
```

The output has 3 channels:

```text
mean temperature, maximum temperature, minimum temperature
```

## Temporal and Spatial Versions

Temporal-generalization runs use year-based train/validation/test splits and the temporal checkpoint `pcr-time.pth`.

Spatial leave-out runs use station-based splits and the spatial checkpoint `pcr-spatial.pth`.

Demo 6 is a sample-level compute profile and does not use temporal/spatial model selection.

## Relationship to Demos

- Demo 1 validates the data pipeline interface and writes a compatible model-ready fixture.
- Demo 2 runs the CatBoost teacher and PCR-Net training/testing workflow.
- Demo 3 runs pretrained inference and visualization.
- Demo 4 runs model ablations.
- Demo 5 runs baseline experiments.
- Demo 6 profiles model compute with THOP and CUDA memory counters when available.
