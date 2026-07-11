# Reproducibility

This repository distinguishes software execution, public demo reproduction, and manuscript numerical reproduction.

## Level A: Software Smoke Test

Purpose:

- Verify imports.
- Verify data interfaces.
- Verify a minimal training or forward path.
- Verify output creation.

Typical data:

- Tracked `assets/demo_data/smoke_case/`.
- Demo 1 generated fixture under `outputs/demos/01_data_fetch/model_ready/`.

Example:

```bash
python examples/06_compute_profile/run_demo.py --dataset smoke_case --num-samples 1 --diffusion-steps 4
```

## Level B: Public Demo Reproduction

Purpose:

- Run public workflows for training, testing, visualization, ablations, baselines, and computational profiling.
- Exercise Release artifact download and pretrained-checkpoint lookup.

Typical data:

- Stable submission snapshot [`v0.9.0`](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0).
- `assets/demo_data/mini_case/`.
- `assets/pretrained/pcr_net/pcr-time.pth`.
- `assets/pretrained/pcr_net/pcr-spatial.pth`.

Expected outcome:

- Valid summaries, metrics, figures, checkpoints, and benchmark tables.
- Values are workflow-validation results and are not manuscript-identical metrics.

## Level C: Manuscript Experiment Reproduction

Purpose:

- Reproduce manuscript numerical results.

Requires:

- Complete 118-station, 2008-2024 research dataset.
- Third-party source products acquired from official providers.
- Full preprocessing.
- Full training schedules.
- Final paper-associated configurations and compute environment.

## Manuscript-Result Mapping

| Manuscript result | Public entry point | Required data | Expected reproduction level |
|---|---|---|---|
| Temporal accuracy table | Demo 2 with `--version time`; full training entry points | Full research dataset for manuscript values | Level B for workflow; Level C for manuscript numbers |
| Spatial leave-out accuracy table | Demo 2 with `--version spatial`; full training entry points | Full research dataset for manuscript values | Level B for workflow; Level C for manuscript numbers |
| Representative maps | Demo 3 | Release checkpoint and mini-case for demo figures; full dataset for manuscript figures | Level B or C depending on data |
| Ablation table | Demo 4 | Full research dataset for manuscript trends | Level B for workflow; Level C for manuscript values |
| Baseline comparison | Demo 5 | Full research dataset for manuscript values | Level B for workflow; Level C for manuscript values |
| Computational-cost table | Demo 6 | Real or demo tensors; hardware affects peak-memory values | Level B for table generation; hardware-dependent comparisons require matching environment |

The public mini-case is intentionally small. It is useful for auditing commands, paths, and outputs, but it is not intended to reproduce final manuscript metrics.
