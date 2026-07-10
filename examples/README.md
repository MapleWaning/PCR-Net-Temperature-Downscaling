# Examples

The repository provides six public demos. Run commands from the repository root.

| Demo | Purpose | Recommended command | Main outputs |
|---|---|---|---|
| [01 Data Fetch](01_data_fetch/README.md) | Minimal preprocessing/interface check. | `python examples/01_data_fetch/run_demo.py` | `outputs/demos/01_data_fetch/summary.json` |
| [02 Train and Test](02_train_and_test/README.md) | CatBoost teacher plus PCR-Net training/testing. | `python examples/02_train_and_test/run_demo.py --dataset mini_case` | `outputs/demos/02_train_and_test/<case>/<split>/summary.json` |
| [03 Visualization](03_visualization/README.md) | Pretrained inference and maps. | `python examples/03_visualization/run_demo.py --dataset mini_case --version time` | `outputs/demos/03_visualization/<case>/<split>/maps/` |
| [04 Ablation Training](04_ablation_training/README.md) | Core ablation variants. | `python examples/04_ablation_training/run_demo.py --dataset mini_case` | `outputs/demos/04_ablation_training/<case>/<split>/summary.json` |
| [05 Baselines](05_baseline_training_and_test/README.md) | RF, CatBoost without LST, and Basic U-Net baselines. | `python examples/05_baseline_training_and_test/run_demo.py --dataset mini_case` | `outputs/demos/05_baseline_training_and_test/<case>/<split>/summary.json` |
| [06 Compute Profile](06_compute_profile/README.md) | Model FLOPs, parameters, size, and peak memory. | `python examples/06_compute_profile/run_demo.py --dataset mini_case` | `outputs/demos/06_compute_profile/<case>/complexity_results.csv` |

## Recommended Order

1. Demo 1 is independent and validates the preprocessing interface.
2. Demo 2 exercises the main CatBoost plus PCR-Net workflow.
3. Demo 3 uses pretrained weights or Demo 2 checkpoints for visualization.
4. Demo 4 reuses prepared data and guidance for ablations.
5. Demo 5 compares baseline methods.
6. Demo 6 profiles model compute.

Demo 2 may use the prebuilt `mini_case` dataset even though Demo 1 writes a compatible model-ready fallback layout.

## Shared Dataset Options

Demos 2-5 use:

```text
--dataset smoke_case|mini_case
--data-root <prepared-data-root>
--split-mode auto|temporal|spatial
--version auto|time|temporal|spatial
```

Demo 6 is sample-level and does not use temporal/spatial version selection.
