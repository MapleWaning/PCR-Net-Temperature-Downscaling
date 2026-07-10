# Outputs

This directory is reserved for generated results. Git tracks this README and
the directory marker only; generated files are ignored.

## Demo Output Layout

Public demos write under:

```text
outputs/demos/
  01_data_fetch/
  02_train_and_test/
  03_visualization/
  04_ablation_training/
  05_baseline_training_and_test/
  06_compute_profile/
```

Typical summary files are named `summary.json`. Demo 6 also writes:

```text
outputs/demos/06_compute_profile/<case>/complexity_results.csv
```

## Reproducibility Notes

Generated outputs depend on the selected dataset, split mode, random seed,
hardware, PyTorch/CUDA stack, and locally available artifacts. Demo outputs are
workflow-validation products unless explicitly produced from the complete
research dataset and finalized paper-associated settings.

Do not commit generated outputs unless a future release process explicitly asks
for a small, documented artifact.
