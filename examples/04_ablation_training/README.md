# Demo 04: Ablation Training

This demo trains the core PCR-Net ablation matrix on a prepared demo dataset. The default dataset is `assets/demo_data/smoke_case`.

The matrix contains:

- `no_attention`: PCR-Net without attention and SFT, using CatBoost guidance.
- `no_gradient_loss`: PCR-Net trained with pure MSE, without the gradient guidance loss.

```bash
python examples/04_ablation_training/run_demo.py
```

Use the Release-managed 288-sample dataset with:

```bash
python examples/04_ablation_training/run_demo.py --dataset mini_case
```

Temporal/spatial versions can be selected explicitly:

```bash
python examples/04_ablation_training/run_demo.py --dataset mini_case --version time
python examples/04_ablation_training/run_demo.py --dataset mini_case --version spatial
```

If both `smoke_case` and `mini_case` are unavailable, Demo 04 falls back to Demo 01 output at `outputs/demos/01_data_fetch/model_ready/`. The `no_attention` ablation needs CatBoost guidance, so run Demo 02 first when using that fallback.
