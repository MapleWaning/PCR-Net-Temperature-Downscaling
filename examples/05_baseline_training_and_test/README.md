# Demo 05: Baseline Training and Testing

This demo runs the baseline model workflow on a prepared demo dataset. The default dataset is `assets/demo_data/smoke_case`.

It trains and tests:

- Random Forest baseline.
- CatBoost baseline without LST.
- Basic U-Net baseline with MSE loss.

All baseline inputs use the raw ERA5-style HDF5 input:

```text
assets/demo_data/<case>/standard/era5/era5_t2m_YYYY.h5
```

This demo does not use LST and does not use `cb_t2m_YYYY.h5`.

```bash
python examples/05_baseline_training_and_test/run_demo.py
```

Use the Release-managed 288-sample dataset with:

```bash
python examples/05_baseline_training_and_test/run_demo.py --dataset mini_case
```

Temporal/spatial versions can be selected explicitly:

```bash
python examples/05_baseline_training_and_test/run_demo.py --dataset mini_case --version time
python examples/05_baseline_training_and_test/run_demo.py --dataset mini_case --version spatial
```

If both `smoke_case` and `mini_case` are unavailable, Demo 05 falls back to Demo 01 output at `outputs/demos/01_data_fetch/model_ready/`.
