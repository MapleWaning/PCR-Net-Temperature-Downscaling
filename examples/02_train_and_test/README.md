# Demo 02: Model Training and Testing

This demo runs the model-side dependency chain on a prepared demo dataset:

1. Build LST CatBoost parquet features from the physical base and LST inputs.
2. Train the LST CatBoost residual model.
3. Run sample-limited CatBoost spatial inference to write `catboost_inference/catboost_lst/cb_t2m_YYYY.h5`.
4. Train PCR-Net with that CatBoost guidance.
5. Evaluate the PCR-Net checkpoint.

```bash
python examples/02_train_and_test/run_demo.py
```

The default dataset is `assets/demo_data/smoke_case`. Use the Release-managed 288-sample dataset with:

```bash
python examples/02_train_and_test/run_demo.py --dataset mini_case
```

`--split-mode auto` uses spatial split for `smoke_case` and temporal split for `mini_case`. The CatBoost guidance used by PCR-Net is produced by this demo, not pre-generated.

If both `smoke_case` and `mini_case` are unavailable, Demo 02 falls back to Demo 01 output at `outputs/demos/01_data_fetch/model_ready/`. Run Demo 01 first to create that fallback data.

Temporal/spatial versions can be selected explicitly:

```bash
python examples/02_train_and_test/run_demo.py --dataset mini_case --version time
python examples/02_train_and_test/run_demo.py --dataset mini_case --version spatial
```
