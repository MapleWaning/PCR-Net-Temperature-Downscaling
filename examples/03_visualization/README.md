# Demo 03: Visualization

This demo renders one sample-level four-panel figure from a prepared demo dataset. The default dataset is `assets/demo_data/smoke_case`.

```bash
python examples/03_visualization/run_demo.py
```

Demo 03 first looks for pretrained weights under `assets/pretrained/pcr_net`:

- temporal/time: `pcr-time.pth`
- spatial: `pcr-spatial.pth`

If the selected pretrained checkpoint is missing, it falls back to the matching Demo 02 checkpoint. If neither exists, the demo exits with an error. The target sample is fixed to station `36982099999` on `2023-07-20` when that sample exists; fallback datasets use their first available sample.

If both `smoke_case` and `mini_case` are unavailable, Demo 03 falls back to Demo 01 output at `outputs/demos/01_data_fetch/model_ready/`. The visualization still needs CatBoost guidance, so run Demo 02 first if the fallback data does not already have guidance.

```bash
python examples/03_visualization/run_demo.py --dataset mini_case
```

Temporal/spatial versions can be selected explicitly:

```bash
python examples/03_visualization/run_demo.py --dataset mini_case --version time
python examples/03_visualization/run_demo.py --dataset mini_case --version spatial
```
