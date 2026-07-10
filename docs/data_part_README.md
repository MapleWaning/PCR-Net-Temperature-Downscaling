# Data Pipeline and Physical Baseline Notes

This legacy page is retained for users who followed earlier repository notes. The current user-facing data documentation is:

- `docs/data_and_artifacts.md`
- `docs/manual_downloads.md`
- `docs/data_contract.md`
- `docs/configuration.md`

## Current Package Layout

```text
src/data_pipeline/
  cli.py
  config.py
  pipeline.py
  registry.py
  common/
  modules/

src/physical_base/
  cli.py
  config.py
  pipeline.py
  factors.py
  tbase.py
  validation.py
  physics/
```

`data_pipeline` prepares standard data products from GSOD, ERA5, SRTM, GLC FCS30D, Landsat-derived albedo, and MODIS LST inputs. `physical_base` consumes the standard metadata, DEM, and ERA5 temperature products plus ERA5 auxiliary physical inputs to produce physical factors and `t_base_advanced_YYYY.h5`.

## Entrypoints

```bash
python scripts/run_pipeline.py plan --config configs/data/default.yaml
python scripts/run_pipeline.py run --config configs/data/default.yaml

python scripts/run_physical.py plan --config configs/physical/physical.yaml
python scripts/run_physical.py run-factors --config configs/physical/physical.yaml
python scripts/run_physical.py run-tbase --config configs/physical/physical.yaml
python scripts/run_physical.py run --config configs/physical/physical.yaml
```

The equivalent module entrypoints are:

```bash
python -m data_pipeline.cli plan --config configs/data/default.yaml
python -m physical_base.cli run --config configs/physical/physical.yaml
```

Use `PYTHONPATH=src` if running module entrypoints from an uninstalled checkout.

## Contract Summary

Station order is defined by `standard/metadata/high-quality-meta.csv`. Arrays with a station dimension must use that row order. Standard products and physical products are documented in `docs/data_contract.md`.

Manual third-party data requirements, provider access notes, MODIS LST task logic, ERA5 pressure-level inputs, and GLC FCS30D tile names are documented in `docs/manual_downloads.md`.
