# Prepared Demo Data

This directory stores lightweight prepared demo datasets.

- `smoke_case/`: a four-sample dataset committed with the repository for fast smoke tests.
- `mini_case/`: a 288-sample dataset with temporal and spatial split views. This dataset is managed through GitHub Releases rather than Git.

Both datasets keep the same inner layout as `data/processed` so demos can switch between them with `--dataset smoke_case` or `--dataset mini_case`.
