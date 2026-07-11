# PCR-Net Pretrained Weights

PCR-Net checkpoints are managed by GitHub Releases instead of Git.

## Download

```bash
python scripts/download_release_artifact.py pcr_time
python scripts/download_release_artifact.py pcr_spatial
```

The active checkpoint release is the stable submission snapshot
[`v0.9.0`](https://github.com/MapleWaning/PCR-Net-Temperature-Downscaling/releases/tag/v0.9.0).
The downloader verifies each checkpoint with the size and SHA256 values in
`configs/artifacts/artifact_manifest.json`.

## Expected Layout

```text
assets/pretrained/pcr_net/
  pcr-time-v0.1.0.pth
  pcr-time.pth
  pcr-spatial-v0.1.0.pth
  pcr-spatial.pth
```

`pcr-time.pth` is the runtime alias for the temporal-generalization checkpoint.
`pcr-spatial.pth` is the runtime alias for the spatial leave-out checkpoint.

Demo 3 selects the matching alias for `--version time`, `--version temporal`,
or `--version spatial`. If a checkpoint is missing and no `--model-path` is
provided, the demo attempts to download it automatically.
