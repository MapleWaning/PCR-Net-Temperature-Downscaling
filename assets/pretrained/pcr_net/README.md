# PCR-Net Pretrained Weights

Full pretrained PCR-Net checkpoints are managed by GitHub Releases.

Expected local layout after download:

```text
assets/pretrained/pcr_net/
  pcr-time.pth
  pcr-spatial.pth
  metadata.json
```

Demo 03 selects `pcr-time.pth` for the temporal version and `pcr-spatial.pth` for the spatial version before falling back to a Demo 02 checkpoint.
