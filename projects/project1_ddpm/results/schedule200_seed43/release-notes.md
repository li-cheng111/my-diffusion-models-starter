# Seed 43 CIFAR-10 schedule continuation to 200 epochs

This release contains the exact epoch-200 checkpoints evaluated in the completed Linear-vs-Cosine continuation, plus compact reproducibility bundles with configs, full training/loss logs, evaluation logs, raw metric files, sample grids, environment versions, and intermediate/final checkpoint hashes. No dataset is included.

## Fixed evaluation protocol

- CIFAR-10 train split: first 5,000 real images and 5,000 generated images
- Generation seed 44; batch size 64; 1,000-step DDPM
- torch-fidelity Inception features; predicted `x0` clipped at every reverse step
- Training seed 43; full U-Net; epoch-50 schedule-specific checkpoints resumed to epoch 200
- Moore Threads MTT S4000 (MUSA), batch size 128, epsilon prediction, `T=1000`, AdamW `2e-4`, 5,000-step warmup

## FID (EMA 0.999 / 0.9995 / 0.9999)

| Schedule | EMA 0.999 | EMA 0.9995 | EMA 0.9999 | Best |
|---|---:|---:|---:|---:|
| Linear | 18.9129 | 18.3699 | 18.9244 | 18.3699 |
| Cosine | 15.9393 | **15.5792** | 16.3553 | **15.5792** |

Cosine is lower at all three EMA decays in this paired seed-43 continuation. The best result is still above the target FID ≤ 15. This is one training seed, not a multi-seed estimate. The exact FID results and comparison with the three-seed 50-epoch baseline are documented in the project README and `results/schedule200_seed43/README.md`.

## Assets

- `cifar10_schedule200_seed43_linear_epoch200.pt`: exact Linear `epoch_latest.pt` used for evaluation
- `cifar10_schedule200_seed43_cosine_epoch200.pt`: exact Cosine `epoch_latest.pt` used for evaluation
- `linear_repro_bundle.tar.gz`, `cosine_repro_bundle.tar.gz`: logs, configs, metrics, six 64-sample grids, environment info, loss histories, and hashes; exclude weights and datasets
- `RELEASE_SHA256SUMS.txt`: SHA-256 for all four downloadable assets
- Intermediate epoch 100/150 checkpoints remain on the AutoDL instances; their SHA-256 values are recorded in the repro bundles and Git-tracked per-run checksum files.
