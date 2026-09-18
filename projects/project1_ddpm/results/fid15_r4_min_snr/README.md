# R4: Cosine + Min-SNR DDPM

This directory records the completed R4 CIFAR-10 experiment. The run finished
the 200-epoch budget on AutoDL and used the full U-Net configuration in the
adjacent `config.yaml` file.

## Fixed evaluation protocol

- Dataset: CIFAR-10 `train` split, first 5,000 real images
- Generated images: 5,000 per evaluation
- Seed: 44
- Reverse process: 1,000-step DDPM sampling
- Predicted `x0` clipping: enabled
- Checkpoint: `final.pt` from the corresponding AutoDL run

## FID results

| Weights | FID |
|---|---:|
| EMA 0.9990 | 16.3013 |
| EMA 0.9995 | **15.7954** |
| EMA 0.9999 | 16.5633 |
| Raw | 29.1437 |

The best R4 result is EMA 0.9995 at FID 15.7954. It does not meet the target
`FID <= 15` under this fixed protocol.

## Training configuration

- Beta schedule: cosine, `T=1000`, `s=0.008`
- U-Net: base channels 128, channel multipliers `[1, 2, 2, 2]`
- Two residual blocks per level, attention at 16x16, full skip layout
- Batch size 128, mixed precision FP16
- Learning rate `2e-4`, 5,000-step warmup, no late decay in this run
- Loss: epsilon prediction with Min-SNR weighting, `gamma=5`
- EMA decays: 0.999, 0.9995, 0.9999
- Target budget: 200 epochs / 78,000 attempted steps
- Wall-clock training time: 154.5 minutes

The loss history contains 28 AMP-skipped optimizer attempts. The final
successful optimizer step is therefore slightly below the attempted-step
target; the skip rate is approximately 0.036%.

## Included artifacts

- `config.yaml`: exact run configuration
- `loss_history.csv` and `loss_curve.png`: training history
- `ckpt/fid_*.txt` and `ckpt/fid_comparison_clipx0.md`: FID results
- `samples/`: periodic sample grids
- `samples_final_ema9995_clipx0/grid.png`: final best-EMA sample grid

The large model checkpoint is distributed through the GitHub Release rather
than regular Git history.
