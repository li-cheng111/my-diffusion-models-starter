# Seed 43 schedule continuation: epoch 50 → 200

## Status and conclusion

Both schedule-specific seed 43 runs reached epoch 200 and all six fixed-protocol FID evaluations completed. The best result in this continuation is **Cosine, EMA 0.9995: FID 15.5792**. It improves on the corresponding linear continuation (18.3699) but does **not** meet the target FID ≤ 15. The project-wide best remains R5 seed 46 / EMA 0.9999 at 15.2086.

## Experimental design

- Dataset: CIFAR-10; training seed 43.
- Two independent continuation branches start from their matching 50-epoch checkpoint: the Linear branch resumes Linear, and the Cosine branch resumes Cosine. Exact-epoch resume carries training to epoch 200 (78,000 total optimizer steps; approximately 58,500 additional updates after epoch 50).
- Apart from the beta schedule, the two 200-epoch configurations match: full U-Net with base width 128, channel multipliers `[1, 2, 2, 2]`, two residual blocks per level, attention at 16×16, dropout 0.1; CIFAR-10 32×32; batch 128; epsilon prediction with uniform MSE; AdamW at `2e-4`, 5,000-step warmup; fp16, gradient clip 1.0, one gradient-accumulation step; `T=1000`; EMA decays 0.999, 0.9995, and 0.9999.
- Linear betas span 0.0001–0.02. Cosine uses offset `s=0.008`.
- Continuation time reported by the training logs: Linear 376.1 minutes; Cosine 390.7 minutes. These are the epoch-50-to-200 continuation durations, not fresh-from-scratch wall times.
- Backend: Moore Threads MTT S4000 / MUSA. Recorded environment: Linux x86_64, Python 3.10.8, `torch 2.2.0a0+git8ac9b20`, `torch_musa 1.3.0`, `torchvision 0.17.2+c1d70fe`.

## Evaluation protocol

All six evaluations use the CIFAR-10 train split's first 5,000 real images and 5,000 generated images, generation seed 44, batch size 64, 1,000 reverse DDPM steps, torch-fidelity Inception features, and per-step clipping of predicted `x0` to `[-1, 1]`. The evaluated weight is the named EMA state in the epoch-200 checkpoint. No real-image subset or generation seed was changed between configurations.

## Metrics

The first three columns below are the seed 43 50-epoch baseline values; the next three are the same schedule-specific branch after continuation to epoch 200. Lower FID is better.

| Schedule | EMA | 50 epoch | 200 epoch | Change |
|---|---:|---:|---:|---:|
| Linear | 0.999 | 28.8330 | 18.9129 | −9.9201 |
| Linear | 0.9995 | 29.2898 | 18.3699 | −10.9199 |
| Linear | 0.9999 | 160.4971 | 18.9244 | −141.5727 |
| Cosine | 0.999 | 24.0592 | 15.9393 | −8.1199 |
| Cosine | 0.9995 | 24.4483 | **15.5792** | −8.8691 |
| Cosine | 0.9999 | 142.6996 | 16.3553 | −126.3443 |

The 50-epoch three-training-seed experiment (seeds 42, 43, 44) provides useful context: at EMA 0.999 and 0.9995, cosine means were 23.9276 and 24.3862, versus 28.9522 and 29.3787 for Linear. EMA 0.9999 was anomalously poor at 50 epochs for both schedules (cosine 152.3518 ± 8.7018; Linear 189.3650 ± 72.2358). At 200 epochs, the paired seed 43 continuation again favors cosine at every EMA, but this 200-epoch result is only one training seed and must not be described as a three-seed estimate.

The large recovery of EMA 0.9999 after continuation is consistent with a long EMA time horizon lagging behind a short training trajectory. It is evidence from these runs, not proof that longer training alone explains all schedule or seed differences. The early base-U-Net challenge matrix used a different model/sampling setup and had a severe unclipped cosine endpoint failure; it is not a directly comparable estimate of the full-U-Net continuation effect.

## Files

- `summary.csv`: machine-readable 50- and 200-epoch FID results.
- `linear/config.yaml`, `cosine/config.yaml`: exact 200-epoch configurations.
- `linear/train.log`, `cosine/train.log`; `linear/loss_history.csv`, `cosine/loss_history.csv`: training logs and per-update loss histories.
- `linear/evaluations/`, `cosine/evaluations/`: FID and sample-generation logs; each `ckpt/fid_*.txt` is the raw metric record.
- `linear/samples_ema_*/grid.png`, `cosine/samples_ema_*/grid.png`: 8×8 grids of 64 samples for each EMA decay, generated with seed 44 and clipped-x0 sampling.
- `linear/environment.txt`, `cosine/environment.txt`: runtime versions.
- `linear/CHECKPOINT_SHA256.txt`, `cosine/CHECKPOINT_SHA256.txt`: hashes for the epoch 100/150/200 snapshots and the exact `epoch_latest.pt` used for evaluation.

Large epoch-200 checkpoints are distributed separately through the [schedule200-seed43-v1 GitHub Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule200-seed43-v1); checkpoints and datasets are intentionally excluded from this Git directory. The seed 42/43/44 50-epoch checkpoint and result bundles are in their respective `schedule50-seed*-v1` Releases.
