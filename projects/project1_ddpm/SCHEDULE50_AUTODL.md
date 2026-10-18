# CIFAR-10 50-epoch linear/cosine comparison

**Status: complete.** All six jobs (linear and cosine at training seeds 42, 43,
and 44) completed 50 epochs and their three EMA evaluations under the fixed
protocol below. Final metrics are recorded near the end of this document.

This run compares linear and cosine diffusion schedules with the same full
U-Net, optimizer settings, 50-epoch budget, EMA bank, and FID protocol. The six
jobs are paired by training seed:

| AutoDL instance | Training order | Seed |
|---|---|---:|
| A | linear, then cosine | 42 |
| B | cosine, then linear | 43 |
| C | linear, then cosine | 44 |

## Fixed setup

- CIFAR-10 train set, 32x32, random horizontal flip during training.
- Batch size 128; full U-Net with base channels 128, channel multipliers
  `[1, 2, 2, 2]`, two ResBlocks per level, and attention at 16x16.
- Epsilon prediction with uniform MSE; `T=1000`.
- Linear betas run from `0.0001` to `0.02`; cosine uses `s=0.008`.
- AdamW, learning rate `2e-4`, 5,000 successful-update warmup steps, then
  constant learning rate; gradient clipping 1.0; fp16.
- EMA decays 0.999, 0.9995, and 0.9999 are kept in the same checkpoint.
- Exactly 50 data epochs per job. With `drop_last=True`, this is 19,500
  attempted batches; AMP-skipped updates are logged and not replaced with
  extra epochs.
- Formal FID uses CIFAR-10 train's first 5,000 real images without augmentation,
  5,000 generated images, evaluation seed 44, batch size 64, 1,000 DDPM steps,
  torch-fidelity Inception features, and predicted-x0 clipping at every reverse
  step. EMA 0.9999 is the predeclared primary metric; the other EMA values are
  supporting comparisons.

## GPU and environment preflight

These three instances use Moore Threads MUSA GPUs (MTT S4000), not NVIDIA
CUDA GPUs. Their installed environment uses `torch_musa`; import that package
before querying `torch.musa` or selecting the device. The project detects CUDA
or MUSA automatically, uses MUSA AMP on these instances, and records the MUSA
device details through `musaInfo` when NVIDIA tooling is absent. Before
starting, verify on each instance:

```bash
/usr/local/musa/bin/musaInfo
/root/miniconda3/bin/python -c 'import torch; import torch_musa; print(torch.__version__, torch.musa.is_available(), torch.musa.device_count())'
df -h /root/autodl-tmp
```

The launcher stops before training if PyTorch can see neither CUDA nor MUSA.
`nvidia-smi` is not expected on a Moore Threads instance and is not required.
The monitor reports MUSA memory from `musaInfo`; utilization may be unavailable
if the image does not provide a MUSA equivalent of `nvidia-smi`.

## Start one assigned instance

The launcher clones the public repository into the data disk, records the Git
commit and environment, starts a read-only dashboard on port 8765, downloads
CIFAR-10 if absent, and runs the assigned pair sequentially. It uses epoch
boundary checkpoints so an interrupted job resumes at the next complete epoch.

First fetch the project scripts to the AutoDL data disk:

```bash
mkdir -p /root/autodl-tmp/schedule50
source /etc/network_turbo 2>/dev/null || true
git clone --filter=blob:none --sparse https://github.com/li-cheng111/my-diffusion-models-starter.git /root/autodl-tmp/schedule50/repo
git -C /root/autodl-tmp/schedule50/repo sparse-checkout set projects/project1_ddpm
```

The runner can also use a pre-provisioned source snapshot under
`/root/autodl-tmp/schedule50/repo/projects/project1_ddpm` if GitHub cloning is
unavailable; in that case set `SOURCE_COMMIT` to the snapshot's exact commit
hash when launching it.

Then launch the assigned pair inside `screen`:

```bash
# Instance A
screen -dmS schedule50_seed42 bash -lc 'bash /root/autodl-tmp/schedule50/repo/projects/project1_ddpm/run_schedule50_instance.sh /root/autodl-tmp/schedule50 42 linear-first'

# Instance B
screen -dmS schedule50_seed43 bash -lc 'bash /root/autodl-tmp/schedule50/repo/projects/project1_ddpm/run_schedule50_instance.sh /root/autodl-tmp/schedule50 43 cosine-first'

# Instance C
screen -dmS schedule50_seed44 bash -lc 'bash /root/autodl-tmp/schedule50/repo/projects/project1_ddpm/run_schedule50_instance.sh /root/autodl-tmp/schedule50 44 linear-first'
```

For unattended runs, start each command inside a separate `screen` session.
The launcher writes logs under `logs/`, training outputs under `runs/seedNN/`,
and environment metadata under `manifest/`. It stores each final checkpoint,
epoch recovery checkpoint, loss curve, per-EMA FID text, and final sample grid.

## Live monitor

Each assigned instance serves its own dashboard on port 8765 and watches both
schedule runs for that seed. Forward each remote port to a different local port
from three PowerShell windows:

```powershell
ssh -N -T -L 8765:127.0.0.1:8765 -p 28657 root@connect.sha1.seetacloud.com
ssh -N -T -L 8766:127.0.0.1:8765 -p 17355 root@connect.sha1.seetacloud.com
ssh -N -T -L 8767:127.0.0.1:8765 -p 22042 root@connect.sha1.seetacloud.com
```

Open `http://127.0.0.1:8765`, `http://127.0.0.1:8766`, and
`http://127.0.0.1:8767` to see the three seed pairs. An SSH tunnel stays open
without printing output after authentication; closing its terminal closes that
forward only, not the `screen` training process.

## Result interpretation

Report all three seed pairs and all EMA values. For each EMA, show linear and
cosine mean and sample standard deviation across training seeds, plus the
per-seed paired difference `cosine - linear`. Treat EMA 0.9999 as the primary
comparison. A 50-epoch result describes this short-training regime; it does not
replace the existing 200-epoch comparison or establish which schedule is best
after full convergence.

## Completed results

Each entry lists FID for EMA 0.999 / 0.9995 / 0.9999. The reported summary is
the mean and sample standard deviation across the three *training* seeds.

| Training seed | Linear | Cosine |
|---:|---:|---:|
| 42 | 29.2144 / 29.6127 / 136.0274 | 23.4786 / 24.0694 / 159.5961 |
| 43 | 28.8330 / 29.2898 / 160.4971 | 24.0592 / 24.4483 / 142.6996 |
| 44 | 28.8092 / 29.2335 / 271.5706 | 24.2449 / 24.6410 / 154.7598 |
| Mean ± sample SD | 28.9522 ± 0.2274 / 29.3787 ± 0.2046 / 189.3650 ± 72.2358 | 23.9276 ± 0.3997 / 24.3862 ± 0.2908 / 152.3518 ± 8.7018 |

Cosine beat linear on every seed for EMA 0.999 and 0.9995, by 4.56–5.74 FID
and 4.59–5.54 FID respectively. EMA 0.9999 was severely degraded for both
schedules. Its approximately 10,000-update averaging time scale is long relative
to this 19,500-batch budget, making excessive lag a plausible explanation, but
the experiment does not isolate or prove that mechanism. These 50-epoch results
are a short-budget comparison and must not be conflated with the separate
200-epoch experiments.

## Release artifacts

- Seed 42: [schedule50-seed42-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed42-v1)
- Seed 44: [schedule50-seed44-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed44-v1)
- Seed 43: [schedule50-seed43-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed43-v1). The Release now contains both final checkpoints, the completed logs/configs/FID/loss/sample archive, and a completed SHA256 manifest. The legacy `final.pt` asset is the cosine checkpoint; the linear checkpoint has an explicit filename. The earlier resume bundle remains for provenance.

Each completed result archive retains the per-run configuration, environment
metadata, training/evaluation logs, loss history/curve, EMA FID text files, and
sample grids. Final checkpoints remain Release attachments, not ordinary Git
objects. CIFAR-10 data is not included.
