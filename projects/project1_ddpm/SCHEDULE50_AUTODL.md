# CIFAR-10 50-epoch linear/cosine comparison

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

Use the same GPU image and Python environment on all instances. Before starting,
verify on each instance:

```bash
nvidia-smi
/root/miniconda3/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")'
df -h /root/autodl-tmp
```

The launcher stops before training if PyTorch cannot see CUDA or `nvidia-smi`
is unavailable. Install a CUDA-enabled PyTorch environment in an AutoDL image
with an attached GPU; installing CUDA wheels alone cannot expose a missing GPU
device to the container.

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
