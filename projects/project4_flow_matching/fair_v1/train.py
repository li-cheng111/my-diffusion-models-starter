"""Fair Project 4 trainer: shared Project 1 U-Net for DDPM and FM."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, subprocess, sys, time
from collections import deque
from pathlib import Path
from typing import Iterator
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from projects.project1_ddpm.schedule import DDPMSchedule
from projects.project4_flow_matching.fair_v1.models import (
    ConditionalProject1UNet, build_model, predict, initial_state_sha256,
)

def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None

class CIFAR10DeterministicAugment(Dataset):
    """Per-epoch/index deterministic horizontal flip independent of worker scheduling."""
    def __init__(self, root, seed):
        self.base = datasets.CIFAR10(root, train=True, download=True, transform=transforms.ToTensor())
        self.seed = int(seed)
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
    @property
    def epoch(self):
        return int(self._epoch.item())
    @epoch.setter
    def epoch(self, value):
        self._epoch.fill_(int(value))
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        image, label = self.base[index]
        mask = (1 << 64) - 1
        value = (self.seed + self.epoch * 0x9E3779B97F4A7C15 + index * 0xD1B54A32D192ED03) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        if value >> 63:
            image = image.flip(-1)
        return image.mul(2.0).sub(1.0), int(label)

class ReflowPairDataset(Dataset):
    """Read immutable float32 teacher-pair shards listed by manifest.json."""
    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        noises, endpoints = [], []
        for shard in manifest["shards"]:
            path = directory / shard["file"]
            if sha256_file(path) != shard["sha256"]:
                raise ValueError(f"Reflow pair shard hash mismatch: {path}")
            data = torch.load(path, map_location="cpu", weights_only=True)
            noises.append(data["noise"].float().contiguous())
            endpoints.append(data["endpoint"].float().contiguous())
        self.noise = torch.cat(noises)
        self.endpoint = torch.cat(endpoints)
        if self.noise.shape != self.endpoint.shape or self.noise.ndim != 4:
            raise ValueError("Reflow noise/endpoints must be matching BCHW tensors")
        if len(self.noise) != int(manifest["num_pairs"]):
            raise ValueError("Reflow pair count differs from manifest")
        if not torch.isfinite(self.noise).all() or not torch.isfinite(self.endpoint).all():
            raise ValueError("Reflow pairs contain non-finite values")
    def __len__(self):
        return len(self.noise)
    def __getitem__(self, index):
        return self.noise[index], self.endpoint[index]

class OrderedSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices.tolist()
    def set_indices(self, indices):
        self.indices = indices.tolist()
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

def _make_generator(device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator

def _fm_loss(model, x, labels, cfg, device, time_rng, noise_rng, drop_rng, pair_noise=None):
    batch = x.shape[0]
    sampling = cfg["train"].get("t_sampling", "uniform")
    if sampling == "uniform":
        t = torch.rand(batch, device=device, generator=time_rng)
    elif sampling == "logit_normal":
        t = torch.randn(batch, device=device, generator=time_rng).sigmoid()
    else:
        raise ValueError(f"unknown t_sampling={sampling!r}")
    noise = (torch.randn(x.shape, device=device, dtype=x.dtype, generator=noise_rng)
             if pair_noise is None else pair_noise)
    t_view = t.view(batch, 1, 1, 1)
    xt, target = (1.0 - t_view) * noise + t_view * x, x - noise
    kind = cfg["model"]["type"]
    conditional = kind in {"p1_unet_conditional", "dit"}
    if conditional:
        num_classes = int(cfg["model"].get("num_classes", 10))
        drop_prob = float(cfg["train"].get("cond_drop_prob", 0.0))
        drop = torch.rand(batch, device=device, generator=drop_rng) < drop_prob
        y = torch.where(drop, torch.full_like(labels, num_classes), labels)
    else:
        y = None
    prediction = predict(model, xt, t, y)
    if prediction.shape != target.shape:
        raise ValueError(f"model output {tuple(prediction.shape)} != target {tuple(target.shape)}")
    return F.mse_loss(prediction.float(), target.float())

def _ddpm_loss(model, x, schedule, device, time_rng, noise_rng):
    batch = x.shape[0]
    t = torch.randint(schedule.T, (batch,), device=device, generator=time_rng)
    noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=noise_rng)
    alpha_bar = schedule.alphas_cumprod[t].view(batch, 1, 1, 1).to(dtype=x.dtype)
    xt = alpha_bar.sqrt() * x + (1.0 - alpha_bar).clamp_min(0).sqrt() * noise
    prediction = predict(model, xt, t.float() / float(schedule.T - 1))
    if prediction.shape != noise.shape:
        raise ValueError(f"model output {tuple(prediction.shape)} != noise {tuple(noise.shape)}")
    return F.mse_loss(prediction.float(), noise.float())

def _lr(step, train_cfg):
    base = float(train_cfg["lr"])
    warmup = int(train_cfg.get("warmup_steps", 5000))
    decay_start = int(train_cfg.get("decay_start_step", 150000))
    max_steps = int(train_cfg["max_steps"])
    floor = float(train_cfg.get("min_lr", 2e-5))
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    if step < decay_start:
        return base
    progress = min(1.0, max(0.0, (step - decay_start) / max(1, max_steps - decay_start)))
    return floor + (base - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

def _cpu_state(state):
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}

def _save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)

def _restore_optimizer_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def _ema_update(ema, model, decay):
    with torch.no_grad():
        for key, value in model.state_dict().items():
            value = value.detach()
            if value.dtype.is_floating_point:
                ema[key].mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                ema[key].copy_(value)

def _ema_from_checkpoint(checkpoint, decay):
    ema = checkpoint.get("ema")
    if isinstance(ema, dict) and "models" in ema:
        index = min(range(len(ema["decays"])), key=lambda i: abs(float(ema["decays"][i]) - decay))
        if abs(float(ema["decays"][index]) - decay) > 1e-8:
            raise ValueError(f"EMA decay {decay} not present: {ema['decays']}")
        return ema["models"][index]
    if isinstance(ema, dict):
        return ema
    return checkpoint.get("ema_state", checkpoint["model"])

def train(cfg, output_dir, seed, resume=None, init_checkpoint=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = build_model(cfg["model"]).to(device)
    parent_sha = None
    if init_checkpoint:
        parent = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(_ema_from_checkpoint(parent, float(cfg["train"].get("ema_decay", 0.9999))), strict=True)
        parent_sha = sha256_file(init_checkpoint)
    init_sha = initial_state_sha256(model)
    count = sum(p.numel() for p in model.parameters())
    print(f"model={cfg['model']['type']} params={count/1e6:.3f}M")
    print(f"device={device} precision={cfg['train'].get('precision', 'bf16')} seed={seed} init_sha256={init_sha}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]),
        betas=tuple(cfg["train"].get("betas", [0.9, 0.999])),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    decay = float(cfg["train"].get("ema_decay", 0.9999))
    ema = {key: value.detach().clone() for key, value in model.state_dict().items()}
    method = cfg["method"]
    if method not in {"fm", "ddpm"}:
        raise ValueError("method must be 'fm' or 'ddpm'")
    schedule = DDPMSchedule(**cfg.get("diffusion", {})).to(device) if method == "ddpm" else None

    data_cfg = cfg["data"]
    pair_mode = data_cfg.get("source", "cifar10") == "reflow_pairs"
    dataset = (ReflowPairDataset(data_cfg["root"]) if pair_mode
               else CIFAR10DeterministicAugment(str(data_cfg["root"]), seed))
    batch_size = int(cfg["train"]["batch_size"])
    microbatch_size = int(cfg["train"].get("microbatch_size", batch_size))
    if microbatch_size <= 0 or batch_size % microbatch_size:
        raise ValueError("microbatch_size must divide the effective batch_size")
    accumulation_steps = batch_size // microbatch_size
    num_workers = int(data_cfg.get("num_workers", 4))
    max_steps = int(cfg["train"]["max_steps"])
    time_rng = _make_generator(device, seed + 100_003)
    noise_rng = _make_generator(device, seed + 200_003)
    drop_rng = _make_generator(device, seed + 300_007)
    step, epoch, batch_offset = 0, 0, 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != method:
            raise ValueError(f"cannot resume {checkpoint.get('method')} checkpoint as {method}")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema = {k: v.to(device).clone() for k, v in checkpoint["ema_state"].items()}
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_optimizer_device(optimizer, device)
        step, epoch, batch_offset = int(checkpoint["step"]), int(checkpoint["epoch"]), int(checkpoint["batch_offset"])
        time_rng.set_state(checkpoint["rng"]["time"].cpu())
        noise_rng.set_state(checkpoint["rng"]["noise"].cpu())
        drop_rng.set_state(checkpoint["rng"]["drop"].cpu())
        torch.set_rng_state(checkpoint["rng"]["torch"].cpu())
        if device.type == "cuda" and checkpoint["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng"]["cuda"]])
        init_sha = checkpoint.get("initial_state_sha256", init_sha)
        print(f"resumed step={step} epoch={epoch} batch_offset={batch_offset}", flush=True)

    metadata = {
        "experiment": cfg.get("experiment"), "method": method, "seed": seed,
        "config": cfg, "initial_state_sha256": init_sha,
        "parent_checkpoint_sha256": parent_sha, "git_revision": git_revision(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "effective_batch_size": batch_size, "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "augmentation": ("fixed teacher pairs; no augmentation" if pair_mode else
                         "CIFAR-10 train; deterministic per-epoch/index horizontal flip p=0.5"),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_path = output / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(["step", "loss_100", "lr", "step_per_sec", "elapsed_seconds"])
    recent = deque(maxlen=100)
    start_time = time.perf_counter()
    starting_step = step
    elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0)) if resume else 0.0
    save_every = int(cfg["train"].get("save_every", 10000))
    snapshot_every = int(cfg["train"].get("snapshot_every", 50000))
    precision = str(cfg["train"].get("precision", "bf16"))
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    amp_enabled = precision == "bf16" and device.type == "cuda"

    sampler = OrderedSampler(torch.empty(0, dtype=torch.long))
    loader_generator = torch.Generator().manual_seed(seed + 77)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0), generator=loader_generator,
    )
    while step < max_steps:
        if hasattr(dataset, "epoch"):
            dataset.epoch = epoch
        order_generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch * 97_409)
        order = torch.randperm(len(dataset), generator=order_generator)
        if batch_offset > len(dataset):
            raise ValueError("saved data cursor exceeds dataset length")
        sampler.set_indices(order[batch_offset:])
        for batch in loader:
            if pair_mode:
                pair_noise, images = batch
                labels = torch.zeros(len(images), dtype=torch.long)
            else:
                images, labels = batch
                pair_noise = None
            if len(images) != batch_size:
                raise RuntimeError(f"incomplete effective batch: {len(images)} != {batch_size}")
            lr = _lr(step, cfg["train"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro_start in range(0, batch_size, microbatch_size):
                micro_end = micro_start + microbatch_size
                x_micro = images[micro_start:micro_end].to(device, non_blocking=True)
                y_micro = labels[micro_start:micro_end].to(device, non_blocking=True)
                noise_micro = (pair_noise[micro_start:micro_end].to(device, non_blocking=True)
                               if pair_noise is not None else None)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    micro_loss = (_fm_loss(model, x_micro, y_micro, cfg, device, time_rng, noise_rng, drop_rng, noise_micro)
                                  if method == "fm" else _ddpm_loss(model, x_micro, schedule, device, time_rng, noise_rng))
                if not torch.isfinite(micro_loss):
                    raise FloatingPointError(f"non-finite loss at step {step + 1}: {micro_loss.item()}")
                (micro_loss / accumulation_steps).backward()
                loss_value += float(micro_loss.detach().float().item()) / accumulation_steps
                del x_micro, y_micro, noise_micro, micro_loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            optimizer.step()
            _ema_update(ema, model, decay)
            recent.append(loss_value)
            step += 1
            batch_offset += batch_size

            if step % int(cfg["train"].get("log_every", 100)) == 0:
                elapsed = time.perf_counter() - start_time
                speed = (step - starting_step) / max(elapsed, 1e-9)
                mean_loss = sum(recent) / len(recent)
                total_elapsed = elapsed_before + elapsed
                with open(log_path, "a", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerow([step, f"{mean_loss:.7f}", f"{lr:.9g}", f"{speed:.5f}", f"{total_elapsed:.1f}"])
                print(f"step {step}/{max_steps} loss100={mean_loss:.5f} lr={lr:.3e} step/s={speed:.3f} elapsed_h={total_elapsed/3600:.2f}", flush=True)

            if step % save_every == 0 or step >= max_steps:
                total_elapsed = elapsed_before + time.perf_counter() - start_time
                checkpoint = {
                    "format": "project4-fair-v1", "experiment": cfg.get("experiment"), "method": method,
                    "step": step, "epoch": epoch, "batch_offset": batch_offset,
                    "model": _cpu_state(model.state_dict()), "ema_state": _cpu_state(ema),
                    "ema_decay": decay, "optimizer": optimizer.state_dict(), "cfg": cfg,
                    "seed": seed, "initial_state_sha256": init_sha,
                    "parent_checkpoint_sha256": parent_sha, "elapsed_seconds": total_elapsed,
                    "rng": {"time": time_rng.get_state(), "noise": noise_rng.get_state(),
                            "drop": drop_rng.get_state(), "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                }
                _save_checkpoint(output / "latest.pt", checkpoint)
                if step % snapshot_every == 0 or step >= max_steps:
                    _save_checkpoint(output / f"step_{step}.pt", checkpoint)
                print(f"saved checkpoint at step {step}", flush=True)
            if step >= max_steps:
                break
        epoch += 1
        batch_offset = 0
    print(f"training_complete step={step} output={output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="temporary step override for smoke runs")
    parser.add_argument("--batch-size", type=int, default=None, help="temporary batch override for smoke runs")
    parser.add_argument("--num-workers", type=int, default=None, help="temporary data-loader override for smoke runs")
    parser.add_argument("--init", dest="init_checkpoint", default=None,
                        help="initialize from a parent EMA checkpoint with fresh optimizer state")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 44))
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
        cfg["train"]["microbatch_size"] = min(int(cfg["train"].get("microbatch_size", args.batch_size)), args.batch_size)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    train(cfg, args.output, seed, args.resume, args.init_checkpoint)

if __name__ == "__main__":
    main()
