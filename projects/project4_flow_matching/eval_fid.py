"""Reproducible CIFAR-10 FID evaluation for Project 4 FM checkpoints."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torchvision

from train import build_model
from sample import sample_ode, select_ema_state


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


@torch.no_grad()
def generate(model, num_samples, batch_size, n_steps, num_classes, solver,
             condition_mode, cfg_scale, device, seed):
    """Generate with a fixed per-seed noise stream and balanced class labels."""
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        noise = torch.randn((bs, 3, 32, 32), generator=generator, device=device)
        labels = (torch.arange(start, start + bs, device=device) % num_classes).long()
        images = sample_ode(model, noise, labels, num_classes, n_steps, solver,
                            condition_mode, cfg_scale)
        out.append((((images.clamp(-1, 1) + 1) * 127.5).round())
                   .to(torch.uint8).cpu())
        count = start + bs
        print(f"    generated {count}/{num_samples}", end="\r", flush=True)
    print()
    return torch.cat(out)


def load_real_images(data_root, num_samples):
    from torchvision import datasets, transforms
    ds = datasets.CIFAR10(data_root, train=True, download=True,
                          transform=transforms.ToTensor())
    if num_samples > len(ds):
        raise ValueError(f"num_samples={num_samples} exceeds CIFAR-10 train size {len(ds)}")
    images = torch.stack([ds[i][0] for i in range(num_samples)])
    return (images * 255).round().to(torch.uint8)


def calculate_fid(fake_uint8, real_uint8, batch_size, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for i in range(0, len(real_uint8), batch_size):
        metric.update(real_uint8[i:i + batch_size].to(device), real=True)
    for i in range(0, len(fake_uint8), batch_size):
        metric.update(fake_uint8[i:i + batch_size].to(device), real=False)
    return float(metric.compute().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--nfe", type=int, nargs="+", default=[4, 8, 16, 32, 50])
    parser.add_argument("--solver", choices=["euler", "heun"], nargs="+", default=["euler"])
    parser.add_argument("--condition", choices=["unconditional", "conditional", "cfg"],
                        default="unconditional")
    parser.add_argument("--cfg", type=float, nargs="+", default=[1.0],
                        help="guidance scale(s), used only when --condition cfg")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output", default="results/fid.json")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.model)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config"))
    model = build_model(cfg).to(device)
    if args.no_ema:
        weights, weight_name = ckpt["model"], "model"
    else:
        weights, weight_name = select_ema_state(ckpt, args.ema_decay)
    model.load_state_dict(weights)
    model.eval()
    num_classes = int(cfg["model"]["num_classes"])
    real_uint8 = load_real_images(args.data_root, args.num_samples)
    scales = args.cfg if args.condition == "cfg" else [1.0]
    results = []

    for seed in args.seeds:
        for solver in args.solver:
            for nfe in args.nfe:
                for scale in scales:
                    print(f"\n[eval] seed={seed} solver={solver} steps={nfe} "
                          f"condition={args.condition} cfg={scale}")
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    start = time.perf_counter()
                    fake = generate(model, args.num_samples, args.batch_size, nfe,
                                    num_classes, solver, args.condition, scale,
                                    device, seed)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    sample_seconds = time.perf_counter() - start
                    fid_start = time.perf_counter()
                    score = calculate_fid(fake, real_uint8, args.batch_size, device)
                    fid_seconds = time.perf_counter() - fid_start
                    solver_evals = nfe * (2 if solver == "heun" else 1)
                    guided_factor = 2 if args.condition == "cfg" else 1
                    row = {
                        "seed": seed, "solver": solver, "steps": nfe,
                        "condition_mode": args.condition,
                        "cfg_scale": scale if args.condition == "cfg" else None,
                        "forward_evaluations_per_sample": solver_evals * guided_factor,
                        "batched_model_calls": solver_evals,
                        "fid": score, "num_samples": args.num_samples,
                        "weights": weight_name,
                        "sample_seconds": sample_seconds,
                        "fid_seconds": fid_seconds,
                        "samples_per_second": args.num_samples / sample_seconds,
                    }
                    results.append(row)
                    print(f"  FID={score:.4f}; sample={sample_seconds/60:.1f} min; "
                          f"{row['samples_per_second']:.1f} images/s")

    metadata = {
        "protocol": "torchmetrics FrechetInceptionDistance(feature=2048), CIFAR-10 train indices [0:N], uint8 RGB",
        "checkpoint": str(ckpt_path), "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_step": ckpt.get("step"), "weights": weight_name,
        "git_revision": git_revision(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dataset_split": "CIFAR-10 train", "real_indices": f"0:{args.num_samples}",
        "num_samples": args.num_samples, "batch_size": args.batch_size,
        "seeds": args.seeds, "class_order": "balanced cyclic 0..9",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
