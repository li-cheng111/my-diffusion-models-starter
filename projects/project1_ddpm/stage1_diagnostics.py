"""Stage-1 diagnostics for the FID<=15 gap.

This script does not train or modify checkpoints. It measures:

1. FID stability across generation seeds for selected checkpoints;
2. real-vs-real FID and the image tensor -> uint8 conversion path;
3. per-timestep epsilon and x0 reconstruction errors on fixed data/noise.

The default protocol intentionally matches Project 1's reported protocol:
5,000 generated images, the first 5,000 unaugmented CIFAR-10 train images,
seed 44, and clipped predicted x0 during DDPM sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Subset

try:  # Package mode and direct execution from the project directory.
    from .dataset import denormalize, get_dataset
    from .diffusion import p_sample_loop, q_sample
    from .evaluate import _load_model, _to_uint8_rgb
except ImportError:  # pragma: no cover - direct script compatibility
    from dataset import denormalize, get_dataset
    from diffusion import p_sample_loop, q_sample
    from evaluate import _load_model, _to_uint8_rgb


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_subset(dataset, start: int, count: int) -> Subset:
    if start < 0 or count <= 0 or start + count > len(dataset):
        raise ValueError(
            f"dataset subset [{start}, {start + count}) is outside dataset length {len(dataset)}"
        )
    return Subset(dataset, range(start, start + count))


def _fid_metric(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:  # pragma: no cover - depends on runtime extras
        raise ImportError(
            "Install torchmetrics[image] and torch-fidelity before running stage1 diagnostics."
        ) from exc
    return FrechetInceptionDistance(feature=2048, normalize=False).to(device)


@torch.no_grad()
def fid_model_against_real_subset(
    model: torch.nn.Module,
    schedule,
    real_subset,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    image_size: int,
    in_channels: int,
    clip_denoised: bool,
) -> float:
    """Compute FID with a fixed real subset and a seeded generated stream."""

    fid = _fid_metric(device)
    real_loader = DataLoader(
        real_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    loaded = 0
    for batch in real_loader:
        if loaded >= num_samples:
            break
        batch = batch[: min(batch.shape[0], num_samples - loaded)].to(device)
        fid.update(_to_uint8_rgb(batch, in_channels), real=True)
        loaded += batch.shape[0]

    generated = 0
    while generated < num_samples:
        current = min(batch_size, num_samples - generated)
        samples = p_sample_loop(
            model,
            (current, in_channels, image_size, image_size),
            schedule,
            device=device,
            clip_denoised=clip_denoised,
        )
        fid.update(_to_uint8_rgb(samples, in_channels), real=False)
        generated += current
    return float(fid.compute().item())


@torch.no_grad()
def real_vs_real_calibration(
    dataset,
    first_start: int,
    second_start: int,
    count: int,
    batch_size: int,
    device: torch.device,
    in_channels: int,
) -> dict[str, Any]:
    """Measure real-vs-real FID and image conversion statistics."""

    first = _dataset_subset(dataset, first_start, count)
    second = _dataset_subset(dataset, second_start, count)
    fid = _fid_metric(device)
    stats: dict[str, dict[str, float]] = {}

    def update_stats(name: str, subset: Subset) -> None:
        values: list[torch.Tensor] = []
        uint8_values: list[torch.Tensor] = []
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        loaded = 0
        for batch in loader:
            batch = batch.to(device)
            converted = _to_uint8_rgb(batch, in_channels)
            fid.update(converted, real=name == "first")
            values.append(denormalize(batch).float().cpu().reshape(-1))
            uint8_values.append(converted.float().cpu().reshape(-1))
            loaded += batch.shape[0]
        raw = torch.cat(values)
        quantized = torch.cat(uint8_values)
        stats[name] = {
            "normalized_min": float((raw * 2.0 - 1.0).min()),
            "normalized_max": float((raw * 2.0 - 1.0).max()),
            "denormalized_min": float(raw.min()),
            "denormalized_max": float(raw.max()),
            "denormalized_mean": float(raw.mean()),
            "denormalized_std": float(raw.std()),
            "uint8_min": float(quantized.min()),
            "uint8_max": float(quantized.max()),
            "uint8_mean": float(quantized.mean()),
            "uint8_std": float(quantized.std()),
            "samples": loaded,
        }

    update_stats("first", first)
    update_stats("second", second)
    return {
        "first_start": first_start,
        "second_start": second_start,
        "count_per_group": count,
        "fid_first_vs_second": float(fid.compute().item()),
        "conversion": stats,
        "conversion_note": "real images use the same [-1,1] -> [0,1] clamp -> uint8 path as generated images",
    }


def _quantile(values: list[torch.Tensor], q: float) -> float:
    return float(torch.quantile(torch.cat(values), q))


@torch.no_grad()
def timestep_diagnostics(
    checkpoint_path: str,
    data_root: str,
    split: str,
    num_images: int,
    batch_size: int,
    seed: int,
    timesteps: Iterable[int],
    ema_decay: float,
) -> dict[str, Any]:
    """Measure supervised epsilon/x0 errors at selected diffusion timesteps."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, schedule, cfg, weight_type = _load_model(
        checkpoint, True, device, ema_decay=ema_decay
    )
    model.eval()
    schedule = schedule.to(device)
    dataset = get_dataset(
        cfg["dataset"]["name"],
        root=data_root,
        image_size=cfg["model"]["image_size"],
        train=split == "train",
        augment=False,
    )
    subset = _dataset_subset(dataset, 0, num_images)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    selected = sorted(set(int(t) for t in timesteps), reverse=True)
    if any(t < 0 or t >= schedule.T for t in selected):
        raise ValueError(f"timesteps must be in [0, {schedule.T})")

    seed_everything(seed)
    buckets: dict[int, dict[str, Any]] = {
        t: {"eps_mse": [], "x0_abs": [], "x0_signed": [], "outside": []}
        for t in selected
    }
    for x0 in loader:
        x0 = x0.to(device)
        noise = torch.randn_like(x0)
        for timestep in selected:
            t = torch.full((x0.shape[0],), timestep, device=device, dtype=torch.long)
            xt = q_sample(
                x0,
                t,
                schedule.sqrt_alphas_cumprod,
                schedule.sqrt_one_minus_alphas_cumprod,
                noise=noise,
            )
            predicted_noise = model(xt, t)
            pred_x0 = (
                torch.rsqrt(schedule.alphas_cumprod[t])[:, None, None, None] * xt
                - torch.sqrt(
                    1.0 / schedule.alphas_cumprod[t] - 1.0
                )[:, None, None, None]
                * predicted_noise
            )
            buckets[timestep]["eps_mse"].append(
                (predicted_noise.float() - noise.float()).pow(2).mean(dim=(1, 2, 3)).cpu()
            )
            buckets[timestep]["x0_abs"].append(
                (pred_x0.float() - x0.float()).abs().reshape(x0.shape[0], -1).mean(dim=1).cpu()
            )
            buckets[timestep]["x0_signed"].append(
                (pred_x0.float() - x0.float()).pow(2).reshape(x0.shape[0], -1).mean(dim=1).cpu()
            )
            buckets[timestep]["outside"].append(
                (pred_x0.float().abs() > 1.0).float().reshape(x0.shape[0], -1).mean(dim=1).cpu()
            )

    rows: list[dict[str, float]] = []
    for timestep in selected:
        bucket = buckets[timestep]
        eps = torch.cat(bucket["eps_mse"])
        x0_abs = torch.cat(bucket["x0_abs"])
        x0_mse = torch.cat(bucket["x0_signed"])
        outside = torch.cat(bucket["outside"])
        alpha_bar = float(schedule.alphas_cumprod[timestep].detach().cpu())
        rows.append(
            {
                "t": timestep,
                "alpha_bar": alpha_bar,
                "snr": alpha_bar / max(1.0 - alpha_bar, 1e-20),
                "eps_mse_mean": float(eps.mean()),
                "eps_mse_p95": float(torch.quantile(eps, 0.95)),
                "x0_abs_error_mean": float(x0_abs.mean()),
                "x0_abs_error_p95": float(torch.quantile(x0_abs, 0.95)),
                "x0_mse_mean": float(x0_mse.mean()),
                "pred_x0_outside_fraction_mean": float(outside.mean()),
                "pred_x0_outside_fraction_p95": float(torch.quantile(outside, 0.95)),
            }
        )
    return {
        "checkpoint": checkpoint_path,
        "weights": weight_type,
        "ema_decay": ema_decay,
        "split": split,
        "num_images": num_images,
        "seed": seed,
        "timesteps": selected,
        "rows": rows,
        "note": "x0 errors use the known clean image and fixed sampled noise; outside fraction is before clipping",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3_ckpt", required=True)
    parser.add_argument("--r5_ckpt", required=True)
    parser.add_argument("--r3_ema_decay", type=float, default=0.9995)
    parser.add_argument("--r5_ema_decay", type=float, default=0.9999)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fid_num_samples", type=int, default=5000)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--stability_seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--calibration_count", type=int, default=5000)
    parser.add_argument("--diagnostic_images", type=int, default=256)
    parser.add_argument("--diagnostic_batch_size", type=int, default=64)
    parser.add_argument(
        "--diagnostic_timesteps",
        type=int,
        nargs="+",
        default=[0, 1, 10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 975, 990, 995, 998, 999],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r3_checkpoint = torch.load(args.r3_ckpt, map_location="cpu")
    r5_checkpoint = torch.load(args.r5_ckpt, map_location="cpu")
    r3_cfg = r3_checkpoint["config"]
    r5_cfg = r5_checkpoint["config"]
    if r3_cfg["dataset"]["name"].lower() != r5_cfg["dataset"]["name"].lower():
        raise ValueError("R3 and R5 must use the same dataset for this comparison")
    dataset = get_dataset(
        r5_cfg["dataset"]["name"],
        root=args.data_root,
        image_size=r5_cfg["model"]["image_size"],
        train=True,
        augment=False,
    )

    stability_rows: list[dict[str, Any]] = []
    for label, ckpt_path, ema_decay in (
        ("R3", args.r3_ckpt, args.r3_ema_decay),
        ("R5", args.r5_ckpt, args.r5_ema_decay),
    ):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model, schedule, cfg, weight_type = _load_model(
            checkpoint, True, device, ema_decay=ema_decay
        )
        real_subset = _dataset_subset(dataset, 0, args.fid_num_samples)
        for seed in args.stability_seeds:
            seed_everything(seed)
            score = fid_model_against_real_subset(
                model,
                schedule,
                real_subset,
                args.fid_num_samples,
                args.fid_batch_size,
                device,
                cfg["model"]["image_size"],
                cfg["model"]["in_channels"],
                clip_denoised=True,
            )
            row = {
                "experiment": label,
                "checkpoint": ckpt_path,
                "ema_decay": ema_decay,
                "weights": weight_type,
                "seed": seed,
                "real_split": "train",
                "real_start": 0,
                "real_samples": args.fid_num_samples,
                "generated_samples": args.fid_num_samples,
                "clip_denoised": True,
                "fid": score,
            }
            stability_rows.append(row)
            print(f"[{label} seed={seed}] FID={score:.4f}", flush=True)
        del model, schedule
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(output_dir / "fid_stability.csv", stability_rows)
    calibration = real_vs_real_calibration(
        dataset,
        first_start=0,
        second_start=args.calibration_count,
        count=args.calibration_count,
        batch_size=args.fid_batch_size,
        device=device,
        in_channels=r5_cfg["model"]["in_channels"],
    )
    (output_dir / "real_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(
        f"[calibration] real first {args.calibration_count} vs next {args.calibration_count} "
        f"FID={calibration['fid_first_vs_second']:.4f}",
        flush=True,
    )

    timestep_files: list[str] = []
    for label, ckpt_path, ema_decay in (
        ("R3", args.r3_ckpt, args.r3_ema_decay),
        ("R5", args.r5_ckpt, args.r5_ema_decay),
    ):
        for split in ("train", "test"):
            result = timestep_diagnostics(
                ckpt_path,
                args.data_root,
                split,
                args.diagnostic_images,
                args.diagnostic_batch_size,
                seed=44,
                timesteps=args.diagnostic_timesteps,
                ema_decay=ema_decay,
            )
            path = output_dir / f"timestep_{label.lower()}_{split}.json"
            path.write_text(json.dumps(result, indent=2))
            timestep_files.append(str(path))
            print(f"[timestep] wrote {path}", flush=True)

    summary = {
        "protocol": {
            "dataset": r5_cfg["dataset"]["name"],
            "real_split": "train",
            "real_start": 0,
            "real_samples": args.fid_num_samples,
            "generated_samples": args.fid_num_samples,
            "clip_denoised": True,
            "fid_seeds": args.stability_seeds,
        },
        "fid_stability_csv": str(output_dir / "fid_stability.csv"),
        "real_calibration": str(output_dir / "real_calibration.json"),
        "timestep_files": timestep_files,
        "interpretation": {
            "formal_seed": 44,
            "extra_seeds_are_diagnostic_only": True,
            "no_training_performed": True,
        },
    }
    (output_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Stage 1 diagnostics written to {output_dir}")


if __name__ == "__main__":
    main()
