"""Measure FID variance caused by real subsets and generation seeds.

This is a no-training diagnostic.  It keeps the official Project 1 protocol
(the first 5,000 CIFAR-10 train images and generation seed 44) and adds four
reproducible, class-stratified real subsets.  Generated uint8 images are
cached once per checkpoint/seed so changing the real subset never repeats
the expensive 1,000-step DDPM sampling loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import get_dataset
    from .diffusion import p_sample_loop
    from .evaluate import _load_model, _to_uint8_rgb
except ImportError:  # pragma: no cover - direct script compatibility
    from dataset import get_dataset
    from diffusion import p_sample_loop
    from evaluate import _load_model, _to_uint8_rgb


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fid_metric(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Install torchmetrics[image] and torch-fidelity before running this diagnostic."
        ) from exc
    return FrechetInceptionDistance(feature=2048, normalize=False).to(device)


def dataset_targets(dataset) -> list[int]:
    base = getattr(dataset, "base", dataset)
    targets = getattr(base, "targets", None)
    if targets is None:
        raise TypeError("The dataset must expose CIFAR-10 targets for stratified subsets.")
    return [int(value) for value in targets]


def label_histogram(indices: list[int], targets: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indices:
        key = str(targets[index])
        counts[key] = counts.get(key, 0) + 1
    return {key: counts[key] for key in sorted(counts, key=int)}


def build_manifest(
    dataset,
    subset_size: int,
    random_subset_count: int,
    seed: int,
) -> dict[str, Any]:
    targets = dataset_targets(dataset)
    if subset_size <= 0 or subset_size > len(dataset):
        raise ValueError(f"subset_size must be in [1, {len(dataset)}]")
    if subset_size % 10 != 0:
        raise ValueError("subset_size must be divisible by 10 for balanced CIFAR-10 subsets")
    if random_subset_count <= 0:
        raise ValueError("random_subset_count must be positive")

    subsets: list[dict[str, Any]] = []
    official = list(range(subset_size))
    subsets.append(
        {
            "name": "official_first5000",
            "kind": "official",
            "seed": None,
            "indices": official,
            "label_histogram": label_histogram(official, targets),
        }
    )

    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(target, []).append(index)
    per_class = subset_size // 10
    for target in range(10):
        if len(by_class.get(target, [])) < per_class * random_subset_count:
            raise ValueError(f"Not enough examples for class {target}")

    shuffled_by_class: dict[int, list[int]] = {}
    for target in range(10):
        values = torch.tensor(by_class[target], dtype=torch.long)
        order = torch.randperm(len(values), generator=generator)
        shuffled_by_class[target] = values[order].tolist()

    for subset_index in range(random_subset_count):
        indices: list[int] = []
        start = subset_index * per_class
        stop = start + per_class
        for target in range(10):
            indices.extend(shuffled_by_class[target][start:stop])
        # Shuffle the final order, while keeping the selected set fixed.
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices = [indices[position] for position in order]
        subsets.append(
            {
                "name": f"stratified_{subset_index + 1}",
                "kind": "stratified_random",
                "seed": seed,
                "indices": indices,
                "label_histogram": label_histogram(indices, targets),
            }
        )

    return {
        "dataset": "cifar10",
        "split": "train",
        "subset_size": subset_size,
        "random_subset_count": random_subset_count,
        "stratification": "500 images per class for each random subset",
        "seed": seed,
        "subsets": subsets,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def generate_cached_images(
    model,
    schedule,
    checkpoint_label: str,
    seed: int,
    output_path: Path,
    device: torch.device,
    image_size: int,
    in_channels: int,
    num_samples: int,
    batch_size: int,
    clip_denoised: bool,
    force: bool,
) -> torch.Tensor:
    if output_path.exists() and not force:
        cached = torch.load(output_path, map_location="cpu")
        images = cached["images"] if isinstance(cached, dict) else cached
        if tuple(images.shape) == (num_samples, 3, image_size, image_size):
            print(f"[cache] using {output_path}", flush=True)
            return images.to(torch.uint8).contiguous()
        raise ValueError(f"Cached image shape is incompatible: {tuple(images.shape)}")

    print(
        f"[generate] {checkpoint_label} seed={seed} samples={num_samples}",
        flush=True,
    )
    seed_everything(seed)
    batches: list[torch.Tensor] = []
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
        batches.append(_to_uint8_rgb(samples, in_channels).cpu())
        generated += current
        if generated % (batch_size * 10) == 0 or generated == num_samples:
            print(f"[generate] {checkpoint_label} seed={seed} {generated}/{num_samples}", flush=True)
    images = torch.cat(batches, dim=0).to(torch.uint8).contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "images": images,
            "checkpoint_label": checkpoint_label,
            "seed": seed,
            "clip_denoised": clip_denoised,
        },
        output_path,
    )
    print(f"[cache] wrote {output_path}", flush=True)
    return images


@torch.no_grad()
def fid_for_subset(
    real_dataset,
    real_indices: list[int],
    generated_images: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    fid = fid_metric(device)
    real_loader = DataLoader(
        Subset(real_dataset, real_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    for batch in real_loader:
        fid.update(_to_uint8_rgb(batch.to(device), 3), real=True)
    for start in range(0, len(generated_images), batch_size):
        fid.update(generated_images[start : start + batch_size].to(device), real=False)
    return float(fid.compute().item())


def bootstrap_ci(values: list[float], seed: int, draws: int = 2000) -> list[float]:
    if len(values) < 2:
        value = values[0] if values else float("nan")
        return [value, value]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row['experiment']}|{row['subset']}"
        groups.setdefault(key, []).append(float(row["fid"]))
    summary: dict[str, Any] = {}
    for key, values in groups.items():
        summary[key] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "bootstrap_95_ci": bootstrap_ci(values, seed=20260919),
        }
    for experiment in sorted({row["experiment"] for row in rows}):
        values = [float(row["fid"]) for row in rows if row["experiment"] == experiment]
        summary[f"{experiment}|ALL"] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "bootstrap_95_ci": bootstrap_ci(values, seed=20260920),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3_ckpt", required=True)
    parser.add_argument("--r5_ckpt", required=True)
    parser.add_argument("--r3_ema_decay", type=float, default=0.9995)
    parser.add_argument("--r5_ema_decay", type=float, default=0.9999)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--random_subset_count", type=int, default=4)
    parser.add_argument("--subset_seed", type=int, default=20260919)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gen_seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--force_regenerate", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    r5_checkpoint = torch.load(args.r5_ckpt, map_location="cpu")
    r5_cfg = r5_checkpoint["config"]
    dataset_name = r5_cfg["dataset"]["name"]
    if dataset_name.lower() != "cifar10":
        raise ValueError("This variance experiment is intended for CIFAR-10.")
    real_dataset = get_dataset(
        dataset_name,
        root=args.data_root,
        image_size=r5_cfg["model"]["image_size"],
        train=True,
        augment=False,
    )
    manifest_path = output_dir / "real_subset_manifest.json"
    if manifest_path.exists() and not args.force_regenerate:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"[manifest] using {manifest_path}", flush=True)
    else:
        manifest = build_manifest(
            real_dataset,
            subset_size=args.subset_size,
            random_subset_count=args.random_subset_count,
            seed=args.subset_seed,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[manifest] wrote {manifest_path}", flush=True)

    experiments = (
        ("R3", args.r3_ckpt, args.r3_ema_decay),
        ("R5", args.r5_ckpt, args.r5_ema_decay),
    )
    rows: list[dict[str, Any]] = []
    for label, checkpoint_path, ema_decay in experiments:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model, schedule, cfg, weight_type = _load_model(
            checkpoint, True, device, ema_decay=ema_decay
        )
        for seed in args.gen_seeds:
            cache_path = cache_dir / f"{label.lower()}_ema{ema_decay:.4f}_seed{seed}.pt"
            generated = generate_cached_images(
                model,
                schedule,
                label,
                seed,
                cache_path,
                device,
                cfg["model"]["image_size"],
                cfg["model"]["in_channels"],
                args.num_samples,
                args.batch_size,
                clip_denoised=True,
                force=args.force_regenerate,
            )
            if len(generated) != args.num_samples:
                raise ValueError("Generated cache has an unexpected number of images")
            for subset in manifest["subsets"]:
                score = fid_for_subset(
                    real_dataset,
                    subset["indices"],
                    generated,
                    args.batch_size,
                    device,
                )
                row = {
                    "experiment": label,
                    "checkpoint": checkpoint_path,
                    "ema_decay": ema_decay,
                    "weights": weight_type,
                    "subset": subset["name"],
                    "subset_kind": subset["kind"],
                    "subset_seed": subset["seed"],
                    "generated_seed": seed,
                    "real_samples": args.subset_size,
                    "generated_samples": args.num_samples,
                    "clip_denoised": True,
                    "fid": score,
                }
                rows.append(row)
                print(
                    f"[fid] {label} real={subset['name']} gen_seed={seed} FID={score:.4f}",
                    flush=True,
                )
        del model, schedule
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "real_subset_fid.csv"
    write_csv(csv_path, rows)
    summary = {
        "protocol": {
            "real_split": "train",
            "subset_size": args.subset_size,
            "generated_samples": args.num_samples,
            "generated_seeds": args.gen_seeds,
            "clip_denoised": True,
            "subset_manifest": str(manifest_path),
        },
        "summary": summarize(rows),
        "cache_dir": str(cache_dir),
        "csv": str(csv_path),
        "no_training_performed": True,
    }
    (output_dir / "real_subset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote {csv_path} and {output_dir / 'real_subset_summary.json'}")


if __name__ == "__main__":
    main()
