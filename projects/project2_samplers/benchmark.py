"""Reproducible FID-versus-NFE benchmark for Project 2 samplers."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .checkpoint_utils import load_model_weights, repository_commit
    from .project1_path import add_project1_to_path
    from .samplers import get_sampler
except ImportError:  # pragma: no cover - direct script compatibility
    from checkpoint_utils import load_model_weights, repository_commit
    from project1_path import add_project1_to_path
    from samplers import get_sampler


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def generate_samples(
    sampler,
    *,
    num_steps: int,
    num_samples: int,
    batch_size: int,
    image_shape: tuple[int, int, int],
    in_channels: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from dataset import denormalize

    generator = torch.Generator(device=device).manual_seed(seed)
    chunks: list[torch.Tensor] = []
    preview: list[torch.Tensor] = []
    generated = 0
    while generated < num_samples:
        current = min(batch_size, num_samples - generated)
        shape = (current, *image_shape)
        output = sampler.sample(shape, num_steps=num_steps, generator=generator)
        images = denormalize(output)
        if in_channels == 1:
            images = images.repeat(1, 3, 1, 1)
        uint8_images = (images * 255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        chunks.append(uint8_images)
        if sum(item.shape[0] for item in preview) < 64:
            preview.append(uint8_images[: 64 - sum(item.shape[0] for item in preview)])
        generated += current
        if generated == num_samples or generated % (batch_size * 10) == 0:
            print(f"    generated {generated}/{num_samples}")
    return torch.cat(chunks, dim=0), torch.cat(preview, dim=0)


@torch.no_grad()
def compute_fid(
    generated: torch.Tensor,
    real_dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    in_channels: int,
) -> float:
    from dataset import denormalize
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for start in range(0, generated.shape[0], batch_size):
        metric.update(generated[start : start + batch_size].to(device), real=False)

    loader = torch.utils.data.DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    real_count = 0
    for batch in loader:
        images = denormalize(batch.to(device))
        if in_channels == 1:
            images = images.repeat(1, 3, 1, 1)
        images = (images * 255.0).round().clamp(0, 255).to(torch.uint8)
        metric.update(images, real=True)
        real_count += images.shape[0]
    if real_count != generated.shape[0]:
        raise RuntimeError(
            f"FID sample mismatch: real={real_count}, generated={generated.shape[0]}"
        )
    return float(metric.compute().item())


def save_preview(images: torch.Tensor, output: Path) -> None:
    from torchvision.utils import save_image

    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(images.float() / 255.0, output, nrow=8)


def plot_pareto(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    # DDIM (eta=0) and the starter Euler sampler can be numerically identical
    # on this checkpoint.  Draw Euler first and DDIM last with a dashed line
    # so the DDIM curve remains visible instead of being painted over.
    plot_order = ("ddpm", "euler", "dpm-solver", "ddim")
    styles = {
        "ddpm": {"color": "#f58518", "linestyle": "-", "marker": "o", "zorder": 2},
        "euler": {"color": "#d62728", "linestyle": "-", "marker": "o", "zorder": 3},
        "dpm-solver": {
            "color": "#2ca02c",
            "linestyle": "-",
            "marker": "o",
            "zorder": 4,
        },
        "ddim": {
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "s",
            "zorder": 5,
        },
    }
    labels = {
        "ddpm": "DDPM",
        "ddim": "DDIM (eta=0)",
        "euler": "Euler",
        "dpm-solver": "DPM-Solver-2",
    }
    names = [
        name for name in plot_order if any(item["sampler"] == name for item in results)
    ]
    names.extend(
        sorted({item["sampler"] for item in results}.difference(names))
    )
    for sampler_name in names:
        points = sorted(
            (item for item in results if item["sampler"] == sampler_name),
            key=lambda item: item["nfe"],
        )
        style = styles.get(sampler_name, {})
        axis.plot(
            [item["nfe"] for item in points],
            [item["fid"] for item in points],
            label=labels.get(sampler_name, sampler_name),
            **style,
        )
    axis.set_xscale("log")
    axis.set_xlabel("NFE (model forward evaluations, log scale)")
    axis.set_ylabel("FID (lower is better)")
    axis.set_title("CIFAR-10 sampler quality versus compute")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def experiment_matrix(args, schedule) -> dict[str, list[int]]:
    if args.preset == "full":
        return {
            "ddpm": [int(schedule.T)],
            "ddim": [10, 20, 50, 100, 250],
            "euler": [10, 20, 50, 100, 250],
            "dpm-solver": [5, 10, 25, 50],
        }
    return {name: list(args.steps) for name in args.sampler}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--sampler",
        nargs="+",
        default=["ddim"],
        choices=["ddpm", "ddim", "dpm-solver", "euler"],
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--preset", choices=["custom", "full"], default="custom")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--ddim_timestep_strategy", default="linear")
    parser.add_argument("--dpm_timestep_strategy", default="lambda")
    parser.add_argument("--output", default="runs/benchmark_all.json")
    parser.add_argument("--plot", default="runs/pareto_fid_nfe.png")
    parser.add_argument("--samples_dir", default="samples")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--project1_path", default=None)
    parser.add_argument("--raw_weights", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 2:
        parser.error("--num_samples must be at least 2 for FID")

    project1 = add_project1_to_path(args.project1_path)
    from dataset import get_dataset
    from model import UNet
    from schedule import DDPMSchedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = UNet(**config["model"]).to(device)
    schedule = DDPMSchedule(**config["diffusion"]).to(device)
    weight_type = load_model_weights(model, checkpoint, use_ema=not args.raw_weights)
    model.eval()

    full_real_dataset = get_dataset(
        config["dataset"]["name"],
        root=args.data_root,
        image_size=config["model"]["image_size"],
        train=True,
        augment=False,
    )
    if args.num_samples > len(full_real_dataset):
        parser.error(
            f"Requested {args.num_samples} real images, dataset has {len(full_real_dataset)}"
        )
    real_dataset = torch.utils.data.Subset(
        full_real_dataset, list(range(args.num_samples))
    )
    image_shape = (
        int(config["model"]["in_channels"]),
        int(config["model"]["image_size"]),
        int(config["model"]["image_size"]),
    )

    results: list[dict] = []
    matrix = experiment_matrix(args, schedule)
    for sampler_name, step_values in matrix.items():
        for num_steps in step_values:
            if sampler_name == "ddpm" and num_steps != schedule.T:
                print(f"Skipping DDPM steps={num_steps}; DDPM requires T={schedule.T}")
                continue
            seed_everything(args.seed)
            timestep_strategy = (
                args.dpm_timestep_strategy
                if sampler_name == "dpm-solver"
                else args.ddim_timestep_strategy
            )
            sampler = get_sampler(
                sampler_name,
                model,
                schedule,
                device=device,
                eta=args.eta,
                timestep_strategy=timestep_strategy,
            )
            nfe = sampler.nfe_for_steps(num_steps)
            print(f"\n[benchmark] sampler={sampler_name}, steps={num_steps}, NFE={nfe}")
            synchronize(device)
            started = time.perf_counter()
            generated, preview = generate_samples(
                sampler,
                num_steps=num_steps,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                image_shape=image_shape,
                in_channels=image_shape[0],
                device=device,
                seed=args.seed,
            )
            synchronize(device)
            sample_seconds = time.perf_counter() - started
            preview_path = (
                Path(args.samples_dir)
                / f"{sampler_name.replace('-', '_')}_nfe{nfe}.png"
            )
            save_preview(preview, preview_path)

            started = time.perf_counter()
            fid = compute_fid(
                generated,
                real_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                in_channels=image_shape[0],
            )
            synchronize(device)
            fid_seconds = time.perf_counter() - started
            result = {
                "sampler": sampler_name,
                "num_steps": num_steps,
                "nfe": nfe,
                "fid": fid,
                "sample_seconds": sample_seconds,
                "fid_seconds": fid_seconds,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "timestep_strategy": timestep_strategy,
                "eta": args.eta if sampler_name == "ddim" else None,
                "preview": preview_path.as_posix(),
            }
            results.append(result)
            print(f"    FID={fid:.4f}, sampling={sample_seconds:.1f}s")

    payload = {
        "metadata": {
            "project1_path": str(project1),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "git_commit": repository_commit(),
            "weights": weight_type,
            "dataset": config["dataset"]["name"],
            "real_split": "train",
            "real_indices": f"0:{args.num_samples}",
            "augmentation": False,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if results:
        plot_pareto(results, Path(args.plot))
    print(f"\nSaved {output_path} and {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
