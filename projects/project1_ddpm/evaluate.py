"""FID evaluation for generated images versus a held-out dataset transform."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:  # 支持 ``python -m`` 和 starter 原有的直接脚本入口。
    from .dataset import denormalize, get_dataset
    from .diffusion import p_sample_loop
    from .model import UNet
    from .schedule import DDPMSchedule
except ImportError:  # pragma: no cover - direct script compatibility
    from dataset import denormalize, get_dataset
    from diffusion import p_sample_loop
    from model import UNet
    from schedule import DDPMSchedule


def _to_uint8_rgb(x: torch.Tensor, in_channels: int) -> torch.Tensor:
    x = denormalize(x)
    if in_channels == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x * 255.0).clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def compute_fid(
    model: torch.nn.Module,
    schedule: DDPMSchedule,
    real_dataset,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    image_size: int,
    in_channels: int,
    clip_denoised: bool = False,
) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "Install torchmetrics[image] and torch-fidelity before FID evaluation."
        ) from exc

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    real_loader = torch.utils.data.DataLoader(
        real_dataset,
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
        current_batch = min(batch_size, num_samples - generated)
        samples = p_sample_loop(
            model,
            (current_batch, in_channels, image_size, image_size),
            schedule,
            device=device,
            clip_denoised=clip_denoised,
        )
        fid.update(_to_uint8_rgb(samples, in_channels), real=False)
        generated += current_batch
    return float(fid.compute().item())


def _load_model(
    checkpoint: dict,
    use_ema: bool,
    device: torch.device,
) -> tuple[torch.nn.Module, DDPMSchedule, dict, str]:
    """Create an evaluator model from either raw or EMA checkpoint weights."""

    cfg = checkpoint["config"]
    model = UNet(**cfg["model"]).to(device)
    schedule = DDPMSchedule(**cfg["diffusion"]).to(device)
    if use_ema:
        if "ema" not in checkpoint:
            raise ValueError("The checkpoint does not contain EMA weights.")
        ema_state = checkpoint["ema"]
        model.load_state_dict(ema_state.get("model", ema_state))
        weight_type = "EMA"
    else:
        model.load_state_dict(checkpoint["model"])
        weight_type = "raw"
    model.eval()
    return model, schedule, cfg, weight_type


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument(
        "--compare_ema",
        action="store_true",
        help="Evaluate both EMA and raw weights using the same random seed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="Clip predicted x_0 to [-1, 1] before each reverse posterior update.",
    )
    parser.add_argument("--data_root", default="./data")
    parser.add_argument(
        "--real_split",
        choices=("train", "test"),
        default="train",
        help="Dataset split used for FID real images; advanced track uses train.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    cfg = checkpoint["config"]
    image_size = cfg["model"]["image_size"]
    in_channels = cfg["model"]["in_channels"]
    if args.compare_ema and args.no_ema:
        parser.error("--compare_ema and --no_ema cannot be used together")
    if args.compare_ema:
        weight_options = (True, False)
    else:
        weight_options = (not args.no_ema,)

    real_dataset = get_dataset(
        cfg["dataset"]["name"],
        root=args.data_root,
        image_size=image_size,
        train=args.real_split == "train",
        augment=False,
    )

    results = {}
    for use_ema in weight_options:
        _seed_everything(args.seed)
        model, schedule, _, weight_type = _load_model(checkpoint, use_ema, device)
        score = compute_fid(
            model,
            schedule,
            real_dataset,
            args.num_samples,
            args.batch_size,
            device,
            image_size,
            in_channels,
            clip_denoised=args.clip_denoised,
        )
        results[weight_type] = score
        clip_suffix = "_clipx0" if args.clip_denoised else "_noclipx0"
        result_path = Path(args.ckpt).parent / f"fid_{args.num_samples}_{weight_type}{clip_suffix}.txt"
        result_path.write_text(
            f"FID: {score:.4f}\n"
            f"num_samples: {args.num_samples}\n"
            f"real_split: {args.real_split}\n"
            f"weights: {weight_type}\n"
            f"checkpoint: {args.ckpt}\n"
            f"seed: {args.seed}\n"
            f"clip_denoised: {args.clip_denoised}\n"
        )
        print(f"FID @ {args.num_samples} samples ({weight_type}, real={args.real_split}): {score:.4f}")

    if len(results) == 2:
        clip_suffix = "_clipx0" if args.clip_denoised else "_noclipx0"
        comparison_path = Path(args.ckpt).parent / f"fid_comparison{clip_suffix}.md"
        ema_score = results["EMA"]
        raw_score = results["raw"]
        comparison_path.write_text(
            "# CIFAR-10 EMA comparison\n\n"
            f"- Real split: `{args.real_split}`\n"
            f"- Real samples: `{args.num_samples}`\n"
            f"- Generated samples per run: `{args.num_samples}`\n"
            f"- Seed: `{args.seed}`\n\n"
            f"- Clip predicted x0: `{args.clip_denoised}`\n\n"
            "| Weights | FID |\n"
            "|---|---:|\n"
            f"| EMA | {ema_score:.4f} |\n"
            f"| Raw | {raw_score:.4f} |\n"
            f"| Raw - EMA | {raw_score - ema_score:+.4f} |\n"
        )
        print(f"EMA comparison written to {comparison_path}")


if __name__ == "__main__":
    main()
