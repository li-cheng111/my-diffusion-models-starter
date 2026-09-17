"""Inspect reverse-sampling stability for a DDPM checkpoint and schedule."""

from __future__ import annotations

import argparse
import json
from statistics import median
from pathlib import Path

import torch

try:
    from .diffusion import p_sample_loop
    from .evaluate import _load_model
except ImportError:  # pragma: no cover - direct script compatibility
    from diffusion import p_sample_loop
    from evaluate import _load_model


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _abs_summary(value: torch.Tensor) -> dict[str, float]:
    """Summarize absolute values without retaining the full sampling trace."""

    flattened = value.detach().float().abs().reshape(-1)
    quantiles = torch.tensor((0.5, 0.95, 0.99), device=flattened.device)
    q50, q95, q99 = torch.quantile(flattened, quantiles).tolist()
    return {
        "median": float(q50),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(flattened.max().cpu()),
    }


@torch.no_grad()
def diagnose(
    checkpoint_path: str,
    num_samples: int,
    batch_size: int,
    seed: int,
    clip_denoised: bool,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, schedule, cfg, weight_type = _load_model(checkpoint, True, device)
    model.eval()
    schedule = schedule.to(device)
    all_steps: list[dict] = []
    generated = 0
    while generated < num_samples:
        current = min(batch_size, num_samples - generated)
        diagnostics: dict = {}
        _seed(seed + generated)
        p_sample_loop(
            model,
            (
                current,
                cfg["model"]["in_channels"],
                cfg["model"]["image_size"],
                cfg["model"]["image_size"],
            ),
            schedule,
            device=device,
            clip_denoised=clip_denoised,
            diagnostics=diagnostics,
        )
        all_steps.extend(diagnostics.get("steps", []))
        generated += current

    by_t: dict[int, list[dict]] = {}
    for row in all_steps:
        by_t.setdefault(int(row["t"]), []).append(row)
    step_summary = []
    for timestep in sorted(by_t, reverse=True):
        rows = by_t[timestep]
        summary = {"t": timestep}
        for prefix in ("xt_abs", "pred_x0_abs", "pred_noise_abs"):
            for statistic in ("median", "p95", "p99"):
                key = f"{prefix}_{statistic}"
                summary[key] = float(median(row[key] for row in rows))
            summary[f"{prefix}_max"] = max(row[f"{prefix}_max"] for row in rows)
        summary["pred_x0_outside_fraction"] = sum(
            row["pred_x0_outside_fraction"] for row in rows
        ) / len(rows)
        step_summary.append(summary)
    betas = schedule.betas.detach().cpu()
    return {
        "checkpoint": checkpoint_path,
        "weights": weight_type,
        "clip_denoised": clip_denoised,
        "num_samples": num_samples,
        "seed": seed,
        "schedule": {
            "name": schedule.beta_schedule,
            "T": schedule.T,
            "beta_min": float(betas.min()),
            "beta_max": float(betas.max()),
            "alpha_bar_final": float(schedule.alphas_cumprod[-1].cpu()),
            "posterior_variance_t0": float(schedule.posterior_variance[0].cpu()),
        },
        "steps": step_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = diagnose(
        args.ckpt,
        args.num_samples,
        args.batch_size,
        args.seed,
        args.clip_denoised,
    )
    output = Path(args.output) if args.output else Path(args.ckpt).parent / (
        "schedule_diagnostics_clipx0.json" if args.clip_denoised else "schedule_diagnostics_noclipx0.json"
    )
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["schedule"], indent=2))
    print(f"Schedule diagnostics written to {output}")


if __name__ == "__main__":
    main()
