"""Compare the local DDPM reverse equations with an independent reference.

The check uses fixed x_t and epsilon predictions, so it does not depend on a
checkpoint or on random sampling noise. It also records robust magnitude
statistics for the x_0 reconstruction at the most sensitive timesteps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .diffusion import p_sample
    from .schedule import DDPMSchedule
except ImportError:  # pragma: no cover - direct script compatibility
    from diffusion import p_sample
    from schedule import DDPMSchedule


def _extract(values: torch.Tensor, t: int, shape: tuple[int, ...]) -> torch.Tensor:
    return values[t].reshape(1, *([1] * (len(shape) - 1)))


def _stats(value: torch.Tensor) -> dict[str, float]:
    absolute = value.abs().reshape(-1)
    q50, q95, q99 = torch.quantile(
        absolute, torch.tensor((0.50, 0.95, 0.99), dtype=absolute.dtype)
    ).tolist()
    return {
        "median": float(q50),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(absolute.max()),
        "outside_fraction": float((absolute > 1.0).double().mean()),
    }


class _FixedEpsilonModel(torch.nn.Module):
    def __init__(self, epsilon: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("epsilon", epsilon)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del x, t
        return self.epsilon


def _reference_from_eps(
    xt: torch.Tensor,
    eps: torch.Tensor,
    t: int,
    schedule: DDPMSchedule,
    clip_denoised: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent q-posterior implementation, following Eq. 9-11."""

    alpha_bar = schedule.alphas_cumprod.double()
    alpha_bar_prev = schedule.alphas_cumprod_prev.double()
    alpha = schedule.alphas.double()
    beta = schedule.betas.double()
    x0 = (
        xt - _extract(torch.sqrt(1.0 - alpha_bar), t, xt.shape) * eps
    ) / _extract(torch.sqrt(alpha_bar), t, xt.shape)
    if clip_denoised:
        x0 = x0.clamp(-1.0, 1.0)
    denominator = 1.0 - _extract(alpha_bar, t, xt.shape)
    coef1 = (
        _extract(beta, t, xt.shape)
        * _extract(torch.sqrt(alpha_bar_prev), t, xt.shape)
        / denominator
    )
    coef2 = (
        _extract(torch.sqrt(alpha), t, xt.shape)
        * (1.0 - _extract(alpha_bar_prev, t, xt.shape))
        / denominator
    )
    mean = coef1 * x0 + coef2 * xt
    variance = _extract(schedule.posterior_variance.double(), t, xt.shape)
    return mean, variance, x0


def _local_equations_from_eps(
    xt: torch.Tensor,
    eps: torch.Tensor,
    t: int,
    schedule: DDPMSchedule,
    clip_denoised: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the equations used by diffusion.p_sample."""

    shape = xt.shape
    beta = _extract(schedule.betas.double(), t, shape)
    alpha = _extract(schedule.alphas.double(), t, shape)
    alpha_bar = _extract(schedule.alphas_cumprod.double(), t, shape)
    alpha_bar_prev = _extract(schedule.alphas_cumprod_prev.double(), t, shape)
    sqrt_one_minus_alpha_bar = _extract(
        schedule.sqrt_one_minus_alphas_cumprod.double(), t, shape
    )
    if clip_denoised:
        x0 = torch.rsqrt(alpha_bar) * xt - torch.sqrt(1.0 / alpha_bar - 1.0) * eps
        x0 = x0.clamp(-1.0, 1.0)
        denominator = 1.0 - alpha_bar
        mean = (
            beta * torch.sqrt(alpha_bar_prev) / denominator * x0
            + torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / denominator * xt
        )
    else:
        x0 = torch.rsqrt(alpha_bar) * xt - torch.sqrt(1.0 / alpha_bar - 1.0) * eps
        mean = torch.rsqrt(alpha) * (
            xt - beta * eps / sqrt_one_minus_alpha_bar.clamp(min=1e-20)
        )
    variance = _extract(schedule.posterior_variance.double(), t, shape)
    return mean, variance, x0


def validate(beta_schedule: str, timesteps: int) -> dict:
    schedule = DDPMSchedule(T=timesteps, beta_schedule=beta_schedule).double()
    values = torch.linspace(-0.75, 0.75, 2 * 3 * 8 * 8, dtype=torch.float64)
    xt = values.reshape(2, 3, 8, 8)
    eps = torch.linspace(0.15, -0.15, xt.numel(), dtype=torch.float64).reshape_as(xt)
    rows = []
    for timestep in (timesteps - 1, timesteps - 2, 0):
        for clip_denoised in (False, True):
            local_mean, local_variance, local_x0 = _local_equations_from_eps(
                xt, eps, timestep, schedule, clip_denoised
            )
            reference_mean, reference_variance, reference_x0 = _reference_from_eps(
                xt, eps, timestep, schedule, clip_denoised
            )
            production_sample = p_sample(
                _FixedEpsilonModel(eps),
                xt,
                torch.full((xt.shape[0],), timestep, dtype=torch.long),
                schedule,
                clip_denoised=clip_denoised,
                noise=torch.zeros_like(xt),
            )
            rows.append(
                {
                    "t": timestep,
                    "clip_denoised": clip_denoised,
                    "mean_max_abs_diff": float((local_mean - reference_mean).abs().max()),
                    "production_p_sample_max_abs_diff": float(
                        (production_sample - reference_mean).abs().max()
                    ),
                    "variance_max_abs_diff": float(
                        (local_variance - reference_variance).abs().max()
                    ),
                    "x0_max_abs_diff": float((local_x0 - reference_x0).abs().max()),
                    "local_x0_abs": _stats(local_x0),
                    "reference_x0_abs": _stats(reference_x0),
                }
            )
    max_error = max(
        max(
            row["mean_max_abs_diff"],
            row["production_p_sample_max_abs_diff"],
            row["variance_max_abs_diff"],
            row["x0_max_abs_diff"],
        )
        for row in rows
    )
    # The schedule stores float32 buffers. At t=0 the direct epsilon form
    # subtracts nearly equal values and can differ from the posterior form by
    # about 1e-4 in float64 after those buffers have been rounded. This is a
    # numerical representation effect, so keep it explicit in the report.
    tolerance = 2e-4
    return {
        "beta_schedule": beta_schedule,
        "T": timesteps,
        "fixed_input": {
            "x_t": "torch.linspace(-0.75, 0.75, 2*3*8*8).reshape(2,3,8,8)",
            "epsilon": "torch.linspace(0.15, -0.15, 2*3*8*8).reshape(2,3,8,8)",
        },
        "schedule": {
            "beta_min": float(schedule.betas.min()),
            "beta_max": float(schedule.betas.max()),
            "alpha_bar_final": float(schedule.alphas_cumprod[-1]),
        },
        "max_formula_error": max_error,
        "tolerance": tolerance,
        "pass": max_error < tolerance,
        "checks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta_schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--output", default="runs/sampler_validation.json")
    args = parser.parse_args()
    result = validate(args.beta_schedule, args.T)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"pass": result["pass"], "max_formula_error": result["max_formula_error"]}))
    print(f"Sampler validation written to {output}")
    if not result["pass"]:
        raise SystemExit("Sampler formula comparison failed")


if __name__ == "__main__":
    main()
