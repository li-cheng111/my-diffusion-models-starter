"""Core DDPM forward-process, loss, and reverse-sampling functions."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

try:  # 支持包模式和从项目目录直接运行 starter 脚本。
    from .schedule import DDPMSchedule
except ImportError:  # pragma: no cover - direct script compatibility
    from schedule import DDPMSchedule


def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: Sequence[int]) -> torch.Tensor:
    """Gather one scalar per batch item and broadcast it over image dimensions."""

    if t.ndim != 1:
        raise ValueError(f"t must have shape (B,), got {tuple(t.shape)}")
    if len(x_shape) < 2 or t.shape[0] != x_shape[0]:
        raise ValueError("t batch size must match the first dimension of x")
    gathered = values.gather(0, t)
    return gathered.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


def _abs_summary(value: torch.Tensor) -> dict[str, float]:
    """Return robust magnitude statistics for a sampling diagnostic."""

    flattened = value.detach().float().abs().reshape(-1)
    q50, q95, q99 = torch.quantile(
        flattened, torch.tensor((0.5, 0.95, 0.99), device=flattened.device)
    ).tolist()
    return {
        "median": float(q50),
        "p95": float(q95),
        "p99": float(q99),
        "max": float(flattened.max().cpu()),
    }


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample x_t directly from x_0 using the closed form of q(x_t | x_0)."""

    if noise is None:
        noise = torch.randn_like(x0)
    if noise.shape != x0.shape:
        raise ValueError("noise and x0 must have identical shapes")
    t = t.to(device=x0.device, dtype=torch.long)
    sqrt_alpha_bar = sqrt_alpha_bar.to(device=x0.device)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device=x0.device)
    scale_signal = _extract(sqrt_alpha_bar, t, x0.shape)
    scale_noise = _extract(sqrt_one_minus_alpha_bar, t, x0.shape)
    return scale_signal * x0 + scale_noise * noise


def p_losses(
    model: torch.nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """Compute the simplified DDPM noise-prediction MSE."""

    noise = torch.randn_like(x0)
    xt = q_sample(
        x0,
        t,
        schedule.sqrt_alphas_cumprod,
        schedule.sqrt_one_minus_alphas_cumprod,
        noise=noise,
    )
    predicted_noise = model(xt, t)
    if predicted_noise.shape != noise.shape:
        raise ValueError(
            "model output must have the same shape as the injected noise: "
            f"{tuple(predicted_noise.shape)} != {tuple(noise.shape)}"
        )
    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def p_sample(
    model: torch.nn.Module,
    xt: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule,
    clip_denoised: bool = False,
    diagnostics: Optional[dict[str, Any]] = None,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Perform one stochastic reverse step x_t -> x_{t-1}.

    ``noise`` is optional and exists so deterministic formula checks can call
    the production sampler with a fixed random term. Normal sampling keeps
    the original behavior when it is omitted.
    """

    if isinstance(t, int):
        t = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
    else:
        t = t.to(device=xt.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(xt.shape[0])
        elif t.ndim == 1 and t.numel() == 1 and xt.shape[0] != 1:
            t = t.expand(xt.shape[0])
    if t.ndim != 1 or t.shape[0] != xt.shape[0]:
        raise ValueError("t must contain one timestep per sample")

    predicted_noise = model(xt, t)
    beta_t = _extract(schedule.betas, t, xt.shape)
    sqrt_recip_alpha_t = _extract(schedule.sqrt_recip_alphas, t, xt.shape)
    sqrt_one_minus_alpha_bar_t = _extract(
        schedule.sqrt_one_minus_alphas_cumprod, t, xt.shape
    )

    pred_x0 = None
    if clip_denoised or diagnostics is not None:
        # Convert epsilon prediction to x_0, constrain it to the data range,
        # and then use q(x_{t-1} | x_t, x_0) for the reverse mean. The noisy
        # state xt is intentionally not clipped.
        alpha_bar_t = _extract(schedule.alphas_cumprod, t, xt.shape)
        alpha_bar_prev_t = _extract(schedule.alphas_cumprod_prev, t, xt.shape)
        pred_x0 = (
            torch.rsqrt(alpha_bar_t) * xt
            - torch.sqrt(1.0 / alpha_bar_t - 1.0) * predicted_noise
        )
        if diagnostics is not None:
            xt_stats = _abs_summary(xt)
            pred_x0_stats = _abs_summary(pred_x0)
            pred_noise_stats = _abs_summary(predicted_noise)
            diagnostics.setdefault("steps", []).append(
                {
                    "t": int(t[0].item()),
                    "xt_abs_median": xt_stats["median"],
                    "xt_abs_p95": xt_stats["p95"],
                    "xt_abs_p99": xt_stats["p99"],
                    "xt_abs_max": xt_stats["max"],
                    "pred_x0_abs_median": pred_x0_stats["median"],
                    "pred_x0_abs_p95": pred_x0_stats["p95"],
                    "pred_x0_abs_p99": pred_x0_stats["p99"],
                    "pred_x0_abs_max": pred_x0_stats["max"],
                    "pred_x0_outside_fraction": float(
                        (pred_x0.detach().abs() > 1.0).float().mean().cpu()
                    ),
                    "pred_noise_abs_median": pred_noise_stats["median"],
                    "pred_noise_abs_p95": pred_noise_stats["p95"],
                    "pred_noise_abs_p99": pred_noise_stats["p99"],
                    "pred_noise_abs_max": pred_noise_stats["max"],
                }
            )
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
    if clip_denoised:
        posterior_mean_coef1 = beta_t * torch.sqrt(alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        posterior_mean_coef2 = (
            _extract(schedule.alphas, t, xt.shape).sqrt()
            * (1.0 - alpha_bar_prev_t)
            / (1.0 - alpha_bar_t)
        )
        mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * xt
    else:
        mean = sqrt_recip_alpha_t * (
            xt - beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t.clamp(min=1e-20)
        )

    variance_t = _extract(schedule.posterior_variance, t, xt.shape)
    if noise is None:
        noise = torch.randn_like(xt)
    elif noise.shape != xt.shape:
        raise ValueError("noise and xt must have identical shapes")
    nonzero_mask = (t != 0).to(dtype=xt.dtype).reshape(xt.shape[0], *([1] * (xt.ndim - 1)))
    return mean + nonzero_mask * torch.sqrt(variance_t.clamp(min=0.0)) * noise


@torch.no_grad()
def p_sample_loop(
    model: torch.nn.Module,
    shape: Sequence[int],
    schedule: DDPMSchedule,
    device: Optional[torch.device] = None,
    clip_denoised: bool = False,
    diagnostics: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Generate a batch by applying all T reverse steps."""

    if len(shape) != 4:
        raise ValueError(f"Expected image shape (B,C,H,W), got {tuple(shape)}")
    if device is None:
        device = next(model.parameters()).device
    x = torch.randn(tuple(shape), device=device)
    for step in reversed(range(schedule.T)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x = p_sample(
            model,
            x,
            t,
            schedule,
            clip_denoised=clip_denoised,
            diagnostics=diagnostics,
        )
    return x
