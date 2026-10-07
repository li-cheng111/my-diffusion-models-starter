"""DDPM ancestral sampler used as the full-trajectory reference."""

from __future__ import annotations

import torch

from .base import BaseSampler, randn, sqrt_one_minus_alpha_bar


class DDPMSampler(BaseSampler):
    def __init__(self, model, schedule, device="cuda", clip_denoised: bool = True):
        super().__init__(model, schedule, device)
        self.clip_denoised = bool(clip_denoised)

    def get_timesteps(self, num_steps: int) -> list[int]:
        if num_steps != self.T:
            print(f"[warning] DDPM requires T={self.T}; ignoring num_steps={num_steps}")
        return list(range(self.T - 1, -1, -1))

    @torch.no_grad()
    def step(self, x_t, eps_pred, t, t_prev, generator=None):
        schedule = self.schedule
        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        if self.clip_denoised:
            alpha_bar = schedule.alphas_cumprod[t]
            alpha_bar_prev = schedule.alphas_cumprod_prev[t]
            pred_x0 = (
                x_t - sqrt_one_minus_alpha_bar(schedule)[t] * eps_pred
            ) / torch.sqrt(alpha_bar.clamp(min=torch.finfo(x_t.dtype).eps))
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
            coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
            coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean = coef_x0 * pred_x0 + coef_xt * x_t
        else:
            mean = (
                x_t - beta_t / sqrt_one_minus_alpha_bar(schedule)[t] * eps_pred
            ) / torch.sqrt(alpha_t)
        if t == 0:
            return mean
        variance = schedule.posterior_variance[t].clamp(min=0)
        noise = randn(
            x_t.shape,
            device=x_t.device,
            dtype=x_t.dtype,
            generator=generator,
        )
        return mean + torch.sqrt(variance) * noise

    @property
    def name(self) -> str:
        return "DDPM"
