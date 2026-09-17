import torch
from torch import nn

from projects.project1_ddpm.diffusion import p_losses, p_sample, q_sample
from projects.project1_ddpm.schedule import DDPMSchedule


class ZeroNoiseModel(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_q_sample_is_reproducible_with_explicit_noise():
    schedule = DDPMSchedule(T=16)
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 1, 8, 15])
    noise = torch.randn_like(x0)
    first = q_sample(
        x0,
        t,
        schedule.sqrt_alphas_cumprod,
        schedule.sqrt_one_minus_alphas_cumprod,
        noise,
    )
    second = q_sample(
        x0,
        t,
        schedule.sqrt_alphas_cumprod,
        schedule.sqrt_one_minus_alphas_cumprod,
        noise,
    )
    assert torch.equal(first, second)


def test_p_losses_produces_gradients():
    model = nn.Conv2d(1, 1, kernel_size=1)
    schedule = DDPMSchedule(T=16)
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.tensor([2, 11])
    loss = p_losses(model, x0, t, schedule)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.weight.grad is not None


def test_min_snr_loss_produces_finite_gradients():
    model = nn.Conv2d(1, 1, kernel_size=1)
    schedule = DDPMSchedule(T=16, beta_schedule="cosine")
    x0 = torch.randn(4, 1, 8, 8)
    t = torch.tensor([0, 1, 8, 15])
    loss = p_losses(model, x0, t, schedule, loss_weighting="min_snr", min_snr_gamma=5.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.weight.grad is not None


def test_invalid_min_snr_configuration_is_rejected():
    model = ZeroNoiseModel()
    schedule = DDPMSchedule(T=16)
    x0 = torch.randn(2, 1, 8, 8)
    t = torch.tensor([2, 11])
    try:
        p_losses(model, x0, t, schedule, loss_weighting="min_snr", min_snr_gamma=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive min_snr_gamma should be rejected")


def test_last_reverse_step_has_no_random_noise():
    model = ZeroNoiseModel()
    schedule = DDPMSchedule(T=16)
    xt = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    first = p_sample(model, xt, t, schedule)
    second = p_sample(model, xt, t, schedule)
    assert torch.equal(first, second)
