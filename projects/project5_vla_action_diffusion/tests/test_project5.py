import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import DemoDataset, collect_demos
from eval import ddim_sample_action, evaluate, make_ddim_timesteps
from env import Reach2DEnv
from model import DiffusionPolicy, ResNetVisionEncoder, VisionEncoder
from train import (DDPMScheduler, _masked_mse, behavior_cloning_loss,
                   diffusion_loss, ema_update, flow_matching_loss)


def batch(horizon=8):
    return {"image": torch.randn(2, 3, 64, 64), "state": torch.randn(2, 2),
            "goal": torch.randn(2, 2), "action": torch.randn(2, horizon, 2),
            "action_mask": torch.ones(2, horizon)}


def test_vision_encoder_shape_and_budget():
    model = VisionEncoder(out_dim=128)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 128)
    assert sum(p.numel() for p in model.parameters()) < 1_000_000


def test_losses_are_scalar_and_differentiable():
    model = DiffusionPolicy(horizon=8, state_dim=2, use_vision=True)
    scheduler = DDPMScheduler(T=10)
    b = batch()
    for loss in (diffusion_loss(model, b, scheduler, "cpu", True),
                 flow_matching_loss(model, b, "cpu", True),
                 behavior_cloning_loss(model, b, "cpu", True)):
        assert loss.ndim == 0 and torch.isfinite(loss)
        model.zero_grad()
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())


def test_continuous_fm_time_changes_conditioning():
    model = DiffusionPolicy(horizon=4, state_dim=2, use_vision=False)
    action = torch.zeros(1, 4, 2)
    state = torch.zeros(1, 2)
    out0 = model(action, torch.tensor([0.0]), state=state)
    out1 = model(action, torch.tensor([1.0]), state=state)
    assert not torch.allclose(out0, out1)


def test_scheduler_shapes():
    scheduler = DDPMScheduler(T=10)
    x = torch.randn(3, 8, 2)
    t = torch.tensor([0, 4, 9])
    out = scheduler.add_noise(x, t, torch.zeros_like(x))
    assert out.shape == x.shape
    assert torch.allclose(out[0], x[0] * scheduler.sqrt_ac[0])


def test_cosine_schedule_and_ddim_endpoints():
    scheduler = DDPMScheduler(T=100, schedule="cosine")
    timesteps = make_ddim_timesteps(100, 20)
    assert timesteps[0] == 99 and timesteps[-1] == 0
    assert all(a > b for a, b in zip(timesteps, timesteps[1:]))
    assert scheduler.alpha_cum[-1] < 1e-5


def test_ddim_sampling_is_reproducible_with_generator():
    class ZeroModel(nn.Module):
        def forward(self, action, t, image=None, state=None):
            return torch.zeros_like(action)

    scheduler = DDPMScheduler(T=20, schedule="cosine")
    image = torch.zeros(1, 3, 64, 64)
    state = torch.zeros(1, 2)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a1 = ddim_sample_action(ZeroModel(), scheduler, image, state, horizon=4,
                            n_steps=5, device="cpu", generator=g1)
    a2 = ddim_sample_action(ZeroModel(), scheduler, image, state, horizon=4,
                            n_steps=5, device="cpu", generator=g2)
    assert torch.equal(a1, a2)


def test_masked_loss_ignores_padding():
    target = torch.zeros(1, 4, 2)
    prediction = target.clone()
    prediction[:, 2:] = 100.0
    b = {"action_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]])}
    assert _masked_mse(prediction, target, b, mask_padding=True).item() == 0.0
    assert _masked_mse(prediction, target, b, mask_padding=False).item() > 0.0


def test_collected_demo_contains_true_padding_mask(tmp_path):
    path = tmp_path / "demo.pkl"
    collect_demos(n_demos=1, chunk_size=8, n_distractors=1,
                  save_path=str(path), seed=5)
    dataset = DemoDataset(path)
    masks = [dataset[i]["action_mask"] for i in range(len(dataset))]
    assert any((mask == 0).any() for mask in masks)
    assert all(mask.shape == (8,) for mask in masks)


def test_ema_copies_batchnorm_buffers():
    model = nn.BatchNorm1d(3)
    ema = nn.BatchNorm1d(3)
    model.running_mean.fill_(2.0)
    model.running_var.fill_(3.0)
    model.num_batches_tracked.fill_(7)
    ema_update(ema, model, decay=0.9)
    assert torch.equal(ema.running_mean, model.running_mean)
    assert torch.equal(ema.running_var, model.running_var)
    assert torch.equal(ema.num_batches_tracked, model.num_batches_tracked)


def test_closed_loop_returns_complete_metrics():
    cfg = {"n_distractors": 1, "use_vision": False, "chunk_size": 8,
           "diffusion_steps": 10}
    model = DiffusionPolicy(horizon=8, state_dim=4, use_vision=False)
    result = evaluate(model, DDPMScheduler(T=10), cfg, "cpu", n_episodes=2,
                      exec_steps=4, chunk_size=8, n_sample_steps=2,
                      seed_start=10000, method="bc")
    assert result["episodes"] == 2
    assert 0 <= result["success_rate"] <= 1
    assert result["successes"] + result["collisions"] + result["timeouts"] == 2


def test_diffusion_evaluation_is_repeatable():
    cfg = {"n_distractors": 1, "use_vision": False, "chunk_size": 4,
           "diffusion_steps": 10, "noise_schedule": "cosine"}
    model = DiffusionPolicy(horizon=4, state_dim=4, use_vision=False, hidden=32)
    scheduler = DDPMScheduler(T=10, schedule="cosine")
    kwargs = dict(n_episodes=2, exec_steps=2, chunk_size=4,
                  n_sample_steps=2, seed_start=12000, method="ddpm")
    first = evaluate(model, scheduler, cfg, "cpu", **kwargs)
    second = evaluate(model, scheduler, cfg, "cpu", **kwargs)
    assert first == second


def test_bonus_target_modes_and_moving_distractors():
    choices = [[-0.65, 0.55], [0.65, 0.55]]
    env = Reach2DEnv(n_distractors=1, target_choices=choices,
                     moving_distractors=True, seed=7)
    assert any(np.allclose(env.target_pos, choice) for choice in choices)
    before = env.distractors.copy()
    env.step([0.0, 0.0])
    assert not np.allclose(before, env.distractors)


def test_resnet_encoder_offline_shape():
    encoder = ResNetVisionEncoder(out_dim=32, pretrained=False)
    assert encoder(torch.randn(2, 3, 64, 64)).shape == (2, 32)
