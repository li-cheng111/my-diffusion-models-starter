"""
Project 5 training script. Contains: TODO 19 (action chunk DDPM loss)
"""
import argparse
import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

from dataset import DemoDataset, collect_demos
from model import DiffusionPolicy
from obs_utils import state_dim_for, state_from_batch


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


# -----------------------------------------------------------------------------
# DDPM utility
# -----------------------------------------------------------------------------
class DDPMScheduler:
    def __init__(self, T=100, beta_min=1e-4, beta_max=0.02, device="cpu",
                 schedule="linear"):
        self.T = T
        self.schedule = schedule
        if schedule == "linear":
            betas = torch.linspace(beta_min, beta_max, T, device=device)
        elif schedule == "cosine":
            # Improved DDPM cosine schedule. Unlike the starter linear
            # schedule at T=100, this reaches an almost pure-noise endpoint,
            # matching inference that starts from N(0, I).
            s = 0.008
            steps = torch.linspace(0, T, T + 1, device=device, dtype=torch.float64)
            alpha_bar = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2).square()
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999).float()
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")
        alphas = 1 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_cum = alphas.cumprod(0)
        self.sqrt_ac = self.alpha_cum.sqrt()
        self.sqrt_1mac = (1 - self.alpha_cum).sqrt()

    def add_noise(self, x_0, t, noise):
        """x_t = sqrt(ac_t) * x_0 + sqrt(1 - ac_t) * noise."""
        # x_0 shape (B, H, A), t shape (B,)
        shape = [t.shape[0]] + [1] * (x_0.ndim - 1)
        sa = self.sqrt_ac[t].reshape(shape)
        s1ma = self.sqrt_1mac[t].reshape(shape)
        return sa * x_0 + s1ma * noise


# -----------------------------------------------------------------------------
# TODO 19: Implement action chunk DDPM loss
# -----------------------------------------------------------------------------
def _masked_mse(prediction, target, batch, mask_padding=False):
    error = (prediction - target).square()
    if not mask_padding or "action_mask" not in batch:
        return error.mean()
    mask = batch["action_mask"].to(error.device, dtype=error.dtype).unsqueeze(-1)
    return (error * mask).sum() / (mask.sum() * error.shape[-1]).clamp_min(1.0)


def diffusion_loss(model, batch, scheduler, device, use_vision=True,
                   mask_padding=False):
    """
    Compute DDPM noise prediction loss on action chunks.

    Args:
        model: DiffusionPolicy, callable as model(a_noisy, t, image, state) -> eps_pred
        batch: dict with keys image (B, 3, H, W), state (B, 2), goal (B, 2), action (B, H, A)
        scheduler: DDPMScheduler instance
        device: cuda or cpu
        use_vision: 决定条件用什么 state（见 obs_utils.py）

    Returns: scalar loss

    Steps:
        1) Sample timestep t ~ U(0, T) per-batch
        2) Sample noise eps ~ N(0, I) with shape of action
        3) Add noise to action: a_t = scheduler.add_noise(action, t, eps)
        4) Forward model: eps_pred = model(a_t, t, image, state)
        5) Return MSE(eps_pred, eps)

    注意 t 的取值范围是 [0, T)，别写成 [1, T] —— scheduler 的表是 0-indexed。
    """
    image = batch["image"].to(device)
    # use_vision=False 时 state 是 concat(agent_pos, target_pos)，见 obs_utils
    state = state_from_batch(batch, use_vision).to(device)
    action = batch["action"].to(device)
    B = action.shape[0]

    # ============================================================
    # TODO 19: Implement DDPM loss for action chunk (≈ 5 lines)
    # ============================================================
    t = torch.randint(0, scheduler.T, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(action)
    noisy_action = scheduler.add_noise(action, t, noise)
    eps_pred = model(noisy_action, t, image=image, state=state)
    return _masked_mse(eps_pred, noise, batch, mask_padding)
    # ============================================================
    # END TODO 19
    # ============================================================


def flow_matching_loss(model, batch, device, use_vision=True, mask_padding=False):
    """Conditional Flow Matching loss for the optional FM action head."""
    image = batch["image"].to(device)
    state = state_from_batch(batch, use_vision).to(device)
    action = batch["action"].to(device)
    noise = torch.randn_like(action)
    t = torch.rand(action.shape[0], device=device)
    view = t.view(-1, *([1] * (action.ndim - 1)))
    noisy_action = (1.0 - view) * noise + view * action
    velocity_pred = model(noisy_action, t, image=image, state=state)
    return _masked_mse(velocity_pred, action - noise, batch, mask_padding)


def behavior_cloning_loss(model, batch, device, use_vision=True, mask_padding=False):
    """Deterministic action-chunk baseline with the same condition encoder."""
    image = batch["image"].to(device)
    state = state_from_batch(batch, use_vision).to(device)
    action = batch["action"].to(device)
    zeros = torch.zeros_like(action)
    t = torch.zeros(action.shape[0], device=device, dtype=torch.long)
    action_pred = model(zeros, t, image=image, state=state)
    return _masked_mse(action_pred, action, batch, mask_padding)


# -----------------------------------------------------------------------------
# EMA helper
# -----------------------------------------------------------------------------
@torch.no_grad()
def ema_update(ema_model, model, decay=0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)
    # BatchNorm running statistics and counters are buffers, not parameters.
    # Keeping their initial values made EMA evaluation invalid for ResNet18.
    model_buffers = dict(model.named_buffers())
    for name, buffer_ema in ema_model.named_buffers():
        buffer_ema.copy_(model_buffers[name])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--collect", action="store_true", help="Collect demo data first")
    parser.add_argument("--run-dir", default=None, help="Directory for metrics/checkpoints")
    parser.add_argument("--method", choices=("ddpm", "fm", "bc"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path, latest, or auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    method = args.method or cfg.get("method", "ddpm")
    seed = cfg.get("seed", 0) if args.seed is None else args.seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir or cfg.get("run_dir", "."))
    if args.run_dir:
        cfg["ckpt_dir"] = str(run_dir / "ckpts")
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    status_path = run_dir / "status.json"
    write_json(status_path, {"state": "starting", "method": method, "seed": seed,
                             "max_steps": cfg["max_steps"], "step": 0})

    # Collect demos if needed
    demo_path = cfg["demo_path"]
    if args.collect or not os.path.exists(demo_path):
        collect_demos(
            n_demos=cfg["n_demos"],
            chunk_size=cfg["chunk_size"],
            n_distractors=cfg["n_distractors"],
            save_path=demo_path,
            seed=cfg.get("data_seed", cfg.get("seed", 0)),
            target_choices=cfg.get("target_choices"),
        )

    # Dataset
    ds = DemoDataset(demo_path)
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 2), drop_last=True, pin_memory=True,
    )
    print(f"Demo dataset: {len(ds)} samples")

    # Model
    use_vision = cfg["use_vision"]
    model = DiffusionPolicy(
        horizon=cfg["chunk_size"],
        action_dim=2,
        state_dim=state_dim_for(use_vision),
        image_size=64,
        vision_out_dim=cfg["vision_out_dim"],
        hidden=cfg["hidden"],
        use_vision=use_vision,
        vision_encoder=cfg.get("vision_encoder", "small"),
        vision_pretrained=cfg.get("vision_pretrained", False),
    ).to(device)
    print(f"条件模式: {'image + agent_pos' if use_vision else 'agent_pos + target_pos（无视觉）'}"
          f"  state_dim={state_dim_for(use_vision)}")
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 0.0))
    scheduler = DDPMScheduler(
        T=cfg["diffusion_steps"], device=device,
        schedule=cfg.get("noise_schedule", "linear"),
    )
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    checkpoint = None
    resume_path = args.resume
    if resume_path and resume_path.lower() in {"latest", "auto"}:
        candidates = sorted(Path(cfg["ckpt_dir"]).glob("model_*.pt"))
        resume_path = str(candidates[-1]) if candidates else None
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        ema_model.load_state_dict(checkpoint.get("ema", checkpoint.get("model", checkpoint)))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resumed model from {resume_path}")

    # Training
    model.train()
    step = int(checkpoint.get("step", 0)) if checkpoint else 0
    losses = []
    t_start = time.time()
    write_json(status_path, {"state": "running", "method": method, "seed": seed,
                             "max_steps": cfg["max_steps"], "step": step})
    while step < cfg["max_steps"]:
        for batch in loader:
            mask_padding = cfg.get("mask_padding", False)
            if method == "fm":
                loss = flow_matching_loss(model, batch, device, use_vision, mask_padding)
            elif method == "bc":
                loss = behavior_cloning_loss(model, batch, device, use_vision, mask_padding)
            else:
                loss = diffusion_loss(model, batch, scheduler, device, use_vision,
                                      mask_padding)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema_update(ema_model, model, decay=cfg.get("ema_decay", 0.999))

            losses.append(loss.item())
            step += 1
            if step % cfg["log_every"] == 0:
                avg = sum(losses[-100:]) / min(len(losses), 100)
                print(f"step {step}/{cfg['max_steps']} | loss {loss.item():.4f} "
                      f"(avg100 {avg:.4f}) | {step / (time.time() - t_start):.1f}/s")
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "loss": loss.item(),
                                             "avg100": avg, "method": method,
                                             "elapsed_s": time.time() - t_start},
                                            ensure_ascii=False) + "\n")
                write_json(status_path, {"state": "running", "method": method, "seed": seed,
                                         "max_steps": cfg["max_steps"], "step": step,
                                         "loss": loss.item(), "avg100": avg,
                                         "elapsed_s": time.time() - t_start})
            if step % cfg["save_every"] == 0:
                p = os.path.join(cfg["ckpt_dir"], f"model_{step:06d}.pt")
                torch.save(
                    {"model": model.state_dict(), "ema": ema_model.state_dict(),
                     "optimizer": optimizer.state_dict(), "step": step,
                     "config": cfg, "method": method, "seed": seed},
                    p,
                )
                print(f"Saved {p}")
            if step >= cfg["max_steps"]:
                break

    final = os.path.join(cfg["ckpt_dir"], "model_final.pt")
    torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step,
                "config": cfg, "method": method, "seed": seed}, final)
    write_json(status_path, {"state": "completed", "method": method, "seed": seed,
                             "max_steps": cfg["max_steps"], "step": step,
                             "elapsed_s": time.time() - t_start,
                             "checkpoint": str(Path(final).resolve())})
    print(f"Done. Final ckpt: {final}")


if __name__ == "__main__":
    main()
