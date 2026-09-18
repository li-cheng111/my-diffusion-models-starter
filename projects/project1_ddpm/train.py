"""Config-driven DDPM training with EMA, AMP, logging, and checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchvision.utils as vutils
import yaml
from torch.cuda.amp import GradScaler, autocast

try:  # 支持 ``python -m`` 和 starter 原有的直接脚本入口。
    from .dataset import denormalize, get_dataloader
    from .diffusion import p_losses, p_sample_loop
    from .model import UNet
    from .schedule import DDPMSchedule
except ImportError:  # pragma: no cover - direct script compatibility
    from dataset import denormalize, get_dataloader
    from diffusion import p_losses, p_sample_loop
    from model import UNet
    from schedule import DDPMSchedule


class EMA:
    """Exponential moving average of model parameters and buffers."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_parameter, parameter in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_parameter.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "model": self.ema_model.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "model" in state:
            self.ema_model.load_state_dict(state["model"])
            self.decay = float(state.get("decay", self.decay))
        else:
            self.ema_model.load_state_dict(state)


class EMABank:
    """Maintain several EMA copies in one training run."""

    def __init__(self, model: torch.nn.Module, decays: list[float] | tuple[float, ...]) -> None:
        values = tuple(sorted({float(decay) for decay in decays}))
        if not values or any(not 0.0 < decay < 1.0 for decay in values):
            raise ValueError("ema_decays must contain values strictly between 0 and 1")
        self.decays = values
        self.models = [copy.deepcopy(model) for _ in values]
        for shadow in self.models:
            for parameter in shadow.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for decay, shadow in zip(self.decays, self.models):
            for shadow_parameter, parameter in zip(shadow.parameters(), model.parameters()):
                shadow_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
            for shadow_buffer, buffer in zip(shadow.buffers(), model.buffers()):
                shadow_buffer.copy_(buffer)

    def primary_model(self) -> torch.nn.Module:
        return self.models[-1]

    def model_for_decay(self, decay: float) -> torch.nn.Module:
        target = float(decay)
        for configured, shadow in zip(self.decays, self.models):
            if math.isclose(configured, target, rel_tol=0.0, abs_tol=1e-8):
                return shadow
        available = ", ".join(f"{item:.6g}" for item in self.decays)
        raise ValueError(f"EMA decay {target} is unavailable; choices: {available}")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decays": list(self.decays),
            "models": [shadow.state_dict() for shadow in self.models],
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "models" in state:
            saved_decays = [float(value) for value in state.get("decays", self.decays)]
            if tuple(saved_decays) != self.decays:
                raise ValueError(
                    f"EMA decay mismatch: checkpoint={saved_decays}, config={list(self.decays)}"
                )
            for shadow, shadow_state in zip(self.models, state["models"]):
                shadow.load_state_dict(shadow_state)
            return
        # Backward compatibility: initialize every requested decay from a
        # legacy single EMA state when architectures are otherwise compatible.
        legacy = state.get("model", state)
        for shadow in self.models:
            shadow.load_state_dict(legacy)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_loss_history(history: list[Dict[str, float]], output_dir: Path) -> None:
    csv_path = output_dir / "loss_history.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "attempt_step",
                "epoch",
                "loss",
                "lr",
                "grad_norm",
                "optimizer_step_skipped",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    # Plotting happens only when the training script is actually run; this
    # function is intentionally not invoked during repository preparation.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    plt.figure(figsize=(8, 4))
    plt.plot([row["step"] for row in history], [row["loss"] for row in history])
    plt.xlabel("Training step")
    plt.ylabel("Noise prediction MSE")
    plt.title("DDPM training loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    ema: Optional[Any],
    scaler: GradScaler,
    cfg: Dict[str, Any],
    epoch: int,
    global_step: int,
    resume_schedule_origin_step: Optional[int] = None,
) -> None:
    state: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    if lr_scheduler is not None:
        state["lr_scheduler"] = lr_scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    if scaler.is_enabled():
        state["scaler"] = scaler.state_dict()
    if resume_schedule_origin_step is not None:
        state["resume_schedule_origin_step"] = resume_schedule_origin_step
    torch.save(state, path)


def _lr_for_update(cfg: Dict[str, Any], step: int) -> float:
    """Return the learning rate used for a zero-based successful update."""

    optimizer_cfg = cfg["optimizer"]
    base_lr = float(optimizer_cfg["lr"])
    warmup_steps = max(0, int(optimizer_cfg.get("warmup_steps", 0)))
    schedule_name = str(optimizer_cfg.get("lr_schedule", "warmup_constant")).lower()
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if schedule_name in {"constant", "warmup_constant"}:
        return base_lr
    if schedule_name == "warmup_late_cosine":
        total_steps = int(cfg["training"].get("max_steps", 0))
        late_decay_start = int(
            optimizer_cfg.get("late_decay_start_step", round(total_steps * 0.75))
        )
        if total_steps <= 0 or not 0 <= late_decay_start < total_steps:
            raise ValueError(
                "optimizer.late_decay_start_step must be in [0, training.max_steps)"
            )
        min_lr = float(optimizer_cfg.get("min_lr", 0.0))
        if not 0.0 <= min_lr <= base_lr:
            raise ValueError("optimizer.min_lr must satisfy 0 <= min_lr <= optimizer.lr")
        if step < late_decay_start:
            return base_lr
        decay_steps = max(total_steps - late_decay_start - 1, 1)
        progress = min(max((step - late_decay_start) / decay_steps, 0.0), 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if schedule_name != "warmup_cosine":
        raise ValueError(
            "optimizer.lr_schedule must be 'constant', 'warmup_constant', "
            "'warmup_cosine', or 'warmup_late_cosine'"
        )
    total_steps = int(cfg["training"].get("max_steps", 0))
    if total_steps <= warmup_steps:
        raise ValueError("training.max_steps must be greater than warmup_steps for warmup_cosine")
    min_lr = float(optimizer_cfg.get("min_lr", 0.0))
    if not 0.0 <= min_lr <= base_lr:
        raise ValueError("optimizer.min_lr must satisfy 0 <= min_lr <= optimizer.lr")
    progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps - 1, 1), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(cfg: Dict[str, Any], resume: Optional[str] = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "ckpt").mkdir(exist_ok=True)

    _set_seed(int(cfg.get("seed", 42)))
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    loader = get_dataloader(
        name=cfg["dataset"]["name"],
        batch_size=cfg["dataset"]["batch_size"],
        root=cfg["dataset"].get("root", "./data"),
        image_size=cfg["dataset"].get("image_size"),
        num_workers=cfg["dataset"].get("num_workers", 4),
    )
    model = UNet(**cfg["model"]).to(device)
    schedule = DDPMSchedule(**cfg["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["optimizer"]["lr"]),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.0)),
        betas=(0.9, 0.999),
    )

    resume_lr_cfg = cfg.get("resume_lr_schedule")
    explicit_lr_schedule = resume_lr_cfg is not None or "lr_schedule" in cfg["optimizer"]
    warmup_steps = int(cfg["optimizer"].get("warmup_steps", 0))
    lr_scheduler = None
    if warmup_steps > 0 and not explicit_lr_schedule:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min((step + 1) / warmup_steps, 1.0)
        )

    ema_decays_cfg = cfg.get("ema_decays")
    if ema_decays_cfg is not None:
        ema = EMABank(model, [float(value) for value in ema_decays_cfg])
    else:
        ema_decay = float(cfg.get("ema_decay", 0.0))
        ema = EMA(model, ema_decay) if ema_decay > 0 else None
    precision = cfg.get("mixed_precision", "no")
    use_amp = precision in {"fp16", "bf16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_epoch = 0
    global_step = 0
    resume_schedule_origin_step: Optional[int] = None
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if ema is not None and "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        if scaler.is_enabled() and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        if resume_lr_cfg is not None:
            resume_schedule_origin_step = int(
                checkpoint.get("resume_schedule_origin_step", global_step)
            )

    if resume_lr_cfg is not None:
        if resume_schedule_origin_step is None:
            resume_schedule_origin_step = global_step
        start_lr = float(resume_lr_cfg.get("start_lr", cfg["optimizer"]["lr"]))
        end_lr = float(resume_lr_cfg.get("end_lr", start_lr))
        decay_steps = max(1, int(resume_lr_cfg.get("steps", 1)))
        for group in optimizer.param_groups:
            group["lr"] = start_lr

    wandb_enabled = bool(cfg.get("wandb", {}).get("enabled", False))
    wandb = None
    if wandb_enabled:
        import wandb as wandb_module

        wandb = wandb_module
        wandb.init(
            project=cfg["wandb"].get("project", "ddpm-course"),
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )

    training_cfg = cfg["training"]
    log_every = int(training_cfg.get("log_every", 50))
    sample_every = int(training_cfg.get("sample_every", 1000))
    ckpt_every = int(training_cfg.get("ckpt_every", 5000))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    max_steps = training_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    grad_accum_steps = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    loss_weighting = str(training_cfg.get("loss_weighting", "uniform"))
    min_snr_gamma = float(training_cfg.get("min_snr_gamma", 5.0))
    history: list[Dict[str, float]] = []
    start_time = time.time()
    last_epoch = start_epoch - 1
    micro_step = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(training_cfg["num_epochs"])):
        last_epoch = epoch
        for x0 in loader:
            if max_steps is not None and global_step >= max_steps:
                break
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, schedule.T, (x0.shape[0],), device=device)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                loss = p_losses(
                    model,
                    x0,
                    t,
                    schedule,
                    loss_weighting=loss_weighting,
                    min_snr_gamma=min_snr_gamma,
                ) / grad_accum_steps

            micro_step += 1

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if micro_step % grad_accum_steps != 0:
                continue

            if resume_lr_cfg is not None:
                progress = min(
                    max((global_step - resume_schedule_origin_step) / decay_steps, 0.0),
                    1.0,
                )
                lr_for_update = end_lr + (start_lr - end_lr) * 0.5 * (
                    1.0 + math.cos(progress * math.pi)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr_for_update
            elif explicit_lr_schedule:
                lr_for_update = _lr_for_update(cfg, global_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr_for_update

            skipped = False
            grad_norm = 0.0
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
                optimizer.step()
            if skipped:
                # A GradScaler overflow did not produce an optimizer update;
                # do not advance the schedule, EMA, or effective step count.
                history.append(
                    {
                        "step": float(global_step),
                        "attempt_step": float(micro_step),
                        "epoch": float(epoch),
                        "loss": float((loss * grad_accum_steps).detach().cpu()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "grad_norm": grad_norm,
                        "optimizer_step_skipped": 1.0,
                    }
                )
                print(
                    f"[epoch {epoch:03d} attempt {micro_step:06d}] "
                    f"optimizer step skipped; scaler={scaler.get_scale():.1f}",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            if lr_scheduler is not None:
                lr_scheduler.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            lr_now = float(optimizer.param_groups[0]["lr"])
            row = {
                "step": float(global_step),
                "attempt_step": float(micro_step),
                "epoch": float(epoch),
                "loss": float((loss * grad_accum_steps).detach().cpu()),
                "lr": lr_now,
                "grad_norm": grad_norm,
                "optimizer_step_skipped": float(skipped),
            }
            history.append(row)
            if global_step % log_every == 0:
                print(
                    f"[epoch {epoch:03d} step {global_step:06d}] "
                    f"loss={row['loss']:.5f} lr={lr_now:.3e}"
                )
                if wandb is not None:
                    wandb.log({"loss": row["loss"], "lr": lr_now}, step=global_step)

            if sample_every > 0 and global_step % sample_every == 0:
                if isinstance(ema, EMABank):
                    sample_model = ema.primary_model()
                else:
                    sample_model = ema.ema_model if ema is not None else model
                was_training = sample_model.training
                sample_model.eval()
                samples = p_sample_loop(
                    sample_model,
                    (
                        16,
                        cfg["model"]["in_channels"],
                        cfg["model"]["image_size"],
                        cfg["model"]["image_size"],
                    ),
                    schedule,
                    device=device,
                )
                vutils.save_image(
                    denormalize(samples),
                    output_dir / "samples" / f"step_{global_step:06d}.png",
                    nrow=4,
                )
                if was_training:
                    sample_model.train()

            if ckpt_every > 0 and global_step % ckpt_every == 0:
                _save_checkpoint(
                    output_dir / "ckpt" / f"step_{global_step:06d}.pt",
                    model,
                    optimizer,
                    lr_scheduler,
                    ema,
                    scaler,
                    cfg,
                    epoch,
                    global_step,
                    resume_schedule_origin_step,
                )

        if max_steps is not None and global_step >= max_steps:
            break

    _save_checkpoint(
        output_dir / "ckpt" / "final.pt",
        model,
        optimizer,
        lr_scheduler,
        ema,
        scaler,
        cfg,
        last_epoch,
        global_step,
        resume_schedule_origin_step,
    )
    _save_loss_history(history, output_dir)
    if wandb is not None:
        wandb.finish()
    print(f"Training complete in {(time.time() - start_time) / 60.0:.1f} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a from-scratch DDPM")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override training.num_epochs without editing the YAML config.",
    )
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_epochs is not None:
        if args.num_epochs <= 0:
            parser.error("--num_epochs must be positive")
        cfg["training"]["num_epochs"] = args.num_epochs
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
