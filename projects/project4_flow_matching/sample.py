"""ODE samplers for Project 4 flow-matching checkpoints."""
import argparse
import os

import torch
import torchvision

from train import build_model


def _guided_velocity(model, x, t, labels, num_classes, condition_mode, cfg_scale):
    if condition_mode == "unconditional":
        y = torch.full_like(labels, num_classes)
        return model(x, t, y)
    if condition_mode == "conditional":
        return model(x, t, labels)
    y_null = torch.full_like(labels, num_classes)
    x_in = torch.cat((x, x), dim=0)
    t_in = torch.cat((t, t), dim=0)
    y_in = torch.cat((y_null, labels), dim=0)
    v_uncond, v_cond = model(x_in, t_in, y_in).chunk(2)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_ode(model, initial_noise, labels, num_classes, n_steps, solver="euler",
               condition_mode="unconditional", cfg_scale=1.0):
    """Integrate from t=0 to 1. Heun uses two model evaluations per step."""
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be 'euler' or 'heun'")
    if condition_mode not in ("unconditional", "conditional", "cfg"):
        raise ValueError("condition_mode must be unconditional, conditional, or cfg")
    x = initial_noise
    batch = x.shape[0]
    labels = labels.to(device=x.device, dtype=torch.long)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device, dtype=torch.float32)
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        v0 = _guided_velocity(model, x, t0.expand(batch), labels, num_classes,
                              condition_mode, cfg_scale)
        if solver == "euler":
            x = x + dt * v0
        else:
            v1 = _guided_velocity(model, x + dt * v0, t1.expand(batch), labels,
                                  num_classes, condition_mode, cfg_scale)
            x = x + 0.5 * dt * (v0 + v1)
    return x


def _sample(model, n_samples, n_steps, num_classes, class_label, image_size, device,
            solver="euler", condition_mode="conditional", cfg_scale=1.0,
            initial_noise=None):
    if n_samples <= 0 or n_steps <= 0:
        raise ValueError("n_samples and n_steps must be positive")
    if initial_noise is None:
        initial_noise = torch.randn(n_samples, 3, image_size, image_size, device=device)
    labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    return sample_ode(model, initial_noise, labels, num_classes, n_steps, solver,
                      condition_mode, cfg_scale)


@torch.no_grad()
def euler_sample(model, n_samples, n_steps, num_classes, class_label, image_size=32, device="cuda"):
    return _sample(model, n_samples, n_steps, num_classes, class_label, image_size,
                   device, "euler", "conditional")


@torch.no_grad()
def heun_sample(model, n_samples, n_steps, num_classes, class_label, image_size=32, device="cuda"):
    return _sample(model, n_samples, n_steps, num_classes, class_label, image_size,
                   device, "heun", "conditional")


@torch.no_grad()
def euler_sample_cfg(model, n_samples, n_steps, num_classes, class_label, cfg_scale,
                     image_size=32, device="cuda"):
    return _sample(model, n_samples, n_steps, num_classes, class_label, image_size,
                   device, "euler", "cfg", cfg_scale)


def select_ema_state(ckpt, ema_decay=None):
    ema = ckpt.get("ema")
    if ema is None:
        return ckpt["model"], "model"
    if isinstance(ema, dict) and "models" in ema:
        decays, models = ema["decays"], ema["models"]
        if ema_decay is None:
            index = max(range(len(decays)), key=lambda i: decays[i])
        else:
            matches = [i for i, d in enumerate(decays) if abs(float(d) - float(ema_decay)) < 1e-8]
            if not matches:
                raise ValueError(f"EMA decay {ema_decay} unavailable; choices: {decays}")
            index = matches[0]
        return models[index], f"EMA_{decays[index]}"
    return ema, "EMA_legacy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="samples/")
    parser.add_argument("--nfe", type=int, nargs="+", default=[4, 8, 16, 32, 50])
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--condition", choices=["unconditional", "conditional", "cfg"], default="conditional")
    parser.add_argument("--cfg", type=float, nargs="+", default=[1.0, 3.0, 7.5])
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", ckpt.get("config"))
    model = build_model(cfg).to(device)
    state, weights = (ckpt["model"], "model") if args.no_ema else select_ema_state(ckpt, args.ema_decay)
    model.load_state_dict(state)
    model.eval()
    num_classes = cfg["model"]["num_classes"]
    os.makedirs(args.output, exist_ok=True)

    scales = args.cfg if args.condition == "cfg" else [1.0]
    for nfe in args.nfe:
        for cfg_scale in scales:
            torch.manual_seed(args.seed)
            noise = torch.randn(args.n_samples, 3, 32, 32, device=device)
            labels = torch.full((args.n_samples,), args.class_label, device=device, dtype=torch.long)
            samples = sample_ode(model, noise, labels, num_classes, nfe, args.solver,
                                 args.condition, cfg_scale)
            grid = torchvision.utils.make_grid((samples.clamp(-1, 1) + 1) / 2, nrow=8)
            fp = os.path.join(args.output, f"{args.solver}_nfe{nfe}_{args.condition}_cfg{cfg_scale}_{weights}.png")
            torchvision.utils.save_image(grid, fp)
            print(f"saved {fp}")


if __name__ == "__main__":
    main()
