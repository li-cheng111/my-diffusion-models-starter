"""Common-noise NFE/FID evaluation for Project 4 fair-v1 checkpoints."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import torch
import torchvision
from torchvision import datasets, transforms
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from projects.project1_ddpm.schedule import DDPMSchedule
from projects.project1_ddpm.diffusion import p_sample
from projects.project4_flow_matching.fair_v1.models import ConditionalProject1UNet, build_model, predict
from projects.project4_flow_matching.fair_v1.train import sha256_file


def load_weights(checkpoint_path, device, ema_decay=0.9999):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    model = build_model(cfg["model"]).to(device)
    if "ema_state" in checkpoint:
        state = checkpoint["ema_state"]
        weights_name = f"EMA_{checkpoint.get('ema_decay', ema_decay)}"
    else:
        ema = checkpoint.get("ema")
        if isinstance(ema, dict) and "models" in ema:
            matches = [i for i, value in enumerate(ema["decays"]) if abs(float(value) - ema_decay) < 1e-8]
            if not matches:
                raise ValueError(f"EMA {ema_decay} absent from {ema['decays']}")
            index = matches[0]
            state, weights_name = ema["models"][index], f"EMA_{ema['decays'][index]}"
        else:
            state, weights_name = (ema if isinstance(ema, dict) else checkpoint["model"]), "EMA"
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, checkpoint, weights_name


def _model_uses_labels(model):
    return isinstance(model, ConditionalProject1UNet) or model.__class__.__name__ == "SimpleDiT"


def _velocity(model, x, t, labels, mode, scale, num_classes):
    if not _model_uses_labels(model):
        return predict(model, x, t)
    null = torch.full_like(labels, num_classes)
    if mode == "unconditional":
        return predict(model, x, t, null)
    if mode == "conditional":
        return predict(model, x, t, labels)
    if mode == "cfg":
        v_cond = predict(model, x, t, labels)
        v_uncond = predict(model, x, t, null)
        return v_uncond + float(scale) * (v_cond - v_uncond)
    raise ValueError(f"unknown condition mode {mode}")


def integrate_velocity(initial_noise, velocity_fn, nfe, solver):
    """Integrate a normalized-time ODE; nfe counts velocity_fn calls exactly."""
    if nfe <= 0:
        raise ValueError("NFE budget must be positive")
    if solver == "euler":
        steps = nfe
    elif solver == "heun":
        if nfe % 2:
            raise ValueError("Heun requires an even NFE budget")
        steps = nfe // 2
    else:
        raise ValueError(f"unknown ODE solver {solver}")
    x = initial_noise
    times = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=torch.float32)
    calls = 0
    for index in range(steps):
        t0, t1 = times[index], times[index + 1]
        dt = t1 - t0
        v0 = velocity_fn(x, t0.expand(x.shape[0]))
        calls += 1
        if solver == "euler":
            x = x + dt * v0
        else:
            v1 = velocity_fn(x + dt * v0, t1.expand(x.shape[0]))
            calls += 1
            x = x + 0.5 * dt * (v0 + v1)
    if calls != nfe:
        raise RuntimeError(f"solver used {calls} evaluations for NFE={nfe}")
    return x


@torch.no_grad()
def sample_fm(model, initial_noise, labels, num_classes, nfe, solver, mode, scale):
    velocity_fn = lambda x, t: _velocity(model, x, t, labels, mode, scale, num_classes)
    return integrate_velocity(initial_noise, velocity_fn, nfe, solver)


@torch.no_grad()
def sample_ddim(model, initial_noise, schedule, nfe, clip_denoised=True):
    if not 1 <= nfe <= schedule.T:
        raise ValueError(f"DDIM NFE must be in [1,{schedule.T}]")
    times = torch.linspace(schedule.T - 1, 0, nfe, device=initial_noise.device).round().long()
    if torch.unique_consecutive(times).numel() != nfe:
        raise ValueError("requested DDIM NFE generated duplicate timesteps")
    x = initial_noise
    for index, timestep in enumerate(times):
        t = torch.full((x.shape[0],), int(timestep.item()), device=x.device, dtype=torch.long)
        t_norm = t.float() / float(schedule.T - 1)
        eps = predict(model, x, t_norm)
        alpha_bar = schedule.alphas_cumprod[timestep].to(dtype=x.dtype)
        pred_x0 = (x - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp_min(1e-12)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)
        if index + 1 < nfe:
            previous = times[index + 1]
            alpha_previous = schedule.alphas_cumprod[previous].to(dtype=x.dtype)
        else:
            alpha_previous = torch.ones((), device=x.device, dtype=x.dtype)
        x = alpha_previous.sqrt() * pred_x0 + (1.0 - alpha_previous).clamp_min(0).sqrt() * eps
    return x


@torch.no_grad()
def sample_ancestral(model, initial_noise, schedule, seed):
    device = initial_noise.device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    x = initial_noise
    for timestep in reversed(range(schedule.T)):
        t = torch.full((x.shape[0],), timestep, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedule, clip_denoised=True)
    return x


def load_noise_bank(path, count, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bank = torch.load(path, map_location="cpu", weights_only=True)
        if bank.shape != (count, 3, 32, 32):
            raise ValueError(f"noise bank shape {tuple(bank.shape)} != {(count, 3, 32, 32)}")
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bank = torch.randn((count, 3, 32, 32), generator=generator, dtype=torch.float32)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(bank, temporary)
        temporary.replace(path)
    return bank, sha256_file(path)


def load_real(data_root, count):
    dataset = datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.ToTensor())
    if count > len(dataset):
        raise ValueError(f"requested {count} real images but train split has {len(dataset)}")
    images = torch.stack([dataset[i][0] for i in range(count)])
    return (images * 255.0).round().to(torch.uint8)


def fid_score(fake, real, batch_size, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for start in range(0, len(real), batch_size):
        metric.update(real[start:start + batch_size].to(device), real=True)
    for start in range(0, len(fake), batch_size):
        metric.update(fake[start:start + batch_size].to(device), real=False)
    return float(metric.compute().item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", choices=["auto", "fm", "ddpm"], default="auto")
    parser.add_argument("--solver", choices=["auto", "euler", "heun", "ddim", "ancestral"], default="auto")
    parser.add_argument("--condition", choices=["auto", "unconditional", "conditional", "cfg"], default="auto")
    parser.add_argument("--cfg", type=float, nargs="+", default=[1.0])
    parser.add_argument("--nfe", type=int, nargs="+", default=[4, 8, 16, 32, 50])
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--ancestral-seed", type=int, default=3001)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--noise-bank", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-dir", default=None)
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint, weights_name = load_weights(args.model, device)
    method = checkpoint.get("method", cfg.get("method")) if args.method == "auto" else args.method
    if method not in {"fm", "ddpm"}:
        raise ValueError(f"unsupported checkpoint method {method}")
    solver = args.solver if args.solver != "auto" else ("euler" if method == "fm" else "ddim")
    mode = args.condition
    if mode == "auto":
        mode = "conditional" if isinstance(model, ConditionalProject1UNet) else "unconditional"
    num_classes = int(cfg["model"].get("num_classes", 10))
    schedule = DDPMSchedule(**cfg.get("diffusion", {})).to(device) if method == "ddpm" else None
    bank_path = args.noise_bank or f"results/project4_fair_v1/noise/noise_seed{args.seed}_n{args.num_samples}.pt"
    noise_bank, noise_sha = load_noise_bank(bank_path, args.num_samples, args.seed)
    real = load_real(args.data_root, args.num_samples)
    result_rows = []
    scales = args.cfg if mode == "cfg" else [1.0]
    samples_dir = Path(args.samples_dir) if args.samples_dir else None
    if samples_dir:
        samples_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    requests = [1000] if solver == "ancestral" else args.nfe
    for scale in scales:
        for nfe_budget in requests:
            if solver == "ancestral":
                integration_steps, effective_nfe = schedule.T, schedule.T
            elif method == "ddpm":
                if solver != "ddim":
                    raise ValueError("DDPM supports only ddim or ancestral sampling")
                integration_steps, effective_nfe = nfe_budget, nfe_budget
            else:
                integration_steps = nfe_budget if solver == "euler" else nfe_budget // 2
                effective_nfe = nfe_budget * (2 if mode == "cfg" else 1)
            print(f"eval method={method} solver={solver} requested_nfe={nfe_budget} cfg={scale} mode={mode}", flush=True)
            generated = []
            start = time.perf_counter()
            for begin in range(0, args.num_samples, args.batch_size):
                end = min(begin + args.batch_size, args.num_samples)
                initial_noise = noise_bank[begin:end].to(device)
                labels = (torch.arange(begin, end, device=device) % num_classes).long()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    if solver == "ancestral":
                        images = sample_ancestral(model, initial_noise, schedule, args.ancestral_seed + begin)
                    elif method == "ddpm":
                        images = sample_ddim(model, initial_noise, schedule, nfe_budget, clip_denoised=not args.no_clip)
                    else:
                        images = sample_fm(model, initial_noise, labels, num_classes, nfe_budget, solver, mode, scale)
                generated.append(((images.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).cpu())
                if end == args.num_samples or end % (args.batch_size * 10) == 0:
                    print(f"  generated {end}/{args.num_samples}", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_seconds = time.perf_counter() - start
            fake = torch.cat(generated)
            fid_start = time.perf_counter()
            score = fid_score(fake, real, args.batch_size, device)
            fid_seconds = time.perf_counter() - fid_start
            row = {
                "method": method, "solver": solver, "nfe_per_branch": nfe_budget,
                "effective_nfe_per_sample": effective_nfe, "integration_steps": integration_steps,
                "condition_mode": mode, "cfg_scale": scale if mode == "cfg" else None,
                "num_samples": args.num_samples, "generation_seed": args.seed,
                "fid": score, "sample_seconds": sample_seconds, "fid_seconds": fid_seconds,
                "samples_per_second": args.num_samples / max(sample_seconds, 1e-9),
                "clip_denoised": (not args.no_clip) if method == "ddpm" else None,
            }
            result_rows.append(row)
            print(f"  FID={score:.4f}; sampling={sample_seconds/60:.2f} min; {row['samples_per_second']:.1f} images/s", flush=True)
            if samples_dir:
                from torchvision.utils import make_grid, save_image
                grid = make_grid(fake[:64].float() / 255.0, nrow=8)
                save_image(grid, samples_dir / f"{method}_{solver}_nfe{nfe_budget}_{mode}_cfg{scale:g}.png")

    import subprocess
    metadata = {
        "protocol": "TorchMetrics FID-2048; CIFAR-10 train indices [0:N]; RGB uint8; common CPU Gaussian bank",
        "checkpoint": str(Path(args.model).resolve()), "checkpoint_sha256": sha256_file(args.model),
        "checkpoint_step": checkpoint.get("step"), "weights": weights_name,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "python": platform.python_version(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "data_split": "CIFAR-10 train", "real_indices": f"0:{args.num_samples}",
        "num_samples": args.num_samples, "batch_size": args.batch_size,
        "generation_seed": args.seed, "noise_bank": str(Path(bank_path).resolve()),
        "noise_bank_sha256": noise_sha, "results": result_rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {destination}", flush=True)

if __name__ == "__main__":
    main()
