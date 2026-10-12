"""Generate immutable, hash-verified pairs for one Reflow round."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from projects.project4_flow_matching.fair_v1.evaluate import load_weights, sample_fm
from projects.project4_flow_matching.fair_v1.train import sha256_file


def _sample(model, noise, nfe, device):
    labels = torch.zeros(len(noise), device=device, dtype=torch.long)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return sample_fm(model, noise.to(device), labels, 10, nfe, "heun", "unconditional", 1.0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", default="results/project4_fair_v1/reflow_pairs")
    parser.add_argument("--num-pairs", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--teacher-nfe", type=int, default=200)
    parser.add_argument("--diagnostic-nfe", type=int, default=400)
    parser.add_argument("--max-relative-rms", type=float, default=0.02)
    args = parser.parse_args()
    if args.teacher_nfe % 2 or args.diagnostic_nfe % 2:
        raise ValueError("Heun NFE must be even")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed pair set: {output / 'manifest.json'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint, weights_name = load_weights(args.teacher, device)
    if cfg["model"]["type"] != "p1_unet":
        raise ValueError("Reflow teacher must be the unconditional Project 1 U-Net F-U")
    diagnostic_generator = torch.Generator(device="cpu").manual_seed(args.seed + 910_001)
    diagnostic_noise = torch.randn((256, 3, 32, 32), generator=diagnostic_generator)
    teacher_coarse = _sample(model, diagnostic_noise, args.teacher_nfe, device)
    teacher_fine = _sample(model, diagnostic_noise, args.diagnostic_nfe, device)
    relative_rms = float((teacher_coarse - teacher_fine).square().mean().sqrt() /
                         teacher_fine.square().mean().sqrt().clamp_min(1e-12))
    diagnostic = {
        "teacher": str(Path(args.teacher).resolve()), "teacher_sha256": sha256_file(args.teacher),
        "weights": weights_name, "teacher_nfe": args.teacher_nfe,
        "diagnostic_nfe": args.diagnostic_nfe, "num_diagnostic_noise": len(diagnostic_noise),
        "relative_rms_difference": relative_rms,
        "max_relative_rms": args.max_relative_rms,
        "stable_at_declared_threshold": relative_rms <= args.max_relative_rms,
    }
    (output / "teacher_stability.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(f"teacher stability: relative_rms={relative_rms:.6g} threshold={args.max_relative_rms:.6g}", flush=True)
    if relative_rms > args.max_relative_rms:
        raise SystemExit("teacher integration failed the frozen stability threshold; revise NFE, record the change, and use a new output directory")

    teacher_sha = sha256_file(args.teacher)
    shard_records = []
    started = time.perf_counter()
    for start in range(0, args.num_pairs, args.shard_size):
        count = min(args.shard_size, args.num_pairs - start)
        shard_index = start // args.shard_size
        generator = torch.Generator(device="cpu").manual_seed(args.seed + start * 1009)
        noise = torch.randn((count, 3, 32, 32), generator=generator, dtype=torch.float32)
        endpoints = []
        for begin in range(0, count, args.batch_size):
            endpoints.append(_sample(model, noise[begin:begin + args.batch_size], args.teacher_nfe, device).cpu())
        endpoint = torch.cat(endpoints)
        filename = f"pairs_{shard_index:03d}.pt"
        temporary = output / (filename + ".tmp")
        torch.save({"noise": noise.contiguous(), "endpoint": endpoint.contiguous()}, temporary)
        temporary.replace(output / filename)
        record = {"file": filename, "sha256": sha256_file(output / filename),
                  "start": start, "count": count}
        shard_records.append(record)
        print(f"saved {filename} pairs={count} elapsed_min={(time.perf_counter()-started)/60:.1f}", flush=True)

    manifest = {
        "format": "project4-fair-v1-reflow-pairs", "num_pairs": args.num_pairs,
        "shape_per_image": [3, 32, 32], "dtype": "float32",
        "pair_definition": "epsilon is the initial Gaussian noise; endpoint is the fixed teacher ODE output",
        "teacher": str(Path(args.teacher).resolve()), "teacher_sha256": teacher_sha,
        "teacher_weights": weights_name, "solver": "Heun",
        "teacher_nfe": args.teacher_nfe, "stability_check": diagnostic,
        "seed": args.seed, "shard_size": args.shard_size,
        "shards": shard_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"complete pairs={args.num_pairs} elapsed_h={(time.perf_counter()-started)/3600:.2f}", flush=True)

if __name__ == "__main__":
    main()
