"""Run controlled Project 5 comparisons with shared data and architecture.

The full matrix contains ten conditions with one fixed training seed:
  * BC / DDIM / FM at H=16 with padding masks;
  * DDIM at H=8/16/32/64 with padding masks;
  * DDIM at H=8/16/32/64 without masks as the padding control.

Every run writes its resolved config, checkpoint, metrics and 100-episode
closed-loop evaluation beneath ``runs/project5_fair``. Re-running with
``--resume`` skips completed training/evaluation and rebuilds the aggregate.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def conditions_for(preset: str) -> list[dict]:
    conditions: "OrderedDict[str, dict]" = OrderedDict()

    def add(method: str, horizon: int, masked: bool) -> None:
        label = f"{method}_h{horizon}_{'masked' if masked else 'unmasked'}"
        conditions[label] = {
            "label": label, "method": method, "horizon": horizon,
            "mask_padding": masked,
        }

    if preset in {"algorithms", "full"}:
        for method in ("bc", "ddpm", "fm"):
            add(method, 16, True)
    if preset in {"chunks", "full"}:
        for horizon in (8, 16, 32, 64):
            add("ddpm", horizon, True)
    if preset in {"padding", "full"}:
        for horizon in (8, 16, 32, 64):
            add("ddpm", horizon, False)
    return list(conditions.values())


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    if episodes == 0:
        return 0.0, 0.0
    p = successes / episodes
    denom = 1 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denom
    radius = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denom
    return centre - radius, centre + radius


def aggregate(run_root: Path, conditions: list[dict], seed: int) -> dict:
    summary = {"protocol": {"training_seed": seed}, "conditions": {}}
    for condition in conditions:
        path = run_root / condition["label"] / f"seed_{seed}" / "eval.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        successes = row["successes"]
        episodes = row["episodes"]
        low, high = wilson_interval(successes, episodes)
        summary["conditions"][condition["label"]] = {
            **condition,
            "training_seed": seed,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "success_rate_wilson95": [low, high],
            "collision_rate": row["collision_rate"],
            "timeout_rate": row["timeout_rate"],
            "avg_steps": row["avg_steps"],
            "mean_final_distance": row["mean_final_distance"],
            "evaluation": row,
        }
    return summary


def write_report(summary: dict, output: Path) -> None:
    lines = [
        "# Controlled Project 5 ablation",
        "",
        "All rows use training seed 42, the same 4x4 spatial CNN, expert-data "
        "seed, cosine noise schedule, closed-loop protocol and paired evaluation "
        "episodes. Wilson intervals quantify episode variation for this one "
        "trained checkpoint; training-seed variation is not estimated.",
        "",
        "| condition | method | H | masked | train seed | success (Wilson 95% CI) | collision | timeout | avg steps |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["conditions"].items():
        lines.append(
            f"| {name} | {row['method']} | {row['horizon']} | "
            f"{'yes' if row['mask_padding'] else 'no'} | 42 | "
            f"{100 * row['success_rate']:.1f}% "
            f"({100 * row['success_rate_wilson95'][0]:.1f}–"
            f"{100 * row['success_rate_wilson95'][1]:.1f}%) | "
            f"{100 * row['collision_rate']:.1f}% | "
            f"{100 * row['timeout_rate']:.1f}% | {row['avg_steps']:.2f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("algorithms", "chunks", "padding", "full"),
                        default="full")
    parser.add_argument("--seed", type=int, default=42,
                        help="Single training seed used for every comparison (default: 42)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--run-root", default="runs/project5_fair")
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing final checkpoints and evaluations")
    args = parser.parse_args()

    base = yaml.safe_load((ROOT / "configs" / "reach2d_fair.yaml").read_text(encoding="utf-8"))
    run_root = (ROOT / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    conditions = conditions_for(args.preset)

    for condition in conditions:
        horizon = condition["horizon"]
        data_path = (ROOT / "data" / f"fair_demos_h{horizon}.pkl").resolve()
        for seed in [args.seed]:
            run_dir = run_root / condition["label"] / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = dict(base)
            config.update({
                "chunk_size": horizon,
                "seed": seed,
                "demo_path": str(data_path),
                "mask_padding": condition["mask_padding"],
                "max_steps": args.steps,
                "ckpt_dir": str(run_dir / "ckpts"),
            })
            config_path = run_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            checkpoint = run_dir / "ckpts" / "model_final.pt"
            if not (args.resume and checkpoint.exists()):
                command = [sys.executable, "train.py", "--config", str(config_path),
                           "--method", condition["method"], "--run-dir", str(run_dir),
                           "--seed", str(seed)]
                if not data_path.exists():
                    command.append("--collect")
                run(command)

            output = run_dir / "eval.json"
            if not (args.resume and output.exists()):
                sample_steps = 1 if condition["method"] == "bc" else (
                    10 if condition["method"] == "fm" else 20
                )
                run([sys.executable, "eval.py", "--config", str(config_path),
                     "--ckpt", str(checkpoint), "--method", condition["method"],
                     "--n_episodes", str(args.episodes), "--seed-start", "10000",
                     "--exec_steps", "4", "--n_sample_steps", str(sample_steps),
                     "--output", str(output), "--rollout-dir", str(run_dir / "rollouts"),
                     "--max-success-plots", "3"])

            summary = aggregate(run_root, conditions, args.seed)
            (run_root / "fair_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_report(summary, run_root / "fair_ablation.md")

    print(json.dumps({"state": "complete", "run_root": str(run_root),
                      "conditions": len(conditions), "training_seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
