"""Re-evaluate the fair-matrix checkpoints restored from the Project 5 release.

This keeps the original training artifacts immutable while producing a fresh,
auditable evaluation tree with the current evaluator and metrics.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from run_fair_experiments import aggregate, conditions_for, write_report


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", default="assets/restored/runs/project5_fair")
    parser.add_argument("--run-root", default="runs/project5_release_eval")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--max-success-plots", type=int, default=3)
    parser.add_argument("--resume", action="store_true",
                        help="Skip evaluations that already have eval.json")
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(checkpoint_root.glob("*/seed_*/ckpts/model_final.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_final.pt checkpoints found under {checkpoint_root}")

    conditions = conditions_for("full")
    for checkpoint in checkpoints:
        source_run = checkpoint.parents[1]
        relative = source_run.relative_to(checkpoint_root)
        label = relative.parts[0]
        condition = next((item for item in conditions if item["label"] == label), None)
        if condition is None:
            print(f"Skipping unrecognized checkpoint condition: {relative}", flush=True)
            continue

        output_run = run_root / relative
        output_run.mkdir(parents=True, exist_ok=True)
        source_config = source_run / "config.yaml"
        config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
        # Data paths in the release point to the original training host. They
        # are not used by evaluation, but rewrite them to avoid stale paths in
        # the copied, self-contained resolved configuration.
        config["demo_path"] = "<not-required-for-checkpoint-evaluation>"
        config_path = output_run / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        output = output_run / "eval.json"
        if args.resume and output.exists():
            print(f"Skipping completed evaluation: {relative}", flush=True)
            continue

        method = condition["method"]
        sample_steps = 1 if method == "bc" else (10 if method == "fm" else 20)
        command = [
            sys.executable, "eval.py", "--config", str(config_path),
            "--ckpt", str(checkpoint), "--method", method,
            "--n_episodes", str(args.episodes), "--seed-start", str(args.seed_start),
            "--exec_steps", "4", "--n_sample_steps", str(sample_steps),
            "--output", str(output), "--rollout-dir", str(output_run / "rollouts"),
            "--max-success-plots", str(args.max_success_plots),
        ]
        print("\n$ " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    evaluated = {path.parents[1].name for path in run_root.glob("*/seed_*/eval.json")}
    expected = {item["label"] for item in conditions}
    missing = sorted(expected - evaluated)
    summary = aggregate(run_root, conditions, seed=42)
    summary["protocol"].update({
        "evaluation_seed_start": args.seed_start,
        "evaluation_episodes_per_condition": args.episodes,
        "source_checkpoint_root": str(checkpoint_root),
        "evaluation_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    })
    summary["protocol"]["missing_conditions"] = missing
    (run_root / "fair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, run_root / "fair_ablation.md")
    print(json.dumps({"state": "complete" if not missing else "incomplete",
                      "evaluated_conditions": len(evaluated),
                      "missing_conditions": missing,
                      "run_root": str(run_root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
