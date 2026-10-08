"""Plot unconditional Project 4 FM solvers against the Project 1 DDIM baseline."""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("results", [])


def nfe(row):
    return int(row.get("forward_evaluations_per_sample", row.get("nfe", row.get("steps", 0))))


def reduce_seeds(rows):
    groups = {}
    for row in rows:
        groups.setdefault(nfe(row), []).append(float(row["fid"]))
    xs = sorted(groups)
    means = [statistics.mean(groups[x]) for x in xs]
    spreads = [statistics.pstdev(groups[x]) if len(groups[x]) > 1 else 0.0 for x in xs]
    return xs, means, spreads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm", default="results/fm_v2_unconditional_solvers.json")
    parser.add_argument("--ddpm", default="../project2_samplers/results/p1_r5_ddim_5k.json")
    parser.add_argument("--output", default="results/nfe_fid_curve.png")
    args = parser.parse_args()

    fm_rows = read_rows(args.fm)
    unconditional = [r for r in fm_rows if r.get("condition_mode") == "unconditional"]
    if not unconditional:
        unconditional = [r for r in fm_rows if float(r.get("cfg_scale", 0) or 0) == 0]
    dd_rows = [r for r in read_rows(args.ddpm) if r.get("sampler", "ddim") == "ddim"]

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    for solver in sorted({r.get("solver", "euler") for r in unconditional}):
        selected = [r for r in unconditional if r.get("solver", "euler") == solver]
        xs, means, spread = reduce_seeds(selected)
        ax.plot(xs, means, marker="o", label=f"FM {solver.title()} (unconditional)")
        if any(spread):
            ax.fill_between(xs, [m-s for m,s in zip(means,spread)],
                            [m+s for m,s in zip(means,spread)], alpha=0.15)
    if dd_rows:
        xs, means, spread = reduce_seeds(dd_rows)
        ax.plot(xs, means, "s-", label="Project 1 R5 + DDIM")
        if any(spread):
            ax.fill_between(xs, [m-s for m,s in zip(means,spread)],
                            [m+s for m,s in zip(means,spread)], alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlabel("Network evaluations per generated image (log scale)")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("CIFAR-10: unconditional Flow Matching vs Project 1 DDPM")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
