"""Plot a compact summary of the Project 5 bonus results."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def load_training_curve(name: str) -> tuple[list[int], list[float]]:
    pattern = re.compile(
        r"step\s+(\d+)/\d+\s+\|\s+loss\s+[\d.]+\s+"
        r"\(avg100\s+([\d.]+)\)\s+\|\s+[\d.]+/s"
    )
    steps: list[int] = []
    losses: list[float] = []
    for line in (ROOT / "logs" / name).read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))
    if not steps:
        raise ValueError(f"No training metrics found in {name}")
    return steps, losses


def plot_resnet_convergence() -> None:
    curves = [
        ("Small CNN (batch 256, lr 1e-3)", "train_spatial.log", "#1f77b4"),
        ("ImageNet ResNet-18, 5k (batch 128, lr 3e-4)", "train_resnet18.log", "#ff7f0e"),
        ("ImageNet ResNet-18, 15k (batch 128, lr 1e-3)", "train_resnet18_long.log", "#2ca02c"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, log_name, color in curves:
        steps, losses = load_training_curve(log_name)
        ax.plot(steps, losses, label=label, color=color, linewidth=2)
        ax.scatter([steps[-1]], [losses[-1]], color=color, s=24, zorder=3)
    ax.set_xlabel("Optimizer updates")
    ax.set_ylabel("Moving-average training loss (100 steps)")
    ax.set_title("Small CNN vs ImageNet-pretrained ResNet-18")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output = RESULTS / "resnet_convergence.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


def main() -> None:
    entries = [
        ("Multimodal\n(two targets)", load("eval_multimodal.json"), "#2ca02c"),
        ("Moving\ndistractors", load("eval_moving.json"), "#ff7f0e"),
        ("ResNet18\n5k steps", load("eval_resnet18.json"), "#9467bd"),
        ("ResNet18\n15k steps", load("eval_resnet18_long.json"), "#d62728"),
    ]
    labels = [e[0] for e in entries]
    success = [100 * e[1]["success_rate"] for e in entries]
    collision = [100 * e[1]["collision_rate"] for e in entries]
    timeout = [100 * e[1]["timeout_rate"] for e in entries]

    fig, (ax, ax_mode) = plt.subplots(1, 2, figsize=(11, 4.8),
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    x = list(range(len(labels)))
    ax.bar(x, success, label="success", color="#2ca02c")
    ax.bar(x, collision, bottom=success, label="collision", color="#d62728")
    ax.bar(x, timeout, bottom=[s + c for s, c in zip(success, collision)],
           label="timeout", color="#7f7f7f")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Episodes (%)")
    ax.set_xticks(x, labels)
    ax.set_title("Bonus closed-loop outcomes (100 episodes)")
    ax.legend(loc="upper right", frameon=False)
    for i, value in enumerate(success):
        ax.text(i, value + 2, f"{value:.0f}%", ha="center", va="bottom", fontsize=10)

    ax_mode.set_xlim(-1, 1)
    ax_mode.set_ylim(-1, 1)
    ax_mode.scatter([-0.65, 0.65], [0.55, 0.55], s=550, c=["#e41a1c", "#377eb8"],
                    edgecolors="black", linewidths=1.2, label="target modes")
    ax_mode.scatter([0], [-0.55], s=420, c="#1f77b4", edgecolors="black", label="agent")
    ax_mode.annotate("mode 0\n96.2%", (-0.65, 0.55), xytext=(-0.95, 0.82),
                     arrowprops={"arrowstyle": "->"}, ha="center")
    ax_mode.annotate("mode 1\n97.9%", (0.65, 0.55), xytext=(0.95, 0.82),
                     arrowprops={"arrowstyle": "->"}, ha="center")
    ax_mode.set_title("Two target modes")
    ax_mode.set_xlabel("Environment x")
    ax_mode.set_ylabel("Environment y")
    ax_mode.grid(alpha=0.2)
    fig.tight_layout()
    output = RESULTS / "bonus_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)
    plot_resnet_convergence()


if __name__ == "__main__":
    main()
