"""Create compact, reproducible figures used by report.md.

All inputs are committed experiment outputs; no model weights or source images
are loaded by this utility.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPORT = OUTPUTS / "report"
REPORT.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def montage(paths: list[Path], labels: list[str], target: Path, ncols: int = 2) -> None:
    rows = (len(paths) + ncols - 1) // ncols
    fig, axes = plt.subplots(rows, ncols, figsize=(6.4 * ncols, 5.2 * rows), squeeze=False)
    for index, (path, label) in enumerate(zip(paths, labels)):
        ax = axes[index // ncols][index % ncols]
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(label)
        ax.axis("off")
    for index in range(len(paths), rows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_loss_curve() -> None:
    records = []
    metrics = OUTPUTS / "lora" / "full" / "training_metrics.jsonl"
    for line in metrics.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "step":
            records.append(row)
    steps = np.asarray([row["step"] for row in records], dtype=float)
    losses = np.asarray([row["loss"] for row in records], dtype=float)
    window = 50
    rolling = np.convolve(losses, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(steps, losses, color="#9aa4b2", linewidth=0.7, alpha=0.45, label="step loss")
    ax.plot(steps[window - 1 :], rolling, color="#1f77b4", linewidth=2.0, label="50-step mean")
    for checkpoint in (200, 400, 600, 800):
        ax.axvline(checkpoint, color="#d62728", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("MSE noise-prediction loss")
    ax.set_title("LoRA training loss on AutoDL RTX 4090")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "lora_loss_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_vae_statistics() -> None:
    metrics = load_json(OUTPUTS / "vae_metrics.json")
    means = np.asarray(metrics["latent_channel_mean"], dtype=float)
    stds = np.asarray(metrics["latent_channel_std"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    channels = np.arange(1, len(means) + 1)
    axes[0].bar(channels, means, yerr=stds, capsize=5, color="#4c78a8", alpha=0.85)
    axes[0].axhline(0, color="#555", linewidth=0.8)
    axes[0].set_xlabel("VAE latent channel")
    axes[0].set_ylabel("mean ± standard deviation")
    axes[0].set_xticks(channels)
    axes[0].set_title("Channel statistics")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].axis("off")
    axes[1].text(
        0.05,
        0.72,
        f"Input: {tuple(metrics['input_shape'])}\n"
        f"Latent: {tuple(metrics['latent_shape'])}\n"
        f"Reconstruction: {tuple(metrics['reconstruction_shape'])}\n\n"
        f"MSE: {metrics['mse_01']:.7f}\n"
        f"PSNR: {metrics['psnr_db']:.4f} dB",
        va="top",
        fontsize=12,
        family="monospace",
    )
    axes[1].set_title("Reconstruction quality")
    fig.tight_layout()
    fig.savefig(REPORT / "vae_statistics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_loss_curve()
    make_vae_statistics()
    montage(
        [
            OUTPUTS / "sweep" / "sweep_cfg.png",
            OUTPUTS / "sweep" / "sweep_steps.png",
            OUTPUTS / "sweep" / "sweep_sampler.png",
            OUTPUTS / "sweep" / "grid_2d.png",
        ],
        ["CFG sweep", "steps sweep", "sampler sweep", "CFG × steps grid"],
        REPORT / "sweep_overview.png",
    )
    montage(
        [
            OUTPUTS / "controlnet" / "controlnet_grid.png",
            OUTPUTS / "controlnet" / "controlnet_conflict_dog_on_same_edges.png",
            OUTPUTS / "lora" / "full_eval" / "lora_before_after.png",
            OUTPUTS / "attention" / "full" / "generated.png",
        ],
        ["ControlNet styles", "ControlNet conflict", "LoRA comparison", "Attention target"],
        REPORT / "conditioned_outputs.png",
    )
    print(f"wrote report figures to {REPORT}")


if __name__ == "__main__":
    main()
