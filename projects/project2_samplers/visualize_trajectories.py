"""Compare sampler trajectories from exactly the same initial noise tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from .checkpoint_utils import load_model_weights, repository_commit
    from .project1_path import add_project1_to_path
    from .samplers import get_sampler
except ImportError:  # pragma: no cover - direct script compatibility
    from checkpoint_utils import load_model_weights, repository_commit
    from project1_path import add_project1_to_path
    from samplers import get_sampler


def trajectory_label(timesteps: list[int], trajectory_index: int) -> str:
    if trajectory_index >= len(timesteps):
        return "x0"
    return f"t={timesteps[trajectory_index]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--samplers",
        nargs="+",
        default=["ddpm", "ddim", "euler", "dpm-solver"],
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_show", type=int, default=10)
    parser.add_argument("--output", default="runs/trajectory_compare.png")
    parser.add_argument("--metrics_output", default="runs/trajectory_metrics.json")
    parser.add_argument("--project1_path", default=None)
    args = parser.parse_args()

    project1 = add_project1_to_path(args.project1_path)
    from dataset import denormalize
    from model import UNet
    from schedule import DDPMSchedule

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = UNet(**config["model"]).to(device)
    schedule = DDPMSchedule(**config["diffusion"]).to(device)
    weight_type = load_model_weights(model, checkpoint, use_ema=True)
    model.eval()

    shape = (
        1,
        int(config["model"]["in_channels"]),
        int(config["model"]["image_size"]),
        int(config["model"]["image_size"]),
    )
    initial_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_noise = torch.randn(shape, device=device, generator=initial_generator)

    all_trajectories: dict[str, list[torch.Tensor]] = {}
    all_timesteps: dict[str, list[int]] = {}
    metrics: list[dict] = []
    model_label = "Project 1 CIFAR-10 epsilon-prediction U-Net"
    for sampler_name in args.samplers:
        strategy = "lambda" if sampler_name.replace("_", "-") == "dpm-solver" else "linear"
        sampler = get_sampler(
            sampler_name,
            model,
            schedule,
            device=device,
            timestep_strategy=strategy,
        )
        num_steps = int(schedule.T) if sampler_name == "ddpm" else args.steps
        noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        _, trajectory = sampler.sample(
            shape,
            num_steps=num_steps,
            return_trajectory=True,
            initial_x=initial_noise,
            generator=noise_generator,
        )
        timesteps = sampler.get_timesteps(num_steps)
        displacements = [
            float(
                torch.linalg.vector_norm(
                    (right - left).reshape(right.shape[0], -1), dim=1
                ).mean()
            )
            for left, right in zip(trajectory, trajectory[1:])
        ]
        all_trajectories[sampler_name] = trajectory
        all_timesteps[sampler_name] = timesteps
        metrics.append(
            {
                "sampler": sampler_name,
                "model": model_label,
                "weights": weight_type,
                "num_steps": num_steps,
                "nfe": sampler.nfe_for_steps(num_steps),
                "timestep_strategy": strategy,
                "initial_noise_seed": args.seed,
                "trajectory_points": len(trajectory),
                "mean_step_l2": sum(displacements) / len(displacements),
                "max_step_l2": max(displacements),
                "final_l2_from_initial": float(
                    torch.linalg.vector_norm(
                        (trajectory[-1] - trajectory[0]).reshape(1, -1), dim=1
                    ).item()
                ),
            }
        )

    figure, axes = plt.subplots(
        len(args.samplers),
        args.num_show,
        figsize=(2.0 * args.num_show, 2.15 * len(args.samplers)),
        squeeze=False,
    )
    in_channels = shape[1]
    for row, sampler_name in enumerate(args.samplers):
        trajectory = all_trajectories[sampler_name]
        timesteps = all_timesteps[sampler_name]
        indices = torch.linspace(0, len(trajectory) - 1, args.num_show).round().long()
        for column, index_tensor in enumerate(indices):
            index = int(index_tensor.item())
            image = denormalize(trajectory[index][0]).detach().cpu()
            if in_channels == 1:
                axes[row, column].imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                axes[row, column].imshow(image.permute(1, 2, 0).numpy().clip(0, 1))
            axes[row, column].axis("off")
            axes[row, column].set_title(trajectory_label(timesteps, index), fontsize=9)
            if column == 0:
                axes[row, column].set_ylabel(sampler_name, fontsize=11)
    figure.suptitle("Sampler trajectories from the same initial noise", fontsize=13)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    metrics_payload = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "project1_path": str(project1),
        "git_commit": repository_commit(),
        "model": model_label,
        "weights": weight_type,
        "seed": args.seed,
        "same_initial_noise": True,
        "results": metrics,
    }
    metrics_output = Path(args.metrics_output)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved {output} and {metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
