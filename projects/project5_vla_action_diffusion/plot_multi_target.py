"""Plot repeated closed-loop trajectories from eval_multi_target.py JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_results(result_path: Path, output_dir: Path, max_scenes: int = 8) -> Path:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    scenes = payload["scenes"]
    ranked = sorted(
        scenes,
        key=lambda scene: (
            all(n > 0 for n in scene["target_success_counts"]),
            scene["successes"],
        ),
        reverse=True,
    )[:max_scenes]
    if not ranked:
        raise ValueError("Evaluation result contains no scenes")

    columns = min(4, len(ranked))
    rows = int(np.ceil(len(ranked) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.0 * rows),
                             squeeze=False)
    colors = {0: "#2474d2", 1: "#ed8a24"}
    for ax, scene in zip(axes.flat, ranked):
        for obstacle in scene["obstacles"]:
            ax.add_patch(Rectangle((obstacle[0] - 0.07, obstacle[1] - 0.07),
                                   0.14, 0.14, facecolor="#888888",
                                   edgecolor="#555555", alpha=0.7, zorder=2))
        for goal_id, goal in enumerate(scene["goals"]):
            ax.scatter(goal[0], goal[1], marker="*", s=180, color="#d62728",
                       edgecolor="white", linewidth=0.7, zorder=5)
            ax.text(goal[0], goal[1] + 0.10, f"G{goal_id}", ha="center", fontsize=9)
        start = scene["start"]
        ax.scatter(start[0], start[1], marker="o", s=55, color="#1f2937", zorder=5)

        for episode in scene["episodes"]:
            target_id = episode["reached_target_id"]
            color = colors.get(target_id, "#9aa0a6")
            alpha = 0.58 if target_id is not None else 0.24
            path = np.asarray(episode["path"])
            ax.plot(path[:, 0], path[:, 1], color=color, alpha=alpha,
                    linewidth=1.1, zorder=3 if target_id is not None else 1)

        counts = scene["target_success_counts"]
        ax.set_title(f"Scene {scene['scene_index']} · G0={counts[0]}, G1={counts[1]}")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    for ax in axes.flat[len(ranked):]:
        ax.axis("off")
    fig.suptitle(
        f"Fixed-observation multi-target rollouts ({payload['method']}, "
        f"{payload['protocol']['samples_per_scene']} samples per scene)",
        fontsize=14,
    )
    fig.legend(handles=[
        plt.Line2D([0], [0], color=colors[0], label="Reached goal 0"),
        plt.Line2D([0], [0], color=colors[1], label="Reached goal 1"),
        plt.Line2D([0], [0], color="#9aa0a6", label="Collision / timeout"),
    ], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multi_target_rollouts.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-scenes", type=int, default=8)
    args = parser.parse_args()
    print(plot_results(Path(args.input), Path(args.output_dir), args.max_scenes))


if __name__ == "__main__":
    main()
