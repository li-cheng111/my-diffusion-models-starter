"""
Closed-loop evaluation for Project 5. Contains: TODO 21 (eval loop).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from env import Reach2DEnv
from model import DiffusionPolicy
from obs_utils import state_dim_for, state_from_obs
from train import DDPMScheduler, load_config


def make_ddim_timesteps(T, n_steps):
    """Return descending, endpoint-inclusive DDIM indices."""
    if T < 1 or n_steps < 1:
        raise ValueError("T and n_steps must be positive")
    count = min(T, n_steps)
    return torch.linspace(T - 1, 0, count).round().long().unique_consecutive().tolist()


@torch.no_grad()
def ddim_sample_action(model, scheduler, image, state, horizon=32, action_dim=2,
                       n_steps=20, device="cuda", generator=None):
    """Deterministic DDIM (eta=0) sampling for a DDPM-trained action model."""
    model.eval()
    B = image.shape[0]
    a = torch.randn(B, horizon, action_dim, device=device, generator=generator)
    T = scheduler.T
    timesteps = make_ddim_timesteps(T, n_steps)
    for i, ti in enumerate(timesteps):
        t_tensor = torch.full((B,), ti, device=device, dtype=torch.long)
        eps_pred = model(a, t_tensor, image=image, state=state)
        # Reconstruct x_0
        ac_t = scheduler.alpha_cum[ti]
        x_0 = (a - (1 - ac_t).sqrt() * eps_pred) / ac_t.sqrt()
        x_0 = x_0.clamp(-1.5, 1.5)
        # Move to previous step
        if i + 1 < len(timesteps):
            ti_prev = timesteps[i + 1]
            ac_prev = scheduler.alpha_cum[ti_prev]
            a = ac_prev.sqrt() * x_0 + (1 - ac_prev).sqrt() * eps_pred
        else:
            a = x_0
    return a  # (B, H, A)


# Backward-compatible import name used by earlier tests/scripts. The sampler
# has always been deterministic; the explicit DDIM name avoids misreporting it
# as ancestral DDPM sampling in new experiments.
ddpm_sample_action = ddim_sample_action


@torch.no_grad()
def fm_sample_action(model, image, state, horizon=32, action_dim=2,
                     n_steps=10, device="cuda", generator=None):
    """Euler integration of the action flow from noise to data."""
    model.eval()
    a = torch.randn(image.shape[0], horizon, action_dim, device=device,
                    generator=generator)
    n_steps = max(1, n_steps)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((image.shape[0],), i / n_steps, device=device)
        a = a + dt * model(a, t, image=image, state=state)
    return a.clamp(-1.5, 1.5)


@torch.no_grad()
def bc_sample_action(model, image, state, horizon=32, action_dim=2, device="cuda"):
    """One deterministic forward pass for the Pure BC baseline."""
    model.eval()
    zeros = torch.zeros(image.shape[0], horizon, action_dim, device=device)
    t = torch.zeros(image.shape[0], device=device, dtype=torch.long)
    return model(zeros, t, image=image, state=state).clamp(-1.0, 1.0)


# =============================================================================
# TODO 21: Implement closed-loop evaluation
# =============================================================================
def evaluate(model, scheduler, cfg, device, n_episodes=100, exec_steps=10,
             chunk_size=32, n_sample_steps=20, verbose=False, seed_start=10000,
             method="ddpm", rollout_dir=None, max_success_plots=3):
    """
    Closed-loop policy evaluation.

    For each of n_episodes:
        1) env.reset() → obs
        2) Loop until done:
            a) Every `exec_steps` step (or step 0), sample fresh action chunk
            b) Execute the next action from the chunk
            c) Update obs, check info['success']/'collision']
        3) Record success / collision / steps

    Args:
        model: trained DiffusionPolicy
        scheduler: DDPMScheduler
        cfg: config dict
        device: cuda or cpu
        n_episodes: number of evaluation episodes
        exec_steps: how many actions from each generated chunk to execute before re-planning
        chunk_size: action chunk length (must match training cfg['chunk_size'])
        n_sample_steps: DDPM sampling steps

    Returns:
        dict with keys 'success_rate', 'collision_rate', 'avg_steps', 'episodes'

    可以直接用的零件（不用自己拼条件表示）：
        env = Reach2DEnv(n_distractors=n_dist, seed=1000 + ep)   # 固定 seed 才可复现
        obs = env.reset()
        image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float() / 127.5 - 1.0
        image = image.unsqueeze(0).to(device)                    # (1, 3, 64, 64)
        state = torch.from_numpy(state_from_obs(obs, use_vision)).unsqueeze(0).to(device)
        chunk = ddpm_sample_action(model, scheduler, image, state,
                                   horizon=chunk_size, n_steps=n_sample_steps,
                                   device=device)[0].cpu().numpy()   # (H, 2)

    ⚠️ 两个最容易错的点：
    1. image 的归一化必须和 dataset.py 里一模一样（/127.5 - 1.0），
       否则训练分布和评估分布对不上，成功率会莫名其妙地低。
    2. 重规划的计数：`exec_steps` 是"执行几步后重新采样"，
       从 chunk 里取的下标是 step % exec_steps，不是 step % chunk_size。
       README §阶段 4 的伪代码给的是前者。

    评估用的 seed 要和训练数据的 seed 区间错开（collect_demos 用的是 0..2*n_demos），
    否则你测的是训练集，成功率虚高。
    """
    n_dist = cfg["n_distractors"]
    use_vision = cfg["use_vision"]

    # ============================================================
    # TODO 21: Implement closed-loop evaluation (≈ 25-40 lines)
    # ============================================================
    if not 1 <= exec_steps <= chunk_size:
        raise ValueError(f"exec_steps must be in [1, {chunk_size}], got {exec_steps}")
    model.eval()
    successes = collisions = timeouts = 0
    total_steps = 0
    final_distances = []
    saved_rollouts = 0
    mode_counts = {}
    mode_successes = {}
    rollout_dir = Path(rollout_dir) if rollout_dir else None
    if rollout_dir:
        rollout_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(n_episodes):
        episode_seed = seed_start + ep
        env = Reach2DEnv(
            n_distractors=n_dist,
            seed=episode_seed,
            target_choices=cfg.get("target_choices"),
            moving_distractors=cfg.get("moving_distractors", False),
            distractor_speed=cfg.get("distractor_speed", 0.025),
        )
        # Reach2DEnv initializes itself for interactive use. Reset with the
        # same explicit seed so evaluation has exactly one reproducible initial
        # state per episode rather than consuming the RNG stream twice.
        obs = env.reset(seed=episode_seed)
        mode = None
        if cfg.get("target_choices") is not None:
            choices = np.asarray(cfg["target_choices"], dtype=np.float32)
            mode = int(np.argmin(np.linalg.norm(choices - env.target_pos, axis=1)))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        generator = torch.Generator(device=device)
        generator.manual_seed(episode_seed)
        positions = [obs["state"].copy()]
        action_chunk = None
        info = {"success": False, "collision": False, "dist_to_target": float("inf"),
                "step_count": env.MAX_STEPS}
        for step in range(env.MAX_STEPS):
            if step % exec_steps == 0:
                image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float()
                image = (image / 127.5 - 1.0).unsqueeze(0).to(device)
                state = torch.from_numpy(state_from_obs(obs, use_vision)).unsqueeze(0).to(device)
                if method == "fm":
                    action_chunk = fm_sample_action(
                        model, image, state, horizon=chunk_size,
                        n_steps=n_sample_steps, device=device, generator=generator,
                    )[0].cpu().numpy()
                elif method == "bc":
                    action_chunk = bc_sample_action(
                        model, image, state, horizon=chunk_size, device=device,
                    )[0].cpu().numpy()
                else:
                    action_chunk = ddim_sample_action(
                        model, scheduler, image, state, horizon=chunk_size,
                        n_steps=n_sample_steps, device=device, generator=generator,
                    )[0].cpu().numpy()
            obs, _, done, info = env.step(action_chunk[step % exec_steps])
            positions.append(obs["state"].copy())
            if done:
                break

        total_steps += info["step_count"]
        final_distances.append(info["dist_to_target"])
        if info["success"]:
            successes += 1
            if mode is not None:
                mode_successes[mode] = mode_successes.get(mode, 0) + 1
            if rollout_dir and saved_rollouts < max_success_plots:
                import matplotlib.pyplot as plt
                path = np.asarray(positions)
                fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
                if len(env.distractors):
                    ax.scatter(env.distractors[:, 0], env.distractors[:, 1], c="gray",
                               marker="s", s=90, label="distractor")
                ax.plot(path[:, 0], path[:, 1], "b.-", linewidth=1.5,
                        markersize=2, label="policy")
                ax.scatter(*path[0], c="royalblue", s=45, label="start")
                ax.scatter(*env.target_pos, c="red", marker="*", s=100, label="target")
                ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal",
                       title=f"episode {ep} · {method}")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7, loc="upper right")
                fig.tight_layout()
                fig.savefig(rollout_dir / f"success_ep{ep:03d}.png")
                plt.close(fig)
                saved_rollouts += 1
        elif info["collision"]:
            collisions += 1
        else:
            timeouts += 1
        if verbose and ((ep + 1) % 10 == 0 or ep == 0):
            print(f"episode {ep + 1}/{n_episodes} | success={successes} collision={collisions}", flush=True)

    result = {
        "method": method,
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "mean_final_distance": float(np.mean(final_distances)),
        "episodes": n_episodes,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "seed_start": seed_start,
        "seed_end": seed_start + n_episodes - 1,
        "exec_steps": exec_steps,
        "chunk_size": chunk_size,
        "n_sample_steps": n_sample_steps,
        "rollouts_saved": saved_rollouts,
    }
    if mode_counts:
        result["mode_stats"] = {
            str(k): {"episodes": mode_counts[k],
                     "successes": mode_successes.get(k, 0),
                     "success_rate": mode_successes.get(k, 0) / mode_counts[k]}
            for k in sorted(mode_counts)
        }
    return result
    # ============================================================
    # END TODO 21
    # ============================================================


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--exec_steps", type=int, default=10)
    parser.add_argument("--n_sample_steps", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--method", choices=("ddpm", "fm", "bc"), default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--rollout-dir", default=None)
    parser.add_argument("--max-success-plots", type=int, default=3)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionPolicy(
        horizon=cfg["chunk_size"],
        action_dim=2,
        state_dim=state_dim_for(cfg["use_vision"]),
        image_size=64,
        vision_out_dim=cfg["vision_out_dim"],
        hidden=cfg["hidden"],
        use_vision=cfg["use_vision"],
        vision_encoder=cfg.get("vision_encoder", "small"),
        vision_pretrained=cfg.get("vision_pretrained", False),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    use_ema = not args.no_ema
    state_dict = ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state_dict)
    method = args.method or ckpt.get("method", "ddpm")
    print(f"Loaded {'EMA' if use_ema else 'main'} model from {args.ckpt} ({method})")

    scheduler = DDPMScheduler(
        T=cfg["diffusion_steps"], device=device,
        schedule=cfg.get("noise_schedule", "linear"),
    )

    status_path = Path(args.output).parent / "status.json" if args.output else None
    if status_path:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps({"state": "evaluating", "method": method,
                                           "episodes": args.n_episodes}, indent=2), encoding="utf-8")

    results = evaluate(
        model, scheduler, cfg, device,
        n_episodes=args.n_episodes,
        exec_steps=args.exec_steps,
        chunk_size=cfg["chunk_size"],
        n_sample_steps=args.n_sample_steps,
        verbose=True,
        seed_start=args.seed_start,
        method=method,
        rollout_dir=args.rollout_dir,
        max_success_plots=args.max_success_plots,
    )

    print("\n=== Evaluation Results ===")
    print(f"Success rate:   {results['success_rate']*100:.1f}%")
    print(f"Collision rate: {results['collision_rate']*100:.1f}%")
    print(f"Avg steps:      {results['avg_steps']:.1f}")
    print(f"Total episodes: {results['episodes']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        status_path.write_text(json.dumps({"state": "completed", "method": method,
                                           "step": ckpt.get("step"),
                                           "eval": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved results: {output}")


if __name__ == "__main__":
    main()
