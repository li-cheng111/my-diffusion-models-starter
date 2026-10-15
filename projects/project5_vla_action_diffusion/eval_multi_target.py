"""Repeated fixed-scene evaluation for two simultaneously visible goals."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from device_utils import get_device, make_generator, seed_device
from env import Reach2DEnv
from eval import (DDPMScheduler, bc_sample_action, ddim_sample_action,
                  classify_episode_outcome)
from model import DiffusionPolicy
from obs_utils import state_dim_for, state_from_obs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _load_policy(config: dict, checkpoint_path: Path, device: torch.device):
    model = DiffusionPolicy(
        horizon=config["chunk_size"],
        action_dim=2,
        state_dim=state_dim_for(config["use_vision"]),
        image_size=64,
        vision_out_dim=config["vision_out_dim"],
        hidden=config["hidden"],
        use_vision=config["use_vision"],
        vision_encoder=config.get("vision_encoder", "small"),
        vision_pretrained=config.get("vision_pretrained", False),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    scheduler = DDPMScheduler(
        T=config["diffusion_steps"], device=device,
        schedule=config.get("noise_schedule", "linear"),
    )
    return model, scheduler, checkpoint


def _episode(model, scheduler, config, device, scene_seed, noise_seed,
             exec_steps, sample_steps, method):
    env = Reach2DEnv(
        n_distractors=config["n_distractors"],
        goal_positions=config["goal_positions"],
        seed=scene_seed,
    )
    obs = env.reset(seed=scene_seed)
    scene = {
        "start": obs["state"].astype(float).tolist(),
        "goals": obs["goal_positions"].astype(float).tolist(),
        "obstacles": env.distractors.astype(float).tolist(),
        "initial_image_sha256": hashlib.sha256(obs["image"].tobytes()).hexdigest(),
    }
    positions = [obs["state"].astype(float).tolist()]
    info = {"success": False, "collision": False, "step_count": env.MAX_STEPS,
            "reached_target_id": None, "dist_to_target": float("inf")}
    generator = make_generator(device, noise_seed)
    action_chunk = None

    for step in range(env.MAX_STEPS):
        if step % exec_steps == 0:
            image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float()
            image = (image / 127.5 - 1.0).unsqueeze(0).to(device)
            state = torch.from_numpy(
                state_from_obs(obs, config["use_vision"])
            ).unsqueeze(0).to(device)
            if method == "bc":
                action_chunk = bc_sample_action(
                    model, image, state, horizon=config["chunk_size"], device=device,
                )[0].cpu().numpy()
            else:
                action_chunk = ddim_sample_action(
                    model, scheduler, image, state,
                    horizon=config["chunk_size"], n_steps=sample_steps,
                    device=device, generator=generator,
                )[0].cpu().numpy()

        obs, _, done, info = env.step(action_chunk[step % exec_steps])
        positions.append(obs["state"].astype(float).tolist())
        if done:
            break

    outcome = classify_episode_outcome(info)
    return {
        **scene,
        "noise_seed": int(noise_seed),
        "outcome": outcome,
        "success": outcome == "success",
        "collision": outcome == "collision",
        "reached_target_id": (info.get("reached_target_id")
                              if outcome == "success" else None),
        "steps": int(info.get("step_count", len(positions) - 1)),
        "final_distance": float(info.get("dist_to_target", float("inf"))),
        "path": positions,
    }


def evaluate_multi_target(config_path: Path, checkpoint_path: Path, run_dir: Path,
                          n_scenes: int, samples_per_scene: int, seed_start: int,
                          noise_seed_start: int, exec_steps: int, sample_steps: int,
                          method: str) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not config.get("use_vision", False):
        raise ValueError("The two-visible-goal task requires use_vision: true")
    if len(config.get("goal_positions", [])) < 2:
        raise ValueError("Config must contain at least two goal_positions")
    if samples_per_scene < 1 or n_scenes < 1:
        raise ValueError("n_scenes and samples_per_scene must be positive")

    seed_device(seed_start)
    device = get_device()
    print(f"Evaluation device: {device}", flush=True)
    model, scheduler, checkpoint = _load_policy(config, checkpoint_path, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    output_path = run_dir / "eval_multi_target.json"
    total = n_scenes * samples_per_scene
    episodes = []
    scenes = []
    started = time.time()

    _write_json(status_path, {
        "state": "running", "stage": "multi_target_evaluation",
        "run": run_dir.name, "method": method, "step": 0, "max_steps": total,
        "checkpoint": str(checkpoint_path),
    })
    episode_index = 0
    for scene_index in range(n_scenes):
        scene_seed = seed_start + scene_index
        scene_episodes = []
        reference_signature = None
        for sample_index in range(samples_per_scene):
            noise_seed = noise_seed_start + scene_index * samples_per_scene + sample_index
            result = _episode(
                model, scheduler, config, device, scene_seed, noise_seed,
                exec_steps, sample_steps, method,
            )
            signature = (result["initial_image_sha256"], result["start"],
                         result["goals"], result["obstacles"])
            if reference_signature is None:
                reference_signature = signature
                scene = {key: result[key] for key in
                         ("start", "goals", "obstacles", "initial_image_sha256")}
            elif signature != reference_signature:
                raise RuntimeError(f"Initial scene/observation changed within scene {scene_index}")
            result.update({"scene_index": scene_index, "scene_seed": scene_seed,
                           "sample_index": sample_index})
            scene_episodes.append(result)
            episodes.append(result)
            episode_index += 1

            if episode_index % max(1, min(20, samples_per_scene)) == 0 or episode_index == total:
                completed_success = sum(x["success"] for x in episodes)
                _write_json(status_path, {
                    "state": "running", "stage": "multi_target_evaluation",
                    "run": run_dir.name, "method": method,
                    "step": episode_index, "max_steps": total,
                    "success_rate_so_far": completed_success / episode_index,
                    "elapsed_s": time.time() - started,
                })

        counts = [sum(ep["reached_target_id"] == i for ep in scene_episodes)
                  for i in range(len(config["goal_positions"]))]
        scene.update({"scene_index": scene_index, "scene_seed": scene_seed,
                      "target_success_counts": counts,
                      "successes": sum(ep["success"] for ep in scene_episodes),
                      "episodes": scene_episodes})
        scenes.append(scene)

    successes = sum(ep["success"] for ep in episodes)
    collisions = sum(ep["collision"] for ep in episodes)
    target_counts = [sum(ep["reached_target_id"] == i for ep in episodes)
                     for i in range(len(config["goal_positions"]))]
    both_covered = sum(all(n >= 1 for n in scene["target_success_counts"]) for scene in scenes)
    stable_covered = sum(all(n >= 3 for n in scene["target_success_counts"]) for scene in scenes)
    elapsed = time.time() - started
    result = {
        "method": method,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "device": str(device),
        "config": config,
        "protocol": {
            "n_scenes": n_scenes,
            "samples_per_scene": samples_per_scene,
            "environment_seed_start": seed_start,
            "noise_seed_start": noise_seed_start,
            "exec_steps": exec_steps,
            "ddim_steps": sample_steps,
            "same_initial_observation_per_scene": True,
        },
        "episodes": total,
        "successes": successes,
        "collisions": collisions,
        "timeouts": total - successes - collisions,
        "success_rate": successes / total,
        "collision_rate": collisions / total,
        "timeout_rate": (total - successes - collisions) / total,
        "avg_steps": float(np.mean([ep["steps"] for ep in episodes])),
        "mean_final_distance": float(np.mean([ep["final_distance"] for ep in episodes])),
        "target_success_counts": target_counts,
        "target_success_rates_per_episode": [n / total for n in target_counts],
        "scenes_with_both_targets_reached": both_covered,
        "both_target_coverage_rate": both_covered / n_scenes,
        "scenes_with_each_target_reached_at_least_3_times": stable_covered,
        "stable_both_target_coverage_rate": stable_covered / n_scenes,
        "elapsed_s": elapsed,
        "scenes": scenes,
    }
    _write_json(output_path, result)
    _write_json(status_path, {
        "state": "completed", "stage": "multi_target_evaluation",
        "run": run_dir.name, "method": method, "step": total, "max_steps": total,
        "elapsed_s": elapsed, "eval": {key: result[key] for key in
            ("success_rate", "collision_rate", "timeout_rate", "avg_steps",
             "both_target_coverage_rate", "stable_both_target_coverage_rate")},
    })
    print(json.dumps({key: result[key] for key in
          ("success_rate", "collision_rate", "timeout_rate", "avg_steps",
           "target_success_counts", "both_target_coverage_rate",
           "stable_both_target_coverage_rate", "elapsed_s")},
          ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {output_path}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reach2d_two_targets.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-scenes", type=int, default=100)
    parser.add_argument("--samples-per-scene", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--noise-seed-start", type=int, default=900000)
    parser.add_argument("--exec-steps", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--method", choices=("ddpm", "bc"), default="ddpm")
    args = parser.parse_args()
    evaluate_multi_target(
        Path(args.config), Path(args.checkpoint), Path(args.run_dir),
        args.n_scenes, args.samples_per_scene, args.seed_start,
        args.noise_seed_start, args.exec_steps, args.sample_steps, args.method,
    )


if __name__ == "__main__":
    main()
