"""
Demo data collection & dataset for Project 5.

- collect_demos: roll out expert policy in env, save (image, state, action_chunk) triples
- DemoDataset: PyTorch Dataset wrapping collected demos
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def collect_demos(n_demos=1000, chunk_size=32, n_distractors=2, save_path=None,
                  seed=0, target_choices=None):
    """Collect n_demos rollouts, return list of (image_seq, state_seq, action_chunk)."""
    from env import Reach2DEnv, expert_policy

    data = []
    n_success = 0
    for i in range(n_demos * 2):  # over-sample to get enough successes
        env = Reach2DEnv(n_distractors=n_distractors, seed=seed + i,
                         target_choices=target_choices)
        env.reset()
        obs_list, act_list, info = expert_policy(env)
        if not info["success"]:
            continue
        # Pack into chunks of length chunk_size with sliding window
        L = len(obs_list)
        for start in range(L):
            end = start + chunk_size
            action_mask = np.ones((chunk_size,), dtype=np.float32)
            if end > L:
                # 末尾不足一个 chunk：**重复最后一个 action** 补齐。
                #
                # ⚠️ 不要用零填充。轨迹末尾的 window 大部分是 padding，
                # 零填充会让模型学到"大量时候输出 0"，闭环时 policy 直接卡住不动
                # （README 常见问题里那条"policy 动不了"就是这么来的）。
                # 重复最后一个 action 表示"保持当前动作"，语义上也更合理——
                # Diffusion Policy / ACT 用的都是这个做法。
                actions = np.zeros((chunk_size, 2), dtype=np.float32)
                valid = np.array(act_list[start:L], dtype=np.float32)
                actions[: L - start] = valid
                actions[L - start:] = valid[-1]
                action_mask[L - start:] = 0.0
            else:
                actions = np.array(act_list[start:end], dtype=np.float32)
            data.append({
                "image": obs_list[start]["image"],
                "state": obs_list[start]["state"],
                "goal": obs_list[start]["goal"],
                "action_chunk": actions,
                "action_mask": action_mask,
            })
        n_success += 1
        if n_success >= n_demos:
            break

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} (image, state, chunk) samples from {n_success} demos → {save_path}")
    return data


def collect_multi_target_demos(n_scenes=500, chunk_size=16, n_distractors=2,
                               goal_positions=((-0.65, 0.55), (0.65, 0.55)),
                               save_path=None, seed=42000, max_attempts=None,
                               progress_callback=None):
    """Collect paired expert trajectories to both visible goals per scene.

    Each pair starts from a byte-identical observation.  The selected expert
    goal is stored only as analysis metadata and is never returned as a model
    input by :class:`DemoDataset`.
    """
    from env import Reach2DEnv, expert_policy

    goals = np.asarray(goal_positions, dtype=np.float32)
    if goals.ndim != 2 or goals.shape[1] != 2 or len(goals) < 2:
        raise ValueError("goal_positions must contain at least two 2D goals")
    if n_scenes < 1 or chunk_size < 1:
        raise ValueError("n_scenes and chunk_size must be positive")

    data = []
    accepted = 0
    attempts = 0
    max_attempts = max_attempts or n_scenes * 8
    while accepted < n_scenes and attempts < max_attempts:
        scene_seed = seed + attempts
        attempts += 1
        paired = []
        reference = None
        valid_pair = True

        for target_id in range(len(goals)):
            env = Reach2DEnv(
                n_distractors=n_distractors,
                seed=scene_seed,
                goal_positions=goals,
            )
            obs = env.reset(seed=scene_seed)
            signature = (obs["image"].copy(), obs["state"].copy(),
                         obs["goal_positions"].copy(), env.distractors.copy())
            if reference is None:
                reference = signature
            elif any(not np.array_equal(a, b) for a, b in zip(reference, signature)):
                raise RuntimeError("Paired demonstrations did not start from the same scene")

            obs_list, act_list, info = expert_policy(env, target_id=target_id)
            if (not info.get("success")
                    or info.get("collision")
                    or info.get("reached_target_id") != target_id
                    or not act_list):
                valid_pair = False
                break

            trajectory = []
            length = len(obs_list)
            for start in range(length):
                valid = min(chunk_size, length - start)
                actions = np.empty((chunk_size, 2), dtype=np.float32)
                actions[:valid] = np.asarray(act_list[start:start + valid], dtype=np.float32)
                actions[valid:] = actions[valid - 1]
                mask = np.zeros((chunk_size,), dtype=np.float32)
                mask[:valid] = 1.0
                trajectory.append({
                    "image": obs_list[start]["image"],
                    "state": obs_list[start]["state"],
                    "goal": obs_list[start]["goal"],
                    "action_chunk": actions,
                    "action_mask": mask,
                    "scene_id": accepted,
                    "scene_seed": scene_seed,
                    "route_mode": target_id,
                    "pair_start": start == 0,
                    "trajectory_length": length,
                })
            paired.extend(trajectory)

        if valid_pair:
            data.extend(paired)
            accepted += 1

        if progress_callback and (
                accepted == n_scenes or attempts % max(1, n_scenes // 20) == 0):
            progress_callback({"accepted_scenes": accepted, "target_scenes": n_scenes,
                               "attempts": attempts, "max_attempts": max_attempts})

    if accepted != n_scenes:
        raise RuntimeError(
            f"Collected {accepted}/{n_scenes} paired scenes after {attempts} attempts; "
            "check expert reachability or increase max_attempts"
        )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(data)} action chunks from {accepted} paired scenes → {save_path}")
    return data


class DemoDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        # image: (H, W, 3) uint8 → (3, H, W) float32 in [-1, 1]
        img = torch.from_numpy(d["image"]).permute(2, 0, 1).float() / 127.5 - 1.0
        state = torch.from_numpy(d["state"])
        goal = torch.from_numpy(d["goal"])
        # action_chunk: (H, 2) float32
        action = torch.from_numpy(d["action_chunk"])
        # Older demo pickles do not contain masks. Treat them as fully valid so
        # legacy checkpoints remain reproducible; fair experiments recollect
        # their data and therefore always use the true valid-action mask.
        action_mask = torch.from_numpy(
            d.get("action_mask", np.ones((action.shape[0],), dtype=np.float32))
        ).float()
        return {"image": img, "state": state, "goal": goal,
                "action": action, "action_mask": action_mask}


if __name__ == "__main__":
    data = collect_demos(n_demos=500, chunk_size=32, save_path="./data/demos.pkl")
    print(f"Total samples: {len(data)}")
    print(f"First sample shapes: image={data[0]['image'].shape}, "
          f"state={data[0]['state'].shape}, goal={data[0]['goal'].shape}, "
          f"action={data[0]['action_chunk'].shape}")
    n_zero = sum(1 for d in data if np.abs(d['action_chunk']).max() < 1e-6)
    print(f"全零 action chunk: {n_zero}（应当为 0——若不为 0 说明 padding 有问题）")
