"""环境与配置自检：确认 env、数据、模型、TODO 都接得上。

动手之前先跑（半分钟，不需要 GPU）：

    python check_setup.py

它会检查：
1. 依赖包
2. env 能否 reset/step/render，observation 里的 image/state/goal 是否齐全
3. 专家策略能否产出成功轨迹（不成功就没有训练数据）
4. 采一小批 demo，检查 action chunk 没有全零（零填充会让 policy 学成不动）
5. 按每个 config 构建模型并前向，state_dim 与条件模式是否一致
6. TODO 19/20/21 的状态

TODO 显示 ⏳ 是预期的——那正是你要写的部分。
"""

import argparse
import importlib
import sys
from pathlib import Path

try:
    # Keep the diagnostic script usable in Windows consoles whose legacy
    # code page cannot encode the check-mark characters below.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

PACKAGES = [('torch', 'torch'), ('numpy', 'numpy'), ('yaml', 'PyYAML'),
            ('tqdm', 'tqdm'), ('matplotlib', 'matplotlib'),
            ('torchvision', 'torchvision')]


def check_packages():
    missing = []
    for mod, pip_name in PACKAGES:
        try:
            importlib.import_module(mod)
            print(f"   ✅ {pip_name}")
        except ImportError:
            missing.append(pip_name)
            print(f"   ❌ {pip_name} 未安装")
    if missing:
        print("     pip install " + " ".join(missing))
    return not missing


def check_env():
    import numpy as np
    from env import Reach2DEnv, expert_policy

    env = Reach2DEnv(n_distractors=2, seed=42)
    obs = env.reset()

    for key, shape in [("image", (64, 64, 3)), ("state", (2,)), ("goal", (2,))]:
        if key not in obs:
            print(f"   ❌ observation 缺少 '{key}'")
            return False
        if tuple(np.shape(obs[key])) != shape:
            print(f"   ❌ obs['{key}'] shape {np.shape(obs[key])} != {shape}")
            return False
    print(f"   ✅ observation: image{obs['image'].shape} state{obs['state'].shape} "
          f"goal{obs['goal'].shape}")

    obs2, r, done, info = env.step(np.array([0.5, 0.5]))
    if not {"success", "collision", "step_count"} <= set(info):
        print(f"   ❌ info 缺字段：{sorted(info)}")
        return False
    print(f"   ✅ step() OK，info 字段齐全")

    # 专家策略的成功率——太低的话根本收不到训练数据
    n_ok = 0
    for i in range(20):
        e = Reach2DEnv(n_distractors=2, seed=100 + i)
        e.reset()
        _, _, info = expert_policy(e)
        n_ok += int(info["success"])
    rate = n_ok / 20
    if rate < 0.3:
        print(f"   ❌ 专家成功率只有 {rate:.0%}，采不到足够 demo")
        return False
    print(f"   ✅ 专家策略成功率 {rate:.0%}（20 次试跑）")
    return True


def check_demos():
    import numpy as np
    from dataset import collect_demos

    data = collect_demos(n_demos=5, chunk_size=16, n_distractors=2, save_path=None, seed=0)
    if not data:
        print("   ❌ 一条 demo 都没采到")
        return False

    n_zero = sum(1 for d in data if np.abs(d["action_chunk"]).max() < 1e-6)
    keys_ok = {"image", "state", "goal", "action_chunk", "action_mask"} <= set(data[0])
    if not keys_ok:
        print(f"   ❌ 样本缺字段：{sorted(data[0])}")
        return False
    if n_zero:
        print(f"   ❌ {n_zero}/{len(data)} 个 action chunk 全是零 —— "
              f"padding 用了零填充，policy 会学成不动")
        return False
    masks_ok = all(d["action_mask"].shape == (16,)
                   and set(np.unique(d["action_mask"])) <= {0.0, 1.0}
                   for d in data)
    has_padding = any((d["action_mask"] == 0).any() for d in data)
    if not masks_ok or not has_padding:
        print("   ❌ action_mask 无效或没有标出尾部 padding")
        return False
    print(f"   ✅ {len(data)} 个样本，action/action_mask 齐全，无全零 chunk")
    return True


def check_model(cfg_path):
    import torch
    from model import DiffusionPolicy
    from obs_utils import state_dim_for
    from train import load_config

    cfg = load_config(cfg_path)
    use_vision = cfg["use_vision"]
    sd = state_dim_for(use_vision)

    try:
        model = DiffusionPolicy(
            horizon=cfg["chunk_size"], action_dim=2, state_dim=sd, image_size=64,
            vision_out_dim=cfg["vision_out_dim"], hidden=cfg["hidden"],
            use_vision=use_vision,
            vision_encoder=cfg.get("vision_encoder", "small"),
            # Setup should be offline-safe; the actual bonus training command
            # enables pretrained weights explicitly.
            vision_pretrained=False,
        )
    except NotImplementedError:
        print(f"   ⏳ {cfg_path.name}: use_vision=true，等 TODO 20 完成（预期）")
        return None
    except Exception as e:
        print(f"   ❌ {cfg_path.name}: 构建失败 → {type(e).__name__}: {e}")
        return False

    B, H = 2, cfg["chunk_size"]
    a = torch.randn(B, H, 2)
    t = torch.randint(0, cfg["diffusion_steps"], (B,))
    img = torch.randn(B, 3, 64, 64) if use_vision else None
    try:
        with torch.no_grad():
            eps = model(a, t, image=img, state=torch.randn(B, sd))
    except Exception as e:
        print(f"   ❌ {cfg_path.name}: 前向失败 → {type(e).__name__}: {e}")
        return False

    if eps.shape != a.shape:
        print(f"   ❌ {cfg_path.name}: 输出 {tuple(eps.shape)} != 输入 {tuple(a.shape)}")
        return False
    n = sum(p.numel() for p in model.parameters()) / 1e6
    mode = "image + agent_pos" if use_vision else "agent_pos + target_pos"
    print(f"   ✅ {cfg_path.name}: {n:.2f}M params, state_dim={sd}, 条件={mode}")
    return model


def check_todos(model, cfg_path):
    import torch
    from obs_utils import state_dim_for
    from train import DDPMScheduler, diffusion_loss, load_config
    from eval import evaluate

    cfg = load_config(cfg_path)
    use_vision = cfg["use_vision"]
    sched = DDPMScheduler(T=cfg["diffusion_steps"], device="cpu",
                          schedule=cfg.get("noise_schedule", "linear"))
    H = cfg["chunk_size"]
    batch = {"image": torch.randn(2, 3, 64, 64), "state": torch.randn(2, 2),
             "goal": torch.randn(2, 2), "action": torch.randn(2, H, 2),
             "action_mask": torch.ones(2, H)}

    checks = [
        ("TODO 19 (diffusion_loss)",
         lambda: diffusion_loss(model, batch, sched, "cpu", use_vision,
                                cfg.get("mask_padding", False))),
        ("TODO 21 (evaluate)",
         lambda: evaluate(model, sched, cfg, "cpu", n_episodes=2,
                          exec_steps=4, chunk_size=H, n_sample_steps=4)),
    ]
    done = True
    for label, fn in checks:
        try:
            out = fn()
            print(f"   ✅ {label}: {out if not hasattr(out, 'shape') else tuple(out.shape)}")
        except NotImplementedError:
            print(f"   ⏳ {label}: 未完成（预期）")
            done = False
        except Exception as e:
            print(f"   ❌ {label}: {type(e).__name__}: {e}")
            done = False
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_demos", action="store_true",
                        help="跳过采 demo 的检查（稍慢的一步）")
    args = parser.parse_args()

    print("[1/5] 依赖包")
    if not check_packages():
        return 1

    print("\n[2/5] 环境 env.py")
    if not check_env():
        return 1

    print("\n[3/5] demo 采集")
    if not args.skip_demos and not check_demos():
        return 1

    print("\n[4/5] 按 config 构建模型")
    cfgs = sorted(Path("configs").glob("*.yaml"))
    models = {}
    for p in cfgs:
        m = check_model(p)
        if m is False:
            return 1
        if m is not None:
            models[p] = m

    if not models:
        print("\n所有 config 都要求 TODO 20 —— 先做 TODO 20，或用 "
              "configs/reach2d_state.yaml 走阶段 2")
        return 0

    print("\n[5/5] TODO 状态")
    p, m = next(iter(models.items()))
    done = check_todos(m, p)

    print()
    print("全部就绪 ✅" if done else "环境与模型 OK ✅，TODO 还没写完 —— 去动手吧")
    return 0


if __name__ == "__main__":
    sys.exit(main())
