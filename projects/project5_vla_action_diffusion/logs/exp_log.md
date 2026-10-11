# Project 5 AutoDL experiment log

## 实验编号与目标

`exp_20260914_01_vla_action_diffusion`

目标：在未见 seed 的 2D reaching 环境上验证视觉条件 action diffusion 是否达到 70% 闭环成功率，并比较 state-only、Pure BC、Flow Matching、chunk 长度和视觉空间表征。

## 1. 环境信息

| 项目 | 内容 |
|---|---|
| 日期 | 2026-09-14（Asia/Shanghai） |
| 硬件 | AutoDL NVIDIA GeForce RTX 4090 24GB，16 vCPU，120GB RAM |
| 软件 | Ubuntu 22.04.5，Python 3.12.3，PyTorch 2.8.0+cu128，CUDA 可用 |
| 仓库 | `https://github.com/li-cheng111/my-diffusion-models-starter` |
| 分支 | `codex/project5-vla-action-diffusion` |
| 训练 seed | 42 |
| 评估 seed | 10000–10099（100 episodes） |
| 代码提交 | 空间 encoder + 结果 artifact：`0e71913e3334e7304c8906c3f660cdc178924743` |

## 2. 假设与预期

- H=16 的 padding 应明显少于 H=32，闭环更稳定。
- 视觉 DDPM 应优于 state-only 之外的无视觉基线；保留空间特征可能比 global average pooling 更能定位 target。
- FM 的采样调用更少，但在相同训练步数下不保证成功率超过 DDPM。

## 3. 完整运行配置

主配置：`configs/reach2d_16.yaml`；H=16、1000 demos、batch=256、lr=1e-3、T=100、EMA=0.999、10000 steps。空间版仅替换 `VisionEncoder` 的末端为 4×4 adaptive pooling，其他超参不变。

## 4. 运行命令

```bash
python check_setup.py
pytest -q tests
python -u run_experiments.py --preset full > logs/full_run.log 2>&1
python -u train.py --config configs/reach2d_16.yaml --method ddpm \
  --run-dir runs/project5/vision_ddpm_spatial --seed 42
python -u eval.py --config configs/reach2d_16.yaml \
  --ckpt runs/project5/vision_ddpm_spatial/ckpts/model_final.pt \
  --method ddpm --n_episodes 100 --seed-start 10000 --exec_steps 4 \
  --n_sample_steps 20 --output runs/project5/vision_ddpm_spatial/eval_spatial.json
```

训练与评估运行在 AutoDL tmux，`monitor_dashboard.py` 只读暴露 loss、GPU、状态和结果；SSH 隧道映射到本地 18765 端口。

## 5. 结果

完整矩阵用时约 17 分钟（五组训练各约 2.4 分钟，评估和数据采集占其余时间）；空间版重新训练约 2.5 分钟。最终空间视觉 DDPM：success=71/100，collision=3/100，timeout=26/100，avg_steps=48.70，mean_final_distance=0.1643。Global pooling 对照为 54%，exec_steps=1 仍为 54%；state-only 为 81%，Pure BC 为 58%，FM 为 57%，H=32 为 14%。

训练曲线：`results/loss_curve.png`；成功 rollout：`results/rollouts/success_ep*.png`。原始指标 JSON 和 JSONL 保存在 `results/`。

## 6. 结论与下一步

空间 encoder 假设成立，最终达到 70% 目标。后续应加入 action mask、更多随机光照/障碍分布和多 seed 置信区间，再评估 FM 的更长训练是否能追平 DDPM。

## Bonus experiments (AutoDL, 2026-09-14)

- **Multimodal**: `reach2d_multimodal.yaml`, two target positions, 10k steps. Eval seeds 20000–20099: success 97%, collision 3%, timeout 0%; per-mode 96.2%/97.9%.
- **ImageNet ResNet18**: `reach2d_resnet18.yaml`, pretrained weights, 5k steps. Eval seeds 30000–30099: 3% success, 8% collision, 89% timeout. A 15k-step run remained 2%/18%/80%; retained as an honest negative comparison against the 0.59M-parameter spatial CNN.
- **Moving distractors**: `reach2d_moving.yaml`, static-trained final checkpoint, distractor speed 0.025. Eval seeds 40000–40099: success 72%, collision 20%, timeout 8%.
- Reproduction: `python run_bonus_experiments.py --preset all`; summary figure: `python plot_bonus.py`.
