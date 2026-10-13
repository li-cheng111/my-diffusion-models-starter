# Project 5：视觉条件动作扩散（VLA Action Diffusion）

本项目在二维目标到达环境中实现并评估视觉条件动作策略，比较行为克隆（BC）、去噪扩散（DDPM）与 Flow Matching（FM），并研究动作块长度、填充掩码、视觉编码器和环境变化。本文档以当前代码、配置和结果为准；方法推导与实验记录见文末索引。

## 任务与实现

`Reach2DEnv` 将场景渲染为 64×64 RGB 图像，包含智能体、目标和两个障碍物。视觉策略接收图像与智能体位置；目标和障碍物坐标不作为显式状态输入。state-only 对照接收智能体和目标坐标，不能视为视觉策略的等价基线。

策略预测长度为 `H` 的二维动作块，每执行前 4 步后重新观测和规划。动作先裁剪到 `[-1, 1]²`，环境再乘以 `ACTION_SCALE=0.05` 更新位置；目标距离小于 `0.06` 判成功，智能体与障碍物距离小于 `0.10` 判碰撞，每回合最多 100 步。

视觉编码器为三层 stride-2 CNN（通道数 3→32→64→128），每层使用 GroupNorm 与 SiLU；4×4 自适应池化保留空间信息，再映射为 128 维特征。状态、图像和时间嵌入共同条件化动作生成器；动作块展平后由隐藏宽度为 256 的 MLP 预测噪声、速度或动作。DDPM 使用余弦噪声调度；BC 直接回归动作块，FM 回归速度场。变长轨迹末尾重复最后一个动作补齐，并记录有效动作掩码。

## 主结果：单训练种子公平对照

10 个条件均使用训练 seed 42、同一批 1,000 条成功专家轨迹（按各自 H 构造动作块）、4×4 spatial CNN、余弦噪声调度、10,000 次更新和相同的 100 个评估 episode（环境 seeds 10000–10099）。batch size 为 256，优化器为 AdamW，learning rate 为 1e-3，weight decay 为 0，EMA decay 为 0.999。每次执行 4 步后重新规划；DDPM 使用 20 步 DDIM 采样，FM 使用 10 步 Euler 积分。表中 checkpoint 于 2026-10-13 从 [Project 5 fair seed 42 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-fair-seed42-v1.0.0) 恢复，并用代码提交 `c062751` 重评估；没有重新训练。成功率区间为 95% Wilson 区间，只反映评估 episode 抽样不确定性，不表示训练 seed 间方差。

| 方法 | H | 填充掩码 | 成功率（95% Wilson CI） | 碰撞率 | 超时率 | 平均步数 | 末端距离 | 沿途目标距离 |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| BC | 16 | 是 | 88%（80.2–93.0%） | 9% | 3% | 26.95 | 0.0913 | 0.4877 |
| DDPM | 16 | 是 | 81%（72.2–87.5%） | 8% | 11% | 35.57 | 0.1339 | 0.4675 |
| FM | 16 | 是 | 87%（79.0–92.2%） | 7% | 6% | 29.53 | 0.0987 | 0.4736 |
| DDPM | 8 | 是 | 89%（81.4–93.7%） | 7% | 4% | 27.57 | 0.0994 | 0.4852 |
| DDPM | 32 | 是 | 48%（38.5–57.7%） | 6% | 46% | 64.39 | 0.2621 | 0.4612 |
| DDPM | 64 | 是 | 26%（18.4–35.4%） | 1% | 73% | 85.86 | 0.3668 | 0.5033 |
| DDPM | 8 | 否 | **91%（83.8–95.2%）** | 7% | 2% | 26.37 | 0.0919 | 0.4887 |
| DDPM | 16 | 否 | 85%（76.7–90.7%） | 8% | 7% | 31.50 | 0.1115 | 0.4769 |
| DDPM | 32 | 否 | 66%（56.3–74.5%） | 7% | 27% | 54.73 | 0.2138 | 0.4716 |
| DDPM | 64 | 否 | 28%（20.1–37.5%） | 9% | 63% | 81.96 | 0.4147 | 0.5955 |

- 主矩阵最高点估计为 DDPM、H=8、未屏蔽填充动作的 91%；每个条件只有一个训练 seed，不能据此断言方法间存在显著差异。
- H=16 时 BC、FM、DDPM masked 分别为 88%、87%、81%；本实验没有显示 DDPM 优于 BC。
- H=32/64 的超时率明显升高。四种 H 下 masked 的成功率点估计均低于 unmasked，但单种子结果不足以证明掩码普遍有害。

## 先导与扩展实验

以下来自不同阶段或配置，不能与上方公平矩阵直接混作同一对照。

| 实验 | 设置 | 成功 / 碰撞 / 超时 | 说明 |
|---|---|---:|---|
| 先导视觉 DDPM | H=16，全局池化 | 54% / 6% / 40% | 早期编码器对照 |
| 先导视觉 DDPM | H=16，4×4 空间池化 | 71% / 3% / 26% | 早期协议，与上方公平矩阵不同 |
| 先导 BC / FM | H=16 | BC 58% / 12% / 30%；FM 57% / 9% / 34% | 先导算法对照 |
| state-only DDPM | H=16，显式输入目标坐标 | 81% / 12% / 7% | 含目标位置信息，不等价于视觉条件 |
| 先导 DDPM | H=32 | 14% / 1% / 85% | 早期配置，不与公平矩阵 H=32 混比 |
| 双目标条件 | H=16，两个固定目标位置 | 97% / 3% / 0% | 两目标成功率 96.2%（50/52）与 97.9%（47/48） |
| ImageNet ResNet-18 | 5,000 步 | 3% / 8% / 89% | seeds 30000–30099 |
| ImageNet ResNet-18 | 15,000 步 | 2% / 18% / 80% | 延长训练未改善该设置 |
| 移动障碍评估 | 静态训练 checkpoint，障碍速度 0.025 | 72% / 20% / 8% | 未在移动环境重新训练；碰撞率上升 |

双目标实验表明模型能完成两种图像中的目标条件，但目标在每回合观测中已明确，不能证明同一观测下生成两类动作轨迹。先导数据中的重复填充比例约为 H=8：12.9%、H=16：27.5%、H=32：51.7%、H=64：74.9%，仅作历史背景参考。

![先导训练损失曲线](results/loss_curve.png)

![扩展实验结果汇总](results/bonus_summary.png)

## 复现

评估同时报告末端距离（FDE）与整段轨迹的目标距离：`mean_path_target_distance` 先对单条 rollout 的所有已访问位置（包括初始位置）计算到目标的欧氏距离均值，再跨 episode 求均值。它反映整个过程的目标接近程度，不是相对专家轨迹的 ADE；因此不能解释为轨迹模仿误差。碰撞与到达若在同一步同时发生，按碰撞失败计数。

实验日志记录的软件环境为 Python 3.12.3、PyTorch 2.8.0+cu128。请先按本机平台安装匹配的 PyTorch 与 torchvision，再在项目目录安装其余依赖并运行自检：

```bash
python -m pip install pyyaml tqdm matplotlib numpy
python check_setup.py
```

先运行短 smoke 对照，再运行完整公平矩阵：

```bash
python run_fair_experiments.py --preset algorithms --seed 42 --episodes 5 --steps 20 --run-root runs/fair_smoke
python run_fair_experiments.py --preset full --seed 42 --episodes 100 --steps 10000 --run-root runs/project5_fair
```

已有 checkpoint 或结果时，可在完整命令末尾加 `--resume` 跳过已完成项。缺少公平实验 demo 文件时，脚本会调用训练入口采集数据。扩展实验与汇总图：

```bash
python run_bonus_experiments.py --preset all
python plot_bonus.py
```

移动障碍 preset 使用仓库中的 `ckpts/model_final.pt`，不在动态环境重新训练。该 checkpoint 属于早期实验；公平矩阵运行时 checkpoint 写入 `runs/`，已提交结果摘要位于 `results/fair_seed42/`。可选的 `monitor_dashboard.py` 用于查看运行状态；`launch_after_gpu_idle.sh` 面向 AutoDL，会等待 Project 4 相关进程结束及 GPU 空闲。

## 文件索引

- `env.py`、`dataset.py`、`obs_utils.py`：环境、专家数据与观测条件处理。
- `model.py`、`train.py`、`eval.py`：策略结构、训练与闭环评估；`configs/`：实验配置。
- [公平矩阵汇总](results/fair_seed42/fair_ablation.md)；[公平矩阵指标](results/fair_seed42/fair_summary.json)；[先导消融](results/ablation.md)；[实验日志](logs/exp_log.md)。
- `results/rollouts/`、`results/bonus_rollouts/`：示例轨迹；`results/*.json`：实验指标。
- [公平实验方案](FAIR_EXPERIMENT_PLAN.md)：对照协议；[调试记录](debug_log.md)：实现与运行问题。
- [实验日志模板](experiment_log_template.md)是通用模板，其中示例数据不是本项目结果。

## 从 GitHub Release 恢复并重评估

训练好的公平矩阵 checkpoint 与原始结果作为 GitHub Release 附件提供，避免把大型二进制文件塞进 Git 历史。克隆本仓库 `main` 后，在项目目录执行以下命令即可从 Release 恢复模型，并用当前版本评估代码重新跑 100 个固定 episode；新结果写到独立目录，不会覆盖发布时的原始结果：

```bash
mkdir -p assets/release assets/restored
cd assets/release
base=https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project5-fair-seed42-v1.0.0
curl -fLO "$base/SHA256SUMS"
curl -fLO "$base/project5-fair-seed42-checkpoints.tar.gz"
curl -fLO "$base/project5-fair-seed42-results.tar.gz"
curl -fLO "$base/project5-vla-action-diffusion-source-14a93fd.tar.gz"
sha256sum -c SHA256SUMS
cd ../..
tar -xzf assets/release/project5-fair-seed42-checkpoints.tar.gz -C assets/restored
python eval_checkpoint_archive.py --checkpoint-root assets/restored/runs/project5_fair --run-root runs/project5_release_eval --episodes 100 --seed-start 10000
```

`eval_checkpoint_archive.py --resume` 会跳过已经生成 `eval.json` 的条件，适合断线后接续；正式复现训练矩阵则使用上一节的 `run_fair_experiments.py` 命令。`SHA256SUMS` 同时校验 checkpoint、结果和 source 三个 Release 附件。Release source 是当时的源码快照；实验应以当前克隆的 `main` 作为评估代码。

## 限制

这是无摩擦、无动力学的二维点质量环境，不代表真实机器人控制能力。公平矩阵只有一个训练 seed；Wilson 区间只量化评估 episode 的不确定性。移动障碍实验的碰撞率升至 20%，静态训练策略在环境变化下仍有安全局限。
