# Project 5：VLA Action Diffusion 实验报告

## 摘要

本项目在 64×64 RGB 的 2D reaching 环境中实现了一个简化 VLA：视觉编码器读取 agent、target 和 distractors，条件 action head 一次生成一段连续二维动作，并在闭环中周期性重规划。完成了 TODO 19（action-chunk DDPM loss）、TODO 20（vision encoder）和 TODO 21（100 集闭环评估），并额外实现了 Flow Matching 与 Pure BC 对照。AutoDL RTX 4090 上最终模型使用 H=16、DDPM 20 步采样和每 4 步重规划，在固定测试集上取得 **71% 成功率、3% 碰撞率、26% 超时率**，达到题目要求的 70% 目标。

## 1. Design note

### 1.1 Action representation 与 chunk size

动作是环境坐标系中的 `(Δx, Δy) ∈ [-0.1, 0.1]^2`。模型不只预测下一步，而是预测 H 步 action chunk，评估时执行前 `exec_steps` 步后重新观察并采样。这样可以把短期运动的相关性作为一个整体学习，同时减少每个控制周期的采样开销。训练 demo 的轨迹中位数为 26 步；在本项目的重复末动作 padding 策略下，H=16 的 padding 为 27.5%，而 H=32 为 51.7%，因此最终选择 H=16。H=32 的实测成功率只有 14%，验证了过长 chunk 会稀释监督。

### 1.2 Vision encoder 与条件融合

视觉编码器是三层 stride=2 的 CNN（3→32→64→128），每层使用 GroupNorm + SiLU。第一版使用 `AdaptiveAvgPool2d(1)`，但它几乎消除了 target 和障碍的空间位置信息，视觉 DDPM 只有 54% 成功率。最终版本保留 `4×4` 特征网格，再用 Linear(2048, 128) 和 LayerNorm 投影；它仍然是小于 1M 参数的 toy encoder，但能表达“目标在左/右、障碍在何处”。视觉特征与 agent 自身的 2D state、时间 embedding 拼接后输入 action denoiser。

### 1.3 Scheduler 与训练目标

主模型使用 T=100 的 DDPM。给定 clean action chunk `a`、噪声 `ε` 和 `ᾱ_t`，训练构造

```text
a_t = sqrt(ᾱ_t) a + sqrt(1-ᾱ_t) ε
L_DDPM = || ε_θ(a_t, t, image, state) - ε ||_2^2
```

采样时从高斯噪声开始，用 20 个反向步得到长度 H 的 action chunk。作为加分项，Flow Matching 采用 `x_t=(1-t)ε+t a`、回归向量场 `a-ε`，用 10 步 Euler 积分；Pure BC 则固定零 action 输入，直接回归 clean action。

## 2. AutoDL 实验设置与可复现性

| 项目 | 设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24GB |
| CPU/RAM | 16 vCPU / 120GB |
| Python / PyTorch | 3.12.3 / 2.8.0+cu128 |
| demos | 1,000 条成功 expert 轨迹，训练 seed=42 |
| optimizer | Adam，lr=1e-3，weight decay=0，EMA=0.999 |
| steps / batch | 10,000 / 256 |
| evaluation | 100 episodes，seed 10000–10099，闭环 exec_steps=4 |

AutoDL 上的启动命令和只读 dashboard 见项目 README。训练、评估和 GPU 状态通过 tmux 与 `monitor_dashboard.py` 持续写入 `status.json`、`metrics.jsonl`；本地 SSH 隧道访问 `http://127.0.0.1:18765/`。原始训练曲线保存在 `results/metrics/`，曲线图为 [results/loss_curve.png](results/loss_curve.png)。

## 3. Ablation 结果

| 实验 | H | 成功率 | 碰撞率 | 超时率 | 平均步数 | 平均末端距离 |
|---|---:|---:|---:|---:|---:|---:|
| Pure BC | 16 | 58% | 12% | 30% | 50.73 | 0.3570 |
| DDPM + global pooling | 16 | 54% | 6% | 40% | 68.93 | 0.4114 |
| **DDPM + 4×4 spatial pooling（最终）** | **16** | **71%** | **3%** | **26%** | **48.70** | **0.1643** |
| Flow Matching + vision | 16 | 57% | 9% | 34% | 52.85 | 0.4804 |
| DDPM + vision，H=32 | 32 | 14% | 1% | 85% | 95.03 | 0.6302 |
| DDPM state-only | 16 | 81% | 12% | 7% | 55.89 | 0.1075 |

每一行都是同一个 100 集 seed 区间，不是训练集重放。最终 `eval_vision.json`、`eval_state.json`、`eval_bc.json` 和 `eval_fm.json` 为可直接读取的 JSON；`results/rollouts/` 中保留了 4 张成功轨迹图。

训练曲线显示 spatial encoder 的 moving-average loss 在 10k 步约 0.097，低于 global pooling 版本约 0.118。单独把 DDPM 每步重规划（exec_steps=1）并没有提升成功率（54%），但碰撞降至 2%，代价是推理调用次数约增加 4 倍；这说明视觉表征而非采样频率是本任务的主要瓶颈。

## 4. 自查问题

1. **为什么 action chunk？** 连续控制相邻动作高度相关，一次预测 H 步能学习局部轨迹并减少采样次数；每步预测虽然反馈最及时，但会增加约 H/`exec_steps` 倍的模型调用。`exec_steps=1` 能更快纠偏和降低碰撞，却牺牲实时性。

2. **多模态问题？** 遇到障碍时左绕和右绕都是合理解。Pure BC 的 MSE 会把两条轨迹平均成穿过障碍的动作；diffusion 从噪声采样，能够表示多峰的 action chunk 分布，并在闭环重规划中选择不同模式。

3. **条件与差值？** state-only 的 81% 比视觉 baseline 的 54% 高 27 个百分点，但 state-only 直接拿到 target 坐标且看不到障碍，不能把这 27 点解释成“视觉增益”。真正有意义的对照是相同 agent state 下 global pooling 与 spatial encoder 的 54%→71%；如果把 distractor 坐标加入 state，视觉 ablation 就失去辨识障碍视觉能力的意义。

4. **padding？** 轨迹中位数 26 步，H=32 时 51.7% 槽位是重复末动作，H=64 时 74.9%。过多重复监督会把策略拉向小/恒定动作，造成超时。改进方法是选择 H≈任务时间尺度（本实验 H=16），使用 mask/weighted loss，或按轨迹长度自适应 chunk。

5. **FM vs DDPM？** FM 直接学习从噪声到动作的连续向量场，推理只需少量 Euler 步，适合实时控制；DDPM 的离散反向链通常需要更多网络调用、但训练和调试更成熟。本实验 FM 10 步的成功率为 57%，尚未超过最终 DDPM，体现了表征与采样质量之间的 trade-off。

## 5. Reflection：从 toy VLA 到 Pi-0/OpenVLA 部署

这个 toy 环境没有体现工业机器人最昂贵的部分：相机和 proprioception 的时间同步、标定漂移、遮挡与光照变化、动作延迟、关节/力矩/速度约束、碰撞安全层、失败恢复和数据闭环。Pi-0/OpenVLA 还需要处理语言目标 grounding、预训练视觉 token 的 domain gap、长时任务的技能切换，以及在有限算力上的量化、缓存和延迟预算。真正部署时不能只看平均成功率；应记录每个场景的风险、置信度和 OOD 检测，给策略配一个可验证的安全控制器，并通过仿真回放、少量真实数据和人工接管逐步扩大覆盖。这个项目最有价值的经验是：空间表征、chunk 时间尺度和闭环评估必须一起设计，低 loss 本身并不等于可执行策略。

## 6. 结论

TODO 19–21 已完成并通过自检/测试。上一轮小型 spatial CNN + H=16 DDPM 的 71% 是先导实验数字，不是后续统一协议重跑的结果；新的公平对照与当前结果以第 8 节为准（DDPM H=16 为 81%）。此前 checkpoint 和 bonus rollout 仍作为历史实验产物保留。

## 7. 加分项实验与讨论

### 7.1 双目标多模态 demo

为验证模型是否能覆盖多个目标模式，`reach2d_multimodal.yaml` 将目标采样改为两个离散位置 `(-0.65, 0.55)` 与 `(0.65, 0.55)`，其余障碍、H=16 和 spatial CNN 均与最终模型一致。评估额外记录 `mode_stats`，而不是只报告总体均值。100 个未见 seed（20000–20099）得到 97% 成功率（97/100）、3% 碰撞、0% 超时；左目标 mode 0 为 50/52=96.2%，右目标 mode 1 为 47/48=97.9%。这说明 action diffusion 在这个设置下没有塌缩到单一目标，两个条件模式都可执行。结果与目标位置示意见 [results/bonus_summary.png](results/bonus_summary.png)。

### 7.2 ImageNet-pretrained ResNet18 对照

`model.py` 增加了真实 ImageNet 权重的 ResNet18 encoder：输入从 `[-1,1]` 转换到 ImageNet mean/std，最后的分类层替换为 128 维条件特征。它约 11.48M 参数，而最终 spatial CNN 约 0.59M 参数。为避免把偶然的短训结果误当结论，我保留了两个预算：5,000 steps 的成功率为 3%（碰撞 8%、超时 89%），15,000 steps 仍为 2%（碰撞 18%、超时 80%），评估 seed 均为 30000–30099。这个负结果是有价值的对照：在 64×64、小规模 demo 和从像素直接微调的条件下，预训练 backbone 的容量与归一化开销反而使优化困难；实际使用应继续比较冻结 backbone、较大数据集、分层学习率和更长训练，而不能只看参数量或“预训练”标签。

### 7.3 移动障碍泛化

`Reach2DEnv` 新增了有速度的 distractor：每个 step 更新位置并在边界反弹。使用静态障碍训练得到的最终 DDPM checkpoint，不重新训练，直接在速度 0.025 的动态测试环境评估 100 个 seed（40000–40099），成功率 72%、碰撞率 20%、超时率 8%，平均步数 34.94。与静态测试的 71%/3%/26% 相比，策略仍能到达目标，但碰撞明显增加，体现了观测延迟和训练分布变化带来的风险。该实验也保留了 6 张成功 rollout 图，便于复核行为。

三项加分实验的完整 JSON、metrics、日志和配置均已提交；`run_bonus_experiments.py` 可在 AutoDL 的 tmux 中按 preset 重新执行，`plot_bonus.py` 重新生成汇总图。Flow Matching 加分项已在第 1 节和主 ablation 中给出（10 步 Euler，57% 成功率）。

## 8. 更新后的公平重跑（单训练种子）

以下是修复随机种子、FM 时间编码、EMA buffers 与 DDIM schedule 后完成的公平对照；此前 Section 3 的结果属于不同协议下的先导实验，不能与此表直接混比。10 个条件均使用训练 seed 42、10,000 个优化步骤、同一 4×4 spatial CNN 与专家数据，并在固定测试 seeds 10000–10099 上运行 100 个闭环 episode（`exec_steps=4`）。成功率附 Wilson 95% episode 区间；因为只有一个训练种子，该区间不表示训练种子间方差或跨种子显著性。

| 条件 | 方法 | H | padding mask | 成功率（Wilson 95% CI） | 碰撞率 | 超时率 | 平均步数 | 平均末端距离 |
|---|---|---:|:---:|---:|---:|---:|---:|---:|
| BC H16 | BC | 16 | 是 | 88% (80.2–93.0%) | 9% | 3% | 26.95 | 0.0913 |
| DDPM H16 | DDPM | 16 | 是 | 81% (72.2–87.5%) | 8% | 11% | 35.57 | 0.1339 |
| FM H16 | FM | 16 | 是 | 87% (79.0–92.2%) | 7% | 6% | 29.53 | 0.0987 |
| DDPM H8 | DDPM | 8 | 是 | 89% (81.4–93.7%) | 7% | 4% | 27.57 | 0.0994 |
| DDPM H32 | DDPM | 32 | 是 | 48% (38.5–57.7%) | 6% | 46% | 64.39 | 0.2621 |
| DDPM H64 | DDPM | 64 | 是 | 26% (18.4–35.4%) | 1% | 73% | 85.86 | 0.3668 |
| DDPM H8 | DDPM | 8 | 否 | 91% (83.8–95.2%) | 7% | 2% | 26.37 | 0.0919 |
| DDPM H16 | DDPM | 16 | 否 | 85% (76.7–90.7%) | 8% | 7% | 31.50 | 0.1115 |
| DDPM H32 | DDPM | 32 | 否 | 66% (56.3–74.5%) | 7% | 27% | 54.73 | 0.2138 |
| DDPM H64 | DDPM | 64 | 否 | 28% (20.1–37.5%) | 9% | 63% | 81.96 | 0.4147 |

在这次预算下，BC 与 FM 的 H16 成功率高于 DDPM H16；因此目前不能声称 diffusion 在此任务上优于 BC。H=8 表现最好，而 H=32/64 的超时随 chunk 变长明显增加，符合过长动作序列削弱闭环纠偏的预期。Padding mask 在本次单种子结果中没有带来提升，H16/32 的 masked 结果反而较低；这需要更多训练种子和损失/数据诊断才能判断原因，不应据此宣称 mask 普遍有害。机器可读完整结果见 [`results/fair_seed42/fair_summary.json`](results/fair_seed42/fair_summary.json) 与 [`results/fair_seed42/fair_ablation.md`](results/fair_seed42/fair_ablation.md)。
