# Project 5：视觉条件动作扩散（VLA Action Diffusion）

本项目在二维目标到达环境中实现并评估视觉条件动作策略，比较行为克隆（BC）、去噪扩散（DDPM）与 Flow Matching（FM），并研究动作块长度、填充掩码、视觉编码器和环境变化。本 README 同时作为 Project 5 实验报告，按原任务要求补入设计说明、闭环对照、训练曲线、自查题和工程反思。公平复评的视觉 DDPM H=16 masked 成功率为 81%（100 个 episode，单训练 seed），达到题目提出的 70% 参考目标；但结果不支持“DDPM 必然优于 BC”，也不能据单 seed 推断方法间显著差异。本文档以当前代码、配置和结果为准；方法推导与实验记录见文末索引。

## 任务与实现

`Reach2DEnv` 将场景渲染为 64×64 RGB 图像，包含智能体、目标和两个障碍物。视觉策略接收图像与智能体位置；目标和障碍物坐标不作为显式状态输入。state-only 对照接收智能体和目标坐标，不能视为视觉策略的等价基线。

策略预测长度为 `H` 的二维动作块，每执行前 4 步后重新观测和规划。动作先裁剪到 `[-1, 1]²`，环境再乘以 `ACTION_SCALE=0.05` 更新位置；目标距离小于 `0.06` 判成功，智能体与障碍物距离小于 `0.10` 判碰撞，每回合最多 100 步。

视觉编码器为三层 stride-2 CNN（通道数 3→32→64→128），每层使用 GroupNorm 与 SiLU；4×4 自适应池化保留空间信息，再映射为 128 维特征。状态、图像和时间嵌入共同条件化动作生成器；动作块展平后由隐藏宽度为 256 的 MLP 预测噪声、速度或动作。DDPM 使用余弦噪声调度；BC 直接回归动作块，FM 回归速度场。变长轨迹末尾重复最后一个动作补齐，并记录有效动作掩码。

### 训练目标

令 $o$ 表示当前图像与允许输入的状态，$\mathbf{a}_0=(a_0,\ldots,a_{H-1})$ 表示一段动作块。DDPM 对动作块加噪并预测噪声：

$$
\begin{aligned}
\mathbf{a}_t &= \sqrt{\bar\alpha_t}\,\mathbf{a}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon, \\
\mathcal{L}_{\mathrm{DDPM}} &= \mathbb{E}_{t,\mathbf{a}_0,\boldsymbol\epsilon,o} \\
&\quad \left[\left\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{a}_t,t,o)\right\|_2^2\right].
\end{aligned}
$$

BC 直接回归动作块；FM 在高斯噪声与数据动作之间线性插值，并学习从噪声指向动作的速度场。开启 padding mask 时，三种损失均只对有效动作槽位计入均方误差。

$$
\begin{aligned}
\mathcal{L}_{\mathrm{BC}} &= \mathbb{E}\left[\|f_\theta(o)-\mathbf{a}_0\|_2^2\right], \\
\mathbf{a}_t &= (1-t)\mathbf{z}+t\mathbf{a}_0, \\
\mathcal{L}_{\mathrm{FM}} &= \mathbb{E}\left[\|v_\theta(\mathbf{a}_t,t,o)-(\mathbf{a}_0-\mathbf{z})\|_2^2\right].
\end{aligned}
$$

### 设计决策

- **动作块长度**：先以 $H=16$ 作为基线，理由是专家轨迹中位数约 26 步，$H=16$ 的重复尾部占比约 27.5%，在规划跨度和有效监督之间折中。后续公平 sweep 显示本环境的最佳观测点估计出现在 $H=8$；这修正了初始选择，不代表单种子下已证明 $H=8$ 普遍最优。
- **视觉编码器**：采用小型 CNN 和 4×4 空间池化。目标与障碍的位置关系对避障至关重要；先导对照中空间池化比全局池化高 17 个百分点（71% vs. 54%）。预训练 ResNet-18 参数更多，但在当前小数据、低分辨率任务上未带来更好的闭环结果。
- **生成方法与采样**：DDPM 使用 $T=100$ 的余弦训练调度，并以 20 步 DDIM 做闭环采样；FM 作为独立生成目标，用 10 步 Euler 积分采样。选择少步数采样以控制重规划成本；当前只比较了采样步数和任务结果，没有测量端到端延迟，因此不把步数比例写成实际加速倍数。
- **条件信息**：视觉策略输入 RGB 图像和智能体位置；state-only 基线显式输入目标坐标但不输入障碍坐标。两者回答的问题不同，不能把 state-only 当作视觉模型的同信息量替代品。

## 主结果：单训练种子公平对照

10 个条件均使用训练 seed 42、同一批 1,000 条成功专家轨迹（按各自 H 构造动作块）、4×4 spatial CNN、余弦噪声调度、10,000 次更新和相同的 100 个评估 episode（环境 seeds 10000–10099）。batch size 为 256，优化器为 AdamW，learning rate 为 1e-3，weight decay 为 0，EMA decay 为 0.999。每次执行 4 步后重新规划；DDPM 使用 20 步 DDIM 采样，FM 使用 10 步 Euler 积分。表中 checkpoint 于 2026-10-13 从 [Project 5 fair seed 42 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-fair-seed42-v1.0.0) 恢复，并用代码提交 `c062751` 重评估；没有重新训练。

| 方法 | H | 填充掩码 | 成功率 | 碰撞率 | 超时率 | 平均步数 | 末端距离 | 沿途目标距离 |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| BC | 16 | 是 | 88% | 9% | 3% | 26.95 | 0.0913 | 0.4877 |
| DDPM | 16 | 是 | 81% | 8% | 11% | 35.57 | 0.1339 | 0.4675 |
| FM | 16 | 是 | 87% | 7% | 6% | 29.53 | 0.0987 | 0.4736 |
| DDPM | 8 | 是 | 89% | 7% | 4% | 27.57 | 0.0994 | 0.4852 |
| DDPM | 32 | 是 | 48% | 6% | 46% | 64.39 | 0.2621 | 0.4612 |
| DDPM | 64 | 是 | 26% | 1% | 73% | 85.86 | 0.3668 | 0.5033 |
| DDPM | 8 | 否 | **91%** | 7% | 2% | 26.37 | 0.0919 | 0.4887 |
| DDPM | 16 | 否 | 85% | 8% | 7% | 31.50 | 0.1115 | 0.4769 |
| DDPM | 32 | 否 | 66% | 7% | 27% | 54.73 | 0.2138 | 0.4716 |
| DDPM | 64 | 否 | 28% | 9% | 63% | 81.96 | 0.4147 | 0.5955 |

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

### 阶段 2 与阶段 3：输入条件对照

下表是最初的阶段对照，不是后文 10 条件公平矩阵的一部分。两者都在 100 个固定评估 episode（seeds 10000–10099）上闭环评估，均为 H=16、每 4 步重规划、DDPM 20 步采样；阶段 2 输入智能体与目标坐标，阶段 3 输入图像与智能体坐标。

| 阶段 | 条件 | 成功率 | 碰撞率 | 超时率 | 平均步数 | 平均末端距离 |
|---|---|---:|---:|---:|---:|---:|
| 2：state-only | agent + target 坐标 | 81% | 12% | 7% | 55.89 | 0.1075 |
| 3：视觉 DDPM（4×4） | RGB 图像 + agent 坐标 | 71% | 3% | 26% | 48.70 | 0.1643 |

阶段 3 的成功率比阶段 2 低 10 个百分点，但碰撞率低 9 个百分点、超时率高 19 个百分点。视觉策略必须从像素中同时辨认目标与障碍，state-only 策略则直接获得目标坐标，因此这个阶段对照衡量的是**两种条件接口下的整体策略差异**，不能单独解释为“视觉让成功率提高/降低”。若给 state-only 输入障碍物坐标，它会成为带特权几何信息的状态基线，也不再回答视觉策略是否能从图像辨认场景的问题。之后的公平矩阵固定使用视觉条件，在相同数据与评估集上比较算法、H 和 padding mask；其中 DDPM H=16 masked 的成功率为 81%。

### 训练曲线与收敛分析

![先导训练损失曲线](results/loss_curve.png)

图：先导运行中每 100 步移动平均 loss；这些曲线不是 10 条件公平矩阵的训练曲线。4×4 spatial DDPM 的末段 loss 为 0.0968，低于全局池化 DDPM 的 0.1177；state DDPM 为 0.1218，FM 为 0.1338。各曲线整体下降并在后段趋稳，但 loss 只度量去噪/回归目标，不等价于闭环成功率。全局池化与空间池化的末段 loss 差距有限，闭环成功率却相差 17 个百分点，说明保留物体位置关系对控制更关键；H=32 的失败也说明单看 loss 可能掩盖过长动作块造成的超时。

ImageNet ResNet-18 加分实验的训练日志还提供了步速和末段 loss。以下是 AutoDL 运行日志中的约数；步数、训练预算及配置不同，不能当作严格的同预算收敛速度基准。

| 编码器 | 参数量 | 更新步数 | 日志末段吞吐 | 末段 100-step 平均 loss | 闭环成功率 |
|---|---:|---:|---:|---:|---:|
| 小型 4×4 spatial CNN | 0.59M | 10,000 | 65.9 step/s | 0.0968 | 71% |
| ImageNet ResNet-18 | 11.48M | 5,000 | 63.8 step/s | 0.1168 | 3% |
| ImageNet ResNet-18（延长） | 11.48M | 15,000 | 67.0 step/s | 0.0950 | 2% |

延长 ResNet-18 训练使 loss 降到与小型 CNN 相近的水平，但闭环成功率没有改善。现有证据支持“更多预训练参数并未自动转化为本任务的控制收益”；由于没有使用统一停止阈值和匹配 wall-clock 预算，本实验只能算有实测吞吐的描述性对照，不能声称已严格测定两种编码器的收敛速度。

### 成功闭环轨迹

以下展示四个成功 rollout。蓝线为策略执行轨迹，圆点为起点，红星为目标，灰色方块为障碍物。采用 2×2 图组，便于横向比较不同场景中的闭环路径；更多轨迹见 `results/rollouts/` 和 `results/bonus_rollouts/`。

<table>
  <tr>
    <td align="center"><img src="results/rollouts/success_ep000.png" alt="成功闭环轨迹 episode 0" width="420"/><br/><sub>Episode 0</sub></td>
    <td align="center"><img src="results/rollouts/success_ep001.png" alt="成功闭环轨迹 episode 1" width="420"/><br/><sub>Episode 1</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="results/rollouts/success_ep002.png" alt="成功闭环轨迹 episode 2" width="420"/><br/><sub>Episode 2</sub></td>
    <td align="center"><img src="results/rollouts/success_ep003.png" alt="成功闭环轨迹 episode 3" width="420"/><br/><sub>Episode 3</sub></td>
  </tr>
</table>

![扩展实验结果汇总](results/bonus_summary.png)

## 对照原始任务要求的自查与反思

### 自查问题

1. **为什么预测动作块？** 一次预测 $H$ 步能减少每个环境步都重新调用生成模型的开销，并让局部动作保持连贯；每执行 `exec_steps` 步再观察和规划，则保留闭环纠偏能力。若 `exec_steps=1`，反馈更及时，但会增加采样调用、推理延迟与动作抖动风险。本项目早期全局池化模型把重规划改为每步一次后仍为 54%，说明重规划频率不能弥补视觉表征丢失的位置信息。
2. **多模态下 MSE BC 会怎样？** 若同一观测有绕障碍物左、右两条合理轨迹，MSE 往往趋向两种动作的条件均值；均值轨迹可能穿过障碍，或落在数据里没有出现过的中间模式。扩散策略可通过不同噪声样本表达多个动作分布模式。但本项目的双目标实验是目标位置不同、图像条件也不同，证明了两种目标条件都可完成，**没有证明同一观测下生成左右两种绕行轨迹**。
3. **阶段 2 与阶段 3 的差值表示什么？** 探索性结果是 state-only 81%、视觉版 71%，差值为 −10 个百分点；碰撞率从 12% 降至 3%，超时率从 7% 升至 26%。它反映“显式目标坐标但无障碍坐标”与“从图像推断目标和障碍”两套输入/任务难度的差异，不是单独的视觉因果效应。若把障碍坐标也放入 state，比较就变成特权状态信息与视觉感知的对照。
4. **padding 如何影响策略？** 专家轨迹短于 H 时重复末动作会形成非有效尾部监督；H 越大，重复槽位越多，动作块中真实运动信息占比越小，可能使预测变迟缓并增加超时。本实验记录了有效槽位掩码并对 masked/unmasked 做了公平比较；单训练 seed 下 masked 点估计反而略低，故不能宣称掩码必然提升成功率。下一步应检查 mask 的损失归一化、尾部动作语义，并增加训练 seeds。
5. **FM 相对 DDPM 对实时控制的潜在收益是什么？** FM 用 ODE 速度场从噪声积分到动作，当前配置用 10 步 Euler；DDPM 使用 20 步 DDIM，因此前者理论上需要更少的模型函数调用。公平矩阵中 FM 成功率 87%、DDPM 81%，但区间重叠且只有一个训练 seed；本项目没有测端到端延迟，不能据此声称 FM 快 5 倍或显著优于 DDPM。

### Reflection（约 300 字）

这个二维实验没有真实相机标定、机械臂动力学、执行器延迟和接触过程。Pi-0 或 OpenVLA 部署到工业机器人前，还需要处理相机内外参漂移、光照变化、遮挡、反光及多传感器时间同步，并验证语言指令与视觉目标是否一致。动作从归一化平面坐标变成关节或末端控制量后，必须满足速度、力矩、工作空间和碰撞约束；网络输出也要经过安全过滤、限幅、异常检测和可验证的急停/回退控制。真实任务还需要覆盖稀有故障和分布外场景的数据，进行域适配、跨设备复现、长时间稳定性与延迟测试，并保留操作员接管、审计日志和版本回滚机制。因而高离线成功率只是进入真实部署评估的起点，不是安全或可靠性的证明。

### 原始要求与仓库证据

| 原始要求 | 当前实现 / 证据 | 结论与边界 |
|---|---|---|
| TODO 19：action-chunk DDPM loss | `train.py`；余弦 scheduler、DDPM 目标及 mask loss；`tests/test_project5.py` | 已实现 |
| TODO 20：视觉编码与 observation conditioning | `model.py`；CNN + 4×4 spatial pooling；图像和 agent state 条件 | 已实现；全局池化负结果保留作对照 |
| TODO 21：闭环评估 | `eval.py`；成功、碰撞、超时、步数、末端距离和轨迹目标距离 | 已实现；每个公平条件 100 episodes |
| test 环境成功率 ≥70% | 公平视觉 DDPM H=16 masked：81%；移动障碍评估：72% | 达到点估计目标；移动障碍碰撞率 20%，安全性仍有限 |
| 消融、训练曲线、阶段 2/3 分析 | 本 README 的主矩阵、先导对照和曲线解释；`results/ablation.md`、`results/fair_seed42/` | 已报告；明确区分探索性结果与公平复评 |
| ≥3 个成功 rollout、实验日志、≥3 条真实 debug 记录 | `results/rollouts/`（本文以 2×2 图组嵌入 4 张）；`logs/exp_log.md`；`debug_log.md`（5 条） | 文件齐全 |
| FM 加分项 | 公平矩阵 H=16 masked：FM 87%，10-step Euler；DDPM 81%，20-step DDIM | 已实现并比较；未测实际 wall-clock 加速倍数 |
| 多目标 / multimodal demos | 双目标数据，两个目标条件成功率 96.2% / 97.9% | 部分满足：未证明同一观测下的双峰动作分布，不夸称完整通过 +10 |
| ResNet-18 与移动障碍加分项 | ResNet-18 5k/15k 训练和闭环结果；静态策略在移动障碍上评估 | 已做扩展实验；ResNet 收敛速度非严格同预算比较，动态场景碰撞仍高 |

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
- [阶段 2 state-only 评估](results/eval_state.json)、[阶段 3 视觉评估](results/eval_vision.json)：原始阶段对照；`results/rollouts/`、`results/bonus_rollouts/`：示例轨迹；`results/*.json`：扩展实验指标。
- [先导训练日志](logs/train_spatial.log)、`logs/train_resnet18*.log` 与 `results/metrics/`：曲线和吞吐数据来源；[公平实验方案](FAIR_EXPERIMENT_PLAN.md)：对照协议；[调试记录](debug_log.md)：5 条实际实现/运行问题。
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
curl -fLO "$base/project5-fair-seed42-reeval-c062751.tar.gz"
curl -fLO "$base/project5-vla-action-diffusion-source-a2b695d.tar.gz"
curl -fLO "$base/SHA256SUMS-project5-c062751.txt"
sha256sum -c SHA256SUMS-project5-c062751.txt
cd ../..
tar -xzf assets/release/project5-fair-seed42-checkpoints.tar.gz -C assets/restored
mkdir -p runs/project5_release_eval_archive
tar -xzf assets/release/project5-fair-seed42-reeval-c062751.tar.gz -C runs/project5_release_eval_archive
python eval_checkpoint_archive.py --checkpoint-root assets/restored/runs/project5_fair --run-root runs/project5_release_eval --episodes 100 --seed-start 10000
```

`eval_checkpoint_archive.py --resume` 会跳过已经生成 `eval.json` 的条件，适合断线后接续；正式复现训练矩阵则使用上一节的 `run_fair_experiments.py` 命令。Release 中原 `SHA256SUMS` 校验 checkpoint、首批结果和源码快照；`SHA256SUMS-project5-c062751.txt` 校验本次复评结果和更新后的源码快照。复评归档可直接检查或引用；从 checkpoint 重新运行脚本则会生成新的完整评估目录。Release source 是源码快照；实验应以当前克隆的 `main` 作为评估代码。

## 限制

这是无摩擦、无动力学的二维点质量环境，不代表真实机器人控制能力。公平矩阵只有一个训练 seed，因此成功率是单次评估集上的观测比例，不能反映训练 seed 间方差。移动障碍实验的碰撞率升至 20%，静态训练策略在环境变化下仍有安全局限。多目标实验尚未验证同一观测下的双峰动作轨迹；ResNet-18 实验也没有按统一 wall-clock 或统一收敛阈值比较速度。因此，这两项结论按已获得的证据限定，不把部分验证写成完整证明。
