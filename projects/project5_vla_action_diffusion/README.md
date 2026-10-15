# Project 5：VLA Action Diffusion 实验报告

本项目在二维环境中实现并评估视觉条件动作策略，比较行为克隆（BC）、去噪扩散（DDPM）与 Flow Matching（FM），并研究动作块长度、填充掩码、视觉编码器和环境变化。

## 一、实验内容

`Reach2DEnv` 将场景渲染为 64×64 RGB 图像，包含智能体、目标和两个障碍物。视觉策略接收图像与智能体位置，no vision (state-only) 则接收智能体和目标坐标，作为一个参照条件。

策略预测长度为 `H` 的二维动作块，每执行前 4 步后重新观测和规划。动作先裁剪到 `[-1, 1]²`，环境再乘以 `ACTION_SCALE=0.05` 更新位置；目标距离小于 `0.06` 判成功，智能体与障碍物距离小于 `0.10` 判碰撞，每回合最多 100 步。

视觉编码器为三层 stride-2 CNN（3→32→64→128），每层使用 GroupNorm 与 SiLU，最后4×4 自适应池化保留空间信息，再映射为 128 维特征。状态、图像和时间嵌入共同条件化动作生成器，动作块展平后由隐藏宽度为 256 的 MLP 预测噪声、速度或动作。

DDPM 使用余弦噪声调度；BC 直接回归动作块，FM 回归速度场。

### 二、Loss 设计

令 $o$ 表示当前图像与允许输入的状态，$`\mathbf{a}_0=(a_0,\ldots,a_{H-1})`$ 表示一段动作块。DDPM 对动作块加噪并预测噪声：

$$
\begin{aligned}
\mathbf{a}_t &= \sqrt{\bar\alpha_t}\,\mathbf{a}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon, \\
\mathcal{L}_{\mathrm{DDPM}} &= \mathbb{E}_{t,\mathbf{a}_0,\boldsymbol\epsilon,o} \\
&\quad \left[\left\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{a}_t,t,o)\right\|_2^2\right].
\end{aligned}
$$

BC 直接回归动作块，FM 在高斯噪声与数据动作之间线性插值，并学习从噪声指向动作的速度场。开启 padding mask 时，三种损失均只对有效动作槽位计入均方误差。

$$
\begin{aligned}
\mathcal{L}_{\mathrm{BC}} &= \mathbb{E}\left[\|f_\theta(o)-\mathbf{a}_0\|_2^2\right], \\
\mathbf{a}_t &= (1-t)\mathbf{z}+t\mathbf{a}_0, \\
\mathcal{L}_{\mathrm{FM}} &= \mathbb{E}\left[\|v_\theta(\mathbf{a}_t,t,o)-(\mathbf{a}_0-\mathbf{z})\|_2^2\right].
\end{aligned}
$$

## 三、实验结果

|实验条件|代表配置|主要结果|核心观察|
|-|-|-|-|
|Pure BC (no diffusion)|H=16，图像 + agent state，MSE|成功 88%，碰撞 9%，超时 3%；平均 26.95 步，末端距离 0.0913|BC 点估计高于视觉 DDPM，接近 FM|
|DDPM (no vision)|H=16，agent + target 坐标，DDIM 20 步|成功 81%，碰撞 12%，超时 7%；平均 55.89 步，末端距离 0.1075|state-only 直接获得目标位置，但效果差于BC|
|DDPM + vision|H=16，图像 + agent state，4×4 spatial，masked|成功 81%，碰撞 8%，超时 11%；平均 35.57 步，末端距离 0.1339|超时比较明显，效果差于BC|
|FM + vision|H=16，图像 + agent state，Euler 10 步|成功 87%，碰撞 7%，超时 6%；平均 29.53 步，末端距离 0.0987|与 BC 接近，比 DDPM 效果好|
|DDPM + vision Chunk 8 vs 32 vs 64|masked H=8/32/64 成功率 89%/48%/26%；unmasked 91%/66%/28%|masked 超时率 4%/46%/73%；unmasked 2%/27%/63%|H=8 效果最优；H=32/64 超时率明显上升|

<p align="center">
  <img src="results/loss_curve.png" alt="训练损失曲线" />
  <br />
  <em>图 1. 训练损失曲线</em>
</p>
### 1）Pure BC (no diffusion)

Pure BC 直接以图像和 agent 位置回归 H=16 动作块，不经过生成式采样。

**成功率为 88%，碰撞率 9%、超时率 3%**，平均 26.95 步完成，末端距离为 0.0913。

与视觉 DDPM（81%）和 FM（87%）相比，BC 没有落后，在当前二维任务和专家数据规模下，生成式动作建模并未自动带来更高的成功率。

### 2）DDPM (no vision)

state-only DDPM 的 H=16 参照实验使用 agent 与 target 显式坐标、20 步 DDIM，成功率为 81%。

| 条件 | 输入条件 | 成功率 | 碰撞率 | 超时率 | 平均步数 | 末端距离 |
|---|---|---:|---:|---:|---:|---:|
| DDPM (no vision) | agent + target 坐标 | 81% | **12%** | 7% | 55.89 | 0.1075 |
| DDPM + vision | RGB 图像 + agent 坐标 | 81% | **8%** | 11% | 35.57 | 0.1339 |

no vision(state-only) 直接获得目标位置，却没有障碍物坐标，视觉策略则从图像识别目标与障碍，可以看到碰撞率上vision版更佳。

### 3）DDPM + vision

视觉 DDPM 接收 64×64 RGB 图像和 agent 坐标，使用 4×4 空间池化 CNN、H=16 masked action chunk 与 20 步 DDIM。

**成功率为 81%，碰撞率 8%、超时率 11%**，达到 70% 参考目标。

以下闭环图展示四条成功轨迹，作为定性示例。

<table align="center">
  <tr>
    <td align="center"><img src="results/rollouts/success_ep000.png" alt="视觉 DDPM 成功轨迹 episode 0" width="320"/><br/><sub>Episode 0</sub></td>
    <td align="center"><img src="results/rollouts/success_ep001.png" alt="视觉 DDPM 成功轨迹 episode 1" width="320"/><br/><sub>Episode 1</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="results/rollouts/success_ep002.png" alt="视觉 DDPM 成功轨迹 episode 2" width="320"/><br/><sub>Episode 2</sub></td>
    <td align="center"><img src="results/rollouts/success_ep003.png" alt="视觉 DDPM 成功轨迹 episode 3" width="320"/><br/><sub>Episode 3</sub></td>
  </tr>
</table>

<p align="center"><em>图 2. H=16 视觉 DDPM 的四条成功轨迹示例。蓝线为策略轨迹，圆点为起点，红星为目标，灰色方块为障碍物</em></p>

### 4）FM + vision

FM 在相同视觉条件、H=16 下达到**87% 成功率，碰撞率 7%、超时率 6%**，采样使用 10 步 Euler。
BC 为 88%，视觉 DDPM 为 81%，在实验条件下FM的效果相较于DDPM更优，但是并没有超过BC，即没有带来实质的正向提升。

**具体实现和数据对比见扩展实验第一项。**

### 5）Chunk 8 vs 32 vs 64

下表列出同一视觉 DDPM 中 masked 与 unmasked 的结果；H=16 作为中间参照。

| H | padding mask | 成功率 | 碰撞率 | 超时率 | 平均步数 | 末端距离 | 沿途目标距离 |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 8 | 是 | 89% | 7% | 4% | 27.57 | 0.0994 | 0.4852 |
| 8 | 否 | **91%** | 7% | 2% | 26.37 | 0.0919 | 0.4887 |
| 16 | 是 | 81% | 8% | 11% | 35.57 | 0.1339 | 0.4675 |
| 16 | 否 | 85% | 8% | 7% | 31.50 | 0.1115 | 0.4769 |
| 32 | 是 | 48% | 6% | 46% | 64.39 | 0.2621 | 0.4612 |
| 32 | 否 | 66% | 7% | 27% | 54.73 | 0.2138 | 0.4716 |
| 64 | 是 | 26% | 1% | 73% | 85.86 | 0.3668 | 0.5033 |
| 64 | 否 | 28% | 9% | 63% | 81.96 | 0.4147 | 0.5955 |

`action_mask` 会标记真实动作位置为 1，补齐位置为 0。
- Masked：计算损失时只统计 mask 为 1 的位置，padding 不产生训练梯度。
- Unmasked：忽略 mask，把重复的末动作也当作真实示范来学习。

**H=8 的观测成功率最高**。H 增至 32、64 后，masked 成功率降至 48%、26%，超时率升至 46%、73%，说明长动作块在当前重规划设置下更难稳定完成任务，且masked样本结果普遍差于unmasked，说明mask在本条件下基本没有正向效果。

## 四、扩展实验

本节按四项加分要求分别报告实现、评估和主要结论。

下图展示扩展实验的总体结果，并标示多目标评测中的目标位置与分模式成功率。

<p align="center">
  <img src="results/bonus_summary.png" alt="扩展实验闭环结果与双目标示意" />
  <br />
  <em>图 3. 扩展实验结果及多目标实验中的目标位置与成功率。</em>
</p>

### 1）Flow Matching 替代 DDPM

FM 将 DDPM 的噪声预测目标替换为速度场回归，并用 Euler ODE 从噪声生成动作块。

下表中的 BC、DDPM 与 FM 使用同一视觉条件、H=16、专家数据及评估集。

| 方法 | 训练目标 | 推理方式 | 成功率 | 碰撞率 | 超时率 | 平均步数 |
|---|---|---|---:|---:|---:|---:|
| Pure BC | 动作块 MSE | 直接回归 | 88% | 9% | 3% | 26.95 |
| DDPM + vision | 噪声预测 | DDIM，20 步 | 81% | 8% | 11% | 35.57 |
| FM + vision | 速度场回归 | Euler，10 步 | 87% | 7% | 6% | 29.53 |

FM 完成对 DDPM 训练目标的替换，成功率点估计与 BC 接近。

### 2）Multi-modal demo 数据集：两个 target

在每张图像中同时绘制左、右两个目标，分别位于 $`(-0.65, 0.55)`$ 和 $`(0.65, 0.55)`$；到达任意目标即成功，并记录实际到达的目标。训练数据按场景成对采集：同一场景的两条专家轨迹从完全相同的初始状态、图像、障碍物和双目标布局出发，分别前往左、右目标。目标编号不作为策略输入，策略只接收图像与状态，因此不同路线由模型采样产生，不由显式目标标签指定。

使用一个 seed，从现有 H=16 视觉 DDPM checkpoint 初始化并微调 5,000 步。正式评测在 100 个固定场景上，每个场景使用 20 个推理噪声样本（推理采样次数），共 2,000 个 episode，DDIM 20 步和每 4 步重规划协议。。

| 模型 | 成功率 | 碰撞率 | 超时率 | 平均步数 | 平均末端距离 | 到达目标 G0 / G1 | 双目标覆盖场景 | 稳定双目标覆盖场景 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 同观测配对数据微调（5k） | **81.40%（1,628/2,000）** | **8.15%** | **10.45%** | **48.24** | **0.1248** | **699 / 929** | **57/100** | **34/100** |

**“双目标覆盖”** 标准为：同一固定初始场景的 20 次采样中，至少各有一次成功到达 G0 和 G1。

**“稳定双目标覆盖”** 要求每个目标至少各成功 3 次。
微调后两种目标都被采到，且 57% 的固定场景覆盖了两种模式，构成同一初始观测下生成不同目标路线的直接证据。但稳定覆盖率为 34%，说明模式覆盖仍不均衡，双模态性能并不稳定。

<p align="center">
  <img src="results/simultaneous_multimodal/finetuned_rollouts.png" alt="同一初始观测下，微调策略从不同采样中分别到达左、右目标的闭环轨迹" />
  <br />
  <em>图 4. 同观测双目标实验的微调模型轨迹。</em>
  <br />
  <em>每个小图固定同一初始场景，蓝线与橙线分别表示成功到达左、右目标的不同采样轨迹，障碍物与两个目标在每次采样中保持不变。</em>
</p>

### 3）ImageNet-pretrained ResNet-18 替换小 CNN

ResNet-18 加载 ImageNet 预训练权重（`vision_pretrained: true`），参数量 11.48M；小 CNN 参数量 0.59M。

| 方法 | 训练目标 | 推理方式 | 成功率 | 碰撞率 | 超时率 | 平均步数 |
|---|---|---|---:|---:|---:|---:|
| Small CNN（DDPM，10k） | DDPM 噪声预测 | DDIM 20 步；H=16，每 4 步重规划 | 81% | 8% | 11% | 35.57 |
| ImageNet-pretrained ResNet-18（5k） | DDPM 噪声预测 | DDIM 20 步；H=16，每 4 步重规划 | 3% | 8% | 89% | 91.92 |
| ImageNet-pretrained ResNet-18（15k） | DDPM 噪声预测 | DDIM 20 步；H=16，每 4 步重规划 | 2% | 18% | 80% | 89.29 |

从结果看，Small CNN 的观测成功率为 81%，远高于 ResNet-18 的 3% 和 2%。ResNet-18 从 5,000 步延长到 15,000 步后，成功率未见提升，超时率虽从 89% 降至 80%，碰撞率却从 8% 升至 18%，平均步数仍接近 90 步，说明当前配置下ResNet策略尚未形成有效的闭环控制能力，ImageNet 预训练本身并未带来更好的任务表现，延长训练也不足以改善此问题。

### 4）移动障碍环境泛化

复用静态障碍训练得到的 H=16 视觉 DDPM checkpoint，测试时障碍速度设为 0.025。

两种环境评测各 100 个 episode，使用相同模型，seed 区间不同。

| 测试环境 | 评估 seeds | 成功率 | 碰撞率 | 超时率 | 平均步数 | 末端距离 |
|---|---|---:|---:|---:|---:|---:|
| 静态障碍 | 10000–10099 | 71% | 3% | 26% | 48.70 | 0.1643 |
| 移动障碍（速度 0.025） | 40000–40099 | 72% | 20% | 8% | 34.94 | 0.1867 |

成功率点估计相近，但碰撞率由 3% 升至 20%，说明策略具备一定迁移能力。下方为移动障碍测试中的两条成功轨迹。

<table align="center">
  <tr>
    <td align="center"><img src="results/bonus_rollouts/moving_success_ep000.png" alt="移动障碍成功轨迹 episode 0" width="320"/><br/><sub>Episode 0</sub></td>
    <td align="center"><img src="results/bonus_rollouts/moving_success_ep005.png" alt="移动障碍成功轨迹 episode 5" width="320"/><br/><sub>Episode 5</sub></td>
  </tr>
</table>

<p align="center"><em>图 5. 移动障碍环境中的两条成功策略轨迹，灰色方块表示回合结束时的障碍位置。</em></p>

## 五、自查问题

1. **Action chunking：为什么要一次预测 H 步而不是每步预测一步？ 如果 exec_steps 设成 1（每步都重规划），会有什么代价和好处？**

    一次预测 $H$ 步能减少每个环境步都重新调用生成模型的开销，并让局部动作保持连贯，每执行 `exec_steps` 步再观察和规划，则保留纠偏能力。若 `exec_steps=1`，反馈更及时，但会增加采样调用与推理开销。本项目中固定每 4 步重规划，在闭环纠偏和调用成本之间折中。

2. **多模态：绕开干扰物有左右两条合理路径。用 MSE 直接回归动作（Pure BC） 在这种情况下会输出什么？为什么 diffusion 不会有这个问题？**

    若同一观测有绕障碍物左、右两条合理轨迹，MSE 往往趋向两种动作的条件均值，均值轨迹可能穿过障碍，或落在数据里没有出现过的中间模式，扩散策略则有机会通过不同噪声样本表达多个动作分布模式。双目标实验在配对的相同初始观测上分别监督左右目标路线，并在 57/100 个固定场景中观察到两种成功模式，可以说明扩散策略学习到了多模态行为。但只有 34/100 个场景达到每个目标至少 3 次成功，反映双模态仍不完整。

3. **条件：阶段 2 与阶段 3 的成功率差多少？这个差值衡量的到底是什么？ 如果把干扰物坐标也加进 state，这个实验还能说明问题吗？**

    虽然两者在当前记录中的成功率点估计均为 81%，但 no vision 直接输入目标坐标，视觉策略则从图像中识别目标和障碍，二者碰撞率、超时率和平均步数不同，可以反映视觉对于锁定目标与避障的效果。若把障碍坐标也放入 state，则无法对比突出视觉的避障效果。

4. **padding：本环境轨迹中位数 23 步而 chunk 是 32，超过一半的监督信号来自补出来的动作。 这对学到的策略有什么影响？你会怎么改进？**

    专家轨迹短于 H 时会重复末动作进行padding，H 越大，重复槽位越多，动作块中真实运动信息占比越小，会使预测变迟缓并增加超时。本实验记录了有效槽位掩码并对 masked/unmasked 做了比较，结果上 masked 点估计反而略低，说明如果末动作本身就是合理的终止行为，完全屏蔽它也未必更好。

5. **FM vs DDPM：Pi-0 用 flow matching 而不是 DDPM 做 action head， 在实时控制这个场景下，这个选择的关键收益是什么？**

    DDPM 需要通过多步迭代去噪生成动作，而 FM 直接学习从高斯噪声到真实动作分布的速度场，并通过 ODE 数值积分生成动作，采用线性插值路径的 FM 通常适合少步数采样。当前实验中，FM 使用 10 步 Euler，而 DDPM 使用 20 步 DDIM，因此 FM 所需的模型函数调用次数减少 50%。实验得到 FM 成功率 87%、DDPM 81%，FM 在尽量保持动作生成质量的前提下降低推理开销，使生成式策略更容易满足高频控制与闭环重规划的实时性要求。

## 六、Reflection

   以上 toy 实验里未体现的工程：Pi-0 或 OpenVLA 部署到工业机器人前，还需要处理相机内外参漂移、光照变化、遮挡、反光及多传感器时间同步，并验证语言指令与视觉目标是否一致等内容。动作从归一化平面坐标变成关节或末端控制量后，必须满足速度、力矩、工作空间和碰撞约束，并且网络输出也要经过安全过滤、限幅、异常检测和可验证的急停/回退控制。
  此外，真实任务还需要覆盖稀有故障和分布外场景的数据，进行域适配、跨设备复现、长时间稳定性与延迟测试，并保留操作员接管、审计日志和版本回滚机制等等。理论与工程落地之间，还有相当一段距离。

## 七、文件索引

| 类别 | 文件 / Release 资产 | 内容 |
|---|---|---|
| 环境与数据 | [`env.py`](env.py)、[`dataset.py`](dataset.py)、[`obs_utils.py`](obs_utils.py) | 环境、专家数据与观测条件处理。 |
| 模型与配置 | [`model.py`](model.py)、[`train.py`](train.py)、[`eval.py`](eval.py)、[`configs/`](configs/) | 策略结构、训练与闭环评估入口、实验配置。 |
| 同观测双目标多模态 | [`eval_multi_target.py`](eval_multi_target.py)、[`plot_multi_target.py`](plot_multi_target.py)、[`results/simultaneous_multimodal/summary.json`](results/simultaneous_multimodal/summary.json) | 100 个固定场景、每场景 20 次采样的零样本/微调配对评测汇总。 |
| 矩阵与历史消融 | [矩阵汇总](results/fair_seed42/fair_ablation.md)、[矩阵指标](results/fair_seed42/fair_summary.json)、[`results/ablation.md`](results/ablation.md)、[`logs/exp_log.md`](logs/exp_log.md) | 当前矩阵结果、历史消融归档与实验日志。 |
| 条件评估与扩展结果 | [state-only 评估](results/eval_state.json)、[视觉评估](results/eval_vision.json)、`results/*.json` | 条件参照及各扩展实验的逐项指标。 |
| 轨迹与训练记录 | [`results/rollouts/`](results/rollouts/)、[`results/bonus_rollouts/`](results/bonus_rollouts/)、`logs/train_spatial.log`、`logs/train_resnet18*.log`、[`results/metrics/`](results/metrics/) | 成功轨迹、ResNet 收敛图数据来源与训练指标。 |
| 同观测双目标轨迹图 | [`results/simultaneous_multimodal/finetuned_rollouts.png`](results/simultaneous_multimodal/finetuned_rollouts.png)、[`results/simultaneous_multimodal/zero_shot_rollouts.png`](results/simultaneous_multimodal/zero_shot_rollouts.png) | 相同双目标场景下，配对微调模型与原 checkpoint 的闭环轨迹对比。 |
| Release：checkpoint | `project5-fair-seed42-checkpoints.tar.gz` | 矩阵模型权重。 |
| Release：结果归档 | `project5-fair-seed42-results.tar.gz`、`project5-fair-seed42-reeval-c062751.tar.gz` | 实验原始结果与复评结果。 |
| Release：同观测双目标复现包 | [project5-simultaneous-multimodal-v1.0.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-simultaneous-multimodal-v1.0.0) | 配对演示数据、微调 checkpoint、评测 JSON、源码、日志、轨迹图与 SHA-256 校验文件。 |
| Release 版本 | [project5-fair-seed42-v1.0.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-fair-seed42-v1.0.0) | 上述复现文件所在的 GitHub Release。 |
