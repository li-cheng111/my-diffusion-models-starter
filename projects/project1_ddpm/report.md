# 项目 1：从零实现 DDPM

## 当前状态

本仓库已完成项目 1 基础档、进阶档和挑战档的源码、配置、静态测试、结果和实验文档。MNIST 基线与 CIFAR-10 实验均在 AutoDL 上完成真实训练、采样和 FID 评估。本报告只记录实际得到的结果；原始进阶档单次 linear EMA FID 为 19.2879，挑战档六组实验的汇总见 `results/challenge/summary.md`。针对 FID≤15 的独立续训实验最终 EMA FID 为 18.1490，仍未达标。

## 方法概述

DDPM 使用固定的前向过程逐步向图像加入高斯噪声：

$$
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

训练时随机采样时间步 `t`，使用闭合形式直接得到 `x_t`，再让 U-Net 预测加入的噪声。优化目标是：

$$
\mathcal L_{\text{simple}}
=\|\epsilon-\epsilon_\theta(x_t,t)\|^2.
$$

反向采样从标准高斯噪声开始，按 `T-1` 到 `0` 的顺序执行采样步骤。最后一步不再添加随机噪声。

## 代码结构

- `schedule.py` 计算 beta、alpha、累积 alpha 和 posterior variance。
- `diffusion.py` 实现前向加噪、训练损失和完整反向采样。
- `model/embedding.py` 实现 sinusoidal timestep embedding。
- `model/unet.py` 实现带时间条件的残差 U-Net。
- `dataset.py` 支持 MNIST 和 CIFAR-10，并将输入归一化到 `[-1,1]`。
- `train.py` 支持 AdamW、warmup、EMA、AMP、梯度裁剪、checkpoint 和 loss history。
- `evaluate.py` 支持从训练集或测试集读取 real images；进阶档默认固定使用无增强的 5,000 张训练图，并可用 `--compare_ema` 同时评估 EMA 与 raw 权重。

## 自查问题

1. `q_sample` 中的线性组合对应 `q(x_t|x_0)` 的闭合形式。
2. `(B,)` 的时间系数要 reshape 为 `(B,1,1,1)`，这样才能对每个 batch 样本的所有通道和像素广播；否则维度可能无法匹配或产生错误广播。
3. schedule 使用 `register_buffer`，因为这些张量不是可学习参数，但必须随模型一起移动到 CPU 或 GPU，并进入 checkpoint 状态。
4. 随机采样 `t` 是对所有时间步期望的 Monte Carlo 估计；每个 batch 都能覆盖不同噪声强度。
5. `t=0` 时 posterior variance 为零，继续添加噪声会破坏最终的干净样本。

## 实验完成情况

- MNIST 50 轮：已完成训练和推理；结果见下方“MNIST 实验结果”。
- CIFAR-10 200 轮：已完成训练、EMA/raw 采样和 FID 对比，结果见下方“进阶档实验结果”。
- 挑战档：linear/cosine 两种 schedule、seed 42/43/44 共六组 200 轮实验已完成，结果见下方“挑战档实验结果”。

## MNIST 实验结果

| 项目 | 实际值 |
|---|---|
| 硬件 | NVIDIA GeForce RTX 5090 |
| 训练配置 | MNIST，50 轮，23,400 步 |
| 训练时间 | 24.8 分钟 |
| 最后一次日志 loss | 0.01151 |
| FID | 32.8913（EMA，5,000 张样本） |

结果文件：

- `runs/exp_mnist_baseline/loss_history.csv`
- `runs/exp_mnist_baseline/loss_curve.png`
- `runs/exp_mnist_baseline/samples_inference_ema/grid.png`
- `runs/exp_mnist_baseline/ckpt/fid_5000_EMA.txt`

大型 MNIST 临时权重不进入普通 Git；本次临时产物压缩包见
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。

## 基础档验收状态

基础档的 8 个代码实现点已经写入仓库；MNIST 50 轮和 CIFAR-10 200 轮均已完成真实训练与推理。进阶档的 EMA/raw 对比已完成，但本次 EMA FID 尚未达到 15 的目标。

## 进阶档实验结果

配置文件：`configs/cifar10.yaml`。

```bash
python train.py --config configs/cifar10.yaml
python sample.py --ckpt runs/exp_cifar10_advanced/ckpt/final.pt \
    --num_samples 64 --save_grid
python evaluate.py --ckpt runs/exp_cifar10_advanced/ckpt/final.pt \
    --num_samples 5000 --batch_size 64 --real_split train --compare_ema
```

评估实际生成：

- `fid_5000_EMA.txt`
- `fid_5000_raw.txt`
- `fid_comparison.md`

### 进阶档实际指标

| 指标 | 实际值 |
|---|---|
| 硬件 | NVIDIA GeForce RTX 5090 |
| CIFAR-10 训练时间 | 98.2 分钟 |
| 总训练步数 | 78,000（50,000 张训练图，批大小 128，drop_last） |
| 最后一次日志 loss | 0.01938（step 78,000） |
| EMA FID（5,000 张训练图像） | 19.2879 |
| Raw FID（5,000 张训练图像） | 28.7464 |
| Raw - EMA（原始权重减 EMA） | +9.4586 |
| FID ≤ 15 | 未达到 |

EMA 的 FID 比 raw 低 9.4586，说明本次训练中 EMA 权重的分布质量更好；该结论仅针对本次 seed、配置和 5,000 样本评估。结果文件：

- `runs/exp_cifar10_advanced/loss_history.csv`
- `runs/exp_cifar10_advanced/loss_curve.png`
- `runs/exp_cifar10_advanced/samples_inference_ema/grid.png`
- `runs/exp_cifar10_advanced/samples_inference_raw/grid.png`
- `runs/exp_cifar10_advanced/ckpt/fid_comparison.md`

进阶档最终 checkpoint 不进入普通 Git；挑战档六组最终 checkpoint 见
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。
本次未达到 FID ≤ 15；后续若继续冲击目标，应在保留已有结果的基础上调整模型、schedule 或训练策略后重新实验。

## 挑战档实验结果

## FID≤15 改进实验

### 实验目的与控制变量

原始 linear/seed44 checkpoint 已经接近当前实验矩阵中的最好结果，因此本实验只改变
训练时长，并加入一个独立的采样消融变量：从该 checkpoint 继续训练到 100,000 次有效
更新，同时分别评估原始反向采样和预测 $x_0$ 逐步裁剪的采样。模型结构、linear beta
schedule、CIFAR-10 数据、EMA、评估样本数、真实数据划分和随机种子保持固定；不加入
learned variance、hybrid loss 或 importance sampling。

配置文件为 `configs/cifar10_fid15_resume.yaml`，输出目录为
`runs/fid15_4090_linear_seed44`。续训使用 batch size 128、AdamW、初始学习率
$2\times10^{-4}$，在新增 22,000 次有效更新内余弦下降到 $2\times10^{-5}$，并恢复
raw、optimizer、EMA 和 AMP scaler 状态。训练每 5,000 步保存一个 checkpoint，最终
保存 `step_100000.pt` 和 `final.pt`。

### 固定协议下的 FID 结果

所有数值均为 EMA 权重、5,000 张生成样本对 5,000 张无增强 CIFAR-10 train 图像的
FID，seed 为 44：

| 模型 | 采样方式 | FID | 相对原始未裁剪基线 |
|---|---|---:|---:|
| 原始 200 epoch / seed44 | 未裁剪 | 18.9577 | 0.0000 |
| 原始 200 epoch / seed44 | 裁剪预测 $x_0$ | 18.9715 | +0.0138 |
| 100,000 有效更新 | 未裁剪 | 18.1502 | -0.8075 |
| 100,000 有效更新 | 裁剪预测 $x_0$ | **18.1490** | **-0.8087** |

续训是主要收益来源：在相同采样方式下，未裁剪 FID 从 18.9577 降到 18.1502。
预测 $x_0$ 裁剪在原始 checkpoint 上略有负收益，在最终 checkpoint 上只比未裁剪低
0.0012，属于很小的差异，不能据此断言该技巧能稳定改善 FID。最终最佳结果为
18.1490，距离目标 15 仍有 3.1490，因此本轮结论是“有改善，但未达标”。

### 训练与产物

- 硬件：NVIDIA GeForce RTX 4090，单卡；
- 续训时间：28.5 分钟；
- 有效更新：从约 78k 续训至 100,000；
- 最后日志：`step=100000`，loss `0.02777`，学习率 `2.000e-05`；
- 稳定性：未观察到 NaN、AMP 跳步或异常中断；AutoDL SSH 断联后，screen 中的评估
  进程被重新检查，已有 checkpoint 未受影响；
- AutoDL 原始结果目录：`runs/fid15_4090_linear_seed44/`；Git 中的复核副本位于
  `results/fid15_4090_linear_seed44/`；
- 最终样本网格：`results/fid15_4090_linear_seed44/samples_final_ema_grid.png`。

大型 checkpoint 和中间样本不进入普通 Git，使用 GitHub Release 附件保存。普通 Git
只保存代码、配置、FID 文本、loss history、loss 曲线、样本网格和摘要，避免仓库被
二进制权重或 CIFAR-10 临时数据占满。

挑战档要求实现 cosine 调度策略，并在相同模型和训练协议下对比 linear/cosine，分别使用随机种子 `42、43、44` 报告均值 ± 标准差，同时提交包含失败案例分析的八页技术报告。本仓库已完成实现、六组真实实验和结果汇总：

- `schedule.py` 中的 `cosine_beta_schedule` 已接入 `DDPMSchedule`；
- `configs/cifar10_linear.yaml` 和 `configs/cifar10_cosine.yaml` 固定了对比实验的控制变量；
- `challenge.py` 负责六组 200 轮训练、64 张 EMA 样本网格、5,000 对 5,000 FID 评估和结果汇总，并支持加入 50 轮控制组；
- `challenge_report.md` 已更新为真实八页技术报告，`results/challenge/` 保存逐 seed FID 和最终样本网格。

### 每个随机种子的结果

| 调度策略 | Seed | EMA FID | Raw FID | Raw - EMA |
|---|---:|---:|---:|---:|
| linear | 42 | 19.2879 | 28.7464 | +9.4585 |
| linear | 43 | 19.6306 | 42.7537 | +23.1231 |
| linear | 44 | 18.9593 | 34.9399 | +15.9806 |
| cosine | 42 | 137.9856 | 283.4196 | +145.4340 |
| cosine | 43 | 129.2708 | 399.1676 | +269.8968 |
| cosine | 44 | 145.2832 | 410.0121 | +264.7289 |

### 均值 ± 标准差

| 调度策略 | 训练轮数 | EMA FID | Raw FID |
|---|---:|---:|---:|
| linear | 200 | 19.2926 ± 0.3357 | 35.4800 ± 7.0193 |
| cosine | 200 | 137.5132 ± 8.0166 | 364.1998 ± 70.1675 |

FID 使用每次 5,000 张生成图像，对比 5,000 张不使用随机增强的 CIFAR-10 训练图像。在线性和 cosine
均保持同一模型、优化器、训练轮数和 seed 集合的前提下，本实验中 linear 明显优于 cosine；但
该结论只适用于当前实验协议，不能外推到其他模型或训练预算。线性 schedule 的 EMA 平均 FID
比 raw 低 16.1874，cosine 的对应差值为 226.6866。

### 样本、失败记录与限制

六组最终 EMA 样本网格保存在 `results/challenge/*/samples_challenge_ema/grid.png`；大型 checkpoint
和压缩后的中间样本保存在 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。
挑战档提交结果中未记录需要单独归因的训练失败；AutoDL 的 GitHub HTTP 503 和数据下载慢属于环境访问问题，
已在 `debug_log.md` 单独记录，不能当作 schedule 失败案例。

本次没有运行 50 轮控制组，因此不能回答 50 轮下的调度策略差异。三组 seed 仍是有限样本，FID
也会受到 Inception 实现、输入范围、真实数据划分和生成样本数影响；结论应视为本配置下的实验观察。

## v2：完整 U-Net、cosine 诊断与学习率策略

为继续冲击 FID≤15，在保持严格 200 epoch 约束的前提下，新增了完整的多级 skip
connection U-Net、sinusoidal time embedding 注入、EMA bank（0.999/0.9995/0.9999）、
warmup 和 cosine-LR 配置，并将 cosine beta schedule 与 linear beta schedule 分开做对照。
三组实验均使用 seed 44、CIFAR-10、5,000 张生成图像对 5,000 张无增强 train 图像计算 FID。

### v2 primary 结果

| 实验 | 训练策略 | 未裁剪 x0 | 裁剪 x0 | 训练状态 |
|---|---|---:|---:|---|
| R1 | full U-Net + linear beta + warmup 后固定 LR | 18.8045 | 18.8144 | 完成 |
| R2 | full U-Net + linear beta + warmup 后 cosine LR | 19.4615 | 19.4633 | 完成 |
| R3 | full U-Net + cosine beta + warmup 后固定 LR | 162.8797 | 16.1686 | 完成 |

默认评估权重为 EMA 0.9999。R3 的 cosine beta 未裁剪采样出现严重反向轨迹异常；逐步
裁剪预测的 x0 后 FID 从 162.8797 降到 16.1686，说明裁剪确实抑制了异常值传播，但仍
没有达到 15。R1 和 R2 的裁剪前后几乎不变，说明 x0 clipping 不是 linear 配置下的主要
瓶颈。R2 从有效的 step 70000 checkpoint 恢复到 step 78000；原有 final.pt 因 AutoDL
数据盘空间不足而损坏，恢复后新 final.pt 已通过加载验证。

### 当前 v2 判断

在 primary 结果中，最佳 FID 为 R3 裁剪 x0 的 16.1686。进一步在同一 R3 checkpoint、同一
采样协议下比较 EMA decay 后，EMA 0.999、0.9995、0.9999 和 raw 的 FID 分别为
15.7886、15.5340、16.1686 和 66.6401；完整记录见
`results/fid15_v2/ema_comparison_clipx0.md`。因此本轮最终最佳为 EMA 0.9995 的
15.5340，距离 FID≤15 仍差 0.5340，结论为未达标。

主要原因包括：

1. 当前训练预算只有严格 200 epoch，完整 U-Net 虽然提升了建模能力，但没有改变 CIFAR-10
   无条件 DDPM 在该预算下的优化难度；
2. linear beta 在本实现和评估协议下明显稳定，cosine beta 的未裁剪反向过程会积累极端
   `pred_x0`，导致 Inception 特征分布严重偏离；clipping 能修复大部分异常，但不能完全
   恢复 linear 的质量；
3. 从 step 70000 到 78000 的 cosine-LR 尾段训练没有带来收益，R2 的 19.4615 高于 R1
   的 18.8045，说明后期 LR 衰减设置在当前 checkpoint 和预算下并非有效改进；
4. FID 使用 5,000 对 5,000 图像，仍受生成 seed、Inception 实现、输入范围和 real split
   影响，因此 16.1686 与 15 的差距应通过固定协议下的复现实验确认，而不能通过更换
   数据子集或随机种子选择性报告。

### v2 补充实验：EMA、R3 历史点与 sampler 校验

为完成对当前结果的无重训诊断，R1、R2 在相同 `train` 前 5,000 张真实图、5,000 张生成图、
seed 44、裁剪 $x_0$ 协议下比较 EMA 0.999 和 0.9995：

| 实验 | EMA 0.999 | EMA 0.9995 |
|---|---:|---:|
| R1 full linear default | 18.9792 | 18.5210 |
| R2 full linear cosine-LR | 20.3343 | 20.1602 |

同一协议下，R3 EMA 0.9995 的历史 FID 为：

| 训练步数 | FID |
|---:|---:|
| 10,000 | 40.9638 |
| 20,000 | 24.6445 |
| 30,000 | 19.8270 |
| 40,000 | 17.9806 |
| 50,000 | 16.9291 |
| 60,000 | 16.4440 |
| 70,000 | 15.9460 |
| 最终 checkpoint | **15.5340** |

历史结果单调接近最终结果，没有发现最终步数之前更低的 checkpoint。最佳 R3 EMA 0.9995
裁剪样本网格单独保存为 `results/fid15_v2/R3_full_cosine_default_ema9995_clipx0_grid.png`；
它不代表旧的 EMA 0.9999 未裁剪网格。

此外，新增 `validate_sampler.py`，用固定 $x_t$、固定 epsilon 预测，在 `t=999,998,0`
分别比较 production `p_sample` 与独立实现的 posterior mean、variance 和裁剪路径。cosine
最大误差为 `5.7964e-7`，linear 最大误差为 `1.2450e-4`，均低于 `2e-4` 容差。linear
误差峰值来自 `t=0` 直接 epsilon 公式与 float32 schedule 系数的近似相减，属于数值表示误差，
不是 posterior 公式不一致。

R3 轨迹诊断补充了 median、P95、P99 和 max。例如无裁剪 t=999 的 `pred_x0` 为
`90.15/280.88/385.49/964.98`，裁剪 t=0 后的 P95/P99 约为 `0.94/0.99`。这说明 cosine
末端小 $α\_bar$ 会放大 epsilon 预测误差；最大值异常本身不能证明 schedule 实现错误，
但也说明逐步裁剪是当前采样协议下必要的稳定化操作。

## R5/R6：最终 200 epoch 复核

在 v2 和 R4 的基础上，继续完成两组独立的严格 200 epoch 实验。两组均使用完整 U-Net、
cosine beta schedule、seed 44、batch size 128、5,000 步 warmup、后期 cosine learning-rate
decay（从 step 58,500 开始由 `2e-4` 衰减到 `2e-5`）、EMA bank `0.999/0.9995/0.9999`，
并固定用 `train` split 前 5,000 张真实图和 5,000 张生成图、seed 44、裁剪预测 $x_0$ 评估。

R5 使用 uniform epsilon MSE，R6 只将训练损失替换为 Min-SNR-$\\gamma=5$，其余变量保持一致。

| 实验 | 训练损失 | EMA 0.9990 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R5 late decay | uniform MSE | 16.3823 | 16.2361 | **15.4385** | 16.1927 |
| R6 late decay | Min-SNR-$\\gamma=5$ | 17.2559 | 16.9445 | **15.8562** | 17.8510 |

R5 训练耗时约 155.0 分钟，R6 训练耗时约 153.1 分钟；评估的 EMA 对照是在训练完成后
按固定协议逐组生成。当前最佳为 R5 EMA 0.9999 的 `15.4385`，距离目标差 `0.4385`，
因此结果仍未达到 FID≤15。R6 最佳值比 R5 高 `0.4177`，表明 Min-SNR-$\\gamma=5$ 在
当前模型容量、噪声 schedule、学习率后期衰减和训练预算下没有带来改善。

这两组结果也补充了三个判断：

1. EMA 0.9999 在最终 checkpoint 上优于 0.999 和 0.9995，说明模型后期更新幅度仍然较大，
   更长时间常数的平均可能抑制了采样噪声；
2. R5 的 raw FID 为 16.1927，而 R6 的 raw FID 为 17.8510，Min-SNR 变体不仅没有改善 EMA，
   raw 权重也更差，因此不能把差异归因于 EMA 选择；
3. R5 与目标的差距只有约 0.44，但仍应通过同一固定协议确认，不能用更换 real split、seed
   或生成数量的方式宣称达标。

R5/R6 的配置、原始 FID 文件、loss history、loss 曲线、周期样本、最终样本网格和 AutoDL
日志保存在 `results/fid15_final/`。最终 checkpoint 与轻量结果归档通过
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)
发布。
另外，已按最佳 EMA 0.9999 权重补生成两组 64 张裁剪样本网格；它们分别保存为
`r5_late_decay/best_grid_ema9999_clipx0.png` 和
`r6_min_snr_late_decay/best_grid_ema9999_clipx0.png`，避免使用 EMA 0.9995 网格代表最佳 FID。

## Stage 1：不重训诊断

为了区分 FID 差距来自随机采样波动、评估实现还是模型本身，针对 R3 和 R5 做了三类
不重训诊断：固定真实图像为 CIFAR-10 train split 前 5,000 张，生成 5,000 张，使用
seed 44/45/46 和裁剪预测 `x0` 的相同协议；用真实 train 前 5,000 张与接下来 5,000 张
做 real-vs-real FID；并对前 256 张 train/test 图像使用固定噪声，在 20 个 timestep 上
统计 epsilon MSE、`x0` 误差和裁剪前越界像素比例。

| 实验 | seed 44 | seed 45 | seed 46 | 均值 | 范围 |
|---|---:|---:|---:|---:|---:|
| R3 EMA 0.9995 | 15.5340 | 15.6575 | 15.4022 | 15.5312 | 0.2553 |
| R5 EMA 0.9999 | 15.4385 | 15.4678 | 15.2086 | 15.3717 | 0.2592 |

R5 相比 R3 在正式 seed 44 上改善 `0.0955`，三 seed 均值改善约 `0.1595`，但没有一次
低于 15。real-vs-real FID 为 `10.2039`；两组真实图像的 uint8 均值为 `120.24/121.42`、
标准差为 `64.679/64.651`，范围均为 `0–255`。因此没有证据表明当前 FID 差距主要由输入
转换或真实数据管线错误造成。

R5 的 `alpha_bar` 在 `t=999` 为约 `2.43e-9`，在 `t=998` 为约 `2.43e-6`。虽然 epsilon
MSE 只有约 `3.6e-5`，但反推 `x0` 时会乘以 `1/sqrt(alpha_bar)`，导致 t=999 的 mean abs
误差约 `94.43`、99.3% 像素越界，t=998 也有约 76.9% 像素越界。这是低信噪比端点的
数值放大现象；先前独立 posterior 对照已确认 production sampler 的 mean、variance 和
clipping 路径公式一致。因此 clipping 是必要的稳定化措施，但不能解释中间 timestep 的
建模误差，也不能单独把 FID 从 15.4 推到 15 以下。

R3/R5 的 train/test timestep 指标基本重合，例如 R5 的 `t=500` mean abs x0 error 为
`0.128902/0.127885`，`t=800` 为 `0.224260/0.222263`。这不支持“模型只记住训练集”
是当前 0.4 FID 差距主因的判断。更合理的结论是：当前完整 U-Net、cosine schedule、
epsilon-prediction、200 epoch 和 EMA 组合已经接近该配置的能力上限，但在中高噪声段仍有
足以影响 Inception 特征的系统性误差，同时 FID 本身还存在约 0.26 的 seed range。

完整数据和复现脚本见 [`results/fid15_stage1/README.md`](results/fid15_stage1/README.md)
与 [`stage1_diagnostics.py`](stage1_diagnostics.py)。

### Stage 1 补充：真实子集方差实验

针对“随机抽取 5,000 张真实图像会不会改变 FID”的问题，使用已有 R3/R5 checkpoint 做了
不重训矩阵。协议固定为 CIFAR-10 train split、5,000 张真实图、5,000 张生成图、裁剪
预测 `x0` 和生成 seed `44/45/46`。除了正式的前 5,000 张真实图，还构造了 4 组由每类
500 张组成的、互不重叠的 class-stratified 子集，子集种子为 `20260919`，因此得到
`2 × 5 × 3 = 30` 个分数。

| 实验 | EMA | 全部 15 格均值 ± 样本标准差 | 最低 | 最高 | 正式子集均值 |
|---|---:|---:|---:|---:|---:|
| R3 | 0.9995 | 15.5096 ± 0.1104 | 15.3157 | 15.6866 | 15.5312 |
| R5 | 0.9999 | 15.3590 ± 0.1270 | 15.1445 | 15.5563 | 15.3716 |

所有 30 个分数都大于 15。最低分 `15.1445` 仍高于目标 `0.1445`；正式报告应继续采用
R5、正式前 5,000 张真实图、生成 seed 44 的 `15.4385`，而不是选择最低随机子集。该
实验证明真实子集选择会影响 FID，但影响属于评估方差，不能被解释为模型质量已经达到
阈值。例如同一 R5/seed44 从正式子集的 `15.4385` 变为 `stratified_1` 的 `15.2631`，
下降 `0.1754`，但在 4 组随机子集和 3 个生成 seed 的完整矩阵中仍没有一组低于 15。

原始 30 行记录、聚合统计和复现脚本见
[`results/fid15_stage1/real_subset/`](results/fid15_stage1/real_subset/) 和
[`stage1_real_subset_fid.py`](stage1_real_subset_fid.py)。

## R7：v-prediction 实验

为验证 v-prediction 是否能缓解 cosine schedule 低信噪比端点的 epsilon 误差放大，
在 R5 的完整 U-Net、cosine beta、200 epoch、78,000 steps、seed 44、warmup 和后期
learning-rate decay 设置上，仅切换训练目标与对应的采样转换。R7 训练耗时 156.6 分钟，
并在固定的 CIFAR-10 train 前 5,000 张真实图、5,000 张生成图、seed 44、裁剪预测
`x0` 协议下比较 EMA bank 与 raw 权重。

| 实验 | prediction target | EMA 0.9990 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R7 | v-prediction | 19.0443 | 18.8484 | **17.4808** | 19.2896 |

R7 最佳结果为 EMA 0.9999 的 `17.4808`，比 R5 最佳 `15.4385` 高 `2.0423`。因此，
在本项目的实现和固定评估协议下，v-prediction 没有带来预期收益。EMA 0.9999 仍明显
优于 EMA 0.999/0.9995 和 raw，说明权重平均仍然有效，但改善不足以抵消 prediction
target 切换造成的性能损失。完整配置、结果、loss、日志和样本网格见
[`results/fid15_r7_v_prediction/`](results/fid15_r7_v_prediction/)。

R7 结果进一步缩小了问题范围：R5 与 R7 使用相同的完整 U-Net、训练预算、beta schedule、
clipping 和评估协议，主要变量是 prediction target。R7 的明显退化表明，当前瓶颈不能
简单归因于 epsilon 参数化在端点的不稳定；更可能是 v-target 与现有 uniform loss、
网络容量、时间步采样及 clipping 之间的组合不匹配。后续改进应以 R5 为基线，采用单变量
消融验证 loss weighting、cosine schedule 参数、时间步重采样和更合适的采样器，避免同时
切换 prediction target、网络结构和训练策略。
