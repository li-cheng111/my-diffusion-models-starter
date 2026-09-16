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
