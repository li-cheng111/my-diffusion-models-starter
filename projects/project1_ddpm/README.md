# Project 1：From-Scratch DDPM 实验报告

本文件整合了 project1_ddpm 基础档、进阶档、挑战档、
FID 改进、学习率筛选、采样诊断和复现说明。

## 一、全部实验结果总览

| 实验条件 | 数据与预算 | 主要设置 | EMA FID（0.999 / 0.9995 / 0.9999） | Raw / 其他指标 | 结论 |
|---|---|---|---:|---:|---|
| MNIST 基础档 | MNIST，50 epoch，23,400 steps | linear beta，$`\epsilon`$ prediction | — | FID 32.8913；末次 loss 0.01151；24.8 min | 基础实现、训练、采样完成 |
| CIFAR-10 进阶基线 | CIFAR-10，200 epoch，78,000 steps | linear beta，$`\epsilon`$ MSE，EMA | — / — / 19.2879 | raw 28.7464；98.2 min | 进阶档完成，未达 $`\mathrm{FID}\le 15`$ |
| Schedule50 linear 汇总 | CIFAR-10，50 epoch，3 seeds，各 19,500 attempted batches | 完整 U-Net，linear，clip $`x_0`$ | 28.9522 ± 0.2274 / 29.3787 ± 0.2046 / 189.3650 ± 72.2358 | 固定 5,000/5,000 FID 协议 | 短预算下 0.999/0.9995 高于 cosine |
| Schedule50 cosine 汇总 | CIFAR-10，50 epoch，3 seeds，各 19,500 attempted batches | 完整 U-Net，cosine，clip $`x_0`$ | 23.9276 ± 0.3997 / 24.3862 ± 0.2908 / 152.3518 ± 8.7018 | 固定 5,000/5,000 FID 协议 | 0.999/0.9995 优于 linear；0.9999 仍很差 |
| seed43 续训 linear | CIFAR-10，50→200 epoch | 完整 U-Net，linear，新增约 58,500 steps | 18.9129 / 18.3699 / 18.9244 | 续训 376.1 min；最佳 EMA 0.9995 |
| seed43 续训 cosine | CIFAR-10，50→200 epoch | 完整 U-Net，cosine，新增约 58,500 steps | 15.9393 / 15.5792 / 16.3553 | 续训 390.7 min；最佳 EMA 0.9995 |
| 改版R1 full linear default | 200 epoch，seed 44 | 完整 U-Net，linear，固定 LR | 18.9792 / 18.5210 / 18.8144 | 未裁剪 18.8045 | 完整 U-Net 基线 |
| 改版R2 full linear cosine-LR | 200 epoch，seed 44 | 完整 U-Net，linear，LR 尾段衰减 | 20.3343 / 20.1602 / 19.4633 | 未裁剪 19.4615 | LR 尾衰减在 linear 上未改善 |
| 改版R3 full cosine default | 200 epoch，seed 44 | 完整 U-Net，cosine，固定 LR | 15.7886 / 15.5340 / 16.1686 | raw 66.6401；未裁剪 162.8797 | 裁剪后大幅恢复，最佳 15.5340 |
| 改版R4 Min-SNR | 200 epoch，seed 44 | cosine + Min-SNR $`\gamma=5`$，无 late decay | 16.3013 / 15.7954 / 16.5633 | raw 29.1437 | 不如 R5 |
| 改版R5 late decay | 200 epoch，seed 44 | cosine + uniform $`\epsilon`$ MSE + LR late decay | 16.3823 / 16.2361 / 15.4385 | raw 16.1927 | **当前正式最佳，但仍高 0.4385** |
| 改版R6 Min-SNR late decay | 200 epoch，seed 44 | R5 + Min-SNR $`\gamma=5`$ | 17.2559 / 16.9445 / 15.8562 | raw 17.8510 | Min-SNR 在本配置下退化 |
| 改版R7 v-prediction | 200 epoch，seed 44 | R5 设置，$`\epsilon\to v`$ target | 19.0443 / 18.8484 / 17.4808 | raw 19.2896 | 单独切换 v-prediction 无效 |
| 调整学习率LR=5e-5 | 200 epoch，seed 44 | cosine，peak 5e-5，late decay | 17.4472 / 17.3407 / 16.5231 | raw 18.6642 | 30 epoch 最优候选未延续 |
| 调整学习率LR=2.5e-4 | 50 → 200 epoch，seed 44 | cosine，peak 2.5e-4，late decay | 16.0761 / 15.8897 / 15.6761 | raw 16.8996 | 不如 R5 的 0.9999 |
| **R5 seed 测试** | 同一 R5 checkpoint | seed 44/45/46，5,000/5,000 | EMA 0.9999条件下，各seed对应**15.4385 / 15.4678 / 15.2086** | 均值 15.3717 | **R5在以上测试中效果最佳，FID最可能接近15** |

当前全项目的最佳固定协议值仍为 seed=46 下的 **R5 EMA 0.9999 = 15.2086**。本次 seed43 续训对照中，cosine 最佳为 15.5792，尚未达到 FID ≤ 15。

## 二、实现内容

- schedule.py：linear/cosine beta schedule、$`\alpha`$、累积 $`\bar{\alpha}`$、posterior
  mean 系数和 posterior variance；
- diffusion.py：闭合形式前向加噪 q_sample、$`\epsilon`$/$`v`$ 训练损失、单步反向采样
  p_sample、完整 p_sample_loop；
- model/embedding.py：sinusoidal timestep embedding；
- model/unet.py：带时间条件的 ResBlock、下采样/上采样和 skip connection，
  并在 16×16 分辨率使用 attention；
- dataset.py：MNIST/CIFAR-10 数据加载，统一归一化到 [-1, 1]；
- train.py：AdamW、warmup、EMA、AMP、梯度裁剪、checkpoint 和 loss history；
- sample.py：按指定 EMA/raw 权重生成样本网格；
- evaluate.py：固定真实数据 split、样本数、seed，支持多 EMA 对照；
- challenge.py：linear/cosine × 多 seed 的挑战档编排和汇总；
- monitor.py、experiment_monitor.py、challenge_monitor.py：只读训练进度监控；
- tests/：schedule、diffusion 和模型的静态测试源码。

前向过程的核心公式为：

```math
\begin{aligned}
x_t &= \sqrt{\bar{\alpha}_t}\,x_0\quad+ \sqrt{1 - \bar{\alpha}_t}\,\epsilon
\end{aligned}
```

训练时随机采样 $`t`$，直接由闭合形式得到 $`x_t`$，模型预测噪声 $`\epsilon_\theta(x_t,t)`$，
默认目标为：

```math
\mathcal{L}_{\mathrm{simple}} = \mathrm{mean}\!\left[\left(\epsilon - \epsilon_\theta(x_t,t)\right)^2\right]
```

反向采样从标准高斯噪声开始，按 $`T-1`$ 到 $`0`$ 逐步计算 posterior mean；$`t=0`$ 不再加入随机噪声。

## 三、评估协议

### CIFAR-10 FID

使用固定的 CIFAR-10 train split 前 5,000 张真实图像，关闭随机增强，
每次生成 5,000 张图，生成 seed 为 44。模型分别评估 EMA 0.999、0.9995、0.9999
和 raw；R3–R7 的主结果使用 clip_denoised=True，即在每个反向步骤将预测的 $`\hat{x}_0`$
裁剪到 [-1, 1]。

此评估方法让不同训练方案在尽量相同真实子集、生成 seed、样本数和采样处理下进行单变量比较。


## 四、基础档与进阶档

### 4.1 MNIST 基础档

MNIST 基线使用 linear schedule、50 epoch、23,400 steps，最后记录 loss 为 0.01151，使用 EMA 生成 5,000 张样本的FID 为 32.8913。该结果的主要意义是验证从加噪、时间嵌入、ResBlock 注入到反向
采样的完整链路。

<p align="center">
  <img src="runs/exp_mnist_baseline/loss_curve.png" alt="图1 MNIST 基础档训练损失曲线" />
</p>
<p align="center"><em>图 1　MNIST 基础档 50 epoch 训练损失曲线</em></p>

<p align="center">
  <img src="runs/exp_mnist_baseline/samples_inference_ema/grid.png" alt="图2 MNIST 基础档 EMA 样本网格" />
</p>
<p align="center"><em>图 2　MNIST 基础档最终 EMA 样本网格</em></p>

### 4.2 CIFAR-10 进阶档

进阶基线使用 CIFAR-10、200 epoch、78,000 steps、linear beta 和 $`\epsilon`$ MSE，维护
raw 与 EMA 权重。最终 loss 为 0.01938，训练约 98.2 分钟：

- EMA FID：19.2879；
- raw FID：28.7464；
  
因此 EMA 对采样质量有明显帮助，但这个单次结果距离 $`\mathrm{FID}\le 15`$ 仍有 4.2879。

<p align="center">
  <img src="runs/exp_cifar10_advanced/loss_curve.png" alt="图3 CIFAR-10 进阶基线训练损失曲线" />
</p>
<p align="center"><em>图 3　CIFAR-10 进阶基线训练损失曲线</em></p>

<p align="center">
  <img src="runs/exp_cifar10_advanced/samples_inference_ema/grid.png" alt="图4 CIFAR-10 进阶基线 EMA 样本网格" />
</p>
<p align="center"><em>图 4　CIFAR-10 进阶基线最终 EMA 样本网格</em></p>

## 五、挑战档：linear 与 cosine

### 1）Schedule50：50 epoch 

使用 seed 42、43、44 分别训练 linear 与 cosine，每组固定完整 U-Net、batch size 128、$`T=1000`$、$`\epsilon`$ prediction、AdamW 学习率 $`2\times10^{-4}`$、5,000 次成功更新 warmup、50 epoch（每组 19,500 个 batch）、EMA 0.999/0.9995/0.9999，并在每个反向步骤裁剪预测的 $`x_0`$。
FID 固定为 CIFAR-10 train split 前 5,000 张真实图、5,000 张生成图、生成 seed 44、batch size 64、1,000 步 DDPM 和 torch-fidelity Inception 特征。每个单元按 EMA 0.999 / 0.9995 / 0.9999 排列，数值越低越好。

| 训练 seed | Linear FID（0.999 / 0.9995 / 0.9999） | Cosine FID（0.999 / 0.9995 / 0.9999） |
|---:|---:|---:|
| 42 | 29.2144 / 29.6127 / 136.0274 | 23.4786 / 24.0694 / 159.5961 |
| 43 | 28.8330 / 29.2898 / 160.4971 | 24.0592 / 24.4483 / 142.6996 |
| 44 | 28.8092 / 29.2335 / 271.5706 | 24.2449 / 24.6410 / 154.7598 |
| 均值 ± 样本标准差 | 28.9522 ± 0.2274 / 29.3787 ± 0.2046 / 189.3650 ± 72.2358 | 23.9276 ± 0.3997 / 24.3862 ± 0.2908 / 152.3518 ± 8.7018 |

在 0.999 和 0.9995 两档 EMA 下，cosine 在三个 seed 上都比 linear 低约 4.6–5.7 FID，跨 seed 均值分别低 5.0246 和 4.9925，方向一致且幅度明显，说明**50 epoch 时 cosine**效果会更好一点。但是在 EMA 0.9999 主指标下，两种 schedule 都出现异常高 FID，最可能的原因是 **EMA 平均周期过长**，当前实验从随机初始化开始使用固定 decay，经过约 19,500 次更新后，初始化参数仍保留约 14.23% 的权重，初始化与前 5,000 步 warmup 阶段合计约占 23.46%，使 EMA 模型明显滞后于已经训练好的模型，这种去噪偏差经过 1,000 步采样累积，可能导致生成质量严重下降。

### 2）Schedule200：200 epoch

在50epoch基础上进行续训，linear 与 cosine 两组仍都使用 CIFAR-10、相同训练 seed、同一套评估协议和每个反向步骤的 $`x_0`$ clipping。FID 固定为 train split 前 5,000 张真实图、5,000 张生成图、生成 seed 44、batch size 64、1,000 步 DDPM 与 torch-fidelity Inception 特征。

| Schedule | 训练 seed / 预算 | EMA 0.999 | EMA 0.9995 | EMA 0.9999 | Raw |
|---|---|---:|---:|---:|---:|
| linear | 43，50→200 epoch | 18.9129 | **18.3699** | 18.9244 | 未评估 |
| cosine | 43，50→200 epoch | 15.9393 | **15.5792** | 16.3553 | 未评估 |

续训到 200 epoch 后，FID 对应值分别降至 18.9129 / 18.3699 / 18.9244 与 15.9393 / 15.5792 / 16.3553。可以看到**异常高的 EMA 0.9999 FID 在增加训练预算后显著恢复**，说明50 epoch下 EMA 平均周期过长可能就是主要的影响因素。

在这个配对续训中，cosine 的 FID 在三档 EMA 下都低于 linear，分别改善 2.9736、2.7907 和 2.5691；两种 schedule 的最佳 EMA 都是 0.9995，说明在本实验条件下，**可以确定 cosine 效果优于 linear**。

## 六、改进实验

### 6.1 R1/R2/R3：完整 U-Net、linear schedule 与固定学习率

三组均严格保持 200 epoch、约 78,000 steps、seed 44 和固定正式 FID 协议：

| 实验 | 变化 | EMA 0.999 | EMA 0.9995 | EMA 0.9999 | 其他 |
|---|---|---:|---:|---:|---|
| R1 | 完整 U-Net + linear beta + warmup 后固定 LR | 18.9792 | 18.5210 | 18.8144 | 未裁剪 0.9999：18.8045 |
| R2 | R1 + linear beta + cosine LR 尾段 | 20.3343 | 20.1602 | 19.4633 | 未裁剪 0.9999：19.4615 |
| R3 | 完整 U-Net + cosine beta + warmup 后固定 LR | 15.7886 | **15.5340** | 16.1686 | 未裁剪 0.9999：162.8797 |

R3 的 EMA decay 直接说明 EMA 不是越大越好，本 checkpoint 上 0.9995 优于 0.9999。

### 6.4 R4/R5/R6：Min-SNR 与后期学习率衰减

R4、R5、R6 都使用完整 U-Net、cosine beta、200 epoch、batch 128、seed 44、warmup、
EMA 对照和裁剪 $`x_0`$：

| 实验 | 训练目标/学习率策略 | EMA 0.999 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R4 | Min-SNR $`\gamma=5`$，无 late decay | 16.3013 | **15.7954** | 16.5633 | 29.1437 |
| R5 | uniform $`\epsilon`$ MSE，late decay $`2\times10^{-4}\to2\times10^{-5}`$ | 16.3823 | 16.2361 | **15.4385** | 16.1927 |
| R6 | Min-SNR $`\gamma=5`$ + late decay | 17.2559 | 16.9445 | **15.8562** | 17.8510 |

R5 比 R4 的最佳值改善 0.3569，说明在当前配置下后期学习率衰减有效；R6 比 R5 高 0.4177，说明直接使用 Min-SNR $`\gamma=5`$ 没有解决当前问题。

<p align="center">
  <img src="results/fid15_final/r5_late_decay/loss_curve.png" alt="图5 R5 训练损失曲线" />
</p>
<p align="center"><em>图 5　R5 cosine beta + uniform epsilon MSE + late decay 训练损失曲线</em></p>

<p align="center">
  <img src="results/fid15_final/r5_late_decay/best_grid_ema9999_clipx0.png" alt="图6 R5 EMA 0.9999 逐步裁剪 x0 样本网格" />
</p>
<p align="center"><em>图 6　R5 EMA 0.9999、逐步裁剪预测 x_0 的最佳样本网格</em></p>

### 6.5 R7：v-prediction

R7 保持 R5 的 U-Net、cosine beta、200 epoch、78,000 steps、warmup、late decay、
seed 和 clipping，只把 $`\epsilon`$ prediction 改为 $`v`$-prediction：

| 权重 | FID |
|---|---:|
| EMA 0.9990 | 19.0443 |
| EMA 0.9995 | 18.8484 |
| EMA 0.9999 | **17.4808** |
| raw | 19.2896 |

R7 最佳值比 R5 高 2.0423，说明当前 uniform v-loss 与 timestep 分布、cosine
schedule 和采样方式组合不适合直接替代 R5；如果以后再研究 v-prediction，必须同时设计与 target 匹配的 loss weighting。

## 七、学习率实验

### 7.1 30 epoch 粗筛

7 组实验使用完整 U-Net、cosine beta、$`\epsilon`$ prediction、seed 44、batch 128、
1,000 steps warmup 和 1,000/1,000 快速 FID：

| 学习率 | 最终 loss | 快速 FID | 判断 |
|---:|---:|---:|---|
| 5e-5 | 0.05284 | **238.0244** | 快速筛选最优 |
| 8e-5 | 0.05245 | 263.5649 | 次优 |
| 1.2e-4 | 0.05214 | 291.7750 | 淘汰 |
| 1.6e-4 | 0.05209 | 284.0970 | 淘汰 |
| 2e-4 | 0.05199 | 307.1943 | 淘汰 |
| 2.5e-4 | 0.05193 | 330.7679 | 淘汰 |
| 3e-4 | 0.05207 | 323.4893 | 淘汰 |

高学习率的 loss 略低而 FID 更差，说明平均 $`\epsilon`$ MSE 不能替代完整反向采样质量。
由于只有 30 epoch 和 1,000 个样本，该结果只能排除明显不合适的候选。

### 7.2 LR=5e-5 完整 200 epoch

5e-5 从粗筛候选进入完整复核，设置为 peak 5e-5、late decay 到 5e-6：

| 权重 | LR=5e-5 | R5 | 差值 |
|---|---:|---:|---:|
| EMA 0.9990 | 17.4472 | 16.3823 | +1.0649 |
| EMA 0.9995 | 17.3407 | 16.2361 | +1.1046 |
| EMA 0.9999 | 16.5231 | **15.4385** | +1.0846 |
| raw | 18.6642 | 16.1927 | +2.4715 |

完整训练证明 30 epoch 最优候选并不等于 200 epoch 最优候选。当前 200 epoch 预算下，R5 的 2e-4 峰值学习率更能完成有效拟合。

### 7.3 LR=2.5e-4 完整 200 epoch

从 50 epoch checkpoint 继续到 200 epoch，新增 58,450 个有效更新：

| 权重 | LR=2.5e-4 | R5 的 LR=2e-4 | 差值 |
|---|---:|---:|---:|
| EMA 0.9990 | 16.0761 | 16.3823 | -0.3062 |
| EMA 0.9995 | 15.8897 | 16.2361 | -0.3464 |
| EMA 0.9999 | 15.6761 | **15.4385** | +0.2376 |
| raw | 16.8996 | 16.1927 | +0.7069 |

0.9990/0.9995 在该复核中较低，但通常选择的 0.9999 反而高于 R5，说明提高峰值学习率的效果并不佳。

## 八、一些诊断

### 8.1 cosine 末端的数值放大

cosine schedule 末端约为：

```math
\begin{aligned}
t=998:\quad \bar{\alpha}_t &\approx 2.43\times10^{-6},\\
t=999:\quad \bar{\alpha}_t &\approx 2.43\times10^{-9}.
\end{aligned}
```

$`\epsilon`$ prediction 反推 $`x_0`$ 时需要除以 $`\sqrt{\bar{\alpha}_t}`$，因此很小的 $`\epsilon`$ 误差会被放大。诊断中记录的典型现象为：

| $`t`$ | $`\epsilon`$ MSE | mean abs $`x_0`$ error | 越界比例 |
|---:|---:|---:|---:|
| 800 | 0.008703 | 0.224260 | 0.0030 |
| 990 | 0.000077 | 0.499628 | 0.0340 |
| 998 | 0.000034 | 2.889602 | 0.7689 |
| 999 | 0.000036 | 94.428703 | 0.9929 |

R3 无裁剪时 pred_x0_abs_max 约 964.98、越界比例约 99.43%，导致 FID达到162.8797；逐步裁剪 $`x_0`$ 后降至约 16，可以说明 clipping 是必要的数值稳定化措施。

### 8.2 seed 与真实子集方差

R5 EMA 0.9999 的生成 seed 结果为：

~~~text
seed44 = 15.4385
seed45 = 15.4678
seed46 = 15.2086
均值    = 15.3717
~~~

对 R3 和 R5 各使用 15 个子集/生成 seed 组合：

| 实验 | 均值 | 样本标准差 | 最低 | 最高 |
|---|---:|---:|---:|---:|
| R3 EMA 0.9995 | 15.5096 | 0.1104 | 15.3157 | 15.6866 |
| R5 EMA 0.9999 | 15.3590 | 0.1270 | 15.1445 | 15.5563 |

随机性可以造成约 0.1–0.2 的变化，但都无法稳定达到 $`\mathrm{FID}\le 15`$。

## 九、对于 FID 仍未达到 15 的分析

全项目此前最佳固定协议结果仍是 R5 seed46、EMA 0.9999 的 15.2086（高于 15 的
目标 0.2086）；本次最新 seed43 续训的最佳值是 cosine EMA 0.9995 = 15.5792，距目标
0.5792。它优于本次 linear 对照，但没有刷新 R5。结合这些结果，FID 未达标不是单一
bug，而是以下因素叠加：

1. **cosine 末端的 $`\epsilon`$ 误差放大。** 极小 $`\bar{\alpha}_t`$ 使最后几个 timestep 的
   $`x_0`$ 反推非常敏感；全程 clipping 能避免灾难性爆炸，但会引入系统性截断。
2. **中高噪声区间的误差仍在多步链中累积。** 即使最后一步不爆炸，中高噪声阶段的
   轮廓、颜色和局部纹理误差会逐步传递到 Inception 特征，像素级 MSE 很小也不代表
   FID 已经足够低。
3. **训练 loss 与 FID 目标不完全一致。** uniform $`\epsilon`$ MSE 对 timestep 等权，
   但不同 timestep 对最终视觉质量的贡献不等价；R6 的 Min-SNR $`\gamma=5`$ 退化说明
   简单地重新加权并不能保证有效。
4. **网络容量与采样器仍有限。** 当前 U-Net 已有完整 skip connection 和 16×16
   attention，但对 32×32 CIFAR-10 的细节建模能力、timestep 覆盖和 ancestral
   sampling 误差仍可能限制最后约半个 FID。
5. **EMA 只能降低参数噪声，不能修正系统误差。** EMA 显著优于 raw，但不同 decay
   的最优点会随训练轨迹变化，EMA 不是一个可以固定解决所有问题的独立变量。
6. **训练预算会改变 EMA 的可用性，短训结果不能外推到长训。** seed43 的
   50→200 epoch 连续轨迹中，EMA 0.9999 从 50 epoch 的 160.4971（linear）/
   142.6996（cosine）降到 18.9244 / 16.3553；EMA 0.999 和 0.9995 也均明显改善。
   这符合较大 decay 在短训练时对近期参数变化响应偏慢、训练延长后逐步追上的解释；
   但单条轨迹无法证明它是唯一原因。200 epoch 时 0.9995 仍是两种 schedule 的最佳
   decay，说明“EMA 0.9999 随训练变好”不等于“EMA 越大越好”。

### 为什么旧实验看起来 linear 更好，而复核中 cosine 更好

这不是同一实验条件下 schedule 排名反转，而是旧实验把 schedule 与实现差异混在了一起：

- 旧挑战矩阵使用基础 U-Net 和未逐步裁剪预测 $`x_0`$ 的采样路径。cosine 端点的
  $`\bar{\alpha}_t`$ 很小时，噪声预测误差会被反推公式放大，生成样本因此严重失真，
  FID 达 129–145；旧表里的 linear 与 cosine 数值不应作为 schedule 的公平比较。
- 修正为完整 U-Net、统一逐步 $`x_0`$ clipping 和固定评估协议后，50 epoch 三 seed
  均值中 cosine 的 EMA 0.999 / 0.9995 分别为 23.9276 / 24.3862，优于 linear 的
  28.9522 / 29.3787。EMA 0.9999 在 50 epoch 都很差，表明短预算下高 decay 评估不稳定。
- 更有针对性的 seed43 50→200 epoch 配对续训中，cosine 的三个 EMA FID 均低于 linear：
  cosine `15.9393 / 15.5792 / 16.3553`，linear `18.9129 / 18.3699 / 18.9244`；
  cosine 分别低 2.9736 / 2.7907 / 2.5691。该结果加强了本配置下 cosine 占优的证据，
 但 200 epoch 只有一个训练 seed，不能据此宣称普遍规律。
- 本次 cosine 最优 15.5792 仍高于目标 0.5792，也高于项目此前 R5 seed46 的 15.2086；
  schedule 胜出并不意味着达到目标。剩余差距需要继续考察训练目标、模型容量、优化轨迹
  与采样质量，且要以固定协议和多 seed 复核。

因此，对当前项目最准确的结论是：旧版 129–145 的 cosine FID 主要暴露了采样实现问题；
修正实现后，50 epoch 三 seed 与 seed43 的 200 epoch 配对续训都显示 cosine 优于 linear。
但长训证据目前仅来自一个 seed，结论仍是“该配置下观察到”，不是 schedule 的普遍定律。

## 十、补充文件说明

### 10.1 仓库中的结果材料

- schedule.py、diffusion.py、dataset.py、train.py、sample.py、evaluate.py：核心代码；
- model/、configs/、tests/：模型、配置和测试源码；
- results/challenge/：六组挑战矩阵的逐 seed FID 原始文本；
- results/fid15_v2/：R1/R2/R3、EMA 对比、历史 checkpoint 评估和采样验证数据；
- results/fid15_4090_linear_seed44/：linear 续训到 100k；
- results/fid15_r4_min_snr/：R4；
- results/fid15_final/：R5/R6；
- results/fid15_stage1/：稳定性、real-vs-real、真实子集和 timestep 诊断数据；
- results/fid15_r7_v_prediction/：R7；
- results/lr_screen_30ep/、results/lr_screen_50ep/：学习率筛选；
- [SCHEDULE50_AUTODL.md](SCHEDULE50_AUTODL.md)：50 epoch linear/cosine 三 seed 对照的配置、结果和复现材料入口；
- results/fid15_lr5e5_full/、results/lr25_200ep/：两组完整学习率复核；
- results/schedule200_seed43/：两种 schedule 的 50→200 epoch 续训配置、FID 指标、评估/采样日志、六张样本网格、环境与 checkpoint 校验和；
- 各结果目录中的 .txt、.csv、.png、.yaml 保留原始数值、曲线、样本网格和实际配置。

### 10.2 GitHub Release 大文件

大型 checkpoint 不进入普通 Git，统一保存在
[challenge-v1 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)，
包括：

- r3_full_cosine_final.pt
- r4_min_snr_final.pt
- r5_uniform_late_decay_final.pt
- r6_min_snr_late_decay_final.pt
- r7_v_prediction_final.pt
- lr5e5_full_200ep_final.pt
- lr25_200ep_final.pt
- 30/50 epoch 学习率筛选 checkpoint、MNIST 临时产物和中间样本压缩包

本次 seed43 schedule 50→200 epoch 续训的两份精确 epoch-200 checkpoint 与复现包，另存于
[schedule200-seed43-v1 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule200-seed43-v1)。该 Release 附带 SHA-256 清单；仓库内保留小型配置、日志、FID 文本、loss history 和样本网格，不包含数据集或模型权重。

三 seed、50 epoch schedule 对照的 checkpoint 和复现包分别存于
[seed42](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed42-v1)、
[seed43](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed43-v1) 和
[seed44](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed44-v1) Releases。

## 十一、自查问题（逐题回答）

### 1. 公式映射：`q_sample` 中的表达式对应论文哪个公式？

代码中的

```text
x_t = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise
```

对应 DDPM 论文前向扩散过程的闭合形式：

```math
q(x_t\mid x_0)=\mathcal{N}\left(x_t;\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I\right),
```

等价的重参数化采样写法为：

```math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad\epsilon\sim\mathcal{N}(0,I).
```

其中 $`\bar{\alpha}_t=\prod_{s=1}^{t}(1-\beta_s)`$。因此 `q_sample` 不需要逐步执行前向链，就能直接从 $`x_0`$ 得到任意时间步的 $`x_t`$。

### 2. 形状广播：为什么要 reshape 成 `(B, 1, 1, 1)`？

按 batch 取出时间步后，`sqrt_alpha_bar[t]` 的形状是 `(B,)`，其中每个样本有一个自己的噪声强度。图像张量通常是 `(B, C, H, W)`，所以要把系数 reshape 为 `(B, 1, 1, 1)`，让同一个样本的标量系数广播到它的全部通道和空间位置。

如果不 reshape，PyTorch 会从最后一个维度尝试对齐 `(B,)`。这通常无法与 `(B, C, H, W)` 正确广播；即使在某些特殊尺寸下意外广播成功，也可能把 batch 维误当成空间或通道维，导致每个样本没有使用对应的 $`t`$。结果是加噪公式的语义和张量形状都不可靠。

### 3. 设备：为什么 `betas` 要使用 `register_buffer`？

`betas`、`alphas_cumprod` 和 posterior 系数是模型计算所需的固定张量，不是需要优化器更新的参数。使用 `register_buffer` 有三点作用：

1. 它们不会出现在可训练参数中，也不会被优化器更新；
2. 调用 `model.to(device)`、保存 `state_dict` 和加载 checkpoint 时，它们会跟随模型正确迁移和保存；
3. `q_sample`、`p_sample` 在 GPU 上索引这些张量时，不会因为仍停留在 CPU 而产生设备不一致错误。

如果只写成 `self.betas = ...`，它只是普通属性，不能可靠地参与模型的设备迁移和 checkpoint 状态管理。

### 4. 训练：为什么每个 step 的 `t` 是随机采样，而不是顺序遍历？

训练目标是对所有时间步的损失取期望。每一步随机采样 batch 内的 `t`，相当于用 Monte Carlo 估计这个期望：同一个 batch 可以覆盖不同噪声强度，长期训练后每个时间步都得到充分更新。

如果按 `0,1,2,...,T-1` 顺序遍历，单个 batch 会集中在一个噪声等级，训练更新会带有很强的顺序相关性；当数据量、epoch 或中途停止发生变化时，部分时间步还可能没有得到均衡训练。随机采样也避免了必须为每个 $`t`$ 单独保存和组织一遍数据。

### 5. 采样：为什么 `p_sample` 的最后一步（`t=0`）不加随机扰动？

反向 DDPM 的随机项来自 $`\sqrt{\tilde{\beta}_t}z`$。当到达 `t=0` 时，目标是得到最终的 $`x_0`$，不再需要把结果送入更低的噪声等级；此时实现中直接返回 posterior mean，不再采样新的 $`z`$。

如果在 `t=0` 仍加入随机扰动，输出会重新偏离模型预测的干净图像，增加最终像素和特征的不确定性，也不符合从 $`x_1$` 到 $`x_0$` 的最后一步定义。

### 6. EMA：实测使用 EMA 和不使用 EMA，FID 差多少？为什么？

进阶基线中，CIFAR-10 的 raw FID 为 `28.7464`，EMA 0.9999 FID 为 `19.2879`，EMA 改善 `9.4595`。在 R5 中，raw FID 为 `16.1927`，EMA 0.9999 FID 为 `15.4385`，改善 `0.7542`。

差异来自训练参数的逐步波动：raw 是最后一个优化器时刻的参数，可能包含 mini-batch 噪声、后期学习率变化或 AMP 更新带来的局部扰动；EMA 对连续训练轨迹做低通滤波，通常能提高反向采样链的稳定性。不过 EMA 不能消除 schedule、网络容量或采样公式的系统误差，所以在 R5 中收益已经小于进阶基线，而且不同 decay 的最优值也会变化。

### 7. Schedule：linear 和 cosine 在 50 epoch 时哪个更好？200 epoch 时呢？

完整 U-Net 的三训练 seed、50-epoch 对照显示，EMA 0.999 / 0.9995 下 cosine 的平均 FID（`23.9276 / 24.3862`）低于 linear（`28.9522 / 29.3787`）。EMA 0.9999 下两种 schedule 都出现严重退化（cosine `152.3518 ± 8.7018`，linear `189.3650 ± 72.2358`），所以该短预算下不能把 0.9999 作为可靠比较点。

在 seed43 的 schedule-specific checkpoint 上分别续训至 200 epoch 后，cosine 在三档 EMA 下都优于 linear：`15.9393 / 15.5792 / 16.3553` 对 `18.9129 / 18.3699 / 18.9244`。最佳 cosine 仍为 `15.5792`，没有达到 FID ≤ 15。该续训对照只有一个训练 seed，属于配对实测而不是三 seed 统计结果。

这与初始挑战矩阵（基础 U-Net、cosine 末端未正确裁剪，linear `19.2926 ± 0.3357`、cosine `137.5132 ± 8.0166`）看似相反，根因是比较条件不同：网络与 skip 结构、端点裁剪、训练阶段/预算以及 EMA 状态都改变了。结论不是“cosine 永远更好”，而是必须固定这些因素，并对足够训练预算下的最终生成结果进行对照。

## 十二、我学到的最有价值的一件事

最有价值的认识是：**训练 loss 下降并不等价于生成质量和 FID 一定改善，必须把 schedule、网络、采样器、EMA 和评估协议作为一条完整链路一起验证。**

本项目中，高学习率候选在 30 epoch 粗筛时的平均噪声预测 loss 略低，但快速 FID 更差；cosine 在初始实现中严重失败，加入逐步 $`x_0`$ clipping 和采样诊断后却成为后续最佳路线；R5 的 raw 与 EMA、不同 EMA decay 也给出了不同结论。这些对照让我从“看单个训练指标”转向“固定协议、隔离变量、用最终采样质量验证假设”。

## 十三、真实调试记录

### 记录 1：cosine 采样端点出现 $`x_0`$ 数值爆炸

- **现象：** 初始 cosine 挑战实验的 FID 达到 `129.2708–145.2832`，明显异常；R3 未裁剪 EMA 0.9999 的 FID 为 `162.8797`。
- **定位：** `diagnose_schedule.py` 和 `validate_sampler.py` 显示 cosine 在 `t=998/999` 时 $`\bar{\alpha}_t`$ 约为 `2.43×10^{-6}` 和 `2.43×10^{-9}`。epsilon prediction 反推 $`x_0`$ 时除以 $`\sqrt{\bar{\alpha}_t}`$，误差被放大；诊断记录的典型 `pred_x0_abs_max` 约为 `964.98`，越界比例约 `99.43%`。
- **处理与结果：** 在每个反向步骤把预测的 $`\hat{x}_0`$ 裁剪到 `[-1,1]`，并保留中位数/P95/P99/最大值诊断。R3 EMA 0.9995 的 FID 从未裁剪异常值恢复到 `15.5340`。

### 记录 2：AutoDL 数据盘空间不足影响 checkpoint 完整写入

- **现象：** 训练过程中 AutoDL 数据盘接近满载，最终 checkpoint 或中间实验产物出现写入不完整风险，监控页面也可能因后台进程或 SSH 断联而显示等待/断开。
- **定位：** 检查数据盘占用、`runs/` 和 checkpoint 文件大小，并对照训练日志中的最近 global step；确认问题来自存储空间和会话稳定性，而不是模型前向或反向公式。
- **处理与结果：** 保留已提交 GitHub/Release 的复现材料，删除确认不再需要的本地大文件，独立管理每个实验目录，并通过中间 checkpoint 恢复训练。后续实验采用后台会话和分阶段保存，避免单个最终文件成为唯一恢复点。

### 记录 3：AMP 更新跳步需要和训练状态一起检查

- **现象：** 混合精度训练日志中出现 scaler 溢出/optimizer step 被跳过的迹象，单看 epoch 进度容易误以为每个 batch 都产生了有效更新。
- **定位：** 同时检查 global step、AMP scaler、optimizer step 和 checkpoint 中保存的 raw/EMA 状态；区分“读过一个 batch”和“完成一次有效参数更新”。
- **处理与结果：** 训练记录改为报告有效更新步数，并在 checkpoint 中恢复 raw、optimizer、EMA 和 scaler 状态；续训配置按有效更新数计算目标，避免恢复后学习率、EMA 时间常数和总训练预算错位。

以上记录均对应本项目已经产生的实验日志、诊断数值或 AutoDL 运行过程；没有把尚未完成的 50 epoch linear/cosine 对照写成已有结果。
