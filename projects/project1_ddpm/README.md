# Project 1：From-Scratch DDPM 实验报告

本文件整合了 project1_ddpm 基础档、进阶档、挑战档、
FID 改进、学习率筛选、采样诊断和复现说明。

## 一、全部实验结果总览

| 实验条件 | 数据与预算 | 主要设置 | EMA FID（0.999 / 0.9995 / 0.9999） | Raw / 其他指标 | 结论 |
|---|---|---|---:|---:|---|
| MNIST 基础档 | MNIST，50 epoch，23,400 steps | linear beta，epsilon prediction | — | FID 32.8913；末次 loss 0.01151；24.8 min | 基础实现、训练、采样完成 |
| CIFAR-10 进阶基线 | CIFAR-10，200 epoch，78,000 steps | linear beta，epsilon MSE，EMA | — / — / 19.2879 | raw 28.7464；98.2 min | 进阶档完成，未达 FID≤15 |
| 挑战 linear，seed 42 | CIFAR-10，200 epoch | linear beta，基础 U-Net | — / — / 19.2879 | raw 28.7464 | 稳定优于同期 cosine |
| 挑战 linear，seed 43 | CIFAR-10，200 epoch | linear beta，基础 U-Net | — / — / 19.6306 | raw 42.7537 | seed 方差可见 |
| 挑战 linear，seed 44 | CIFAR-10，200 epoch | linear beta，基础 U-Net | — / — / 18.9593 | raw 34.9399 | 初始挑战矩阵中的最好 linear |
| 挑战 linear 汇总 | 3 seeds，200 epoch | linear beta | — / — / 19.2926 ± 0.3357 | raw 35.4800 ± 7.0193 | 初始挑战主线 |
| 挑战 cosine，seed 42 | CIFAR-10，200 epoch | cosine beta，基础 U-Net | — / — / 137.9856 | raw 283.4196 | 未裁剪端点异常明显 |
| 挑战 cosine，seed 43 | CIFAR-10，200 epoch | cosine beta，基础 U-Net | — / — / 129.2708 | raw 399.1676 | 失败案例 |
| 挑战 cosine，seed 44 | CIFAR-10，200 epoch | cosine beta，基础 U-Net | — / — / 145.2832 | raw 410.0121 | 失败案例 |
| 挑战 cosine 汇总 | 3 seeds，200 epoch | cosine beta | — / — / 137.5132 ± 8.0166 | raw 364.1998 ± 70.1675 | 初始 cosine 采样异常 |
| 改版R1 full linear default | 200 epoch，seed 44 | 完整 U-Net，linear，固定 LR | 18.9792 / 18.5210 / 18.8144 | 未裁剪 18.8045 | 完整 U-Net 基线 |
| 改版R2 full linear cosine-LR | 200 epoch，seed 44 | 完整 U-Net，linear，LR 尾段衰减 | 20.3343 / 20.1602 / 19.4633 | 未裁剪 19.4615 | LR 尾衰减在 linear 上未改善 |
| 改版R3 full cosine default | 200 epoch，seed 44 | 完整 U-Net，cosine，固定 LR | 15.7886 / 15.5340 / 16.1686 | raw 66.6401；未裁剪 162.8797 | 裁剪后大幅恢复，最佳 15.5340 |
| 改版R4 Min-SNR | 200 epoch，seed 44 | cosine + Min-SNR gamma=5，无 late decay | 16.3013 / 15.7954 / 16.5633 | raw 29.1437 | 不如 R5 |
| 改版R5 late decay | 200 epoch，seed 44 | cosine + uniform epsilon MSE + LR late decay | 16.3823 / 16.2361 / 15.4385 | raw 16.1927 | **当前正式最佳，但仍高 0.4385** |
| 改版R6 Min-SNR late decay | 200 epoch，seed 44 | R5 + Min-SNR gamma=5 | 17.2559 / 16.9445 / 15.8562 | raw 17.8510 | Min-SNR 在本配置下退化 |
| 改版R7 v-prediction | 200 epoch，seed 44 | R5 设置，epsilon → v target | 19.0443 / 18.8484 / 17.4808 | raw 19.2896 | 单独切换 v-prediction 无效 |
| 调整学习率LR=5e-5 | 200 epoch，seed 44 | cosine，peak 5e-5，late decay | 17.4472 / 17.3407 / 16.5231 | raw 18.6642 | 30 epoch 最优候选未延续 |
| 调整学习率LR=2.5e-4 | 50 → 200 epoch，seed 44 | cosine，peak 2.5e-4，late decay | 16.0761 / 15.8897 / 15.6761 | raw 16.8996 | 不如 R5 的 0.9999 |
| **R5 seed 测试** | 同一 R5 checkpoint | seed 44/45/46，5,000/5,000 | EMA 0.9999条件下，各seed对应**15.4385 / 15.4678 / 15.2086** | 均值 15.3717 | **R5在以上测试中效果最佳，FID最可能接近15** |

当前实现的最佳值为seed=46下的 **R5 EMA 0.9999 = 15.2086**。

## 二、实现内容

- schedule.py：linear/cosine beta schedule、alpha、累积 alpha_bar、posterior
  mean 系数和 posterior variance；
- diffusion.py：闭合形式前向加噪 q_sample、epsilon/v 训练损失、单步反向采样
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

~~~text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
~~~

训练时随机采样 t，直接由闭合形式得到 x_t，模型预测噪声 epsilon_theta(x_t,t)，
默认目标为：

~~~text
L_simple = mean((epsilon - epsilon_theta(x_t,t))^2)
~~~

反向采样从标准高斯噪声开始，按 T-1 到 0 逐步计算 posterior mean；t=0 不再加入随机噪声。

## 三、评估协议

### CIFAR-10 FID

使用固定的 CIFAR-10 train split 前 5,000 张真实图像，关闭随机增强，
每次生成 5,000 张图，生成 seed 为 44。模型分别评估 EMA 0.999、0.9995、0.9999
和 raw；R3–R7 的主结果使用 clip_denoised=True，即在每个反向步骤将预测的 x0
裁剪到 [-1, 1]。

此评估方法让不同训练方案在尽量相同真实子集、生成 seed、样本数和采样处理下进行单变量比较。


## 四、基础档与进阶档

### 4.1 MNIST 基础档

MNIST 基线使用 linear schedule、50 epoch、23,400 steps，最后记录 loss 为 0.01151，使用 EMA 生成 5,000 张样本的FID 为 32.8913。该结果的主要意义是验证从加噪、时间嵌入、ResBlock 注入到反向
采样的完整链路。

### 4.2 CIFAR-10 进阶档

进阶基线使用 CIFAR-10、200 epoch、78,000 steps、linear beta 和 epsilon MSE，维护
raw 与 EMA 权重。最终 loss 为 0.01938，训练约 98.2 分钟：

- EMA FID：19.2879；
- raw FID：28.7464；
  
因此 EMA 对采样质量有明显帮助，但这个单次结果距离 FID≤15 仍有 4.2879。

## 五、挑战档：linear 与 cosine

挑战档在相同模型、200 epoch、CIFAR-10 和 5,000/5,000 FID 协议下比较两个
schedule，并使用 seed 42、43、44：

| Schedule | Seed 42 | Seed 43 | Seed 44 | 均值 ± 标准差 |
|---|---:|---:|---:|---:|
| linear EMA | 19.2879 | 19.6306 | 18.9593 | 19.2926 ± 0.3357 |
| cosine EMA | 137.9856 | 129.2708 | 145.2832 | 137.5132 ± 8.0166 |
| linear raw | 28.7464 | 42.7537 | 34.9399 | 35.4800 ± 7.0193 |
| cosine raw | 283.4196 | 399.1676 | 410.0121 | 364.1998 ± 70.1675 |

初看结果会得出“linear 明显优于 cosine”的结论，但这只适用于初始矩阵的模型和采样实现。后续发现 cosine schedule 在 **末端极小 alpha_bar 下会产生巨大的x0** 反推值；在完整 U-Net、明确的 clipping 和诊断加入后，cosine 可以达到约 15–16的 FID。因此 schedule 不能脱离网络、loss、采样处理和训练预算单独评价。

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
EMA 对照和裁剪 x0：

| 实验 | 训练目标/学习率策略 | EMA 0.999 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R4 | Min-SNR gamma=5，无 late decay | 16.3013 | **15.7954** | 16.5633 | 29.1437 |
| R5 | uniform epsilon MSE，late decay 2e-4 → 2e-5 | 16.3823 | 16.2361 | **15.4385** | 16.1927 |
| R6 | Min-SNR gamma=5 + late decay | 17.2559 | 16.9445 | **15.8562** | 17.8510 |

R5 比 R4 的最佳值改善 0.3569，说明在当前配置下后期学习率衰减有效；R6 比 R5 高 0.4177，说明直接使用 Min-SNR gamma=5 没有解决当前问题。

### 6.5 R7：v-prediction

R7 保持 R5 的 U-Net、cosine beta、200 epoch、78,000 steps、warmup、late decay、
seed 和 clipping，只把 epsilon prediction 改为 v-prediction：

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

7 组实验使用完整 U-Net、cosine beta、epsilon prediction、seed 44、batch 128、
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

高学习率的 loss 略低而 FID 更差，说明平均 epsilon MSE 不能替代完整反向采样质量。
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

~~~text
t=998: alpha_bar ≈ 2.43e-6
t=999: alpha_bar ≈ 2.43e-9
~~~

epsilon prediction 反推 x0 时需要除以 sqrt(alpha_bar)，因此很小的 epsilon 误差会被放大。诊断中记录的典型现象为：

| t | epsilon MSE | mean abs x0 error | 越界比例 |
|---:|---:|---:|---:|
| 800 | 0.008703 | 0.224260 | 0.0030 |
| 990 | 0.000077 | 0.499628 | 0.0340 |
| 998 | 0.000034 | 2.889602 | 0.7689 |
| 999 | 0.000036 | 94.428703 | 0.9929 |

R3 无裁剪时 pred_x0_abs_max 约 964.98、越界比例约 99.43%，导致 FID 达到162.8797；逐步裁剪 x0 后降至约 16，可以说明 clipping 是必要的数值稳定化措施。

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

随机性可以造成约 0.1–0.2 的变化，但都无法稳定达到FID≤15。

## 九、对于 FID 仍未达到 15 的分析

当前最佳固定协议结果是 R5 的 15.4385，距离目标 0.4385。综合所有对照，根因
不是单一 bug，而是以下因素叠加：

1. **cosine 末端的 epsilon 误差放大。** 极小 alpha_bar 使最后几个 timestep 的
   x0 反推非常敏感；全程 clipping 能避免灾难性爆炸，但会引入系统性截断。
2. **中高噪声区间的误差仍在多步链中累积。** 即使最后一步不爆炸，中高噪声阶段的
   轮廓、颜色和局部纹理误差会逐步传递到 Inception 特征，像素级 MSE 很小也不代表
   FID 已经足够低。
3. **训练 loss 与 FID 目标不完全一致。** uniform epsilon MSE 对 timestep 等权，
   但不同 timestep 对最终视觉质量的贡献不等价；R6 的 Min-SNR gamma=5 退化说明
   简单地重新加权并不能保证有效。
4. **网络容量与采样器仍有限。** 当前 U-Net 已有完整 skip connection 和 16×16
   attention，但对 32×32 CIFAR-10 的细节建模能力、timestep 覆盖和 ancestral
   sampling 误差仍可能限制最后约半个 FID。
5. **EMA 只能降低参数噪声，不能修正系统误差。** EMA 显著优于 raw，但不同 decay
   的最优点会随训练轨迹变化，EMA 不是一个可以固定解决所有问题的独立变量。

### 为什么早期 linear 比 cosine 好，后来却是 cosine 更好

两次结论来自不同条件：

- 初始挑战矩阵使用的模型和采样路径在 cosine 末端没有处理好 x0 爆炸，因此
  cosine 均值为 137.5132；
- v2 使用完整 U-Net、明确的逐步 x0 clipping 和采样诊断后，cosine 降到 16.1686，
 继续优化到 R5 达到 15.4385；
- schedule 优劣依赖模型、loss、采样端点、EMA 和训练预算，不能脱离这些条件说 linear 或 cosine 永远更好。

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
- results/fid15_lr5e5_full/、results/lr25_200ep/：两组完整学习率复核；
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
