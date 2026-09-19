# Project 1 DDPM：实现、实验结果与 FID 分析总报告

> 本文是 `projects/project1_ddpm/` 的统一说明文档，汇总基础档、进阶档、挑战档、
> FID≤15 改进实验、R7 v-prediction 实验和所有已完成诊断。原有的 `report.md`、
> `challenge_report.md`、`debug_log.md` 及各结果目录中的 README 仍然保留，用于保存
> 原始记录、运行日志和阶段性说明；本文作为面向复现和结论阅读的主入口。

## 1. 项目目标与最终结论

Project 1 的目标是使用 PyTorch 从零实现无条件 DDPM，并在 MNIST 与 CIFAR-10 上完成
训练、采样、FID 评估和实验分析。实现不依赖 `diffusers` 或 `lucidrains`，核心流程包括：

1. 构造 beta/alpha 累积量和 posterior 系数；
2. 通过闭合公式执行前向加噪；
3. 用带 timestep embedding 的 U-Net 预测噪声或 v；
4. 计算训练损失并维护 raw/EMA 权重；
5. 用 DDPM 反向链生成图像；
6. 使用固定协议计算 FID，并记录 loss、样本、checkpoint 和诊断数据。

最终可靠的 CIFAR-10 基线是 R5：

```text
R5 / uniform epsilon MSE / EMA 0.9999 / FID = 15.4385
```

在当前固定协议下尚未达到 FID≤15。R7 的 v-prediction 实验没有改善结果，最佳 FID
为 17.4808。现有证据表明，剩余差距主要来自中高噪声时间步的建模误差、cosine schedule
端点的病态条件、loss weighting 与采样 clipping 的组合，而不是单一的代码公式错误、
EMA 选择、真实图像子集或随机 seed。

## 2. 代码实现方法

### 2.1 调度与 DDPM 系数

`schedule.py` 实现两类 beta schedule：

- **linear**：在 `[beta_start, beta_end]` 之间等间隔生成 beta；
- **cosine**：先定义目标累积量

  $$
  \bar\alpha(t)=\cos^2\left(\frac{t/T+s}{1+s}\frac{\pi}{2}\right),
  $$

  再由相邻累积量计算

  $$
  \beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}.
  $$

调度对象预计算并注册以下量为 buffer：

```text
betas
alphas = 1 - betas
alphas_cumprod = alpha_bar
sqrt_alpha_bar
sqrt_one_minus_alpha_bar
posterior_variance
posterior_mean_coef1 / posterior_mean_coef2
```

时间步从 0 开始索引；batch 系数 reshape 为 `(B, 1, 1, 1)` 后广播到图像张量；`t=0`
的 posterior variance 显式设为 0，最后一步不添加随机噪声。

### 2.2 前向过程与训练目标

给定干净图像 `x0`、时间步 `t` 和标准高斯噪声 `epsilon`，前向过程使用闭合式：

$$
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

epsilon-prediction 的简化损失为：

$$
\mathcal L_\epsilon=
\mathbb E\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].
$$

R5 使用 uniform epsilon MSE，是当前效果最好的已完成训练基线。R6 只将该损失替换为
Min-SNR-$\gamma=5$ 加权，结果变差。R7 使用 v-prediction：

$$
v=\sqrt{\bar\alpha_t}\epsilon-
\sqrt{1-\bar\alpha_t}x_0.
$$

采样时将模型输出转换回噪声和 `x0`：

$$
\hat\epsilon=\sqrt{1-\bar\alpha_t}x_t+\sqrt{\bar\alpha_t}\hat v,
$$

$$
\hat x_0=\sqrt{\bar\alpha_t}x_t-
\sqrt{1-\bar\alpha_t}\hat v.
$$

### 2.3 反向采样

epsilon-prediction 下先计算：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

随后使用 DDPM posterior mean：

$$
\mu_t=c_1(t)\hat x_0+c_2(t)x_t,
$$

并按 posterior variance 加入随机噪声。R3 之后的固定 FID 协议在每个反向步骤将预测的
`x0` 裁剪到 `[-1,1]`，再计算 posterior mean。该操作能阻止 cosine 末端轨迹爆炸，
但会引入采样偏差，因此必须在结果中明确记录，不能把裁剪与未裁剪结果混为一谈。

### 2.4 时间 embedding 与 U-Net

`model/embedding.py` 使用 sinusoidal timestep embedding，将离散时间步编码为正弦/余弦
特征。`model/unet.py` 中的 ResBlock 将时间 embedding 经过线性层投影后广播到空间维度，
注入卷积特征。

基础 MNIST 模型使用较小 U-Net；CIFAR-10 进阶和挑战实验使用完整 skip-connection U-Net：

- base channels：128；
- channel multipliers：`[1, 2, 2, 2]`；
- 每个 level 两个 residual blocks；
- 16×16 分辨率加入 attention；
- dropout：0.1；
- 上采样阶段与对应下采样特征进行 skip connection；
- 同时维护 raw 权重和 EMA 权重。

### 2.5 数据、训练和评估入口

- `dataset.py`：CIFAR-10/MNIST 数据加载、resize、归一化到 `[-1,1]`；训练阶段可使用水平翻转，
  评估阶段关闭随机增强；
- `train.py`：配置驱动的训练、AMP、梯度裁剪、warmup、后期学习率衰减、EMA、周期 checkpoint
  和样本；
- `sample.py`：加载 raw 或指定 EMA decay，生成单图和 grid；
- `evaluate.py`：固定 real split、生成 seed、样本数量和 clipping，计算 raw/EMA FID；
- `experiment_monitor.py`：只读 HTTP 页面，显示训练 step、loss、LR、GPU、checkpoint 和样本。

## 3. 统一 FID 评估协议

除特别注明的早期对照外，FID 改进实验使用以下固定协议：

| 项目 | 固定值 |
|---|---|
| 数据集 | CIFAR-10 |
| 真实 split | `train` 前 5,000 张 |
| 真实图像增强 | 关闭 |
| 生成图像 | 5,000 张 |
| 生成 seed | 44 |
| batch size | 64 |
| 反向步数 | 1,000 |
| 采样权重 | 指定 EMA 或 raw |
| `x0` clipping | 明确记录，R4–R7 主结果为开启 |
| FID | torch-fidelity Inception feature=2048 |

seed 45/46、其他真实子集和 real-vs-real 只用于稳定性诊断，不替换正式结果。这样可以
避免通过更换真实图像、生成 seed 或样本数量选择最低分数。

## 4. 全部实验结果

### 4.1 挑战档：linear/cosine × seed 42/43/44

初始挑战矩阵每组训练 200 轮，比较两个 beta schedule：

| Schedule | Seed | EMA FID | Raw FID |
|---|---:|---:|---:|
| cosine | 42 | 137.9856 | 283.4196 |
| cosine | 43 | 129.2708 | 399.1676 |
| cosine | 44 | 145.2832 | 410.0121 |
| linear | 42 | 19.2879 | 28.7464 |
| linear | 43 | 19.6306 | 42.7537 |
| linear | 44 | 18.9593 | 34.9399 |

均值 ± 标准差：

| Schedule | EMA FID | Raw FID |
|---|---:|---:|
| linear | 19.2926 ± 0.3357 | 35.4800 ± 7.0193 |
| cosine | 137.5132 ± 8.0166 | 364.1998 ± 70.1675 |

该矩阵说明在当时的实现和评估条件下 cosine 出现严重采样异常。但它不能单独证明
“linear 永远优于 cosine”：后续 R3 使用完整 U-Net、明确的 `x0` clipping 和采样诊断后，
cosine 的 FID 从 100 以上降到约 16。挑战档和后续 R3 并非可以忽略差异的完全同条件实验，
因此 schedule 结论必须绑定到模型、采样和 clipping 协议。

### 4.2 v2：完整 U-Net 与早期 cosine 诊断

| 实验 | 主要变化 | 未裁剪 x0 | 裁剪 x0 |
|---|---|---:|---:|
| R1 | 完整 U-Net + linear + 常规 LR | 18.8045 | 18.8144 |
| R2 | 完整 U-Net + linear + cosine-LR 尾段 | 19.4615 | 19.4633 |
| R3 | 完整 U-Net + cosine + 常规 LR | 162.8797 | 16.1686 |

R3 的 EMA decay 对照为：

| 权重 | FID |
|---|---:|
| EMA 0.999 | 15.7886 |
| EMA 0.9995 | **15.5340** |
| EMA 0.9999 | 16.1686 |
| raw | 66.6401 |

R3 历史 checkpoint 的 EMA 0.9995、裁剪 `x0` FID：

| Step | FID |
|---:|---:|
| 10,000 | 40.9638 |
| 20,000 | 24.6445 |
| 30,000 | 19.8270 |
| 40,000 | 17.9806 |
| 50,000 | 16.9291 |
| 60,000 | 16.4440 |
| 70,000 | 15.9460 |
| 78,000 | 15.5340 |

没有观察到最终步数之前更低的 FID，说明该配置至少在 200 epoch 内仍随训练改善。

### 4.3 Linear 续训到 100,000 steps

在 linear/seed44 checkpoint 基础上续训到 100,000 次有效更新：

| 模型 | 采样 | FID |
|---|---|---:|
| 原始 200 epoch | 未裁剪 | 18.9577 |
| 原始 200 epoch | 裁剪 x0 | 18.9715 |
| 100,000 steps | 未裁剪 | 18.1502 |
| 100,000 steps | 裁剪 x0 | **18.1490** |

续训带来约 `0.81` 的改善，但仍距离 15 较远，说明简单延长 linear 训练不能解决主要
瓶颈。后续实验因此保持 200 epoch，并转向 schedule、loss、EMA 和参数化的单变量比较。

### 4.4 R4/R5/R6：cosine 主线改进

| 实验 | 训练损失/策略 | EMA 0.9990 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R4 | Min-SNR-$\gamma=5$，无 late decay | 16.3013 | **15.7954** | 16.5633 | 29.1437 |
| R5 | uniform epsilon MSE + late decay | 16.3823 | 16.2361 | **15.4385** | 16.1927 |
| R6 | Min-SNR-$\gamma=5$ + late decay | 17.2559 | 16.9445 | **15.8562** | 17.8510 |

R5 比 R4 改善约 `0.3569`，表明后期 learning-rate decay 在当前配置下有帮助。R6 比 R5
差 `0.4177`，说明 Min-SNR-$\gamma=5$ 并不是当前设置的有效改进；不能把“Min-SNR
通常有帮助”直接外推到本项目。

### 4.5 R7：v-prediction

R7 保持 R5 的完整 U-Net、cosine beta、200 epoch、78,000 steps、seed、warmup、late decay
和采样 clipping，仅切换为 v-prediction：

| 权重 | FID |
|---|---:|
| EMA 0.9990 | 19.0443 |
| EMA 0.9995 | 18.8484 |
| EMA 0.9999 | **17.4808** |
| raw | 19.2896 |

R7 的最佳结果比 R5 高 `2.0423`。这证明单独切换 prediction target 不能解决当前问题，
并且当前 uniform v-loss 与 timestep 分布、cosine schedule 和 clipping 的组合明显不如
R5 的 uniform epsilon MSE。

## 5. 诊断结果：哪些问题已经排除

### 5.1 posterior 公式和采样实现

固定 `x_t`、固定模型噪声预测和 `t=999,998,0`，将生产 `p_sample` 与独立 posterior
实现逐项比较：

| Schedule | 最大绝对误差 | 容差 |
|---|---:|---:|
| cosine | 5.80e-7 | 2e-4 |
| linear | 1.25e-4 | 2e-4 |

两者均通过。linear 在 `t=0` 的误差来自 float32 系数近似相减，不是 posterior mean 或
variance 公式不一致。因此目前没有证据支持“FID 高主要因为 posterior 代码写错”。

### 5.2 数据范围和 FID 输入转换

真实图像和生成图像均采用 `[-1,1] -> [0,1] -> uint8`。real-vs-real 校准为：

```text
train 前 5,000 张 vs 接下来 5,000 张：FID = 10.2039
```

两组真实图像的 uint8 均值、标准差和 `0–255` 范围一致，没有发现整体偏移、范围错误或
明显截断。因此数据转换不是当前差距的主要来源。

### 5.3 生成 seed 和真实子集方差

R5 EMA 0.9999 的三个生成 seed：

```text
seed44 = 15.4385
seed45 = 15.4678
seed46 = 15.2086
```

真实子集矩阵共 30 个单元格，R5 的均值为 `15.3590`，标准差 `0.1270`，最低 `15.1445`；
全部 30 个分数均大于 15。随机子集可能造成约 `0.18` 的单次变化，但不能稳定解释
FID≤15，也不能用事后挑选的最低子集作为正式结果。

### 5.4 过拟合和 checkpoint 损坏

R3/R5 的 train/test timestep 误差曲线基本重合，例如 R5 的 mean absolute `x0` error：

```text
t=500: train 0.128902 / test 0.127885
t=800: train 0.224260 / test 0.222263
```

没有明显 train-test 过拟合证据。历史上曾发生过 AutoDL 数据盘空间不足导致某个 final
checkpoint 损坏，但有效的中间 checkpoint 已恢复并重新生成最终 checkpoint；R5/R6/R7 的
最终文件均已实际加载并完成 FID 评估。因此当前结论不是由损坏 checkpoint 造成的。

## 6. FID 未达到 15 的根因分析

### 6.1 cosine 末端的极小 `alpha_bar`

R5 cosine schedule 在末端约为：

```text
t=998: alpha_bar ≈ 2.43e-6
t=999: alpha_bar ≈ 2.43e-9
```

epsilon-prediction 反推 `x0` 会除以 `sqrt(alpha_bar)`。即使 epsilon MSE 已很小，也会在
`t=998/999` 放大成巨大的 `x0` 误差：

| t | epsilon MSE | mean abs x0 error | 越界比例 |
|---:|---:|---:|---:|
| 800 | 0.008703 | 0.224260 | 0.0030 |
| 990 | 0.000077 | 0.499628 | 0.0340 |
| 998 | 0.000034 | 2.889602 | 0.7689 |
| 999 | 0.000036 | 94.428703 | 0.9929 |

这解释了为什么 cosine 未裁剪采样会出现 FID=162.8797 甚至更高，而 clipping 后能降到
16 左右。clipping 是必要的数值稳定化，但它不是完美的生成修复：被裁剪的 `x0` 不再是
模型原始后验均值对应的无偏估计，会对颜色和纹理造成系统性改变。

### 6.2 中高噪声区间的预测误差仍然影响 Inception 特征

末端 t=999 的异常容易观察，但真正决定 FID 的不只是最后一个 timestep。中高噪声区间
的误差会在多步反向链中累积，影响物体轮廓、颜色布局和局部纹理；这些误差可能对像素
MSE 不大，却会显著改变 Inception 特征。R5 的 FID 已接近 15，但说明该区间仍有足够的
系统性偏差。

### 6.3 loss weighting 与 FID 目标不一致

uniform epsilon MSE 对所有随机 timestep 等权，但不同 timestep 对最终 FID 的贡献并不等价。
R6 的 Min-SNR-$\gamma=5$ 直接改变了训练权重，却使最佳 FID 从 15.4385 退化到 15.8562；
这说明当前问题不是“随意增加低噪声权重”即可解决，而是需要测量每个 timestep 的误差与
FID 贡献后再设计权重。

R7 进一步说明 prediction target 的改变会重新定义有效权重。v-target 本身不保证在当前
uniform sampling 和 uniform loss 下更适合 CIFAR-10 FID；如果不同时校准 target-specific
weighting，模型可能在高噪声结构恢复上得到更合理的训练目标，却牺牲对最终视觉特征更
敏感的中间区域。

### 6.4 有限训练预算与模型容量

R3 从 10k 到 78k 的 FID 持续下降，表明在 200 epoch 内尚未完全进入平台区。R5 的晚期
学习率衰减有小幅收益，但 R7 在相同预算下退化，说明简单延长训练或单独换 target 都不够。
当前模型已经是完整 U-Net，但其宽度、attention 位置、训练步数和 timestep 采样仍可能限制
32×32 CIFAR-10 的细节建模能力。

## 7. 为什么早期看起来 linear 比 cosine 好，后来又反过来

这两个结论来自不同实验阶段，不能直接矛盾化：

1. 初始挑战矩阵中，cosine 出现 137.5 的均值，说明当时的采样轨迹对末端数值问题极其敏感；
2. v2 中加入完整 U-Net、明确 `x0` clipping 和 schedule 诊断后，cosine 结果降到 16.1686；
3. 后续 R4/R5 继续使用 cosine 并优化 EMA、warmup 和 late decay，最佳达到 15.4385；
4. 因此真正的结论是：schedule 的优劣依赖于网络、loss、采样 clipping、训练预算和评估协议。

初始 linear 的 19.29 不能与修订后的 cosine 15.44 直接比较，因为后者同时受益于完整 U-Net、
更稳定的采样处理、EMA 对比和后期学习率衰减。公平结论必须使用只改变一个变量的消融。

## 8. 可复现文件与仓库结构

### 8.1 代码入口

```text
schedule.py              beta schedule 与 posterior 系数
diffusion.py             q_sample、loss、p_sample、采样循环
dataset.py               MNIST/CIFAR-10 数据加载
train.py                 训练、EMA、checkpoint、loss 记录
sample.py                生成图像与 grid
evaluate.py              固定协议 FID 评估
model/embedding.py       sinusoidal timestep embedding
model/unet.py            ResBlock、attention、完整 U-Net
configs/                 各阶段实际配置
tests/                   静态测试代码
```

### 8.2 结果索引

- [`results/challenge/`](results/challenge/)：linear/cosine × seed42/43/44 挑战矩阵；
- [`results/fid15_v2/`](results/fid15_v2/)：R1/R2/R3、EMA 对照、历史 checkpoint 和 sampler 验证；
- [`results/fid15_4090_linear_seed44/`](results/fid15_4090_linear_seed44/)：linear 续训到 100k；
- [`results/fid15_r4_min_snr/`](results/fid15_r4_min_snr/)：R4；
- [`results/fid15_final/`](results/fid15_final/)：R5/R6；
- [`results/fid15_stage1/`](results/fid15_stage1/)：FID 稳定性、real-vs-real、timestep 和真实子集诊断；
- [`results/fid15_r7_v_prediction/`](results/fid15_r7_v_prediction/)：R7 v-prediction。

### 8.3 Release 大文件

大型 checkpoint、中间样本和 MNIST 临时产物不进入普通 Git，统一放在
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。
R7 最终 checkpoint 为 [`r7_v_prediction_final.pt`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/r7_v_prediction_final.pt)。

## 9. 复现命令

在仓库的 `projects/project1_ddpm/` 目录中安装依赖后，下载所需 CIFAR-10 数据和 Release
checkpoint，再执行：

```bash
python evaluate.py \
  --ckpt runs/fid15_r5_late_decay/ckpt/final.pt \
  --num_samples 5000 \
  --batch_size 64 \
  --real_split train \
  --seed 44 \
  --compare_ema \
  --clip_denoised \
  --data_root ./data
```

R7 使用相同命令，只需替换为 R7 checkpoint。重新训练时使用：

```bash
python train.py --config configs/cifar10_fid15_r5_late_decay.yaml
python train.py --config configs/cifar10_fid15_r7_v_prediction.yaml
```

训练、评估和测试本报告只记录实际执行过的命令及结果；仓库中不提交虚构的样本、FID 或
checkpoint。`runs/`、原始数据和大型二进制仍属于运行环境产物。

## 10. 后续改进优先级

### 优先级 1：不重训的采样消融

以 R5 EMA 0.9999 为唯一基线，固定 5,000/5,000、seed44：

1. 比较末端若干步 clipping 与全程 clipping；
2. 比较动态阈值 clipping 与固定 `[-1,1]` clipping；
3. 记录不同 timestep 的 `x0` 越界率、P50/P95/P99 和 FID；
4. 保留最低分之外的全部协议和结果，避免选择性报告。

### 优先级 2：loss/timestep 单变量训练

保持 R5 的 epsilon-prediction、网络、schedule 和 200 epoch，只改变一个因素：

- Min-SNR 的多个 gamma，而不是只测 gamma=5；
- P2 weighting 或针对中高噪声区间的温和权重；
- timestep importance sampling；
- 每个候选都比较 EMA 0.999/0.9995/0.9999/raw。

### 优先级 3：schedule 和模型容量

只有在优先级 1/2 有正收益后再做：

- cosine `s` 和末端 `alpha_bar` floor 的小范围消融；
- 更宽的中层/高分辨率特征；
- attention 位置和 residual block 数量；
- 保持 200 epoch，比较每个配置的实际有效更新数和 wall-clock。

不建议继续单独延长 linear 训练，也不建议在没有 target-specific weighting 的情况下继续
使用 R7 v-prediction 作为主线。

## 11. 最终判断

Project 1 的代码、配置、测试、训练、采样、评估、监控和结果归档已经完成；基础档、进阶档、
挑战档和多轮 FID 改进实验均有可复现记录。当前最优固定协议结果为 R5 EMA 0.9999 的
FID `15.4385`，尚未达到 FID≤15。该差距已经通过 seed、真实子集、real-vs-real、sampler
公式和 checkpoint 完整性检查，不能简单归因于随机性或基础实现错误。

当前最合理的技术结论是：在 200 epoch、32×32 CIFAR-10、现有完整 U-Net 和 DDPM ancestral
sampling 下，模型在中高噪声 timestep 的学习目标和采样校正仍不够匹配。下一步应该围绕
 `R5 epsilon-prediction + loss/timestep weighting + schedule endpoint/sampler` 做可控单变量
实验；只有出现稳定、可复现且不依赖挑选 seed/real subset 的低于 15 结果，才能宣布达标。

## 附录 A：30 epoch 学习率粗筛

在完成上述实验后，AutoDL 上又完成了 7 组学习率粗筛。实验固定完整 U-Net、cosine beta、
epsilon prediction、seed 44、batch 128 和 1,000 steps warmup；每组运行 30 epoch、
11,730 steps，warmup 后保持常数学习率。评估固定 EMA 0.9999、裁剪预测 x0、seed 44、
train split，并使用 1,000 张真实图与 1,000 张生成图。

| 学习率 | 快速 FID | 最终 loss |
|---:|---:|---:|
| 5e-5 | **238.0244** | 0.05284 |
| 8e-5 | 263.5649 | 0.05245 |
| 1.2e-4 | 291.7750 | 0.05214 |
| 1.6e-4 | 284.0970 | 0.05209 |
| 2e-4 | 307.1943 | 0.05199 |
| 2.5e-4 | 330.7679 | 0.05193 |
| 3e-4 | 323.4893 | 0.05207 |

7/7 任务正常完成，没有 OOM、NaN 或 AMP 异常。该结果支持在早期筛选阶段优先测试较低
学习率，但不能直接推翻 R5：粗筛只运行 30 epoch、使用 1,000 样本，并且没有使用 R5 的
后期 cosine 学习率衰减。高学习率的最终 loss 略低而 FID 更差，进一步证明训练 loss
不是完整采样质量的充分指标。因此后续正式复核应只保留 `5e-5` 和 `8e-5`，恢复 200 epoch
及正式 5,000 样本 FID 协议。详细日志、配置和图像见 `results/lr_screen_30ep/`，LR05
大型 checkpoint 位于 `challenge-v1 Release`。
