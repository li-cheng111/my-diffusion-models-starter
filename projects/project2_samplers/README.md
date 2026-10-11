# Project 2: 采样器对比 (DDIM / DPM-Solver / Euler) 实验报告

本项目直接复用项目 1 已训练的 CIFAR-10 epsilon-prediction U-Net，比较不同采样器的生成质量、模型调用次数（NFE）和 DDIM 反演效果。代码、配置、测量结果和图表均保存在本目录。

## 一、实现内容

| 方法 | 实现与用途 |
|---|---|
| DDPM ancestral | 完整 1,000 步随机采样，作为基线。 |
| DDIM | 支持跳步、线性或二次时间步及 `eta`；`eta=0` 时为确定性采样。 |
| Euler | Probability-flow 一阶采样。当前实现与 `eta=0` 的 DDIM 使用相同的离散更新。 |
| DPM-Solver-2 | 基于指数中点法的二阶求解器；`S` 个外层步实际调用模型 `2S-1` 次。 |
| DDIM inversion | 将图像反演到噪声潜变量，再确定性重构并测量误差。 |

### 二、实验结果

| 采样器 | 步数 / 外层步 | NFE | FID ↓ | 采样时间 |
|---|---:|---:|---:|---:|
| DDPM | 1,000 | 1,000 | **19.290** | 1,010.7 s |
| DDIM (`eta=0`) | 10 | 10 | 34.555 | 10.1 s |
| DDIM (`eta=0`) | 20 | 20 | 28.159 | 20.1 s |
| DDIM (`eta=0`) | 50 | 50 | 24.574 | 50.9 s |
| DDIM (`eta=0`) | 100 | 100 | 22.875 | 100.6 s |
| DDIM (`eta=0`) | 250 | 250 | 21.241 | 251.9 s |
| Euler | 10 | 10 | 34.555 | 10.1 s |
| Euler | 20 | 20 | 28.159 | 20.3 s |
| Euler | 50 | 50 | 24.574 | 50.4 s |
| Euler | 100 | 100 | 22.875 | 100.9 s |
| Euler | 250 | 250 | 21.241 | 251.4 s |
| DPM-Solver-2 | 5 | 9 | 29.295 | 9.3 s |
| DPM-Solver-2 | 10 | 19 | **21.074** | 19.3 s |
| DPM-Solver-2 | 25 | 49 | 21.361 | 49.4 s |
| DPM-Solver-2 | 50 | 99 | 21.192 | 100.5 s |

![FID–NFE Pareto 图](runs/pareto_fid_nfe.png)

在这组固定权重和协议下，DDPM 的绝对 FID 最低，但需要 1,000 NFE。DPM-Solver-2 在 19 NFE 时 FID 为 21.074，比相近预算的 DDIM（20 NFE，28.159）低 7.085。**DDIM 和 Euler 的测量曲线重合，符合本实现中 `eta=0` DDIM 与一阶 probability-flow Euler 更新相同的离散形式。**

此外，仓库记录了另一组独立实验：使用项目 1 的 R5 EMA 0.9999 权重 `r5_uniform_late_decay_final.pt`（[challenge-v1 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)，做 DDIM 评估，真实图、生成 seed 和生成图数量与正式协议相同。**可见，即便DDPM的模型训练得更好，对于低步数的DDIM也不一定有正面的提升。**

| DDIM NFE | 10 | 20 | 50 | 100 | 250 |
|---|---:|---:|---:|---:|---:|
| R5 FID ↓ | 53.091 | 33.610 | 23.121 | 20.111 | **18.362** |

数据见 [`results/p1_r5_ddim_5k.json`](results/p1_r5_ddim_5k.json)，250 NFE 样图见 [`samples/r5_baseline/ddim_nfe250.png`](samples/r5_baseline/ddim_nfe250.png)。

## 三、采样轨迹对比

轨迹实验对四种方法传入同一个初始噪声 tensor，减少随机起点造成的差异。下列数据使用 CIFAR-10 seed 44 EMA 模型、初始噪声 seed 44；DDIM、Euler 使用 50 步，DPM-Solver-2 使用 50 个外层步。

| 采样器 | 步数 | 实际 NFE | 平均单步 L2 | 最大单步 L2 | 终点距初始噪声 L2 |
|---|---:|---:|---:|---:|---:|
| DDPM ancestral | 1,000 | 1,000 | 5.2548 | 8.0443 | 67.5356 |
| DDIM (`eta=0`) | 50 | 50 | 1.1307 | 3.2183 | 53.7210 |
| Euler | 50 | 50 | 1.1307 | 3.2183 | 53.7210 |
| DPM-Solver-2 | 50 | 99 | 1.1225 | 4.2199 | 53.0901 |

![相同初始噪声下的采样轨迹](runs/trajectory_comparison.png)

## 四、DDIM 反演与重构

反演使用 CIFAR-10 测试集前 64 张图，按采样时间步的反方向运行；往返过程关闭 `x0` 裁剪，避免裁剪影响重构误差。表中指标是 64 张图逐图测量后的均值。

| 步数 | 平均 L2  | MAE  | MSE  | PSNR  |
|---:|---:|---:|---:|---:|
| 10 | 18.2820 | 0.26955 | 0.11417 | 15.891 dB |
| 20 | 12.5195 | 0.18369 | 0.05343 | 19.175 dB |
| 50 | 5.0287 | 0.07314 | 0.00864 | 27.057 dB |
| 100 | 2.3665 | 0.03419 | 0.00197 | 33.644 dB |
| 250 | **0.9316** | **0.01332** | **0.00032** | **41.836 dB** |

![DDIM 反演误差曲线](runs/inversion_errors.png)

## 五、原理方法说明与适用范围

DDIM 根据噪声预测估计干净样本：

$$
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}}.
$$

对跳步 `t > t_prev`，更新为：

$$
\begin{aligned}
\sigma_t^2 &= \eta^2\frac{1-\bar{\alpha}_{t_{prev}}}{1-\bar{\alpha}_t}
\left(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t_{prev}}}\right),\\
x_{t_{prev}} &= \sqrt{\bar{\alpha}_{t_{prev}}}\hat{x}_0
+\sqrt{1-\bar{\alpha}_{t_{prev}}-\sigma_t^2}\,\epsilon_\theta
+\sigma_t z.
\end{aligned}
$$

`eta=0` 时 `sigma=0`，固定权重、时间步和初始噪声后采样结果确定。`eta` 增大时会注入额外噪声，可能增加多样性，也会改变低步数下的保真度。实现会保护平方根和除法的数值范围；普通生成默认裁剪 `x0` 到 `[-1,1]`，反演与重构不裁剪。

DPM-Solver-2 令 `lambda = log(alpha / sigma)`，在 lambda 中点进行二阶指数积分。每个中间区间调用模型两次，最后的终止步只需一次，因此实际 NFE 为 `2S-1`。本实验使用 lambda 时间步；报告的 NFE 由采样器调用次数验证。

当 DDIM 不裁剪 `x0`、采用连续相邻训练时间步，并令随机方差等于 DDPM posterior variance

$$
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
$$

其更新可化为 DDPM ancestral 更新。跳步时即使 `eta=1`，也只是类似 DDPM 的随机更新，并非原始 DDPM Markov 链。

## 六、验证与结果文件说明

| 文件 | 内容 |
|---|---|
| `runs/benchmark_all.json` | 15 个正式配置、FID、NFE、耗时、checkpoint 摘要和运行环境。 |
| `runs/benchmark_ddim.json` | 五个 DDIM 步数的独立结果。 |
| `runs/pareto_fid_nfe.png` | 完整 FID–NFE Pareto 图。 |
| `runs/trajectory_comparison.json`、`runs/trajectory_comparison.png` | 共同初始噪声下的轨迹指标和图像。 |
| `runs/inversion_results.json`、`runs/inversion_errors.png` | 反演重构指标和误差图。 |
| `results/p1_r5_ddim_5k.json`、`results/p1_r5_ddim_5k.png` | Project 1 R5 checkpoint 的独立 DDIM 对照。 |

## 七、自查问题与回答

为避免符号混淆，记单步保留系数为 $\alpha_t=1-\beta_t$，累积保留系数为 $\bar\alpha_t=\prod_{i=1}^{t}\alpha_i$，干净端点为 $\bar\alpha_0=1$。这里使用数学上的 $t=1,\ldots,T$；代码中的训练索引为 `0,...,T-1`，干净端点用 `-1` 表示。$\sigma_{t\to s}$ 专指 DDIM 更新时新注入噪声的标准差。

### 1. 确定性：相同初始噪声是否一定得到相同输出？

从较大的时间 $t$ 跳到较小的时间 $s$，DDIM 更新为：

$$
\begin{aligned}
\epsilon_t &= \epsilon_\theta(x_t,t),\\
\hat x_0 &= \frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_t}{\sqrt{\bar\alpha_t}},\\
\sigma_{t\to s} &= \eta\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right)},\\
x_s &= \sqrt{\bar\alpha_s}\,\hat x_0
+\sqrt{1-\bar\alpha_s-\sigma_{t\to s}^{2}}\,\epsilon_t
+\sigma_{t\to s}z,\qquad z\sim\mathcal N(0,I).
\end{aligned}
$$

当 $\eta=0$ 时，$\sigma_{t\to s}=0$，最后的随机项消失。给定 $x_t$ 后，模型预测和下一步状态都是确定的函数，逐步复合后得到 $x_0=G_\theta(x_T)$。因此，在数学层面，固定同一个 $x_T$、模型权重、条件、时间步序列和裁剪规则，确定性网络会给出相同输出。但是若初始噪声不同，也还会得到不同的输出。

本项目的 [`DDIMSampler.step`](samplers/ddim.py) 只有在 `eta > 0` 且 `sigma > 0` 时才调用 `randn`。当通过 `initial_x` 显式传入初始噪声且 `eta=0` 时，采样过程无需再抽取随机数。

工程上还要区分“算法确定性”和“逐位一致”，GPU 非确定性算子、并行归约顺序、浮点精度以及不同硬件或库版本，都可能产生数值差异，算法上的确定不能消除所有这些差异。

### 2. 时间步：cosine 模型如何选取 100 个 DDIM 推理步？

训练噪声调度决定每个时间索引对应的 $\bar\alpha_t$；推理时间步策略决定从这些索引中选取哪些点。`timestep_strategy="linear"` 表示均匀选择索引，并不表示把模型的训练噪声调度改成 linear。

对于使用 cosine schedule 训练、共有 1,000 个训练时间步的模型，可以从代码索引 `0,...,999` 中均匀选择 100 个不同索引，再按降序采样。选择索引的算法可以与 linear 模型相同，但每一步必须读取 cosine 调度对应的 `alphas_cumprod`，并将选中的原始训练索引传给模型。

相同的索引选择不代表相同的噪声间隔。cosine 调度按归一化时间 $u\in[0,1]$ 定义的目标累积系数为：

$$
f(u)=\cos^2\!\left(\frac{u+s_0}{1+s_0}\frac{\pi}{2}\right),
\qquad \bar\alpha(u)=\frac{f(u)}{f(0)}.
$$

其中 $s_0$ 是小偏移量。离散实现还会裁剪 $\beta_t$，所以推理应使用实际保存或一致重建的系数。

可用半对数信噪比检查每一步跨越的噪声范围：

$$
\lambda_t=\log\frac{\sqrt{\bar\alpha_t}}{\sqrt{1-\bar\alpha_t}}
=\frac12\log\frac{\bar\alpha_t}{1-\bar\alpha_t}.
$$

如果希望按噪声水平分配预算，可以在实际训练索引对应的有限 $\lambda$ 区间中选择 100 个目标值，再映射到最接近的训练索引。

### 3. 插值：为什么改变 eta 会影响生成质量？小步数时哪一端通常更好？

对固定区间 $t\to s$，记 $\eta=1$ 时的方差为 $v_{t\to s}$，则：

$$
\sigma_{t\to s}^{2}=\eta^2v_{t\to s},\qquad
x_s=\sqrt{\bar\alpha_s}\hat x_0
+\sqrt{1-\bar\alpha_s-\eta^2v_{t\to s}}\,\epsilon_t
+\eta\sqrt{v_{t\to s}}\,z.
$$

因此，增大 $\eta$ 不只是额外加一份噪声，它同时减小沿当前噪声预测 $\epsilon_t$ 前进的系数，并增大独立随机噪声的系数，即不同 $\eta$ 改变了采样路径及后续模型看到的输入。

在理想条件下，不同生成过程可以对应相同的目标数据分布，但实际模型预测不精确，而且跳步会放大相邻状态之间的差异，因此不同路径上的误差传播不同，最终 FID 也会不同。

在很少的推理步数下，$\eta=0$ 或较小的 $\eta$ 往往是更好的起点，确定性更新利用当前预测跨越较大的时间间隔，而强随机更新在有限的后续去噪次数下更难得到充分修正。

### 4. 退化：证明 DDIM 在 posterior variance 下得到 DDPM 均值

考虑相邻步 $t\to t-1$，先不裁剪 $\hat x_0$，并令：

$$
\sigma_{t\to t-1}^{2}=\tilde\beta_t
=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
\qquad \bar\alpha_t=\alpha_t\bar\alpha_{t-1}.
$$

第一步，化简 DDIM 噪声方向的系数：

$$
\begin{aligned}
1-\bar\alpha_{t-1}-\tilde\beta_t
&=(1-\bar\alpha_{t-1})\left(1-\frac{\beta_t}{1-\bar\alpha_t}\right)\\
&=\frac{\alpha_t(1-\bar\alpha_{t-1})^2}{1-\bar\alpha_t}.
\end{aligned}
$$

因此其平方根为 $\sqrt{\alpha_t}(1-\bar\alpha_{t-1})/\sqrt{1-\bar\alpha_t}$。
第二步，把 $\hat x_0$ 代入 DDIM 的非随机部分 $m_t$：

$$
\begin{aligned}
m_t
&=\sqrt{\bar\alpha_{t-1}}\hat x_0
+\sqrt{1-\bar\alpha_{t-1}-\tilde\beta_t}\,\epsilon_t\\
&=\frac{x_t}{\sqrt{\alpha_t}}
+\left[-\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\alpha_t}}
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{\sqrt{1-\bar\alpha_t}}\right]\epsilon_t\\
&=\frac{1}{\sqrt{\alpha_t}}\left(x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_t\right)
=\mu_\theta(x_t,t).
\end{aligned}
$$

最后一步使用恒等式 $-(1-\bar\alpha_t)+\alpha_t(1-\bar\alpha_{t-1})=-\beta_t$。加入随机项后即为：

$$
x_{t-1}=\mu_\theta(x_t,t)+\sqrt{\tilde\beta_t}\,z.
$$

为了与题目中的 $\tilde\mu_t$ 对应，真实前向后验的均值为：

$$
\tilde\mu_t(x_t,x_0)
=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

代入 $x_0=\hat x_0$ 后，正好得到上面的 $\mu_\theta(x_t,t)$；若 $\hat x_0$ 恰好等于真实 $x_0$，则得到真实后验均值。

### 5. 误差来源：DDIM Inversion 为什么不能保证精确重构？

记 $a_t$ 和 $b_t$ 分别为：

$$
a_t=\sqrt{\bar\alpha_t},\qquad
b_t=\sqrt{1-\bar\alpha_t}.
$$

无裁剪的确定性 DDIM 可写为：

$$
x_s=\frac{a_s}{a_t}x_t+
\left(b_s-\frac{a_s}{a_t}b_t\right)\epsilon_\theta(x_t,t), \qquad s<t.
$$

若给定 $x_s$ 求 $x_t$，右侧模型预测依赖尚未知晓的 $x_t$，一般需要解一个非线性方程。标准 DDIM inversion 用已知端点的预测替代未知端点的预测，得到显式近似：

$$
\hat x_t=\frac{a_t}{a_s}x_s+
\left(b_t-\frac{a_t}{a_s}b_s\right)\epsilon_\theta(x_s,s).
$$

再从 $\hat x_t$ 采样回 $s$，单区间往返残差满足：

$$
x_s^{\mathrm{rec}}-x_s
=\left(b_s-\frac{a_s}{a_t}b_t\right)
\left[\epsilon_\theta(\hat x_t,t)-\epsilon_\theta(x_s,s)\right].
$$

这个等式直接显示误差来源：**两个方向在不同状态和时间上调用模型，噪声预测通常不相等，因此不会严格抵消。** 同一时间步序列只是保证两边的区间匹配，不能消除这一近似误差。
