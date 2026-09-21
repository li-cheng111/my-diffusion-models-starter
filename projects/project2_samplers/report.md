# 项目 2 ：Samplers 报告

> 实现、理论分析和 AutoDL 正式测量均已完成。下列数值全部来自已记录的
> JSON 结果文件。

## 1. 实验协议

- 模型：项目 1 CIFAR-10 linear 调度策略的随机种子 44 checkpoint，使用 EMA 权重。
- FID 真实图集合：固定使用前 5,000 张 CIFAR-10 训练图，不使用随机增强。
- 生成图集合：每个配置生成 5,000 张图，每个配置开始前都将随机种子重置为 42。
- 数据范围：训练和采样数据在 `[-1,1]`；FID 输入转换为 `[0,255]` 的 RGB
  `uint8`。
- 计算环境：AutoDL NVIDIA RTX 4090（24,564 MiB），Python 3.12.3，PyTorch
  2.8.0+cu128，CUDA 12.8。
- checkpoint SHA256：`937853559a1377660f7d4cbd1dd3f7c6cf022aed4ab84e9daa918aa6e34541412`。
- AutoDL 仓库提交：`e8f8af6698570485b9909ca1755f0266a09e4da8`；权重从嵌套 EMA
  状态中加载。

## 2. DDIM 推导和实现

训练好的模型预测前向过程中的噪声：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

由上式求解干净图像得到：

$$
\hat x_0(x_t)=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
{\sqrt{\bar\alpha_t}}.
$$

对于跳步时间序列中任意相邻的 `t > t_prev`：

$$
\sigma^2=
\eta^2\frac{1-\bar\alpha_{t_{prev}}}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_{t_{prev}}}\right),
$$

$$
x_{t_{prev}}=
\sqrt{\bar\alpha_{t_{prev}}}\hat x_0+
\sqrt{1-\bar\alpha_{t_{prev}}-\sigma^2}\epsilon_\theta+
\sigma z.
$$

实现会将平方根参数截断到不小于零，保护 `alpha_bar` 很小时的除法，并将生成
的干净图像预测裁剪到 `[-1,1]`。当 `eta=0` 时，不会额外采样随机噪声。

## 3. FID 与 NFE

| 采样器 | 步数 | 真实 NFE | FID | 采样时间 |
|---|---:|---:|---:|---:|
| DDPM | 1000 | 1000 | 19.290 | 1010.7 s |
| DDIM | 10 | 10 | 34.555 | 10.1 s |
| DDIM | 20 | 20 | 28.159 | 20.1 s |
| DDIM | 50 | 50 | 24.574 | 50.9 s |
| DDIM | 100 | 100 | 22.875 | 100.6 s |
| DDIM | 250 | 250 | 21.241 | 251.9 s |
| Euler | 10 | 10 | 34.555 | 10.1 s |
| Euler | 20 | 20 | 28.159 | 20.3 s |
| Euler | 50 | 50 | 24.574 | 50.4 s |
| Euler | 100 | 100 | 22.875 | 100.9 s |
| Euler | 250 | 250 | 21.241 | 251.4 s |
| DPM-Solver-2 | 5 | 9 | 29.295 | 9.3 s |
| DPM-Solver-2 | 10 | 19 | 21.074 | 19.3 s |
| DPM-Solver-2 | 25 | 49 | 21.361 | 49.4 s |
| DPM-Solver-2 | 50 | 99 | 21.192 | 100.5 s |

![FID 与 NFE 的 Pareto 图](runs/pareto_fid_nfe.png)

图中 DDIM 使用蓝色虚线和方形标记，并在 Euler 曲线之后绘制。两者在本次
`eta=0`、linear 时间步和同一 checkpoint 下数值重合，因此如果只使用相同的
实线和圆点，后绘制的 Euler 会把 DDIM 完全覆盖；现在图例和曲线均能明确区分
两种采样器。

每个点都使用相同的训练网络、checkpoint、5,000 张真实图和随机种子，因此曲线
变化反映的是离散化或求解器行为，而不是重新训练的影响。DDIM 随步数增加而
稳定改善，FID 从 10 NFE 时的 34.555 降至 250 NFE 时的 21.241。课程 starter
中的 Euler 实现与 `eta=0` 的 DDIM 使用相同的确定性离散 probability-flow 更新，
所以数值完全一致。DPM-Solver-2 在低计算量下最有优势：19 NFE 时 FID 为
21.074，接近 DDIM 250 NFE 的结果，且明显优于 DDIM 20 NFE。DDPM 的绝对 FID
最佳（19.290），但需要 1,000 NFE。

## 4. 采样轨迹

![共享初始噪声的采样轨迹对比](runs/trajectory_comparison.png)

测量结果见 `runs/trajectory_comparison.json`。图中四行从上到下依次对应表中的
DDPM、DDIM、Euler 和 DPM-Solver-2；每一行都使用同一个 Project 1
epsilon-prediction U-Net（CIFAR-10、seed44、EMA 权重），变化项只有采样器、
时间步策略和 NFE。所有行使用相同的初始噪声（`same_initial_noise=true`）和
真实时间步标签。

| 轨迹（图中行） | 采样器与更新方式 | 神经网络模型 | 时间步策略 / 外层步数 | 实际 NFE | 平均单步 L2 | 最大单步 L2 | 终点距初始噪声 L2 |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | DDPM ancestral | Project 1 CIFAR-10 epsilon U-Net（EMA，seed44） | linear / 1000 | 1000 | 5.2548 | 8.0443 | 67.5356 |
| 2 | DDIM，`eta=0` | 同上 | linear / 50 | 50 | 1.1307 | 3.2183 | 53.7210 |
| 3 | Euler probability-flow | 同上 | linear / 50 | 50 | 1.1307 | 3.2183 | 53.7210 |
| 4 | DPM-Solver-2 指数中点 | 同上 | lambda / 50 | 99 | 1.1225 | 4.2199 | 53.0901 |

在 50 个外层步时，DDPM 使用 1,000 NFE，平均单步 L2 最大；DDIM 和 Euler
完全重合；DPM-Solver-2 以约两倍 NFE 换取略小的平均单步位移。DDPM 在初始化
之后仍然是随机过程，而其他三种配置在 `eta=0` 或确定性 ODE 更新下不再注入
新的随机噪声。

## 5. DPM-Solver-2

令 $\lambda=\log(\alpha/\sigma)$，扩散 ODE 的线性部分可以解析积分。实现
分别在源点和离散 lambda 中点的最近时间步评估网络，然后应用指数中点更新。
当运行 `S` 个外层步时，网络实际评估次数为 `2S-1`，因为最后的干净图像投影
只需要一次评估。

正式对比表明，二阶中点修正在低 NFE 区间最有用。为避免只罗列 DPM-Solver-2
自己的数值，下面同时列出 NFE 最接近的 DDIM 配置；$\Delta$FID 定义为
`DPM-Solver-2 FID - 匹配 DDIM FID`，负值表示 DPM-Solver-2 更好。

| DPM 外层步数 $S$ | DPM 实际 NFE $2S-1$ | DPM FID | 采样时间 | 匹配 DDIM NFE | 匹配 DDIM FID | $\Delta$FID（DPM - DDIM） |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 9 | 29.295 | 9.3 s | 10 | 34.555 | -5.261 |
| 10 | 19 | 21.074 | 19.3 s | 20 | 28.159 | -7.085 |
| 25 | 49 | 21.361 | 49.4 s | 50 | 24.574 | -3.213 |
| 50 | 99 | 21.192 | 100.5 s | 100 | 22.875 | -1.684 |

这张表支持三个结论。第一，$S=10$、19 NFE 是本实验中最有效的工作点，FID
比相近 NFE 的 DDIM 低 7.085。第二，增加外层步数并不保证 FID 单调下降：25
步（49 NFE）的 21.361 反而略差于 10 步（19 NFE）的 21.074，说明有限容量
模型的预测误差和 lambda 离散网格误差开始占主导。第三，99 NFE 时 DPM-Solver-2
仍优于 100 NFE DDIM，但距离 DDPM 的 19.290 还有差距；因此它的优势是用较少
模型调用达到接近高质量的结果，而不是在所有 NFE 下都超过完整 DDPM。真实模型
调用次数 `2S-1` 已同时记录在 benchmark JSON 和 sampler 自测中。

## 6. DDIM 反演

反演严格沿采样时间序列的反方向执行，并使用当前状态的噪声预测近似无法直接
获得的目标时间噪声。反演和重构都关闭干净图像裁剪，避免裁剪人为破坏往返过程。

![DDIM 反演误差](runs/inversion_errors.png)

评估使用 CIFAR-10 测试集索引 0–63，并在往返过程中关闭裁剪。结果见
`runs/inversion_results.json`：

| 步数 | 平均 L2 | MAE | MSE | PSNR |
|---:|---:|---:|---:|---:|
| 10 | 18.2820 | 0.26955 | 0.11417 | 15.891 dB |
| 20 | 12.5195 | 0.18369 | 0.05343 | 19.175 dB |
| 50 | 5.0287 | 0.07314 | 0.00864 | 27.057 dB |
| 100 | 2.3665 | 0.03419 | 0.00197 | 33.644 dB |
| 250 | 0.9316 | 0.01332 | 0.00032 | 41.836 dB |

误差随反演步数增加单调且显著下降。对于本 checkpoint 和时间步策略，增加步数
带来的局部截断误差下降足以抵消模型预测误差的累积影响。

## 7. 必答自查问题

### 7.1 为什么 DDIM 是确定性的？

当 `eta=0` 时，$\sigma=0$，更新式中不再包含新采样的 `z`。因此，相同的
模型、调度策略、时间步序列、权重和初始 $x_T$ 会定义相同的数学输出。在 GPU
上要实现逐比特完全一致，还需要确定性 kernel 以及固定的软件和硬件环境。

### 7.2 cosine 训练模型应如何选择 100 个 DDIM 时间步？

均匀的整数索引并不代表噪声水平的等量变化，而且 cosine 调度策略对
$\bar\alpha_t$ 的分布不同于 linear 调度策略。更合理的做法是按 log-SNR
均匀布置推理点（或通过 $\bar\alpha_t$ 匹配目标噪声水平），再将它们映射回
不重复的离散训练索引。复用相同整数索引在形式上可行，但无法在去噪路径上提供
相同的数值分辨率。

### 7.3 为什么改变 eta 会改变质量？

`eta` 在确定性的 ODE-like 轨迹和随机 ancestral 更新之间进行折中。注入噪声
可以增加轨迹多样性，但在步数较少、单步跨度较大时，也会引入剩余更新难以完全
消除的方差。因此 `eta=0` 通常在低步数下具有更好的保真度，而偏大的 eta 在
更重视多样性时可能更有价值。

### 7.4 为什么 DDIM 更新可以退化为 DDPM？

对于连续时间步，并令

$$
\sigma_t^2=\tilde\beta_t=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
$$

将
$\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/
\sqrt{\bar\alpha_t}$ 代入 DDIM 均值，并收集 $x_t$ 与 $\epsilon_\theta$
的系数，可得

$$
\mu_\theta=
\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta\right).
$$

剩余的随机项为 $\sqrt{\tilde\beta_t}z$，正好是 DDPM ancestral 更新。该
等价关系要求时间步连续且不额外进行干净图像裁剪；跳步时的 `eta=1` 更新只是
类似 DDPM，并不是原始 DDPM 的 Markov 链。
