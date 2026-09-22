# 项目 2 报告：采样器对比

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

## 5. DPM-Solver-2：高阶 ODE 求解器

### 5.1 关键思想：半解析积分

扩散模型的 probability-flow ODE 可以抽象写成：

$$
\frac{dx}{dt}=a(t)x+b_\theta(x,t),
$$

其中 $a(t)x$ 是只与噪声调度有关的线性项，$b_\theta(x,t)$ 是包含模型预测
$\epsilon_\theta(x,t)$ 的非线性项。对这样的半线性 ODE 使用积分因子，可以把
解写成：

$$
x(s)=\exp\left(\int_t^s a(u)\,du\right)x(t)
+\int_t^s\exp\left(\int_\tau^s a(u)\,du\right)b_\theta(x(\tau),\tau)\,d\tau.
$$

DPM-Solver 不再像普通 Euler 一样把整个右端都近似为
$f_\theta(x_t,t)\Delta t$，而是先解析计算线性项，只对包含神经网络的积分项
做数值近似。这样可以在较大的时间区间上保留扩散调度的精确结构，显著降低
离散化误差；但“更精确”依赖于模型预测足够平滑，并不是无条件精确。

在 epsilon 预测参数化下，代码使用：

$$
\alpha_t=\sqrt{\bar\alpha_t},\qquad
\sigma_t=\sqrt{1-\bar\alpha_t},\qquad
\lambda_t=\log\frac{\alpha_t}{\sigma_t}
$$

作为噪声坐标。其中 $\lambda_t$ 是 log-SNR 的一半。采样从噪声端向数据端
进行时，$\lambda$ 单调增加，因此比原始整数时间步更适合均匀分配求解区间。
本实现先在 $\lambda$ 空间均匀取点，再映射到最近的离散训练时间步。

若在一个区间内暂时把模型预测视为常数，记

$$
h=\lambda_s-\lambda_t>0,
$$

则半解析的一阶更新可以写为：

$$
x_s=\frac{\alpha_s}{\alpha_t}x_t
-\sigma_s\left(e^h-1\right)\epsilon_\theta(x_t,t).
$$

第一项是线性扩散部分的解析传播，第二项是冻结模型预测后得到的积分结果。
这说明 DPM-Solver 的效率来源不是减少模型计算本身，而是减少每次模型预测之间
因大步长而产生的数值误差。

### 5.2 DPM-Solver-2：指数中点法

只在区间起点使用 $\epsilon_t$ 仍然是一阶近似。DPM-Solver-2 再增加一次
中点模型评估，以估计模型预测在整个区间内的变化：

$$
\lambda_m=\frac{\lambda_t+\lambda_s}{2},\qquad
h_m=\lambda_m-\lambda_t.
$$

首先用起点预测构造中点状态：

$$
x_m=\frac{\alpha_m}{\alpha_t}x_t
-\sigma_m\left(e^{h_m}-1\right)\epsilon_t.
$$

然后在中点重新调用网络：

$$
\epsilon_m=\epsilon_\theta(x_m,t_m).
$$

最后使用中点预测完成整个区间：

$$
x_s=\frac{\alpha_s}{\alpha_t}x_t
-\sigma_s\left(e^h-1\right)\epsilon_m.
$$

中点预测相当于对模型积分项使用指数加权的 midpoint quadrature。在模型预测
随 $\lambda$ 平滑变化时，其局部误差为 $O(h^3)$，全局误差为 $O(h^2)$，因此
称为二阶 DPM-Solver。这里的“二阶”数值积分阶数，不是每个采样步固定需要
两次模型调用。

当前实现对应 `DPMSolver2Sampler._dpm_solver_2_step`：先计算源点预测，再构造
离散 lambda 中点并计算 `eps_mid`，最后使用指数中点更新。最后一步只需要用当前
预测投影到干净图像，不再构造中点，因此运行 `S` 个外层步时：

$$
\mathrm{NFE}=2S-1.
$$

例如 10 个外层步实际是 19 NFE，而不是 10 或 20 NFE。

### 5.3 “高阶”具体提高了什么

从一阶到更高阶，本质上是对同一个模型积分项使用更高阶的近似：

- 一阶 DPM-Solver 将区间内的 $\epsilon_\theta$ 视为常数，只使用一个时间点；
- 二阶 DPM-Solver-2 使用中点预测，能够捕捉 $\epsilon_\theta$ 随 $\lambda$ 的
  一阶变化；
- 三阶及更高阶方法会使用更多中间点，或复用前几个区间的模型预测，拟合更高次
  的时间变化。

如果模型预测足够平滑，$p$ 阶方法的全局离散误差通常随步长按
$O(h^p)$ 缩小。但阶数越高并不意味着 NFE 一定更低：单步高阶方法需要更多
模型评估，多步方法则需要保存并复用历史预测。因此实际比较必须同时报告 NFE，
不能只比较外层步数。当前实验中 DPM-Solver-2 的 `S=10` 实际是 19 NFE，
而 DDIM 的 50 步是 50 NFE；“10 步 DPM 优于 50 步 DDIM”准确地说是较低模型
调用预算下的比较，而不是同 NFE 的严格对照。

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
