# Project 4：CIFAR-10 Flow Matching 实验报告

> **历史结果说明**：以下数字来自 Project4 v1 的 RTX 4090 训练。旧采样脚本把
> `CFG=0` 作为“无 guidance”，但仍输入类别 0，因此这些数字实际是类别条件生成，
> 不能作为无条件 FM 与 Project 1 的公平对照。代码和评测协议现已更新为 v2；新的
> AutoDL 4080 SUPER 结果、运行时间及 R5 对照曲线将在训练和统一评测完成后替换本节。

## 1. 实验目标与环境

本项目在 CIFAR-10 训练集上实现 Conditional Flow Matching（Rectified Flow），并用同一份真实样本、同一套 TorchMetrics FID 配置比较 Flow Matching（FM）和 Project 2 的 DDPM + DDIM 基线。核心交付物是 TODO 16 的训练损失、TODO 17 的 Euler ODE 采样、TODO 18 的 velocity-space CFG，以及不同 NFE 下的 FID 曲线。

正式实验运行在 AutoDL 单张 NVIDIA GeForce RTX 4090（24 GB）上，Python 3.12.3、PyTorch 2.8.0+cu128、TorchVision 0.23.0+cu128、TorchMetrics 1.9.0。模型采用 DiT-S，约 32.62M 参数；batch size 128，200,000 个 optimizer steps，AdamW，EMA decay 0.9999，conditional dropout 0.1，随机种子 42，训练使用 BF16 autocast。CIFAR-10 数据直接复用 AutoDL 数据盘上的缓存，避免重复下载。

正式训练从 10:27 左右开始，16:45 保存 step 200,000，耗时约 6 小时 18 分钟，稳定吞吐约 8.81 step/s，最终训练 loss 为 0.1676。BF16 预检时 batch 128 的峰值显存约 7.47 GiB，正式运行没有显存溢出。训练 checkpoint 只保存在 AutoDL，不进入 Git。

## 2. Rectified Flow loss 推导

设噪声 $\epsilon\sim\mathcal N(0,I)$，数据样本为 $x_1$。线性概率路径为

$$x_t=(1-t)\epsilon+t x_1,\qquad t\sim U[0,1].$$

对时间求导得到条件速度

$$u_t(x_t\mid x_1)=\frac{d x_t}{dt}=x_1-\epsilon.$$

因此网络 $v_\theta(x_t,t,y)$ 的 Conditional Flow Matching 目标为

$$\mathcal L_{CFM}=\mathbb E_{t,\epsilon,x_1}\left[\lVert v_\theta(x_t,t,y)-(x_1-\epsilon)\rVert_2^2\right].$$

代码先从 batch 图像得到 $x_1$，采样同形状标准高斯噪声和 batch 时间，再构造 $x_t$ 与 target。训练时以 10% 概率将类别替换为索引 10；两个 backbone 的 embedding 大小是 `num_classes + 1`，所以索引 10 是合法的 null token。这一设计同时完成了 CFG 所需的 conditional/unconditional 训练。

最小化 CFM loss 与不可直接计算的 FM loss 的梯度相同。原因是对条件速度的平方误差取条件分布平均后，关于网络输出的期望梯度等于关于边缘真实速度场的期望梯度；两者的标量损失不必相等，但优化方向一致。

## 3. 采样与 CFG

Euler sampler 从 $x(0)\sim\mathcal N(0,I)$ 出发，将区间 $[0,1]$ 等分为 $N$ 步，并执行

$$x_{i+1}=x_i+v_\theta(x_i,t_i,y)\Delta t.$$

FM 的积分方向是从 0 到 1，符号是加号。CFG 在 velocity 空间中使用

$$v=v_u+s(v_c-v_u),$$

其中 $v_u$ 使用 null token，$v_c$ 使用目标类别。实现将 conditional 和 unconditional 输入沿 batch 维拼接，一次前向后拆分，降低了 Python 循环和 kernel launch 开销。另实现了 Heun 二阶 sampler：先用 Euler 预测新位置，再在新时间点评估速度并取两次速度的平均，CPU 冒烟测试已通过。

## 4. NFE-FID 结果

所有 FM FID 使用 EMA 权重、5,000 张生成图、CIFAR-10 train 的前 5,000 张真实图和 TorchMetrics `FrechetInceptionDistance(feature=2048, normalize=False)`。Project 2 DDPM 基线使用相同样本数、真实数据切分和 FID 实现；因此数值可以直接放在同一张图中观察趋势。

### FM Euler（CFG=0）

| NFE | forward passes | FID |
|---:|---:|---:|
| 4 | 4 | 47.377 |
| 8 | 8 | 28.305 |
| 16 | 16 | 22.121 |
| 32 | 32 | 19.311 |
| 50 | 50 | 18.299 |

### DDPM + DDIM 基线

| NFE | FID |
|---:|---:|
| 10 | 34.555 |
| 20 | 28.159 |
| 50 | 24.574 |
| 100 | 22.875 |
| 250 | 21.241 |

FM 在低 NFE 区间下降很快：8 NFE 的 FID 已为 28.305，优于 DDIM 10 NFE 的 34.555；16 NFE 为 22.121，优于 DDIM 20 NFE 的 28.159；50 NFE 为 18.299，优于 DDIM 50 NFE 的 24.574。曲线在 32–50 NFE 后开始变平，说明当前 200K、5,000 样本设置下继续增加 Euler 步数的收益有限。绝对数值高于论文级结果是预期现象，原因包括模型规模、训练步数和 FID 样本数都较小。

## 5. CFG scale 扫描

固定 Euler NFE=20，结果如下：

| CFG scale | FID |
|---:|---:|
| 1.0 | 20.988 |
| 2.0 | **18.349** |
| 3.0 | 23.693 |
| 5.0 | 35.316 |
| 7.5 | 43.962 |

scale=2.0 在本次训练中最好。scale 从 3.0 开始 FID 反弹，5.0 和 7.5 明显恶化，说明过强 guidance 放大了类别条件方向，带来过饱和和多样性下降。实际使用应在 1–3 附近搜索，而不能直接照搬扩散模型常用的 7.5。

## 6. 自查问题

1. 线性路径的导数直接给出 $u_t=x_1-\epsilon$；这也是代码 target 的来源。
2. CFM 和 FM 的网络输出梯度期望相同，因此可以用可采样的条件路径训练边缘速度场；“等价”指优化梯度，不是每个样本的 loss 数值相同。
3. DDPM 的反向过程在低步数下会累积离散化误差，而 FM 学到的是从噪声到数据的连续速度场，轨迹更适合少步 ODE 积分，所以低 NFE 下优势明显。
4. Reflow 用模型生成的配对重新训练，使路径更接近直线，减少曲率，从而改善一步或少步积分。
5. CFG 组合的是同一时刻、同一状态下的速度估计；速度场对条件分布的线性组合仍可用于 ODE 积分，因此形式与在噪声/score 空间做 CFG 相同。

## 7. 结论与后续

TODO 16–18 已完成并通过自检、500 步 debug、BF16 预检和正式训练。正式 FM 的 NFE-FID 曲线在低 NFE 下优于 DDIM 基线，CFG=2.0 是当前设置的较优点。最终提交包含源代码、原始 JSON、NFE-FID 图、采样图、实验日志和调试记录；大 checkpoint 与 CIFAR-10 数据不提交。
