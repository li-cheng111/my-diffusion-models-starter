# Project 4：CIFAR-10 Flow Matching v2 实验报告

## 1. 实验设置与运行时间

本实验在 AutoDL 单张 NVIDIA GeForce RTX 4080 SUPER（32,760 MiB）上运行，环境为 Python
3.12.3、PyTorch 2.8.0+cu128 和 TorchVision 0.23.0+cu128。训练数据为 CIFAR-10 train，
模型为 32.62M 参数的 DiT-S，batch size 64，BF16 autocast，AdamW，随机种子 42，保存
50K 间隔快照并以 10K 间隔更新续训 checkpoint。两组实验分别使用无条件 dropout=1.0
和 conditional dropout=0.1；都训练到 200,000 optimizer steps。

| 阶段 | 实测用时 | 结果 |
|---|---:|---|
| CIFAR-10 下载与 MD5 校验 | 约 15 秒 | 固定镜像、170,498,071 字节 |
| 100-step BF16 冒烟运行 | 约 9.3 秒 | 约 10.8 step/s |
| Project 1 R5 + DDIM 五点扫描 | 34 分 49 秒 | 采样 2,006.3 秒，FID 计算 83.0 秒 |
| FM-U 训练 | 4.83 小时 | step 200,000，最终 loss 0.2013 |
| FM-C 训练 checkpoint 记录的累计时间 | 4.36 小时 | step 200,000，最终 loss 0.1665 |
| FM-U Euler/Heun 的 10 组 FID | 20.2 分钟 | 5,000 张生成图/组 |
| FM-C CFG 的 5 组 FID | 13.0 分钟 | 5,000 张生成图/组 |

FM-U 训练时与约 35 分钟的 Project 1 基线评测共用 GPU，因此两段时间不能简单相加为
总墙钟时间。FM-C 在日志达到 193K 后进程意外退出；从最后完整的 190K checkpoint 恢复，
重新计算未保存的 3K 步并继续到 200K。最终 checkpoint 的 4.36 小时累计计时不包括第一
次运行中这段被丢弃的计算。两次运行的原始训练日志与合并后的 CSV 均随结果提交。

## 2. Rectified Flow loss 推导

令噪声 $x_0\sim\mathcal N(0,I)$，数据样本为 $x_1$，时间 $t\sim U[0,1]$。采用线性概率路径

$$x_t=(1-t)x_0+t x_1.$$

对 $t$ 求导得到条件速度

$$u_t(x_t\mid x_1)=\frac{d x_t}{dt}=x_1-x_0.$$

网络 $v_\theta(x_t,t,y)$ 的 Conditional Flow Matching 目标为

$$\mathcal L_{CFM}=\mathbb E_{t,x_0,x_1,y}
  \left[\lVert v_\theta(x_t,t,y)-(x_1-x_0)\rVert_2^2\right].$$

实现先从 batch 图像构造 $x_1$，采样同形状高斯噪声和时间，再构造 $x_t$ 与 target
$x_1-x_0$。Conditional 训练以 10% 概率把标签替换为 null token 10；类别 embedding
大小为 `num_classes + 1`，所以该索引有效，并可供 CFG 使用。

最小化 CFM loss 与不可直接计算的边缘 FM loss 对网络输出的期望梯度相同。这里的等价是
优化梯度等价，不是单样本 loss 或两个 loss 的数值相等。

## 3. 采样方法与 CFG

Euler 从 $x(0)\sim\mathcal N(0,I)$ 出发，按 $t_i=i/N$ 积分：

$$x_{i+1}=x_i+v_\theta(x_i,t_i,y)\Delta t.$$

FM 的积分方向是从 0 到 1，更新项是加号。Heun 先做 Euler 预测，再在新位置和新时间
计算速度，并对两次速度取平均。一个 Heun 积分步需要两次网络前向，因此公平比较时按
实际网络求值次数（NFE）作横轴。

CFG 在线性速度场上组合无条件与条件预测：

$$v=v_u+s(v_c-v_u).$$

实现将两种标签的输入沿 batch 维合并为一次 batched model call，但其计算量仍相当于
两次单路前向；结果 JSON 同时记录 batched calls 和每张图的实际 network evaluations。

## 4. 统一 NFE-FID 评测

所有结果使用 EMA 0.9999、CIFAR-10 train 前 5,000 张真实图、5,000 张生成图、batch size
64 和 TorchMetrics `FrechetInceptionDistance(feature=2048, normalize=False)`。FM 与
Project 1 DDIM 都使用 seed 42 和同一 real-image 子集。FM 的完整原始数据见
[`results/fm_v2_unconditional_solvers.json`](results/fm_v2_unconditional_solvers.json)，
对照 DDIM 的原始数据见
[`../project2_samplers/results/p1_r5_ddim_5k.json`](../project2_samplers/results/p1_r5_ddim_5k.json)。

### FM-U 与 Project 1 R5 + DDIM

| 方法 | 实际 NFE/图 | 积分步数 | FID |
|---|---:|---:|---:|
| FM Euler | 4 | 4 | 80.5531 |
| FM Euler | 8 | 8 | 43.2211 |
| FM Euler | 16 | 16 | 32.1931 |
| FM Euler | 32 | 32 | 28.0684 |
| FM Euler | 50 | 50 | 26.7610 |
| FM Heun | 8 | 4 | 154.8813 |
| FM Heun | 16 | 8 | 82.1227 |
| FM Heun | 32 | 16 | 43.2588 |
| FM Heun | 64 | 32 | 26.4840 |
| FM Heun | 100 | 50 | 24.0062 |
| R5 + DDIM | 10 | 10 | 53.0907 |
| R5 + DDIM | 20 | 20 | 33.6099 |
| R5 + DDIM | 50 | 50 | 23.1206 |
| R5 + DDIM | 100 | 100 | 20.1105 |
| R5 + DDIM | 250 | 250 | 18.3625 |

![FM 与 Project 1 DDIM 的 NFE-FID 曲线](results/nfe_fid_curve_v2.png)

FM Euler 在 8 NFE 时优于 DDIM 10 NFE；16 NFE 的 FID 32.19 也略好于 DDIM 20 NFE
的 33.61。到更高采样预算时，DDIM 表现更好：NFE 50 时 DDIM 为 23.12，FM Euler 为
26.76；DDIM NFE 100/250 又进一步降至 20.11/18.36。Heun 以 64 次评估达到 26.48，
只略好于 Euler 50 次评估的 26.76；在粗步数设置下 Heun 的 FID 很差。因而本次实验
支持“FM Euler 在很低 NFE 下有优势”，但不支持“FM 在整个预算范围全面领先”。

## 5. CFG scale 扫描

FM-C 固定 Euler 20 个积分步，扫描 CFG scale；因为每步需要无条件和条件两次预测，
每张图的有效网络求值数均为 40。

| CFG scale | 实际 NFE/图 | FID |
|---:|---:|---:|
| 1.0 | 40 | 23.5773 |
| 2.0 | 40 | **18.8105** |
| 3.0 | 40 | 23.6983 |
| 5.0 | 40 | 35.2425 |
| 7.5 | 40 | 44.9110 |

scale=2.0 在本次设置下最好；scale 从 3.0 起 FID 反弹，较大的 guidance 会放大条件方向，
在本实验中降低质量。完整数据见
[`results/fm_v2_conditional_cfg.json`](results/fm_v2_conditional_cfg.json)。预览图包括
[FM-U Heun 50 步网格](samples/fm_v2_unconditional/heun_nfe50_unconditional_cfg1.0_EMA_0.9999.png)
和 [类别 0 / airplane 的 FM-C 网格](samples/fm_v2_conditional/class_0/euler_nfe20_cfg_cfg3.0_EMA_0.9999.png)。

## 6. 自查问题

1. **Linear path 的条件速度**：$x_t=(1-t)x_0+t x_1$，直接对 $t$ 求导得到
   $u_t=x_1-x_0$，因此 target 不依赖 $t$。
2. **CFM 与 FM 的关系**：条件速度对 $x_1$ 取条件期望给出边缘速度场；平方损失关于
   网络输出的期望梯度相同，所以可以用可采样的条件路径训练边缘场。loss 数值不必相等。
3. **低 NFE 对比**：FM 直接学习连续时间速度场，适合少步 ODE 积分，所以 Euler 在本次
   8 次评估时优于 DDIM 10 次；但这只是当前小模型、训练预算和单 seed 的测量，不能推出
   FM 在所有计算预算上都优于 DDIM。实际 50 次评估结果中 DDIM 更好。
4. **Reflow**：用模型生成的新 $(x_0,x_1)$ 配对重新训练，使轨迹更直、更低曲率，减少
   少步数值积分误差，从而改善极少步甚至一步生成。
5. **Velocity CFG**：在同一 $(x_t,t)$ 上计算条件与无条件速度，然后线性组合。ODE 的
   瞬时速度可以被引导，因此形式是 $v_u+s(v_c-v_u)$，与噪声或 score 空间的 CFG 类似。

## 7. 结论与限制

TODO 16–18 的 loss、Euler sampler 和 velocity-space CFG 均在训练和评测流程中使用；
两组 DiT-S 都完成 200K 训练，发布了只含推理权重的 EMA checkpoint。最终仓库保留原始
评测 JSON、NFE-FID 图、训练 CSV、训练/评测日志、类别样图、哈希文件与复现清单；权重
作为 GitHub Release 附件，不放入 Git 对象库。

FID 每个设置只跑一个 seed，5,000 张生成图带有单次采样噪声，微小差异应谨慎解释。若要
估计稳定性，可对多个 seed 重复生成并报告均值和标准差。CIFAR-10 低分辨率和 32.62M
模型规模也限制了绝对生成质量。
