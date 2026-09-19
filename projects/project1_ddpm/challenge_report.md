# 项目 1 挑战档技术报告

> 全部 Project 1 的统一方法与结果分析见 [`PROJECT1_COMPLETE_REPORT.md`](PROJECT1_COMPLETE_REPORT.md)。

> 本文按八页技术报告组织，内容基于已完成的六组真实实验和仓库结果文件。大型 checkpoint 与中间样本保存在 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。

## 第 1 页：摘要与研究问题

### 研究问题

在 U-Net、优化器、数据增强、训练轮数和随机种子集合保持一致的前提下，比较 linear 与 cosine beta 调度策略对 CIFAR-10 无条件 DDPM 的影响。主要指标是使用 EMA 权重生成的 FID，辅助指标是 raw 权重 FID、训练 loss 曲线和 64 张样本网格。

### 可复现性声明

挑战实验矩阵默认是两个调度策略和三个随机种子：`42、43、44`，每个组合训练 200 轮；`challenge.py` 也支持显式加入 50 轮控制组。每个组合评估 5,000 张生成图像与 5,000 张无增强 CIFAR-10 训练图像，并写入 `results.csv` 和 `summary.md`。

### 实验状态

挑战档的 linear/cosine × seed `42/43/44` 六组 200 轮实验均已完成。结果显示，在当前模型、优化器、训练预算和 FID 协议下，linear 的 EMA FID 明显低于 cosine；逐 seed 结果和统计汇总见第 5 页。

## 第 2 页：DDPM 数学背景

前向过程使用固定的高斯转移：

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I).
$$

令 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$，则可以用闭合形式直接采样：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\quad\epsilon\sim\mathcal N(0,I).
$$

模型 $\epsilon_\theta(x_t,t)$ 预测噪声，训练目标为：

$$
\mathcal L_{simple}=\mathbb E_{x_0,t,\epsilon}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2\right].
$$

调度策略不改变模型结构，而是改变每个时间步的噪声强度以及训练样本在不同信噪比区域的分布。

## 第 3 页：linear 与 cosine 调度策略实现

### linear 调度策略

`linear_beta_schedule` 在 `[beta_start, beta_end]` 之间等间隔生成 `T` 个 beta。它直接对应原始 DDPM 常用基线，优点是简单、易解释；缺点是时间步上的信噪比变化不一定均衡。

### cosine 调度策略

`cosine_beta_schedule` 先构造：

$$
\bar\alpha(t)=\cos^2\left(\frac{t/T+s}{1+s}\frac{\pi}{2}\right),
$$

再通过相邻累积量之比得到：

$$
\beta_t=1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}.
$$

实现对 beta 做有限性和 `(0,1)` 范围检查，并使用数值截断避免极端时间步造成不稳定。两种调度策略都进入同一个 `DDPMSchedule`，因此训练和采样逻辑无需分叉。

### 实现自查

- `betas`、`alphas_cumprod` 和 posterior 系数注册为 buffer。
- `t` 从 0 开始索引。
- batch 时间系数 reshape 为 `(B, 1, 1, 1)` 后再广播到图像张量。
- `t=0` 的 posterior variance 显式设为 0，最后一步不添加随机扰动。

## 第 4 页：实验设计

### 控制变量

| 项目 | 固定值 |
|---|---|
| 数据集 | CIFAR-10 训练集 |
| 分辨率 | 32 × 32 |
| Batch size | 128 |
| 训练轮数 | 200 |
| 模型 | base 128，channel mult 1/2/2/2 |
| 优化器 | AdamW |
| 学习率 | 2e-4 |
| Warmup | 5,000 步 |
| EMA | 衰减 0.9999 |
| 精度 | fp16 |
| 数据增强 | 仅训练阶段随机水平翻转 |
| FID 真实图 | 5,000 张无增强训练图 |
| FID 生成图 | 每次运行 5,000 张 |

唯一实验变量是 `beta_schedule`。两个调度策略各运行随机种子 `42、43、44`，并用同一评估随机种子规则生成可比较的随机样本。

### 运行方式

```bash
python challenge.py run \
  --schedules linear cosine \
  --seeds 42 43 44 \
  --output_root runs/challenge
```

若要同时完成 50 轮/200 轮的调度策略自查：

```bash
python challenge.py run \
  --schedules linear cosine \
  --epoch_budgets 50 200 \
  --seeds 42 43 44 \
  --output_root runs/challenge
```

运行时每个实验独立保存 checkpoint、loss history、loss curve、EMA 样本网格和 raw/EMA FID；当前 Git 提交只保留小型 FID 结果和最终样本网格，大型 checkpoint 与中间样本见 Release。若训练中断，可保留已完成目录，修复后使用单独的 `train.py --resume` 命令恢复；恢复前应在日志中记录原因。

## 第 5 页：结果表与统计方法

本页使用真实实验产物生成，不能用估算值替代。

### 每个随机种子的结果

| 调度策略 | 随机种子 | EMA FID | Raw FID | Raw - EMA | 状态 |
|---|---:|---:|---:|---:|---|
| linear | 42 | 19.2879 | 28.7464 | +9.4585 | 已完成 |
| linear | 43 | 19.6306 | 42.7537 | +23.1231 | 已完成 |
| linear | 44 | 18.9593 | 34.9399 | +15.9806 | 已完成 |
| cosine | 42 | 137.9856 | 283.4196 | +145.4340 | 已完成 |
| cosine | 43 | 129.2708 | 399.1676 | +269.8968 | 已完成 |
| cosine | 44 | 145.2832 | 410.0121 | +264.7289 | 已完成 |

### 均值 ± 标准差

`challenge.py summarize` 使用三个随机种子的算术平均值和样本标准差：

$$
\bar x=\frac{1}{n}\sum_i x_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_i(x_i-\bar x)^2}.
$$

最终统计如下，FID 的真实数据划分为 5,000 张无增强训练图，生成图数量为每次 5,000 张：

| 调度策略 | 训练轮数 | EMA FID | Raw FID |
|---|---:|---:|---:|
| linear | 200 | 19.2926 ± 0.3357 | 35.4800 ± 7.0193 |
| cosine | 200 | 137.5132 ± 8.0166 | 364.1998 ± 70.1675 |

## 第 6 页：训练曲线与样本质量分析

本页使用已提交的最终样本网格和 Release 中的中间样本：

- `results/challenge/cifar10_linear_200ep_seed*/samples_challenge_ema/grid.png`
- `results/challenge/cifar10_cosine_200ep_seed*/samples_challenge_ema/grid.png`
- Release 附件 `challenge_intermediate_samples.tgz`

挑战档完整 loss history/loss curve 未提交到 Git，因此本报告不虚构六组训练曲线或最终 loss。分析应区分训练 loss 和生成质量：loss 更低不必然意味着 FID 更低；样本网格应观察颜色、轮廓、多样性、重复样本和明显伪影。EMA/raw 的差异也应结合 FID 和图像，而不是只凭单张样本下结论。

本实验中 linear 的 EMA 平均 FID 比 raw 低 `16.1874`，cosine 的对应差值为 `226.6866`。这说明 EMA 在本次实验的两种权重比较中均有帮助，但不能据此推断任何其他训练设置。

对于“50 轮与 200 轮哪个调度策略更好”的自查问题，本仓库只提供 200 轮挑战矩阵。若要回答 50 轮，必须新增同样的 50 轮控制实验，不能从 200 轮结果外推。

## 第 7 页：失败案例与威胁有效性

### 真实失败案例

本页只允许写入实际发生且有日志证据的问题，例如：数据下载中断、显存不足、NaN、checkpoint 恢复错位、FID 输入范围错误或某个调度策略的训练失败。每条记录都应包含命令、日志片段或文件证据、根因和修复验证，格式见 `logs/challenge_experiment_log_template.md` 和 `debug_log.md`。

挑战矩阵的六组 FID 文件和结果汇总均已生成，仓库中没有记录需要单独归因的挑战档训练失败。已有的 AutoDL GitHub HTTP 503、SSH 连接和数据下载慢等问题属于环境访问问题，已在 `debug_log.md` 记录，不能冒充调度策略失败。

### 威胁有效性

- 三个随机种子仍然是小样本，均值 ± 标准差不能等同于统计显著性检验。
- FID 对 Inception 实现、输入范围和 real split 敏感。
- 同一训练时长不代表两个调度策略达到相同优化程度。
- 仅比较一个 U-Net 容量和一个学习率，结论只适用于本实验设置。
- 评估样本数为 5,000，结果可能比大规模评估有更高方差。

## 第 8 页：结论与复现清单

### 结论

> 在 CIFAR-10、200 轮、相同 U-Net/优化器和随机种子 `42/43/44` 的条件下，linear 的 EMA FID 为 `19.2926 ± 0.3357`，cosine 的 EMA FID 为 `137.5132 ± 8.0166`。在本实验协议下 linear 优于 cosine，但两者均未达到 FID ≤ 15。linear 的 EMA 平均 FID 比 raw 低 `16.1874`，cosine 的对应差值为 `226.6866`。该结论仅适用于当前配置。

### 复现清单

- [x] `configs/cifar10_linear.yaml` 与 `configs/cifar10_cosine.yaml` 已固定控制变量
- [x] 两个调度策略均完成随机种子 42、43、44 的 200 轮实验
- [x] 六个 FID 结果均为 5,000 对 5,000，并已保存到 `results/challenge/`
- [x] 六个最终 EMA 样本网格已提交到 Git
- [x] 大型 checkpoint 和中间样本已上传到 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)
- [x] `python challenge.py summarize` 生成的 `results.csv` 和 `summary.md` 已提交
- [x] 失败和环境问题均按证据记录，未将推测写成 schedule 失败结论
- [ ] 50 轮控制组：本次未运行
