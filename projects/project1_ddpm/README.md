# 项目 1：从零实现 DDPM

本目录包含项目 1 基础档、进阶档和挑战档的源码、配置、静态测试、结果与实验说明。实现目标是用 PyTorch 手写一个不依赖 `diffusers` 或 `lucidrains` 的无条件 DDPM。

## 当前提交状态

基础档 MNIST 实验、进阶档 CIFAR-10 单次实验、挑战档 linear/cosine × seed 42/43/44 六组实验均已完成。FID≤15 v2 的三组严格 200 epoch 实验也已完成；小型结果、摘要和最终样本网格在仓库中，大型 checkpoint 与中间样本在 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1) 或 AutoDL 实验目录中。

## 基础档完成清单

- [x] linear beta 调度策略和 DDPM 系数预计算
- [x] 闭合形式前向加噪 `q_sample`
- [x] 简化的噪声预测损失 `p_losses`
- [x] 单步反向采样 `p_sample`
- [x] 完整反向采样循环 `p_sample_loop`
- [x] sinusoidal timestep embedding
- [x] ResBlock 的时间 embedding 广播注入
- [x] MNIST 50 轮配置和训练入口
- [x] 64 张样本生成与最终产物的后续命令
- [x] 实际运行 MNIST 训练并补充最终 loss 曲线、样本网格和 checkpoint

## 进阶档实现状态

- [x] CIFAR-10 200 轮配置（`configs/cifar10.yaml`）
- [x] 训练过程维护 raw 与 EMA 两套权重
- [x] FID 默认使用 5,000 张无增强训练图作为 real split
- [x] `evaluate.py --compare_ema` 一键生成 EMA/raw 对比记录
- [x] CIFAR-10 实际训练、EMA/raw 样本和 FID 对比报告
- [ ] FID ≤ 15（本次 EMA FID 为 19.2879，仍需调参或重训）

## FID≤15 改进实验状态

在 AutoDL NVIDIA GeForce RTX 4090 上，基于 linear/seed44 的原始 CIFAR-10
checkpoint 继续训练到 100,000 次有效更新，并固定使用 5,000 张训练集图像、EMA
权重和 seed 44 评估。实验同时比较了预测 $x_0$ 的逐步裁剪与原始采样；裁剪只影响
采样阶段，不改变训练目标。

| checkpoint | 采样方式 | FID |
|---|---|---:|
| 原始 200 epoch / seed44 | 未裁剪 | 18.9577 |
| 原始 200 epoch / seed44 | 裁剪预测 $x_0$ | 18.9715 |
| 100,000 有效更新 | 未裁剪 | **18.1502** |
| 100,000 有效更新 | 裁剪预测 $x_0$ | **18.1490** |

续训相对原始未裁剪基线下降 0.8087 FID；本次仍未达到 FID≤15。裁剪预测 $x_0$
在最终 checkpoint 上只带来 0.0012 的额外下降，不能视为稳定收益。详细配置、日志
和限制见 `report.md`、`debug_log.md` 以及 `results/fid15_4090_linear_seed44/summary.md`。

## 挑战档实现状态

- [x] `cosine_beta_schedule` 已实现并接入 `DDPMSchedule`
- [x] linear/cosine 两套 CIFAR-10 200 轮配置已建立
- [x] `challenge.py` 已提供 2 个调度策略 × 3 个随机种子的可复现实验编排
- [x] `challenge.py` 支持 `--epoch_budgets 50 200`，可回答 50/200 轮自查问题
- [x] `challenge.py summarize` 已提供均值 ± 标准差汇总
- [x] `challenge_report.md` 已建立八页技术报告结构和失败案例记录规范
- [x] 挑战档六组 200 轮实验已经完成，结果见 `results/challenge/summary.md`
- [x] 真实均值 ± 标准差和调度策略对比已记录；linear 在本实验协议下优于 cosine
- [x] 大型 checkpoint、中间样本和 MNIST 临时产物已通过 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1) 保存

如需在新的输出目录复现实验矩阵，可明确执行：

```bash
python challenge.py run \
  --schedules linear cosine \
  --seeds 42 43 44 \
  --output_root runs/challenge_repro
```

如需先检查将要执行的命令而不运行程序，可使用 `--dry_run`。已有提交的可复核汇总位于
`results/challenge/summary.md`；若重新生成实验目录，再使用
`python challenge.py summarize --output_root runs/challenge_repro` 汇总。

默认挑战矩阵为 200 轮、2 个调度策略 × 3 个随机种子，共 6 组。本次没有运行 50 轮控制组，
因此不能从当前结果外推 50 轮下的调度策略结论。

### 挑战档实际结果

| 调度策略 | EMA FID | Raw FID |
|---|---:|---:|
| linear | 19.2926 ± 0.3357 | 35.4800 ± 7.0193 |
| cosine | 137.5132 ± 8.0166 | 364.1998 ± 70.1675 |

FID 使用每次 5,000 张生成图与 5,000 张无增强 CIFAR-10 训练图计算。当前最佳单次 EMA FID
为 linear/seed44 的 18.9593，仍未达到作业目标 FID ≤ 15。逐 seed 数值见
`results/challenge/summary.md`，六个 checkpoint 下载地址见上方 Release。

## 目录

```text
schedule.py       # beta 调度策略与 DDPM 系数
diffusion.py      # q_sample、训练损失和反向采样
dataset.py        # MNIST / CIFAR-10 加载与 [-1, 1] 归一化
train.py          # 配置驱动训练、EMA、AMP、checkpoint
sample.py         # checkpoint 采样
evaluate.py       # FID 评估
model/            # sinusoidal embedding、ResBlock、U-Net
configs/          # MNIST 和 CIFAR-10 配置
tests/            # 单元测试源码（本次未在仓库环境执行完整回归）
challenge.py      # 挑战档多调度策略、多随机种子实验编排与汇总
challenge_report.md # 挑战档八页技术报告结构稿
challenge_monitor.py # 挑战矩阵只读实时监控页面
monitor.py        # 本地只读实时训练进度监控
report.md         # 理论、实现和实际实验结果
debug_log.md      # 实际调试记录
logs/             # 实验日志模板
```

## FID≤15 v2 严格 200 epoch 实验

为保持和作业协议一致，本轮每个变体固定运行 200 epoch、约 78,000 次有效更新、seed 44，
并把输出写入互不覆盖的目录：

- `configs/cifar10_fid15_full_linear_default.yaml`：完整 DDPM U-Net + linear schedule + warmup 后固定学习率；
- `configs/cifar10_fid15_full_linear_cosinelr.yaml`：完整 DDPM U-Net + linear schedule + warmup 后 cosine 学习率衰减；
- `configs/cifar10_fid15_full_cosine_default.yaml`：完整 DDPM U-Net + 修正后的 cosine schedule + warmup 后固定学习率。

每个运行同时保存 EMA decay `0.999、0.9995、0.9999`，以便在相同 checkpoint 上比较 EMA；
`diagnose_schedule.py` 可记录反向采样中的 `pred_x0` 越界比例和各时间步幅值。

AutoDL 上可用 `run_fid15_v2.sh` 按顺序运行三个实验。训练期间启动只读多实验监控：

```bash
python -u experiment_monitor.py \
  --run R1=runs/fid15_v2_full_linear_default \
  --run R2=runs/fid15_v2_full_linear_cosinelr \
  --run R3=runs/fid15_v2_full_cosine_default \
  --total-steps 78000 \
  --pid-file runs/fid15_v2_runner.pid \
  --log-path logs/fid15_v2_runner.log \
  --host 127.0.0.1 --port 8765
```

本页面每 2 秒刷新，训练结束后只读显示最终 checkpoint。v2 实际结果已经记录在
`results/fid15_v2/primary_results.md`；R2 的恢复过程和磁盘问题见 `debug_log.md`。

### v2 实际 primary 结果

| 实验 | 未裁剪 x0 | 裁剪 x0 |
|---|---:|---:|
| R1 full linear default | 18.8045 | 18.8144 |
| R2 full linear cosine-LR | 19.4615 | 19.4633 |
| R3 full cosine default | 162.8797 | 16.1686 |

三组均使用 5,000 张生成图像、5,000 张无增强训练图像、seed 44 和 EMA 0.9999。
primary 最优为 R3 裁剪 x0 的 16.1686，仍高于 FID≤15；EMA 多 decay 对比结果见
`results/fid15_v2/ema_comparison_clipx0.md`。

R3 的 EMA decay 对比为：EMA 0.999=`15.7886`、EMA 0.9995=`15.5340`、EMA
0.9999=`16.1686`、raw=`66.6401`。因此 v2 最终最佳为 EMA 0.9995 的 `15.5340`，
距离目标差 `0.5340`，仍未达标。cosine 未裁剪诊断在 `t=999` 的 `pred_x0_abs_max`
约为 `964.98`、越界比例约 `99.43%`；逐步裁剪 x0 后 FID 大幅下降，但仍不足以达到 15。

### v2 补充评估与采样验证

R1/R2 在固定 `train` 前 5,000 张真实图、5,000 张生成图、seed 44、裁剪预测 $x_0$
协议下补充比较 EMA：R1 的 EMA 0.999/0.9995 为 `18.9792/18.5210`，R2 为
`20.3343/20.1602`。R3 历史 checkpoint 的 EMA 0.9995 FID 为：10k=`40.9638`、
20k=`24.6445`、30k=`19.8270`、40k=`17.9806`、50k=`16.9291`、60k=`16.4440`、
70k=`15.9460`；没有早于最终步数的更低值。

已单独生成 `results/fid15_v2/R3_full_cosine_default_ema9995_clipx0_grid.png`，它对应
最佳 EMA 0.9995 和裁剪采样，不使用旧的默认 EMA 0.9999 未裁剪网格。采样公式验证使用
固定 $x_t$、固定预测噪声和 `t=999,998,0` 对照独立 posterior 实现：cosine 最大误差
`5.7964e-7`，linear 最大误差 `1.2450e-4`，均通过 `2e-4` 容差。R3 诊断 JSON 现在同时
记录 median/P95/P99/max，用来区分低 $α\_bar$ 下正常的噪声放大与实现错误。

## 核心公式

前向过程使用闭合形式：

```text
x_t = sqrt(alpha_bar_t) * x_0
    + sqrt(1 - alpha_bar_t) * epsilon
```

模型预测加入的噪声，训练目标为噪声 MSE。采样时从标准高斯噪声开始，依次执行 `T-1` 到 `0` 的反向步骤；最后一步不加入随机扰动。

## 后续运行方式

安装依赖后，可运行：

```bash
python train.py --config configs/mnist.yaml
python sample.py --ckpt runs/exp_mnist_baseline/ckpt/final.pt --num_samples 64 --save_grid
```

训练进行中时，可以启动本地只读监控页面。页面每 2 秒读取一次样本、checkpoint 和 loss 文件：

```bash
python monitor.py --run-dir runs/exp_mnist_baseline \
    --total-steps 23400 --train-pid <训练进程 PID> --port 8765
```

然后打开 <http://127.0.0.1:8765/>。步数会随着最新持久化产物更新；`loss_history.csv` 出现后会自动绘制 loss 曲线。

CIFAR-10 的长训练使用：

```bash
python train.py --config configs/cifar10.yaml
python sample.py --ckpt runs/exp_cifar10_advanced/ckpt/final.pt \
    --num_samples 64 --save_grid
python evaluate.py --ckpt runs/exp_cifar10_advanced/ckpt/final.pt \
    --num_samples 5000 --batch_size 64 --real_split train --compare_ema
```

实验结果已经同步到 `report.md`、`debug_log.md`、`challenge_report.md`、`results/challenge/` 和 `logs/`。
由于大型二进制文件不进入普通 Git，挑战档 checkpoint 和中间样本请从
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1) 下载。

FID 改进实验的最终 EMA checkpoint、各阶段 checkpoint 和原始 source checkpoint 不进入普通 Git；
请从对应 Release 下载。仓库中保留可复核的 FID 文本、配置、loss history、loss 曲线、最终样本网格
和实验摘要。

## R5/R6 最终复核结果

在 R5、R6 中保持完整 U-Net、cosine beta、seed 44、200 epoch、约 78,000 次有效更新和固定
`train` 前 5,000 张真实图像的评估协议。两组均使用 5,000 张生成图像、裁剪预测 `x0`，并比较
EMA 0.999、0.9995、0.9999 与 raw：

| 实验 | 训练损失 | EMA 0.9990 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R5 late decay | uniform MSE | 16.3823 | 16.2361 | **15.4385** | 16.1927 |
| R6 late decay | Min-SNR-$\\gamma=5$ | 17.2559 | 16.9445 | **15.8562** | 17.8510 |

当前最佳为 R5 EMA 0.9999 的 `15.4385`，距离 FID≤15 仍差 `0.4385`，所以本轮没有达标。
R6 相比 R5 的最佳结果高 `0.4177`，说明在当前 200 epoch、学习率和 cosine beta 设置下，
Min-SNR-$\\gamma=5$ 没有带来收益。完整结果、配置、loss 历史、样本网格和日志见
[`results/fid15_final/`](results/fid15_final/)，大型 checkpoint 见
[`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。
其中 `best_grid_ema9999_clipx0.png` 与各自当前最佳 FID 权重一致；`final_grid_ema9995_clipx0.png`
保留为训练完成时的 EMA 0.9995 参考网格。

## FID≤15 最终改进运行：R4 Min-SNR

最终执行方案只新增一组训练，不改变 R3 的 200 epoch、78,000 次有效更新、cosine
beta、完整 U-Net、batch size 128、seed 44、warmup、AMP 和 EMA 设置；唯一训练改动是
对 epsilon MSE 使用 Min-SNR-$\gamma=5$ 权重。配置为
`configs/cifar10_fid15_r4_min_snr.yaml`，独立输出目录为
`runs/fid15_r4_min_snr`，启动脚本为 `run_fid15_r4.sh`。结果产生前不填写 FID，避免虚构
实验结论。

训练与评估期间可用以下只读页面实时查看 step、loss、GPU、checkpoint 和样本：

```bash
python -u experiment_monitor.py \\
  --run R4_min_snr=runs/fid15_r4_min_snr \\
  --total-steps 78000 \\
  --pid-file runs/fid15_r4_min_snr.pid \\
  --log-path logs/fid15_r4_min_snr.log \\
  --host 127.0.0.1 --port 8765
```

然后通过 SSH 本地端口转发访问 `http://127.0.0.1:8765/`。训练完成后脚本首先用 EMA
0.9995、裁剪 $x_0$ 和固定 5,000/5,000 train split 评估，再补充 EMA bank 对比。

## Stage 1：FID 稳定性与 timestep 诊断

在不重训的前提下，新增 `stage1_diagnostics.py` 对 R3 EMA 0.9995 与 R5 EMA 0.9999
进行 seed 44/45/46 稳定性评估、real-vs-real 校准和固定噪声的逐 timestep 误差分析。R3
三 seed 均值为 `15.5312`，R5 为 `15.3717`；R5 的 seed 44 正式值为 `15.4385`，
seed 46 的最低值为 `15.2086`，但没有一次低于 15。real-vs-real FID 为 `10.2039`，
输入转换统计一致，未发现简单的数据范围或 uint8 转换错误。

诊断确认 cosine schedule 在 `t=998/999` 的极小 `alpha_bar` 会放大 epsilon 误差，
所以逐步 clipping 是必要的稳定化措施；R3/R5 的 train/test timestep 曲线基本重合，
没有发现明显过拟合或 checkpoint 损坏。完整数据和解释见
[`results/fid15_stage1/README.md`](results/fid15_stage1/README.md)。

### Stage 1 补充：真实子集与生成种子方差

为检验“随机抽取 5,000 张真实图像是否可能把 FID 推过 15”，新增
`stage1_real_subset_fid.py`。在不重训、不更换模型的前提下，固定 CIFAR-10 train split，
对 R3 EMA 0.9995 和 R5 EMA 0.9999 分别使用正式的前 5,000 张图，以及 4 组每类 500 张、
互不重叠的 class-stratified 子集；每个真实子集又使用生成 seed 44/45/46，共 30 个 FID
单元格。生成图只对每个模型/seed 采样一次并缓存，避免真实子集变化重复 1,000 步反向采样。

| 实验 | 15 个单元格均值 | 样本标准差 | 最低 | 最高 | 正式子集 3-seed 均值 |
|---|---:|---:|---:|---:|---:|
| R3 EMA 0.9995 | 15.5096 | 0.1104 | 15.3157 | 15.6866 | 15.5312 |
| R5 EMA 0.9999 | 15.3590 | 0.1270 | 15.1445 | 15.5563 | 15.3716 |

30 组全部高于 15；最低值为 R5、`stratified_1`、seed 46 的 `15.1445`。R5 正式协议
（前 5,000 张、seed 44）仍为 `15.4385`，因此不能用事后挑选的随机子集宣称达标。随机
子集会造成可见变化，例如 R5 seed 44 从正式子集的 `15.4385` 变为 `stratified_1` 的
`15.2631`，但这只能说明评估方差，不能解释当前模型稳定达到 FID≤15。完整 CSV、聚合
JSON、协议和复现命令见 [`results/fid15_stage1/real_subset/`](results/fid15_stage1/real_subset/)。
