# FID≤15 v2：primary 结果

本目录记录严格 200 epoch 的 v2 改进实验。三组实验均使用单卡 NVIDIA GeForce RTX 4090、CIFAR-10、seed 44、5,000 张生成图像对 5,000 张无增强训练图像计算 FID，默认使用 EMA 0.9999。

| 实验 | 主要变化 | 未裁剪 x0 | 裁剪 x0 |
|---|---|---:|---:|
| R1 full linear default | 完整 U-Net + linear beta + warmup 后固定学习率 | 18.8045 | 18.8144 |
| R2 full linear cosine-LR | 完整 U-Net + linear beta + warmup 后余弦学习率衰减 | 19.4615 | 19.4633 |
| R3 full cosine default | 完整 U-Net + cosine beta + warmup 后固定学习率 | 162.8797 | 16.1686 |

评估命令固定使用 `--real_split train --seed 44 --num_samples 5000 --batch_size 64`。R2 的原始 `final.pt` 曾因数据盘空间不足写入损坏；本次从有效的 `step_070000.pt` 恢复到 step 78000，并重新生成最终 checkpoint 后再评估。

当前 primary 最优为 R3 的裁剪 x0 结果 `16.1686`，距离 FID≤15 还差 `1.1686`。

## R1/R2 EMA 补充评估

为排除 EMA decay 选择造成的影响，R1、R2 在同一固定协议下再次评估 EMA 0.999 和
0.9995：`train` split 前 5,000 张真实图、5,000 张生成图、seed 44、裁剪预测 $x_0$。

| 实验 | EMA 0.999 | EMA 0.9995 |
|---|---:|---:|
| R1 full linear default | 18.9792 | 18.5210 |
| R2 full linear cosine-LR | 20.3343 | 20.1602 |

在这次固定评估中，R1 的 EMA 0.9995 比 EMA 0.999 低 `0.4582`，R2 低 `0.1741`；
但两者仍高于 R3 最终 checkpoint 的 EMA 0.9995 结果 `15.5340`。

## R3 历史 checkpoint 评估

本节记录 `step_010000.pt` 至 `step_070000.pt` 的 EMA 0.9995、裁剪预测 $x_0$ 结果。
评估协议与上表一致。

| checkpoint | FID |
|---|---:|
| step 010000 | 40.9638 |
| step 020000 | 24.6445 |
| step 030000 | 19.8270 |
| step 040000 | 17.9806 |
| step 050000 | 16.9291 |
| step 060000 | 16.4440 |
| step 070000 | 15.9460 |

R3 历史 checkpoint 的 FID 随训练步数持续下降，但在 70k 仍高于最终 checkpoint 的
`15.5340`；因此本次未观察到“最终步数之前更低”的情况。历史点的独立 FID 文本文件为
`R3_step*_ema9995_clipx0.txt`，主任务日志为 `fid15_point1.log`。

## 采样实现验证

固定 $x_t$、预测噪声和 `t=999,998,0` 的独立公式对照已通过。cosine 的最大误差为
`5.7964e-7`，linear 的最大误差为 `1.2450e-4`，均低于 `2e-4` 容差。linear 的误差
峰值出现在 `t=0` 的直接 epsilon 形式，是 float32 schedule 系数在近似相减时的数值误差，
不是 posterior mean 或 variance 公式错误。统计详情见 `sampler_validation_*.json`。

## R3 EMA decay 对比

同一 R3 checkpoint、同一 5,000/5,000 train split、seed44、裁剪 x0 协议：

| 权重 | FID |
|---|---:|
| EMA 0.999 | 15.7886 |
| EMA 0.9995 | **15.5340** |
| EMA 0.9999 | 16.1686 |
| raw | 66.6401 |

最终最佳值为 EMA 0.9995 的 `15.5340`，距离目标仍差 `0.5340`，因此本轮未达标。

cosine 反向诊断见 `schedule_diagnostics_noclipx0.json` 和 `schedule_diagnostics_clipx0.json`：
未裁剪在 `t=999` 出现 `pred_x0_abs_max≈964.98`、越界比例≈99.43%；裁剪后末端
`x_t_abs_max≈1.04`、越界比例约 0.3%。
