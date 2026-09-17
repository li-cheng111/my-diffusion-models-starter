# FID≤15 v2 结果索引

本目录保存 AutoDL RTX 4090 上完成的三组严格 200 epoch 实验的小型可复核结果。
主结果、固定评估协议和最终判断见 [primary_results.md](primary_results.md)，EMA 对比见
[ema_comparison_clipx0.md](ema_comparison_clipx0.md)。

## 实验目录对应关系

- `R1_full_linear_default_*`：完整 U-Net + linear beta + 常规学习率。
- `R2_full_linear_cosinelr_*`：完整 U-Net + linear beta + cosine 学习率尾段。
- `R3_full_cosine_default_*`：完整 U-Net + cosine beta + 常规学习率。

每组的 `grid.png` 和 `loss_curve.png` 是最终小型可视化产物；`schedule_diagnostics_*.json`
记录 R3 采样轨迹诊断。checkpoint、MNIST/CIFAR-10 数据和大量中间样本不放入普通 Git。

## 本轮补充评估

R1、R2 的 EMA 对比固定使用 `train` split 前 5,000 张真实图、5,000 张生成图、seed 44
和裁剪预测 $x_0$。结果如下：

| 实验 | EMA 0.999 | EMA 0.9995 |
|---|---:|---:|
| R1 full linear default | 18.9792 | 18.5210 |
| R2 full linear cosine-LR | 20.3343 | 20.1602 |

R3 早期 checkpoint 的同协议评估已在 `primary_results.md` 中按 checkpoint 步数记录；
不要将早期高 FID 误解为 schedule 实现失败。当前 R3 最终 checkpoint 另生成了
`R3_full_cosine_default_ema9995_clipx0_grid.png`，用于对应最佳 EMA 0.9995 的裁剪采样，
不再沿用默认 EMA 0.9999 的历史网格作为最佳结果。

## 采样公式验证

`sampler_validation_cosine.json` 和 `sampler_validation_linear.json` 使用固定 $x_t$、固定噪声
预测和 `t=999,998,0`，将生产 `p_sample` 与独立实现的 posterior mean、variance 及裁剪路径
逐项比较。两种 schedule 均通过容差 `2e-4`：cosine 最大误差约 `5.80e-7`，linear 最大误差
约 `1.25e-4`。linear 在 `t=0` 的较大误差来自 float32 schedule 系数造成的近似相减数值误差，
不是 posterior 公式不一致。

R3 cosine 的轨迹统计位于 `schedule_diagnostics_noclipx0.json` 和
`schedule_diagnostics_clipx0.json`，每个时间步同时记录 median、P95、P99、max 和越界比例。
这些统计用于区分“正常的高噪声放大”与实现错误：例如无裁剪在 `t=999` 的 `pred_x0` median/P95/P99/max
约为 `90.15/280.88/385.49/964.98`，而不是只看最后一个 max；裁剪在 `t=0` 将相应的
P95/P99 控制到约 `0.94/0.99`。
