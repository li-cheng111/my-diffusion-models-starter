# FID≤15 v2：primary 结果

本目录记录严格 200 epoch 的 v2 改进实验。三组实验均使用单卡 NVIDIA GeForce RTX 4090、CIFAR-10、seed 44、5,000 张生成图像对 5,000 张无增强训练图像计算 FID，默认使用 EMA 0.9999。

| 实验 | 主要变化 | 未裁剪 x0 | 裁剪 x0 |
|---|---|---:|---:|
| R1 full linear default | 完整 U-Net + linear beta + warmup 后固定学习率 | 18.8045 | 18.8144 |
| R2 full linear cosine-LR | 完整 U-Net + linear beta + warmup 后余弦学习率衰减 | 19.4615 | 19.4633 |
| R3 full cosine default | 完整 U-Net + cosine beta + warmup 后固定学习率 | 162.8797 | 16.1686 |

评估命令固定使用 `--real_split train --seed 44 --num_samples 5000 --batch_size 64`。R2 的原始 `final.pt` 曾因数据盘空间不足写入损坏；本次从有效的 `step_070000.pt` 恢复到 step 78000，并重新生成最终 checkpoint 后再评估。

当前 primary 最优为 R3 的裁剪 x0 结果 `16.1686`，距离 FID≤15 还差 `1.1686`。

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
