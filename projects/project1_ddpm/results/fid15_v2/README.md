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
