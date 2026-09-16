# FID≤15 改进实验摘要

## 固定评估协议

- 数据：CIFAR-10，real split 为无增强 `train`；
- 生成样本：5,000；
- 模型权重：EMA；
- seed：44；
- 硬件：AutoDL NVIDIA GeForce RTX 4090；
- 采样差异：是否逐步裁剪预测的 `x_0` 到 `[-1, 1]`。

## 结果

| checkpoint | 采样方式 | FID |
|---|---|---:|
| 原始 200 epoch / seed44 | 未裁剪 | 18.9577 |
| 原始 200 epoch / seed44 | 裁剪预测 `x_0` | 18.9715 |
| 100,000 有效更新 | 未裁剪 | 18.1502 |
| 100,000 有效更新 | 裁剪预测 `x_0` | **18.1490** |

续训将原始未裁剪基线降低 0.8075；最终裁剪版降低 0.8087。最佳 FID 为 18.1490，
仍未达到 FID≤15。

## 运行产物

- 配置：`configs/cifar10_fid15_resume.yaml`
- 训练日志：`results/fid15_4090_linear_seed44/train_resume.log`
- loss history：`results/fid15_4090_linear_seed44/loss_history.csv`
- loss 曲线：`results/fid15_4090_linear_seed44/loss_curve.png`
- 最终样本网格：`results/fid15_4090_linear_seed44/samples_final_ema_grid.png`
- 大型 checkpoint：对应 GitHub Release 附件，不放入普通 Git。
