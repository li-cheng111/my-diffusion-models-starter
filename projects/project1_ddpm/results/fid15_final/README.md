# R5/R6 最终 FID 复核结果

本目录保存 R5、R6 两组严格 200 epoch 实验的轻量复现材料。大型 checkpoint 不进入普通 Git，下载地址见仓库的 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)。

## 固定评估协议

- 数据集：CIFAR-10
- 真实图像：`train` split 前 5,000 张，无增强
- 生成图像：5,000 张
- 采样 seed：44
- batch size：64
- 采样：逐步裁剪预测的 `x0`（`clip_denoised=True`）
- 评估：torch-fidelity FID
- 硬件：AutoDL NVIDIA GeForce RTX 4090
- 每组训练：200 epoch，最大 78,000 次有效更新

## 结果

| 实验 | 训练损失 | EMA 0.9990 | EMA 0.9995 | EMA 0.9999 | raw |
|---|---|---:|---:|---:|---:|
| R5 late decay | uniform MSE | 16.3823 | 16.2361 | **15.4385** | 16.1927 |
| R6 late decay | Min-SNR-$\\gamma=5$ | 17.2559 | 16.9445 | **15.8562** | 17.8510 |

当前最佳是 R5 的 EMA 0.9999，FID `15.4385`，距离目标 FID≤15 还差 `0.4385`。因此本轮结论是：**结果已完成复核，但尚未达标**。

R6 的 Min-SNR 加权在本实验设置下没有优于 R5：最佳结果高 `0.4177` FID。两组实验都显示 EMA 0.9999 优于 0.9990、0.9995 和 raw，说明后期参数平均仍然是当前配置的重要组成部分。

## 训练设置差异

R5 使用完整 U-Net、cosine beta schedule、warmup 后 late cosine learning-rate decay，学习率从 `2e-4` 衰减到 `2e-5`，训练损失为 uniform epsilon MSE。

R6 与 R5 保持模型、schedule、学习率、seed、训练预算和评估协议一致，只把训练损失改为 Min-SNR-$\\gamma=5$。

## 文件说明

- `r5_late_decay/config.yaml`、`r6_min_snr_late_decay/config.yaml`：实际训练配置；
- `r5_late_decay/fid_comparison_clipx0.md`、`r6_min_snr_late_decay/fid_comparison_clipx0.md`：EMA/raw 对照；
- `fid_*.txt`：逐个 FID 的原始记录；
- `loss_history.csv`、`loss_curve.png`：训练 loss 历史和曲线；
- `best_grid_ema9999_clipx0.png`：与当前最佳 FID EMA 0.9999 对应的 64 张裁剪样本网格；
- `final_grid_ema9995_clipx0.png`：训练完成时按默认命令生成的 EMA 0.9995 参考网格；
- `samples/`：训练过程中的周期样本网格；
- `logs/`：R5/R6 训练与恢复评估日志。

轻量 Release 附件 `r5_r6_results_light.tar.gz` 包含上述复现材料；两个最终 checkpoint 以独立 Release 附件提供。

## 复现命令

下载 checkpoint 后，在项目目录中执行：

```bash
python evaluate.py \
  --ckpt runs/fid15_r5_late_decay/ckpt/final.pt \
  --num_samples 5000 \
  --batch_size 64 \
  --real_split train \
  --seed 44 \
  --compare_ema \
  --clip_denoised \
  --data_root ./data
```

R6 只需将 checkpoint 路径替换为对应 Release 附件即可。评估协议不能通过更换 real split、生成 seed 或样本数量来选择性优化。
