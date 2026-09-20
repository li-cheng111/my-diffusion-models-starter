# LR=5e-5 完整 200 epoch 复核

## 结论

LR=5e-5 已完成完整 200 epoch、78,000 有效更新，并按正式协议完成 5,000 样本 FID
评估。最佳结果为 EMA 0.9999 的 `16.5231`，没有超过 R5 的 `15.4385`，距离
FID≤15 仍差 `1.5231`。

## 固定协议

- 数据集：CIFAR-10 train split 前 5,000 张真实图，无增强
- 生成图：5,000 张；生成 seed：44；batch size：64
- 采样：`clip_denoised=True`，逐步裁剪预测的 x0
- 模型：完整 skip-connection U-Net，base channels 128，16×16 attention
- beta：cosine schedule，`T=1000`
- 训练目标：uniform epsilon MSE
- 学习率：峰值 `5e-5`，warmup 5,000 steps；58,500 步后 cosine 衰减到 `5e-6`
- EMA：`0.999 / 0.9995 / 0.9999`
- 硬件：AutoDL NVIDIA GPU；训练耗时 156.5 分钟

## FID 结果

| 权重 | FID |
|---|---:|
| EMA 0.9990 | 17.4472 |
| EMA 0.9995 | 17.3407 |
| EMA 0.9999 | **16.5231** |
| raw | 18.6642 |

相同正式协议下，R5 的结果为 `16.3823 / 16.2361 / 15.4385 / 16.1927`。因此 LR=5e-5
相对 R5 的差值为：

| 权重 | LR=5e-5 | R5 | 差值 |
|---|---:|---:|---:|
| EMA 0.9990 | 17.4472 | 16.3823 | +1.0649 |
| EMA 0.9995 | 17.3407 | 16.2361 | +1.1046 |
| EMA 0.9999 | 16.5231 | 15.4385 | +1.0846 |
| raw | 18.6642 | 16.1927 | +2.4715 |

## 分析

1. 30 epoch 粗筛中 LR=5e-5 的快速 FID 最低，但完整 200 epoch 后仍明显差于 R5。这说明
   30 epoch 粗筛只适合排除高学习率，不能直接预测完整训练的最终 FID。
2. 在同一 seed、真实子集、模型、beta schedule、EMA 和采样协议下，LR=5e-5 没有改善，
   说明当前 R5 的 `2e-4` 峰值学习率更适合在有限 200 epoch 内完成有效拟合。
3. 本次将最小学习率按峰值比例设置为 `5e-6`，以保持 R5 的 10% 衰减比例；R5 使用的是
   绝对值 `2e-5`。因此结果说明“低峰值 + 同比例晚期衰减”不如 R5，但不能单独分离
   `min_lr` 绝对值的影响。
4. EMA 0.9999 仍然显著优于 raw，说明参数平均仍是必要的；但 EMA 不能弥补峰值学习率
   过低导致的有效学习不足。
5. 一次 AMP step skip 出现在 step 76,471 附近，训练随后正常完成；没有 OOM、NaN 或
   checkpoint 损坏证据，不足以解释 1.0846 的 FID 差距。

## 产物

- `config.yaml`：实际运行配置
- `loss_history.csv`、`loss_curve.png`：训练曲线
- `train.log`、`eval.log`：训练和评估日志
- `ckpt/fid_*.txt`、`ckpt/fid_comparison_clipx0.md`：正式 FID 记录
- `samples/`：训练过程样本网格

最终 checkpoint 已上传到 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)：

`https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/lr5e5_full_200ep_final.pt`
