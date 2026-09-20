# 50 epoch 学习率续训粗筛

## 实验目的

本实验不是从头训练，而是将 30 epoch 粗筛中保留的 `2.5e-4` 与 `3.0e-4` 两个候选分别从其 30 epoch `final.pt` 继续训练到 50 epoch，用较低成本判断较高学习率是否会在更长训练阶段追上或超过低学习率候选。

## 训练设置

- 数据集：CIFAR-10，`train` split，无增强
- 模型：完整 skip-connection U-Net
- beta schedule：cosine，`T=1000`
- prediction：epsilon
- loss：uniform epsilon MSE
- batch size：128
- seed：44
- EMA：`0.999 / 0.9995 / 0.9999`
- warmup：1,000 steps，之后保持候选峰值学习率
- 总训练量：50 epoch，`19,550` 个有效更新
- 起点：30 epoch，`11,730` 个有效更新
- 本轮新增：`7,820` 个有效更新
- 输出 checkpoint：每个候选的 `final.pt` 未进入普通 Git

每组的配置副本见 [`configs/`](configs/)，训练编排脚本见项目根目录的 [`run_lr_screen_resume_50ep.sh`](../../run_lr_screen_resume_50ep.sh)。

## 快速 FID 协议

本轮只用于筛选，固定使用：

- 1,000 张生成图与 1,000 张 `train` split 真实图
- 生成 seed `44`
- EMA `0.9999`
- 逐步裁剪预测的 `x0`
- batch size `64`

因此下面的数值不是正式的 5,000/5,000 FID，不能直接与 R5 的正式 FID `15.4385` 比较，也不能据此宣称达到 FID≤15。

## 结果

| 候选 | 30 epoch 快速 FID | 50 epoch 快速 FID | 变化 |
|---|---:|---:|---:|
| LR `2.5e-4` | 330.7679 | **193.6462** | -137.1217 |
| LR `3.0e-4` | 323.4893 | 202.6441 | -120.8452 |

两组训练均正常完成，最终训练日志的最后记录为：

- LR `2.5e-4`：loss `0.04731`，训练耗时约 `16.1` 分钟；
- LR `3.0e-4`：loss `0.04724`，训练耗时约 `16.1` 分钟。

训练日志中出现少量 AMP optimizer step skip，scaler 随后继续工作；未见 OOM、NaN 或 checkpoint 损坏。原始轻量日志、FID 文本和配置保存在本目录的 `runs/`、`combined.log` 和 `configs/` 中。

## 结论

1. 从 30 epoch 继续到 50 epoch 对两个候选都带来了明显的快速 FID 改善，说明 30 epoch 的粗筛不能代表较长训练的最终排序。
2. 在本次固定的 50 epoch 快速协议下，`2.5e-4` 优于 `3.0e-4`，因此如果继续做正式复核，应优先选择 `2.5e-4`。
3. 这只是 1,000 样本快速 FID。正式结论仍需固定同一协议完成 5,000/5,000 评估，并记录完整 EMA bank 对照。
4. 由于 R5 已经在 200 epoch、后期学习率衰减和正式协议下达到 `15.4385`，本轮 50 epoch 结果不足以证明更高学习率会突破 FID 15；它只说明 `2.5e-4` 值得作为后续单变量实验候选。

## 复现

先准备与 30 epoch 粗筛相同的数据和环境，再从相应的 30 epoch checkpoint 运行：

```bash
bash run_lr_screen_resume_50ep.sh
```

如果需要正式比较，应将评估样本数改为 5,000，并保持 `train` 前 5,000 张真实图、seed `44`、EMA decay、`clip_denoised` 和 torch-fidelity 版本不变。
