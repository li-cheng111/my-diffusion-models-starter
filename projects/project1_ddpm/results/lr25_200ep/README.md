# LR=2.5e-4：50→200 epoch 正式复核

## 实验设置

本实验从 50 epoch 学习率粗筛的 LR `2.5e-4` checkpoint 继续训练到总计 200 epoch，未从头训练。

- 起始 checkpoint：50 epoch、总步数 `19,550`
- 最终训练步数：`78,000`
- 新增有效更新：`58,450`
- 数据集：CIFAR-10 `train` split，无增强
- 模型：完整 skip-connection U-Net
- beta schedule：cosine，`T=1000`
- prediction：epsilon
- loss：uniform epsilon MSE
- batch size：128，seed `44`
- EMA：`0.999 / 0.9995 / 0.9999`
- 学习率：峰值 `2.5e-4`，step `58,500` 后 late cosine decay，最低 `2.5e-5`
- 采样：逐步裁剪预测的 `x0`

配置和启动脚本分别见 [`configs/cifar10_lr25_200ep_resume.yaml`](../../configs/cifar10_lr25_200ep_resume.yaml) 和 [`run_lr25_200ep_resume.sh`](../../run_lr25_200ep_resume.sh)。

本目录同时保存了实际运行配置 `config.yaml`、完整 loss 历史 `loss_history.csv`、loss 曲线
`loss_curve.png` 和 EMA 0.9999 裁剪采样网格 `final_grid_ema9999_clipx0.png`。

## 正式 FID

固定协议为 5,000 张生成图、CIFAR-10 train split 前 5,000 张真实图、seed `44`、batch size `64`、逐步裁剪 `x0`。

| 权重 | FID |
|---|---:|
| EMA `0.9990` | 16.0761 |
| EMA `0.9995` | 15.8897 |
| EMA `0.9999` | **15.6761** |
| raw | 16.8996 |

训练耗时约 `122.5` 分钟，最终 EMA 0.9999 样本网格已生成。训练期间出现少量 AMP optimizer step skip，随后自动恢复；没有 OOM、NaN 或 checkpoint 损坏。

## 结论

- 本次最佳结果为 EMA `0.9999` 的 `15.6761`，仍未达到 FID≤15。
- 与此前 R5 的最佳正式结果 `15.4385` 相比，高 `0.2376`；在当前 200 epoch 预算下，LR `2.5e-4` 没有优于 R5 的 LR `2e-4` 配置。
- EMA 明显优于 raw，raw 与 EMA 0.9999 的差值为 `1.2235`，说明参数平均仍然有效，但不足以弥补学习率/训练轨迹差异。
- 50 epoch 快速筛选中的优势没有延续到正式 200 epoch，因此早期 1,000 样本 FID 只能用于排除候选，不能代替完整训练判断。

完整结果文件、loss 历史、loss 曲线和最终样本网格位于本目录；大型 checkpoint 不进入普通 Git，
通过 [`challenge-v1 Release`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)
提供：[`lr25_200ep_final.pt`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/lr25_200ep_final.pt)。

Release checkpoint SHA-256：
`5c854456880228d31975db4927e986768fbbc0acb7a92bddef9b5f1559ac8070`。
