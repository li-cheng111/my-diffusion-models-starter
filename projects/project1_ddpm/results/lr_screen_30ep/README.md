# 30-epoch 学习率粗筛结果

## 结论

本轮在 AutoDL 的 NVIDIA GPU 上完成了 7 组独立 CIFAR-10 粗筛，所有任务均正常结束：

```text
7 / 7 runs completed
每组 11,730 个有效更新，总计 82,110 个有效更新
无 OOM、NaN 或 AMP 异常记录
```

在固定的快速评估协议下，`5e-5` 是本轮最优候选，FID 为 `238.0244`。但该数值使用
1,000 张生成图和 1,000 张 train split 真实图，仅用于粗筛，不能与正式的 5,000 对 5,000、
200 epoch FID 直接比较，也不能据此宣称达到 FID≤15。

## 固定实验条件

- 数据集：CIFAR-10 train split，图像尺寸 32×32，seed 44
- 模型：完整 skip-connection U-Net，base channels 128，attention resolution 16
- diffusion：cosine beta schedule，`T=1000`
- 预测目标：epsilon prediction
- batch size：128；梯度累积：1；AMP：fp16
- 训练：30 epoch，11,730 steps，前 1,000 steps warmup，之后保持常数学习率
- EMA：训练时维护 `0.999 / 0.9995 / 0.9999`
- 快速 FID：EMA `0.9999`、`clip_denoised=True`、seed 44、real split=train、1,000 对 1,000
- 本轮没有使用 R5 的后期 cosine 学习率衰减，因此不能把粗筛配置视为 R5 的完全复刻

## FID 结果

| 标签 | 学习率 | 最终记录 loss | 快速 FID | 判断 |
|---|---:|---:|---:|---|
| LR05 | 5.0e-5 | 0.05284 | **238.0244** | 首选 |
| LR08 | 8.0e-5 | 0.05245 | 263.5649 | 次选 |
| LR12 | 1.2e-4 | 0.05214 | 291.7750 | 淘汰 |
| LR16 | 1.6e-4 | 0.05209 | 284.0970 | 淘汰 |
| LR20 | 2.0e-4 | 0.05199 | 307.1943 | 淘汰 |
| LR25 | 2.5e-4 | 0.05193 | 330.7679 | 淘汰 |
| LR30 | 3.0e-4 | 0.05207 | 323.4893 | 淘汰 |

## 分析

1. `5e-5` 比第二名 `8e-5` 低 25.5405 FID 点，比 `2e-4` 低 69.1699 点。在当前早期
   预算下，较高学习率明显破坏了采样质量。
2. 各组最终 loss 非常接近，甚至高学习率的 loss 略低，但 FID 更差。这说明平均 epsilon
   MSE 不能单独代表完整反向链的质量，尤其不能替代中高噪声时间步误差、EMA 时间常数和
   采样轨迹的检查。
3. 30 epoch 仍处在明显的早期阶段；此时高学习率可能在训练 loss 上收敛更快，但会让 EMA
   权重和反向采样分布尚未稳定。因此本轮只能排除 `1.2e-4` 以上的候选，不能证明
   `5e-5` 在完整 200 epoch 中一定优于 R5 的 `2e-4`。
4. 快速 FID 只用 1,000 张图，数值包含较大的估计方差。候选之间差距很大，粗排序有参考
   价值；最终结论必须用相同 seed、5,000 对 5,000 的正式协议复核。

## 后续建议

优先对 `5e-5` 和 `8e-5` 做完整 200 epoch 复核，并恢复正式实验的后期学习率衰减策略；
`1.2e-4`、`1.6e-4`、`2.0e-4`、`2.5e-4`、`3.0e-4` 暂不值得继续消耗完整训练预算。
正式复核至少记录 EMA `0.999 / 0.9995 / 0.9999`，再使用 5,000 样本确定最佳 EMA。

## 复现文件

- 候选配置：`configs/LR05.yaml` 至 `configs/LR30.yaml`
- 每组训练日志：`runs/LR*/train.log`
- 每组 loss 曲线与历史：`runs/LR*/loss_curve.png`、`runs/LR*/loss_history.csv`
- 每组快速 FID：`runs/LR*/ckpt/fid_1000_EMA_0.9999_clipx0.txt`
- 每组中间样本网格：`runs/LR*/samples/`
- 总日志：`combined.log`
- 本轮通用配置与启动脚本：`../../configs/cifar10_lr_screen_30ep_base.yaml`、
  `../../run_lr_screen_30ep.sh`

最优 LR05 的大型最终 checkpoint 不进入普通 Git，已作为 Release 附件保存：

`https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/lr_screen_30ep_LR05_final.pt`
