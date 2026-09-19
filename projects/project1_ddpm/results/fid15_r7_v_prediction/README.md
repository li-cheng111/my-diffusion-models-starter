# R7：v-prediction 复现实验

## 实验目的

R7 在保持 R5 主要设置不变的条件下，将 DDPM 的 epsilon-prediction 改为
v-prediction，检验 cosine beta schedule 在低信噪比端点的数值条件是否会得到改善。

## 训练配置

- 数据集：CIFAR-10
- 输入：32×32，RGB，训练集
- 训练预算：200 epoch，78,000 次有效更新
- 模型：完整 skip-connection U-Net，base channels=128，channel multipliers=1/2/2/2
- beta schedule：cosine，T=1000，s=0.008
- prediction target：`v_prediction`
- batch size：128
- optimizer：AdamW，初始学习率 `2e-4`
- warmup：5,000 steps
- 后期余弦衰减：step 58,500 开始，最低学习率 `2e-5`
- EMA：0.999、0.9995、0.9999
- seed：44
- AMP：fp16
- 训练耗时：156.6 分钟，NVIDIA RTX 4090

完整配置保存在 [`config.yaml`](config.yaml)，与仓库中的
[`configs/cifar10_fid15_r7_v_prediction.yaml`](../../configs/cifar10_fid15_r7_v_prediction.yaml)
对应。

## 固定 FID 协议

使用 CIFAR-10 `train` split 前 5,000 张真实图，生成 5,000 张图，生成 seed=44，
batch size=64；每个反向步骤都对预测的 `x0` 执行 `[-1, 1]` clipping。

## 结果

| 权重 | FID |
|---|---:|
| EMA 0.9990 | 19.0443 |
| EMA 0.9995 | 18.8484 |
| EMA 0.9999 | **17.4808** |
| raw | 19.2896 |

原始结果见 [`fid_comparison_clipx0.md`](fid_comparison_clipx0.md) 和各个
`fid_5000_*.txt` 文件。训练 loss 见 [`loss_curve.png`](loss_curve.png) 与
[`loss_history.csv`](loss_history.csv)，评估日志见 [`eval.log`](eval.log)，训练日志见
[`train.log`](train.log)。最佳 EMA 0.9999 的 64 张裁剪样本网格见
[`grid.png`](grid.png)。

## 复现命令

```bash
python train.py \
  --config configs/cifar10_fid15_r7_v_prediction.yaml \
  --output_dir runs/fid15_r7_v_prediction

python evaluate.py \
  --ckpt runs/fid15_r7_v_prediction/ckpt/final.pt \
  --num_samples 5000 \
  --batch_size 64 \
  --real_split train \
  --seed 44 \
  --compare_ema \
  --clip_denoised \
  --data_root ./data

python sample.py \
  --ckpt runs/fid15_r7_v_prediction/ckpt/final.pt \
  --num_samples 64 \
  --batch_size 64 \
  --ema_decay 0.9999 \
  --clip_denoised \
  --save_grid
```

大型 checkpoint 不进入普通 Git；复现实验使用的 `final.pt` 应从项目对应的 GitHub
Release asset 下载：[`r7_v_prediction_final.pt`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/r7_v_prediction_final.pt)。

## 结论

R7 没有改善 FID。最佳值 17.4808 比 R5 的 15.4385 高 2.0423，说明在当前 U-Net、
cosine schedule、训练预算和 clipping 组合下，单独切换到 v-prediction 不是有效改进。
后续应优先回到 R5 epsilon-prediction 基线，针对中高噪声 timestep 的 loss weighting、
schedule 参数和采样误差做单变量消融，而不是继续延长当前 R7。
