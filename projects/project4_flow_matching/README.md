# Project 4：CIFAR-10 Flow Matching

本项目在 CIFAR-10 上训练 Rectified Flow 图像生成模型，实现 Euler、Heun 和
Classifier-Free Guidance（CFG）采样，并用 FID 与 Project 1 的 DDIM 基线比较。v2版本里无条件模型 FM-U 与类别条件模型 FM-C 均已训练 200K 步，权重和完整评测归档发布在 [GitHub Release `project4-v2.0`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0)。

> 仓库还保留 v1 的配置、结果和复现清单，供历史参考。

## 一、方法与模型

线性路径从标准高斯噪声 `epsilon` 插值到数据图像 `x1`：

$$
x_t=(1-t)\epsilon+t x_1,\qquad
u_t=\frac{d x_t}{dt}=x_1-\epsilon,\qquad t\sim U[0,1].
$$

网络以均方误差预测速度场 `u_t`。采样从噪声端 `t=0` 积分到数据端 `t=1`：Euler
每步调用一次网络；Heun 先预测再校正，每步调用两次。表格中的 NFE 是每张图的实际
网络评估次数，不是积分步数。CFG 使用
`v = v_uncond + scale * (v_cond - v_uncond)`；其每步需两路预测，计算成本为普通采样的两倍。

| 实验 | 配置 | 训练方式 |
|---|---|---|
| FM-U | [`cifar10_fm_v2_unconditional.yaml`](configs/cifar10_fm_v2_unconditional.yaml) | 每个样本都使用 null 类别 token（`cond_drop_prob=1.0`），用于无条件生成 |
| FM-C | [`cifar10_fm_v2_conditional.yaml`](configs/cifar10_fm_v2_conditional.yaml) | 类别条件训练，10% 概率将标签替换为 null token，用于条件生成和 CFG |

两组模型都是 DiT-S，32.62M 参数，连续时间嵌入尺度为 1000。null token 的类别索引为
10（CIFAR-10 的 10 个类别索引为 0–9）。

## 环境、数据与训练

v2 训练环境为 Python 3.12.3、PyTorch 2.8.0+cu128、TorchVision 0.23.0+cu128，硬件为
AutoDL 单卡 NVIDIA RTX 4080 SUPER（32,760 MiB）。在仓库根目录安装依赖：

```bash
pip install -r requirements/project4.txt
```

训练脚本会通过 TorchVision 自动下载 CIFAR-10 到配置指定的 `data/`。如需使用实验所用镜像
预下载，压缩包为 170,498,071 字节，MD5 为 `c58f30108f718f92721af3b95e74349a`：

```bash
mkdir -p data
curl -fL -o data/cifar-10-python.tar.gz \
  https://hf-mirror.com/datasets/MIT-OL-AI-D/cifar-10-python/resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz
echo 'c58f30108f718f92721af3b95e74349a  data/cifar-10-python.tar.gz' | md5sum -c -
```

从仓库根目录分别启动两组训练：

```bash
python projects/project4_flow_matching/train.py \
  --config projects/project4_flow_matching/configs/cifar10_fm_v2_unconditional.yaml \
  --output runs/fm_v2_unconditional --seed 42 --precision bf16

python projects/project4_flow_matching/train.py \
  --config projects/project4_flow_matching/configs/cifar10_fm_v2_conditional.yaml \
  --output runs/fm_v2_conditional --seed 42 --precision bf16
```

两组配置均为 batch size 64、AdamW、200,000 optimizer steps、seed 42，维护 EMA 0.999、
0.9995、0.9999；前 2,000 步 warmup，从 100,000 步起 cosine 衰减。训练每 100 步记录
loss、学习率、速度和耗时；每 10,000 步更新 `latest.pt`，每 50,000 步保存快照。checkpoint
包含模型、优化器、scheduler、EMA 和随机数状态，可从最后完整保存点续训：

```bash
python projects/project4_flow_matching/train.py \
  --config projects/project4_flow_matching/configs/cifar10_fm_v2_unconditional.yaml \
  --output runs/fm_v2_unconditional --seed 42 --precision bf16 \
  --resume runs/fm_v2_unconditional/latest.pt
```

将配置和输出目录改为 FM-C 对应路径即可恢复 conditional 训练。v2 实测 FM-U 用时 4.83
小时，最终 loss 0.2013。FM-C 最终 checkpoint 记录 4.36 小时、loss 0.1665；训练进程在
193K 后退出，从 190K checkpoint 恢复到 200K，重算的 3K 步不计入该累计时间。FM-U 训练
期间 GPU 还运行了约 35 分钟的 Project 1 评测；因此这些时间是本次运行记录，不是硬件性能保证。

## 使用已发布权重

Release 提供 EMA 0.9999 的推理权重与评测归档。权重文件不纳入 Git：

| Release 文件 | 大小 | SHA-256 |
|---|---:|---|
| `project4-v2-unconditional-step200000.pt` | 522,147,167 B | `b75cbe3d4f9aeb16c66cb53e4acb03f9c2c6a18421efd945f78475cd2e114571` |
| `project4-v2-conditional-step200000.pt` | 522,146,099 B | `25666b32e914002778825a02f8e14c89f9f5ff081f9dded9f3f0ad6e0283c5aa` |
| `project4_v2_results_only.tar.gz` | 654,483 B | `e8ea255f9ae63f666b47bbdab30d494307285d9ba30b7276e248ce800bf8802a` |

下载权重及结果归档，并校验哈希：

```bash
mkdir -p checkpoints
wget -O checkpoints/project4-v2-unconditional-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4-v2-unconditional-step200000.pt
wget -O checkpoints/project4-v2-conditional-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4-v2-conditional-step200000.pt
wget -O project4_v2_results_only.tar.gz \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4_v2_results_only.tar.gz
wget -O project4_v2_results_only.tar.gz.sha256 \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4_v2_results_only.tar.gz.sha256
sha256sum -c project4_v2_results_only.tar.gz.sha256
echo 'b75cbe3d4f9aeb16c66cb53e4acb03f9c2c6a18421efd945f78475cd2e114571  checkpoints/project4-v2-unconditional-step200000.pt' | sha256sum -c -
echo '25666b32e914002778825a02f8e14c89f9f5ff081f9dded9f3f0ad6e0283c5aa  checkpoints/project4-v2-conditional-step200000.pt' | sha256sum -c -
```

结果归档包含评测 JSON、样本网格、训练 CSV 及训练/评测日志。训练 checkpoint 如需发布，先用
[`export_checkpoint.py`](export_checkpoint.py) 移除优化器与 RNG 状态，再发布推理权重。

## 采样与评测

`sample.py --nfe` 参数表示积分步数。无条件采样示例：

```bash
python projects/project4_flow_matching/sample.py \
  --model checkpoints/project4-v2-unconditional-step200000.pt \
  --condition unconditional --solver euler --nfe 50 \
  --n_samples 64 --seed 42 --output samples/fm_v2_unconditional
```

FM-C 的类别 0 CFG 示例：

```bash
python projects/project4_flow_matching/sample.py \
  --model checkpoints/project4-v2-conditional-step200000.pt \
  --condition cfg --solver euler --nfe 20 --cfg 2 \
  --class_label 0 --n_samples 64 --seed 42 \
  --output samples/fm_v2_conditional
```

复现 FM-U 的 Euler / Heun 扫描和 FM-C 的 CFG 扫描：

```bash
python projects/project4_flow_matching/eval_fid.py \
  --model checkpoints/project4-v2-unconditional-step200000.pt \
  --condition unconditional --solver euler heun --nfe 4 8 16 32 50 \
  --num_samples 5000 --batch_size 64 --seeds 42 \
  --output projects/project4_flow_matching/results/fm_v2_unconditional_solvers.json

python projects/project4_flow_matching/eval_fid.py \
  --model checkpoints/project4-v2-conditional-step200000.pt \
  --condition cfg --solver euler --nfe 20 --cfg 1 2 3 5 7.5 \
  --num_samples 5000 --batch_size 64 --seeds 42 \
  --output projects/project4_flow_matching/results/fm_v2_conditional_cfg.json
```

评测使用 CIFAR-10 train 前 5,000 张真实图、5,000 张生成图、TorchMetrics
`FrechetInceptionDistance(feature=2048, normalize=False)`、batch size 64、seed 42 和
EMA 0.9999。CFG 将无条件与条件输入合并为一次 batch 前向；JSON 中的
`batched_model_calls` 记录批量调用数，`forward_evaluations_per_sample` 计入两路预测的实际成本。
要估计种子波动，可将 `--seeds 42` 改为多个 seed，并报告分布而非单次值。

## v2 实测结果

### FM-U 与 Project 1 DDIM

| 采样器 | 积分步数 | 实际 NFE/图 | FID ↓ |
|---|---:|---:|---:|
| FM Euler | 4 | 4 | 80.5531 |
| FM Euler | 8 | 8 | 43.2211 |
| FM Euler | 16 | 16 | 32.1931 |
| FM Euler | 32 | 32 | 28.0684 |
| FM Euler | 50 | 50 | 26.7610 |
| FM Heun | 4 | 8 | 154.8813 |
| FM Heun | 8 | 16 | 82.1227 |
| FM Heun | 16 | 32 | 43.2588 |
| FM Heun | 32 | 64 | 26.4840 |
| FM Heun | 50 | 100 | 24.0062 |
| Project 1 R5 + DDIM | 10 | 10 | 53.0907 |
| Project 1 R5 + DDIM | 20 | 20 | 33.6099 |
| Project 1 R5 + DDIM | 50 | 50 | 23.1206 |
| Project 1 R5 + DDIM | 100 | 100 | 20.1105 |
| Project 1 R5 + DDIM | 250 | 250 | 18.3625 |

![FM 与 Project 1 DDIM 的 NFE–FID 曲线](results/nfe_fid_curve_v2.png)

FM Euler 在 NFE 8、16 的 FID 分别低于 DDIM NFE 10、20 的测点；这两组计算预算并未对齐。
预算对齐的 NFE 50 结果中 DDIM 优于 FM Euler；NFE 100 时 DDIM 也优于已测的 FM Heun。
因此，本组数据不支持“FM 在低 NFE 下总体优于 DDIM”的结论。Heun 在 NFE 8、16 的 FID
较高，也说明结果对求解器和积分预算敏感。

两边评测协议较一致，但模型与训练过程不同：FM 使用 32.62M 参数 DiT-S，DDIM 使用
Project 1 R5 U-Net；训练架构与预算并未控制一致。每项只有 seed 42 的一次测量，没有种子
方差估计，不能将局部差异归因于采样方法或视为稳定结论。更完整的评测记录在
[`Project 2 的基线 JSON`](../project2_samplers/results/p1_r5_ddim_5k.json)。

### FM-C 的 CFG 扫描

固定 Euler 20 个积分步（每图 40 次实际网络评估）：

| CFG scale | 实际 NFE/图 | FID ↓ |
|---:|---:|---:|
| 1.0 | 40 | 23.5773 |
| 2.0 | 40 | **18.8105** |
| 3.0 | 40 | 23.6983 |
| 5.0 | 40 | 35.2425 |
| 7.5 | 40 | 44.9110 |

本次测量中 scale 2.0 的 FID 最低；scale 从 3.0 起 FID 变差。单 seed 的 FID 不代表
跨 seed 的稳定性。

![FM-C 类别条件样本网格](samples/fm_v2_conditional/class_0/euler_nfe20_cfg_cfg3.0_EMA_0.9999.png)

## 代码与结果文件说明

| 路径 | 内容 |
|---|---|
| [`train.py`](train.py) | 训练、EMA、checkpoint 保存与恢复 |
| [`sample.py`](sample.py) | Euler / Heun 图像采样与 CFG |
| [`eval_fid.py`](eval_fid.py) | FID 与 NFE 扫描，输出评测 JSON |
| [`plot_nfe_curve.py`](plot_nfe_curve.py) | 由评测结果绘制 NFE–FID 曲线 |
| [`export_checkpoint.py`](export_checkpoint.py) | 将续训 checkpoint 导出为推理权重 |
| [`model/dit.py`](model/dit.py)、[`model/unet.py`](model/unet.py) | DiT 与简化 conditional U-Net |
| [`results/fm_v2_unconditional_solvers.json`](results/fm_v2_unconditional_solvers.json) | FM-U Euler / Heun 的原始 FID 结果 |
| [`results/fm_v2_conditional_cfg.json`](results/fm_v2_conditional_cfg.json) | FM-C CFG 扫描的原始 FID 结果 |
| [`results/nfe_fid_curve_v2.png`](results/nfe_fid_curve_v2.png) | FM 与 Project 1 DDIM 的 NFE–FID 曲线 |
| [`repro_manifest_v2.json`](repro_manifest_v2.json) | 软硬件、配置、数据、权重哈希和评测协议 |

