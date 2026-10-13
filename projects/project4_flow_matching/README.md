# Project 4：CIFAR-10 Flow Matching 实验报告

本项目在 CIFAR-10 上训练 Rectified Flow（RF）图像生成模型，实现 Euler、Heun 和
Classifier-Free Guidance（CFG）采样，并用 FID 与 DDPM/DDIM 基线比较。README 同时记录了早期 v2 探索实验和后续 fair-v1 单种子对照实验；两轮实验使用了不同的模型、训练种子和评测设置，结果不能合并成同一张排名表。

当前主要结论以 fair-v1 为准：在 Project 1 U-Net 上直接替换为 Flow Matching 后，普通 FM-U Euler 并没有在中高 NFE 上稳定超过 DDIM；重流（F-RF）在低 NFE 下明显更好，但到 NFE 50–100 时优势消失；Heun 则在高 NFE 表现较好、低 NFE 较差。A/B/C 三个 AutoDL 节点的 checkpoint、评测 JSON、日志、样图和复现包已发布在 [GitHub Release `project4-fair-v1.0`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0)。实验设置及逐项数据见 [fair-v1 README](fair_v1/README.md) 和[完整结果表](fair_v1/ARTIFACTS.md)。

早期 v2 的 DiT-S、seed 42 结果仍保留在下文作为历史记录，原权重与评测归档位于 [GitHub Release `project4-v2.0`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0)。

## fair-v1 结果速览

fair-v1 用 seed 44，A 节点在相同 Project 1 U-Net 家族上训练 D-U（DDPM）和 F-U（普通 Flow Matching），并评估 F-CT、F-RF；B 节点评估条件 CFG（F-C）和 logit-normal 时间采样（F-LN）；C 节点额外评估 32.62M 参数 DiT-S（F-DiT）。所有评测使用 CIFAR-10 train 的前 5,000 张图、每档 5,000 张生成图、FID-2048、generation seed 1001 和同一固定噪声库。除 F-DiT 外，主比较使用 Project 1 U-Net；F-DiT 的架构不同，应视为进阶实验而非严格的架构控制对照。

下表按**实际网络评估次数（NFE）**对齐，并列出最能说明差异的 20、50、100 NFE 测点。完整 2–100 NFE 曲线及采样耗时见[实验归档表](fair_v1/ARTIFACTS.md)。

| 实际 NFE/图 | DDIM（D-U） | FM-U Euler | FM-U Heun | F-CT Euler | F-RF Euler |
|---:|---:|---:|---:|---:|---:|
| 20 | 19.9765 | 20.8186 | 53.9351 | 21.6280 | **16.6678** |
| 50 | **16.3803** | 17.3398 | 20.1862 | 17.9956 | 16.4426 |
| 100 | 15.4242 | 15.9969 | **14.5048** | 16.6269 | 16.3683 |

这组测点表明结果取决于训练变体和求解器：普通 FM-U Euler 在 NFE 20、50、100 的 FID 均略高于 DDIM；F-RF 在 NFE 2–32 优于 DDIM，但 NFE 50–100 略逊；Heun 到 NFE 100 才优于 DDIM。因而本实验不支持“Flow Matching 普遍在低 NFE 下优于 DDIM”，但显示重流可以改善低预算采样。以上都是单种子实测值，不是统计显著性结论。

## 一、方法原理

线性路径从标准高斯噪声 `epsilon` 插值到数据图像 `x1`：

$$
x_t=(1-t)\epsilon+t x_1,\qquad
u_t=\frac{d x_t}{dt}=x_1-\epsilon,\qquad t\sim U[0,1].
$$

网络以均方误差预测速度场 `u_t`。采样从噪声端 `t=0` 积分到数据端 `t=1`：Euler每步调用一次网络；Heun 先预测再校正，每步调用两次。表格中的 NFE 是每张图的实际网络评估次数，不是积分步数。CFG 使用`v = v_uncond + scale * (v_cond - v_uncond)`；其每步需两路预测，计算成本为普通采样的两倍。

## 二、历史 v2 实验参数配置（seed 42）

### 训练配置

| 参数 | FM-U（无条件） | FM-C（类别条件） |
|---|---|---|
| 配置文件 | [cifar10_fm_v2_unconditional.yaml](configs/cifar10_fm_v2_unconditional.yaml) | [cifar10_fm_v2_conditional.yaml](configs/cifar10_fm_v2_conditional.yaml) |
| 条件 dropout 概率 | 1.0；所有样本使用 null token | 0.1；随机将标签替换为 null token |
| 输出目录 | runs/fm_v2_unconditional | runs/fm_v2_conditional |

| 两组共用参数 | 实际设置 |
|---|---|
| 模型 | DiT-S，32.62M 参数；embed dim 384、depth 12、6 heads、patch size 2 |
| 时间嵌入尺度 | 1000 |
| 优化器 | AdamW；learning rate 2e-4；weight decay 0 |
| Batch size | 64 |
| 训练步数 | 200,000 optimizer steps |
| 随机种子 | 42 |
| 梯度裁剪 | 1.0 |
| EMA decay | 0.999、0.9995、0.9999 |
| 学习率计划 | 前 2,000 步 warmup；从 100,000 步起 cosine decay；最低为初始学习率的 0.1 |
| 日志频率 | 每 100 步 |
| checkpoint 频率 | 每 10,000 步更新 latest.pt；每 50,000 步保存快照 |

这些用时是本次运行记录，不代表其他 GPU 或负载下的固定训练时间。

### 采样配置

| 实验 | 条件模式 | 求解器 | 积分步数 | 实际网络评估次数/图 |
|---|---|---|---|---|
| FM-U NFE 扫描 | 无条件，使用 null token | Euler | 4、8、16、32、50 | 4、8、16、32、50 |
| FM-U NFE 扫描 | 无条件，使用 null token | Heun | 4、8、16、32、50 | 8、16、32、64、100 |
| FM-C CFG 扫描 | 无条件与类别条件预测做 CFG | Euler | 20 | 40 |

### FID 评测配置

| 参数 | 实际设置 |
|---|---|
| 指标 | TorchMetrics FrechetInceptionDistance(feature=2048, normalize=False) |
| 真实图像 | CIFAR-10 train 的前 5,000 张（索引 0–4,999） |
| 生成图像 | 每组 5,000 张 |
| 图像格式 | uint8 RGB |
| Batch size | 64 |
| 评测随机种子 | 42；类别条件评测按 0–9 循环平衡采样 |
| checkpoint 权重 | EMA 0.9999 |
| 原始结果 | [fm_v2_unconditional_solvers.json](results/fm_v2_unconditional_solvers.json)、[fm_v2_conditional_cfg.json](results/fm_v2_conditional_cfg.json) |

## 三、历史 v2 实测结果（DiT-S FM 与 Project 1 R5 U-Net DDIM）

### FM-U 与 DDIM

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

FM Euler 在 NFE 8、16 的结果与 DDIM NFE 10、20 不属于预算对齐比较。预算对齐的 NFE 50 测点中 DDIM 优于 FM Euler，NFE 100 时 DDIM 也优于已测的 FM Heun。因此，本组数据总体上不支持“FM 在低 NFE 下优于 DDIM”的结论。

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

## 四、fair-v1 完整对照实验（seed 44）

fair-v1 的目的，是补上早期 v2 比较中的架构差异：先在 Project 1 的 U-Net 上训练 D-U（DDPM）基线和 F-U（Flow Matching），再在相同 U-Net 家族上测试不同 FM 训练变体与采样器。C 节点的 F-DiT 保留为 DiT-S 进阶实验，不纳入架构匹配结论。

| 节点 | 实验 | 模型/方法 | 训练步数 |
|---|---|---|---:|
| A | D-U | Project 1 U-Net，DDPM | 200,000 |
| A | F-U | Project 1 U-Net，Flow Matching | 200,000 |
| A | F-CT | Project 1 U-Net，Flow Matching 变体 | 50,000 |
| A | F-RF | Project 1 U-Net，Reflow；从 step 10,000 恢复至 50,000 | 50,000 |
| B | F-C | Project 1 U-Net，类别条件 Flow Matching + CFG | 200,000 |
| B | F-LN | Project 1 U-Net，logit-normal 时间采样 | 200,000 |
| C | F-DiT | DiT-S，32.62M 参数，Flow Matching | 200,000 |

D-U 与 F-U 是主方法对照：两者使用相同的 Project 1 U-Net 架构与核心训练设置（batch 128、学习率 2×10⁻⁴、200,000 步、EMA 0.9999、BF16、seed 44），主要区别是 DDPM 与 Flow Matching 的训练目标/概率路径。F-CT 和 F-RF 使用不同训练流程且只训练到 50,000 步；F-RF 还通过教师生成的轨迹进行重流，因此这些变体与 D-U 的差异不能全部归因于“只替换扩散目标”。F-C 是类别条件模型，F-LN 改了时间采样分布，F-DiT 改了架构；它们分别回答进阶问题，不与无条件 D-U/F-U 主对照混为一谈。

以上训练 seed 均为 44。评测采用 TorchMetrics FID-2048，以 CIFAR-10 train 索引 0–4,999 为真实图像；每个测点生成 5,000 张 RGB uint8 图像，batch size 64，generation seed 1001。节点使用相同的固定高斯噪声库（SHA-256：`a81b164722002a7b4649ba8519a1dff2befad6dd015a77aea97e7d28f261793e`）。Flow Matching Euler 和 DDIM 的 NFE 是每张图的网络调用数；Heun 表格也按实际调用数计，因此与 DDIM 对齐的是 NFE 列，而非积分步数。

### 条件生成与训练变体

F-C 固定 Euler 20 个积分步，即每张图 40 次模型调用。FID 在 CFG scale=2 时最低，scale 增大到 3 以上后质量下降：

| CFG scale | 实际 NFE/图 | FID ↓ |
|---:|---:|---:|
| 1 | 40 | 18.5864 |
| 2 | 40 | **18.3478** |
| 3 | 40 | 23.0368 |
| 5 | 40 | 33.3663 |
| 7.5 | 40 | 46.1879 |

F-LN 是无条件 Euler。代表性结果为 NFE 20：21.6078、NFE 50：17.7216、NFE 100：16.2048。它在高 NFE 下接近 F-U Euler，但本次数据没有显示它在所有预算点都更优。

F-DiT 是不同架构的进阶结果：Euler NFE 2/8/20/50/100 的 FID 分别为 294.5080、61.0893、28.9317、31.0744、31.7707；本次最低点出现在 NFE 20，而非最高 NFE。它和 U-Net 结果可以展示架构/训练组合的表现，但不单独用于证明 RF 或某种求解器普遍更好。

### 额外基线与结果边界

- D-U 的完整 1,000 步 ancestral 采样 FID 为 13.7594，采样耗时 1,486.54 秒；这是高计算预算参考，不能与 20 或 100 NFE 视为等成本结果。
- 关闭去噪 clipping 后的 DDIM-20 FID 为 372.2270，说明该基线的 clipping 设定对本次结果影响很大。
- FID 不保证随 NFE 单调下降：例如 F-RF 从 NFE 64 的 16.3228 到 NFE 100 的 16.3683 略有回升。README 保留实测值，不对曲线作单调化处理。
- 每个配置只有 seed 44 的一次实验。共同噪声和一致的 5,000 张样本协议改善了配对比较，但没有跨种子方差，不能据此声称统计显著或推广到所有模型、数据集和算力预算。

逐档 FID、采样秒数、checkpoint 哈希、训练/评测 metadata 与归档说明见 [fair-v1 实验归档表](fair_v1/ARTIFACTS.md)。下载并校验 A/B/C 的 checkpoint 与复现包：

```bash
bash projects/project4_flow_matching/fair_v1/download_artifacts.sh artifacts/project4-fair-v1 all
```

### 结论

在架构匹配的 U-Net 比较中，普通 FM-U Euler 于 NFE 20/50/100 的 FID 分别为 20.8186/17.3398/15.9969，略高于 DDIM 的 19.9765/16.3803/15.4242；这不支持普通 Flow Matching 在中高 NFE 下稳定优于 DDIM。F-RF 在 NFE 2–32 明显更好，但 NFE 50–100 时稍逊；Heun 在低预算下较差，到 NFE 100 才取得本组较好结果。较准确的结论是：采样质量取决于训练目标、重流策略、求解器和 NFE；这次单种子实验看到 F-RF 的低预算收益，但没有证据支持“FM 普遍低步数更优”。

## 五、代码与结果文件说明

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
| [`fair_v1/README.md`](fair_v1/README.md)、[`fair_v1/ARTIFACTS.md`](fair_v1/ARTIFACTS.md) | 最终 fair-v1 的执行说明、A/B/C 完整 FID 表和 Release 清单 |

[GitHub Release project4-v2.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0) 提供历史 v2 的 EMA 0.9999 推理权重和评测归档；checkpoint 不存入 Git 仓库。最终 fair-v1 的 checkpoint 与复现包位于 [project4-fair-v1.0 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0)。

| Release 文件 | 用途 | 大小（bytes） | SHA-256 |
|---|---|---:|---|
| `project4-v2-unconditional-step200000.pt` | 历史 v2 FM-U，step 200,000 | 522,147,167 | `b75cbe3d4f9aeb16c66cb53e4acb03f9c2c6a18421efd945f78475cd2e114571` |
| `project4-v2-conditional-step200000.pt` | 历史 v2 FM-C，step 200,000 | 522,146,099 | `25666b32e914002778825a02f8e14c89f9f5ff081f9dded9f3f0ad6e0283c5aa` |
| `project4_v2_results_only.tar.gz` | 评测 JSON、样本网格、训练 CSV、训练与评测日志 | 654,483 | `e8ea255f9ae63f666b47bbdab30d494307285d9ba30b7276e248ce800bf8802a` |
