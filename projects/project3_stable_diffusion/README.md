# Project 3：Stable Diffusion 实验报告

> 本文将本地 README 与项目仓库的实验报告合并为一个可读、可审计的报告入口。数据口径以已提交的 AutoDL 产物、JSON/JSONL metadata 和图像为准；实验图像链接固定到提交 `6b14f26cd7ea2a67d3b231a97cde80baea636ca2`，便于核对对应快照。

> 图片展示：图像直接引用上述提交中的已提交实验 PNG，不依赖额外的 `README_assets/` 目录；并列图按等宽显示，不改变原图比例。

## 1. 摘要与完成度

本项目以 Stable Diffusion 1.5 为对象，逐段复现并检查文本到图像的 latent diffusion 路径，完成手写推理、参数扫描、VAE 解剖、LoRA 微调、Canny ControlNet、cross-attention 可视化，以及 LDM/SDXL 阅读笔记。

| 原始作业模块 | 结果 | 主要证据 |
|---|---|---|
| A：不调用高层 pipeline 的手写推理 | 已完成并保留 notebook 输出 | 01_inference_walkthrough.ipynb、outputs/manual/ |
| B：CFG、steps、sampler 对比 | 已完成 full preset，seed 42 | outputs/sweep/、metadata.json |
| C：VAE latent 通道与重构 | 已执行，含四通道图、crop、MSE/PSNR | 05_vae_anatomy.ipynb、outputs/vae_*.png、vae_metrics.json |
| D：LoRA 微调、分段保存、独立加载 | AutoDL RTX 4090 上完成 800 optimizer steps | outputs/lora/full/、outputs/lora/full_eval/ |
| E：Canny ControlNet 与 attention challenge | 四种风格、结构/文本冲突、四个 token attention map | outputs/controlnet/、outputs/attention/full/ |
| LDM §3–4 与 SDXL 阅读笔记 | 已完成 | notes/reading_note_ldm.md、reading_notes/ldm_sdxl_reading_note.md |
| 环境、复现与 debug | 有环境快照、运行日志、SHA256 和真实问题记录 | outputs/autodl_environment.json、logs/、本报告第 10 节 |

**完成判断：**按功能、执行输出和实验图表，Project 3 的 A–E 与阅读笔记均已完成。复现时需特别留意三项边界：AutoDL 使用 ModelScope 模型镜像；LoRA 数据使用 ModelScope 镜像而非原计划的 Wikimedia Commons 原图；部分 AutoDL metadata 的 git_commit 为 null，因此当前能确认产物和运行参数，但不能从 metadata 单独证明每个运行进程对应哪个精确代码提交。

### 数据口径校正

合并过程中对照了本地 README、GitHub report、PNG 文件清单和 JSON metadata，并统一了以下表述：

1. B 的 metadata 实际列出 **16 个 PNG 文件**，它们由 12 张单样本结果图（5 个 CFG、4 个 steps、3 个 sampler）、1 张 3×3 CFG×steps 网格、3 张汇总图组成；不是“16 张单图再加 4 张汇总图”。二维网格本身含 9 个生成样本格。
2. C 的 AutoDL VAE 指标使用 outputs/sweep_smoke/grid_2d.png，MSE 为 0.00197918、PSNR 为 27.0351 dB。本机早期 smoke 的另一组数字不与这组 AutoDL 结果混算。
3. LoRA 数据来源按 manifest 记为 ModelScope huggan/vangogh2photo；由于 AutoDL 访问 Wikimedia Commons 返回网络错误，不能把这 20 张镜像图描述成 Commons 原图或逐幅公有领域作品。
4. 代码默认模型 revision 与 AutoDL 实际下载源不同：代码默认固定 Hugging Face revision；AutoDL 使用 ModelScope 的 master 镜像，并以实际权重 SHA256 辅助识别。本报告会区分二者。

## 2. 环境、模型与数据

### AutoDL 实测环境

运行环境为 AutoDL Linux、Python 3.12.3、NVIDIA GeForce RTX 4090，显存 24 GB（metadata 记录 25,252,724,736 bytes），CUDA 12.8。依赖基线如下：

| 依赖 | 版本 |
|---|---|
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| Diffusers | 0.40.0 |
| Transformers | 5.15.1 |
| PEFT | 0.20.0 |
| Accelerate | 1.14.0 |
| Hugging Face Hub | 1.31.0 |
| safetensors | 0.8.0 |

requirements/project3.txt 是仓库中记录的安装基线；本机 smoke 的 torch 是 CUDA 12.6 build，不替代上述 AutoDL 实测。完整环境、时间戳、设备、模型来源和关键权重摘要见 [autodl_environment.json](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/autodl_environment.json)。

### 基础模型与权重指纹

代码默认模型为 stable-diffusion-v1-5/stable-diffusion-v1-5，固定 revision 451f4fe16113bff5a5d2269ed5ad43b0592e9a14。但 AutoDL 当时无法访问 Hugging Face endpoint，实际使用 ModelScope 镜像 AI-ModelScope/stable-diffusion-v1-5@master；ControlNet 使用 lllyasviel/sd-controlnet-canny 的 ModelScope 镜像。

AutoDL 环境文件记录的关键权重如下：

| 组件 | 大小（bytes） | SHA256 |
|---|---:|---|
| SD 1.5 UNet | 3,438,167,540 | 19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1 |
| SD 1.5 VAE | 334,643,276 | a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c |
| SD 1.5 text encoder | 492,265,874 | d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334 |
| Canny ControlNet | 1,445,157,124 | e19821a00e6d1817b37286a21d5c4f8915076949b0e81846c4f92c96ffb46db7 |

镜像标签 master 本身不是不可变 revision；因此复现实验时应同时核对上表 SHA256，而不能只记录仓库名和标签。

### LoRA 训练数据

AutoDL 从公开 ModelScope 镜像 huggan/vangogh2photo 的 imageA 抽取 20 张图片。manifest 中记录了来源 URL、镜像 metadata 中标注的 Apache-2.0 许可、每张图片尺寸（原始样本为 256×256）与 SHA256；训练预处理再调整到 512×512。原始图像留在 AutoDL 被忽略目录 .local/datasets/project3_vangogh，没有提交到 Git。

原计划的 Wikimedia Commons 下载器仍保留在代码中，但 AutoDL 请求 Commons API 时遇到 OSError: [Errno 99] Cannot assign requested address。因此实际训练集必须准确称为 ModelScope 镜像数据，许可描述也限于该镜像的 metadata。逐图清单见 [vangogh_manifest.json](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/vangogh_manifest.json)。

## 3. Stable Diffusion 端到端结构

文本到图像的主要张量流：

**prompt → tokenizer → CLIP text encoder → text embedding；随机 latent → UNet + CFG + scheduler → 去噪 latent → VAE decoder → RGB 图像。**

对于 SD 1.5、512×512 输入/输出：

| 阶段 | 张量 shape | 来源/含义 |
|---|---|---|
| token ids | (1, 77) | CLIP 固定长度 padding/truncation |
| text embedding | (1, 77, 768) | CLIP 序列长度 77、hidden size 768 |
| 初始 latent | (1, 4, 64, 64) | 图像经 VAE 下采样 8 倍；4 为 latent channels |
| CFG UNet 输入 embedding | (2, 77, 768) | unconditional 与 conditional embedding 沿 batch 拼接 |
| 去噪 latent | (1, 4, 64, 64) | UNet 预测与 scheduler 更新后的 latent |
| 解码 RGB | (1, 3, 512, 512) | 解码前按 VAE scaling factor 0.18215 还原 latent 尺度 |

Classifier-Free Guidance 的预测组合为：

**ε = ε_uncond + guidance_scale × (ε_cond − ε_uncond)**

其中 UNet 单次前向先处理拼接 batch，再拆分 unconditional/conditional 预测。0.18215 是 SD 1.5 VAE 的 latent scaling factor，解码时必须撤销该缩放。A/C notebook 中的中间值和 shape 输出可用于逐步核对这些接口。

## 4. A：手写 Stable Diffusion 推理

文件：01_inference_walkthrough.ipynb。该 notebook 逐步运行 tokenizer、CLIP、噪声初始化、DDIM、CFG、scheduler 更新和 VAE decode，不以 StableDiffusionPipeline(prompt) 取代手写推理。它保留了 AutoDL 执行计数、文本和图像输出。

固定配置：prompt 为 astronaut/horse/mars 场景，negative prompt 为 low quality, blurry, distorted，seed=42，50 个 DDIM steps，CFG=7.5。中间 latent 仍为 (1,4,64,64)，最终图像为 (1,3,512,512)。

### 采样过程中的 latent 统计

| 循环 step | scheduler timestep | mean | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0 | 981 | 0.00173 | 0.99620 | -3.92188 | 3.92578 |
| 9 | 801 | -0.00029 | 0.98591 | -3.81445 | 3.77148 |
| 24 | 501 | -0.00298 | 0.94525 | -3.60352 | 3.93555 |
| 49 | 1 | -0.00781 | 0.80868 | -4.90625 | 5.94922 |

标准差总体下降，表示 latent 的整体噪声尺度随去噪过程减小；但 min/max 不要求单调收缩，UNet 预测和 scheduler 更新会改变每个空间位置的值。反向扩散因此不是把 forward 加噪公式逐项反演。

产物：[手写推理生成图](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_seed42.png)、[latent 统计 JSON](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_stats.json)。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_seed42.png" alt="手写 Stable Diffusion 推理结果" width="960"></p>
<p align="center"><strong>图 1｜手写推理生成结果</strong><br><sub>seed=42、50 DDIM steps、CFG=7.5；图像由 notebook 显式执行 tokenizer、CLIP、UNet、CFG、scheduler 和 VAE decode。</sub></p>

原始图与统计：[manual_sd_seed42.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_seed42.png) · [manual_sd_stats.json](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_stats.json)

### A 的思考题结论

1. CFG=0 时公式退化为 unconditional 预测，prompt 条件方向不参与结果。
2. 只用 conditional embedding 仍是条件生成，但失去 unconditional-to-conditional 的 CFG 外推。
3. 把 init_noise_sigma 错设成 0.5 会让初始 latent 尺度不符合 scheduler 预期，可能影响亮度、对比度和细节；应读取 scheduler 的值。
4. 反向过程噪声尺度总体下降，但包含 UNet 预测、CFG 和 scheduler 参数化，不是 forward 的简单逐项逆映射。
5. 仅 5 steps 时离散采样校正机会少，通常结构/纹理更不稳定；增加 steps 有帮助，但计算成本增加且收益会递减。

## 5. B：CFG、steps 与 sampler 扫描

脚本：02_parameter_sweep.py。full preset 在同一 prompt、negative prompt 和 seed=42 下改变一个因素或绘制二维交互图。prompt 为 “a photograph of a cat wearing a wizard hat, fantasy art”，negative prompt 为 “low quality, blurry”；输出尺寸 512×512。

| 对比 | 扫描配置 | 控制变量/图 |
|---|---|---|
| CFG scale | 1、3、7.5、15、25 | 其余配置固定；sweep_cfg.png |
| inference steps | 10、20、50、100 | CFG=7.5；sweep_steps.png |
| sampler | DDIM、Euler-A、DPM++ 2M | 30 steps、CFG=7.5；sweep_sampler.png |
| CFG × steps | CFG=3、7.5、15 × steps=10、20、50 | 3×3 grid_2d.png |

full sweep 用时 **36.642 秒**。同 seed 用于条件间对比，输出按 metadata 记录 16 个 PNG 文件：12 张独立条件结果、1 张含 9 个样本格的二维网格、3 张横向汇总图。参数扫描 metadata 同时保存 prompt、seed、依赖版本、运行时间、文件清单和每张图的 SHA256。

### 实测图与观察

CFG 与 steps 的单因素对照、sampler 比较和 CFG×steps 交互网格分别排成独立图版；同一组卡片等宽、等高，便于横向比较。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_cfg.png" alt="CFG scale 五档对照图" width="960"></p>
<p align="center"><strong>图 2a｜CFG scale 对照</strong><br><sub>固定 prompt、seed=42 与其余设置，比较 CFG=1、3、7.5、15、25。</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_steps.png" alt="采样步数四档对照图" width="960"></p>
<p align="center"><strong>图 2b｜采样步数对照</strong><br><sub>固定 CFG=7.5，比较 10、20、50、100 steps。</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_sampler.png" alt="DDIM、Euler-A、DPM++ 2M 对照图" width="960"></p>
<p align="center"><strong>图 2c｜Sampler 对照</strong><br><sub>固定 30 steps 与 CFG=7.5；sampler 更新方式不同，因此同 seed 不保证像素一致。</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/grid_2d.png" alt="CFG 与 steps 的三乘三交互网格" width="960"></p>
<p align="center"><strong>图 2d｜CFG × steps 交互</strong><br><sub>行列分别对应 CFG=3、7.5、15 与 steps=10、20、50；这是子网格，不是完整 5×4 组合。</sub></p>

CFG 从 1 增至 3/7.5 后，猫和巫师帽的语义更明确；到 15/25 时颜色和边缘更强烈，出现过饱和倾向。steps 从 10 到 20 可见结构明显成形，50/100 继续细化但增益较小。以上是单一 prompt/seed 的定性观察，不是跨数据集质量评测。

原始高清图：[sweep_cfg.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_cfg.png) · [sweep_steps.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_steps.png) · [sweep_sampler.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_sampler.png) · [grid_2d.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/grid_2d.png)

程序支持 --preset smoke|full、--model_revision 与 --metadata_output。full AutoDL 运行的参数与产物清单见 [sweep metadata](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/metadata.json)。

## 6. C：VAE anatomy

文件：05_vae_anatomy.ipynb。使用 encoder posterior 的 mode（确定性重构，而非随机采样），记录四个 latent channel、原图/重构图、局部细节 crop 和 metrics JSON。本次 AutoDL 输入为 outputs/sweep_smoke/grid_2d.png；指标张量取值范围归一化至 [0,1]。

| 张量 | shape |
|---|---|
| 输入 RGB | (1, 3, 512, 512) |
| latent | (1, 4, 64, 64) |
| 重构 RGB | (1, 3, 512, 512) |

**实测 MSE = 0.0019791808，PSNR = 27.0351 dB。**四个通道统计如下：

| latent channel | mean | std |
|---:|---:|---:|
| 1 | 0.47984 | 1.11195 |
| 2 | -0.01039 | 1.36113 |
| 3 | 0.14115 | 0.59798 |
| 4 | -0.17118 | 0.73707 |

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_channels.png" alt="VAE 四个 latent 通道" width="230"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_reconstruction.png" alt="VAE 输入与重构" width="230"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_detail_crop.png" alt="VAE 局部细节" width="230"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/report/vae_statistics.png" alt="VAE 通道统计与重构指标" width="230"></p>
<p align="center"><strong>图 3｜VAE anatomy：通道、重构与指标</strong><br><sub>A：四个 latent channel；B：输入/重构；C：局部细节；D：通道统计和 MSE/PSNR。</sub></p>

原始图：[vae_channels.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_channels.png) · [vae_reconstruction.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_reconstruction.png) · [vae_detail_crop.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_detail_crop.png) · [vae_statistics.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/report/vae_statistics.png)

**结果解释：**低频色彩和整体形状较容易保留，细小文字、锐利边缘和高频纹理会平滑；VAE 是降低空间计算成本的压缩环节，也构成最终细节的信息瓶颈。当前指标只针对一张由 smoke 生成的合成网格图，不能当作在大规模自然图像验证集上的 VAE 通用质量结论。本机早期 smoke 的 MSE/PSNR 是另一运行环境和输入，不纳入 AutoDL 结果表。

## 7. D：LoRA 风格微调与独立加载

文件：03_lora_finetune.py、evaluate_lora.py。

### 训练实现与配置

- 冻结 VAE、text encoder 和基础 UNet，仅在 UNet attention 的 to_q、to_k、to_v、to_out.0 注入 LoRA。
- rank=8、alpha=8；LoRA master 参数保持 FP32，前向采用 FP16 autocast/GradScaler。
- 输入训练分辨率 512×512，batch size=1，AdamW，learning rate=1e-4，800 optimizer steps，seed=42。
- 开启 gradient checkpointing，max_grad_norm=1.0；每 200 步保存 adapter checkpoint、验证图和 JSONL 训练记录。
- 数据集共 20 张镜像图，训练 instance prompt 为 “a painting in the style of van gogh”。

训练耗时 **209.47 秒**。loss 从初始 **0.2810367** 到最后一步 **0.1527308**；最后 50 步均值 **0.2282097**。单步 loss 在 batch=1 且 timestep/噪声随机时波动很大，所以用滑动均值和固定 prompt/seed 的图像一起判断。800 步训练没有 NaN/Inf；adapter 文件大小 **6,414,448 bytes（约 6.41 MB）**。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/report/lora_loss_curve.png" alt="LoRA 训练 loss 与检查点" width="960"></p>
<p align="center"><strong>图 4a｜LoRA 训练损失</strong><br><sub>单步噪声预测 MSE、50-step 均值，以及 200/400/600/800-step checkpoints。</sub></p>

原始图：[lora_loss_curve.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/report/lora_loss_curve.png)

均值曲线有起伏而非逐步单调下降；损失下降是优化信号，不等于风格泛化的定量指标。

### 独立加载评估

evaluate_lora.py 在全新 SD pipeline 中评估 base、checkpoint-0200、0400、0600、0800。比较使用 seed=42、20 inference steps、CFG=7.5，四个 prompt 是 lighthouse by the sea、quiet street at night、vase of sunflowers、mountain landscape，均使用 sks style token；因此得到 5 个模型状态 × 4 个固定 prompt 的 20 张对比图。另有训练过程使用的两条 validation prompts：landscape 与 vase of sunflowers。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/full_eval/lora_before_after.png" alt="LoRA 四组 prompt 与五个模型状态对比" width="960"></p>
<p align="center"><strong>图 4b｜LoRA adapter 独立加载对照</strong><br><sub>列为 base、200、400、600、800 steps；行对应 lighthouse、street、sunflowers、mountain。</sub></p>

原始图：[lora_before_after.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/full_eval/lora_before_after.png)

这组对比证明 adapter 能从保存文件重新加载且输出与 base 不同；风格变化应结合 200/400/600/800 checkpoint 图观察。它是小数据、小样本的定性对比，不代表对任意 prompt 的稳定风格迁移。

**重载实现注意：**本次保存的是 UNet-only safetensors。直接调用 pipeline adapter loader 可能因 keys 缺少 unet. 前缀而静默忽略权重；正确方式是通过 unet.load_lora_adapter(..., prefix=None, weight_name=...) 明确加载。该问题曾导致 adapter 输出与 base 的 SHA256 完全相同，修复后重新加载评估图有可解释变化。详细配置和 summary 在 [training_summary.json](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/full/training_summary.json) 与 [evaluation metadata](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/full_eval/metadata.json)。

## 8. E：Canny ControlNet 与 cross-attention

### 8.1 Canny ControlNet

文件：04_controlnet_demo.ipynb。输入图先用 Canny 提取边缘（thresholds=[100, 200]），然后固定结构，使用四种 prompt 做重绘；另加要求 “a realistic golden retriever dog, high detail” 的冲突 prompt，观察文本与边缘条件争夺控制权。生成 seed=42。

四种风格 prompt 为现代油画、漫画、赛博朋克角色、Pixar 3D 渲染，分别保存结果并组合成网格：

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/control_input_and_canny.png" alt="ControlNet 输入与 Canny 边缘" width="300"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/controlnet_grid.png" alt="ControlNet 四种风格结果" width="300"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/controlnet_conflict_dog_on_same_edges.png" alt="ControlNet 结构与文本冲突实验" width="300"></p>
<p align="center"><strong>图 5a｜Canny ControlNet 条件实验</strong><br><sub>A：输入/Canny；B：四种风格；C：结构与 golden retriever 文本冲突；统一 seed=42。</sub></p>

原始图：[control_input_and_canny.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/control_input_and_canny.png) · [controlnet_grid.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/controlnet_grid.png) · [controlnet_conflict_dog_on_same_edges.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/controlnet_conflict_dog_on_same_edges.png)

metadata 中的逐图 SHA256 及 prompt/阈值见 [controlnet metadata](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/metadata.json)。原理上，ControlNet 从冻结的 SD 主干复制条件分支，并用 zero-convolution 注入 residual；零初始化令训练开始时条件 residual 为零，减少对原模型输出的突变。训练后条件分支逐步学会通过边缘结构调节生成内容。冲突试验的意义是观察结构条件和文本语义的权衡，不预设其中某一条件总会胜出。

### 8.2 Cross-attention challenge

脚本：06_cross_attention_visualization.py。prompt 为 “a red cat wearing a blue wizard hat in a green forest”，seed=42、30 steps、CFG=7.5。自定义 attention processor 在 CFG 条件分支采集 prompt 相关 attention，在去噪中后段聚合 16×16 和 32×32 query map，并排除 BOS、EOS 及 padding token。有效 token 为 cat、wizard、hat、forest，token indices 分别是 3、7、8、12；metadata 记录了 32 个 attention 模块名、两种空间分辨率、token IDs 与各输出 SHA256。本次记录耗时 9.031 秒。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/generated.png" alt="Cross-attention 生成图" width="180"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_03_cat_w.png" alt="cat token attention map" width="180"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_07_wizard_w.png" alt="wizard token attention map" width="180"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_08_hat_w.png" alt="hat token attention map" width="180"> <img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_12_forest_w.png" alt="forest token attention map" width="180"></p>
<p align="center"><strong>图 5b｜Cross-attention token 空间响应</strong><br><sub>生成图与四个有效 token map；热图是模型响应可视化，并非像素级分割真值。</sub></p>

原始图：[generated.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/generated.png) · [attention_03_cat_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_03_cat_w.png) · [attention_07_wizard_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_07_wizard_w.png) · [attention_08_hat_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_08_hat_w.png) · [attention_12_forest_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_12_forest_w.png)

热力图是 token 到空间位置响应的可视化，不是像素级语义分割真值；低分辨率 query map 经插值叠加后只能解释模型注意区域的大致位置。相关原始结果与采集配置见 [attention metadata](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/metadata.json)。

## 9. LDM 与 SDXL 阅读笔记

### LDM §3–4

Latent Diffusion Models 先训练感知压缩 autoencoder：encoder E 将图像 x 编为 z=E(x)，decoder D 使 x≈D(z)。扩散过程在较低维 z 空间中进行，而不是直接对高分辨率像素做 UNet 计算。前向扩散和噪声预测目标可概括为：

- z_t = sqrt(alpha_bar_t) z_0 + sqrt(1 - alpha_bar_t) epsilon
- L_simple = E[ || epsilon - epsilon_theta(z_t, t, c) ||² ]

文本条件通过 cross-attention 注入：Q 来自当前 latent feature，K/V 来自文本 token embedding，softmax(QKᵀ/sqrt(d))V 把条件语义写入空间特征。CFG 则将 unconditional 与 conditional 预测做线性外推。A 的 shape 跟踪、C 的 VAE 重构和 E 的 attention map 分别对应这些组成机制。

优点是大幅降低空间计算和显存；代价是 VAE 压缩误差、scaling factor、解码能力和文本 tokenizer 都会影响最终细节。

### SDXL 前半部分

SDXL 不只是加宽 SD 1.5 UNet：它使用两个 CLIP text encoder，组合不同 embedding 空间，并加入 pooled text embedding、原始图像尺寸、裁剪坐标与目标尺寸等 additional conditioning。Base 负责主体构图，refiner 在低噪声阶段侧重恢复高频纹理；相应代价是更高显存、更复杂的双文本编码和严格的尺寸 metadata。

与本项目的联系：D 的 LoRA 在固定 VAE/text interface 上学习 UNet attention 的低秩偏移；E 显示 QKᵀ 中 token 对空间 query 的响应；迁移 SDXL 时不能只替换 UNet，还须同步处理双文本编码器与额外条件输入。

## 10. AutoDL 执行、复现与真实 debug

### 执行顺序与验收记录

AutoDL smoke 阶段通过 Project 3 pytest（**5 passed**）、5-step 手写推理/扫描 smoke、2-step LoRA 保存与重载、单 prompt ControlNet、单 token attention map；随后完成 full sweep、A/C/E 三个 notebook 的 execute 输出、800-step LoRA、四风格加冲突 ControlNet、30-step cross-attention。耗时可核验值包括 full sweep 36.642 秒、LoRA 209.47 秒、LoRA 独立评估 37.549 秒、full attention 9.031 秒。日志、JSONL、环境快照和图像 hashes 均在仓库 outputs/ 与 logs/ 下。

| 验收点 | 本次记录 |
|---|---|
| 手写推理与 notebook 输出 | 记录 seed/steps/latent 统计并生成 512×512 图 |
| 扫描范围与可视化 | 五档 CFG、四档 steps、三种 sampler；含网格、汇总图和 metadata |
| VAE | 输入/latent/重构 shape、四通道统计及 MSE/PSNR |
| LoRA | 800 steps、loss 有限、20 张数据、adapter 小于 25 MB、独立重载 |
| ControlNet / attention | 四种风格 + 冲突图；有效 token heatmap 与 16/32 map |
| 仓库卫生 | 基础模型权重、HF cache、原图和临时训练 checkpoint 不入 Git |

运行产物 metadata 中 git_commit 为 null，因此它们未携带一个可直接验证的运行时源代码提交号；GitHub 这份统一文稿及对应输出采用已读取快照 main@6b14f26。若未来重跑，建议把 git_commit 和不可变模型 revision 写入每一份 metadata。

### 已记录的真实问题与修复

1. 旧 huggingface_hub 与 Diffusers 0.40 的 cached_download 接口不兼容，导入失败；升级到锁定环境组合中的 huggingface_hub 1.31.0 后恢复。
2. Windows 工作站仓库输出目录 ACL 对普通 Python 进程只读，运行产物改放仓库外可写路径；AutoDL 则按计划使用持久目录及仓库 outputs。
3. LoRA 初次评估图与 base SHA256 相同；原因是 pipeline loader 忽略缺少 unet. 前缀的 UNet-only adapter。改用 UNet loader 并传 prefix=None 后通过图像变化验证修复。
4. cross-attention 初版在 norm_cross=None 的模块调用 norm_encoder_hidden_states，触发 assertion；仅在 norm_cross 为真时做归一化。
5. tokenizer token 名中含尖括号，在 Windows 上直接做文件名失败；现用安全字符过滤 label，同时保留 token index。
6. AutoDL 访问 Wikimedia Commons 出现 OSError 99；于是使用 ModelScope 的 20 张 huggan/vangogh2photo imageA 数据，记录来源、镜像 metadata 许可说明和 SHA256，未把镜像伪称为 Commons 原图。
7. AutoDL 网络不可达导致 check_env 的远端 HEAD 检查不能单独作为成功依据；最终以镜像权重 hash、实际运行输出和实验 metadata 验收。

## 11. 自查问题与结论

1. **CFG=0 会怎样？** 只使用 unconditional 预测，prompt 的条件方向不被 CFG 采用。
2. **只用 conditional embedding 会怎样？** 仍可条件采样，但不再有 unconditional-to-conditional 外推，prompt adherence 往往不同。
3. **init_noise_sigma 误设为 0.5 会怎样？** 初始 latent 尺度不匹配 scheduler 的预期，可能造成亮度、对比度或细节异常；应使用 scheduler 给定值。
4. **反向 latent 是否是 forward process 的简单逆过程？** 否。噪声整体下降，但 UNet、CFG 和 scheduler 参数化都会影响更新。
5. **为什么 5 steps 往往质量较差？** 采样校正次数太少，离散化误差更明显；更多 steps 有帮助但收益递减。
6. **ControlNet 为什么有条件分支？** 冻结的基础模型保留原有能力，条件分支学习结构 residual，而不是从头破坏主体生成网络。
7. **zero-convolution 为什么重要？** 其零初始化令 ControlNet 初始 residual 为零，让训练初期接近原模型；之后权重学习到结构控制。
8. **结构条件和 prompt 冲突时如何理解？** ControlNet 提供空间结构，文本提供语义/风格，最终由控制强度、CFG 和采样共同决定折中；冲突图用于观察实际取舍。
9. **ControlNet 增加什么成本？** 条件分支的特征提取和 residual 计算会增加显存和推理时间，尽管 base 权重可冻结复用。
10. **CFG 为什么常用 7.5？** 它是 adherence 与自然度/伪影的经验折中，不是理论常数；最合适范围依 prompt、模型和 scheduler 而变。
11. **为什么 SD 1.5 常见 512×512？** VAE 将图像压缩至 64×64 latent，成本可控；分辨率翻倍会让 latent 位置数量约增至四倍。
12. **为什么扩散模型常画不好文字？** CLIP/tokenizer 更偏词级语义而非字符排版；训练目标、数据对齐与 VAE 对细线条的平滑共同限制文字准确度。

**总的实验结论：**手写路径验证了 SD 1.5 从 CLIP 条件到 latent 去噪再到 VAE 解码的接口；参数扫描展示了 guidance 强度、采样步数、sampler 选择之间的可视化取舍；VAE 量化揭示了压缩/细节之间的折衷；LoRA 能以小型 adapter 产生可重载的风格变化；ControlNet 在固定边缘上改变风格，attention heatmap 则展示文本 token 对生成空间的软性响应。所有结论的适用范围是本项目所记录的模型、数据、单组或少量 prompt 与 seed，不外推为 FID 等大规模统计质量结论。

## 12. 代码与产物索引

- A：01_inference_walkthrough.ipynb、outputs/manual/
- B：02_parameter_sweep.py、outputs/sweep/、outputs/sweep_smoke/
- C：05_vae_anatomy.ipynb、outputs/vae_*.png、outputs/vae_metrics.json
- D：03_lora_finetune.py、evaluate_lora.py、outputs/lora/full/、outputs/lora/full_eval/
- E：04_controlnet_demo.ipynb、06_cross_attention_visualization.py、outputs/controlnet/、outputs/attention/
- 环境与日志：outputs/autodl_environment.json、logs/
- 阅读笔记：notes/、reading_notes/
- 报告图生成：make_report_figures.py；仓库中的汇总图保存在 outputs/report/，其他内嵌图直接引用对应的已提交实验 PNG。

基础 SD/ControlNet 权重、Hugging Face cache、20 张训练原图和临时训练 checkpoint 不应放入 Git。完整文件结构请以项目目录为准。

## 13. 参考资料

- [Project 3 原始作业要求](https://github.com/Qi-StarterTrain/diffusion-project3-stable-diffusion)
- [本报告引用的项目代码与实验产物快照](https://github.com/li-cheng111/my-diffusion-models-starter/tree/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion)
- [Stable Diffusion 1.5 模型仓库](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Latent Diffusion Models 论文](https://arxiv.org/abs/2112.10752)
- [SDXL 论文](https://arxiv.org/abs/2307.01952)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [ControlNet 论文](https://arxiv.org/abs/2302.05543)
- [Diffusers LoRA 训练文档](https://huggingface.co/docs/diffusers/en/training/lora)
