# Project 3：Stable Diffusion 完整解剖、实验与微调

本 README 是 Project 3 的统一入口，合并了任务说明、实现方案、AutoDL 实测报告、阅读笔记、实验记录规范和验收清单。原始的 report.md、logs/ 与 reading_notes/ 仍保留，作为逐项证据和可复用模板；提交、复现实验和阅卷优先以本 README 的结论与产物路径为准。

## 1. 项目摘要

本项目不训练一个 Stable Diffusion 基础模型，而是拆解并复现 Stable Diffusion 1.5 的关键路径：

1. 手写 tokenizer、CLIP、latent 初始化、UNet、CFG、scheduler 和 VAE decode。
2. 扫描 CFG、采样步数和 sampler，观察质量、结构和速度的变化。
3. 解剖 VAE 的四个 latent channel，量化重构误差。
4. 在冻结基础模型的前提下，用 LoRA 学习梵高风格偏移，并从全新 pipeline 重载验证。
5. 运行 Canny ControlNet，并提取 cross-attention token 热力图。

当前文档与产物位于 my-diffusion-models-starter 的 main 分支；历史实现分支 codex/monorepo-organization 仍保留完整提交记录：

<https://github.com/li-cheng111/my-diffusion-models-starter/tree/main/projects/project3_stable_diffusion>

## 2. 实测结果总览

| 模块 | AutoDL 配置与结果 | 主要产物 |
|---|---|---|
| A 手写推理 | RTX 4090，seed=42，50 DDIM steps；记录 step 0/9/24/49 latent 统计 | outputs/manual/、执行后的 01_inference_walkthrough.ipynb |
| B 参数扫描 | CFG [1, 3, 7.5, 15, 25]，steps [10, 20, 50, 100]，DDIM/Euler-A/DPM++ 2M；full sweep 36.642 s | outputs/sweep/、outputs/sweep/metadata.json |
| C VAE | 输入/重构 (1,3,512,512)，latent (1,4,64,64)；MSE 0.0019791808，PSNR 27.0351 dB | outputs/vae_*.png、outputs/vae_metrics.json |
| D LoRA | 20 张图，rank=8、alpha=8、800 steps；209.47 s；loss 0.2810367 → 0.1527308；adapter 6.4 MB | outputs/lora/full/、outputs/lora/full_eval/ |
| E ControlNet | Canny 四风格 prompt 加一组结构/prompt 冲突实验 | outputs/controlnet/ |
| E Attention | 30 steps，捕获 cat/wizard/hat/forest，32 层、16×16/32×32 聚合 | outputs/attention/full/ |

验收状态：AutoDL Project 3 tests 5 passed；无 NaN/Inf；LoRA adapter 小于 25 MB；scripts/check_repo.py 通过；基础模型、HF cache、训练原图和临时 checkpoint 均未提交。

## 3. 环境、模型与数据来源

### 3.1 经过验证的环境

AutoDL 实例：NVIDIA RTX 4090 24 GB，Python 3.12.3，CUDA 12.8。

| 依赖 | 版本 |
|---|---|
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0 |
| diffusers | 0.40.0 |
| transformers | 5.15.1 |
| peft | 0.20.0 |
| accelerate | 1.14.0 |
| huggingface_hub | 1.31.0 |

依赖锁定记录见仓库根目录的 requirements/project3.txt；开始前可运行：

~~~bash
python check_env.py
python -m pytest projects/project3_stable_diffusion/tests -q
~~~

### 3.2 基础模型

代码默认使用：

~~~text
stable-diffusion-v1-5/stable-diffusion-v1-5
revision=451f4fe16113bff5a5d2269ed5ad43b0592e9a14
~~~

AutoDL 运行时 Hugging Face endpoint 不可达，因此实际使用了内容等价的 ModelScope Diffusers 镜像：

~~~text
AI-ModelScope/stable-diffusion-v1-5@master
~~~

ControlNet 使用：

~~~text
lllyasviel/sd-controlnet-canny
~~~

实际来源、文件 SHA256、GPU、依赖版本和运行参数见 outputs/autodl_environment.json。模型权重不进入 Git；脚本也支持把 --model_id 指向 AutoDL 持久盘中的本地模型目录。

### 3.3 LoRA 数据

原计划使用 Wikimedia Commons 梵高公共领域作品。AutoDL 访问 Commons API 时出现 OSError: [Errno 99] Cannot assign requested address，因此实际实验使用公开的 ModelScope huggan/vangogh2photo 镜像，抽取 20 张 imageA 图像。训练原图只保留在被忽略的 .local/datasets/project3_vangogh，来源、许可说明和 SHA256 清单见 outputs/lora/vangogh_manifest.json。源码中的 Commons downloader 仍保留，未把镜像冒充为 Commons 原图。

## 4. 端到端流程与 tensor shape

~~~text
prompt
  │
  ├─ tokenizer → token ids                         (1, 77)
  └─ CLIP text encoder → text embeddings           (1, 77, 768)
                               │
Gaussian noise                  │
(1, 4, 64, 64) ── scheduler + UNet + CFG ──────────┘
                               │
denoised latent                                      (1, 4, 64, 64)
                               │ divide by 0.18215
VAE decoder ────────────────────────────────────────┘
                               │
RGB image                                             (1, 3, 512, 512)
~~~

SD 1.5 的 VAE 把 512×512 图像下采样 8 倍，所以 latent 空间是 (batch, 4, 64, 64)；4 是 latent channel 数。CLIP tokenizer 固定 padding/truncation 到 77，text encoder hidden size 为 768。CFG 将 unconditional 和 conditional embedding 拼成 (2,77,768)，UNet 一次前向后拆分：

~~~text
eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
~~~

0.18215 是 SD 1.5 VAE 的 scaling factor，只在 latent 和 decoder 之间变换，不能省略。

## 5. 任务 A：手写 Stable Diffusion 推理

文件：01_inference_walkthrough.ipynb

### 实现要求

整个 notebook 禁止调用 pipe(prompt)，必须显式完成：

1. tokenizer：prompt → token ids；
2. CLIP text encoder：token ids → (1,77,768)；
3. 初始化 (1,4,64,64) Gaussian noise；
4. 50 个 DDIM steps：batch concat、UNet、CFG、scheduler step；
5. VAE decode：latent → (1,3,512,512)。

固定配置为 seed=42、CFG=7.5，negative prompt 为 low quality, blurry, distorted；运行中保存 step 0、9、24、49 的 latent 统计。notebook 已用 nbconvert --execute --inplace 执行并保留 execution count、文本和图片输出。

### 产物

- outputs/manual/manual_sd_seed42.png
- outputs/manual/manual_sd_stats.json
- 01_inference_walkthrough.ipynb

## 6. 任务 B：CFG、steps 与 sampler 扫描

文件：02_parameter_sweep.py

### 配置

| 参数 | full 取值 |
|---|---|
| CFG scale | [1, 3, 7.5, 15, 25] |
| inference steps | [10, 20, 50, 100] |
| sampler | DDIM、Euler-A、DPM++ 2M |
| seed | 42 |

脚本支持 --preset smoke|full、--model_revision 和 --metadata_output。smoke 用于快速验证依赖和显存，full 才是交付矩阵：

~~~bash
python 02_parameter_sweep.py \
  --preset full \
  --model_id /path/to/sd15-clean \
  --model_revision "" \
  --output_dir outputs/sweep \
  --metadata_output outputs/sweep/metadata.json \
  --seed 42
~~~

### 结果与解释

full 运行生成 16 张单图、CFG/steps/sampler 三张总图和一个二维网格，512×512，耗时 36.642 秒。CFG 从 1 增大到中等值时 prompt adherence 增强；过大时出现过饱和、边缘发硬和结构伪影。增加 steps 通常改善早期结构和局部细节，但收益递减；sampler 的离散化方式会同时改变风格和稳定性，不能把 steps 的影响单独归因于 sampler。

产物：outputs/sweep/sweep_cfg.png、sweep_steps.png、sweep_sampler.png、grid_2d.png、全部单图和 metadata.json。

## 7. 任务 C：VAE anatomy

文件：05_vae_anatomy.ipynb

使用 posterior mode 而不是随机采样，保存四个 latent channel、原图/重构图、局部 crop 和 metrics JSON。AutoDL 实测：

| 张量 | shape |
|---|---|
| 输入 RGB | (1,3,512,512) |
| VAE latent | (1,4,64,64) |
| 重构 RGB | (1,3,512,512) |

输入为 outputs/sweep_smoke/grid_2d.png，MSE=0.0019791808，PSNR=27.0351 dB。四个 channel 的 mean/std 写入 outputs/vae_metrics.json。结果表明 VAE 对低频颜色和大形状保持较好，但细小文字、尖锐边缘和高频纹理会被平滑；这也是 latent diffusion 降低计算量时的主要信息瓶颈。

产物：outputs/vae_channels.png、outputs/vae_reconstruction.png、outputs/vae_detail_crop.png、outputs/vae_metrics.json。

## 8. 任务 D：LoRA 微调与独立重载

文件：03_lora_finetune.py、evaluate_lora.py

### 实现约束

- 冻结 VAE、text encoder 和基础 UNet；
- 只向 attention 的 to_q/to_k/to_v/to_out.0 注入 LoRA；
- rank=8、alpha=8；
- LoRA master 参数保留 FP32，前向使用 FP16 autocast/GradScaler；
- 512×512、batch=1、AdamW、lr=1e-4、800 optimizer steps、seed=42；
- 支持 gradient checkpointing、max-grad-norm=1、JSONL 日志和每 200 步 checkpoint。

### 训练与评估

~~~bash
python 03_lora_finetune.py \
  --train_data_dir .local/datasets/project3_vangogh \
  --instance_prompt "a painting in the style of van gogh" \
  --output_dir outputs/lora/full \
  --model_id /path/to/sd15-clean \
  --model_revision "" \
  --seed 42 --rank 8 --num_train_steps 800 \
  --checkpointing_steps 200 --gradient_checkpointing \
  --mixed_precision fp16 --max_grad_norm 1.0 --num_workers 2 \
  --validation_prompts "a landscape in the style of van gogh" \
                       "a vase of sunflowers in the style of van gogh"

python evaluate_lora.py \
  --lora_dir outputs/lora/full \
  --output_dir outputs/lora/full_eval \
  --model_id /path/to/sd15-clean \
  --model_revision "" --seed 42 --steps 20
~~~

AutoDL 训练 20 张图耗时 209.47 秒，初始 loss=0.2810367，最终 loss=0.1527308，last-50 mean=0.2282097，无 NaN/Inf；最终 pytorch_lora_weights.safetensors 为 6,414,448 bytes。评估从全新 pipeline 依次加载 base、200/400/600/800-step adapter，结果在 outputs/lora/full_eval/。

一个关键兼容性问题是：本实验的 adapter 是 UNet-only safetensors，必须使用 unet.load_lora_adapter(..., prefix=None, weight_name=...) 重载；若直接调用 pipeline loader，缺少 unet. 前缀的 keys 可能被静默忽略。该问题已由 2-step smoke 复现、修复并写入 debug log。

可视化结果：outputs/lora/full_eval/lora_before_after.png。

## 9. 任务 E：ControlNet 与 cross-attention

### 9.1 Canny ControlNet

文件：04_controlnet_demo.ipynb

notebook 先计算 Canny，再用同一结构运行四个风格 prompt，额外执行“边缘图是猫、prompt 要求 golden retriever dog”的冲突实验。ControlNet 通过 zero-convolution 将条件分支的 residual 注入冻结的 SD UNet：zero 初始化让训练初期近似原模型，训练后才逐渐加入结构控制。

产物：outputs/controlnet/control_input_and_canny.png、controlnet_grid.png、四个 prompt 结果、controlnet_conflict_dog_on_same_edges.png 和 metadata.json。

### 9.2 Cross-attention challenge

文件：06_cross_attention_visualization.py

自定义 attention processor 只收集 CFG conditional 分支，在中后期 timestep 聚合 16×16/32×32 query map，排除 BOS/EOS/padding，再插值叠加到生成图。AutoDL 30-step 实测捕获 cat/wizard/hat/forest 四个有效 token，聚合 32 个 attention layers，数值有限且 metadata 包含 token id、层数、空间分辨率和 seed。

产物：outputs/attention/full/generated.png、四张 token heatmap overlay 和 metadata.json；outputs/attention/smoke/ 提供快速回归样例。

## 10. LDM 与 SDXL 阅读笔记

### LDM §3–4

LDM 先训练感知压缩 autoencoder：encoder E 将图像 x 映射到 latent z=E(x)，decoder D 重构 x≈D(z)，扩散只在低维 latent 上运行。前向过程可写为：

~~~text
z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
L_simple = E[ || epsilon - epsilon_theta(z_t, t, c) ||^2 ]
~~~

文本条件通过 cross-attention 注入：Q 来自当前 latent feature，K/V 来自文本 token embedding，softmax(QK^T/sqrt(d))V 将 token 语义写回空间位置；CFG 用 unconditional 与 conditional 预测的线性外推提高 prompt adherence。

LDM 的优势是降低 UNet 的空间计算和显存，限制是 VAE 压缩误差、latent scaling、VAE 解码质量和 tokenizer 都会影响最终细节。Project 3 的 A/C 直接验证了 (1,4,64,64) latent 和 0.18215 scaling factor 的作用。

### SDXL 前半部分

SDXL 不只是放大 SD 1.5 的 UNet，还使用两个 CLIP text encoder、拼接不同 embedding 空间、pooled text embedding 以及原图尺寸/裁剪坐标/目标尺寸等 added conditioning。base model 负责构图，refiner 在低噪声阶段恢复高频纹理。代价是更多显存、两个模型和更严格的尺寸 metadata。

与本项目的连接：

1. A 的 latent diffusion 正是 LDM 的压缩空间；
2. D 的 LoRA 在固定 VAE/text interface 上学习 UNet attention 的低秩偏移；
3. E 的 attention map 直接观察 QK^T 的 token 空间响应；
4. 若迁移到 SDXL，不能只替换 UNet，必须同步处理两个 text encoder 和 added conditioning。

论文指标依赖大规模数据和特定采样器；本项目的单 prompt 图片不能等价为 FID 结论。因此每次运行都记录 model revision、seed、scheduler、prompt、环境和产物 SHA256，并区分本机 smoke 与 AutoDL full run。

## 11. 自查问题与结论

1. **CFG=0 会怎样？** 只使用 unconditional 预测，prompt 的条件方向不参与外推。
2. **只用 conditional embedding 会怎样？** 仍可条件采样，但不再有 CFG 的 unconditional-to-conditional 外推，prompt 约束通常变弱。
3. **init_noise_sigma=0.5 会怎样？** 初始 latent 与 scheduler 预期尺度不匹配，可能造成亮度、对比度和细节异常，应使用 scheduler 提供的值。
4. **反向 latent 演化是否是 forward process 的简单逆过程？** 噪声尺度总体下降，但每一步还受 UNet 预测、scheduler 参数化和 CFG 影响，并非逐项取逆。
5. **steps=5 为什么质量差？** scheduler 校正机会太少，离散化误差来不及消除，结构和纹理都会变差。
6. **ControlNet 为什么复制一份 UNet？** 保留冻结的 base 分支，条件分支只学习 residual，避免训练初期破坏原有生成能力。
7. **zero convolution 的意义？** 初始 residual 为零，ControlNet 初期近似 identity；训练后卷积权重不再保持零，会逐步学会结构控制。
8. **结构与 prompt 冲突时怎么办？** ControlNet 提供空间结构，prompt 提供语义和风格，模型会在条件强度、CFG 和 denoising 过程中折中；冲突图用于观察这一点。
9. **ControlNet 多多少计算？** 至少增加一个条件分支的特征提取和 residual 计算，显存与时间高于纯 SD，但基础 UNet 可冻结复用。
10. **为什么 CFG 常用 7.5？** 它是 prompt adherence 与自然度/伪影之间的经验折中，不是理论常数；B 的扫描用于验证本 prompt 下的最佳区间。
11. **为什么 SD 1.5 常见 512×512？** VAE 将它压到 64×64 latent，训练和推理成本可控；1024×1024 会使 latent 空间边长翻倍、位置数量约增四倍。
12. **为什么 SD 画不好文字？** tokenizer/CLIP 更擅长词级语义，训练目标不是逐字符排版；latent/VAE 还会平滑细笔画，数据中的文字也缺少稳定的字符级对齐。

## 12. 真实 debug 记录

1. 旧 huggingface_hub 与 diffusers 0.40 的 cached_download 不兼容；升级到 1.31.0 后恢复。
2. Windows 工作站的输出目录 ACL 对普通 Python 进程只读；运行产物改放到仓库外持久盘，AutoDL 使用 .local/ 和仓库 outputs/。
3. LoRA 初版评估图与 base SHA256 相同；检查 state dict 后发现 loader 因缺少 unet. 前缀忽略 adapter，改用 UNet loader 并传 prefix=None 后恢复。
4. cross-attention 初版在 norm_cross=None 的模块调用 normalization，触发 assertion；现在只在 norm_cross 为真时归一化。
5. tokenizer token 含 <、>，Windows 文件名保存失败；输出 label 现在过滤为安全字符并保留 token index。
6. AutoDL Commons API 返回 OSError: [Errno 99]；改用 ModelScope 公共镜像，并在 manifest 中明确记录来源、许可证和 SHA256。

## 13. 目录与产物索引

~~~text
project3_stable_diffusion/
├── 01_inference_walkthrough.ipynb   # A：手写推理，保留执行输出
├── 02_parameter_sweep.py             # B：CFG/steps/sampler 扫描
├── 03_lora_finetune.py               # D：LoRA 训练
├── 04_controlnet_demo.ipynb          # E：Canny ControlNet，保留执行输出
├── 05_vae_anatomy.ipynb              # C：VAE 四通道与重构
├── 06_cross_attention_visualization.py
├── evaluate_lora.py                  # adapter 独立重载评估
├── prepare_vangogh_dataset.py        # 数据下载与 manifest
├── check_env.py
├── experiment_utils.py
├── tests/
├── outputs/
│   ├── manual/
│   ├── sweep/                        # full 扫描图与 metadata
│   ├── vae_*.png / vae_metrics.json
│   ├── lora/full/                    # 6.4 MB 最终 adapter、日志、validation
│   ├── lora/full_eval/               # base 与 200/400/600/800 对比
│   ├── controlnet/
│   ├── attention/full/
│   └── autodl_environment.json
├── logs/                             # smoke 与 AutoDL 原始日志
├── reading_notes/                    # LDM/SDXL 阅读笔记
├── notes/                            # 按原始作业路径提供的 reading note
├── make_report_figures.py            # 从已提交产物重绘报告图表
├── outputs/report/                   # 报告内嵌的汇总图与指标图
├── report.md                         # 可审计的原始长报告
└── README.md                         # 本统一入口
~~~

## 14. 复现与提交清单

### 最小 smoke 顺序

1. 运行 check_env.py 和 CPU tests；
2. 执行 5-step 手写推理和缩小 sweep；
3. 执行 2-step LoRA 保存/重载；
4. 执行单 prompt ControlNet 和单 token attention；
5. 通过后再运行 full sweep、A/C notebook、800-step LoRA、ControlNet 和 attention challenge。

长任务使用 tmux 或其他持久会话；日志写入 logs/，结束后核对 GPU 显存峰值、loss 是否有限、输出数量和 SHA256。

### 交付前检查

~~~bash
python -m pytest projects/project3_stable_diffusion/tests -q
python scripts/check_repo.py
git status --short
~~~

应满足：

- 执行后的 A/C notebook 保留 cell 输出；
- B 的四张总图和 metadata 齐全；
- C 的四通道、重构图、crop 和 metrics 齐全；
- D 的 800-step loss 无 NaN/Inf，adapter 可由全新 pipeline 重载且小于 25 MB；
- E 至少有四张 ControlNet 风格结果、一张冲突结果和有效 token attention map；
- Git 不包含基础模型、HF cache、训练原图或临时 checkpoint；
- 所有数字、图片、命令和结论都能追溯到 outputs、metadata 或 logs。

## 15. 参考资料

- [LDM: High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- [Diffusers LoRA documentation](https://huggingface.co/docs/diffusers/en/training/lora)

本项目的目标不是成为“只会调 prompt 的人”，而是能够解释 latent、CFG、VAE、attention 和条件控制之间的关系，并能定位一个真实的 Stable Diffusion bug。
