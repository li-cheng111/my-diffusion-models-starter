# 项目 3：Stable Diffusion 完整解剖与微调

> 本目录的实现与 AutoDL 实验产物位于 monorepo 分支
> `codex/monorepo-organization`。基础模型固定为
> `stable-diffusion-v1-5/stable-diffusion-v1-5` revision；权重、缓存和原始训练图像不提交。

> **难度**：中-高
> **预期完成时间**：1.5-2 周
> **前置**：完成 L08-L09，已有 SD 概念基础。任务 A-C 学完 L09 即可动手；
> **任务 D（LoRA）和 `04_controlnet_demo.ipynb` 依赖 L10**（W9，P2 阶段），
> 按课程节奏它们会晚于任务 A-C 开放——先把 A-C 做完，等 L10 讲完再回来补 D。

## 配套教材

讲义、公式推导、论文导读在教材库，开始前请确认已 clone：

```bash
git clone https://github.com/Qi-StarterTrain/diffusion-models-starter-materials.git
```

本项目用到：

| 教材库文件 | 用途 |
|-----------|------|
| `slides/L09_latent_diffusion.md` | 主线讲义。§4 SD 架构、**§5 推理流程代码级解剖（任务 A 的蓝本）**、§6 显存优化、§8 LDM 的限制（任务 C 的分析角度） |
| `slides/L08_cfg_conditional.md` | §4-§6 CFG 与 guidance scale（任务 B 的理论依据）、§8 negative prompt 的本质 |
| `slides/L10_controlnet_lora.md` | §2 ControlNet、**§3 LoRA（任务 D）**，§3.5 给出了该改哪些 `target_modules` |
| `derivations/derive_06_cfg.md` | CFG 与 classifier guidance 的等价性证明 |
| `notebooks/nb07_sd_pipeline_anatomy.ipynb` | SD 单步内部解剖——任务 A 卡住时先看它 |
| `notebooks/nb06_cfg_scale_sweep.ipynb` | CFG scale 实验的小规模版本，任务 B 的预演 |
| `paper_notes/07_LDM_Rombach2022.md` | LDM/SD 论文导读（任务 A 的 reading note 要读 §3-§4） |
| `paper_notes/06_CFG_HoSalimans2021.md` | CFG 论文导读 |
| `paper_notes/10_ControlNet_Zhang2023.md`<br>`paper_notes/11_LoRA_Hu2021.md` | 任务 D 与 ControlNet demo |

> L09 §10 的课后任务把参数扫描写成 `02_parameter_sweep.ipynb`，本仓库提供的是
> 命令行脚本 `02_parameter_sweep.py`——跑脚本即可，不必再改写成 notebook。

---

## 任务概述

不是"训练一个 Stable Diffusion"——那需要数百 GPU-days。
而是**深入剖析**预训练 SD 模型的工作机制，并做轻量微调实验。

学习目标：
1. 理解 SD 推理流水线每一步（不当黑盒用户）
2. 掌握 VAE、UNet、CLIP text encoder 各自的作用
3. 学会调试 SD 推理出现的问题
4. 上手 LoRA 微调
5. 了解 ControlNet 等条件控制方法

---

## 目录结构

```
（仓库根目录）
├── README.md                       ← 本文件
├── check_env.py                    ← 环境自检（动手前先跑）
├── 01_inference_walkthrough.ipynb  ← 手写 SD 推理（核心，任务 A）
├── 02_parameter_sweep.py           ← CFG/steps/sampler 扫描（任务 B）
├── 03_lora_finetune.py             ← LoRA 微调脚手架 ← TODO 13-15（任务 D）
├── 04_controlnet_demo.ipynb        ← ControlNet demo（依赖 L10）
├── experiment_log_template.md      ← 实验日志模板
├── reading_note_template.md        ← 论文笔记模板（任务 A 的 reading note 用）
├── 05_vae_anatomy.ipynb            ← 任务 C：VAE 四通道与重构
├── 06_cross_attention_visualization.py ← 任务 E：token 热力图
├── evaluate_lora.py                ← LoRA checkpoint 重载与前后对比
├── prepare_vangogh_dataset.py      ← 公共领域数据下载与 SHA256 manifest
├── report.md                        ← 中文总报告、流程图、自查题与验收说明
├── reading_notes/                   ← LDM §3–4 与 SDXL 前半部分阅读笔记
├── logs/                            ← 真实 smoke/AutoDL 实验日志（不写虚构结果）
├── tests/                          ← CPU 合约测试与训练步测试
└── outputs/                        ← 生成结果保存（自己建）
```

**动手之前先跑环境自检**——SD 权重 7 GB，跑到一半才发现依赖缺失很浪费时间：

```bash
python check_env.py
```

从 monorepo 根目录运行 Project 3 的静态与训练步测试：

```bash
python -m pytest projects/project3_stable_diffusion/tests -q
```

它会检查依赖包、显存、以及 HuggingFace 是否连得上（只发 HEAD 请求，不下载权重）。

---

## 任务清单

### 任务 A（必做）：手写 SD 推理流水线

完成 `01_inference_walkthrough.ipynb`。

**禁止使用 `pipe(prompt)`**！必须分阶段手写：
1. Tokenize：text → token ids
2. Text encode：token ids → text embeddings (77, 768)
3. Init noise：sample (4, 64, 64)
4. Loop：for 50 DDIM steps
   - Batch concat (cond + uncond)
   - UNet forward
   - CFG combination
   - DDIM step
5. VAE decode：(4, 64, 64) → (3, 512, 512)

每一步要打印 tensor shape 和统计量，确认你理解了。

对照 L09 §5 的代码级解剖来写——那里给的是同一套流程的精简版，
notebook 里多出来的 `scale_model_input` / `init_noise_sigma` 是为了兼容非 DDIM sampler。

**同时提交一份 reading note**（L09 §10 必做第 3 项）：LDM 论文 §3-§4 + SDXL
technical report 第一部分，按 `reading_note_template.md` 写，放到 `notes/`。

---

### 任务 B（必做）：参数扫描实验

运行 `02_parameter_sweep.py`，固定 prompt 和 seed，扫描：

| 参数 | 取值 |
|------|------|
| CFG scale | [1, 3, 7.5, 15, 25] |
| Inference steps | [10, 20, 50, 100] |
| Sampler | DDIM, Euler-A, DPM++ 2M |

```bash
python 02_parameter_sweep.py \
    --prompt "a photograph of a cat wearing a wizard hat, fantasy art" \
    --seed 42 --output_dir ./outputs/sweep
```

AutoDL 上首次验证环境时使用缩小矩阵：

```bash
python 02_parameter_sweep.py --preset smoke --output_dir ./outputs/sweep_smoke
```

脚本会跑 4 组实验，输出 `sweep_cfg.png`、`sweep_steps.png`、`sweep_sampler.png`、
`grid_2d.png`（CFG × steps 二维网格）到 `--output_dir`，同时单独存每张图。

观察：
- 大 CFG 的伪影特征（过饱和、结构崩坏从哪个 scale 开始出现）
- 少步采样的失败模式
- 不同 sampler 的"风格差异"——同 seed 同 prompt 下差多少

提交这 4 张图 + 2 页观察分析。做过 Project 2 的话，
可以顺带对比一下：SD 上的 sampler 差异和你在 CIFAR-10 上量到的 FID-NFE 趋势一致吗？

---

### 任务 C（必做）：VAE 解剖

本任务没有单独的脚手架——**在 `01_inference_walkthrough.ipynb` 末尾自己加 cell**
（VAE 已经在那里加载好了），或者新建 `05_vae_anatomy.ipynb`：

1. 加载真实图像，用 SD 的 VAE encoder → 得到 latent
2. 可视化 latent 的 4 个 channels（每个独立看）
3. 注意 channel 之间的差异（有的偏向亮度，有的偏向色彩）
4. 用 VAE decoder 还原，与原图对比

观察问题：
- VAE 重构在哪些细节上失真？
- 不同 channel 编码了什么信息？
- 文字、小目标是否丢失？

---

### 任务 D（进阶）：LoRA 微调

完成 `03_lora_finetune.py` 中的 **TODO 13-15**。

任务：用 10-30 张特定风格的图像（如某画风、某 object），LoRA 微调 SD。

```bash
python 03_lora_finetune.py \
    --train_data_dir ./my_dataset \
    --instance_prompt "a photo of sks dog" \
    --output_dir ./outputs/lora \
    --rank 8 --num_train_steps 800
```

完整实验显式固定 seed、revision，并在 200 步间隔保存 adapter：

```bash
python 03_lora_finetune.py \
    --train_data_dir .local/datasets/project3_vangogh \
    --instance_prompt "a painting in sks style" \
    --output_dir outputs/lora --seed 42 --rank 8 \
    --num_train_steps 800 --checkpointing_steps 200 \
    --gradient_checkpointing --mixed_precision fp16 \
    --validation_prompts "a landscape in the style of van gogh" \
                       "a vase of sunflowers in the style of van gogh"

python evaluate_lora.py --lora_dir outputs/lora \
    --output_dir outputs/lora/evaluation --seed 42
```

下载公共领域训练图并生成来源/许可/SHA256 清单：

```bash
python prepare_vangogh_dataset.py \
    --output_dir .local/datasets/project3_vangogh \
    --manifest_output outputs/lora/vangogh_manifest.json
```

参考：
- **L10 §3**（LoRA）——§3.5 直接给出了 SD 该改哪些 `target_modules`，§3.6 给了数据量/步数的量级
- `paper_notes/11_LoRA_Hu2021.md`
- diffusers 的 LoRA 教程

超参起点（照 L10 §3.6）：单 object 用 5-30 张图训 500-2000 步，rank 8-32，lr 1e-4。

> ⚠️ 脚手架里 vae / text_encoder 用 fp16 冻结，**UNet 与 LoRA 参数保持 fp32**，
> 混合精度靠 `autocast` + `GradScaler`。TODO 14 的提示里写了正确写法——
> 直接把 LoRA 参数放进 fp16 再喂给 AdamW，loss 会一路 NaN，这是最常见的翻车点。

提交：
- LoRA 权重文件（几 MB，可以直接提交）
- 微调前后对同一 prompt + 同一 seed 的对比图
- 训练日志（按 `experiment_log_template.md`）

---

### 任务 E（挑战）：Cross-Attention 可视化

在采样过程中提取 cross-attention map，可视化每个 token 对应的图像区域。

参考：
- Hertz et al., *Prompt-to-Prompt Image Editing* (ICLR 2023)
- `diffusers` 的 attention processor 接口

需要 hook 进 UNet 的内部模块，提取 attention 矩阵。

命令行挑战实现：

```bash
python 06_cross_attention_visualization.py \
    --output_dir outputs/attention --seed 42
```

**或者**做 L09 §10 挑战档那道：自己组装 **SDXL 双模型（base + refiner）推理流水线**。
两道二选一即可，计同样的 bonus 分。

---

## 自查问题（在报告中回答）

实现状态：A–E 的代码路径、测试、AutoDL 命令和交付物目录均已准备；A/C notebook
已在本机成功执行并保留输出，模型权重、缓存和训练原图继续保持被忽略。AutoDL
full 产物应按 `report.md` 与 `logs/` 的 runbook 回填，不能用本机 smoke 数字冒充。

前 5 道来自 `01_inference_walkthrough.ipynb` 末尾的思考题，
第 6-9 道来自 `04_controlnet_demo.ipynb`（做了 ControlNet 再答）：

1. 把 `cfg=0`（无 guidance）会生成什么？为什么？
2. 只用 `cond_emb`、不做 batch concat，结果会怎样？
3. 把 `init_noise_sigma` 改成 `0.5` 会怎样？
4. 在采样循环里打印 `z` 的统计量——它的演化和 DDPM 训练时的 forward process 反过来是否一致？
5. 把 `num_steps` 改成 5，结合 L07 解释为什么质量这么差。
6. ControlNet 为什么不直接改 SD UNet，而要复制一份再加？
7. Zero convolution 初始化为 0 有什么意义？训练后还是 0 吗？
8. ControlNet 条件与 prompt 冲突时（canny 是猫、prompt 说狗），SD 怎么处理？
9. ControlNet 推理比纯 SD 多多少计算？

另外，下面这三个问题请务必能答上来——它们是本 Project 真正的验收标准
（第二个在 L09 §12 FAQ 里有答案，另外两个要靠你做完任务 B、C 的观察）：
**为什么 CFG 默认 7.5？为什么 SD 出 512×512 而不是 1024×1024？为什么 SD 画不好文字？**

---

## 评分细则

| 项 | 权重 | 评分要点 |
|----|------|---------|
| 任务 A（手写推理） | 30% | 不可调高层 API；每步注释清晰 |
| 任务 B（参数扫描） | 25% | 实验严谨，观察有 insight |
| 任务 C（VAE 解剖） | 20% | 可视化清晰，分析合理 |
| 任务 D（LoRA） | 20% | 进阶 |
| 任务 E（attention） | 5% | 挑战（bonus） |

---

## 环境要求

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install matplotlib     # 任务 B 画网格图
pip install peft           # 任务 D：LoRA
pip install opencv-python  # ControlNet 的 canny 预处理
```

装完跑 `python check_env.py` 确认——它会逐项列出缺哪个包、显存够不够、模型连不连得上。

**显存要求**：
- 推理：8 GB 起步（fp16 6 GB）
- LoRA 微调：12 GB 起步（fp16 + gradient checkpointing 8 GB）

如显存不够，建议租 Colab Pro 或类似服务。

---

## 模型下载

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")
```

第一次会下载 ~7GB 权重。

**关于 model id**：`runwayml/stable-diffusion-v1-5` 这个仓库已经转到社区托管的
`stable-diffusion-v1-5/stable-diffusion-v1-5`，HuggingFace 做了重定向，
旧 id 目前仍然可用（`check_env.py` 会实测）。教材 L09 §10 和 `nb07` 用的都是旧 id，
本仓库保持一致；哪天旧 id 失效，把 `MODEL_ID` / `--model_id` 换成新的即可，别的都不用改。

**下载慢或连不上**（国内网络常见）：

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 用镜像
export HF_HOME=/path/with/space              # 权重缓存换个大盘
python check_env.py                          # 确认通道 OK 再开始下载
```

notebook 里的 `MODEL_ID` 是普通变量，改成本地已下载的目录路径也可以。

---

## 提交要求

**截止前将以下内容 push 到你的作业仓库 main 分支**，助教直接在仓库里评分。

```
（仓库根目录）
├── 01_inference_walkthrough.ipynb  # 保留输出（助教要看 shape 打印）
├── 03_lora_finetune.py             # 含你实现的 TODO 13-15
├── 04_controlnet_demo.ipynb        # 保留输出（如完成）
├── outputs/
│   ├── sweep/                      # 任务 B 的扫描图（sweep_cfg / sweep_steps / sweep_sampler / grid_2d）
│   ├── vae_channels.png            # 任务 C 的 latent 4 通道可视化
│   ├── lora/                       # 任务 D 的 LoRA 权重（几 MB，直接提交）
│   └── lora_before_after.png       # 任务 D 的微调前后对比
├── notes/reading_note_ldm.md       # 任务 A 的 reading note
├── logs/exp_log.md                 # 实验日志（按 experiment_log_template.md）
├── report.md                       # 4-6 页报告
└── debug_log.md                    # 至少 3 条踩坑记录
```

`report.md` 需包含：
1. 任务 A 的推理流程图（手画或 mermaid）+ 每个 tensor shape 的来源
2. 任务 B 的对比图与分析（2 页）
3. 任务 C 的 VAE 可视化与发现
4. 上面「自查问题」
5. 任务 D/E（如完成）

> **notebook 请带输出提交**——清掉输出等于没交。
> SD 权重、数据集图片不要提交（见 `.gitignore`）；LoRA 权重只有几 MB，可以交。

---

## 常见 Bug

### 1. CUDA OOM

`pipe.enable_attention_slicing()` 或 `pipe.enable_vae_slicing()`。

### 2. 黑色 NSFW filter 图

`pipe.safety_checker = None`（仅自用，不要分享）。

### 3. LoRA 训练 loss 是 NaN

**最常见原因**：LoRA 参数被放进 fp16 直接喂给 AdamW。

正确做法见 TODO 14 的提示：冻结的 vae / text_encoder 用 fp16，**UNet 与 LoRA 参数
保持 fp32**，前向用 `autocast`，反向用 `GradScaler`。脚手架的 `main()` 已经按这个
方式加载模型了，你只要在 `train_one_step` 里照着写。

先用 `--mixed_precision no` 跑通，确认 loss 正常下降，再开 fp16。

### 4. LoRA 训练 loss 不降

- 检查 LoRA rank 是否合理（L10 §3.5 建议 8-32）
- learning rate 不要太大（1e-4 起步）
- 数据集太小（<5 张）很难学到
- 步数不够：单 object 通常要 500-2000 步（L10 §3.6），几十步是看不出效果的

### 5. 微调后 catastrophic forgetting

LoRA 比 full fine-tune 好得多，但仍可能。
- 降低 rank
- 别训过头——效果饱和后继续训只会加剧过拟合，在 500-2000 步区间里多存几个
  checkpoint 做对比，挑最好的那个
- 用更多样化的训练 prompt

---

## 学术诚信

⚠️ `diffusers` 的 pipeline 源码和官方 LoRA 训练脚本都是很好的参考，但：

- 任务 A **明令禁止调 `pipe(prompt)`**——整个任务的意义就在于自己把 6 步串起来
- TODO 13-15 允许参考 diffusers 的 `train_text_to_image_lora.py` 的**结构**，
  但必须自己写、自己调
- 直接复制粘贴并提交将记 0 分
- 报告里的"踩坑记录"必须真实——抄来的踩坑是看得出来的

---

## 推荐阅读

1. L09 讲义全文；LoRA / ControlNet 部分看 L10 §2-§3
2. SD 论文（Rombach 2022）§3-§4 —— 导读见 `paper_notes/07_LDM_Rombach2022.md`
3. LoRA 论文（Hu 2021）—— 导读见 `paper_notes/11_LoRA_Hu2021.md`
4. ControlNet 论文（Zhang 2023）—— 导读见 `paper_notes/10_ControlNet_Zhang2023.md`
5. HuggingFace blog 《Annotated Diffusion》《How does Stable Diffusion work?》

---

## 给学生的话

很多人能用 SD 但不理解它。本 project 让你"成为能修 SD bug 的人"，而不是"调 prompt 的人"。

完成后你应当能：
- 解释为什么 CFG=7.5
- 解释 SD 为什么生成 512×512 而非 1024×1024
- 解释为什么 SD 不能生成清晰文字
- 写一个自己的 SD inference 函数

这些是产业级 know-how，比训自己的 SD 更有用。
