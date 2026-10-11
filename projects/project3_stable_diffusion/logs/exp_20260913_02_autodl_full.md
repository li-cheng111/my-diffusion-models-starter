# AutoDL 全量实验日志（2026-09-13）

> 本文依据 AutoDL 运行后保存的 metadata、训练摘要、控制台日志和当前仓库中的产物回填。实验前没有保存预注册假设，故第 2 节是事后整理，不冒充事前记录。所有未留存的信息均明确标为“未记录”。

## 实验编号

`exp_20260913_02_autodl_full`

## 一句话目标

> 在 AutoDL RTX 4090 上，Project 3 的手写推理、参数扫描、VAE、LoRA、ControlNet 和 cross-attention 可视化是否都能运行完成，并留下可核验的输出？

**判定：是。** 各部分均有对应的图像或指标文件；pytest 记录为 5 passed。此判断只表示本次任务完成并留下证据，不代表所有视觉结果都经过盲评，也不代表模型效果达到某个 FID/IS 门槛。

---

## 1. 环境信息

| 项目 | 实际记录 |
|---|---|
| 实验时间 | 2026-09-13，AutoDL metadata 时间为北京时间 23:24–23:31 左右 |
| 实验者 | 仓库记录未注明 |
| 运行平台 | AutoDL Linux，`Linux-5.15.0-97-generic-x86_64-with-glibc2.35` |
| 硬件 | NVIDIA GeForce RTX 4090，单卡；CUDA capability 8.9 |
| Python / PyTorch / CUDA | Python 3.12.3 / PyTorch 2.8.0+cu128 / CUDA 12.8 |
| 关键依赖 | Diffusers 0.40.0、Transformers 5.15.1、Accelerate 1.14.0、PEFT 0.20.0、safetensors 0.8.0、torchvision 0.23.0+cu128 |
| 其他记录的依赖 | huggingface_hub 1.31.0、opencv-python-headless 5.0.0.93、Matplotlib 3.10.5、pyarrow 20.0.0 |
| 仓库 | `li-cheng111/my-diffusion-models-starter`，当前检出 `main` |
| 当前日志所在仓库提交 | `6b14f26`（当前检出状态）；**不能证明 AutoDL 执行时使用的源码正是此提交** |
| 实验 metadata 中的 commit | 各任务记录均为 `git_commit: null`，运行源码 commit 未留存 |
| 实验分支 | AutoDL 执行时分支未记录；当前日志编辑工作区为 `main` |

### 模型与数据来源

AutoDL 无法连接 Hugging Face endpoint，因此本次从 ModelScope 镜像下载并使用本地缓存。`master` 是可移动分支，不能单独作为不可变版本标识；以下组件 SHA256 可用于核对本次缓存权重。

| 组件 | 镜像仓库与 revision | 文件大小 | SHA256 |
|---|---|---:|---|
| SD 1.5 UNet | `AI-ModelScope/stable-diffusion-v1-5@master` | 3,438,167,540 bytes | `19da7aaa4b880e59d56843f1fcb4dd9b599c28a1d9d9af7c1143057c8ffae9f1` |
| SD 1.5 VAE | 同上 | 334,643,276 bytes | `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c` |
| SD 1.5 text encoder | 同上 | 492,265,874 bytes | `d008943c017f0092921106440254dbbe00b6a285f7883ec8ba160c3faad88334` |
| Canny ControlNet | `lllyasviel/sd-controlnet-canny@master` | 1,445,157,124 bytes | `e19821a00e6d1817b37286a21d5c4f8915076949b0e81846c4f92c96ffb46db7` |
| LoRA 训练图 | ModelScope 镜像 `huggan/vangogh2photo` 的 `imageA`，20 张，存于 AutoDL 忽略目录 `.local/datasets/project3_vangogh` | 原图未提交 Git | 镜像 metadata 标示 `Apache-2.0`；此许可标注属于镜像 metadata，不等同于逐张核验原始版权 |

模型和环境的详细记录见 [`outputs/autodl_environment.json`](../outputs/autodl_environment.json)。原计划的 Wikimedia Commons 下载在 AutoDL 网络请求阶段失败，故没有把 Commons 作为实际训练数据来源。

### 显存记录限制

`autodl_environment.json` 记录 `gpu_memory_bytes: 25,252,724,736`；原始短日志另记为 `24,564 MiB`，两者口径不一致。没有保存 `nvidia-smi` 原始快照或各任务峰值统计，因此本日志不把其中任一数值解释为实验峰值显存。

---

## 2. 假设与预期

**事前假设：未记录。** 当时没有在运行前填写可证伪假设或预期阈值；下面是根据已有产物整理的事后验收问题，不是预注册结论：

- 预期各项任务在固定 seed 和配置下完成，并保存图像/指标/metadata。
- 预期 LoRA 训练可从基础模型训练出可独立加载的 adapter；通过相同 prompt、seed 对比 base 与不同 checkpoint。
- 没有预先规定 FID、IS、PSNR 或人工偏好分数的通过阈值；本次也未记录 FID/IS。

---

## 3. 配置（按任务记录）

| 任务 | 配置与实验变量 |
|---|---|
| A. 手写推理 | SD 1.5；prompt `a photograph of an astronaut riding a horse on mars`；negative prompt `low quality, blurry, distorted`；seed 42、CFG 7.5、50 steps。Notebook 按 tokenization、文本编码、初始 latent、UNet 预测、CFG、scheduler 更新及 VAE 解码逐步实现；不以 `pipe(prompt)` 代替去噪循环。 |
| B. 参数扫描 | 固定 prompt `a photograph of a cat wearing a wizard hat, fantasy art`、negative prompt `low quality, blurry`、seed 42、512×512；CFG `[1, 3, 7.5, 15, 25]`，steps `[10, 20, 50, 100]`，sampler `DDIM / Euler-A / DPM++ 2M`。扫描期间每次只改变对应维度；DPM++ 2M 属多步方法，结果受 solver 实现影响。 |
| C. VAE | 使用 posterior mode（确定性编码）；输入 `outputs/sweep_smoke/grid_2d.png`，512×512。输入本身是此前 smoke sweep 生成的合成图，不是自然图像基准集。统计 latent 四通道、重构图、MSE 与 PSNR。 |
| D. LoRA | 数据集 20 张；instance prompt `a painting in the style of van gogh`；512×512；rank 8、alpha 8；AdamW，lr `1e-4`；batch 1；seed 42；800 optimizer steps；每 200 步 checkpoint/validation；gradient checkpointing；FP16 mixed precision；max grad norm 1.0；`random_flip=false`；2 data-loader workers。 |
| D. LoRA 验证/独立评估 | 评估 seed 42、20 steps、CFG 7.5，四个固定 prompt；对比 base、200、400、600、800-step checkpoint，共 5×4 张生成图与一张对比表。 |
| E. ControlNet | Canny thresholds `[100, 200]`；seed 42；4 个风格 prompt：现代油画、漫画、赛博朋克、Pixar 3D；另用 `a realistic golden retriever dog, high detail` 做与固定边缘结构冲突的 prompt。ControlNet 是 `sd-controlnet-canny` 镜像权重。 |
| E. Cross-attention | prompt `a red cat wearing a blue wizard hat in a green forest`；seed 42、30 steps、CFG 7.5；收集条件 CFG 分支；关注 token `cat / wizard / hat / forest`；聚合 16×16、32×32 分辨率的层输出。 |

完整 sweep、LoRA 与评估参数分别保存在 [`sweep/metadata.json`](../outputs/sweep/metadata.json)、[`lora/full/training_summary.json`](../outputs/lora/full/training_summary.json)、[`lora/full_eval/metadata.json`](../outputs/lora/full_eval/metadata.json)；ControlNet 和 attention 配置见各自 `metadata.json`。

---

## 4. 运行命令与复现说明

结果 metadata 保存了主要脚本的参数，但命令字符串中的 `--model_revision` 值为空，且 `git_commit` 为空。下面按已记录参数整理为可执行形式：**移除了空值的 `--model_revision` 参数**，并保留 AutoDL 当时的本地模型缓存路径。需从 `projects/project3_stable_diffusion/` 目录运行，且需先准备同一模型权重与数据集。

参数扫描：

```bash
python 02_parameter_sweep.py \
  --preset full \
  --model_id /root/autodl-tmp/models/sd15-clean \
  --output_dir outputs/sweep \
  --metadata_output outputs/sweep/metadata.json \
  --seed 42
```

LoRA 训练：

```bash
python 03_lora_finetune.py \
  --train_data_dir .local/datasets/project3_vangogh \
  --instance_prompt "a painting in the style of van gogh" \
  --output_dir outputs/lora/full \
  --model_id /root/autodl-tmp/models/sd15-clean \
  --seed 42 --rank 8 --num_train_steps 800 --checkpointing_steps 200 \
  --gradient_checkpointing --mixed_precision fp16 --num_workers 2 \
  --max_grad_norm 1.0 \
  --validation_prompts "a landscape in the style of van gogh" "a vase of sunflowers in the style of van gogh"
```

LoRA 独立评估：

```bash
python evaluate_lora.py \
  --lora_dir outputs/lora/full \
  --output_dir outputs/lora/full_eval \
  --model_id /root/autodl-tmp/models/sd15-clean \
  --seed 42 --steps 20
```

Cross-attention 可视化：

```bash
python 06_cross_attention_visualization.py \
  --model_id /root/autodl-tmp/models/sd15-clean \
  --output_dir outputs/attention/full \
  --prompt "a red cat wearing a blue wizard hat in a green forest" \
  --tokens cat wizard hat forest --seed 42 --steps 30 --cfg 7.5
```

手写推理和 VAE 通过 notebook 执行；本次没有保存对应的原始启动命令/完整 shell history。ControlNet metadata 没有保存 CLI 命令或耗时。不要将本节整理后的命令误认为逐字保存的历史命令。

---

## 5. 结果

### 5.1 数值指标与产物核对

| 部分 | 实测结果 | 产物/说明 |
|---|---|---|
| Project 3 tests | `5 passed`（原始日志记载） | 测试命令和完整测试输出未单独归档 |
| 手写推理 | seed 42、50 steps、CFG 7.5；初始 latent std `0.9962`，第 9/24/49 步 std 分别 `0.9859 / 0.9453 / 0.8087`；最终 latent min/max `-4.9063 / 5.9492` | [`manual_sd_seed42.png`](../outputs/manual/manual_sd_seed42.png)、[`manual_sd_stats.json`](../outputs/manual/manual_sd_stats.json)。各时刻统计只描述 latent 演化，不等价于图像质量分数。 |
| 参数扫描 | 36.642 秒；16 张 PNG **总计**：12 张单参数输出（5 CFG + 4 steps + 3 samplers）、1 张 2D grid、3 张 sweep summary plots | [`outputs/sweep/`](../outputs/sweep/)；所有文件及 SHA256 列于 metadata。旧短日志写成“16 张单图 + 4 张总图/网格”，与实际文件数不符，现已按 metadata 更正。 |
| VAE | input/reconstruction `(1,3,512,512)`；latent `(1,4,64,64)`；MSE `0.0019791808`；PSNR `27.0351 dB`（像素范围按 `[0,1]`） | [`vae_metrics.json`](../outputs/vae_metrics.json)。输入是 smoke sweep 合成 grid，指标仅适用于这张图。 |
| LoRA 训练 | 800/800 steps；209.47 秒；初始记录 loss `0.2810367`、最终 `0.1527308`、最后 50 步均值 `0.2282097`；adapter 6,414,448 bytes（约 6.12 MiB / 6.41 MB），SHA256 `82b93c827e18fc4024f3bb03c87163159acc6c1c981db751c572f9f3e0bf9410` | [`training_summary.json`](../outputs/lora/full/training_summary.json)、[`training_metrics.jsonl`](../outputs/lora/full/training_metrics.jsonl)、8 张 200/400/600/800 validation 图。JSONL 没有完整 800 步逐步 loss 曲线；不要将首末值解读为严格的全程单调收敛。 |
| LoRA 独立评估 | 37.549 秒；seed 42、20 steps、CFG 7.5；base 与 4 个 checkpoint 各对 4 个 prompt 生成，共 20 张图；另有 `lora_before_after.png` | [`outputs/lora/full_eval/`](../outputs/lora/full_eval/)。SHA256 在评估 metadata 中；由全新评估进程加载 adapter 的产物已保存。 |
| ControlNet | 7 张 PNG：输入/边缘图、4 张风格结果、冲突结果、grid；Canny thresholds 100/200 | [`outputs/controlnet/`](../outputs/controlnet/)，SHA256 在 metadata。未记录运行耗时或数值化结构保持指标。 |
| Cross-attention | 9.031 秒；32 个 attention module；16×16 与 32×32 map；输出 4 个选定 token map 和一张生成图 | [`outputs/attention/full/`](../outputs/attention/full/)，token index 为 cat=3、wizard=7、hat=8、forest=12；SHA256 在 metadata。 |

**未测量的数值项：** FID、IS、感知偏好评分、LoRA 训练峰值显存、ControlNet 结构相似度。没有对应的实验记录，故不作估算或补填。

### 5.2 关键可视化结果

以下图像直接来自本次 AutoDL 产物目录；所有链接相对于本日志文件。

**手写推理（seed 42）**

![手写 Stable Diffusion 推理结果](../outputs/manual/manual_sd_seed42.png)

**参数扫描 2D 网格（固定 prompt/seed）**

![CFG 与采样步数 2D 网格](../outputs/sweep/grid_2d.png)

**LoRA base 与训练 checkpoint 对比**

![LoRA base 与 200/400/600/800 step 对比](../outputs/lora/full_eval/lora_before_after.png)

**ControlNet 多 prompt 结果**

![ControlNet 风格 prompt 结果](../outputs/controlnet/controlnet_grid.png)

**Cross-attention token map 示例（cat）**

![cat token cross-attention map](../outputs/attention/full/attention_03_cat_w.png)

训练 loss 曲线图没有留存；LoRA 的训练图是每 200 步的 validation samples，而不是 loss 曲线。对应的逐步 loss 数据也没有完整保存，因此这里不重建或伪绘曲线。

### 5.3 与 baseline 对比

| 对比 | 控制条件 | 有证据支持的结论 | 限制 |
|---|---|---|---|
| LoRA base vs. checkpoint 200/400/600/800 | 同一评估进程、4 个 prompt、seed 42、20 steps、CFG 7.5 | 5 个状态均生成了结果图，hash 不同，adapter 能在训练进程之外加载 | 评估 prompt 含 `in sks style`，而训练 prompt 是 `a painting in the style of van gogh`。`sks` 不是本次明确训练的触发词；因此这些图不能证明模型学会了 `sks` token 的语义，也没有客观质量分数。 |
| ControlNet 固定边缘 + 不同风格文本 | 同一 Canny 输入和 thresholds，4 种风格 prompt | 保存 4 种风格结果以及一组“金毛犬”文本/固定边缘冲突结果，可作定性观察 | 没有结构一致性指标或盲评；不量化宣称“结构保持率”。 |
| CFG / steps / sampler | 固定 prompt、negative prompt、seed 42、512×512；分别扫描指定参数 | 有完整图像网格和各组单图，可以视觉比较参数效果 | 本次没有对图像质量打分，不能据此声称某 sampler 或 CFG 在统计意义上最优。 |

### 5.4 结果解释

- 手写推理的 latent 标准差随着去噪进程从约 `0.9962` 降至 `0.8087`，与逐步更新 latent 的流程相符；单凭该统计不能推出生成质量优劣。
- VAE 的 latent 空间尺寸从 512×512 图像压至 64×64，通道数为 4；本次示例重构 MSE/PSNR 可复核，但由于输入只有一张合成 grid，不能推广到自然图像整体分布。
- LoRA 的最终 loss 小于首个记录值，但最后 50 步均值高于最后一步值，说明不能据此描述为平滑单调下降。权重能重新加载且输出已保存；由于评估触发词与训练 prompt 不一致，风格泛化的结论需保守。
- ControlNet 和 attention 的结果证明本次推理与 map 捕获流程产出了预期文件；缺少定量指标时只作定性展示。

---

## 6. 结论

### 6.1 假设是否成立

**事后验收结论：通过（任务与产物层面）。** A–E 各任务均有相应产物或 metadata；LoRA adapter 已保存并能在评估脚本中独立加载。此结论不等同于已证明某项方法在生成质量上优于 baseline。

### 6.2 现象记录

- Hugging Face endpoint 不可用，模型从 ModelScope 镜像获取；本次实际来源与原始计划中的 HF 模型来源不同。
- LoRA training loss 的首个、末个和末 50 步均值均已保存；完整 per-step loss 曲线没有留存。
- attention metadata 捕获 32 个模块，但只对 16×16、32×32 分辨率的 cross-attention 做目标 token 图示；图中不是模型全部注意力层的完整可视化。
- `outputs/lora/full_eval/` 中使用 `sks style` 的评估 prompt 与训练 instance prompt 不匹配，解释时应视为提示词设置问题，而不是确认存在一个已训练的 `sks` 风格 token。

---

## 7. 失败记录与踩坑

### 坑 1：AutoDL 无法连接 Hugging Face endpoint

- **现象：** HF 模型/数据 endpoint 无法访问。
- **验证与处理：** 使用 ModelScope 镜像及本地缓存继续实验。
- **结果影响：** 实验能够完成；仓库/模型的不可变 revision 没保存，metadata 中 `model_revision` 为空，不能仅凭 `@master` 精确复现。
- **教训：** 离线或镜像环境应同时归档不可变源 revision、实际文件哈希和下载来源。文件哈希已保存，但镜像 commit/revision 未锁定。

### 坑 2：Wikimedia Commons 下载失败

- **现象：** Commons API 请求报 `OSError: [Errno 99] Cannot assign requested address`。
- **处理：** 改用 ModelScope `huggan/vangogh2photo` 镜像的 `imageA`，共 20 张图；原图留在 AutoDL 忽略目录，不进入 Git。
- **结果影响：** 训练实际数据不是原计划中的 Wikimedia Commons 梵高公有领域作品。镜像 metadata 标示 Apache-2.0，但没有逐张核验原始来源/许可；不能把它描述为 Commons 数据或已逐张确认的公有领域数据。
- **教训：** 发生数据源替换时，必须同步更正数据 provenance 和许可表述；本日志按运行时记录更正。

### 坑 3：实验 commit 和模型 revision 未记录

- **现象：** 各 JSON metadata 的 `git_commit` 为 null，`model_revision` 为空；命令字段保留了无值的 `--model_revision`。
- **处理：** 本日志给出按 metadata 整理的可运行命令，并省略空值参数；当前仓库提交只代表当前日志所在代码状态，不回填为 AutoDL 实验 commit。
- **教训：** 启动任务前记录 `git rev-parse HEAD` 和真实 model commit，空 revision 不等于一个可复现版本。

### 坑 4：运行记录中的显存容量口径不一致

- **现象：** environment JSON 为 `25,252,724,736` bytes，原短日志为 `24,564 MiB`；没有原始 `nvidia-smi` 快照。
- **处理：** 保留来源与差异，不将其当作峰值显存。
- **教训：** 容量、已用显存和峰值显存应分别记录，避免混为一个“显存”数字。

### 坑 5：LoRA 评估提示词未与训练提示词对齐

- **现象：** 训练 prompt 是 `a painting in the style of van gogh`，独立评估 prompt 用 `in sks style`。
- **处理：** 本次没有事后改写图像或伪称 `sks` 是训练 token；记录为限制。
- **结果影响：** 可验证 adapter 的重新加载和输出差异，但不能据此断言 `sks` 是学到的触发词。
- **教训：** 训练与评估提示词应共享目标概念；触发词需明确写入训练 caption/instance prompt 后再测试。

### 坑 6：参数扫描输出数量被旧摘要写错

- **现象：** 旧短日志写“16 张单图 + 4 张总图/网格”。metadata 的 files 清单只有 16 张 PNG。
- **核对：** 12 张单参数结果 + 1 张 2D grid + 3 张 summary plot = 16 张总计。
- **处理：** 本日志按实际文件清单更正。

---

## 8. 下一步计划

- [ ] 后续如重跑，先把基础模型和 ControlNet 固定到不可变 commit，并保存真实 model revision 与 SHA256。
- [ ] 在训练 prompt 和评估 prompt 中统一目标触发词；若计划使用 `sks`，先在训练 caption/instance prompt 中明确训练，再单独验证。
- [ ] 训练时逐步保存有限精度、可解析的 loss JSONL，并绘制 loss curve；现有 `training_metrics.jsonl` 不能替代逐步 loss 记录。
- [ ] 每项长任务开始/结束时记录 `nvidia-smi` 和峰值显存，明确区分显存总容量与使用峰值。
- [ ] 把执行时 `git rev-parse HEAD`、完整命令、开始/结束时间、环境和输出 hash 自动写入 metadata。
- [ ] 若需要量化质量，再设计同一 prompt/seed/样本数下的 baseline 对比和相应指标；当前结果不含 FID/IS。

---

## 9. 给未来自己的提醒

> 镜像、数据源或提示词一旦与计划不同，先把真实来源和限制写进日志；没有保存的 commit、loss 曲线或峰值显存，就明确标为未记录，不要事后补猜。

---

## 10. 提交清单（自检）

- [x] 实验日志已按模板补全，且包含真实失败/调试记录。
- [x] 模型、数据来源和关键权重 SHA256 已记录（ModelScope master 本身未锁定不可变 revision）。
- [x] 参数扫描图像、VAE 指标、LoRA adapter/验证图/独立评估图、ControlNet 图、attention map 均有对应路径和 metadata。
- [x] 至少 3 张代表性结果图已在本日志中引用。
- [x] 参数扫描输出数量已按 metadata 与文件列表核对并更正。
- [ ] AutoDL 执行时的 Git commit 已记录：未记录，不能由当前提交代替。
- [ ] 实验日志变更已提交并 push：本次编辑尚未提交/推送。
- [ ] LoRA 全程 loss 曲线与逐步 loss 数据：未留存。
- [ ] 峰值显存：未留存；容量记录口径不一致。
- [ ] FID/IS 和 WandB run：本次未记录/未使用记录。
- [ ] 当前仓库外 AutoDL 原始数据集文件：未纳入 Git，位置 `.local/datasets/project3_vangogh`。
