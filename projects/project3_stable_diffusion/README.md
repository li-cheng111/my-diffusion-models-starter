# Project 3：Stable Diffusion 实验报告

## 1. 实验内容

本项目以 Stable Diffusion 1.5 为对象，逐段复现并检查文本到图像的 latent diffusion 路径，完成手写推理、参数扫描、VAE 解剖、LoRA 微调、Canny ControlNet、cross-attention 可视化，以及 LDM/SDXL 阅读笔记。

| 作业模块 | 结果 | 相关文件 |
|---|---|---|
| A：不调用高层 pipeline 的手写推理 | 已完成并保留 notebook 输出 | 01_inference_walkthrough.ipynb、outputs/manual/ |
| B：CFG、steps、sampler 对比 | 已完成 full preset，seed 42 | outputs/sweep/、metadata.json |
| C：VAE latent 通道与重构 | 已执行，含四通道图、crop、MSE/PSNR | 05_vae_anatomy.ipynb、outputs/vae_*.png、vae_metrics.json |
| D：LoRA 微调、分段保存、独立加载 | 完成 800 optimizer steps | outputs/lora/full/、outputs/lora/full_eval/ |
| E：Canny ControlNet 与 attention challenge | 四种风格、结构/文本冲突、四个 token attention map | outputs/controlnet/、outputs/attention/full/ |
| LDM §3–4 与 SDXL 阅读笔记 | 已完成 | notes/reading_note_ldm.md、reading_notes/ldm_sdxl_reading_note.md |
| 环境、复现与 debug | 环境快照、运行日志和真实问题记录 | outputs/autodl_environment.json、logs/、本报告第 10 节 |

## 2. Stable Diffusion 结构

文本到图像的主要张量流：

**prompt → tokenizer → CLIP text encoder → text embedding；
随机 latent → UNet + CFG + scheduler → 去噪 latent → VAE decoder → RGB 图像。**

对于 SD 1.5、512×512 输入/输出：

| 阶段 | 张量 shape | 来源/含义 |
|---|---|---|
| token ids | (1, 77) | CLIP 固定长度 padding/truncation |
| text embedding | (1, 77, 768) | CLIP 序列长度 77、hidden size 768 |
| 初始 latent | (1, 4, 64, 64) | 图像经 VAE 下采样 8 倍；4 为 latent channels |
| CFG UNet 输入 embedding | (2, 77, 768) | unconditional 与 conditional embedding 沿 batch 拼接 |
| 去噪 latent | (1, 4, 64, 64) | UNet 预测与 scheduler 更新后的 latent |
| 解码 RGB | (1, 3, 512, 512) | 解码前按 VAE scaling factor 0.18215 还原 latent 尺度 |

## 4. A：Stable Diffusion 推理框架实现

文件：01_inference_walkthrough.ipynb，该 notebook 逐步运行 tokenizer、CLIP、噪声初始化、DDIM、CFG、scheduler 更新和 VAE decode。

固定配置：prompt 为 **astronaut/horse/mars** 场景，negative prompt 为 **low quality, blurry, distorted**，seed=42，50 个 DDIM steps，CFG=7.5。

### 采样过程中的 latent 统计

| 循环 step | scheduler timestep | mean | std | min | max |
|---:|---:|---:|---:|---:|---:|
| 0 | 981 | 0.00173 | 0.99620 | -3.92188 | 3.92578 |
| 9 | 801 | -0.00029 | 0.98591 | -3.81445 | 3.77148 |
| 24 | 501 | -0.00298 | 0.94525 | -3.60352 | 3.93555 |
| 49 | 1 | -0.00781 | 0.80868 | -4.90625 | 5.94922 |

标准差总体下降，表示 latent 的整体噪声尺度随去噪过程减小

产物：[手写推理生成图](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_seed42.png)、[latent 统计 JSON](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_stats.json)。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/manual/manual_sd_seed42.png" alt="手写 Stable Diffusion 推理结果" width="960"></p>
<p align="center"><strong>图 1｜手写推理生成结果</strong><br><sub>seed=42、50 DDIM steps、CFG=7.5</sub></p>

### Reading Notes：LDM和SDXL
链接：[reading_note_ldm.md](https://github.com/li-cheng111/my-diffusion-models-starter/blob/main/projects/project3_stable_diffusion/notes/reading_note_ldm.md)

## 5. B：CFG、steps 与 sampler 扫描

脚本：02_parameter_sweep.py。full preset 在同一 prompt、negative prompt 和 seed=42 下改变一个因素或绘制二维交互图。prompt 为 **“a photograph of a cat wearing a wizard hat, fantasy art”**，negative prompt 为 **“low quality, blurry”**；输出尺寸 512×512。

| 对比量 | 扫描配置 | 控制变量/结果图 |
|---|---|---|
| CFG scale | 1、3、7.5、15、25 | 其余配置固定；sweep_cfg.png |
| inference steps | 10、20、50、100 | CFG=7.5；sweep_steps.png |
| sampler | DDIM、Euler-A、DPM++ 2M | 30 steps、CFG=7.5；sweep_sampler.png |
| CFG × steps | CFG=3、7.5、15 × steps=10、20、50 | 3×3 grid_2d.png |

输出按 metadata 记录 16 个 PNG 文件：12 张独立条件结果、1 张含 9 个样本格的二维网格、3 张横向汇总图。

### 实测图

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_cfg.png" alt="CFG scale 五档对照图" width="960"></p>
<p align="center"><strong>图 2a｜CFG scale 对照</strong><br><sub>固定 prompt、seed=42 与其余设置，比较 CFG=1、3、7.5、15、25。</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_steps.png" alt="采样步数四档对照图" width="960"></p>
<p align="center"><strong>图 2b｜采样步数对照</strong><br><sub>固定 CFG=7.5，比较 10、20、50、100 steps。</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_sampler.png" alt="DDIM、Euler-A、DPM++ 2M 对照图" width="960"></p>
<p align="center"><strong>图 2c｜Sampler 对照</strong><br><sub>固定 30 steps 与 CFG=7.5；sampler 更新方式不同，因此同 seed 不保证像素一致。</sub></p>

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/grid_2d.png" alt="CFG 与 steps 的三乘三交互网格" width="960"></p>
<p align="center"><strong>图 2d｜CFG × steps 交互</strong><br><sub>行列分别对应 CFG=3、7.5、15 与 steps=10、20、50；这是子网格，不是完整 5×4 组合。</sub></p>

**可以看到：** CFG 从 1 增至 3/7.5 后，猫和巫师帽的语义更明确；到 15/25 时颜色和边缘更强烈，出现过饱和倾向。steps 从 10 到 20 可见结构明显成形，50/100 继续细化但增益较小。

高清图像链接：[sweep_cfg.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_cfg.png) · [sweep_steps.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_steps.png) · [sweep_sampler.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/sweep_sampler.png) · [grid_2d.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/sweep/grid_2d.png)

## 6. C：VAE anatomy

文件：05_vae_anatomy.ipynb。使用 encoder posterior 的 mode，记录四个 latent channel、原图/重构图、局部细节 crop 和 metrics JSON。

| 张量 | shape |
|---|---|
| 输入 RGB | (1, 3, 512, 512) |
| latent | (1, 4, 64, 64) |
| 重构 RGB | (1, 3, 512, 512) |

**实测 MSE = 0.0019791808，PSNR = 27.0351 dB。** 四个通道统计如下：

| latent channel | mean | std |
|---:|---:|---:|
| 1 | 0.47984 | 1.11195 |
| 2 | -0.01039 | 1.36113 |
| 3 | 0.14115 | 0.59798 |
| 4 | -0.17118 | 0.73707 |

<p align="center"><a href="outputs/vae_channels.png"><img src="outputs/vae_channels.png" alt="VAE 四个 latent 通道" width="960"></a><br><strong>A · 四个 latent 通道</strong></p>
<table>
  <tr>
    <td align="center" width="33%"><a href="outputs/vae_reconstruction.png"><img src="outputs/vae_reconstruction.png" alt="VAE 输入与重构" width="300"></a><br><strong>B · 输入与重构对比</strong></td>
    <td align="center" width="33%"><a href="outputs/vae_detail_crop.png"><img src="outputs/vae_detail_crop.png" alt="VAE 局部细节" width="300"></a><br><strong>C · 局部细节</strong></td>
    <td align="center" width="33%"><a href="outputs/report/vae_statistics.png"><img src="outputs/report/vae_statistics.png" alt="VAE 通道统计与重构指标" width="300"></a><br><strong>D · 通道统计与重构指标</strong></td>
  </tr>
</table>
<p align="center"><strong>图 3｜VAE anatomy：通道、重构与指标</strong><br><sub>A：四个 latent channel；B：输入/重构；C：局部细节；D：通道统计和 MSE/PSNR。</sub></p>

原始图链接：[vae_channels.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_channels.png) · [vae_reconstruction.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_reconstruction.png) · [vae_detail_crop.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/vae_detail_crop.png) · [vae_statistics.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/report/vae_statistics.png)

**结果解释：** 低频色彩和整体形状较容易保留，细小文字、锐利边缘和高频纹理会平滑。VAE 是降低空间计算成本的压缩环节，也构成最终细节的信息瓶颈。

## 7. D：LoRA 风格微调与独立加载

文件：03_lora_finetune.py、evaluate_lora.py。

### 训练实现与配置

- 冻结 VAE、text encoder 和基础 UNet，仅在 UNet attention 的 to_q、to_k、to_v、to_out.0 注入 LoRA。
- rank=8、alpha=8；LoRA master 参数保持 FP32，前向采用 FP16 autocast/GradScaler。
- 输入训练分辨率 512×512，batch size=1，AdamW，learning rate=1e-4，800 optimizer steps，seed=42。
- 开启 gradient checkpointing，max_grad_norm=1.0；每 200 步保存 adapter checkpoint、验证图和 JSONL 训练记录。
- 数据集共 20 张镜像图，训练 instance prompt 为 **“a painting in the style of van gogh”。**

### LoRA评估

evaluate_lora.py 在 SD pipeline 中评估 base、checkpoint-0200、0400、0600、0800。比较使用 seed=42、20 inference steps、CFG=7.5，四个 prompt 是 **lighthouse by the sea、quiet street at night、vase of sunflowers、mountain landscape**，因此得到 5 个模型状态 × 4 个固定 prompt 的 20 张对比图。

<p align="center"><img src="https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/full_eval/lora_before_after.png" alt="LoRA 四组 prompt 与五个模型状态对比" width="960"></p>
<p align="center"><strong>图 4b｜LoRA adapter 独立加载对照</strong><br><sub>列为 base、200、400、600、800 steps；行对应 lighthouse、street、sunflowers、mountain。</sub></p>

原始图：[lora_before_after.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/lora/full_eval/lora_before_after.png)

## 8. E：Canny ControlNet 与 cross-attention

### 8.1 Canny ControlNet

文件：04_controlnet_demo.ipynb。输入图先用 Canny 提取边缘（thresholds=[100, 200]），然后固定结构，使用四种 prompt 做重绘；另加要求 **“a realistic golden retriever dog, high detail”** 的冲突 prompt，观察文本与边缘条件争夺控制权。生成 seed=42。

四种风格 prompt 为现代油画、漫画、赛博朋克角色、Pixar 3D 渲染，分别保存结果并组合成网格：

<p align="center"><a href="outputs/controlnet/control_input_and_canny.png"><img src="outputs/controlnet/control_input_and_canny.png" alt="输入图与 Canny 边缘" width="720"></a><br><strong>A · 输入图与 Canny 边缘</strong></p>
<table>
  <tr>
    <td align="center" width="50%"><a href="outputs/controlnet/controlnet_prompt_00.png"><img src="outputs/controlnet/controlnet_prompt_00.png" alt="现代油画风格 ControlNet 结果" width="460"></a><br><strong>B · 现代油画</strong></td>
    <td align="center" width="50%"><a href="outputs/controlnet/controlnet_prompt_01.png"><img src="outputs/controlnet/controlnet_prompt_01.png" alt="漫画风格 ControlNet 结果" width="460"></a><br><strong>C · 漫画</strong></td>
  </tr>
  <tr>
    <td align="center" width="50%"><a href="outputs/controlnet/controlnet_prompt_02.png"><img src="outputs/controlnet/controlnet_prompt_02.png" alt="赛博朋克风格 ControlNet 结果" width="460"></a><br><strong>D · 赛博朋克</strong></td>
    <td align="center" width="50%"><a href="outputs/controlnet/controlnet_prompt_03.png"><img src="outputs/controlnet/controlnet_prompt_03.png" alt="Pixar 3D 风格 ControlNet 结果" width="460"></a><br><strong>E · Pixar 3D</strong></td>
  </tr>
</table>
<p align="center"><a href="outputs/controlnet/controlnet_conflict_dog_on_same_edges.png"><img src="outputs/controlnet/controlnet_conflict_dog_on_same_edges.png" alt="固定边缘与 golden retriever prompt 冲突实验" width="460"></a><br><strong>F · 固定边缘与 golden retriever 文本冲突</strong></p>
<p align="center"><strong>图 5a｜Canny ControlNet 条件实验</strong><br><sub>A：输入与边缘图；B–E：四种风格 prompt；F：结构/文本冲突。统一 seed=42。</sub></p>

原始图：[control_input_and_canny.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/control_input_and_canny.png) · [controlnet_grid.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/controlnet_grid.png) · [controlnet_conflict_dog_on_same_edges.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/controlnet/controlnet_conflict_dog_on_same_edges.png)

### 8.2 Cross-attention challenge

脚本：06_cross_attention_visualization.py。prompt 为 **“a red cat wearing a blue wizard hat in a green forest”**，seed=42、30 steps、CFG=7.5。自定义 attention processor 在 CFG 条件分支采集 prompt 相关 attention，在去噪中后段聚合 16×16 和 32×32 query map，并排除 BOS、EOS 及 padding token。

有效 token 为 cat、wizard、hat、forest，token indices 分别是 3、7、8、12；metadata 记录了 32 个 attention 模块名、两种空间分辨率、token IDs。

<table>
  <tr>
    <td align="center" width="50%"><a href="outputs/attention/full/attention_03_cat_w.png"><img src="outputs/attention/full/attention_03_cat_w.png" alt="生成图与 cat token attention map" width="480"></a><br><strong>A · cat</strong></td>
    <td align="center" width="50%"><a href="outputs/attention/full/attention_07_wizard_w.png"><img src="outputs/attention/full/attention_07_wizard_w.png" alt="生成图与 wizard token attention map" width="480"></a><br><strong>B · wizard</strong></td>
  </tr>
  <tr>
    <td align="center" width="50%"><a href="outputs/attention/full/attention_08_hat_w.png"><img src="outputs/attention/full/attention_08_hat_w.png" alt="生成图与 hat token attention map" width="480"></a><br><strong>C · hat</strong></td>
    <td align="center" width="50%"><a href="outputs/attention/full/attention_12_forest_w.png"><img src="outputs/attention/full/attention_12_forest_w.png" alt="生成图与 forest token attention map" width="480"></a><br><strong>D · forest</strong></td>
  </tr>
</table>
<p align="center"><strong>图 5b｜Cross-attention token 空间响应</strong><br><sub>每格左侧是同一生成图，右侧是对应 token 的 attention map；热图表示模型响应，不是像素级分割真值。</sub></p>

原始图链接：[generated.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/generated.png) · [attention_03_cat_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_03_cat_w.png) · [attention_07_wizard_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_07_wizard_w.png) · [attention_08_hat_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_08_hat_w.png) · [attention_12_forest_w.png](https://raw.githubusercontent.com/li-cheng111/my-diffusion-models-starter/6b14f26cd7ea2a67d3b231a97cde80baea636ca2/projects/project3_stable_diffusion/outputs/attention/full/attention_12_forest_w.png)

## 9. 自查问题与结论

### 推理与采样

1. **CFG scale 设为 0，prompt 还会影响结果吗？** CFG 公式为 `ε = ε_uncond + s(ε_cond - ε_uncond)`。当 `s=0` 时结果只取 unconditional 预测，因此 prompt 不再通过 CFG 引导生成。
2. **只把 conditional embedding 输入 UNet 会怎样？** 模型仍按 prompt 条件去噪，但没有 unconditional 预测作对照，因而无法计算 CFG 的外推方向，效果相当于标准 CFG 的 s=1，图像仍能生成，但 prompt 约束通常弱于 s>1 的 CFG。
3. **把初始 latent 乘以 0.5 会怎样？** latent 的形状仍正确，但噪声幅度与 scheduler 预期不符，后续去噪轨迹随之改变，可能影响构图、对比度和细节。
4. **反向去噪是不是 forward 加噪的简单逆过程？** 不是。forward 是逐步加入随机噪声；反向过程由 UNet 估计噪声或速度，再由 scheduler 近似更新。它不会逐步还原某次加噪所丢失的全部信息，不是简单的逆过程。
5. **为什么 5 个采样步通常不如更多步？** 每一步都要跨越更大的噪声区间，有限次更新难以充分修正误差，细节和结构更容易不稳定。增加步数通常提高细化程度，但会增加耗时，且收益并非线性增长。

### ControlNet 条件控制

6. **为什么 ControlNet 使用额外的条件分支？** 分支把边缘等空间条件编码成多尺度特征，再以 residual 注入冻结的 Stable Diffusion 主干。这样既保留原模型的生成能力，也能学习结构控制，不必从头训练或大幅改写主干。
7. **Zero convolution 的作用是什么？** 它是零初始化的卷积层，因此训练开始时分支注入主干的 residual 为零，模型行为接近原始 Stable Diffusion。训练中这些卷积逐渐学会传递有效控制信号。
8. **边缘结构与文字 prompt 冲突时，谁优先？** 没有固定的优先者：ControlNet 倾向保留输入的轮廓和布局，prompt 倾向指定对象语义与风格。控制强度、CFG 和模型先验共同影响取舍。
9. **冻结基础模型后，ControlNet 还增加成本吗？** 会。冻结可以减少训练时主干的梯度与优化器状态，但推理时仍须运行 ControlNet 分支并计算注入特征，因此显存占用和每个去噪步的耗时都会增加。

### Stable Diffusion 的经验与局限

10. **为什么常见 CFG scale 是 7.5？** 这是 SD 1.5 工作流中常用的经验起点，不是理论常数。调高通常会加强 prompt 对齐，也可能带来过饱和或伪影；调低可能更自然，但文本约束变弱。合适值需结合 prompt、模型与 sampler 比较。
11. **为什么 SD 1.5 常用 512×512？** 512 是其典型训练分辨率。VAE 的 8 倍下采样会把 512×512 图像映射到 `4×64×64` latent；宽高同时翻倍时，latent 空间位置数变为 4 倍，计算和显存开销也会上升。更高分辨率不保证细节质量同步提升。
12. **为什么扩散模型容易把文字画错？** 主要瓶颈不只是 tokenizer：文本条件更擅长表达整体语义，模型并未被专门训练成精确排版器；低分辨率训练与 VAE 压缩又会损失细小笔画。因此拼写、字数和位置都可能不准确。

## 10. 代码、产物与 Release 索引

### 实验代码与仓库内结果

| 内容 | 主要代码 | 结果与记录 |
| --- | --- | --- |
| A · 手写推理 | [`01_inference_walkthrough.ipynb`](01_inference_walkthrough.ipynb) | [`outputs/manual/`](outputs/manual/) |
| B · 参数扫描 | [`02_parameter_sweep.py`](02_parameter_sweep.py) | [`outputs/sweep/`](outputs/sweep/)、[`outputs/sweep_smoke/`](outputs/sweep_smoke/) |
| C · VAE 解剖 | [`05_vae_anatomy.ipynb`](05_vae_anatomy.ipynb) | `outputs/vae_*.png`、[`outputs/vae_metrics.json`](outputs/vae_metrics.json) |
| D · LoRA 训练与评估 | [`03_lora_finetune.py`](03_lora_finetune.py)、[`evaluate_lora.py`](evaluate_lora.py) | [`outputs/lora/full/`](outputs/lora/full/)、[`outputs/lora/full_eval/`](outputs/lora/full_eval/) |
| E · ControlNet 与 cross-attention | [`04_controlnet_demo.ipynb`](04_controlnet_demo.ipynb)、[`06_cross_attention_visualization.py`](06_cross_attention_visualization.py) | [`outputs/controlnet/`](outputs/controlnet/)、[`outputs/attention/`](outputs/attention/) |
| 环境、实验日志与阅读笔记 | [`outputs/autodl_environment.json`](outputs/autodl_environment.json)、[`logs/`](logs/)、[`notes/`](notes/)、[`reading_notes/`](reading_notes/) | 实验环境、实验配置、运行记录及论文阅读笔记 |
| 报告图 | [`make_report_figures.py`](make_report_figures.py) | [`outputs/report/`](outputs/report/)；其他插图直接引用已提交的实验 PNG |

### GitHub Release

| 发布内容 | 说明 |
| --- | --- |
| [Project3 reproducibility v1.0 · `project3-v1.0`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project3-v1.0) | GitHub Release 页面；另含该 tag 对应的自动生成源码归档 |
| [`project3-lora-vangogh-full.safetensors`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project3-v1.0/project3-lora-vangogh-full.safetensors) | 完整训练的 Van Gogh LoRA adapter，6,414,448 bytes（约 6.12 MiB） |
| [`commit.txt`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project3-v1.0/commit.txt) | 发布时对应的代码提交标识 |
| [`repro_manifest.json`](https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project3-v1.0/repro_manifest.json) · [仓库副本](repro_manifest.json) | 复现配置、训练环境、数据说明和 adapter 校验信息 |
