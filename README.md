# my-diffusion-models-starter

项目代码、配置、实验结果和报告分别保存在 `projects/projectN_*/`，`shared/` 保存跨项目通用工具。

## 项目状态

| 项目 | 内容与当前结果 |
|---|---|
| [项目 1：DDPM](projects/project1_ddpm/README.md) | 基础、进阶、挑战、改进、学习率筛选和采样诊断均有记录。R5 EMA 0.9999 在 seed 44/45/46 测试中的 FID 为 15.4385 / 15.4678 / **15.2086**（均值 15.3717），当前最低值未达到 FID ≤ 15。
| [项目 2：采样器对比](projects/project2_samplers/README.md) | DDPM、DDIM、Euler、二阶 DPM-Solver、DDIM 反演及 FID–NFE 对比均有实现和记录，复用项目 1 的模型。基线 DPM-Solver-2 在 19 NFE 下 FID 为 21.074；R5 权重的 250 NFE DDIM FID 为 18.362。 |
| [项目 3：Stable Diffusion](projects/project3_stable_diffusion/README.md) | SD 1.5 手写推理、CFG/步数/sampler 扫描、VAE 解剖、LoRA、Canny ControlNet 和 cross-attention 可视化均已完成，LoRA 训练 800 步，adapter 已发布。 |
| [项目 4：Flow Matching](projects/project4_flow_matching/README.md) | 包含 U-Net 与 DiT-S 的 Flow Matching、Euler/Heun、CFG、Reflow 和时间采样对照。
| [项目 5：VLA Action Diffusion](projects/project5_vla_action_diffusion/README.md) | 比较 BC、DDPM、Flow Matching、动作块长度、padding mask、视觉编码和环境泛化。seed 42 的 10 配置矩阵每项训练 10,000 步、评估 100 回合，最高成功率为 **91%**（DDPM，H=8，未屏蔽 padding）。|


## 目录与数据约定

```text
requirements/              按项目划分的依赖文件
projects/project1_ddpm/    
projects/project2_samplers/  
projects/project3_stable_diffusion/
projects/project4_flow_matching/
projects/project5_vla_action_diffusion/
shared/                    路径、随机种子、校验、元数据和通用评估工具
docs/                      仓库架构、迁移和存储说明
scripts/                   仓库级检查脚本
```

实验产物位于各项目自己的 `results/`、`samples/` 和 `logs/` 中，原始运行目录位于项目 `runs/`。

项目 2 复用项目 1 的模型、调度器和 checkpoint 格式；项目 4 使用项目 1和2 的标准化评估结果进行对比，；项目 3 独立使用 Hugging Face 预训练模型；项目 5 的基础实验独立运行，Flow Matching 是可选对照。

## 复现权重与下载

大型权重、结果归档和复现材料通过 GitHub Releases 分发，历史版本仍可用于复现对应实验。

| 项目 |  Release内容 |
|---|---|
| **项目 1：DDPM** | `challenge-v1` 收录 R3–R7、学习率筛选及原挑战档权重与结果；`schedule50-seed42-v1`、`schedule50-seed43-v1`、`schedule50-seed44-v1` 分别保存三个训练 seed 的 50 epoch 调度对比；最新的 `schedule200-seed43-v1` 保存 seed 43 线性/余弦续训至 200 epoch 的权重、日志、评估和校验清单。<br>[challenge-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1) · [schedule50-seed42-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed42-v1) · [schedule50-seed43-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed43-v1) · [schedule50-seed44-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule50-seed44-v1) · [schedule200-seed43-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/schedule200-seed43-v1) |
| **项目 2：采样器对比** | `project2-v1.0` 发布采样器基准输入权重 `project2-input-ddpm-cifar10-linear-seed44.pt`，并附代码提交号与复现清单。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project2-v1.0) |
| **项目 3：Stable Diffusion** | `project3-v1.0` 发布完整训练的梵高风格 LoRA adapter `project3-lora-vangogh-full.safetensors`（6,414,448 bytes），并附代码提交号、复现清单和校验信息。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project3-v1.0) |
| **项目 4：Flow Matching** | `project4-v2.0` 发布 seed 42、EMA 0.9999 的条件与无条件 DiT-S 200,000 步 checkpoint，以及结果 JSON、样图、训练 CSV 和日志；`project4-fair-v1.0` 发布 fair-v1 完整 U-Net/DiT-S 对照权重和复现包。<br>[project4-v2.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0) · [project4-fair-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0)  |
| **项目 5：VLA Action Diffusion** | `project5-fair-seed42-v1.0.0` 发布 10 个配置矩阵的 checkpoint、结果/复评归档和源码；`project5-simultaneous-multimodal-v1.0.0` 发布同观测双目标演示与复现材料。<br>[fair-seed42](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-fair-seed42-v1.0.0) · [simultaneous-multimodal](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-simultaneous-multimodal-v1.0.0)  |

各项目的训练协议、完整指标、复现步骤和限制请以对应项目的README实验报告文档说明及 Release 附件中的清单为准。

