# my-diffusion-models-starter

项目代码、配置、实验结果和报告分别保存在 `projects/projectN_*/`，`shared/` 保存跨项目通用工具。

## 项目状态

| 项目 | 内容与当前结果 |
|---|---|
| [项目 1：DDPM](projects/project1_ddpm/README.md) | 基础、进阶、挑战及改进实验均有实现和记录。固定正式协议下当前最佳为 R5 EMA 0.9999，FID **15.4385** |
| [项目 2：采样器对比](projects/project2_samplers/README.md) | DDIM、Euler、二阶 DPM-Solver、DDIM 反演及 FID–NFE 对比均有实现和记录；复用项目 1 的模型。 |
| [项目 3：Stable Diffusion](projects/project3_stable_diffusion/README.md) | 手写推理、参数扫描、VAE、LoRA、ControlNet 和 cross-attention 可视化均有实现和记录；LoRA 权重已发布。 |
| [项目 4：Flow Matching](projects/project4_flow_matching/README.md) | loss、Euler/Heun 采样和 CFG 已用于训练与评估；无条件和条件 DiT-S 均完成 200,000 步训练，v2 结果与推理权重已归档。 |
| [项目 5：VLA Action Diffusion](projects/project5_vla_action_diffusion/README.md) | DDPM、Flow Matching、BC、视觉编码和闭环评估均有实现，覆盖 10 个配置，每个训练 10,000 步并评估 100 回合。报告中的最高成功率为 **91%**（DDPM，H=8，未屏蔽 padding）。 |


## 目录与数据约定

```text
requirements/              按项目划分的依赖文件
projects/project1_ddpm/    DDPM
projects/project2_samplers/  # 采样器对比
projects/project3_stable_diffusion/
projects/project4_flow_matching/
projects/project5_vla_action_diffusion/
shared/                    路径、随机种子、校验、元数据和通用评估工具
docs/                      仓库架构、迁移和存储说明
scripts/                   仓库级检查脚本
```

实验产物位于各项目自己的 `results/`、`samples/` 和 `logs/` 中，原始运行目录位于项目 `runs/`。

项目 2 复用项目 1 的模型、调度器和 checkpoint 格式；项目 4 使用项目 1/2 的标准化评估结果进行对比，不依赖其网络实现；项目 3 独立使用 Hugging Face 预训练模型；项目 5 的基础实验独立运行，Flow Matching 是可选对照。

## 复现权重与下载

大型权重通过 GitHub Releases 分发。下表记录本仓库文档中的主要复现材料；实验设置、完整结果和环境说明以各项目文档及清单为准。

| 项目 | Release内容 | 
|---|---|
| 1 | [challenge-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)：包含 `cifar10_{linear,cosine}_200ep_seed{42,43,44}_final.pt` 及后续改进实验权重。固定协议最佳 R5 权重为 `r5_uniform_late_decay_final.pt`。 |
| 2 | [project2-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project2-v1.0)：`project2-input-ddpm-cifar10-linear-seed44.pt`。 | 
| 3 | [project3-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project3-v1.0)：LoRA adapter `project3-lora-vangogh-full.safetensors`（6,414,448 B）。基础模型需另从 Hugging Face 获取：`stable-diffusion-v1-5/stable-diffusion-v1-5`。 |
| 4 | [project4-v2.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0)：条件与无条件 DiT-S EMA 0.9999 checkpoint，以及 `project4_v2_results_only.tar.gz`。 | 
