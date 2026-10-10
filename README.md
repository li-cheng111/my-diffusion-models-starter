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

大型权重和复现材料通过 GitHub Releases 分发。截至 **2026-10-11**，仓库共有 7 个 Release；最新版本为 Project 5 公平复跑。历史版本仍可用于复现对应实验，不能与后续协议下的结果混为一谈。

| 项目 / Release | 内容与适用范围 |
|---|---|
| **1 · `challenge-v1`** · 2026-09-13 | 共 20 个资源：CIFAR-10 线性/余弦调度、seed 42/43/44 的 6 个基线权重，以及 R3–R7、学习率筛选和正式训练权重；另含 MNIST 部分权重、中间采样、R4 结果、R5/R6 结果与网格归档。固定正式协议下当前最佳为 `r5_uniform_late_decay_final.pt`（EMA 0.9999，FID **15.4385**）。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1) |
| **2 · `project2-v1.0`** · 2026-09-14 | 采样器基准输入权重 `project2-input-ddpm-cifar10-linear-seed44.pt`（459,538,769 B），并附代码提交号与复现清单。复用 Project 1 的 seed 44 线性调度 checkpoint。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project2-v1.0) |
| **3 · `project3-v1.0`** · 2026-09-14 | 梵高风格 LoRA adapter `project3-lora-vangogh-full.safetensors`（6,414,448 B），并附代码提交号与复现清单。基础模型需另从 Hugging Face 获取：`stable-diffusion-v1-5/stable-diffusion-v1-5`。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project3-v1.0) |
| **4 · `project4-v1.0`** · 2026-09-14（历史版） | 早期 DiT-S Flow Matching 复现权重 `project4-fm-dit-s-step200000.pt`（522,163,965 B），训练 200,000 步；附代码提交号与复现清单。当前结果请使用 v2。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v1.0) |
| **4 · `project4-v2.0`** · 2026-10-07 | 当前 Project 4 归档：条件与无条件 DiT-S 各训练 200,000 步（seed 42、EMA 0.9999），包含 `project4-v2-{conditional,unconditional}-step200000.pt` 和 `project4_v2_results_only.tar.gz`。无条件 Euler 50 NFE 的 FID 为 26.7610，Heun 100 NFE 为 24.0062；条件 20-step Euler、CFG=2 的 FID 为 18.8105。FID 采用 5,000 张真实图和 5,000 张生成图，为单 seed 测量。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0) |
| **5 · `project5-bonus-v1.0.0`** · 2026-09-14（历史 bonus） | 含 `project5-vla-action-diffusion-checkpoint-30c0685.pt`、源码包和复现归档。早期实验：空间 DDPM 成功率 71%、10 步 Euler Flow Matching 57%；双目标 97.0%（mode0 96.2%、mode1 97.9%）；移动干扰物 72%（碰撞 20%、超时 8%）。ResNet18 对照在 5k/15k 步为 3%/2%。这些是 bonus 实验，不是后续公平复跑结果。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-bonus-v1.0.0) |
| **5 · `project5-fair-seed42-v1.0.0`** · 2026-10-08（最新） | 同一训练 seed 42 下比较 10 个配置，每个训练 10,000 步，并在固定测试 seed 10000–10099 上评估。附件为 `project5-fair-seed42-checkpoints.tar.gz`（60 个快照）、`project5-fair-seed42-results.tar.gz`（指标、rollout 与汇总）、源码包 `project5-vla-action-diffusion-source-14a93fd.tar.gz` 及 `SHA256SUMS`。最高成功率为 H=8、未屏蔽 padding 的 DDPM **91%**。单训练 seed 无法估计跨 seed 方差。[Release 页面](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project5-fair-seed42-v1.0.0) |

使用 GitHub CLI 下载指定版本的全部附件：

```bash
gh release download project5-fair-seed42-v1.0.0 \
  --repo li-cheng111/my-diffusion-models-starter \
  --dir releases/project5-fair-seed42
```

该版本含 `SHA256SUMS`，可在 Linux、macOS、Git Bash 或 WSL 的下载目录中运行 `sha256sum -c SHA256SUMS` 校验归档。Project 2、3 和 Project 4 v1 附有 `repro_manifest.sha256`；Project 4 v2 为结果归档附带 `.sha256` 文件。其他 Release 附件的 SHA-256 摘要见各 Release 页面的资产信息；Project 1 的 R5 权重摘要为 `88eff8c3db7a2c9350d740247755f1d22abea3de0073b2b504b4b77c2ab412bb`。

各项目的训练协议、完整指标、复现步骤和限制以对应项目文档及 Release 附件中的清单为准。
