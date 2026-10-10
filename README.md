# Diffusion Models 五项目仓库

本仓库将课程的五个扩散模型项目整理为一个 monorepo。项目代码、配置、实验结果和报告分别保存在 `projects/projectN_*/`；`shared/` 仅放跨项目通用工具。

## 项目状态

| 项目 | 内容与当前结果 |
|---|---|
| [项目 1：DDPM](projects/project1_ddpm/README.md) | 基础、进阶、挑战及改进实验均有实现和记录。固定正式协议下当前最佳为 R5 EMA 0.9999，FID **15.4385**；目标 FID ≤ 15 尚未达到。项目文档未记录完整 pytest 回归已执行。 |
| [项目 2：采样器对比](projects/project2_samplers/README.md) | DDIM、Euler、二阶 DPM-Solver、DDIM 反演及 FID–NFE 对比均有实现和记录；复用项目 1 的模型。 |
| [项目 3：Stable Diffusion](projects/project3_stable_diffusion/README.md) | 手写推理、参数扫描、VAE、LoRA、ControlNet 和 cross-attention 可视化均有实现和记录；LoRA 权重已发布。项目文档记录 AutoDL 测试 5 项通过。 |
| [项目 4：Flow Matching](projects/project4_flow_matching/README.md) | loss、Euler/Heun 采样和 CFG 已用于训练与评估；无条件和条件 DiT-S 均完成 200,000 步训练，v2 结果与推理权重已归档。每种设置为单训练种子，细小 FID 差异需谨慎解读。 |
| [项目 5：VLA Action Diffusion](projects/project5_vla_action_diffusion/README.md) | DDPM、Flow Matching、BC、视觉编码和闭环评估均有实现；[公平重跑结果](projects/project5_vla_action_diffusion/results/fair_seed42/fair_ablation.md)覆盖 10 个配置，每个训练 10,000 步并评估 100 回合。报告中的最高成功率为 **91%**（DDPM，H=8，未屏蔽 padding）。公平重跑仅使用训练种子 42，不能据此估计跨训练种子的方差。 |

项目 1 的 FID 使用其报告中注明的固定评估协议；项目 5 的 91% 属于其公平重跑设置。不同项目或不同协议的指标不可直接比较。项目 5 README 保留课程原始 TODO 教学说明；实现状态以代码为准，实验结论以报告和结果文件为准。

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
.local/                    本地数据、权重、缓存和临时运行文件
  datasets/
  checkpoints/
  hf_cache/
  temporary_runs/
```

实验产物位于各项目自己的 `results/`、`samples/` 和 `logs/` 中；原始运行目录通常位于项目 `runs/`。Git 忽略 `.local/`、数据集、缓存、运行目录及常见大型权重格式。经检查后允许的例外包括项目 3 的 `outputs/lora/` 和项目 5 的 `ckpts/model_final.pt`。报告应记录数据划分、随机种子、软件环境和所用 checkpoint 的 SHA256。

仓库只在根目录保留一个 `.git`，上游代码以普通文件快照导入，不使用 submodule。提交时指定文件路径，避免无差别添加数据或运行产物。

## 安装与常用命令

项目要求 Python ≥ 3.10。每份项目依赖文件都包含 `requirements/base.txt`：

```bash
# 项目 1–2
python -m pip install -r requirements/project1-2.txt

# 项目 3：使用其固定版本环境；GPU 环境按项目文档安装匹配的 PyTorch/CUDA wheel
python -m pip install -r requirements/project3.txt

# 项目 4 或项目 5
python -m pip install -r requirements/project4.txt
python -m pip install -r requirements/project5.txt
```

项目 3 的依赖文件固定了 PyTorch、torchvision 及 Hugging Face 相关包；AutoDL 镜像已提供 CUDA 版 PyTorch 时，不要用不匹配的 wheel 覆盖它。正式运行应记录实际环境版本。各项目专用的训练、评估与复现命令见对应 README。

仓库级检查命令：

```bash
python -m pytest projects/project1_ddpm/tests -v
python -m unittest discover -s projects/project2_samplers/tests -v
python -m projects.project2_samplers.check_compat
python scripts/check_repo.py
```

项目 1 的挑战结果汇总入口：

```bash
python -m projects.project1_ddpm.challenge summarize --output_root runs/challenge
```

## 项目关系与上游版本

课程 starter 材料仓库：[diffusion-models-starter-materials](https://github.com/Qi-StarterTrain/diffusion-models-starter-materials)。表中提交号是导入本仓库时固定的上游 `main` 快照；之后上游更新不会自动覆盖本仓库。同步时应记录迁移并重新检查对应项目。

| 项目 | 上游模板 | 固定提交 |
|---|---|---|
| 1 | [diffusion-project1-ddpm](https://github.com/Qi-StarterTrain/diffusion-project1-ddpm) | `56bd97e27655e1e22db5610877c8bf9e024cff96` |
| 2 | [diffusion-project2-samplers](https://github.com/Qi-StarterTrain/diffusion-project2-samplers) | `774d640f3915c3396e034068beff318fbf421719` |
| 3 | [diffusion-project3-stable-diffusion](https://github.com/Qi-StarterTrain/diffusion-project3-stable-diffusion) | `e07f756042ddd7329292624ad12a73b582867c18` |
| 4 | [diffusion-project4-flow-matching](https://github.com/Qi-StarterTrain/diffusion-project4-flow-matching) | `0e0074f41c168d0f69225e7f1e06ed42e75e911d` |
| 5 | [diffusion-project5-vla-action-diffusion](https://github.com/Qi-StarterTrain/diffusion-project5-vla-action-diffusion) | `381d52b1e826ad3adf447e864c938025cfb55c87` |

项目 2 复用项目 1 的模型、调度器和 checkpoint 格式；项目 4 使用项目 1/2 的标准化评估结果进行对比，不依赖其网络实现；项目 3 独立使用 Hugging Face 预训练模型；项目 5 的基础实验独立运行，Flow Matching 是可选对照。

## 复现权重与下载

大型权重通过 GitHub Releases 分发。下表记录本仓库文档中的主要复现资产；实验设置、完整结果和环境说明以各项目文档及清单为准。

| 项目 | Release 与资产 | 校验信息 |
|---|---|---|
| 1 | [challenge-v1](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)：包含 `cifar10_{linear,cosine}_200ep_seed{42,43,44}_final.pt` 及后续改进实验权重。固定协议最佳 R5 权重为 `r5_uniform_late_decay_final.pt`。 | R5 SHA256：`88eff8c3db7a2c9350d740247755f1d22abea3de0073b2b504b4b77c2ab412bb` |
| 2 | [project2-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project2-v1.0)：`project2-input-ddpm-cifar10-linear-seed44.pt`。 | SHA256：`937853559a1377660f7d4cbd1dd3f7c6cf022aed4ab84e9daa918a6e34541412` |
| 3 | [project3-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project3-v1.0)：LoRA adapter `project3-lora-vangogh-full.safetensors`（6,414,448 B）。基础模型需另从 Hugging Face 获取：`stable-diffusion-v1-5/stable-diffusion-v1-5`。 | SHA256：`82b93c827e18fc4024f3bb03c87163159acc6c1c981db751c572f9f3e0bf9410` |
| 4 | [project4-v2.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-v2.0)：条件与无条件 DiT-S EMA 0.9999 checkpoint，以及 `project4_v2_results_only.tar.gz`。 | 条件：`25666b32e914002778825a02f8e14c89f9f5ff081f9dded9f3e0ad6e0283c5aa`；无条件：`b75cbe3d4f9aeb16c66cb53e4acb03f9c2c6a18421efd945f78475cd2e114571`。结果包校验文件随 Release 提供。 |

以下命令适用于 Linux/AutoDL，从仓库根目录执行。只下载当前实验需要的权重；项目 1 的 R5 是项目 4 对照实验使用的基线。

```bash
mkdir -p .local/checkpoints/project{1,2,3,4}

# 项目 1：正式协议最佳 R5 checkpoint
wget -O .local/checkpoints/project1/r5_uniform_late_decay_final.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/r5_uniform_late_decay_final.pt
echo '88eff8c3db7a2c9350d740247755f1d22abea3de0073b2b504b4b77c2ab412bb  .local/checkpoints/project1/r5_uniform_late_decay_final.pt' | sha256sum -c -

# 项目 2：采样器评估输入
wget -O .local/checkpoints/project2/project2-input-ddpm-cifar10-linear-seed44.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project2-v1.0/project2-input-ddpm-cifar10-linear-seed44.pt
echo '937853559a1377660f7d4cbd1dd3f7c6cf022aed4ab84e9daa918a6e34541412  .local/checkpoints/project2/project2-input-ddpm-cifar10-linear-seed44.pt' | sha256sum -c -

# 项目 3：LoRA adapter（基础模型需另从 Hugging Face 下载）
wget -O .local/checkpoints/project3/project3-lora-vangogh-full.safetensors \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project3-v1.0/project3-lora-vangogh-full.safetensors
echo '82b93c827e18fc4024f3bb03c87163159acc6c1c981db751c572f9f3e0bf9410  .local/checkpoints/project3/project3-lora-vangogh-full.safetensors' | sha256sum -c -

# 项目 4：v2 无条件与条件 EMA checkpoint
wget -O .local/checkpoints/project4/project4-v2-unconditional-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4-v2-unconditional-step200000.pt
wget -O .local/checkpoints/project4/project4-v2-conditional-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4-v2-conditional-step200000.pt
echo 'b75cbe3d4f9aeb16c66cb53e4acb03f9c2c6a18421efd945f78475cd2e114571  .local/checkpoints/project4/project4-v2-unconditional-step200000.pt' | sha256sum -c -
echo '25666b32e914002778825a02f8e14c89f9f5ff081f9dded9f3f0ad6e0283c5aa  .local/checkpoints/project4/project4-v2-conditional-step200000.pt' | sha256sum -c -

# 可选：下载归档的评测与训练结果
wget -O .local/checkpoints/project4/project4_v2_results_only.tar.gz \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4_v2_results_only.tar.gz
echo 'e8ea255f9ae63f666b47bbdab30d494307285d9ba30b7276e248ce800bf8802a  .local/checkpoints/project4/project4_v2_results_only.tar.gz' | sha256sum -c -
```

项目 1 的六个初始 schedule/seed 对照权重可按需下载：

```bash
for asset in \
  cifar10_linear_200ep_seed42_final.pt \
  cifar10_linear_200ep_seed43_final.pt \
  cifar10_linear_200ep_seed44_final.pt \
  cifar10_cosine_200ep_seed42_final.pt \
  cifar10_cosine_200ep_seed43_final.pt \
  cifar10_cosine_200ep_seed44_final.pt; do
  wget -O ".local/checkpoints/project1/$asset" \
    "https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/$asset"
done
```

项目 2–4 的复现清单位于各自目录的 `repro_manifest*.json` 和 `.sha256` 文件；项目 4 的 v2 资产清单及评测口径见 `projects/project4_flow_matching/repro_manifest_v2.json`。不要将数据集、缓存、原始运行目录或未获准的大型 checkpoint 提交到 Git。
