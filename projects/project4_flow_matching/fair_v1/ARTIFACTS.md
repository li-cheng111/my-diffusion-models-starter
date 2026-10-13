# Project4 fair-v1 实验归档记录

本页记录 Project4 fair-v1 单种子实验在 AutoDL 节点 A、B、C 上的训练、评估和归档状态，方便后续直接从 GitHub 恢复文件。实验使用单个训练种子，数值用于本次运行的可复现记录，不构成跨种子统计结论。

## 节点 C：F-DiT seed 44

## 设置与结果

- 训练：DiT-S Flow Matching，无条件模式，训练 seed 44，200,000 步，EMA 0.9999，BF16，batch size 128。
- 环境：AutoDL NVIDIA GeForce RTX 4080 SUPER；Python 3.12.3；PyTorch 2.8.0+cu128；CUDA 12.8。
- FID：TorchMetrics FID-2048；真实图像为 CIFAR-10 train 索引 0–4,999；每档生成 5,000 张；evaluation/noise seed 1001；固定共同噪声库。
- 源码：训练 metadata 记录的 Git revision 为 ed3357e16904e9c3d4fc5128eddb75aedd4c7d08。Release 归档包含训练源码及之后新增的节点 C 监控、结果核验脚本。

| Euler NFE/图 | FID ↓ | 采样时间（秒） |
|---:|---:|---:|
| 2 | 294.5080 | 3.90 |
| 4 | 138.9369 | 6.74 |
| 8 | 61.0893 | 13.33 |
| 10 | 48.8798 | 16.68 |
| 16 | 33.2682 | 26.65 |
| 20 | 28.9317 | 33.52 |
| 32 | 30.2821 | 53.22 |
| 50 | 31.0744 | 82.93 |
| 64 | 31.7211 | 106.48 |
| 100 | 31.7707 | 166.95 |

原始结果在 projects/project4_flow_matching/results/fair_v1/F-DiT/euler.json，完整评测日志和样图在同目录。训练曲线、控制台日志与 run metadata 在 projects/project4_flow_matching/logs/fair_v1/F-DiT。

## GitHub Release 资产

Release: https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0

- project4-fair-v1-fdit-seed44-step200000.pt：完整 step-200000 checkpoint，522,172,710 bytes；SHA-256：d0bd447c276c08a2188835c827c2f7a5c77eab58c533f99d1544c6e04ef26710
- project4-fair-v1-fdit-seed44-repro.tar.gz：源码与复现文件归档，58,510,817 bytes；SHA-256：515997d69d8401917e7294d960f0df43e76174ba5dbce621ba22fd6271727072

复现归档内 SHA256SUMS 逐文件列出哈希。固定噪声库 SHA-256 为 a81b164722002a7b4649ba8519a1dff2befad6dd015a77aea97e7d28f261793e。

## 从 GitHub 获取

克隆仓库后，在仓库根目录执行：

    bash projects/project4_flow_matching/fair_v1/download_artifacts.sh

也可用 GitHub CLI 手动获取两个 Release 资产：

    gh release download project4-fair-v1.0 --repo li-cheng111/my-diffusion-models-starter --pattern 'project4-fair-v1-fdit-seed44-*'

数据集本身不在 Release 中；按 fair_v1 README 的 CIFAR-10 设置下载。checkpoint 可以直接用于采样和后续比较，复现归档中的固定噪声库用于保持评测输入一致。

## 节点 B：F-C 与 F-LN（seed 44）

B 节点两组训练均完成 200,000 步；训练 metadata 记录源码 revision `ed3357e16904e9c3d4fc5128eddb75aedd4c7d08`、PyTorch 2.8.0+cu128、RTX 4080 SUPER。评估均使用 CIFAR-10 train 索引 0–4,999、每档生成 5,000 张、generation seed 1001，以及共同噪声库（SHA-256：`a81b164722002a7b4649ba8519a1dff2befad6dd015a77aea97e7d28f261793e`）。日志确认各档均生成完 5,000 张；原始 JSON、完整日志和每档样图已经复制到仓库。

F-C 使用 Euler、CFG NFE 20/branch（有效模型调用数 40），FID 随 CFG scale 如下：

| CFG scale | FID ↓ | 采样时间（秒） |
|---:|---:|---:|
| 1 | 18.5864 | 63.36 |
| 2 | 18.3478 | 61.89 |
| 3 | 23.0368 | 61.78 |
| 5 | 33.3663 | 61.08 |
| 7.5 | 46.1879 | 62.40 |

F-LN 使用无条件 Euler：

| NFE/图 | FID ↓ | 采样时间（秒） |
|---:|---:|---:|
| 2 | 148.3074 | 3.90 |
| 4 | 59.5022 | 6.39 |
| 8 | 31.6195 | 12.27 |
| 10 | 28.1830 | 15.20 |
| 16 | 22.7265 | 24.47 |
| 20 | 21.6078 | 30.28 |
| 32 | 18.9864 | 48.35 |
| 50 | 17.7216 | 75.67 |
| 64 | 16.8549 | 96.67 |
| 100 | 16.2048 | 150.73 |

文件位置：`results/fair_v1/F-C/cfg.json`、`results/fair_v1/F-LN/euler.json`，对应 `.log` 和样图同目录；训练曲线、console log 与 metadata 在 `logs/fair_v1/F-C`、`logs/fair_v1/F-LN`。F-C checkpoint SHA-256 为 `0953f923b183d98dc422508909297472fab62826d89efa2f64de7adf87cfaf92`，F-LN 为 `797f1ca10a66aa37f3be785f1d4c7d8bf369a91b405cbc9ec5708e68213d425c`。B 节点复现包 `project4-fair-v1-node-b-seed44-repro.tar.gz` 为 58,919,911 bytes，SHA-256：`79caaec8cb920155fb0e279a152ffb4a679aaef1bdf0419399647a374ce2dec2`；按仓库根目录结构打包，包含源码快照、训练日志、metadata、全部 B 评测 JSON/日志/样图、固定噪声库和复评命令。两个最终 checkpoint 作为独立 Release 文件保存。

## 节点 A：checkpoint 已归档，评测进行中

A 的 D-U、F-U、F-CT checkpoint 与 F-RF 最终 checkpoint 已从实例取回并核验 SHA-256。F-RF 从 step 10,000 的恢复点续训到 step 50,000；checkpoint 中已恢复其教师 F-U 的 lineage 哈希（`ce3f58438a2ed29bdbff4e97ba2849dc09959b787bdc533f8fc5cf7c70963b96`）。训练目标已完成，但 A 的评测套件尚在运行，因此本次 Release 先上传 A 的四个最终 checkpoint；评测 JSON、日志、样图和 A 复现包将在评测及归档完成后补入。

| Release checkpoint | 大小（bytes） | SHA-256 |
|---|---:|---|
| `project4-fair-v1-a-d-u-seed44-step200000.pt` | 610,942,975 | `b57e30c02aa03b47e587ced38fd91471ac3dc8e50a11be2056ce20b5370df78e` |
| `project4-fair-v1-a-f-u-seed44-step200000.pt` | 610,942,911 | `ce3f58438a2ed29bdbff4e97ba2849dc09959b787bdc533f8fc5cf7c70963b96` |
| `project4-fair-v1-a-f-ct-seed44-step50000.pt` | 610,941,444 | `8532bc0335214f71271896a33de8459aaa5b103239c9b2780185e8a1e06b9d59` |
| `project4-fair-v1-a-f-rf-seed44-step50000.pt` | 610,941,547 | `0fea3d161b7eb9a9d4396f75dab438e230f9fe1910914aa1e4b4f25167bed5cd` |

The shared checksum manifest `ASSET_SHA256SUMS.txt` lists the current Release checkpoints and completed B/C reproduction bundles. The AutoDL A instance must remain available until its evaluation suite and self-contained archive finish.
