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

## 节点 A：D-U、F-U、F-CT 与 F-RF（seed 44）

A 节点所有训练及评测已完成。F-RF 从保存的 step 10,000 检查点恢复并训练到 step 50,000；恢复后检查点保留教师 F-U 的 lineage 哈希 `ce3f58438a2ed29bdbff4e97ba2849dc09959b787bdc533f8fc5cf7c70963b96`。D-U 与 F-U 完成 200,000 步，F-CT 与 F-RF 完成 50,000 步。F-RF 的最终权重、来源关系和评测均已核验。

评估统一使用 TorchMetrics FID-2048；CIFAR-10 train 索引 `0:5000` 作为真实图像，每档生成 5,000 张，batch size 64，generation seed 1001，共同 CPU Gaussian noise bank 的 SHA-256 为 `a81b164722002a7b4649ba8519a1dff2befad6dd015a77aea97e7d28f261793e`。环境为 RTX 4080 SUPER、Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。DDIM、ancestral/no-clip 与四个 FM 结果 JSON 都记录了 checkpoint SHA、步数、EMA、源码 revision 和逐项采样时间。

### D-U：DDIM、ancestral 与 no-clip

DDIM 使用固定噪声，对 2–100 NFE 评估；另测完整 ancestral 1,000 步，以及关闭 denoised clipping 的 DDIM-20。

| DDIM NFE/图 | FID ↓ | 采样时间（秒） |
|---:|---:|---:|
| 2 | 389.6992 | 4.29 |
| 4 | 56.7488 | 7.02 |
| 8 | 29.1369 | 13.32 |
| 10 | 26.2918 | 16.40 |
| 16 | 22.0175 | 26.09 |
| 20 | 19.9765 | 32.62 |
| 32 | 17.9058 | 51.22 |
| 50 | 16.3803 | 78.93 |
| 64 | 16.0098 | 98.92 |
| 100 | 15.4242 | 154.90 |

| 附加评估 | FID ↓ | 采样时间（秒） |
|---|---:|---:|
| Ancestral，1,000 步 | 13.7594 | 1,486.54 |
| DDIM-20，no-clip | 372.2270 | 32.67 |

### F-U：Euler 与 Heun

| NFE/图 | Euler FID ↓ | Euler 采样时间（秒） | Heun FID ↓ | Heun 采样时间（秒） |
|---:|---:|---:|---:|---:|
| 2 | 174.2554 | 3.88 | 404.2330 | 3.98 |
| 4 | 68.3946 | 6.32 | 311.5474 | 6.36 |
| 8 | 31.7916 | 12.09 | 139.7589 | 12.11 |
| 10 | 27.6283 | 14.99 | 109.3425 | 15.00 |
| 16 | 21.9929 | 24.00 | 67.4344 | 24.00 |
| 20 | 20.8186 | 29.76 | 53.9351 | 29.70 |
| 32 | 18.4714 | 47.22 | 32.1175 | 47.28 |
| 50 | 17.3398 | 74.08 | 20.1862 | 74.05 |
| 64 | 16.5395 | 94.39 | 16.9381 | 94.34 |
| 100 | 15.9969 | 147.73 | 14.5048 | 147.57 |

### F-CT 与 F-RF：Euler

| NFE/图 | F-CT FID ↓ | 采样时间（秒） | F-RF FID ↓ | 采样时间（秒） |
|---:|---:|---:|---:|---:|
| 2 | 176.9113 | 4.02 | 20.2793 | 3.95 |
| 4 | 69.8698 | 6.53 | 18.2201 | 6.30 |
| 8 | 33.1324 | 12.10 | 17.2271 | 12.10 |
| 10 | 28.7829 | 14.98 | 17.0783 | 14.97 |
| 16 | 22.9350 | 24.01 | 16.7047 | 24.02 |
| 20 | 21.6280 | 29.71 | 16.6678 | 29.72 |
| 32 | 19.1418 | 47.22 | 16.4586 | 47.22 |
| 50 | 17.9956 | 74.17 | 16.4426 | 74.10 |
| 64 | 17.2232 | 94.45 | 16.3228 | 94.37 |
| 100 | 16.6269 | 147.54 | 16.3683 | 147.58 |

这些数值是单次 seed 44 结果，不构成跨种子显著性结论。FID 并不保证随 NFE 单调下降；表格保留原始实测值。逐项评测输出、日志、样图、A 的训练日志、配置、评测脚本和固定噪声库都包含在 A 复现归档中。

| Release 资产 | 大小（bytes） | SHA-256 |
|---|---:|---|
| `project4-fair-v1-a-d-u-seed44-step200000.pt` | 610,942,975 | `b57e30c02aa03b47e587ced38fd91471ac3dc8e50a11be2056ce20b5370df78e` |
| `project4-fair-v1-a-f-u-seed44-step200000.pt` | 610,942,911 | `ce3f58438a2ed29bdbff4e97ba2849dc09959b787bdc533f8fc5cf7c70963b96` |
| `project4-fair-v1-a-f-ct-seed44-step50000.pt` | 610,941,444 | `8532bc0335214f71271896a33de8459aaa5b103239c9b2780185e8a1e06b9d59` |
| `project4-fair-v1-a-f-rf-seed44-step50000.pt` | 610,941,547 | `0fea3d161b7eb9a9d4396f75dab438e230f9fe1910914aa1e4b4f25167bed5cd` |
| `project4-fair-v1-node-a-seed44-repro.tar.gz` | 64,107,873 | `1f845b46c206f0bb1e242fdb4e1f56d93268d4bd4d904940b279185487d80344` |

复现包内部的 `CHECKPOINT_SHA256SUMS.txt` 列出四个独立 Release checkpoint 的哈希；checkpoint 大文件单独提供，归档包含代码快照、配置、训练日志/metadata、评测 JSON/完整日志/样图、噪声库和来源 revision。训练 metadata 中 D-U 记录 revision `8eb7c8d90b4feb35d9e6a734fdaa538ef919bb99`，FM 检查点记录 `ce4174ae46ad2a0a763fad4b4a4a6330c383fe6b`。A bundle SHA 在本地下载副本、AutoDL 原件及 Release 资产间一致。

## 所有节点的下载与校验

Release：<https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0>。从仓库根目录运行 `bash projects/project4_flow_matching/fair_v1/download_artifacts.sh artifacts/project4-fair-v1 all` 可下载并校验所有节点；将最后参数设为 `A`、`B` 或 `C` 可以只取对应节点。脚本按 `ASSET_SHA256SUMS.txt` 逐项核验 Release 二进制，并解压复现包。CIFAR-10 数据集不在 Release 中，需自行准备。
