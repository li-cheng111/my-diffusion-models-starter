# F-DiT seed 44 归档记录

本页记录 AutoDL 节点 C 完成的 F-DiT 训练和 Euler 评估，方便后续节点直接从 GitHub 恢复文件。实验在单个训练种子上完成，数值用于本次运行的可复现记录，不构成跨种子统计结论。

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
