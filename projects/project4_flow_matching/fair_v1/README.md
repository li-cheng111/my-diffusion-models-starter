# Project 4 fair-v1 实验实现

该目录实现 Project 4 fair-v1 的单种子对照实验。实验矩阵分布在三个 AutoDL 节点：A 运行 D-U、F-U、F-CT、F-RF；B 运行 F-C、F-LN；C 运行 F-DiT。训练 seed 均为 44。GitHub Release [project4-fair-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0) 已有 B、C 的完整复现归档和 A 的最终 checkpoint；A 的评测结果及完整复现包仍在生成，完成后会补入同一 Release。

各节点评测覆盖：A 计划运行 DDIM、ancestral、no-clip、Euler 与 Heun；B 的条件 CFG 和 logit-normal 时间采样及 C 的 DiT-S Euler 已完成。每项结果和实验边界见 [ARTIFACTS.md](ARTIFACTS.md)。所有表格都是单次单种子运行，不代表跨种子统计显著性。

## 下载训练和评测产物

脚本参数为 `bash projects/project4_flow_matching/fair_v1/download_artifacts.sh [输出目录] [C|A|B|all]`。省略参数时保持兼容，仅下载 C 节点 F-DiT。下载完整矩阵：

    bash projects/project4_flow_matching/fair_v1/download_artifacts.sh artifacts/project4-fair-v1 all

也可以单独指定 `A`、`B` 或 `C`。当前 A 范围下载四个最终 checkpoint；B、C 范围还会下载完整复现包。脚本按仓库内的 `ASSET_SHA256SUMS.txt` 校验每个 Release 文件，再解压已完成节点的源码、训练日志、评测 JSON、日志、样图和固定噪声库。A 的评测归档发布后，A 范围也会包含该归档。CIFAR-10 数据集不在归档中，按项目设置说明准备。

## 训练与评测设置

训练使用 P1 U-Net 完成 D-U、F-U、F-C、F-LN、F-CT、F-RF，并使用 DiT-S 完成 F-DiT；配置、步数、求解器和评测 NFE 见对应章节与原始 JSON。评测采用 CIFAR-10 train 前 5,000 张图、每档 5,000 个生成样本、seed 1001 和 TorchMetrics FID-2048。各节点使用哈希相同的固定噪声库，文件 SHA-256 为 `a81b164722002a7b4649ba8519a1dff2befad6dd015a77aea97e7d28f261793e`。

每个 checkpoint 的 metadata 保留源码 revision 与运行环境。数据集本身不包含在 Release 中。

## 目录

- README.md：fair-v1 方案与执行入口
- train.py、evaluate.py、models.py：训练、评测和模型实现
- configs：实验配置
- `run_core_pair.sh`、`run_remaining.sh`、`resume_node_a_frf.sh`、`run_eval_node_a.sh`、`run_node_b.sh`、`run_node_c.sh`：节点运行与恢复入口
- postprocess_node_c.py：节点 C 结果核验与备份状态更新
- ARTIFACTS.md：完整实验矩阵、FID 表、Release 附件和校验信息
- ASSET_SHA256SUMS.txt：最终 checkpoint 与复现包哈希
- download_artifacts.sh：按节点下载、校验并解压 Release 资产
