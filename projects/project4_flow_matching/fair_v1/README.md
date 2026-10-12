# Project 4 fair-v1 实验实现

该目录实现 Project 4 fair-v1 的严格对照实验。F-DiT 使用随机初始化 DiT-S 做无条件 Flow Matching，seed 44 训练至 200,000 步；已完成 Euler NFE 2–100、每档 5,000 张图像的 FID 评测。checkpoint 和完整复现归档发布在 GitHub Release [project4-fair-v1.0](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project4-fair-v1.0)。

当前归档覆盖 AutoDL 节点 C 的 F-DiT 任务。fair-v1 其他实验的结果需要在各自完成后单独归档；不要把本次 F-DiT 结果误当成完整实验矩阵。复现流程、FID 表格、文件哈希和下载命令见 [ARTIFACTS.md](ARTIFACTS.md)。

## 获取本次训练与评测产物

克隆仓库后运行：

    bash projects/project4_flow_matching/fair_v1/download_artifacts.sh

脚本从 Release 下载约 498 MiB 的 step-200000 checkpoint 和约 56 MiB 的复现包，校验 SHA-256 后解压。复现包包含本次 fair-v1 源码快照、训练配置与 metadata、训练日志、Euler FID JSON 和日志、十组样图及固定噪声库。CIFAR-10 数据集不包含在归档内，按项目设置说明下载。

## 训练与评测设置

F-DiT 训练 seed 为 44，200,000 optimizer steps，EMA decay 0.9999，BF16，effective batch size 128。评测使用 CIFAR-10 train 前 5,000 张图、5,000 个生成样本、evaluation/noise seed 1001、TorchMetrics FID-2048，Euler NFE 为 2、4、8、10、16、20、32、50、64、100。

结果是单训练种子的一次运行，适用于记录和后续实验复现，不代表统计显著性。checkpoint 的 run metadata 记录了训练时的代码 revision 与环境信息。此归档不包含其它 fair-v1 实验的 checkpoint。

## 目录

- README.md：fair-v1 方案与执行入口
- train.py、evaluate.py、models.py：训练、评测和模型实现
- configs：实验配置
- run_node_c.sh、postprocess_node_c.py：节点 C 执行与结果核验/备份状态更新
- ARTIFACTS.md：本次 F-DiT 结果、Release 附件和校验信息
- download_artifacts.sh：下载、校验并解压 Release 资产
