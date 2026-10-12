# fair-v1 F-DiT 节点 C 完成记录

记录时间：2026-10-12（Asia/Shanghai）。本记录只覆盖节点 C 的 F-DiT 任务，不表示 fair-v1 的其它对照和进阶实验均已完成。

## 训练

- 配置：F-DiT 无条件 Flow Matching；DiT-S（embed dim 384、12 层、6 heads、patch size 2）；seed 44。
- 训练预算：200,000 optimizer steps；batch size 128；BF16；EMA decay 0.9999。
- 环境：AutoDL NVIDIA GeForce RTX 4080 SUPER；Python 3.12.3；PyTorch 2.8.0+cu128；CUDA 12.8。
- 结束记录：step 200,000；loss 0.1781408；累计训练时间 33,414.9 秒。
- Checkpoint：step_200000.pt；522,172,710 bytes；SHA-256 d0bd447c276c08a2188835c827c2f7a5c77eab58c533f99d1544c6e04ef26710。
- 训练 metadata 与 CSV 位于 logs/fair_v1/F-DiT；完整控制台输出也已归档。

## 评估与核验

使用 CIFAR-10 train 索引 0–4,999、5,000 个生成样本、评测/噪声 seed 1001、固定共同噪声库和 TorchMetrics FID-2048。Euler NFE 网格为 2、4、8、10、16、20、32、50、64、100。评估 JSON、完整日志和十张样本网格均已保存并逐项纳入哈希清单。

监控和后处理核验确认评估正常结束、10 个 NFE 点齐全、checkpoint 训练至 step 200,000 且哈希一致。单训练种子的结果只描述本次运行，不用于推断统计显著性。

训练 revision 记录为 ed3357e16904e9c3d4fc5128eddb75aedd4c7d08；Release 另包含节点 C 的监控和后处理脚本。
