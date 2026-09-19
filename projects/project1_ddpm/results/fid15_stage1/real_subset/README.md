# Real-subset FID variance experiment

这是一次不重训的评估诊断。目标是量化 FID 对真实图像子集和生成随机种子的敏感性，不能用来替换 Project 1 的正式评估协议。

## 固定协议

- 数据：CIFAR-10 `train` split。
- 真实图：每组 5,000 张。
- 正式子集：前 5,000 张 `official_first5000`。
- 随机子集：4 组互不重叠的 class-stratified 子集，每类 500 张，子集种子 `20260919`。
- 生成图：每组 5,000 张，生成 seed 为 `44/45/46`。
- 采样：逐步裁剪预测的 `x0`，FID Inception feature=2048。
- 模型：R3 EMA 0.9995、R5 EMA 0.9999。

`real_subset_fid.csv` 保存 30 个单元格的原始分数；`real_subset_summary.json` 保存聚合统计。实验脚本为仓库根目录的 `stage1_real_subset_fid.py`。脚本会缓存每个模型/生成 seed 的 uint8 生成图，因此切换真实子集时不会重复昂贵的反向采样。

## 结果

| 实验 | 15 个单元格均值 | 样本标准差 | 最低 | 最高 | 正式子集 3-seed 均值 |
|---|---:|---:|---:|---:|---:|
| R3 EMA 0.9995 | 15.5096 | 0.1104 | 15.3157 | 15.6866 | 15.5312 |
| R5 EMA 0.9999 | 15.3590 | 0.1270 | 15.1445 | 15.5563 | 15.3716 |

30 组中没有一组低于 15。最低值是 R5、`stratified_1`、生成 seed 46 的 `15.1445`，仍高于阈值 `0.1445`。R5 在正式协议下 seed 44 的结果为 `15.4385`，这是应继续用于正式报告的值。

随机真实子集确实会改变 FID：在同一个 R5、seed 44 条件下，正式子集为 `15.4385`，`stratified_1` 为 `15.2631`，差异为 `-0.1754`；这说明“随机抽 5,000 张”可能带来明显变化，但不能把事后选择的最低子集当成模型质量达标证据。完整 30 组结果仍显示，当前模型没有通过 FID≤15。

## 来源与复现

实验在 AutoDL RTX 4090 上完成，没有训练、没有修改 checkpoint。原始运行使用 detached screen，日志在远端 `logs/fid15_real_subset.log`；本目录的 CSV 和 JSON 是从该完整日志整理出的可审计结果。远端运行时生成图缓存不进入 Git；checkpoint 仍通过仓库 Release 分发。

复现命令示例：

```bash
python -u stage1_real_subset_fid.py \
  --r3_ckpt runs/fid15_stage1_inputs/r3_full_cosine_final.pt \
  --r5_ckpt runs/fid15_r5_late_decay/ckpt/final.pt \
  --r3_ema_decay 0.9995 --r5_ema_decay 0.9999 \
  --data_root ./data \
  --output_dir runs/fid15_stage1/real_subset \
  --cache_dir /root/fid15_stage1_cache \
  --subset_size 5000 --random_subset_count 4 \
  --subset_seed 20260919 --num_samples 5000 \
  --batch_size 64 --gen_seeds 44 45 46
```
