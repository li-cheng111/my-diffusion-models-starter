# 调试日志

> 统一版方法、实验结果和根因分析见 [`PROJECT1_COMPLETE_REPORT.md`](PROJECT1_COMPLETE_REPORT.md)；本文件仅保留逐条调试证据。

## 条目 1——AutoDL 克隆 GitHub 仓库

- 状态：已解决
- 现象：启用 AutoDL 网络加速后，`git clone` 返回 HTTP 503。
- 假设：GitHub 或加速代理暂时不可用。
- 验证：改用浅克隆和独立目标目录重试后，仓库成功克隆。
- 修复：使用 `--depth 1` 重试；不需要修改代码或数据集。
- 经验：AutoDL 学术加速有帮助但不保证稳定，应保留重试或备用下载路径。

## 条目 2——AutoDL SSH 认证

- 状态：已解决
- 现象：第一次使用公钥登录时返回 `Permission denied (publickey,password)`。
- 假设：当前 AutoDL 实例尚未登记该公钥。
- 验证：同一主机和端口的密码认证成功，远端实例和仓库均可访问。
- 修复：在获得授权的会话中使用实例登录凭据，不修改训练代码。
- 经验：AutoDL 控制台的密钥登记必须与当前实例和端口一致；绝不把密码保存到项目文件。

## 条目 3——MNIST 基线

- 状态：已完成
- 现象：训练、采样和 EMA FID 评估过程中没有运行时错误。
- 假设：当前基线配置适合 RTX 5090 环境。
- 验证：训练达到第 23,400 步，生成最终 checkpoint 和 loss 曲线，并完成 5,000 样本的 FID 评估。
- 修复：无需修复。
- 经验：该基线可在记录的 AutoDL 环境中复现，应将配置和 seed 与实验产物一起保留。

## 条目 4——CIFAR-10 数据准备

- 状态：已解决
- 现象：原始 CIFAR-10 下载地址在 AutoDL 上速度约为 10–55 KB/s。
- 假设：瓶颈在 Toronto 外部下载地址，而不在 GPU 或项目代码。
- 验证：公共 Hugging Face 镜像成功下载相同的 170,498,071 字节压缩包，MD5 与 `c58f30108f718f92721af3b95e74349a` 一致。
- 修复：从镜像继续下载，并将验证过的文件保存在 `data/cifar-10-python.tar.gz`。
- 经验：训练前验证数据字节，并将数据文件放在 AutoDL 数据盘。

## 条目 5——CIFAR-10 进阶实验

- 状态：已完成
- 现象：200 轮训练和训练后评估均未出现运行时错误。
- 验证：训练在 98.2 分钟内达到第 78,000 步；在 5,000 张训练图上，EMA FID 为 19.2879，raw FID 为 28.7464。
- 修复：稳定性方面无需修复。FID 未达到 15 的目标，因此如实记录结果。
- 经验：EMA 明显改善了本次运行的 FID，但运行稳定和达到目标指标是两个独立的验收条件。

## 条目 6——CIFAR-10 挑战档矩阵

- 状态：已完成
- 现象：需要在相同模型、优化器、训练预算和评估协议下比较 linear/cosine 两种 schedule，并覆盖 seed 42、43、44。
- 验证：六组结果均已写入 `results/challenge/results.csv` 和 `results/challenge/summary.md`；每组均有 EMA/raw FID 文件和最终 EMA 样本网格。
- 结果：linear EMA FID 为 `19.2926 ± 0.3357`，cosine EMA FID 为 `137.5132 ± 8.0166`；在本实验协议下 linear 更好，但两者均未达到 FID ≤ 15。
- 大文件处理：六个最终 checkpoint 和中间样本未进入普通 Git，已上传到 `challenge-v1 Release`；MNIST 临时产物也以 Release 压缩附件保存。
- 经验：三组 seed 只能提供有限的方差观察，不能将本矩阵结果外推为 schedule 的普适结论；50 轮控制组本次未运行。

## 条目 9——30 epoch 学习率粗筛

- 状态：已完成，作为正式训练前的筛选实验；没有宣称达到 FID≤15。
- 实施：固定完整 U-Net、cosine beta、epsilon prediction、seed44、batch128 和 1,000 steps
  warmup，依次测试 `5e-5、8e-5、1.2e-4、1.6e-4、2e-4、2.5e-4、3e-4`，每组 30 epoch、
  11,730 steps。
- 结果：使用 EMA0.9999、裁剪 x0、1,000 对 1,000 的快速 FID，结果分别为
  `238.0244、263.5649、291.7750、284.0970、307.1943、330.7679、323.4893`。
- 验证：7/7 任务完成；训练日志没有 OOM、NaN 或 AMP 异常。各组末尾 loss 均约 0.052，
  但 loss 排名与 FID 排名不一致。
- 结论：本预算下优先保留 LR05 和 LR08；其余候选暂不进入完整 200 epoch。快速 FID 的
  样本数较小，且没有复刻 R5 后期学习率衰减，因此必须用正式协议复核。
- 产物：轻量配置、日志、loss、样本网格和 FID 文件进入 `results/lr_screen_30ep/`；
  LR05 最终 checkpoint 进入 `challenge-v1 Release`。AutoDL 原始粗筛目录保留，待确认
  Release 与 GitHub 资产完整后再清理中间 checkpoint。

## 条目 7——FID≤15 续训与采样消融

- 状态：已完成，目标未达标
- 目的：验证在固定 linear/seed44 协议下继续训练到 100,000 次有效更新，以及裁剪预测
  $x_0$ 是否能降低 FID。
- 实施：从原始约 78k 步 checkpoint 独立续训，新增 22,000 次有效更新；使用独立输出目录，
  每 5,000 步保存 checkpoint，并分别运行未裁剪和裁剪预测 $x_0$ 的 5,000 样本 EMA FID。
- 结果：原始 checkpoint 为 18.9577（未裁剪）/18.9715（裁剪）；100k checkpoint 为
  18.1502（未裁剪）/18.1490（裁剪）。
- 结论：续训带来约 0.81 FID 改善；裁剪在最终模型上仅改善 0.0012，效果很小；最终仍高于
  FID 15，不能报告为达标。
- 运行环境：AutoDL NVIDIA GeForce RTX 4090，续训 28.5 分钟；未观察到 NaN、AMP 跳步或
  训练异常。AutoDL 断联后重新登录，checkpoint、loss history 和评估结果均已保留。
- 经验：评估脚本当前只在 5,000 张样本全部生成后写入 FID，因此实时监控应结合 screen、
  Python 进程和 GPU 利用率；不能把暂时为空的评估日志误判为卡死。

## 条目 8——FID≤15 v2：R2 checkpoint 恢复与完整对照

- 状态：训练、primary 评估和 EMA decay 对比均已完成；目标未达标，固定协议下最低 FID 为 EMA 0.9995 的 `15.5340`。
- 实施：在 RTX 4090 上按严格 200 epoch 运行三个独立变体：完整 skip-connection U-Net、
  linear/cosine beta 对照，以及 linear beta 的 cosine-LR 尾段；每组保存 EMA 0.999、0.9995、
  0.9999，并以 seed44、5,000 对 5,000 train split 评估未裁剪和裁剪 x0。
- 结果：R1 为 `18.8045/18.8144`，R2 为 `19.4615/19.4633`，R3 为
  `162.8797/16.1686`（顺序均为未裁剪/裁剪）。primary 最好为 R3 裁剪的 `16.1686`。
- 现象：R2 原始 `final.pt` 在前一次训练结束时因 AutoDL 数据盘空间不足而损坏；有效的
  `step_070000.pt` 仍可加载。扩容后从该 checkpoint 恢复到 step `78000`，新 checkpoint
  大小约 916 MB，并通过 `torch.load` 检查 `global_step=78000、epoch=200`。
- 监控：AutoDL SSH 控制连接断开会使本地转发失效，但 detached screen 中的训练/评估通常
  可以继续；本次曾发现 dead screen，随后重新启动监控服务。恢复训练期间监控还修正为优先
  使用当前 runner 日志，避免旧 loss history 的末尾 step 遮住恢复进度。
- 经验：cosine beta 的未裁剪反向采样会产生极端轨迹，逐步裁剪 x0 能把 FID 从 `162.8797`
  降至 `16.1686`，但仍未达到 15；线性 beta 下裁剪影响约 `0.002`，cosine-LR 尾段也没有
  带来收益。EMA 多 decay 对比显示 EMA 0.9995=`15.5340` 最优，仍比目标高 `0.5340`，
  因此不能把 primary 的 EMA 0.9999 结果误认为本轮最优或达标。

## 条目 9——FID≤15 v2：EMA 补充评估与采样公式验证

- 状态：已完成。
- 固定协议：R1、R2 和 R3 历史点均使用 `train` split 前 5,000 张真实图、5,000 张生成图、
  seed `44`、batch size `64`、裁剪预测 $x_0$；R1/R2 额外比较 EMA `0.999` 和 `0.9995`。
- 已完成结果：R1 为 `18.9792/18.5210`，R2 为 `20.3343/20.1602`，顺序为 EMA
  `0.999/0.9995`。R3 历史 EMA 0.9995 FID 依次为 10k=`40.9638`、20k=`24.6445`、
  30k=`19.8270`、40k=`17.9806`、50k=`16.9291`、60k=`16.4440`、70k=`15.9460`；
  最终 checkpoint 为 `15.5340`。未发现早于最终步数的更低值，最佳 R3 EMA 0.9995 裁剪
  网格已单独生成。
- 采样验证：使用固定 $x_t$、固定噪声预测和 `t=999,998,0`，生产 `p_sample` 与独立
  posterior mean、variance、裁剪路径逐项对照。cosine 最大误差 `5.7964e-7`，linear 最大
  误差 `1.2450e-4`，均通过 `2e-4` 容差。
- 诊断：新增 median/P95/P99/max 统计。cosine 无裁剪在 `t=999` 的 `pred_x0` 为
  `90.15/280.88/385.49/964.98`，裁剪在 `t=0` 的 P95/P99 约为 `0.94/0.99`；
  这些结果表明极端幅值是低 alpha_bar 下 epsilon 误差放大的轨迹现象，不能单独作为 schedule
  公式错误的证据。

## 条目 10——R5/R6 最终复核

- 状态：训练和固定协议评估均已完成；FID≤15 未达标。
- R5：uniform epsilon MSE、late cosine LR，训练约 155.0 分钟；EMA 0.999/0.9995/0.9999/raw
  分别为 `16.3823/16.2361/15.4385/16.1927`。
- R6：Min-SNR-$\\gamma=5$、其余设置与 R5 相同，训练约 153.1 分钟；EMA 0.999/0.9995/0.9999/raw
  分别为 `17.2559/16.9445/15.8562/17.8510`。
- 固定协议：CIFAR-10 `train` split 前 5,000 张真实图、5,000 张生成图、seed 44、batch size 64、
  裁剪预测 $x_0$。
- 结论：最佳 R5 EMA 0.9999 为 `15.4385`，距离目标差 `0.4385`；R6 最佳比 R5 高 `0.4177`，
  当前 Min-SNR 设定没有收益。
- 经验：AutoDL 断联期间训练和评估均通过 detached screen 保留；恢复评估时重新生成 R6 的四组
  EMA/raw FID，最终结果写入 `results/fid15_final/`，没有使用中断前可能不完整的临时输出。

## 条目 11——Stage 1 FID 稳定性与 timestep 诊断

- 状态：已完成；不包含训练或 checkpoint 修改。
- 远程执行：AutoDL RTX 4090；原 screen 在实例重启后死亡，但日志保留了 R3 三个 seed 和 R5
  seed44。随后使用可断点参数只补跑 R5 seed45/46，避免重复已完成的 FID。
- FID：R3 EMA 0.9995 为 `15.5340/15.6575/15.4022`（seed44/45/46，均值 `15.5312`）；
  R5 EMA 0.9999 为 `15.4385/15.4678/15.2086`（均值 `15.3717`）。两组 range 分别为
  `0.2553/0.2592`，没有一次低于 15。
- 校准：real train 前 5,000 张与后 5,000 张的 FID 为 `10.2039`。归一化到 uint8 的范围、
  均值和标准差一致，没有发现明显输入转换错误。
- timestep：R3/R5 的 train/test 曲线基本一致；cosine `t=999` 的 `alpha_bar≈2.43e-9`，
  因此很小的 epsilon 误差会把 x0 反推误差放大到约 94，99.3% 像素越界。该现象与低信噪比
  端点的数值条件有关，不能单独证明 schedule 公式错误；生产采样的 x0 clipping 仍是必要的。
- 结论：随机 seed 可造成约 0.26 的 FID range，但不足以证明稳定 FID≤15；当前主要瓶颈更
  可能在中高噪声段的建模误差、prediction target/loss weighting 与有限训练预算的组合。
- 结果文件：`results/fid15_stage1/`，脚本为 `stage1_diagnostics.py`。

## 条目 12——真实子集 FID 方差诊断

- 状态：已完成；不包含训练或 checkpoint 修改。
- 目的：检验真实图像子集和生成随机种子是否足以造成约 0.4 的 FID 差距。
- 方法：R3 EMA 0.9995、R5 EMA 0.9999；正式前 5,000 张真实图加 4 组每类 500 张的
  class-stratified 子集；生成 seed 44/45/46；每组 5,000 对 5,000 图，裁剪预测 `x0`。
- 结果：R3 的 15 格均值/标准差为 `15.5096/0.1104`，R5 为 `15.3590/0.1270`；
  全部 30 格的最低值为 R5、`stratified_1`、seed46 的 `15.1445`，没有一组低于 15。
- 正式值：R5 在正式前 5,000 张和 seed44 下仍为 `15.4385`。R5 同一 seed44 换到
  `stratified_1` 为 `15.2631`，说明真实子集会影响 FID，但不能事后挑选最低子集作为达标
  证据。
- 结论：随机抽样确实能改变单次 FID，且变化量可以达到约 `0.18`（个别组合相对正式值）；
  但当前 30 格矩阵没有稳定跨过 15，因此不能把未达标归因于单一真实子集。正式报告继续
  使用固定协议，完整记录见 `results/fid15_stage1/real_subset/`。

## 条目 13——R7 v-prediction

- 状态：训练、四组 EMA/raw 评估和最佳样本网格均已完成；FID≤15 未达标。
- 训练：完整 U-Net、cosine beta、v-prediction、200 epoch、78,000 steps，RTX 4090，
  训练耗时 156.6 分钟。
- 固定协议：CIFAR-10 train split 前 5,000 张真实图、5,000 张生成图、seed 44、batch 64、
  每一步裁剪预测 `x0`。
- FID：EMA 0.999=`19.0443`，EMA 0.9995=`18.8484`，EMA 0.9999=`17.4808`，raw=`19.2896`。
- 结论：R7 最佳值比 R5 EMA 0.9999 的 `15.4385` 高 `2.0423`。单独切换 v-prediction
  没有改善结果，后续应回到 R5 epsilon-prediction 基线进行单变量消融。
- 产物：结果目录为 `results/fid15_r7_v_prediction/`；大型 checkpoint 不进入普通 Git，
  通过 GitHub Release 提供。

## 条目 14——LR=5e-5 完整 200 epoch 复核

- 状态：已完成，目标未达标。
- 实施：完整 U-Net、cosine beta、uniform epsilon MSE、seed44、200 epoch、78,000 steps、
  warmup 5,000，step 58,500 后进行 late cosine decay；峰值学习率 `5e-5`，最小学习率
  `5e-6`，EMA bank 为 `0.999/0.9995/0.9999`。
- 正式评估：5,000 张 train split 真实图、5,000 张生成图、seed44、逐步裁剪预测 x0。
- 结果：EMA0.9990=`17.4472`，EMA0.9995=`17.3407`，EMA0.9999=`16.5231`，raw=`18.6642`。
- 对比：R5 EMA0.9999=`15.4385`，因此 LR=5e-5 高 `1.0846`；低学习率没有改善 R5。
- 异常记录：step 76,471 附近有一次 AMP optimizer step skip，训练随后正常完成；没有
  OOM、NaN 或 checkpoint 损坏。
- 产物：轻量结果进入 `results/fid15_lr5e5_full/`，最终 checkpoint 进入
  `challenge-v1 Release`。

## 条目 15——30→50 epoch 学习率续训粗筛

- 状态：已完成，目标 FID≤15 尚未被正式协议验证。
- 实施：从 30 epoch 的 LR `2.5e-4` 和 `3.0e-4` `final.pt` 继续到 50 epoch；每组从
  `11,730` 步到 `19,550` 步，新增 `7,820` 个有效更新。
- 固定设置：CIFAR-10 train split、完整 U-Net、cosine beta、epsilon prediction、
  uniform loss、seed44、batch128、EMA bank、逐步裁剪 `x0`。
- 快速评估：1,000 对 1,000、EMA0.9999、seed44；LR2.5e-4=`193.6462`，LR3.0e-4=
  `202.6441`。对应 30 epoch 快速值分别为 `330.7679` 和 `323.4893`。
- 训练：两组均正常结束；最后 loss 分别为 `0.04731/0.04724`，每组约 16.1 分钟；出现
  少量 AMP optimizer step skip，但没有 OOM、NaN 或 checkpoint 损坏。
- 结论：50 epoch 快速筛选中 LR2.5e-4 优于 LR3.0e-4，但快速 FID 不能替代正式
  5,000/5,000 评估，也不能宣称达到 FID≤15。大型 checkpoint 不进入普通 Git，轻量结果
  见 `results/lr_screen_50ep/`。
