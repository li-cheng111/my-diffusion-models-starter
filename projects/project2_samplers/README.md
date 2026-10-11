# Project 2: 采样器对比 (DDIM / DPM-Solver / Euler) 实验报告

本项目不重新训练模型，直接复用项目 1 已训练的 CIFAR-10 epsilon-prediction U-Net，比较不同采样器的生成质量、模型调用次数（NFE）和 DDIM 反演效果。代码、配置、测量结果和图表均保存在本目录。

## 实现内容

| 方法 | 实现与用途 |
|---|---|
| DDPM ancestral | 完整 1,000 步随机采样，作为基线。 |
| DDIM | 支持跳步、线性或二次时间步及 `eta`；`eta=0` 时为确定性采样。 |
| Euler | Probability-flow 一阶采样。当前实现与 `eta=0` 的 DDIM 使用相同的离散更新。 |
| DPM-Solver-2 | 基于指数中点法的二阶求解器；`S` 个外层步实际调用模型 `2S-1` 次。 |
| DDIM inversion | 将图像反演到噪声潜变量，再确定性重构并测量误差。 |

另提供同一初始噪声下的采样轨迹可视化、FID–NFE benchmark、样本网格、checkpoint 兼容性检查和只读 AutoDL 监控页面。

## 项目依赖与文件

Project 2 与 `projects/project1_ddpm/` 并列。`project1_path.py` 会优先发现该目录，并导入项目 1 的 `schedule.py`、`dataset.py` 和 `model/`。若项目 1 放在其他位置，可通过 `--project1_path` 或环境变量 `PROJECT1_PATH` 指定项目 1 源码目录（需包含上述文件）。

正式实验使用 linear 调度、训练 seed 44 的 EMA checkpoint。旧运行记录中的本地文件名为 `.local/checkpoints/cifar10_linear_200ep_seed44_final.pt`；Release 下载后的文件名为 `.local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt`，两者 SHA-256 相同。

```text
samplers/       DDPM、DDIM、Euler、DPM-Solver-2 与 DDIM 反演
benchmark.py   FID–NFE 评估和 Pareto 图
evaluate_inversion.py
visualize_trajectories.py
sample_grid.py  定性样本网格（不需要 FID 依赖）
check_compat.py 项目 1、调度器和 checkpoint 兼容性检查
configs/       各采样器的默认评估配置
results/       固定结果文件及样图
runs/          正式测量 JSON、图表及轨迹/反演结果
samples/       各配置生成的样本网格
logs/          正式运行日志
```

数据集、模型权重和缓存放在仓库外或 `.local/` 中；该目录已被 Git 忽略。

## 环境与 checkpoint

完整 FID 评估使用 Python 3.12.3、PyTorch 2.8.0+cu128、CUDA 12.8 和 AutoDL NVIDIA RTX 4090。依赖从仓库根目录安装：

```bash
pip install -r requirements/project1-2.txt
```

Project 2 的固定输入权重通过 [`project2-v1.0` Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/project2-v1.0) 发布。它是项目 1 CIFAR-10 linear、训练 seed 44 的 EMA checkpoint，文件大小为 459,538,769 B。下载到本地忽略目录：

```bash
gh release download project2-v1.0 \
  --repo li-cheng111/my-diffusion-models-starter \
  --pattern project2-input-ddpm-cifar10-linear-seed44.pt \
  --dir .local/checkpoints
```

SHA-256 为：

```text
937853559a1377660f7d4cbd1dd3f7c6cf022aed4ab84e9daa918a6e34541412
```

在 Linux、macOS、Git Bash 或 WSL 中，从下载目录校验随仓库提供的清单：

```bash
(cd .local/checkpoints && sha256sum -c ../../projects/project2_samplers/repro_manifest.sha256)
```

Windows PowerShell 可运行：

```powershell
(Get-FileHash .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt -Algorithm SHA256).Hash
```

输出应与上述摘要一致。Release 同时提供 `commit.txt`、`repro_manifest.json` 和 `repro_manifest.sha256`。

## 安装后检查与快速预览

从仓库根目录检查项目 1 发现、采样器、反演接口和 checkpoint 格式，并运行单元测试：

```bash
python -m projects.project2_samplers.check_compat \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt
python -m unittest discover -s projects/project2_samplers/tests -v
```

生成一个定性样本网格：

```bash
python -m projects.project2_samplers.sample_grid \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt \
  --sampler ddim --steps 50 \
  --output projects/project2_samplers/samples/smoke/cifar10_ddim50.png
```

小样本 FID 只能用于确认评估流程可以运行，不能作为最终结果报告。

如需同时检查 FID 计算链路，可运行 128 张图的小规模 benchmark：

```bash
python -m projects.project2_samplers.benchmark \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt \
  --sampler ddim dpm-solver --steps 5 10 \
  --num_samples 128 --batch_size 32 --seed 42 \
  --samples_dir projects/project2_samplers/samples \
  --output projects/project2_samplers/runs/smoke_benchmark.json \
  --plot projects/project2_samplers/runs/smoke_pareto.png
```

## 正式 FID–NFE benchmark

在仓库根目录运行完整矩阵：

```bash
python -m projects.project2_samplers.benchmark \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt \
  --preset full --num_samples 5000 --batch_size 64 --seed 42 \
  --data_root .local/datasets \
  --samples_dir projects/project2_samplers/samples \
  --output projects/project2_samplers/runs/benchmark_all.json \
  --plot projects/project2_samplers/runs/pareto_fid_nfe.png
```

正式协议固定如下：

- checkpoint：项目 1 CIFAR-10 linear、训练 seed 44、EMA 权重；checkpoint SHA-256 见上文。
- 每个配置以 batch size 64 生成 5,000 张图，并使用 CIFAR-10 训练集前 5,000 张无增强图作为真实图；每个配置开始时将生成随机种子重置为 42。
- 若数据集尚未下载，项目 1 的数据加载器会将 CIFAR-10 下载到 `--data_root` 指定目录。
- 图像从 `[-1,1]` 转为 RGB `uint8` `[0,255]`，使用 TorchMetrics FID-2048。表中采样时间不包含 FID 计算时间。
- 完整矩阵包括 DDPM 1,000 步；DDIM 和 Euler 各 10、20、50、100、250 步；DPM-Solver-2 为 5、10、25、50 个外层步。结果是单次 seed 42 测量，不包含采样随机种子方差。

### 结果

| 采样器 | 步数 / 外层步 | NFE | FID ↓ | 采样时间 |
|---|---:|---:|---:|---:|
| DDPM | 1,000 | 1,000 | **19.290** | 1,010.7 s |
| DDIM (`eta=0`) | 10 | 10 | 34.555 | 10.1 s |
| DDIM (`eta=0`) | 20 | 20 | 28.159 | 20.1 s |
| DDIM (`eta=0`) | 50 | 50 | 24.574 | 50.9 s |
| DDIM (`eta=0`) | 100 | 100 | 22.875 | 100.6 s |
| DDIM (`eta=0`) | 250 | 250 | 21.241 | 251.9 s |
| Euler | 10 | 10 | 34.555 | 10.1 s |
| Euler | 20 | 20 | 28.159 | 20.3 s |
| Euler | 50 | 50 | 24.574 | 50.4 s |
| Euler | 100 | 100 | 22.875 | 100.9 s |
| Euler | 250 | 250 | 21.241 | 251.4 s |
| DPM-Solver-2 | 5 | 9 | 29.295 | 9.3 s |
| DPM-Solver-2 | 10 | 19 | **21.074** | 19.3 s |
| DPM-Solver-2 | 25 | 49 | 21.361 | 49.4 s |
| DPM-Solver-2 | 50 | 99 | 21.192 | 100.5 s |

![FID–NFE Pareto 图](runs/pareto_fid_nfe.png)

在这组固定权重和协议下，DDPM 的绝对 FID 最低，但需要 1,000 NFE。DPM-Solver-2 在 19 NFE 时 FID 为 21.074，比相近预算的 DDIM（20 NFE，28.159）低 7.085；增加外层步数不保证 FID 持续下降。DDIM 和 Euler 的测量曲线重合，符合本实现中 `eta=0` DDIM 与一阶 probability-flow Euler 更新相同的离散形式。

此外，仓库记录了另一组独立实验：使用项目 1 的 R5 EMA 0.9999 权重 `r5_uniform_late_decay_final.pt`（[challenge-v1 Release](https://github.com/li-cheng111/my-diffusion-models-starter/releases/tag/challenge-v1)，SHA-256 `88eff8c3db7a2c9350d740247755f1d22abea3de0073b2b504b4b77c2ab412bb`）做 DDIM 评估，真实图、生成 seed 和生成图数量与正式协议相同。它更换了模型权重，不属于上表的同权重采样器对比。

| DDIM NFE | 10 | 20 | 50 | 100 | 250 |
|---|---:|---:|---:|---:|---:|
| R5 FID ↓ | 53.091 | 33.610 | 23.121 | 20.111 | **18.362** |

数据见 [`results/p1_r5_ddim_5k.json`](results/p1_r5_ddim_5k.json)，250 NFE 样图见 [`samples/r5_baseline/ddim_nfe250.png`](samples/r5_baseline/ddim_nfe250.png)。

### 单独运行 DDIM 矩阵

```bash
python -m projects.project2_samplers.benchmark \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt \
  --sampler ddim --steps 10 20 50 100 250 \
  --num_samples 5000 --batch_size 64 --seed 42 \
  --data_root .local/datasets \
  --samples_dir projects/project2_samplers/samples \
  --output projects/project2_samplers/runs/benchmark_ddim.json \
  --plot projects/project2_samplers/runs/pareto_ddim.png
```

## 采样轨迹对比

轨迹实验对四种方法传入同一个初始噪声 tensor，减少随机起点造成的差异。下列数据使用 CIFAR-10 seed 44 EMA 模型、初始噪声 seed 44；DDIM、Euler 使用 50 步，DPM-Solver-2 使用 50 个外层步。

| 采样器 | 步数 | 实际 NFE | 平均单步 L2 | 最大单步 L2 | 终点距初始噪声 L2 |
|---|---:|---:|---:|---:|---:|
| DDPM ancestral | 1,000 | 1,000 | 5.2548 | 8.0443 | 67.5356 |
| DDIM (`eta=0`) | 50 | 50 | 1.1307 | 3.2183 | 53.7210 |
| Euler | 50 | 50 | 1.1307 | 3.2183 | 53.7210 |
| DPM-Solver-2 | 50 | 99 | 1.1225 | 4.2199 | 53.0901 |

![相同初始噪声下的采样轨迹](runs/trajectory_comparison.png)

从仓库根目录重现：

```bash
python -m projects.project2_samplers.visualize_trajectories \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt \
  --samplers ddpm ddim euler dpm-solver --steps 50 --seed 44 \
  --output projects/project2_samplers/runs/trajectory_comparison.png \
  --metrics_output projects/project2_samplers/runs/trajectory_comparison.json
```

## DDIM 反演与重构

反演使用 CIFAR-10 测试集前 64 张图，按采样时间步的反方向运行；往返过程关闭 `x0` 裁剪，避免裁剪影响重构误差。表中指标是 64 张图逐图测量后的均值。

| 步数 | 平均 L2 ↓ | MAE ↓ | MSE ↓ | PSNR ↑ |
|---:|---:|---:|---:|---:|
| 10 | 18.2820 | 0.26955 | 0.11417 | 15.891 dB |
| 20 | 12.5195 | 0.18369 | 0.05343 | 19.175 dB |
| 50 | 5.0287 | 0.07314 | 0.00864 | 27.057 dB |
| 100 | 2.3665 | 0.03419 | 0.00197 | 33.644 dB |
| 250 | **0.9316** | **0.01332** | **0.00032** | **41.836 dB** |

![DDIM 反演误差曲线](runs/inversion_errors.png)

```bash
python -m projects.project2_samplers.evaluate_inversion \
  --ckpt .local/checkpoints/project2-input-ddpm-cifar10-linear-seed44.pt \
  --steps 10 20 50 100 250 --num_images 64 \
  --data_root .local/datasets \
  --output projects/project2_samplers/runs/inversion_results.json \
  --plot projects/project2_samplers/runs/inversion_errors.png
```

## 方法说明与适用范围

DDIM 根据噪声预测估计干净样本：

$$
\hat{x}_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t,t)}
{\sqrt{\bar{\alpha}_t}}.
$$

对跳步 `t > t_prev`，更新为：

$$
\begin{aligned}
\sigma_t^2 &= \eta^2\frac{1-\bar{\alpha}_{t_{prev}}}{1-\bar{\alpha}_t}
\left(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t_{prev}}}\right),\\
x_{t_{prev}} &= \sqrt{\bar{\alpha}_{t_{prev}}}\hat{x}_0
+\sqrt{1-\bar{\alpha}_{t_{prev}}-\sigma_t^2}\,\epsilon_\theta
+\sigma_t z.
\end{aligned}
$$

`eta=0` 时 `sigma=0`，固定权重、时间步和初始噪声后采样结果确定。`eta` 增大时会注入额外噪声，可能增加多样性，也会改变低步数下的保真度。实现会保护平方根和除法的数值范围；普通生成默认裁剪 `x0` 到 `[-1,1]`，反演与重构不裁剪。

DPM-Solver-2 令 `lambda = log(alpha / sigma)`，在 lambda 中点进行二阶指数积分。每个中间区间调用模型两次，最后的终止步只需一次，因此实际 NFE 为 `2S-1`。本实验使用 lambda 时间步；报告的 NFE 由采样器调用次数验证。

当 DDIM 不裁剪 `x0`、采用连续相邻训练时间步，并令随机方差等于 DDPM posterior variance

$$
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
$$

其更新可化为 DDPM ancestral 更新。跳步时即使 `eta=1`，也只是类似 DDPM 的随机更新，并非原始 DDPM Markov 链。

若模型采用 cosine 训练调度，推理时间步宜按 log-SNR（或 `alpha_bar` 噪声水平）布点，再映射为不重复的训练索引；均匀整数索引不代表等量噪声变化。本文正式测量使用 linear 调度的 checkpoint，不把该组数值外推到 cosine 模型。

## 实现与复现注意事项

- 调度器 buffer 名称在 starter 与项目 1 之间不同；采样器通过统一访问器接受单数和复数命名。
- checkpoint loader 支持嵌套 EMA、扁平 EMA，以及多衰减率 EMA；多 EMA 默认选最大 decay，也可显式指定。
- FID 比较固定真实数据子集与样本数量，并对每个配置重置生成 seed；小样本 smoke test 不作为正式指标。
- 轨迹比较显式传入共同初始噪声；DPM-Solver-2 报告真实模型调用数，而非外层步数。
- 这些规则用于避免把数据增强、随机数状态、checkpoint 格式或 NFE 统计差异误当成采样器效果。

## AutoDL 可选监控

`monitor_dashboard.py` 是只读页面，每 2 秒刷新 GPU、benchmark 进度、已完成 FID、预览文件和日志。AutoDL 运行：

```bash
python -m projects.project2_samplers.monitor_dashboard --host 127.0.0.1 --port 18765
```

在本地终端建立 SSH 端口转发后访问 `http://127.0.0.1:18765/`：

```bash
ssh -N -L 18765:127.0.0.1:18765 -p SSH_PORT root@SSH_HOST
```

将 `SSH_PORT` 和 `SSH_HOST` 替换为 AutoDL 实例提供的值；实例更换后需更新连接信息。

## 验证与结果文件

历史运行日志记录了兼容性检查、标准库单元测试和正式 FID 测量通过；正式 5,000 样本评估在 AutoDL RTX 4090 上完成。当前测试入口为 `python -m unittest discover -s projects/project2_samplers/tests -v`。本说明整理过程中未重新运行实验。

| 文件 | 内容 |
|---|---|
| `runs/benchmark_all.json` | 15 个正式配置、FID、NFE、耗时、checkpoint 摘要和运行环境。 |
| `runs/benchmark_ddim.json` | 五个 DDIM 步数的独立结果。 |
| `runs/pareto_fid_nfe.png` | 完整 FID–NFE Pareto 图。 |
| `runs/trajectory_comparison.json`、`runs/trajectory_comparison.png` | 共同初始噪声下的轨迹指标和图像。 |
| `runs/inversion_results.json`、`runs/inversion_errors.png` | 反演重构指标和误差图。 |
| `results/p1_r5_ddim_5k.json`、`results/p1_r5_ddim_5k.png` | Project 1 R5 checkpoint 的独立 DDIM 对照。 |

正式 FID、轨迹和反演测量均已完成；代码测试可按上面的命令在当前环境中重新验证。
