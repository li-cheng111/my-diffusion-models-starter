# 项目 4：CIFAR-10 上的 Flow Matching

> **难度**：中-高
> **预期完成时间**：2 周（含训练时间）
> **前置**：完成 L12；做过 Project 1（DDPM 基线）与 Project 2（FID-NFE 评估）会顺很多

用 Rectified Flow / Conditional Flow Matching 训一个 32×32 RGB 图像生成模型，
对比 DDPM 与 FM 在 NFE-FID 曲线上的差异。

> **v2 复现说明（2026-10）**：本目录现在提供可恢复训练、多档 EMA、Euler/Heun
> 采样和统一 FID 评测。若要复现新的 Project 4 结果，请使用下方「v2 实施与复现」；
> 后续旧版示例仅用于课程 TODO 的背景说明。v1 的旧 FID 表是历史数据，不能当作
> 与 Project 1 R5 基线已严格对齐的 v2 对照结果。

## 配套教材

讲义、公式推导、论文导读在教材库，开始前请确认已 clone：

```bash
git clone https://github.com/Qi-StarterTrain/diffusion-models-starter-materials.git
```

本项目用到：

| 教材库文件 | 用途 |
|-----------|------|
| `slides/L12_flow_matching.md` | 主线讲义。**§5 linear path 与训练 loss（TODO 16）**、§5.3 ODE 积分（TODO 17）、§10.3 CFG with FM（TODO 18）、§8 sigmoid schedule（进阶任务 1） |
| `derivations/derive_07_flow_matching.md` | §7.3-§7.4 linear path 的 $u_t$ 与训练 loss 的完整推导、§9 Reflow（进阶任务 3） |
| `notebooks/nb09_flow_matching_2d.ipynb` | 2D 玩具版 FM——**先跑通它再写 TODO 16**，2D 上调试比 CIFAR-10 快两个数量级 |
| `slides/L11_dit.md` | §3 DiT 架构、§6 训练细节。用 `model.type: dit` 时对照着看 |
| `notebooks/nb08_dit_blocks.ipynb` | AdaLN-Zero 的最小实现 |
| `paper_notes/13_FlowMatching_Lipman2023.md`<br>`paper_notes/14_RectifiedFlow_Liu2022.md` | 两篇原始论文的导读 |

> **记号提醒**：L12 和 derive_07 用 $x_0 \sim \mathcal{N}(0,I)$ 表示噪声、$x_1$ 表示数据；
> 本仓库代码里把噪声叫 `epsilon`（`x_t = (1-t)·epsilon + t·x_1`，target = `x_1 - epsilon`）。
> 只是换了个名字，公式完全一致——但推导时别把 $x_0$ 和代码里的 `x` 搞混了。

## 学习目标

完成本项目后，你能够：
1. 实现 Rectified Flow 的训练 loss（线性路径 + velocity prediction）
2. 实现 Euler 与 Heun ODE 积分采样
3. 实现 Classifier-Free Guidance for FM
4. 评估并对比 DDPM vs FM 在不同 NFE 下的生成质量

---

## 文件结构

```
（仓库根目录）
├── README.md                       # 本文件
├── check_setup.py                  # 配置/模型/TODO 自检（动手前先跑）
├── train.py                        # 训练入口 ← TODO 16
├── sample.py                       # 采样入口 ← TODO 17, 18
├── eval_fid.py                     # NFE-FID 扫描与评估
├── model/
│   ├── __init__.py
│   ├── unet.py                     # 简化版 conditional UNet（~30M）
│   └── dit.py                      # 简化版 DiT-S（供切换 backbone）
├── configs/
│   ├── cifar10_fm.yaml             # 正式训练配置（DiT-S，200K steps）
│   └── cifar10_fm_debug.yaml       # 冒烟测试配置（小 UNet，500 steps）
└── experiment_log_template.md
```

CIFAR-10 的下载与 dataloader 直接写在 `train.py` 里（`torchvision.datasets.CIFAR10`），
不需要单独的 data 模块。

**动手之前先跑自检**（几秒钟）：

```bash
python check_setup.py
```

它会核对 config 的键、按 config 构建模型跑一次前向（包括把 CFG 的 null token
喂进 embedding），并验证 TODO 16/17/18 的 loss 和采样输出。

## v2 实施与复现

v2 将 Project 4 拆成两个 200K-step 训练配置，避免把类别条件模型的 CFG=0
误称为无条件模型，并可以直接和 Project 1 的无条件 DDPM 比较：

| 实验 | 配置 | 用途 |
|------|------|------|
| FM-U | `configs/cifar10_fm_v2_unconditional.yaml` | 每张训练图都用 null 类别 token；与 Project 1 对照 |
| FM-C | `configs/cifar10_fm_v2_conditional.yaml` | 类别条件 + 10% 条件 dropout；评估条件生成与 CFG |

两者均使用 DiT-S（32.62M 参数）、时间嵌入尺度 1000、batch 64、BF16、200K
optimizer steps、seed 42、EMA 0.999/0.9995/0.9999、2K warmup，并在后半程做
cosine 学习率衰减。50K 间隔保留可交付 checkpoint，10K 间隔原子更新 `latest.pt`；
续训恢复模型、optimizer、scheduler 和随机数生成器状态。每 100 steps 将 loss、LR、
速度与累计耗时写入 `train_log.csv`。

AutoDL 单卡 RTX 4080 SUPER（实例报告 32,760 MiB 显存）运行时，从仓库根目录执行：

```bash
pip install -r requirements/project4.txt
mkdir -p data
curl -fL -o data/cifar-10-python.tar.gz \
  https://hf-mirror.com/datasets/MIT-OL-AI-D/cifar-10-python/resolve/a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz
echo 'c58f30108f718f92721af3b95e74349a  data/cifar-10-python.tar.gz' | md5sum -c -
python projects/project4_flow_matching/train.py \
  --config projects/project4_flow_matching/configs/cifar10_fm_v2_unconditional.yaml \
  --output runs/fm_v2_unconditional --seed 42 --precision bf16
python projects/project4_flow_matching/train.py \
  --config projects/project4_flow_matching/configs/cifar10_fm_v2_conditional.yaml \
  --output runs/fm_v2_conditional --seed 42 --precision bf16
```

若进程中断，用同一条命令并追加 `--resume runs/<实验>/latest.pt`。4080 SUPER 上
100-step 预跑为 10.8 step/s；单个 200K 训练预算约 4.6 小时。此次 FM-U 实测 4.83 小时，
期间与约 35 分钟的 Project1 基线评测共用 GPU；基线结束后稳定在约 12.7 step/s。FM-C
最终 checkpoint 记录的累计训练时间为 4.36 小时，最终 loss 为 0.1665。进程在记录到
193K 后意外退出；从最后完整的 190K checkpoint 恢复，补跑至 200K。未保存的 3K 步不计入
checkpoint 的 4.36 小时累计值，恢复过程和两段原始日志均已归档。CIFAR-10 压缩包约
170 MB，固定镜像下载约 15 秒，并通过 MD5 校验。可续训 checkpoint 和数据保存在 AutoDL
数据盘；最终推理 checkpoint 附在 GitHub Release，不放入 Git 对象库。

发布 release 前，将最终续训 checkpoint 转为不含 optimizer/RNG 的推理文件：

```bash
python projects/project4_flow_matching/export_checkpoint.py \
  --input runs/fm_v2_unconditional/latest.pt \
  --output /root/autodl-tmp/checkpoints/project4-v2-unconditional-step200000.pt
sha256sum /root/autodl-tmp/checkpoints/project4-v2-unconditional-step200000.pt
```

已训练完成的 EMA 推理权重和评测/日志归档可从 GitHub Release 下载。权重哈希也记录在
[`results/checkpoint_sha256.txt`](results/checkpoint_sha256.txt) 与
[`repro_manifest_v2.json`](repro_manifest_v2.json) 中：

```bash
mkdir -p checkpoints
wget -O checkpoints/project4-v2-unconditional-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4-v2-unconditional-step200000.pt
wget -O checkpoints/project4-v2-conditional-step200000.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4-v2-conditional-step200000.pt
wget -O project4_v2_results_only.tar.gz \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4_v2_results_only.tar.gz
wget -O project4_v2_results_only.tar.gz.sha256 \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/project4-v2.0/project4_v2_results_only.tar.gz.sha256
sha256sum -c project4_v2_results_only.tar.gz.sha256
echo 'b75cbe3d4f9aeb16c66cb53e4acb03f9c2c6a18421efd945f78475cd2e114571  checkpoints/project4-v2-unconditional-step200000.pt' | sha256sum -c -
echo '25666b32e914002778825a02f8e14c89f9f5ff081f9dded9f3e0ad6e0283c5aa  checkpoints/project4-v2-conditional-step200000.pt' | sha256sum -c -
```

### Project 1 对照与一致评测

先下载已归档的 Project 1 R5 checkpoint，并校验 SHA256：

```bash
mkdir -p /root/autodl-tmp/checkpoints
wget -O /root/autodl-tmp/checkpoints/p1-r5.pt \
  https://github.com/li-cheng111/my-diffusion-models-starter/releases/download/challenge-v1/r5_uniform_late_decay_final.pt
echo '88eff8c3db7a2c9350d740247755f1d22abea3de0073b2b504b4b77c2ab412bb  /root/autodl-tmp/checkpoints/p1-r5.pt' | sha256sum -c -
```

固定 CIFAR-10 train 前 5000 张真实图、TorchMetrics FID-2048、EMA 0.9999、
seed 42 和同一份项目依赖，测 Project 1 DDIM 基线：

```bash
python -m projects.project2_samplers.benchmark \
  --ckpt /root/autodl-tmp/checkpoints/p1-r5.pt --sampler ddim \
  --steps 10 20 50 100 250 --num_samples 5000 --batch_size 64 --seed 42 \
  --ema_decay 0.9999 --num_workers 4 \
  --output projects/project2_samplers/results/p1_r5_ddim_5k.json \
  --plot projects/project2_samplers/results/p1_r5_ddim_5k.png
```

AutoDL RTX 4080 SUPER 上已完成的 R5 基线结果如下。FID 计时使用同一组 5,000 张真实图；五点采样总计 2,006.3 秒，FID 计算总计 83.0 秒，完整扫描约 34 分 49 秒。原始 JSON、曲线、运行日志和每个 NFE 的预览图保存在 `projects/project2_samplers/results/` 与 `projects/project2_samplers/samples/r5_baseline/`。

| DDIM NFE | FID | 采样时间 | FID 计算时间 |
|---:|---:|---:|---:|
| 10 | 53.0907 | 47.7 秒 | 18.0 秒 |
| 20 | 33.6099 | 93.6 秒 | 16.6 秒 |
| 50 | 23.1206 | 233.4 秒 | 15.0 秒 |
| 100 | 20.1105 | 466.5 秒 | 16.9 秒 |
| 250 | 18.3625 | 1,165.2 秒 | 16.5 秒 |

FM-U 的无条件 Euler 曲线：

```bash
python projects/project4_flow_matching/eval_fid.py \
  --model runs/fm_v2_unconditional/latest.pt \
  --condition unconditional --solver euler heun --nfe 4 8 16 32 50 \
  --num_samples 5000 --batch_size 64 --seeds 42 \
  --output projects/project4_flow_matching/results/fm_v2_unconditional_solvers.json
```

绘图横轴按实际网络求值次数对齐；Heun 每步需要两次前向，例如 Euler 8 步与
Heun 4 步都记为每图 8 次网络求值。原始 JSON 会同时保存积分步数和网络求值数。
FM-C 的 CFG 扫描显式使用两种标签预测：

```bash
python projects/project4_flow_matching/eval_fid.py \
  --model runs/fm_v2_conditional/latest.pt \
  --condition cfg --solver euler --nfe 20 --cfg 1 2 3 5 7.5 \
  --num_samples 5000 --batch_size 64 --seeds 42 \
  --output projects/project4_flow_matching/results/fm_v2_conditional_cfg.json
```

`--condition unconditional` 始终输入 null token，`conditional` 始终输入平衡的真实类别，
`cfg` 才执行 $v_u+s(v_c-v_u)$；CFG 每步的网络求值成本是条件或无条件单路的两倍。
重复种子时用 `--seeds 42 43 44`，报告均值和离散程度，不把单个 seed 说成稳定结论。

### v2 本次实测结果

两组模型均训练 200,000 步，使用 EMA 0.9999；FID 使用 seed 42、5,000 张 CIFAR-10
训练集真实图和 TorchMetrics FID-2048。该表是单 seed 的一次测量，不包含随机种子方差。

| Solver | 积分步数 | 实际网络评估/图 | FID |
|---|---:|---:|---:|
| Euler | 4 | 4 | 80.5531 |
| Euler | 8 | 8 | 43.2211 |
| Euler | 16 | 16 | 32.1931 |
| Euler | 32 | 32 | 28.0684 |
| Euler | 50 | 50 | 26.7610 |
| Heun | 4 | 8 | 154.8813 |
| Heun | 8 | 16 | 82.1227 |
| Heun | 16 | 32 | 43.2588 |
| Heun | 32 | 64 | 26.4840 |
| Heun | 50 | 100 | 24.0062 |

Project 1 R5 + DDIM 基线在 NFE 10/20/50/100/250 的 FID 为 53.0907/33.6099/23.1206/
20.1105/18.3625。FM Euler 在 NFE 8 对 DDIM NFE 10、NFE 16 对 DDIM NFE 20 的两个
错位比较点 FID 较低；但同 NFE 50 时 DDIM 优于 FM Euler，同 NFE 100 时 DDIM 优于已测
的 FM Heun。由于低预算点没有对齐、每项仅有一个评测 seed，且模型架构和训练预算不同，
本次数据不足以支持“FM 在低步数下总体优于 DDIM”，也不能把局部差异归因于采样方法本身。
Heun 在 NFE 8/16 的 FID 分别为 154.8813/82.1227，在 NFE 64 达到 26.4840，显示结果对
求解器与积分预算敏感。详细比较与补测建议见 [实验报告](report.md)。对比曲线见
[`results/nfe_fid_curve_v2.png`](results/nfe_fid_curve_v2.png)。

固定 Euler 20 个积分步（CFG 每图实际进行 40 次网络评估）的 guidance 扫描：

| CFG scale | FID |
|---:|---:|
| 1.0 | 23.5773 |
| 2.0 | **18.8105** |
| 3.0 | 23.6983 |
| 5.0 | 35.2425 |
| 7.5 | 44.9110 |

本次最佳 scale 为 2.0；更强 guidance 从 3.0 起使 FID 变差。完整评测 JSON、checkpoint
哈希、训练 CSV、训练与评测日志、类别样本网格和无条件样图均已归档在 `results/`、
`logs/`、`samples/`，推理权重随 `project4-v2.0` GitHub Release 发布。

### AutoDL 实时监控

训练进度页面只读训练 CSV、GPU 使用率和最近日志，每两秒刷新。实例中启动：

```bash
mkdir -p logs runs
nohup python projects/project4_flow_matching/monitor_dashboard.py \
  --host 0.0.0.0 --port 6006 > logs/dashboard.log 2>&1 &
nohup python projects/project4_flow_matching/train.py \
  --config projects/project4_flow_matching/configs/cifar10_fm_v2_unconditional.yaml \
  --output runs/fm_v2_unconditional \
  --seed 42 --precision bf16 \
  > logs/fm_v2_unconditional.log 2>&1 &
```

第一组训练结束后，再启动 FM-C（不要同时占用这张卡训练两个模型）。在 AutoDL
控制台「自定义服务」中打开实例 6006 端口对应的网址，即可在外部浏览器
查看实时进度。实例被释放后服务地址会失效。

---

## TODO 任务

### TODO 16: 实现 Rectified Flow Loss（train.py）

文件 `train.py` 中的 `compute_fm_loss` 函数。

任务：
- 采样 t ~ U[0, 1]
- 采样 epsilon ~ N(0, I)
- 构造 x_t = (1-t) * epsilon + t * x_1
- 训练 v_theta 预测 (x_1 - epsilon)

提示：参考 L12 §5.2 与 derive_07 §7.3-§7.4。

**null token 约定**：`num_classes` 传的是真实类别数（CIFAR-10 是 10），
索引 `0..9` 是真实类别，**索引 10 是 CFG 的 null token**——
两个 backbone 内部都按 `Embedding(num_classes + 1)` 分配。
做 conditional dropout 时把 `y` 替换成 `num_classes` 即可。

---

### TODO 17: 实现 Euler ODE Sampler（sample.py）

函数 `euler_sample`。

任务：
- 从 x_0 ~ N(0, I) 开始
- 用 Euler 一阶积分 dx/dt = v_theta(x, t)
- 从 t=0 到 t=1，n_steps 步

注意：与 DDPM 的 reverse loop **方向相反**（t 从 0 增到 1）。

---

### TODO 18: 实现 CFG for FM（sample.py）

函数 `euler_sample_cfg`。

CFG for FM：
```
v_cond = model(x, t, y_cond)
v_uncond = model(x, t, y_null)
v = v_uncond + s * (v_cond - v_uncond)
```

注意：训练时要 10% conditional dropout（已在 train.py 中实现）。

---

## 评估实验

完成 TODO 后，按这个顺序走：

```bash
# 0. 冒烟测试：小模型跑 500 步，10 分钟内确认整条流水线通了
python train.py --config configs/cifar10_fm_debug.yaml --output checkpoints/debug
python sample.py --model checkpoints/debug/latest.pt --nfe 8 --cfg 0 3
#    出图会很糊，只看「有没有跑通」，不看质量

# 1. 正式训练（旧版教学估算：RTX 4090 上 12-24h；v2 在 AutoDL RTX 4080 SUPER 上的实测时间见上文）
python train.py --config configs/cifar10_fm.yaml --output checkpoints/fm

# 2. 出样本网格图
python sample.py --model checkpoints/fm/latest.pt --nfe 4 8 16 32 50 --cfg 0 1 3 7.5

# 3. NFE-FID 扫描（FM 的核心卖点就在这条曲线上）
python eval_fid.py --model checkpoints/fm/latest.pt \
    --nfe 4 8 16 32 50 --num_samples 5000 --output results/fm_nfe.json

# 4. CFG scale 扫描（固定 NFE=20）
python eval_fid.py --model checkpoints/fm/latest.pt \
    --nfe 20 --cfg 1 2 3 5 7.5 --output results/fm_cfg.json
```

训练脚本存的是 `latest.pt` 和 `step_N.pt`（没有 `best.pt`——FM 训练过程中
没有便宜的验证指标可以用来选“best”，用最后一个 checkpoint 或自己挑一个 step）。

### DDPM 基线从哪来

**不需要在本项目里再训一个 DDPM。** 你在 Project 1 已经训过 CIFAR-10 DDPM，
Project 2 的 `benchmark.py` 已经在同一数据集上量过 FID-NFE 曲线——
直接把那份 `benchmark_ddim.json` 的数搬过来做对比即可。

⚠️ 两条曲线要可比，必须对齐口径：**同样的 `num_samples`（5000）、
同样的 FID 实现（都用 torchmetrics）、同样的真实样本来源（CIFAR-10 train）**。
本项目的 `eval_fid.py` 与 Project 2 的 `benchmark.py` 用的是同一套 torchmetrics 配置，
所以只要 `num_samples` 对齐就能直接比。口径不一致的两组 FID 放在一张图上是没有意义的。

如果你没做过 Project 1/2，在报告里说明，只交 FM 自己的 NFE-FID 曲线即可。

---

## 预期结果

| NFE | DDPM FID | FM FID |
|-----|---------|--------|
| 10 | 25+ | **8-12** |
| 25 | 8-10 | 6-8 |
| 50 | 5-7 | 5-6 |
| 250 | 4-5 | 4-5 |

观察：**FM 在低 NFE 下显著优于 DDPM**。这是 FM 的核心卖点。

> ⚠️ **这张表是论文级训练的量级**（1M+ steps、更大模型、50000 样本算 FID）。
> 本项目的教学配置是 200K steps + 5000 样本，**FID 绝对值会明显更高**（十几到二十几都正常），
> 样本数少还会让 FID 系统性偏高且方差变大。
>
> **评分看的是趋势不是绝对值**：你的 FM 曲线在低 NFE 区间是否显著优于 DDPM、
> 曲线在多少 NFE 之后开始变平。别为了追这张表里的数去反复重训。
>
> **本仓库 v2 的实测结论**：两个错位的低预算测点中 FM Euler 的 FID 较低，但现有数据
> 不足以证明 FM 在低步数下总体优于 DDIM；结果详见上方“v2 本次实测结果”。这张论文级
> 预期表不能当成本项目的实际结果或保证趋势。

---

## 训练资源

- 单 GPU (V100 / A100 / RTX 4090) 即可
- 旧版估算：DiT-S 200K steps 约 12-24 小时（RTX 4090）。v2 在 AutoDL RTX 4080 SUPER 上实测 FM-U 为 4.83 小时、FM-C checkpoint 记录 4.36 小时；FM-C 曾恢复训练，细节见上文。
- 推荐 model：DiT-S（`configs/cifar10_fm.yaml`，~33M）或 UNet（~30M）
- 显存不够就调小 `train.batch_size`，同时把 `train.lr` 按比例调小
- **先跑 `cifar10_fm_debug.yaml`**（500 步、10 分钟）确认流水线通了再开长训练

---

## 调试清单

如果 loss 不收敛：
- [ ] 检查 t 采样是否覆盖 [0, 1]（不要漏 t=0 或 t=1）
- [ ] 检查 x_t 维度与 x_1 一致
- [ ] 检查 target 是 `x_1 - epsilon`（不是 `epsilon - x_1`）
- [ ] 检查 epsilon 是 standard Gaussian（mean=0, std=1）

如果生成质量差：
- [ ] 增大 batch size 到 256+
- [ ] 训练 100K+ steps
- [ ] 用 EMA 权重采样（`train.py` 已经在维护 EMA，`sample.py` 默认就用 EMA）
- [ ] 检查 sampler 的 dt 是否累加到 1

如果**采样出来是纯噪声/纯灰**：
- [ ] 步数少的时候检查 `ema_decay`——`0.9999` 意味着大约需要 1 万步 EMA 才追得上模型。
      只训了几百步的话 EMA 权重几乎还是随机初始化，采出来当然是废的。
      `cifar10_fm_debug.yaml` 里已经把它调成 `0.99`；用 `--no_ema` 也可以绕过
- [ ] 检查积分方向：FM 是 t 从 0 走到 1，`x = x + v * dt`（**加号**），
      和 DDPM 的 reverse loop 方向相反

如果报 `IndexError` 在 embedding 上：
- [ ] null token 索引是 `num_classes`（=10），embedding 大小要是 `num_classes + 1`。
      `python check_setup.py` 会专门测这一条

---

## 进阶任务

1. **Logit-normal t-sampling (SD 3)**：用 `t = sigmoid(N(0, 1))` 而非均匀。看 FID 变化（L12 §8）
2. **Heun 二阶 sampler**：用一阶估计走一步，再用新位置估二阶（L12 §11 进阶 4）
3. **Reflow**：拿训好的 FM 生成大量 $(\epsilon, x_1)$ pairs，重新训一个"拉直"的 FM（derive_07 §9）
4. **Switch backbone**：`model.type` 从 `unet` 改成 `dit`，同预算下对比 FID（L11 §7）

---

## 自查问题（在报告中回答）

前 3 道来自 L12 §11，第 4-5 道来自 derive_07 §11：

1. **推导**：不看讲义，推出 linear path 的 $u_t(x|x_1) = x_1 - x_0$。
2. **等价性**：为什么最小化 CFM loss 等价于最小化不可计算的 FM loss？
   （关键在于两者的梯度相同，不是两者的值相同——说清楚差在哪）
3. **对比**：同一数据集上 FM 与 DDPM 的 NFE-FID 曲线，在低 NFE 区间差距为什么这么大？
4. **Reflow**：为什么 reflow 能改善 1-step 的生成质量？它把什么"拉直"了？
5. **CFG**：为什么 CFG 在 velocity 空间可以直接线性组合（`v = v_u + s(v_c - v_u)`），
   形式和 DDPM 的 $\epsilon$ 空间 CFG 一模一样？（L12 §10.3 给了一句话的答案，展开讲）

---

## 提交要求

**截止前将以下内容 push 到你的作业仓库 main 分支**，助教直接在仓库里评分。

```
（仓库根目录）
├── train.py / sample.py            # 含你实现的 TODO 16-18
├── results/
│   ├── fm_nfe.json                 # NFE 扫描的原始结果
│   ├── fm_cfg.json                 # CFG scale 扫描的原始结果
│   └── nfe_fid_curve.png           # FM vs DDPM 的 NFE-FID 曲线（核心交付物）
├── samples/                        # 各 NFE / CFG 下的样本网格图
├── logs/exp_log.md                 # 实验日志（按 experiment_log_template.md）
├── report.md                       # 3-5 页报告
└── debug_log.md                    # 至少 3 条踩坑记录
```

`report.md` 需包含：
1. TODO 16 的推导：从 linear path 到训练 loss（手写扫描或 LaTeX）
2. NFE-FID 曲线 + 与 DDPM 基线的对比分析（说明两边的 FID 口径如何对齐）
3. CFG scale 的影响：多大开始出现过饱和/多样性坍塌
4. 上面「自查问题」
5. 进阶任务（如完成）

> checkpoint 不要提交（几百 MB，`.gitignore` 已经拦了）。
> `results/*.json` 和图必须提交——助教看的是这些。

---

## 学术诚信

⚠️ Rectified Flow 的实现网上到处都是（`torchcfm`、SD3 的开源复现等），但：

- TODO 16-18 加起来不到 30 行，抄了等于这个 Project 白做
- 允许参考它们的**结构**，但必须自己写、自己调
- 直接复制粘贴并提交将记 0 分
- 报告里的"踩坑记录"必须真实——抄来的踩坑是看得出来的

---

## 评分细则

| 项 | 权重 | 评分要点 |
|----|------|---------|
| TODO 16（FM loss） | 25% | 公式正确、conditional dropout 正确、能收敛 |
| TODO 17（Euler sampler） | 15% | 积分方向与 dt 正确，能出图 |
| TODO 18（CFG） | 15% | 实现正确，batch 合并优化加分 |
| NFE-FID 实验 | 25% | 曲线完整、口径对齐、分析到位 |
| 报告与自查问题 | 15% | 推导正确，回答有深度 |
| 踩坑记录 | 5% | ≥3 条真实踩坑 |
| **加分** | +15 | 每个进阶任务 +5，最多 3 个 |

---

## 关联材料

见开头的「配套教材」表。论文：Lipman 2023、Liu 2022、Esser 2024 (SD 3)。
