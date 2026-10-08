# 项目 5：VLA Action Diffusion——课程综合项目

> **目标**：在玩具 2D 环境上实现 vision-conditioned diffusion policy，复现 Pi-0 / Diffusion Policy 的核心思想。
>
> **难度**：⭐⭐⭐⭐⭐（综合 P0-P4 所有内容）
> **预期耗时**：2-3 周
> **前置**：完成 L17；Project 1（DDPM loss）与 Project 4（action chunk / FM）的经验直接复用

## 配套教材

讲义、论文导读在教材库，开始前请确认已 clone：

```bash
git clone https://github.com/Qi-StarterTrain/diffusion-models-starter-materials.git
```

| 教材库文件 | 用途 |
|-----------|------|
| `slides/L17_vla_embodied.md` | 主线讲义。§7 Diffusion Policy（本项目的直接原型）、§8 关键设计选择（chunk size / 重规划频率）、§5 Pi-0、§9 数据问题 |
| `notebooks/nb12_vla_action_diffusion.ipynb` | 2D action diffusion 玩具版——**先跑通它再写 TODO 19** |
| `paper_notes/19_Pi0_Black2024.md` | Pi-0 的 flow matching action head（加分项要复现的就是它） |
| `paper_notes/18_OpenVLA_Kim2024.md` | OpenVLA，L17 §12 必做的精读对象 |
| `slides/L12_flow_matching.md` | §9 专门讲 Pi-0 的 action FM 与 action chunking，做加分项（FM 替代 DDPM）时看这里 |
| `quizzes/quiz4_vla_world/` | 本阶段的自测题 |

---

## 项目背景

本项目是整个课程的**综合 capstone**。你将：
- 设计一个简化版本的 VLA：vision → action trajectory
- 用 Diffusion Policy / Flow Matching 训练 action head
- 在 toy 2D 环境上做闭环评估
- 体会 Pi-0 / OpenVLA 在工程上的取舍

---

## 任务清单（TODO 19-21）

- **TODO 19**：实现 action chunk DDPM loss
- **TODO 20**：实现 vision encoder + observation conditioning
- **TODO 21**：实现闭环评估（成功率 + trajectory error）

---

## 任务设置：2D Reaching with Distractors

**环境**：
- 一个 64×64 像素的 RGB 渲染场景
- 一个 agent（蓝色圆点）
- 一个目标（红色星）
- 2-3 个干扰物（灰色方块）
- Action 空间：连续 2D delta position $(\Delta x, \Delta y) \in [-0.1, 0.1]^2$

**任务**：从 agent 当前位置走到目标，避开干扰物。

**为什么这个 task 有意义**：
- 视觉条件（需要"看"目标在哪、干扰物在哪）
- 多模态分布（多种合理路径绕开干扰物）
- 长 horizon（32-64 步）
- 闭环评估清晰（成功 = 到达目标 + 不碰干扰物）

---

## 目录结构

```
（仓库根目录）
├── README.md                  # 本文件
├── check_setup.py             # 环境/数据/模型/TODO 自检（动手前先跑）
├── env.py                     # 2D toy 环境（已写好）
├── dataset.py                 # 专家 demo 生成 + 加载（已写好）
├── obs_utils.py               # 观测 → 模型输入的统一约定（已写好，train/eval 共用）
├── model.py                   # 模型定义 ← TODO 20
├── train.py                   # 训练入口 ← TODO 19
├── eval.py                    # 闭环评估 ← TODO 21
├── experiment_log_template.md
└── configs/
    ├── reach2d_debug.yaml     # 冒烟测试（50 demo / 500 步，几分钟）
    ├── reach2d_state.yaml     # 阶段 2：state-only baseline（不需要 TODO 20）
    └── reach2d.yaml           # 阶段 3：视觉版（需要 TODO 20）
```

**动手之前先跑自检**（半分钟，不用 GPU）：

```bash
python check_setup.py
```

它会验证 env 的 observation 字段、专家策略成功率、采出来的 action chunk 有没有退化成
零填充、按每个 config 构建模型并前向、以及三个 TODO 的状态。

### 条件表示的约定（重要）

| 阶段 | config | 送进模型的 state | 维度 | 干扰物信息从哪来 |
|------|--------|-----------------|------|-----------------|
| 阶段 2 | `reach2d_state.yaml` | `concat(agent_pos, target_pos)` | 4 | **没有** → 学不会避障 |
| 阶段 3 | `reach2d.yaml` | `agent_pos`（本体感知） | 2 | 图像 |

这个划分是 ablation 的基础：两阶段的成功率差值就是"视觉带来了什么"的答案。
所以 `env.observation()` 里**故意不提供干扰物坐标**——给了这个对照实验就没意义了。
train.py 和 eval.py 都从 `obs_utils.py` 取条件表示，不要各写一份，否则很容易出现
"训练 loss 好看但闭环成功率接近 0"这种极难查的 bug。

---

## 环境配置

```bash
pip install torch torchvision pyyaml tqdm matplotlib numpy
# 可选：用 pygame 渲染（这里我们直接用 numpy 画图）
```

不需要 mujoco / Isaac Gym 等仿真器；env.py 自己处理碰撞与渲染。

## AutoDL 运行与实时监控

项目目录中提供了 `run_experiments.py` 和只读 `monitor_dashboard.py`。在 AutoDL
上建议把两者放在 `tmux` 会话中运行：

```bash
python -u monitor_dashboard.py --run-root runs/project5 --total-runs 5 --port 18765
python -u run_experiments.py --preset full > logs/full_run.log 2>&1
```

本地使用 SSH 隧道访问页面（端口和实例 SSH 指令按实际值替换）：

```bash
ssh -N -L 18765:127.0.0.1:18765 root@<autodl-host> -p <ssh-port>
```

然后打开 `http://127.0.0.1:18765/`。页面每 2 秒读取训练状态、loss、GPU、评估
结果和 rollout 图片；页面本身没有训练控制接口。

### 严格控制变量的复现实验

`run_fair_experiments.py` 用同一个 4×4 spatial CNN、专家数据和测试 episodes
重跑算法与 chunk 消融。默认使用 cosine noise schedule、padding mask、训练 seed
42，并在每次评估中固定环境与策略采样噪声。评估报告会给出 100 个 episode 的
Wilson 95% 区间；单个训练 seed 无法估计训练随机性的方差：

```bash
# 先做 20-step smoke test
python run_fair_experiments.py --preset algorithms --seed 42 \
  --episodes 5 --steps 20 --run-root runs/fair_smoke

# 完整 10 条件 × 1 seed 矩阵，可断点继续
python -u monitor_dashboard.py --run-root runs/project5_fair \
  --total-runs 10 --port 18765
bash launch_after_gpu_idle.sh
```

矩阵和验收规则见 `FAIR_EXPERIMENT_PLAN.md`。汇总会持续写入
`runs/project5_fair/fair_summary.json` 和 `fair_ablation.md`。

### 最新公平重跑结果

本次单训练种子（42）的 10 个条件均完成 10,000 步训练，并在固定的 100 个
测试 episode（seed 10000–10099）上评估。H=16 下 BC、FM、DDPM 成功率分别为
88%、87%、81%；DDPM 的 H=8 为 89%（masked）/91%（unmasked），H=32/64 的超时率
明显上升。详细区间、失败率与复现输出见
[`results/fair_seed42/fair_ablation.md`](results/fair_seed42/fair_ablation.md)。
Wilson 区间反映 episode 抽样误差；单个训练种子不能估计训练随机性，因此这些数字
不应解读为跨种子显著性结论。本节公平重跑是对照分析的最新结果；上文早期消融属于
先导实验，实验协议不同，不应与本节数值直接混比。

---

## 任务流

### 阶段 1：阅读 + 设计（2-3 天）

1. 重读 L17 + paper note 18 (OpenVLA) + 19 (Pi-0)
2. 跑通 nb12（2D action diffusion 玩具实验）
3. 写 design note，决定：
   - Action representation: chunk size H=32 or 64?
   - Vision encoder: small CNN or ImageNet-pretrained ResNet?
   - Diffusion scheduler: DDPM (50 steps) or FM (10 steps)?

---

### 阶段 2：跑通 baseline（不带视觉，TODO 19）

先不用图像，只用 `(agent_pos, target_pos)` 做条件——**不含干扰物坐标**，
所以模型知道往哪走、但不知道要躲什么。

只需要完成 `train.py` 的 **TODO 19**，不用碰 TODO 20：

```python
# 给定 obs (state), generate action chunk via DDPM
loss = MSE(eps_pred, eps_gt)
# 其中 eps_pred = model(action_noisy, t, obs)
```

```bash
python train.py --config configs/reach2d_state.yaml
python eval.py  --config configs/reach2d_state.yaml --ckpt ckpts_state/model_final.pt
```

预期：能学到"朝目标方向走"，但撞干扰物的比例明显偏高——**这正是阶段 3 要解决的问题**。

> 别用 `configs/reach2d.yaml` 做这一步：它 `use_vision: true`，
> 构建模型时就会撞上 TODO 20 的 `NotImplementedError`。

---

### 阶段 3：加视觉（TODO 20）

完成 vision encoder。建议先用一个小 CNN：
```python
class VisionEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32→32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), # 32→16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),# 16→8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, out_dim),
        )
    def forward(self, x): return self.cnn(x)
```

把 vision features 与 state 一起作为 condition。预期：成功率 +20-30%。

`DiffusionPolicy` 的融合部分已经写好了——你只要实现 `VisionEncoder`，
**把网络赋给 `self.net`**（现成的 `forward` 调的就是它）。

```bash
python train.py --config configs/reach2d.yaml
python eval.py  --config configs/reach2d.yaml --ckpt ckpts/model_final.pt
```

⚠️ 阶段 3 的 state 只有 `agent_pos`（2 维），不再含 target ——
这是真实 VLA 的设置：机器人知道自己在哪，目标和障碍都得从相机看出来。
所以阶段 3 的模型必须**从图像里同时读出目标和干扰物**，比阶段 2 难，
一开始成功率低于 baseline 是正常的，多训一会儿再看。

---

### 阶段 4：闭环评估（TODO 21）

完成 `eval.py`：
- 跑 100 个 episode
- 每 episode：reset env → policy rollout → 检测成功 / 失败 / 碰撞
- 统计：成功率、平均 step 数、碰撞率

**重要**：评估是 **closed-loop**——policy 每 K 步重新规划（action chunking 的关键）。

参考 evaluation pseudocode：
```python
def evaluate(policy, env, n_episodes=100, chunk_size=32, exec_steps=10):
    successes, collisions = 0, 0
    for ep in range(n_episodes):
        obs = env.reset()
        for step in range(env.max_steps):
            if step % exec_steps == 0:
                action_chunk = policy.sample(obs, chunk_size)
            action = action_chunk[step % exec_steps]
            obs, reward, done, info = env.step(action)
            if info['collision']: collisions += 1; break
            if info['success']: successes += 1; break
    return successes / n_episodes, collisions / n_episodes
```

---

### 阶段 5：实验对比（3-5 天）

跑 ablation：

| 实验 | 配置 | 预期成功率 |
|------|------|-----------|
| Pure BC (no diffusion) | MSE on action | ~30% |
| DDPM (no vision) | TODO 19 only | ~50% |
| DDPM + vision | + TODO 20 | ~75% |
| FM + vision | replace DDPM with FM | ~75% (但快 5x) |
| Chunk 8 vs 32 vs 64 | varying H | 见下方说明 |

写 experiment_log。

> **这些数字是参考量级，不是验收线。** 实测下来专家策略本身成功率约 99%，
> 但 policy 学到多少取决于你的 encoder、训练步数和重规划频率，浮动很大。
> 报告里写你**实际测到**的数，并说明评估用的 episode 数与 seed 区间。
>
> **关于 chunk size**：本环境专家轨迹长度中位数只有 23 步（范围 11-54），
> 而默认 `chunk_size: 32` 已经比大多数轨迹还长——数据里约 54% 的 action 槽位是
> 补出来的（重复末帧）。所以 chunk 64 会有约四分之三是 padding，
> "64 略好"这个预期**大概率复现不出来**。这本身就是个值得写进报告的发现：
> action chunk 的长度应该和任务的时间尺度匹配，盲目加长只会稀释监督信号。
> 建议扫 8 / 16 / 32，并在报告里给出各自的 padding 占比。

---

## 评分标准

- **TODO 19 完成 + 训练收敛**：25 分
- **TODO 20 完成 + 视觉模块工作**：25 分
- **TODO 21 完成 + 评估合理**：20 分
- **成功率 ≥ 70% on test envs**：15 分
- **实验报告 + ablation table**：15 分

满分 100。

---

## 加分项

- 实现 Flow Matching 替代 DDPM（+10）
- 实现 multi-modal demo 数据集（多个 target，模型能展示双模态）（+10）
- 用真实 ImageNet-pretrained ResNet18 替换小 CNN，比较收敛速度（+5）
- 在更难的环境（如 obstacles 移动）上测试 generalization（+10）

本次提交已完成以上四项，并保留了可复现实验材料：

| 加分项 | 配置/入口 | 已提交结果 |
|---|---|---|
| Flow Matching | `configs/reach2d_16.yaml` + `run_experiments.py --preset full` | `results/eval_fm.json`，57% |
| 多模态 demo | `configs/reach2d_multimodal.yaml` | `results/eval_multimodal.json`，97%；含 `mode_stats` |
| ImageNet ResNet18 | `configs/reach2d_resnet18.yaml` 或 `_long.yaml` | `results/eval_resnet18*.json`，记录 3%/2% 的真实对照结果 |
| 移动障碍泛化 | `configs/reach2d_moving.yaml` | `results/eval_moving.json`，72% |

在 AutoDL 上复现实验：

```bash
python -u run_bonus_experiments.py --preset all > logs/bonus_run.log 2>&1
```

也可按 `multimodal`、`resnet18`、`resnet18-long`、`moving` 单独运行。动态障碍 preset 默认复用 `ckpts/model_final.pt`，不需要重新采 demo。完成后运行 `python plot_bonus.py` 可生成 `results/bonus_summary.png`。报告中的 ResNet18 数字是负结果，未做挑选或隐藏。

---

## 常见问题

**Q: Policy 训完只会朝中心走，不去 target？**
A: **先确认模型到底看没看到 target。** 阶段 2 必须用 `configs/reach2d_state.yaml`
（state = `concat(agent_pos, target_pos)`，4 维）；如果 state 只有 `agent_pos`，
模型根本没有目标信息，学到的只能是"所有 demo 动作的平均"——看起来就是朝中心漂。
`python check_setup.py` 会打印当前的条件模式和 `state_dim`，先核对这一行。
排除之后再看：(1) demo 数量够不够；(2) obs/action 归一化是否与 dataset.py 一致。

**Q: Vision encoder 不收敛？**
A: 看 LR——CNN 通常需要比 MLP 大 5-10× 的 LR；同时 BatchNorm 在 small batch 下不稳，换 GroupNorm。
另外确认 `VisionEncoder` 的输出量级正常（print 一下 mean/std）：
如果最后一层后面还接了未初始化好的 Linear，特征可能整体偏爆或全塌成 0。

**Q: 评估时 policy 卡死（动不了）？**
A: 大概率是**训练目标里 padding 太多**。轨迹末尾不足一个 chunk 时如果用零填充，
而本环境轨迹中位数只有 23 步、chunk 却是 32，会有一半以上的回归目标是 0——
模型学到的最优解就是"输出接近 0"，闭环时自然不动。
`dataset.py` 现在用**重复最后一个 action** 补齐（Diffusion Policy / ACT 的标准做法），
`check_setup.py` 也会检查这一点。如果你改过 `collect_demos`，先回头核对填充方式。
其次才是检查 noise schedule 和 action 反归一化。

**Q: 闭环成功率远低于训练 loss 给人的预期？**
A: 训练和评估的条件表示不一致是头号原因——图像归一化写法不同、
或者 state 一边用了 `agent_pos` 另一边用了 `concat(agent_pos, target_pos)`。
两边都走 `obs_utils.py` 就能避免。其次检查评估 seed 是否与训练数据重叠
（`collect_demos` 用的是 `0..2*n_demos`，评估请用 1000 以上）。

**Q: 我的 task 太简单，没意义？**
A: 这个 toy task 是 **Pi-0** 在真机器人上的简化版。同样的代码框架，把 env 换成 LIBERO / RLBench，把 vision 换成 ResNet18，就能上真任务。

---

## 提交要求

**截止前将以下内容 push 到你的作业仓库 main 分支**，助教直接在仓库里评分。

```
（仓库根目录）
├── model.py / train.py / eval.py   # 含你实现的 TODO 19-21
├── results/
│   ├── ablation.md                 # ablation 表（实测数字）
│   ├── eval_state.json             # 阶段 2 评估结果
│   ├── eval_vision.json            # 阶段 3 评估结果
│   └── rollouts/                   # ≥3 个成功 rollout 的 trajectory plot
├── ckpts/model_final.pt            # 训好的 checkpoint（本模型很小，几 MB，可以提交）
├── logs/exp_log.md                 # 实验日志（按 experiment_log_template.md）
├── report.md                       # 4-6 页报告
└── debug_log.md                    # 至少 3 条踩坑记录
```

`report.md` 需包含：
1. Design note（阶段 1 的决策：chunk size / encoder / scheduler，以及为什么）
2. ablation table（实测数字 + 评估 episode 数与 seed 区间）
3. 训练曲线 + 阶段 2 vs 阶段 3 的对比分析
4. 下面的「自查问题」
5. Reflection（300 字）：Pi-0 / OpenVLA 在工业部署上需要哪些 toy 实验里没体现的工程？

> demo 数据（`data/*.pkl`，约 600 MB）不要提交，`.gitignore` 已经拦了。

---

## 自查问题（在报告中回答）

1. **Action chunking**：为什么要一次预测 H 步而不是每步预测一步？
   如果 `exec_steps` 设成 1（每步都重规划），会有什么代价和好处？（L17 §8）
2. **多模态**：绕开干扰物有左右两条合理路径。用 MSE 直接回归动作（Pure BC）
   在这种情况下会输出什么？为什么 diffusion 不会有这个问题？
3. **条件**：阶段 2 与阶段 3 的成功率差多少？这个差值衡量的到底是什么？
   如果把干扰物坐标也加进 state，这个实验还能说明问题吗？
4. **padding**：本环境轨迹中位数 23 步而 chunk 是 32，超过一半的监督信号来自补出来的动作。
   这对学到的策略有什么影响？你会怎么改进？
5. **FM vs DDPM**（做了加分项再答）：Pi-0 用 flow matching 而不是 DDPM 做 action head，
   在**实时控制**这个场景下，这个选择的关键收益是什么？（L12 §9）

---

## 学术诚信

⚠️ Diffusion Policy 的官方实现是公开的，但：

- TODO 19 约 5 行、TODO 20 约 10 行、TODO 21 约 30 行，抄了这个 capstone 就白做了
- 允许参考结构，但必须自己写、自己调
- 直接复制粘贴并提交将记 0 分
- 报告里的"踩坑记录"必须真实——抄来的踩坑是看得出来的

---

## 后续方向（不是作业，但鼓励）

完成 Project 5 后你可以：
1. **挑战**：在 RLBench / LIBERO / robotsuite 上跑 OpenVLA 或 Octo
2. **扩成研究课题**：本项目的 ablation 已经是一份像样的 preliminary result，
   补上更真实的环境和基线就能撑起一个完整的研究计划
3. **论文方向**：
   - "动态环境下的 action chunking 重规划"
   - "极少示教的 VLA LoRA 微调"
   - "Diffusion world model + diffusion policy 的端到端训练"

---

> 完成 Project 5 = 你已经做了"简化版 Pi-0"。下一步是把 64×64 toy env 换成真机器人 — 工作量不变，只是数据采集变贵。
