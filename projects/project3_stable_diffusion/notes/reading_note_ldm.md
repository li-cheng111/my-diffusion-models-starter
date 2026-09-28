# LDM §3–4 与 SDXL 前半部分阅读笔记

## 论文与范围

- High-Resolution Image Synthesis with Latent Diffusion Models，Rombach et al.，CVPR 2022。
- SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis，Podell et al.，2023。
- 本笔记覆盖 LDM 的 latent diffusion、cross-attention 条件化，以及 SDXL 架构改动前半部分。

## LDM §3–4：从像素空间到 latent 空间

像素空间扩散直接在高维 RGB 网格上运行，计算昂贵且存在空间冗余。LDM 先训练感知压缩 autoencoder：encoder E 把图像 x 映射到较小 latent z=E(x)，decoder D 重构 x≈D(z)，扩散模型只在 z 上学习。

前向过程可写为：

~~~text
q(z_t | z_{t-1}) = N(sqrt(1-beta_t) z_{t-1}, beta_t I)
z_t = sqrt(alpha_bar_t) z_0 + sqrt(1-alpha_bar_t) epsilon
L_simple = E[ || epsilon - epsilon_theta(z_t,t,c) ||^2 ]
~~~

文本条件通过 cross-attention 注入：Q 来自当前 latent feature，K/V 来自文本 token embedding，softmax(QK^T/sqrt(d))V 将 token 语义写回空间位置。classifier-free guidance 用 unconditional 和 conditional 预测做线性外推，提高 prompt adherence。

LDM 的贡献是把扩散计算从像素空间转移到低维 latent 空间，并用 cross-attention 统一文本、类别和布局条件。限制是 autoencoder 压缩误差、latent scaling、VAE 解码质量和 tokenizer 都会影响最终细节。Project 3 中的 (1,4,64,64) latent 和 0.18215 scaling factor 正是这一设计的直接体现。

## SDXL 前半部分

SDXL 不只是增大 SD 1.5 的 UNet，还使用两个 CLIP text encoder、拼接不同 embedding 空间、pooled text embedding，以及原始尺寸、裁剪坐标和目标尺寸等 added conditioning。base model 负责构图，refiner 在低噪声阶段恢复高频纹理。

这种设计适合高分辨率生成，但需要更多显存、两个 text encoder 和更严格的尺寸 metadata。若把 Project 3 的手写流程迁移到 SDXL，不能只替换 UNet，还要同步处理两个 text encoder 和 added conditioning。

## 与 Project 3 的连接

1. A 部分的 (1,4,64,64) latent 是 LDM 的压缩扩散空间。
2. D 的 LoRA 只修改 UNet attention projection，在固定 VAE/text interface 上学习低秩风格偏移。
3. E 的 attention challenge 直接观察 QK^T 对 token 的空间响应。
4. B、C 的结果说明 sampler、VAE 压缩和 latent scaling 都会影响最终图片，单 prompt 图片不能等价为 FID 结论。

## 局限与复现实验问题

论文指标依赖大规模数据、特定 VAE 和采样器；课程实验应报告 seed、model revision、scheduler、prompt、运行环境和产物 SHA256，并区分本机 smoke 与 AutoDL full run。
