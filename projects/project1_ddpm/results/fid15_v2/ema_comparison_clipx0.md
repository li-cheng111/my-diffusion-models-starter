# R3 EMA decay comparison（clip x0）

- checkpoint：`runs/fid15_v2_full_cosine_default/ckpt/final.pt`
- real split：`train`
- real/generated samples：`5000 / 5000`
- seed：`44`
- clip predicted x0：`true`

| 权重 | FID | raw - weight |
|---|---:|---:|
| EMA 0.9990 | 15.7886 | +50.8515 |
| EMA 0.9995 | **15.5340** | +51.1061 |
| EMA 0.9999 | 16.1686 | +50.4715 |
| raw | 66.6401 | — |

EMA 0.9995 是本轮固定协议下的最佳结果，但仍比目标 FID≤15 高 `0.5340`，因此挑战档
目标未达成。raw 与 EMA 的差异说明 EMA 对该模型的采样质量有显著帮助。
