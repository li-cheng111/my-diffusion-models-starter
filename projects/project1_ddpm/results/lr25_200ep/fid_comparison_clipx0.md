# CIFAR-10 EMA comparison

- Real split: `train`
- Real samples: `5000`
- Generated samples per run: `5000`
- Seed: `44`
- Clip predicted x0: `True`

| Weights | FID |
|---|---:|
| EMA_0.9990 | 16.0761 |
| EMA_0.9995 | 15.8897 |
| EMA_0.9999 | **15.6761** |
| raw | 16.8996 |
| raw - EMA_0.9990 | +0.8235 |
| raw - EMA_0.9995 | +1.0099 |
| raw - EMA_0.9999 | +1.2235 |
