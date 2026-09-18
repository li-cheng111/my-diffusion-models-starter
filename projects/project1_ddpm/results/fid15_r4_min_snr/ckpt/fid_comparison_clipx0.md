# CIFAR-10 EMA comparison

- Real split: `train`
- Real samples: `5000`
- Generated samples per run: `5000`
- Seed: `44`

- Clip predicted x0: `True`

| Weights | FID |
|---|---:|
| EMA_0.9990 | 16.3013 |
| EMA_0.9995 | 15.7954 |
| EMA_0.9999 | 16.5633 |
| raw | 29.1437 |
| raw - EMA_0.9990 | +12.8423 |
| raw - EMA_0.9995 | +13.3483 |
| raw - EMA_0.9999 | +12.5804 |
