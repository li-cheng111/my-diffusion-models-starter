# CIFAR-10 EMA comparison

- Real split: `train`
- Real samples: `5000`
- Generated samples per run: `5000`
- Seed: `44`

- Clip predicted x0: `True`

| Weights | FID |
|---|---:|
| EMA_0.9990 | 16.3823 |
| EMA_0.9995 | 16.2361 |
| EMA_0.9999 | 15.4385 |
| raw | 16.1927 |
| raw - EMA_0.9990 | -0.1896 |
| raw - EMA_0.9995 | -0.0434 |
| raw - EMA_0.9999 | +0.7542 |
