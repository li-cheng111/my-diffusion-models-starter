# CIFAR-10 EMA comparison

- Real split: `train`
- Real samples: `5000`
- Generated samples per run: `5000`
- Seed: `44`

- Clip predicted x0: `True`

| Weights | FID |
|---|---:|
| EMA_0.9990 | 17.2559 |
| EMA_0.9995 | 16.9445 |
| EMA_0.9999 | 15.8562 |
| raw | 17.8510 |
| raw - EMA_0.9990 | +0.5952 |
| raw - EMA_0.9995 | +0.9066 |
| raw - EMA_0.9999 | +1.9948 |
