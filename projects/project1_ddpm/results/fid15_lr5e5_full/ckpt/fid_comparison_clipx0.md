# CIFAR-10 EMA comparison

- Real split: `train`
- Real samples: `5000`
- Generated samples per run: `5000`
- Seed: `44`

- Clip predicted x0: `True`

| Weights | FID |
|---|---:|
| EMA_0.9990 | 17.4472 |
| EMA_0.9995 | 17.3407 |
| EMA_0.9999 | 16.5231 |
| raw | 18.6642 |
| raw - EMA_0.9990 | +1.2171 |
| raw - EMA_0.9995 | +1.3235 |
| raw - EMA_0.9999 | +2.1411 |
