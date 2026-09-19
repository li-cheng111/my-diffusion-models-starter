# CIFAR-10 EMA comparison

- Real split: `train`
- Real samples: `5000`
- Generated samples per run: `5000`
- Seed: `44`

- Clip predicted x0: `True`

| Weights | FID |
|---|---:|
| EMA_0.9990 | 19.0443 |
| EMA_0.9995 | 18.8484 |
| EMA_0.9999 | 17.4808 |
| raw | 19.2896 |
| raw - EMA_0.9990 | +0.2453 |
| raw - EMA_0.9995 | +0.4411 |
| raw - EMA_0.9999 | +1.8087 |
