# Controlled Project 5 ablation

All rows use training seed 42, the same 4x4 spatial CNN, expert-data seed, cosine noise schedule, closed-loop protocol and paired evaluation episodes. Wilson intervals quantify episode variation for this one trained checkpoint; training-seed variation is not estimated.

| condition | method | H | masked | train seed | success (Wilson 95% CI) | collision | timeout | avg steps | final distance | path target distance |
|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| bc_h16_masked | bc | 16 | yes | 42 | 88.0% (80.2–93.0%) | 9.0% | 3.0% | 26.95 | 0.0913 | 0.4877 |
| ddpm_h16_masked | ddpm | 16 | yes | 42 | 81.0% (72.2–87.5%) | 8.0% | 11.0% | 35.57 | 0.1339 | 0.4675 |
| fm_h16_masked | fm | 16 | yes | 42 | 87.0% (79.0–92.2%) | 7.0% | 6.0% | 29.53 | 0.0987 | 0.4736 |
| ddpm_h8_masked | ddpm | 8 | yes | 42 | 89.0% (81.4–93.7%) | 7.0% | 4.0% | 27.57 | 0.0994 | 0.4852 |
| ddpm_h32_masked | ddpm | 32 | yes | 42 | 48.0% (38.5–57.7%) | 6.0% | 46.0% | 64.39 | 0.2621 | 0.4612 |
| ddpm_h64_masked | ddpm | 64 | yes | 42 | 26.0% (18.4–35.4%) | 1.0% | 73.0% | 85.86 | 0.3668 | 0.5033 |
| ddpm_h8_unmasked | ddpm | 8 | no | 42 | 91.0% (83.8–95.2%) | 7.0% | 2.0% | 26.37 | 0.0919 | 0.4887 |
| ddpm_h16_unmasked | ddpm | 16 | no | 42 | 85.0% (76.7–90.7%) | 8.0% | 7.0% | 31.50 | 0.1115 | 0.4769 |
| ddpm_h32_unmasked | ddpm | 32 | no | 42 | 66.0% (56.3–74.5%) | 7.0% | 27.0% | 54.73 | 0.2138 | 0.4716 |
| ddpm_h64_unmasked | ddpm | 64 | no | 42 | 28.0% (20.1–37.5%) | 9.0% | 63.0% | 81.96 | 0.4147 | 0.5955 |
