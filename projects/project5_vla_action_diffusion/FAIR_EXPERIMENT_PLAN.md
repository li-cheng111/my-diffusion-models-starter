# Project 5 controlled rerun plan

## Objective

Replace the exploratory ablation with a controlled comparison that changes one
factor at a time. Every run uses the 4x4 spatial CNN, the same expert trajectories,
cosine diffusion schedule, deterministic evaluation noise and the same closed-loop
episodes.

## Matrix

1. Algorithm comparison: BC, DDIM and FM; H=16; masked loss.
2. Chunk comparison: DDIM with H=8, 16, 32 and 64; masked loss.
3. Padding control: DDIM with H=8, 16, 32 and 64; unmasked loss.
4. Each condition uses the single fixed training seed 42 and 100 paired
   evaluation episodes (seeds 10000-10099).

The full matrix has 10 unique conditions and 10 training runs. H=16 masked DDIM
is shared by the algorithm and chunk comparisons. A single training seed does
not measure variation between training runs; Wilson intervals describe only
the 100 evaluation episodes for the resulting checkpoint.

## Commands

Smoke test before renting GPU time:

```bash
python check_setup.py
python -m pytest -q tests
python run_fair_experiments.py --preset algorithms --episodes 5 --steps 20
```

Full AutoDL run:

```bash
python -u monitor_dashboard.py --run-root runs/project5_fair --total-runs 10 --port 18765
bash launch_after_gpu_idle.sh
```

Outputs are written to `runs/project5_fair/<condition>/seed_42/`. The matrix
also maintains `fair_summary.json` and `fair_ablation.md` after every completed
evaluation, so an interrupted run still has a readable partial result.

## Acceptance criteria

- Tests verify cosine schedule endpoint, endpoint-inclusive DDIM timesteps,
  deterministic sampling, real demo masks and EMA BatchNorm buffers.
- Repeating an evaluation with the same checkpoint and seed range produces the
  same JSON metrics.
- BC/DDIM/FM rows have identical encoder, H, data, optimizer, step budget and
  evaluation episodes.
- Every chunk row reports masked and unmasked results, allowing the padding
  hypothesis to be tested directly.
- Final report includes the fixed training seed and the 95% Wilson interval
  across 100 evaluation episodes. It states that training-seed variation is not
  estimated.

## Estimated AutoDL time

| Part | Runs | RTX 4080 SUPER estimate |
|---|---:|---:|
| Data collection for H=8/16/32/64 | 4 datasets | 10-20 min |
| Algorithm comparison | 3 | 10-20 min |
| Remaining masked H sweep (H=8/32/64) | 3 | 15-30 min |
| Unmasked padding controls | 4 | 20-40 min |
| 1,000 closed-loop episodes and aggregation | included | 10-25 min |
| Total, including checks and artifact collection | 10 | about 1-2 h |

The launcher waits for the already-running Project 4 training process to exit,
then waits for three low-utilization GPU samples before starting Project 5. The
dashboard shows this queued state and counts only the ten experiment runs.
`--resume` safely continues the same matrix.
