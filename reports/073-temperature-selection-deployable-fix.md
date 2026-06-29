# 073 - Temperature and Deployable Selection

## Context

From report 072:

- Euler steps `10 -> 80` did not improve long-horizon ADE.
- Single-sample full ADE was around `0.8m`.
- Oracle best-of-8 reached `0.32m`, proving good long-horizon trajectories exist in the sample distribution.
- Remaining problem: choose or aggregate without GT.

This report tests inference-only fixes on W2 best checkpoint:

`outputs/action_expert/student_ae28/stage0_overfit_32_s3000_seed42_recipe_draw16_full444k_retry_20260531/best.pt`

Diagnostic script:

- `scripts/100_y_temperature_selection_diagnostics.py`
- Y1 output: `outputs/action_expert/y1_temperature_single_full32/summary.json`
- Y2 output: `outputs/action_expert/y2_temperature_selection_full32/summary.json`
- N16 output: `outputs/action_expert/y2_temperature_selection_n16_full32/summary.json`

## Code Path

Official FlowMatching sampler exposes `temperature`:

```python
def sample(..., temperature: float = 1.0, ...):
    ...
    return self._euler(..., temperature=temperature)
```

Official Euler sampler applies it as initial noise scale:

```python
x = torch.randn(batch_size, *self.x_dims, device=device) * temperature
```

I added inference options to `scripts/84_train_student_ae28_official.py`:

- `--eval-temperature`
- `--eval-selection-method {single,oracle_best,medoid,mean_traj}`

Default behavior remains unchanged:

- `eval_temperature=1.0`
- `eval_selection_method=single`

## Y1 - Temperature Sweep, Single Path

32 eval samples, 10 Euler steps, one sampled path.

| temperature | full ADE | full ADE p50 | full FDE | h1.6 ADE | h3.2 ADE |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.8061 | 0.6374 | 2.1512 | 0.0742 | 0.2436 |
| 0.85 | 0.7365 | 0.6216 | 1.9852 | 0.0665 | 0.2197 |
| 0.7 | 0.6990 | 0.6339 | 1.9400 | 0.0579 | 0.1990 |
| 0.5 | 0.6836 | 0.5632 | 1.9952 | 0.0495 | 0.1780 |
| 0.3 | 0.7331 | 0.6163 | 2.2644 | 0.0434 | 0.1686 |

Verdict: lowering temperature helps, but single-path inference still does not reach strict `<0.5m`. Best single-path setting here is `temperature=0.5`, full ADE `0.6836m`.

## Y2 - N=8 Deployable Selection

32 eval samples, 10 Euler steps, 8 sampled paths. `oracle_best` is diagnostic only; `medoid` and `mean_traj` are deployable.

The official sampler returns only the sampled tensor, not a likelihood or per-path FM score, so I tested the two GT-free post-processing choices available without model changes: geometric medoid and mean trajectory.

| temperature | single | oracle best | medoid | mean traj | path ADE std |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.8630 | 0.3582 | 0.6160 | 0.5700 | 0.4455 |
| 0.85 | 0.7912 | 0.3397 | 0.5629 | 0.5481 | 0.3680 |
| 0.7 | 0.7326 | 0.3256 | 0.5744 | 0.5571 | 0.3186 |
| 0.5 | 0.7141 | 0.3569 | 0.6081 | 0.5958 | 0.2576 |
| 0.3 | 0.7452 | 0.4526 | 0.6695 | 0.6726 | 0.1891 |

Verdict: N=8 deployable selection improves over single path, but medoid/mean still do not pass strict `<0.5m`. The best deployable N=8 result is `mean_traj @ temperature=0.85`, full ADE `0.5481m`.

## Y3 - N=16 Temperature + Selection

Because N=8 mean was close, I tested N=16 at the two best temperatures.

| temperature | single | oracle best | medoid | mean traj | path ADE std |
|---:|---:|---:|---:|---:|---:|
| 0.85 | 0.6439 | 0.2179 | 0.5040 | 0.4564 | 0.3621 |
| 0.7 | 0.5983 | 0.2202 | 0.5099 | 0.4713 | 0.3095 |

Horizon breakdown for the winning deployable setting:

`temperature=0.85`, `num_paths=16`, `selection=mean_traj`

| horizon | ADE | FDE |
|---|---:|---:|
| h1.6 | 0.0378 | 0.0941 |
| h3.2 | 0.1252 | 0.3538 |
| h6.4 | 0.4564 | 1.3494 |

Verdict: N=16 mean trajectory is the first GT-free deployable method that passes strict Stage 0. Medoid is nearly enough but stays just above threshold (`0.5040m`).

