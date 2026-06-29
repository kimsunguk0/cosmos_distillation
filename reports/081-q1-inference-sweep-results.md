# 081 Q1 Inference Sweep Results

Date: 2026-06-02

## Run

- Output dir: `outputs/action_expert/q1_inference_sweep_step9000_val512_20260602_0142_aggr_b8pb8`
- Checkpoint: `outputs/action_expert/stage1_fast_resume_s5000_b8_fa2_20260601_081126/best.pt`
- Payload step: 9000
- Held-out split cache: `outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/split_cache_20k_2k_seed42.json`
- Eval samples: 512
- Eval batch size: 8
- Eval path batch size: 8
- Status: completed, 7/7 sweep items

## Results

| sweep | N | temp | selection | full ADE | p50 ADE | FDE | h1.6 ADE | h3.2 ADE | delta vs 2.50 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| single_temp1p0 | 1 | 1.00 | single | 3.081 | 2.195 | 8.821 | 0.201 | 0.795 | +0.58 |
| single_temp0p85 | 1 | 0.85 | single | 2.857 | 2.015 | 8.193 | 0.184 | 0.733 | +0.36 |
| single_temp0p7 | 1 | 0.70 | single | 2.698 | 1.860 | 7.754 | 0.171 | 0.688 | +0.20 |
| single_temp0p6 | 1 | 0.60 | single | 2.614 | 1.809 | 7.490 | 0.165 | 0.666 | +0.11 |
| single_temp0p5 | 1 | 0.50 | single | 2.550 | 1.824 | 7.338 | 0.160 | 0.647 | +0.05 |
| mean_traj_n16_temp0p85 | 16 | 0.85 | mean_traj | 2.525 | 1.868 | 7.251 | 0.160 | 0.646 | +0.02 |
| oracle_best_n16_temp0p85 | 16 | 0.85 | oracle_best | 1.316 | 0.746 | 3.906 | 0.123 | 0.402 | -1.18 |

Note: `mean_traj_n16_temp0p85` also logged best-of-16 oracle metrics from the same 16 sampled paths: full ADE 1.316, h1.6 ADE 0.071, h3.2 ADE 0.310.

## Interpretation

- Temperature helps monotonically for single-path inference, but only from 3.081 to 2.550. It does not move the deployable result into the 1.5-2.0 m range.
- `mean_traj` at N=16 gives 2.525, effectively the same as the Stage 1 plateau around 2.50.
- Oracle best-of-16 is 1.316, so the model distribution contains much better trajectories than the single/mean deployable selector is extracting.
- Therefore Q1 does not support "plateau is only because inference was not swept." The deployable inference recipes tested here do not break the plateau. The remaining gap is selection/ranking or training/data/schedule, to be separated by Q2 and Q3.

One-line Q1 result: inference sweep is completed; deployable inference remains around 2.52-2.55 m, oracle is 1.316 m, so Stage 1 plateau is not solved by temperature or mean_traj inference alone.
