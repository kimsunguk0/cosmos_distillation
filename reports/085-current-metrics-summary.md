# 085 Current Metrics Summary

Generated: 2026-06-05

## Paper Baselines

User-provided AlpamayoR1 paper/reference baselines:

| Model | Training | ADE | minADE@6.4s |
|---|---|---:|---:|
| AlpamayoR1 0.5B | SFT only | 2.12 | 0.913 |
| AlpamayoR1 10B | SFT only | - | 0.849 |

Notes:

- Our `minADE@6` below means best-of-6 trajectory ADE over the full 6.4s / 64 waypoint horizon.
- Our older training logs used `eval_num_paths=16`; those are `minADE@16`, not paper-comparable `minADE@6`.
- `single` eval is paper-style single selected path with best-of-6 diagnostic also logged.
- `mean_traj N16` is the earlier deployable multi-path aggregation recipe and is not the same metric as paper-style single/N6 eval.

## Best Numbers So Far

| Run | Eval set | Selection | Temp | Paths | ADE | minADE | FDE | minFDE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Q3 | Stage1 val 512 | single | 1.0 | 6 | 2.631 | 1.243 | 7.574 | 3.591 |
| Q3 | Stage1 val 512 | single | 0.85 | 6 | 2.506 | 1.276 | 7.239 | 3.691 |
| Stage2 200k best.pt | Stage2 val 1024 | single | 1.0 | 6 | 2.970 | 1.364 | 8.732 | 4.019 |
| Stage2 200k best.pt | Stage2 val 1024 | single | 0.85 | 6 | 2.768 | 1.350 | 8.129 | 3.951 |
| Stage2 200k final.pt | Stage2 val 1024 | single | 1.0 | 6 | 3.032 | 1.343 | 8.827 | 3.977 |
| Stage2 200k final.pt | Stage2 val 1024 | single | 0.85 | 6 | 2.814 | 1.339 | 8.214 | 3.963 |

Current read:

- Best paper-style ADE among these: Q3 temp 0.85, ADE 2.506.
- Best paper-style minADE@6 among these: Q3 temp 1.0, minADE@6 1.243.
- Stage2 200k 1-epoch does not yet beat Q3 on paper-style ADE or minADE@6.
- Stage2 final.pt slightly improves minADE@6 over Stage2 best.pt, but worsens ADE.

## Q2 Results

Run:

- `outputs/action_expert/q2_continue_s10000_to_s30000_b8pb8_20260602_0220`
- Purpose: continue the 20k held-out Stage1 run for more epochs on the same data.
- Eval condition: temp 0.85, `eval_num_paths=16`, `mean_traj`.
- Important: Q2 has no paper-style `minADE@6` eval yet in current artifacts.

| Metric pick | Step | ADE | minADE@16 | FDE |
|---|---:|---:|---:|---:|
| Best ADE | 26000 | 2.127 | 1.020 | 6.189 |
| Best minADE@16 | 21000 | 2.140 | 1.011 | 6.214 |
| Last logged val | 28000 | 2.281 | 1.061 | 6.805 |

Recent curve:

| Step | ADE | minADE@16 | FDE |
|---:|---:|---:|---:|
| 24000 | 2.177 | 1.073 | 6.339 |
| 25000 | 2.228 | 1.021 | 6.483 |
| 26000 | 2.127 | 1.020 | 6.189 |
| 27000 | 2.271 | 1.066 | 6.686 |
| 28000 | 2.281 | 1.061 | 6.805 |

## Q3 Results

Run:

- `outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505`
- Purpose: LR schedule / cooldown test from Q2 best.
- Training-log eval condition: temp 0.85, `eval_num_paths=16`, `mean_traj`.

Training-log N16 result:

| Step | ADE | minADE@16 | FDE |
|---:|---:|---:|---:|
| 1000 | 2.174 | 1.034 | 6.328 |
| 2000 | 2.105 | 0.957 | 6.094 |

Paper-style Q3 N6 eval:

- `outputs/action_expert/q3_minade6_temp_sweep_seed42_evalbase1042_20260604_145757_q3`
- Eval set: Stage1 held-out val 512.

| Temp | Selection | Paths | ADE | minADE@6 | FDE | minFDE@6 |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | single | 6 | 2.631 | 1.243 | 7.574 | 3.591 |
| 0.85 | single | 6 | 2.506 | 1.276 | 7.239 | 3.691 |

## Stage2 200k 1-Epoch Results

Run:

- `outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603`
- Data: train 200k, val 10k split cache.
- Training length: 25000 steps = 1 epoch at batch size 8.
- Eval during training: temp 0.85, `eval_num_paths=16`, `mean_traj`.

Training-log N16 result:

| Metric pick | Step | ADE | minADE@16 | FDE |
|---|---:|---:|---:|---:|
| Best ADE | 15000 | 2.221 | 1.002 | 6.545 |
| Best minADE@16 | 25000 | 2.273 | 0.989 | 6.667 |
| Final eval | 25000 | 2.273 | 0.989 | 6.667 |

Recent curve:

| Step | ADE | minADE@16 | FDE |
|---:|---:|---:|---:|
| 15000 | 2.221 | 1.002 | 6.545 |
| 17500 | 2.330 | 0.995 | 6.831 |
| 20000 | 2.318 | 1.019 | 6.845 |
| 22500 | 2.539 | 1.175 | 7.532 |
| 25000 | 2.273 | 0.989 | 6.667 |

## Stage2 200k Paper-Style N6 Eval

Stage2 best.pt:

- Checkpoint payload step: 15000.
- Eval artifact: `outputs/action_expert/stage2_200k_best_minade6_eval_20260604_145757_stage2`
- Eval set: Stage2 held-out val 1024, train 512.

| Split | Temp | ADE | minADE@6 | FDE | minFDE@6 |
|---|---:|---:|---:|---:|---:|
| val | 1.0 | 2.970 | 1.364 | 8.732 | 4.019 |
| val | 0.85 | 2.768 | 1.350 | 8.129 | 3.951 |
| train | 1.0 | 2.871 | 1.369 | 8.545 | 4.124 |
| train | 0.85 | 2.685 | 1.384 | 7.972 | 4.145 |

Stage2 final.pt:

- Checkpoint payload step: 25000.
- Eval artifact: `outputs/action_expert/stage2_200k_final_minade6_eval_20260605_final_eval_more2ep`
- Eval set: Stage2 held-out val 1024, train 512.

| Split | Temp | ADE | minADE@6 | FDE | minFDE@6 |
|---|---:|---:|---:|---:|---:|
| val | 1.0 | 3.032 | 1.343 | 8.827 | 3.977 |
| val | 0.85 | 2.814 | 1.339 | 8.214 | 3.963 |
| train | 1.0 | 2.900 | 1.359 | 8.504 | 4.096 |
| train | 0.85 | 2.706 | 1.375 | 7.972 | 4.190 |

## Stage2 200k Additional 2-Epoch Run

Current live run:

- `outputs/action_expert/stage2_200k_more2ep_b8_nt16_minade6_20260605_final_eval_more2ep`
- Resume checkpoint: Stage2 1-epoch `final.pt`
- Step range: 25000 -> 75000
- Target total: 3 epochs over 200k samples.
- Eval condition: temp 1.0, `eval_num_paths=6`, `single`.
- This run is the first one whose training-time eval is directly paper-style `minADE@6`.

Current status at collection time:

| Field | Value |
|---|---:|
| Latest train step | 25200 |
| Latest loss | 0.185 |
| pred_v_abs_mean | 0.828 |
| target_v_abs_mean | 0.883 |
| Latest val eval | not reached yet |

## Comparison Against Paper Baseline

Closest paper-style numbers so far:

| Candidate | ADE | Delta vs 0.5B ADE 2.12 | minADE@6 | Delta vs 0.5B 0.913 | Delta vs 10B 0.849 |
|---|---:|---:|---:|---:|---:|
| Q3 temp 1.0 | 2.631 | +0.511 | 1.243 | +0.330 | +0.394 |
| Q3 temp 0.85 | 2.506 | +0.386 | 1.276 | +0.363 | +0.427 |
| Stage2 best temp 0.85 | 2.768 | +0.648 | 1.350 | +0.437 | +0.501 |
| Stage2 final temp 0.85 | 2.814 | +0.694 | 1.339 | +0.426 | +0.490 |

Current conclusion:

- Q3 remains the best current paper-style result.
- Stage2 200k 1-epoch did not beat Q3, despite more data.
- The active 2-more-epoch Stage2 run is the decisive check for whether the 200k split benefits from the official 3-epoch training length.
- If Stage2 more2ep does not approach Q3 by the first few evals, the bottleneck is likely not just epoch count; revisit LR schedule, checkpoint choice, and deployable selection.

