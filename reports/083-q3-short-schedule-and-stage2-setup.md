# 083 Q3 Short Schedule Check And Stage2 Setup

Date: 2026-06-03

## Current Q2 Status

- Stopped Q2 PID 17058 after it stalled idle at step 28100.
- Preserved checkpoints through step 28000.
- Best Q2 checkpoint:
  - `outputs/action_expert/q2_continue_s10000_to_s30000_b8pb8_20260602_0220/best.pt`
  - payload step: 26000
  - val full ADE: 2.1268 m
  - train full ADE: 1.8181 m
  - val best-of-16 oracle: 1.0204 m

Interpretation: constant LR continuation improved over the original 10k run, but 20k data is still plateauing near 2.1-2.3 m.

## Q3 Short Run

Purpose: test whether a short cosine cooldown from the Q2 best checkpoint improves the plateau behavior versus the observed constant-LR degradation after step 26000.

Output:

- `outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505`

Launch summary:

- resume checkpoint: Q2 `best.pt` from step 26000
- steps: 2000 local steps
- batch size: 8
- num_time_samples: 16
- expert_lr/proj_lr: 1e-4
- scheduler: cosine with warmup 100 steps, min_lr 1e-5
- eval: every 1000 steps
- eval samples: 512 val, 256 train
- inference: temperature 0.85, N16, mean_traj
- speed flags:
  - `--allow-train-cache-mutation`
  - `--fused-adamw`

Initial health:

- PID: 22124
- GPU memory: about 125.7 GB
- GPU util: about 99%
- `train_cache_deepcopy=0.0`
- `optimizer_created.fused=true`

Decision rule:

- If Q3 best val ADE clearly beats 2.1268 m, schedule cooldown has some value.
- If Q3 stays around 2.1-2.3 m, schedule is not the main blocker; move to 200k data.

## Stage2 200k Setup

Stage2 is set up as a clean 200k run from `student_backbone_init`, not a warm-start from Q2. This avoids leakage risk from the 20k Q2 train split into a newly created 200k validation split.

Launcher:

- `scripts/launch_stage2_ae28_200k.sh`

Watcher:

- `scripts/watch_q3_then_launch_stage2_ae28_200k.sh`
- log: `outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/stage2_after_q3_watcher.log`

Stage2 output:

- `outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603`

Stage2 split cache:

- `outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json`

Stage2 config:

- train samples: 200000
- held-out val samples: 10000
- eval samples: 1024 val, 512 train
- batch size: 8
- num_time_samples: 16
- expert_lr/proj_lr: 1e-4
- scheduler: constant LR
- eval every: 2500 steps
- total steps: 25000
- inference: temperature 0.85, N16, mean_traj
- speed flags enabled:
  - `--allow-train-cache-mutation`
  - `--fused-adamw`

One-line status: Q3 short schedule check is running; Stage2 200k clean run is queued to launch automatically after Q3 completes successfully.
