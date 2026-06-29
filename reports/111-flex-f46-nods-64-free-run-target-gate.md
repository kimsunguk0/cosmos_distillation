# 111 - FLEX F46 No-DeepStack 64 Free-Run Target Gate

Date: 2026-06-07

## Purpose

Find the scale break point between:

- F42: 16-sample B0 free-run parity success, `0.380 m` ADE
- F45: 256-sample B0 free-run parity failure, final `3.249 m` ADE

This run keeps the F45 recipe but uses the first 64 samples.

## Run

- tmux session: `flex_f46_nods64`
- script: `scripts/tmp_run_flex_f46_nods_free_run_64_from_f42_chain.sh`
- log: `outputs/logs/flex_f46_nods_free_run_target64_from_f42_s8000_lr2e7_20260607_chain.log`
- corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- B0 target summary: `outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json`
- student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- output checkpoint: `outputs/checkpoints/flex_f46_nods_free_run_target64_from_f42_s8000_lr2e7_20260607`

## Settings

- samples: `64`
- steps: `8000`
- LR: `2e-7`
- actual compressed DeepStack: off
- trainable: FLEX scene encoder + last 4 LoRA layers + multimodal projector
- loss: B0 free-run token CE + trajectory state alignment

## Decision Logic

- If 64 passes near F42 quality: F45 failure is scale/capacity/generalization pressure; try staged curriculum 16 -> 64 -> 256.
- If 64 fails like F45: the objective breaks well before 256; next branch must change objective/placement, not data size.

Current status:

`running: tmux session started`

Live checks:

- tmux: `flex_f46_nods64`
- start event logged
- collated cache complete: `64 / 64`
- teacher cache complete: `64 / 64`
- train started: actual compressed DeepStack `false`, trainable params `66,513,920`
- step 1: loss `0.2508`, free-run token acc `1.000`, traj state cosine `0.7979`
- step 300: loss `1.1925`, free-run token acc `0.914`, traj state cosine `0.5899`
- step 1900: loss `0.9920`, free-run token acc `0.926`, traj state cosine `0.6151`, no errors
- step 2000 checkpoint exists: `outputs/checkpoints/flex_f46_nods_free_run_target64_from_f42_s8000_lr2e7_20260607/step_002000`
- step 3300: loss `0.9259`, free-run token acc `0.934`, traj state cosine `0.6301`, no errors
- step 6000 checkpoint exists: `outputs/checkpoints/flex_f46_nods_free_run_target64_from_f42_s8000_lr2e7_20260607/step_006000`
- step 6300: loss `0.8828`, free-run token acc `0.941`, traj state cosine `0.6317`, no errors
- final / step 8000: loss `0.9412`, free-run token acc `0.929`, traj state cosine `0.6138`, no errors

## Result

Final B0-target parity on 64 samples:

| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| final / step 8000 | 0.367 | 2.614 | 8.729 | 19.45 | 3.06 |

Artifact:

`outputs/reports/flex_f46_nods_free_run_target64_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json`

## Interpretation

F46 fails the 64-sample gate.

It is better than F45-256 (`3.249 m` ADE) but far worse than F42-16 (`0.380 m` ADE). Therefore the break point is between 16 and 64 samples, not at full 256 scale.

Current status:

`completed: 64-sample gate failed`
