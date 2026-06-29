# 110 - FLEX F45 No-DeepStack 256 Free-Run Target Gate

Date: 2026-06-07

## Purpose

Scale the current usable FLEX branch from 16-sample B0 free-run parity to the full 256-sample heldout FLEX corpus.

This deliberately keeps actual compressed DeepStack off. Report 109 showed that actual compressed DeepStack is not ready:

- repeated scene DeepStack: `9.575 m` ADE
- rank64 projector best: `0.534 m` ADE
- current best without actual compressed DeepStack: `0.380 m` ADE

## Run

- tmux session: `flex_f45_nods256`
- script: `scripts/tmp_run_flex_f45_nods_free_run_256_from_f42_chain.sh`
- log: `outputs/logs/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607_chain.log`
- corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- teacher/B0: `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`
- student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- output checkpoint: `outputs/checkpoints/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607`
- B0 target summary: `outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json`

## Settings

- samples: `256`
- split: `val`
- prompt mode: `joint`
- target mode: `traj_only`
- steps: `8000`
- batch size: `1`
- LR: `2e-7`
- trainable: FLEX scene encoder + last 4 LoRA layers + multimodal projector
- actual compressed DeepStack: off
- loss: B0 free-run token CE + trajectory state alignment

## Pass Criteria

Primary gate:

- final 256-sample B0 free-run parity materially below the old F1 512 free-run ADE (`3.984 m`)
- no repetition collapse
- ideally close enough to F42's 16-sample `0.380 m` to justify moving to a 512 or full-val branch

Interpretation:

- If 256 overfit stays good: the F42 recipe is not just a 16-sample artifact.
- If 256 overfit fails: the remaining issue is not actual DeepStack; the free-run token/state objective or adaptation capacity does not scale beyond tiny overfit.

Current status:

`completed: 256-sample gate failed`

Live checks at launch:

- tmux: `flex_f45_nods256`
- checkpoint eval watcher tmux: `flex_f45_nods256_ckpteval`
- target decode PID: `4580`
- progress checks: `4 / 256`, `29 / 256`, `49 / 256`, `69 / 256`, `99 / 256`, `136 / 256`, `179 / 256`, `216 / 256`, `248 / 256`
- target decode complete: `256 / 256`
- target summary validated: `target_mode=traj_only`, `image_ablation=normal`, `samples=256`, trajectory token length `128`
- `free_run_token_targets_loaded`: `256`
- train initialization: collated cache complete `256 / 256`; teacher cache complete `256 / 256`
- train started: actual compressed DeepStack `false`, trainable params `66,513,920`
- trainable groups: FLEX scene encoder `31,378,432`, language LoRA `9,961,472`, multimodal projector `25,174,016`
- step 1: loss `0.2508`, free-run token acc `1.000`, traj state cosine `0.7979`
- step 1300: loss `0.9694`, free-run token acc `0.916`, traj state cosine `0.6488`, no errors
- step 2000 checkpoint exists: `outputs/checkpoints/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607/step_002000`
- live step 2000 eval started in tmux `flex_f45_s2000_eval_now`; observed `34 / 256`
- step 2800 train status: loss `0.9772`, free-run token acc `0.906`, traj state cosine `0.6527`, no errors
- live step 2000 eval later observed `89 / 256`
- step 3400 train status: loss `0.9700`, free-run token acc `0.914`, traj state cosine `0.6333`, no errors
- step 4000 checkpoint exists: `outputs/checkpoints/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607/step_004000`
- live step 2000 eval later observed `156 / 256`
- step 4200 train status: loss `0.9857`, free-run token acc `0.919`, traj state cosine `0.6113`, no errors
- live step 2000 eval later observed `223 / 256`
- step 5000 train status: loss `0.9246`, free-run token acc `0.920`, traj state cosine `0.6398`, no errors
- step 5700 train status: loss `0.9391`, free-run token acc `0.917`, traj state cosine `0.6376`, no errors

## Interim Result

Step 2000 B0-target parity on 256 samples:

| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| step 2000 | 0.221 | 3.468 | 10.957 | 12.34 | 3.81 |
| step 4000 | 0.240 | 3.520 | 11.208 | 12.14 | 3.77 |
| final / step 8000 | 0.271 | 3.249 | 10.362 | 11.77 | 2.96 |

Artifact:

`outputs/reports/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607_step_002000_b0_trajonly_parity_summary.json`

`outputs/reports/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607_step_004000_b0_trajonly_parity_summary.json`

`outputs/reports/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json`

Interpretation so far:

- step 2000 does not pass the 256-sample free-run parity gate.
- step 4000 also does not recover.
- final/step 8000 improves slightly over step 2000/4000 but still fails.
- It is much worse than F42's 16-sample result (`0.380 m` ADE), so the 16-sample overfit result does not trivially scale.
- B0 target diversity is `23.55` unique trajectory tokens on average; F45 final is `11.77`, so the branch has substantial trajectory-token diversity loss.
- Automatic step 6000/8000 watcher was stopped after step 2000 and step 4000 both failed; final decode was completed by the main chain.
- step 4000 live eval started and was observed at `6 / 256`.
- step 4000 live eval later observed at `77 / 256`.
- step 6000 checkpoint exists: `outputs/checkpoints/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607/step_006000`
- step 6500 train status: loss `0.9541`, free-run token acc `0.923`, traj state cosine `0.6134`, no errors
- step 4000 live eval later observed at `153 / 256`.
- step 7400 train status: loss `0.9347`, free-run token acc `0.911`, traj state cosine `0.6622`, no errors
- train complete at step 8000: loss `0.9161`, free-run token acc `0.915`, traj state cosine `0.6424`, no errors
- final checkpoint exists: `outputs/checkpoints/flex_f45_nods_free_run_target256_from_f42_s8000_lr2e7_20260607/final`
- final decode started in the main chain
- GPU after model load: about `105.7 GB / 143.8 GB`, util `100%`

Watcher:

- script: `scripts/tmp_wait_eval_flex_f45_nods256_checkpoints.sh`
- stopped after step 2000 and step 4000 both failed; final decode was completed in the main chain
