# 112 - FLEX F47 No-DeepStack 32 Free-Run Target Gate

Date: 2026-06-07

## Purpose

Find the scale break point after:

- F42 16-sample success: `0.380 m` ADE
- F46 64-sample failure: `2.614 m` ADE

This run keeps the same F42 init and F45/F46 recipe but uses the first 32 samples.

## Run

- tmux session: `flex_f47_nods32`
- script: `scripts/tmp_run_flex_f47_nods_free_run_32_from_f42_chain.sh`
- log: `outputs/logs/flex_f47_nods_free_run_target32_from_f42_s8000_lr2e7_20260607_chain.log`
- B0 target summary: `outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json`
- student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- output checkpoint: `outputs/checkpoints/flex_f47_nods_free_run_target32_from_f42_s8000_lr2e7_20260607`

## Settings

- samples: `32`
- steps: `8000`
- LR: `2e-7`
- actual compressed DeepStack: off
- trainable: FLEX scene encoder + last 4 LoRA layers + multimodal projector
- loss: B0 free-run token CE + trajectory state alignment

## Decision Logic

- If 32 passes near F42 quality: failure threshold is between 32 and 64.
- If 32 fails: F42's 16-sample result is a tiny-overfit exception, and this objective is not scalable.

Current status:

`completed: 32-sample target-context gate failed`

Live checks:

- tmux: `flex_f47_nods32`
- start event logged
- collated cache complete: `32 / 32`
- teacher cache complete: `32 / 32`
- train started: actual compressed DeepStack `false`, trainable params `66,513,920`
- trainable groups: FLEX scene encoder `31,378,432`, language LoRA `9,961,472`, multimodal projector `25,174,016`
- step 1: loss `0.2508`, free-run token acc `1.000`, traj state cosine `0.7979`
- step 500: loss `0.9353`, free-run token acc `0.942`, traj state cosine `0.6302`
- step 1000: loss `0.8749`, free-run token acc `0.948`, traj state cosine `0.6469`
- step 1900: loss `0.8663`, free-run token acc `0.949`, traj state cosine `0.6446`
- step 2000 checkpoint exists: `outputs/checkpoints/flex_f47_nods_free_run_target32_from_f42_s8000_lr2e7_20260607/step_002000`
- step 2500: loss `0.8406`, free-run token acc `0.952`, traj state cosine `0.6547`
- step 3400: loss `0.8272`, free-run token acc `0.954`, traj state cosine `0.6576`
- step 4300: loss `0.8358`, free-run token acc `0.955`, traj state cosine `0.6520`
- step 5200: loss `0.8152`, free-run token acc `0.960`, traj state cosine `0.6591`
- step 6400: loss `0.8315`, free-run token acc `0.958`, traj state cosine `0.6559`
- final / step 8000: loss `0.8216`, free-run token acc `0.961`, traj state cosine `0.6560`

## Result

Final B0-target parity on 32 samples:

| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| final / step 8000 | 0.483 | 2.333 | 7.951 | 18.53 | 5.09 |

Artifact:

`outputs/reports/flex_f47_nods_free_run_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json`

Interim interpretation:

- Target-context CE improves slowly, but trajectory-state alignment remains near `0.65` cosine.
- The final free-run decode is much worse than F42's 16-sample result (`0.380 m` ADE).
- F47 is better than F46-64 (`2.614 m`) but still clearly fails. The break point is between 16 and 32 samples.
- The next diagnostic is F48: same 32 samples, but train under current student greedy rollout context instead of B0 target context.
