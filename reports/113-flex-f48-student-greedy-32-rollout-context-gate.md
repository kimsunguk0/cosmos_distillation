# 113 - FLEX F48 Student-Greedy 32 Rollout-Context Gate

Date: 2026-06-07

## Purpose

Test whether the F45/F46/F47 failure is caused by target-context teacher forcing rather than FLEX capacity alone.

F45/F46/F47 train with B0 target trajectory tokens inserted into the trajectory context. Their train CE/token accuracy can look good while final free-run decode still fails. F48 keeps the same 32-sample overfit setting as F47, but trains the trajectory CE under the current student greedy rollout prefix.

## Run

- tmux session: `flex_f48_sg32`
- script: `scripts/tmp_run_flex_f48_nods_free_run_32_student_greedy_from_f42_chain.sh`
- log: `outputs/logs/flex_f48_nods_studentgreedy_target32_from_f42_s8000_lr2e7_20260607_chain.log`
- B0 target summary: `outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json`
- student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- output checkpoint: `outputs/checkpoints/flex_f48_nods_studentgreedy_target32_from_f42_s8000_lr2e7_20260607`

## Settings

- samples: `32`
- steps: `8000`
- LR: `2e-7`
- actual compressed DeepStack: off
- trainable: FLEX scene encoder + last 4 LoRA layers + multimodal projector
- loss: B0 free-run token CE + trajectory state alignment
- context source: `student_greedy`
- greedy context refresh: every `250` steps
- invalid greedy context fallback: `raw`

## Decision Logic

- If F48 improves strongly over F47: main failure is exposure/rollout-context mismatch.
- If F48 also fails: the current FLEX placement/objective cannot scale even in direct rollout-context overfit; next branch should change placement or add a true rollout-state/distillation objective.

Current status:

`stopped early: rollout-context training diverged from target`

Live checks:

- tmux: `flex_f48_sg32`
- start event logged
- collated cache complete: `32 / 32`
- teacher cache complete: `32 / 32`
- train started: actual compressed DeepStack `false`, trainable params `66,513,920`
- GPU: F48 process about `99 GB`, stage2/eval process about `9 GB`; no VRAM collision
- first student-greedy context refresh:
  - exact 128 token rate `1.0`
  - mean invalid count `0.0`
  - target match `0.415`
  - fallback target rate `0.0`
- step 1: loss `0.2508`, free-run token acc `1.000`, traj state cosine `0.7979`
- step 100: loss `10.6555`, free-run token acc `0.429`, traj state cosine `0.6328`, grad norm `650.8`
- step 200: loss `9.5589`, free-run token acc `0.428`, traj state cosine `0.6562`, grad norm `617.7`
- second student-greedy refresh target match: `0.322`
- step 500: loss `8.7843`, free-run token acc `0.308`, traj state cosine `0.6637`, grad norm `707.9`
- third student-greedy refresh target match: `0.247`
- step 700: loss `8.3255`, free-run token acc `0.251`, traj state cosine `0.6808`, grad norm `768.4`
- fourth student-greedy refresh target match: `0.204`
- step 900: loss `8.7637`, free-run token acc `0.211`, traj state cosine `0.7047`, grad norm `792.7`
- stopped via `tmux send-keys C-c`; no FLEX process left running

F47 comparison baseline:

- F47 final target-context 32-sample B0 parity: token match `0.483`, ADE/FDE `2.333 / 7.951`, unique `18.53`, max same-token run `5.09`

## Result

F48 fails as a rollout-context fix.

The generated context stays syntactically valid, but its target match degrades monotonically:

| Refresh | Greedy target match |
|---:|---:|
| 1 | 0.415 |
| 2 | 0.322 |
| 3 | 0.247 |
| 4 | 0.204 |

Interpretation:

- F47 proves target-context training does not scale from 16 to 32 samples.
- F48 proves simply replacing target context with student-greedy rollout context is unstable under the same recipe.
- The next branch should test capacity/placement, not continue this F48 run: either increase trainable adaptation capacity on the 32-sample overfit, or use a position/residual-preserving FLEX placement diagnostic.
