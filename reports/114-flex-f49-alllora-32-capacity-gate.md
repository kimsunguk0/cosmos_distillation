# 114 - FLEX F49 All-LoRA 32 Capacity Gate

Date: 2026-06-07

## Purpose

Separate adaptation capacity from objective/placement failure after F47/F48.

Known results:

- F42 no-actual-DeepStack 16-sample B0 parity succeeds best: `0.380 m` ADE.
- F47 no-actual-DeepStack 32-sample target-context gate fails: token match `0.483`, ADE/FDE `2.333 / 7.951`.
- F48 32-sample student-greedy rollout-context gate is unstable: greedy target match degrades `0.415 -> 0.322 -> 0.247 -> 0.204`.

F49 keeps the F47 target-context/state-heavy objective and F42 init, but opens all language LoRA layers instead of only the last 4.

## Run

- tmux session: `flex_f49_alllora32`
- script: `scripts/tmp_run_flex_f49_nods_alllora_target32_from_f42_chain.sh`
- log: `outputs/logs/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_chain.log`
- B0 target summary: `outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json`
- student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- output checkpoint: `outputs/checkpoints/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607`

## Settings

- samples: `32`
- steps: `8000`
- LR: `2e-7`
- actual compressed DeepStack: off
- trainable: FLEX scene encoder + all language LoRA + multimodal projector
- loss: B0 free-run token CE + trajectory state alignment
- context source: B0 target trajectory tokens

## Decision Logic

- If F49 strongly improves over F47 and approaches F42 16-sample quality: the 32-sample break is mainly adaptation capacity.
- If F49 still fails: the current compressed replacement/objective is the blocker, not LoRA depth. The next branch should be residual/side-channel or true rollout-state distillation, not more target-context CE.

Current status:

`queued: waiting for GPU availability; stage2 AE training currently occupies GPU`

Queue status:

- wait tmux session: `flex_f49_wait_alllora32`
- wait script: `scripts/tmp_wait_and_run_flex_f49_when_gpu_free.sh`
- wait log: `outputs/logs/flex_f49_wait_gpu_20260607.log`
- start condition: stage2 AE training process absent and GPU memory used `< 60000 MB`
- polling interval: `60s`
- first poll after wait-script hardening: `used_mb=99095`, `stage2_tmux_alive=1`, `stage2_train_alive=1`
- stage2 progress at queue setup: `70800 / 75000`
- stage2 recent train speed: about `10.18 sec/step`
- rough remaining train-only time at setup: about `11.9 h`, plus remaining eval/checkpoint overhead
- follow-up poll: stage2 advanced `70800 -> 71000` over the check window, so it is actively training; F49 remains queued.
- latest wait poll observed: `used_mb=99095`, `stage2_tmux_alive=1`, `stage2_train_alive=1`
- live recheck after wait restart: stage2 advanced `71000 -> 71100`; latest loss `0.1013`; F49 still queued.
- later live recheck: stage2 advanced `71200 -> 71300`; latest loss `0.4921`; F49 still queued.
- latest rough remaining time at `71300 / 75000`: train-only `~2.9-3.2 h` from the recent 500-1000 step window, but eval/checkpoint overhead can push wall time higher.
- 5-minute monitor confirmed active training: stage2 advanced `71300 -> 71500`; latest loss `0.3971`; GPU utilization sampled up to `100%`.
- latest rough remaining time at `71500 / 75000`: train-only `~2.7-2.8 h` from the recent 500-1000 step window, with eval/checkpoint overhead still expected before completion.
- latest live recheck: stage2 advanced to `71600 / 75000`; latest loss `0.1250`; F49/F50 summaries still absent.
- latest rough remaining time at `71600 / 75000`: train-only `~2.65-2.69 h` from the recent 500-1000 step window; `~6.0 h` from the 2000-step window that includes eval/checkpoint overhead.
- latest live recheck: stage2 advanced to `71700 / 75000`; latest loss `0.1367`; F49/F50 summaries still absent.
- latest rough remaining time at `71700 / 75000`: train-only `~2.56-2.57 h` from the recent 500-1000 step window; `~5.75 h` from the 2000-step window that includes eval/checkpoint overhead.
- latest live recheck: stage2 advanced to `71800 / 75000`; latest loss `0.0798`; F49/F50 summaries still absent.
- latest rough remaining time at `71800 / 75000`: train-only `~2.49-2.50 h` from the recent 500-1000 step window; `~5.5 h` from the 2000-step window that includes eval/checkpoint overhead.
- latest live recheck: stage2 advanced to `71900 / 75000`; latest loss `0.0797`; F49/F50 summaries still absent.
- next stage2 eval/checkpoint boundary: `72500` (`600` train steps remaining).
- latest rough remaining time at `71900 / 75000`: train-only `~2.42-2.43 h` from the recent 500-1000 step window; `~5.27 h` from the 2000-step window that includes eval/checkpoint overhead.
- latest live recheck: stage2 advanced to `72000 / 75000`; latest loss `0.3413`; F49/F50 summaries still absent.
- next stage2 eval/checkpoint boundary: `72500` (`500` train steps remaining).
- latest rough remaining time at `72000 / 75000`: train-only `~2.34-2.35 h` from the recent 500-1000 step window; `~5.04 h` from the 2000-step window that includes eval/checkpoint overhead.
- follow-up liveness check at `72000`: no newer train-step log yet, but PID `2909` remains `R (running)` and CPU time/context switches continue increasing; stage2 is alive, F49 remains queued.
- latest live recheck: stage2 advanced to `72100 / 75000`; latest loss `0.2935`; F49/F50 summaries still absent.
- next stage2 eval/checkpoint boundary: `72500` (`400` train steps remaining).
- latest rough remaining time at `72100 / 75000`: train-only `~2.27-2.28 h` from the recent 500-1000 step window.
- follow-up recheck: still `72100 / 75000`; GPU memory `99095 MB`; stage2 PID still active; F49 remains queued.
- latest live recheck: stage2 advanced to `72200 / 75000`; latest loss `0.2778`; F49/F50 summaries still absent.
- next stage2 eval/checkpoint boundary: `72500` (`300` train steps remaining).
- latest rough remaining time at `72200 / 75000`: train-only `~2.20-2.22 h` from the recent 500-1000 step window.
- latest live recheck: stage2 advanced to `72300 / 75000`; latest loss `0.2369`; F49/F50 summaries still absent.
- next stage2 eval/checkpoint boundary: `72500` (`200` train steps remaining).
- latest rough remaining time at `72300 / 75000`: train-only `~2.12-2.13 h` from the recent 500-1000 step window.
- latest live recheck: stage2 advanced to `72400 / 75000`; latest loss `0.3438`; F49/F50 summaries still absent.
- next stage2 eval/checkpoint boundary: `72500` (`100` train steps remaining).
- latest rough remaining time at `72400 / 75000`: train-only `~2.05 h`, with imminent `72500` eval/checkpoint overhead expected.

Preflight:

- `scripts/tmp_run_flex_f49_nods_alllora_target32_from_f42_chain.sh`: `bash -n` pass
- `scripts/tmp_wait_and_run_flex_f49_when_gpu_free.sh`: `bash -n` pass
- F42 checkpoint LoRA adapter contains all `28` language layer indices (`0..27`), so `--unfreeze-all-lora` has real adapter weights to train.
- Static code check:
  - `configure_trainable_parameters()` first freezes all student params.
  - `--train-flex` re-enables `flex_scene_encoder`.
  - `--unfreeze-all-lora` re-enables every parameter whose name contains `lora_`.
  - `--unfreeze-multimodal-projector` re-enables multimodal projector / visual merger parameters.
  - AdamW is built from `[parameter for parameter in student.parameters() if parameter.requires_grad]`.
  - Therefore F49 should train FLEX scene encoder + all LoRA + multimodal projector as intended.
- Wait script was hardened to check the actual stage2 `84_train_student_ae28_official.py` process instead of relying only on the stage2 tmux session. This avoids indefinite waiting if the tmux session remains after training exits.
- Wait script was also quieted for transient `/proc/*/cmdline` races and the `flex_f49_wait_alllora32` session was restarted with the fixed script.
- Wait session was restarted with `CHECK_EVERY_SEC=60` so F49 should start within about one minute after stage2 exits and GPU memory drops below the threshold.
- F49 run script now auto-appends the final parity table and pass/fail decision to this report after decode comparison completes.
- Pipeline watchdog started:
  - tmux session: `flex_status_watch`
  - script: `scripts/tmp_watch_flex_pipeline_status.sh`
  - log: `outputs/logs/flex_pipeline_watch_20260607.log`
  - interval: `600s`
  - purpose: preserve stage2/F49/F50 state snapshots across SSH/session interruptions.

Memory/load-order preflight:

- `scripts/105_train_flex_teacher_parity.py` now builds `--cache-teacher-targets` before loading the FLEX student, then unloads teacher and calls `torch.cuda.empty_cache()`.
- Purpose: avoid the previous unnecessary peak where teacher and student were resident before teacher cache construction.
- Preflight script: `scripts/tmp_run_flex_f49_mem_preflight_1step.sh`
- Preflight log: `outputs/logs/flex_f49_mem_preflight_1sample_1step_20260607.log`
- Result: PASS. A 1-sample/1-step F49-equivalent path ran during stage2 without OOM.
- Preflight trainable params: `126,282,752`
  - FLEX scene encoder: `31,378,432`
  - language LoRA: `69,730,304`
  - multimodal projector: `25,174,016`
- Preflight step metrics: loss `0.2508`, token acc `1.0`, grad norm `5.28`, traj_state cosine `0.7979`.

Concurrent-start update:

- `scripts/tmp_wait_and_run_flex_f49_when_gpu_free.sh` now supports `ALLOW_CONCURRENT_AFTER_VAL_STEP`.
- Current wait restart: `MAX_USED_MB=115000`, `ALLOW_CONCURRENT_AFTER_VAL_STEP=72500`, `CHECK_EVERY_SEC=60`.
- This keeps F49 blocked until the stage2 `72500` validation eval is written, then starts F49 if GPU memory is below the threshold.

Live launch override:

- Because stage2 `72500` eval remained long-running and the 1-step F49 preflight already passed, the F49 wait session was replaced with direct F49 execution at `2026-06-07 20:01`.
- F49 reached `flex_parity_train_start` and completed step `1` without OOM while stage2 was still resident.
- Live VRAM after F49 step `1`: about `111.7 GB / 143.8 GB`.
- Step `1` metrics: loss `0.2508`, free-run token acc `1.0`, traj_state cosine `0.7979`, grad norm `5.28`.
- Effective trainable params confirmed: `126,282,752` across FLEX scene encoder, all language LoRA, and multimodal projector.
- Step `100` metrics: loss `1.2388`, free-run token acc `0.9417`, traj_state cosine `0.6207`, grad norm `96.1`.
- Interpretation: step `1` was one cached row and is not comparable to step `100`; step `100` is the first 32-row cyclic training average. Direction should be judged from later 100-step logs.
- Step trend while running concurrently with stage2 eval:

| Step | Loss | Token acc | Traj-state cosine | Grad norm |
|---:|---:|---:|---:|---:|
| 100 | 1.2388 | 0.9417 | 0.6207 | 96.1 |
| 200 | 1.0584 | 0.9420 | 0.6238 | 111.4 |
| 300 | 0.9917 | 0.9421 | 0.6171 | 68.0 |
| 500 | 0.9351 | 0.9424 | 0.6303 | 43.6 |
| 1000 | 0.8741 | 0.9476 | 0.6474 | 19.4 |
| 1400 | 0.8755 | 0.9473 | 0.6498 | 16.4 |
| 1800 | 0.8504 | 0.9485 | 0.6536 | 14.3 |
| 2100 | 0.8565 | 0.9489 | 0.6506 | 14.2 |
| 2600 | 0.8366 | 0.9514 | 0.6548 | 13.7 |
| 3000 | 0.8459 | 0.9510 | 0.6555 | 15.0 |
| 3300 | 0.8308 | 0.9545 | 0.6548 | 13.4 |
| 3700 | 0.8354 | 0.9538 | 0.6544 | 15.4 |
| 4100 | 0.8224 | 0.9551 | 0.6574 | 13.7 |

Early read: F49 is learning enough to reduce loss and token acc, but traj-state cosine only improves to about `0.65-0.66` by step `4100`. This is not a fast 32-sample overfit. Need final checkpoint decode before capacity-pass/fail judgment.

## F49 Result

Auto-appended: 2026-06-07 20:50:06

| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| final / step 8000 | 0.513 | 1.754 | 6.318 | 17.47 | 5.06 |

Reference:

- F42 16-sample target: ADE `0.380 m`
- F47 32-sample last4-LoRA target: ADE `2.333 m`

Decision: partial improvement: all-layer LoRA helps, but still not near F42 16-sample quality.

Artifact:

`outputs/reports/flex_f49_nods_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json`

