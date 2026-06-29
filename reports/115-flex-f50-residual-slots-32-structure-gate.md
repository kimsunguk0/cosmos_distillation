# 115 - FLEX F50 Residual-Slot 32 Structure Gate

Date: 2026-06-07

## Purpose

If F49 fails, separate compressed-replacement failure from broader FLEX/objective failure.

F50 keeps original Qwen visual embeddings and original visual-token length, then adds FLEX scene tokens as a residual to the first K visual slots per image block.

This is not the deployable compressed path. It is a diagnostic:

- If F50 passes while F49 fails: the blocker is compressed replacement / position / information loss.
- If F50 also fails: the blocker is not just compression; the current FLEX objective/placement cannot scale to 32-sample B0 parity.

## Run

- conditional wait tmux: `flex_f50_wait_if_f49fail`
- wait script: `scripts/tmp_wait_and_run_flex_f50_if_f49_fails.sh`
- run script: `scripts/tmp_run_flex_f50_residual_alllora_target32_from_f42_chain.sh`
- wait log: `outputs/logs/flex_f50_wait_f49_20260607.log`
- run log: `outputs/logs/flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607_chain.log`
- B0 target summary: `outputs/reports/b0_step006250_flexheldout256_decode_trajonly_summary.json`
- student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- output checkpoint: `outputs/checkpoints/flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607`

## Settings

- samples: `32`
- steps: `8000`
- LR: `2e-7`
- actual compressed path: off
- residual diagnostic: `--flex-residual-image-slots --flex-residual-scale 1.0`
- trainable: FLEX scene encoder + all language LoRA + multimodal projector
- loss: B0 free-run token CE + trajectory state alignment
- context source: B0 target trajectory tokens

## Conditional Start

F50 starts only if F49 final B0-target ADE is not below `0.8 m`.

If F49 ADE `< 0.8 m`, F50 is skipped because all-LoRA compressed mode already passes the 32-sample capacity gate.

## Current Status

`queued: waiting for F49 final summary`

Preflight:

- `scripts/tmp_run_flex_f50_residual_alllora_target32_from_f42_chain.sh`: `bash -n` pass
- `scripts/tmp_wait_and_run_flex_f50_if_f49_fails.sh`: `bash -n` pass
- conditional wait tmux started: `flex_f50_wait_if_f49fail`
- Wait script hardened: if F49 wait/run session exits without producing the F49 parity summary, F50 wait logs `f50_wait_error_f49_ended_without_summary` and exits instead of silently waiting forever.
- Conditional wait tmux was restarted with the hardened script at `2026-06-07 19:18`.

## Live Start

F49 failed the pass threshold:

- F49 final token match: `0.513`
- F49 final ADE/FDE: `1.754 / 6.318`
- F49 decision: partial improvement over F47, but capacity gate fail for `<0.8 m`.

F50 therefore started automatically.

Live F50 status:

- teacher cache: `32 / 32` done
- collated cache: `32 / 32` done
- train start: reached
- trainable params: `126,282,752`
  - FLEX scene encoder: `31,378,432`
  - language LoRA: `69,730,304`
  - multimodal projector: `25,174,016`
- step `1`: loss `1.2344`, token acc `1.0`, end-token acc `0.0`, traj-state cosine `0.7241`, grad norm `12.53`

Early training trend:

| Step | Loss | Token acc | End-token acc | Traj-state cosine | Grad norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.2344 | 1.0000 | 0.0000 | 0.7241 | 12.5 |
| 200 | 2.0915 | 0.9691 | 0.0000 | 0.7000 | 7.2 |
| 300 | 2.0982 | 0.9653 | 0.0000 | 0.6872 | 7.1 |
| 400 | 2.0680 | 0.9632 | 0.0000 | 0.6941 | 7.1 |
| 500 | 2.0443 | 0.9559 | 0.0000 | 0.6865 | 7.3 |
| 1000 | 1.5296 | 0.8903 | 0.0500 | 0.7114 | 29.6 |
| 1200 | 1.3753 | 0.8777 | 0.3800 | 0.6908 | 36.7 |
| 1400 | 1.2903 | 0.8795 | 0.5900 | 0.6773 | 40.9 |
| 1800 | 1.2350 | 0.8843 | 0.6700 | 0.6718 | 52.2 |
| 2100 | 1.2007 | 0.8839 | 0.7800 | 0.6592 | 51.9 |
| 3000 | 1.1401 | 0.8888 | 0.8000 | 0.6640 | 61.2 |
| 3400 | 1.1133 | 0.8929 | 0.7700 | 0.6686 | 65.7 |
| 4000 | 1.1028 | 0.8913 | 0.8300 | 0.6631 | 63.8 |

Early read: residual-slot mode is not immediately fixing the 32-sample target-context gate. End-token accuracy recovers after step `1000`, but full trajectory token accuracy stays around `0.89` and traj-state cosine stays around `0.66`. Continue to final decode before hard fail judgment.

## F50 Result

Auto-appended: 2026-06-07 22:00:49

| Checkpoint | Token match | ADE m | FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| final / step 8000 | 0.248 | 3.334 | 11.210 | 16.50 | 5.09 |

Decision: residual-slot FAIL: preserving original visual positions is not enough.

Artifact:

`outputs/reports/flex_f50_residualslots_alllora_target32_from_f42_s8000_lr2e7_20260607_final_b0_trajonly_parity_summary.json`

