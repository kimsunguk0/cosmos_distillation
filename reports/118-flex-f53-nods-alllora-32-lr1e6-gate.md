# 118 - FLEX F53 No-DeepStack All-LoRA 32 LR 1e-6 Gate

Date: 2026-06-07

## Purpose

Test whether F49's 32-sample failure was mainly under-training from LR `2e-7`.

## Run

- script: `scripts/tmp_run_flex_f53_nods_alllora_target32_lr1e6_chain.sh`
- run name: `flex_f53_nods_alllora_target32_from_f42_s4000_lr1e6_20260607`
- init: F42 best `step_002000`
- samples: `32`
- actual compressed DeepStack: off
- trainable: FLEX scene encoder + all LoRA + multimodal projector
- trainable params: `126,282,752`
- LR: `1e-6`
- loss: B0 free-run token CE + trajectory-state alignment

## Early Result

| Checkpoint | B0 token match | B0 ADE m | B0 FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| early step_000500 | 0.318 | 3.096 | 9.550 | 19.47 | 5.16 |

Reference:

- F49 all-LoRA LR `2e-7` final: ADE `1.754 m`
- F47 last4-LoRA LR `2e-7` final: ADE `2.333 m`
- F42 16-sample best: ADE `0.380 m`

## Decision

F53 was stopped at step `500`. LR `1e-6` makes the 32-sample free-run parity worse, not better.

One-line status:

`F53 pass: N. The 32-sample no-DeepStack break is not fixed by simply increasing LR to 1e-6.`

Artifact:

`outputs/reports/flex_f53_nods_alllora_target32_from_f42_s4000_lr1e6_20260607_early_step000500_b0_trajonly_parity_summary.json`
