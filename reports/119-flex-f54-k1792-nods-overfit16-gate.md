# 119 - FLEX F54 K1792 No-DeepStack 16-Sample Gate

Date: 2026-06-07

## Purpose

Test whether K896 is over-compressed by increasing retained visual tokens from `56/image` to `112/image`.

## Run

- script: `scripts/tmp_run_flex_f54_k1792_nods_overfit16_chain.sh`
- run name: `flex_f54_k1792_nods_overfit16_from_f0_s3000_lr5e7_20260607`
- F0 source: B0 step_006250 + untrained per-image FLEX
- tokens per image: `112`
- scene tokens: `1792`
- original visual tokens: `2880`
- actual compressed DeepStack: off
- samples: first 16 held-out val rows
- trainable: FLEX scene encoder + last4 LoRA + multimodal projector
- trainable params: `67,431,424`
- LR: `5e-7`
- loss: B0 free-run token CE + trajectory-state alignment

## Early Results

| Checkpoint | B0 token match | B0 ADE m | B0 FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| early step_000500 | 0.192 | 11.643 | 35.470 | 12.94 | 1.12 |
| early step_001000 | 0.221 | 7.274 | 23.164 | 11.62 | 1.00 |

Reference:

- F42 K896 no-actual-DeepStack 16-sample best: ADE `0.380 m`
- F32 K896 clean F0 16-sample CE-only: ADE `2.346 m`

## Decision

F54 was stopped after step `1000`. Increasing retained tokens from K896 to K1792 did not recover the 16-sample B0 parity gate.

This suggests the current failure is not simply “too few visual tokens.” Fresh compressed-prefix training is unstable under the current target-context/state objective, even when compression is relaxed.

One-line status:

`F54 pass: N. K1792 alone does not make FLEX problem-free.`

Artifacts:

- `outputs/reports/flex_f54_k1792_nods_overfit16_from_f0_s3000_lr5e7_20260607_early_step000500_b0_trajonly_parity_summary.json`
- `outputs/reports/flex_f54_k1792_nods_overfit16_from_f0_s3000_lr5e7_20260607_early_step001000_b0_trajonly_parity_summary.json`
