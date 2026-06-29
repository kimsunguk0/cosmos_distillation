# 117 - FLEX F52 DeepStack Feature MSE Projector Gate

Date: 2026-06-07

## Purpose

Test whether actual compressed DeepStack can be recovered by directly matching no-FLEX teacher DeepStack visual features at the retained FLEX slots.

## Code Change

- `scripts/105_train_flex_teacher_parity.py`
  - Added `--deepstack-feature-tokens-per-image`.
  - Added `--deepstack-feature-cos-weight`, `--deepstack-feature-norm-weight`, `--deepstack-feature-mse-weight`.
  - Teacher cache now stores selected no-FLEX DeepStack visual targets.
  - Student loss can compare compressed DeepStack projector outputs against those targets.

Validation:

- `py_compile` pass.
- 1-step MSE-only projector preflight pass.
- Projector-only preflight trainable params: `790,528`.
- Projector-only preflight grad norm: `0.667`.

## Target Construction

Teacher no-FLEX visual features:

- image embeds: `16 x [180, 2048]`
- DeepStack hooks: `3 x [2880, 2048]`
- active hook indexes: `[5, 11, 17]`

FLEX keeps `56` tokens per image, so F52 selected the first `56` teacher DeepStack features from each original `180`-token image block:

`16 images x 56 = 896` DeepStack target tokens per hook.

## Run

- script: `scripts/tmp_run_flex_f52_deepstack_feature_mse_projector_chain.sh`
- run name: `flex_f52_deepstack_feature_mse_projector_from_f42_overfit16_s3000_dsp1e4_20260607`
- init: F42 best `step_002000`
- trainable: FLEX DeepStack projector only
- projector params: `790,528`
- projector LR: `1e-4`
- loss: DeepStack feature MSE + B0 free-run token CE + trajectory-state alignment

## Early Result

| Checkpoint | B0 token match | B0 ADE m | B0 FDE m | Unique tokens | Max same-token run |
|---|---:|---:|---:|---:|---:|
| early step_000500 | 0.584 | 0.962 | 3.421 | 20.06 | 1.12 |

Reference:

- F42 no-actual-DeepStack best: `0.380 m`
- F51 actual-DeepStack joint best: `0.398 m`
- F44b actual-DeepStack projector-only best: `0.534 m`

## Decision

F52 was stopped at step `500`. The DeepStack feature MSE target did not reduce feature MSE and made B0 parity worse than all previous actual-DeepStack gates.

One-line status:

`F52 actual DeepStack feature parity pass: N. Selected-slot teacher DeepStack MSE is not sufficient; actual DeepStack branch remains blocked.`

Artifact:

`outputs/reports/flex_f52_deepstack_feature_mse_projector_from_f42_overfit16_s3000_dsp1e4_20260607_early_step000500_b0_trajonly_parity_summary.json`
