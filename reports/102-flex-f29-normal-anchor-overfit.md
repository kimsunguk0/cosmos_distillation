# FLEX F29 Normal Free-Run Anchor Overfit

## Purpose

Test whether FLEX can memorize B0 no-FLEX normal free-run trajectory tokens on a tiny 16-sample heldout subset.

This is not a generalization test. It is a wiring/capacity/exposure-bias test.

## Setup

- Run: `flex_f29_normal_anchor_from_f28_overfit16_s3000_lr2e6_20260607`
- Init: `flex_f28b_perimage_preservepos_margin_from_f28a_heldout256_s3000_lr1e6_20260607/final`
- Teacher/reference: B0 no-FLEX `step_006250`
- Corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- Samples: first 16 val samples
- Trainable params: 66.5M
  - FLEX scene encoder: 31.4M
  - language LoRA last 4 layers: 10.0M
  - multimodal projector: 25.2M
- Objective:
  - B0 normal free-run token CE: `1.0`
  - free-run end-token CE: `0.05`
  - prefix token CE: `0.05`
  - teacher-forced KL/hidden boundary losses disabled
- Steps: 3000
- LR: `2e-6`

## Train Result

Final step 3000:

- loss: `0.2702`
- free-run token CE: `0.2655`
- free-run token acc: `0.9634`
- prefix CE: `0.0938`
- prefix acc: `0.9847`
- action_pre cosine vs B0: `0.7668`
- cot_end cosine vs B0: `0.8355`

The token CE learns under teacher-forced B0 target context, but it does not reach full memorization and it drifts hidden parity because the hidden losses were intentionally disabled.

## Decode Result

Compressed actual decode, compared against B0 free-run targets:

- exact 128-token rate: `1.0`
- B0 target token match: `0.5522`
- B0 target ADE/FDE: `2.014 / 7.213`
- avg generated unique tokens: `10.0`
- avg max same-token run: `1.19`

Position-preserving diagnostic decode:

- exact 128-token rate: `0.0`
- B0 target token match: `0.0303`
- B0 target ADE/FDE: not measurable because generated token count is not 128
- avg generated unique tokens: `34.1`
- avg max same-token run: `43.25`

## Judgment

Rejected as a FLEX solution.

F29 shows that target-context teacher-forced free-run CE is insufficient. Even on 16 samples, it produces decent conditional token accuracy but poor autoregressive B0 parity during actual decode.

## Next

Run a student-greedy-context overfit. The next test should train on the current student's own generated trajectory prefix while labeling toward B0 targets. If this improves 16-sample B0 parity, the main blocker is exposure bias. If it also fails, the issue is deeper than teacher-forced context mismatch.
