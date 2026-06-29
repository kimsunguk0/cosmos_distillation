# FLEX F32 Clean F0 Trajectory-Only Capacity Test

## Purpose

Test the cleanest remaining overfit gate:

- start from clean untrained per-image FLEX F0,
- avoid joint CoT rollout as much as possible,
- train only trajectory-body parity on 16 samples,
- check whether actual autoregressive decode can reproduce B0 no-FLEX trajectory-only outputs.

This is a capacity / placement / adaptation-surface test, not a generalization test.

## Setup

- Run: `flex_f32_clean_f0_trajonly_anchor_overfit16_s3000_lr2e6_20260607`
- Init: `outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606`
- Teacher/reference: B0 no-FLEX `step_006250`
- Corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- Samples: first 16 val samples
- Prompt mode: `joint`
- Target mode: `traj_only`
- Objective:
  - B0 trajectory-only free-run token CE: `1.0`
  - free-run end-token CE: enabled
  - hidden/KL/pair losses disabled
- Steps: `3000`
- LR: `2e-6`

## Train Result

Final step 3000:

- loss: `0.3293`
- free-run token CE: `0.3279`
- free-run token acc: `0.9436`
- free-run end-token CE/acc: `0.0271 / 1.0000`
- grad norm: `18.08`

The target-context training objective learns, but it does not reach perfect 16-sample memorization.

## Decode Result

B0 trajectory-only target summary, decoded on the same 16 samples:

- teacher/GT decode ADE/FDE: `5.793 / 19.071`
- avg unique trajectory ids: `22.125`
- avg max same-token run: `1.125`

F32 actual trajectory-only decode:

- teacher/GT decode ADE/FDE: `6.382 / 21.949`
- avg unique trajectory ids: `9.938`
- avg max same-token run: `8.500`

F32 compared against B0 trajectory-only free-run targets:

- exact 128-token rate: `1.000`
- B0 target token match: `0.401`
- B0-target ADE/FDE: `2.346 / 8.214`
- compared samples: `16 / 16`

## Judgment

Rejected as a scale-up candidate.

F32 fails the minimum FLEX overfit gate. Even with clean F0, trajectory-only target-context CE, and only 16 samples, actual autoregressive decode does not preserve B0 trajectory-body distribution.

## Diagnosis

The blocker is not only:

- camera order,
- compressed-position shift,
- joint CoT/prefix generation,
- malformed student-greedy context extraction,
- or lack of a B0 token target.

The current K896 per-image FLEX plus limited projector/LoRA adaptation surface is insufficient to preserve B0 trajectory-body conditional distribution under compressed visual prefix.

## Next

Do not scale current FLEX training to held-out/full data yet.

Next controlled gate should increase the adaptation surface before any generalization run:

- all-layer LoRA or at least last-12-layer LoRA,
- projector open,
- FLEX trainable,
- same 16-sample trajectory-only B0 parity gate first,
- promote only if B0-target ADE drops near B0's own trajectory-only diagnostic gap and repetition stays near B0.

