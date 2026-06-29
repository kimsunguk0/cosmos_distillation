# FLEX F33 Last-12 LoRA Adaptation-Surface Test

## Purpose

Follow F32 by increasing the adaptation surface while keeping the same clean 16-sample trajectory-only overfit gate.

Question:

Does opening more language LoRA capacity fix the failed B0 trajectory-body preservation seen in F32?

## Setup

- Run: `flex_f33_clean_f0_trajonly_last12_overfit16_s3000_lr2e6_20260607`
- Init: `outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606`
- Teacher/reference: B0 no-FLEX `step_006250`
- Corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- Samples: first 16 val samples
- Prompt mode: `joint`
- Target mode: `traj_only`
- Objective: same as F32
  - B0 trajectory-only free-run token CE: `1.0`
  - free-run end-token CE: `0.05`
  - hidden/KL/pair losses disabled
- Steps: `3000`
- LR: `2e-6`

Trainable params:

| Run | FLEX | LoRA | projector | total trainable |
| --- | ---: | ---: | ---: | ---: |
| F32 last4 | 31.4M | 10.0M | 25.2M | 66.5M |
| F33 last12 | 31.4M | 29.9M | 25.2M | 86.4M |

## Train Result

Final step 3000:

| Run | token CE | token acc | end CE / acc | loss |
| --- | ---: | ---: | ---: | ---: |
| F32 last4 | 0.3279 | 0.9436 | 0.0271 / 1.0000 | 0.3293 |
| F33 last12 | 0.3151 | 0.9495 | 0.1141 / 0.9800 | 0.3208 |

F33 slightly improves target-context token CE/accuracy over F32, but the gain is small relative to the added LoRA surface.

## Decode Result

B0 trajectory-only target summary on the same 16 samples:

- teacher/GT decode ADE/FDE: `5.793 / 19.071`
- avg unique trajectory ids: `22.125`
- avg max same-token run: `1.125`

F32 actual trajectory-only decode:

- teacher/GT decode ADE/FDE: `6.382 / 21.949`
- avg unique trajectory ids: `9.938`
- avg max same-token run: `8.500`
- B0-target ADE/FDE: `2.346 / 8.214`
- B0 target token match: `0.401`

F33 actual trajectory-only decode:

- teacher/GT decode ADE/FDE: `6.466 / 20.313`
- avg unique trajectory ids: `18.313`
- avg max same-token run: `1.125`
- B0-target ADE/FDE: `4.255 / 13.526`
- B0 target token match: `0.414`

## Judgment

Rejected as a fix.

Opening last-12 LoRA improves repetition/diversity but does not recover B0 trajectory geometry. B0-target token match barely changes, while B0-target ADE/FDE gets worse than F32.

## Diagnosis

The failure is not simply "last4 LoRA too small."

F33 shows that a larger adaptation surface can reduce repetitive decoding, but the generated trajectory lands in a different geometric mode than B0. That points to a content/placement/objective mismatch: the compressed visual prefix plus current target-context CE does not preserve the B0 autoregressive trajectory-body distribution.

## Next

Do not scale this to held-out/full FLEX training.

The next useful test is not another longer last12 run. Use one of these stricter gates:

1. all-layer LoRA only if the question is pure capacity, but require both B0-target ADE and repetition to improve;
2. change placement/objective: distill trajectory-body hidden/KV or action_pre state during autoregressive rollout, not only target-context token CE;
3. test no-compression/dummy visual-slot diagnostic with the same F32/F33 objective to separate FLEX content loss from replacement/placement effects.

