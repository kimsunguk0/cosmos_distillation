# FLEX F34 Dummy-Slot Placement Diagnostic

## Purpose

Separate sequence-length compression / downstream position shift from FLEX content and replacement effects.

F34 keeps the original image-token slot count and inserts FLEX scene tokens into dummy visual slots. This has no deployment speed benefit. It is only a diagnostic: if F34 recovered B0 trajectory parity, compressed-position shift would be the main blocker.

## Setup

- Run: `flex_f34_dummy_slots_trajonly_overfit16_s3000_lr2e6_20260607`
- Init: `outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606`
- Teacher/reference: B0 no-FLEX `step_006250`
- Corpus: `data/corpus/flex_heldout256_stage2val_seed42.jsonl`
- Samples: first 16 val samples
- Prompt mode: `joint`
- Target mode: `traj_only`
- Objective: same clean trajectory-only gate as F32/F33
  - B0 trajectory-only free-run token CE: `1.0`
  - free-run end-token CE: `0.05`
  - hidden/KL/pair losses disabled
- Steps: `3000`
- LR: `2e-6`
- Special mode: `--flex-dummy-image-slots`

## Train Result

Final step 3000:

| Run | token CE | token acc | end CE / acc | loss |
| --- | ---: | ---: | ---: | ---: |
| F32 compressed last4 | 0.3279 | 0.9436 | 0.0271 / 1.0000 | 0.3293 |
| F33 compressed last12 | 0.3151 | 0.9495 | 0.1141 / 0.9800 | 0.3208 |
| F34 dummy-slot last4 | 0.3101 | 0.9572 | 0.2231 / 0.9600 | 0.3213 |

F34 learns the target-context token objective slightly better than F32/F33.

## Decode Result

B0 trajectory-only target summary on the same 16 samples:

- teacher/GT decode ADE/FDE: `5.793 / 19.071`
- avg unique trajectory ids: `22.125`
- avg max same-token run: `1.125`

F34 actual trajectory-only decode:

- teacher/GT ADE/FDE: `6.335 / 21.351`
- teacher-target token match: `0.092`
- avg unique trajectory ids: `15.000`
- avg max same-token run: `1.125`

F34 compared against B0 trajectory-only free-run targets:

- exact 128-token rate: `1.000`
- B0 target token match: `0.443`
- B0-target ADE/FDE: `3.093 / 10.633`
- compared samples: `16 / 16`

## Comparison

| Run | Position mode | LoRA | token CE | B0 token match | B0-target ADE/FDE | unique | max run |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 target | no FLEX | - | - | 1.000 | 0.000 / 0.000 | 22.125 | 1.125 |
| F32 | compressed | last4 | 0.3279 | 0.401 | 2.346 / 8.214 | 9.938 | 8.500 |
| F33 | compressed | last12 | 0.3151 | 0.414 | 4.255 / 13.526 | 18.313 | 1.125 |
| F34 | dummy slots | last4 | 0.3101 | 0.443 | 3.093 / 10.633 | 15.000 | 1.125 |

## Judgment

Rejected as a fix.

F34 improves target-context token CE and removes the obvious repetition failure, but it still does not recover B0 trajectory geometry on a 16-sample overfit gate.

## Diagnosis

Compressed-position shift is not the sole cause.

Because F34 keeps the original image-token slot count and still fails B0 trajectory-body parity, the remaining blocker is FLEX content / replacement / objective mismatch. The current method replaces rich per-image visual token structure with a small set of scene tokens, and target-context trajectory CE does not force the autoregressive trajectory-body state to match B0.

## Next

Do not scale current FLEX to held-out/full training.

The next useful gate should change the training signal or placement, not just run longer:

1. train FLEX against B0 autoregressive rollout states, not only target-context CE;
2. add hidden/action-pre parity at trajectory-body rollout positions under the same generated prefix;
3. test a residual/side-channel FLEX mode that keeps original visual embeddings and adds FLEX information before attempting replacement compression again.

