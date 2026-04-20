# 048. H0a Frozen Latent Recovery on 958-Sample Cache

## What Changed

- Switched teacher traj hidden supervision from a **trainable teacher bridge** to a **frozen latent target** loaded from `/data/teacher_cache/traj15/latent`.
- Built a new corpus on the expanded dataset:
  - `train=754`
  - `val=204`
- Added a reusable hidden probe:
  - [`scripts/21_probe_hidden_latent.py`](/workspace/cosmos_distillation/scripts/21_probe_hidden_latent.py)

## H0a Setup

- Stage config:
  - [`stage_h0a_frozen_teacher_latent_prefix16.yaml`](/workspace/cosmos_distillation/configs/train/stage_h0a_frozen_teacher_latent_prefix16.yaml)
- Key properties:
  - `teacher_traj_hidden_source=latent`
  - `teacher_traj_latent_suffix=lat32`
  - `prefix16` mask
  - `teacher_traj_hidden_align_loss=1.0`
  - `relation / variance / covariance` on
  - `LM traj CE/top-k = 0`
  - `boundary/format` only for decode contract preservation

## Runs

### 1. Baseline

- Checkpoint:
  - `outputs/checkpoints/stage_a_v3_2_format_lr/final`
- Probe:
  - [`stage_a_v3_2_format_hiddenprobe_val40_lat32.json`](/workspace/cosmos_distillation/outputs/reports/stage_a_v3_2_format_hiddenprobe_val40_lat32.json)

### 2. H0a train64 / step64

- Summary:
  - [`subset_h0a_frozenlatent_prefix16_train64_step64.json`](/workspace/cosmos_distillation/outputs/reports/subset_h0a_frozenlatent_prefix16_train64_step64.json)
- Probe:
  - [`subset_h0a_frozenlatent_prefix16_train64_step64_hiddenprobe_val40.json`](/workspace/cosmos_distillation/outputs/reports/subset_h0a_frozenlatent_prefix16_train64_step64_hiddenprobe_val40.json)
- LM-only decode sanity:
  - [`subset_h0a_frozenlatent_prefix16_train64_step64_infer_val10_lm.json`](/workspace/cosmos_distillation/outputs/reports/subset_h0a_frozenlatent_prefix16_train64_step64_infer_val10_lm.json)

### 3. H0a full754 / step128

- Summary:
  - [`full_h0a_frozenlatent_prefix16_step128.json`](/workspace/cosmos_distillation/outputs/reports/full_h0a_frozenlatent_prefix16_step128.json)
- Probe:
  - [`full_h0a_frozenlatent_prefix16_step128_hiddenprobe_val40.json`](/workspace/cosmos_distillation/outputs/reports/full_h0a_frozenlatent_prefix16_step128_hiddenprobe_val40.json)
- LM-only decode sanity:
  - [`full_h0a_frozenlatent_prefix16_step128_infer_val5_lm.json`](/workspace/cosmos_distillation/outputs/reports/full_h0a_frozenlatent_prefix16_step128_infer_val5_lm.json)

## Main Comparison

- Aggregated compare:
  - [`h0a_hiddenprobe_val40_compare.json`](/workspace/cosmos_distillation/outputs/reports/h0a_hiddenprobe_val40_compare.json)

| Run | Student rank | Student offdiag cosine | Token cosine |
| --- | ---: | ---: | ---: |
| Baseline | 2.18 | 0.984 | 0.024 |
| H0a train64/64 | 5.19 | 0.793 | 0.082 |
| H0a full754/128 | 4.63 | 0.730 | 0.137 |

Teacher reference on the same latent target:

- teacher effective rank mean: `4.54`
- teacher offdiag cosine mean: `0.272`

## Interpretation

- H0a is a **real hidden-manifold improvement**, not noise.
- The biggest gain is vs baseline:
  - effective rank roughly doubled+ (`2.18 -> 4.63~5.19`)
  - offdiag cosine dropped strongly (`0.984 -> 0.73~0.79`)
  - token cosine improved (`0.024 -> 0.082~0.137`)
- `train64` gave the best rank.
- `full754/128` gave the best offdiag and token cosine.
- So the expanded dataset does help hidden recovery, but the objective still trades off rank vs direct alignment.

## What Did Not Change

- LM-only decode sanity stayed alive:
  - `CoT -> <|traj_future_start|> -> 128 body -> <|traj_future_end|>` still emits.
- But LM-only traj body is **still `<i1499>` plateau**.
- So this stage improved the upstream hidden manifold without yet fixing the downstream LM readout contract.

## Decision

- H0a should be treated as **partial success**.
- It is enough evidence to keep the hidden-first ordering:
  - `hidden manifold recovery -> LM contract coupling`
- The next logical step is **H0b**:
  - add temporal delta / second-delta
  - keep control anchor very weak
  - keep LM traj CE/top-k off
