# 030 - Trainer Traj Bug Fix And A0 Rerun

## Status

Completed.

Trainer-side trajectory objective bugs were patched, then `A0 traj-only` subset probes were rerun on `train1 / train4 / train16`.

The patch fixed real implementation mismatches, but the rerun did **not** improve collapse behavior overall. In the cleaned objective, `A0` remained strongly collapse-prone and in some settings regressed relative to the earlier decoded-geometry baseline.

## Why This Report Exists

We had identified three trainer-level issues that could distort Stage A interpretation:

1. `hard_traj_ce` and `teacher_traj_ce` were computed on `traj_span_mask`, while `hard_traj_acc` only measured `traj_token_mask`.
2. `A0` batches with no teacher view still flowed through fallback teacher averaging, which could dilute hard trajectory supervision.
3. `teacher_traj_ce` was effectively "GT traj under teacher prompt" unless explicitly overridden, which made the name and behavior misleading.

This report records the patch and the first rerun after those fixes.

## Patch Summary

### 1. Body-only trajectory CE

`hard_traj_ce` and `teacher_traj_ce` now use `traj_token_mask` instead of `traj_span_mask`.

Interpretation:

- trajectory body tokens are now optimized by traj CE
- boundary / format tokens are no longer mixed into traj CE
- format supervision remains the responsibility of `format_ce`

Key file:

- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)

### 2. Remove fallback teacher averaging for traj-only A0

When `teacher_view` is absent, the trainer no longer creates a fallback teacher path and averages hard traj with a zero-weight teacher traj branch.

Interpretation:

- `A0 traj-only` now behaves much closer to a true hard-only traj objective
- hard trajectory signal is no longer implicitly halved by the trainer path

Key file:

- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)

### 3. Make teacher-view traj supervision explicit opt-in

Teacher-view trajectory loss is now only active when `teacher_traj_loss` is explicitly configured.

Interpretation:

- default teacher-view traj leakage is removed
- "teacher traj" is no longer silently "GT traj under teacher prompt" by default

Key files:

- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- [collator.py](/workspace/cosmos_distillation/src/training/collator.py)

### 4. Regression coverage

Unit tests were updated to cover:

- no teacher-traj leakage by default
- explicit teacher-traj opt-in
- body-only traj mask behavior

Key file:

- [test_trainer.py](/workspace/cosmos_distillation/tests/unit/test_trainer.py)

## Rerun Setup

We reran the same `A0 traj-only` subset probe family after the trainer fix.

Configuration:

- stage config: [stage_a0_traj_only.yaml](/workspace/cosmos_distillation/configs/train/stage_a0_traj_only.yaml)
- teacher disabled
- `max_steps = 20`
- constrained traj decoding at inference
- subsets: `train1`, `train4`, `train16`

Artifacts:

- compare JSON: [subset_a0_fix_1_4_16_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_1_4_16_compare.json)
- train summaries:
  - [subset_a0_fix_train1_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train1_step20.json)
  - [subset_a0_fix_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train4_step20.json)
  - [subset_a0_fix_train16_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train16_step20.json)
- infer outputs:
  - [subset_a0_fix_train1_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train1_step20_infer_train.json)
  - [subset_a0_fix_train1_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train1_step20_infer_val.json)
  - [subset_a0_fix_train4_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train4_step20_infer_train.json)
  - [subset_a0_fix_train4_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train4_step20_infer_val.json)
  - [subset_a0_fix_train16_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train16_step20_infer_train.json)
  - [subset_a0_fix_train16_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_fix_train16_step20_infer_val.json)

Reference baselines:

- contract probe: [025-subset-collapse-threshold-1-4-16.md](/workspace/cosmos_distillation/reports/025-subset-collapse-threshold-1-4-16.md)
- decoded-geometry A0 probe: [026-a0-geometry-subset-threshold-1-4-16.md](/workspace/cosmos_distillation/reports/026-a0-geometry-subset-threshold-1-4-16.md)

## Results

### train1

Compared to the earlier decoded-geometry A0 baseline:

- train anti-collapse: `0.2837 -> 0.1840`
- train unique ids: `8.0 -> 4.0`
- train max run: `25.0 -> 55.0`
- val anti-collapse: `0.2608 -> 0.2785`
- val unique ids: `7.3 -> 3.7`
- val max run: `33.6 -> 56.5`

Interpretation:

- the post-fix run no longer memorized the single train sample as well as the earlier A0 geometry probe
- val anti-collapse score ticked up only because coarse-motion agreement happened to improve on a few samples
- trajectory-body richness still regressed

### train4

Compared to the earlier decoded-geometry A0 baseline:

- train anti-collapse: `0.2705 -> 0.0440`
- train unique ids: `10.0 -> 2.75`
- train max run: `38.25 -> 119.5`
- val anti-collapse: `0.2900 -> 0.0482`
- val unique ids: `8.7 -> 3.0`
- val max run: `35.2 -> 119.3`

Interpretation:

- this is a strong regression
- the rerun collapsed almost completely into a small plateau token set
- the earlier A0 geometry gain at `train4` was not stable under the cleaned trainer path

Observed pattern:

- train / val generations repeatedly fell into a narrow `1519 / 1527 / 1353` plateau

### train16

Compared to the earlier decoded-geometry A0 baseline:

- train anti-collapse: `0.1189 -> 0.1179`
- train unique ids: `5.44 -> 2.0`
- train max run: `108.25 -> 97.88`
- val anti-collapse: `0.1055 -> 0.0985`
- val unique ids: `4.9 -> 2.0`
- val max run: `109.1 -> 98.3`

Interpretation:

- repetition length improved slightly
- but diversity collapsed further
- net anti-collapse quality was effectively unchanged to slightly worse

Observed pattern:

- train / val generations mostly reduced to a two-band plateau: `1538 -> 1493`

## Main Takeaway

The trainer bugs were real and needed to be fixed.

But after fixing them, `A0 traj-only` did **not** suddenly become healthy. Instead:

- `train4` became much worse
- `train16` stayed effectively collapsed
- `train1` lost some of its earlier memorization behavior

This suggests an important interpretation:

The earlier decoded-geometry `A0` improvements were at least partly entangled with unintended trainer behavior. Once the objective path was cleaned up, the underlying collapse problem became more exposed rather than less.

## Conclusion

The project blocker is still unchanged:

**Stage A cannot yet learn GT trajectory body robustly in a multi-sample setting without collapse.**

The trainer patch removes interpretation noise, but it does not solve the core optimization attractor.

## Follow-up

Recommended next steps:

1. Keep `Stage B` paused.
2. Treat the cleaned `A0` path as the new baseline for future experiments.
3. Do not keep turning the same decoded-geometry knob harder.
4. Move next to:
   - parity / horizon-conditioned weighting
   - direct anti-plateau penalty
   - true prefix-horizon curriculum
   - if needed later, representation-level trajectory factorization
