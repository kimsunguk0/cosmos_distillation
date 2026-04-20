# 031 Teacher Traj Token Top-K Subset Probe

## Status

Completed.

We wired the imported `traj15` teacher cache into training and ran the first `teacher-traj-only` subset probe on `train1 / train4 / train16`.

## Why This Report Exists

After the cleaned `A0` rerun in [030-trainer-traj-bug-fix-and-a0-rerun.md](/workspace/cosmos_distillation/reports/030-trainer-traj-bug-fix-and-a0-rerun.md), the core blocker remained unchanged:

- `GT traj` supervision still collapsed in multi-sample training
- `Stage B` remained blocked

So the next question was:

**Can teacher trajectory token distribution help where GT-centric traj CE keeps collapsing?**

## What Was Added

The imported cache at `/data/teacher_cache/traj15` was wired into the training path.

Added support for:

- teacher trajectory token sequence supervision
- teacher trajectory top-k KD
- teacher trajectory hidden loading

Key files:

- [collator.py](/workspace/cosmos_distillation/src/training/collator.py)
- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- [losses.py](/workspace/cosmos_distillation/src/training/losses.py)
- [09_train_distill.py](/workspace/cosmos_distillation/scripts/09_train_distill.py)
- [stage_t0_teacher_traj_only.yaml](/workspace/cosmos_distillation/configs/train/stage_t0_teacher_traj_only.yaml)

## Important Constraint

Teacher trajectory **hidden align** did not activate in this experiment.

Reason:

- teacher traj hidden cache shape: `128 x 4096`
- student hidden size: `2048`

Without a projector, hidden states cannot be aligned directly.

So this probe used:

- `teacher_traj_topk_kd_loss = 0.5`
- `teacher_traj_loss = 0.15`
- `teacher_traj_hidden_align_loss = 0.0`
- `GT traj loss = 0.0`

This was intentionally a clean `teacher-traj-only` token/top-k probe.

## Experiment Setup

Config:

- [stage_t0_teacher_traj_only.yaml](/workspace/cosmos_distillation/configs/train/stage_t0_teacher_traj_only.yaml)

Properties:

- `prompt_mode = traj_only`
- `target_mode = traj_only`
- teacher CoT off
- GT CoT off
- GT traj off
- output format loss still on
- trajectory reweighting off

Subset runs:

- `train1`, `train4`, `train16`
- `20` steps each
- constrained trajectory decoding at inference

Artifacts:

- compare JSON: [subset_t0_teacher_1_4_16_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_1_4_16_compare.json)
- run summaries:
  - [subset_t0_teacher_traj_train1_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train1_step20.json)
  - [subset_t0_teacher_traj_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train4_step20.json)
  - [subset_t0_teacher_traj_train16_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train16_step20.json)
- infer outputs:
  - [subset_t0_teacher_traj_train1_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train1_step20_infer_train.json)
  - [subset_t0_teacher_traj_train1_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train1_step20_infer_val.json)
  - [subset_t0_teacher_traj_train4_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train4_step20_infer_train.json)
  - [subset_t0_teacher_traj_train4_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train4_step20_infer_val.json)
  - [subset_t0_teacher_traj_train16_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train16_step20_infer_train.json)
  - [subset_t0_teacher_traj_train16_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_train16_step20_infer_val.json)

Reference baselines:

- [025-subset-collapse-threshold-1-4-16.md](/workspace/cosmos_distillation/reports/025-subset-collapse-threshold-1-4-16.md)
- [026-a0-geometry-subset-threshold-1-4-16.md](/workspace/cosmos_distillation/reports/026-a0-geometry-subset-threshold-1-4-16.md)
- [030-trainer-traj-bug-fix-and-a0-rerun.md](/workspace/cosmos_distillation/reports/030-trainer-traj-bug-fix-and-a0-rerun.md)

## Results

### train1

Teacher-traj-only vs cleaned `A0`:

- train anti-collapse: `0.1840 -> 0.1006`
- train unique ids: `4.0 -> 4.0`
- train max run: `55.0 -> 102.0`
- val anti-collapse: `0.2785 -> 0.1945`
- val unique ids: `3.7 -> 4.0`
- val max run: `56.5 -> 102.7`

Interpretation:

- teacher supervision did **not** rescue memorization
- it drove the model into a longer repeated-token plateau

### train4

Teacher-traj-only vs cleaned `A0`:

- train anti-collapse: `0.0440 -> 0.0502`
- train unique ids: `2.75 -> 2.5`
- train max run: `119.5 -> 114.75`
- val anti-collapse: `0.0482 -> 0.0559`
- val unique ids: `3.0 -> 2.9`
- val max run: `119.3 -> 114.9`

Interpretation:

- this is only a **tiny** improvement over the cleaned `A0`
- it remains far worse than the earlier decoded-geometry `A0`
- the model is still heavily collapsed

### train16

Teacher-traj-only vs cleaned `A0`:

- train anti-collapse: `0.1179 -> 0.0890`
- train unique ids: `2.0 -> 3.44`
- train max run: `97.88 -> 98.94`
- val anti-collapse: `0.0985 -> 0.0879`
- val unique ids: `2.0 -> 3.3`
- val max run: `98.3 -> 99.0`

Interpretation:

- diversity widened slightly
- but overlap and coarse motion collapsed to near zero
- net score got worse

## Qualitative Pattern

This probe did **not** remove collapse.

Instead, it changed the attractor.

Observed behavior:

- previous GT-centric collapse tended to fall into central-band plateaus
- teacher-traj-only fell into a different, teacher-shaped plateau
- examples looked like:
  - `1631 -> 1498 -> 1495 -> 1490`
  - `1296 -> 1183`
  - `739 / 778 / 1094 / 1183`

So the student did not become collapse-free.
It mostly learned a different collapsed template.

## Main Takeaway

`teacher traj token + top-k` by itself is **not enough** to solve the Stage A blocker.

More precisely:

- it can shift the collapse pattern
- it can slightly widen plateau diversity in some settings
- but it does not restore robust trajectory body learning
- and it can destroy GT overlap / coarse motion even when unique token count rises

This means the current problem is deeper than:

- missing teacher distribution information

The optimization attractor is still there even with teacher token/top-k supervision.

## Conclusion

The project blocker is still unchanged:

**Stage A does not yet learn a stable multi-sample trajectory body, even with teacher trajectory token/top-k supervision.**

`Stage B` should remain paused.

## Follow-up

Recommended next steps:

1. Do not interpret teacher token/top-k as a sufficient fix.
2. Keep `teacher traj hidden` on hold until a `4096 -> 2048` projector exists.
3. If teacher trajectory signal is revisited, prefer:
   - projected hidden alignment
   - short-horizon interface alignment
   - or factorized trajectory representation
4. Continue treating `Stage A` stabilization as the main blocker, not reasoning KD.
