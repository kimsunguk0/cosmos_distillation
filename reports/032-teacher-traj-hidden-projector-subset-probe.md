# 032 Teacher Traj Hidden Projector Subset Probe

## Status

Completed.

We added a trainable `4096 -> 2048` compatibility bridge in the student path so `teacher traj hidden` could actually be used, then reran the `teacher-traj-only` subset probe with hidden alignment enabled.

## Why This Report Exists

[031-teacher-traj-token-topk-subset-probe.md](/workspace/cosmos_distillation/reports/031-teacher-traj-token-topk-subset-probe.md) showed that `teacher traj token + top-k` alone did **not** fix Stage A collapse.

At that point, the missing teacher signal was:

- `teacher traj hidden`

But the imported cache and the student backbone were mismatched:

- teacher traj hidden: `4096`
- student hidden: `2048`

So hidden alignment was inactive until a projector existed.

## Implementation

### Student-side projector

The student wrapper now supports an optional trainable trajectory-hidden projector.

Key changes:

- [student_wrapper.py](/workspace/cosmos_distillation/src/model/student_wrapper.py)
- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- [losses.py](/workspace/cosmos_distillation/src/training/losses.py)
- [checkpoint_io.py](/workspace/cosmos_distillation/src/model/checkpoint_io.py)
- [09_train_distill.py](/workspace/cosmos_distillation/scripts/09_train_distill.py)

Behavior:

- when `teacher_traj_hidden_align_loss > 0`
- and a teacher traj cache directory is configured
- training now infers the teacher hidden width from cache
- builds a trainable student projector
- routes `traj_hidden_states` through it
- and saves / reloads projector weights in LoRA checkpoints

Checkpoint manifests now include:

- `traj_hidden_projector`
- `traj_teacher_hidden_size`

### Config

New config:

- [stage_t0_teacher_traj_hidden_projector.yaml](/workspace/cosmos_distillation/configs/train/stage_t0_teacher_traj_hidden_projector.yaml)

Loss setup:

- `teacher_traj_loss = 0.15`
- `teacher_traj_topk_kd_loss = 0.5`
- `teacher_traj_hidden_align_loss = 0.02`

The hidden-align weight was intentionally kept small after smoke validation showed raw hidden MSE is large.

## Validation

### Unit tests

Passed:

- `17 passed`

Relevant tests:

- [test_student_wrapper.py](/workspace/cosmos_distillation/tests/unit/test_student_wrapper.py)
- [test_losses.py](/workspace/cosmos_distillation/tests/unit/test_losses.py)
- [test_trainer.py](/workspace/cosmos_distillation/tests/unit/test_trainer.py)

### Smoke

Smoke artifact:

- [smoke_t0_teacher_traj_hidden_projector.json](/workspace/cosmos_distillation/outputs/reports/smoke_t0_teacher_traj_hidden_projector.json)

Smoke metrics:

- [metrics.jsonl](/workspace/cosmos_distillation/outputs/checkpoints/smoke_t0_teacher_traj_hidden_projector/metrics.jsonl)

Important result:

- `teacher_traj_hidden_align_loss` is now **nonzero**
- smoke value was roughly `355.38` before weighting

So hidden alignment is no longer dead code.

## Subset Experiment

Artifacts:

- compare JSON: [subset_t0_teacher_hiddenproj_1_4_16_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_hiddenproj_1_4_16_compare.json)
- run summaries:
  - [subset_t0_teacher_traj_hiddenproj_train1_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train1_step20.json)
  - [subset_t0_teacher_traj_hiddenproj_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train4_step20.json)
  - [subset_t0_teacher_traj_hiddenproj_train16_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train16_step20.json)
- infer outputs:
  - [subset_t0_teacher_traj_hiddenproj_train1_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train1_step20_infer_train.json)
  - [subset_t0_teacher_traj_hiddenproj_train1_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train1_step20_infer_val.json)
  - [subset_t0_teacher_traj_hiddenproj_train4_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train4_step20_infer_train.json)
  - [subset_t0_teacher_traj_hiddenproj_train4_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train4_step20_infer_val.json)
  - [subset_t0_teacher_traj_hiddenproj_train16_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train16_step20_infer_train.json)
  - [subset_t0_teacher_traj_hiddenproj_train16_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t0_teacher_traj_hiddenproj_train16_step20_infer_val.json)

## Results

### train1

Teacher-only vs hidden-projector:

- train anti-collapse: `0.1006 -> 0.2578`
- train unique ids: `4.0 -> 3.0`
- train max run: `102.0 -> 117.0`
- val anti-collapse: `0.1945 -> 0.1600`
- val unique ids: `4.0 -> 3.4`
- val max run: `102.7 -> 116.8`

Interpretation:

- train score improved only because coarse-motion agreement jumped to `1.0`
- actual trajectory richness did not improve
- val regressed

### train4

Teacher-only vs hidden-projector:

- train anti-collapse: `0.0502 -> 0.0442`
- train unique ids: `2.5 -> 3.0`
- train max run: `114.75 -> 119.75`
- val anti-collapse: `0.0559 -> 0.0448`
- val unique ids: `2.9 -> 3.0`
- val max run: `114.9 -> 119.6`

Interpretation:

- no meaningful rescue
- tiny diversity gain was offset by worse run-length and worse overlap

### train16

Teacher-only vs hidden-projector:

- train anti-collapse: `0.0890 -> 0.0094`
- train unique ids: `3.44 -> 1.0`
- train max run: `98.94 -> 128.0`
- val anti-collapse: `0.0879 -> 0.0094`
- val unique ids: `3.3 -> 1.0`
- val max run: `99.0 -> 128.0`

Interpretation:

- this is a hard regression
- hidden-projector training collapsed to an almost pure single-token plateau
- generated outputs were dominated by repeated `<i1183>`

## Main Takeaway

Adding a projector made `teacher traj hidden` technically usable, but it did **not** solve the optimization problem.

Instead:

- `train1` got a misleading memorization-style bump
- `train4` stayed collapsed
- `train16` got much worse

So the project is now beyond the question of “can we wire hidden align in?”

The answer is:

- yes, we can
- but in the current loss stack, it does not stabilize Stage A

## Conclusion

`teacher traj hidden + token/top-k` is **not** a sufficient fix in the current setup.

The strongest evidence is `train16`, where hidden-projector supervision collapsed harder than both:

- cleaned `A0`
- teacher token/top-k only

`Stage B` should remain paused.

## Follow-up

Recommended next steps:

1. Do not keep pushing the same hidden-align design harder.
2. Treat this result as evidence that direct token-space LM-head supervision is still the dominant attractor.
3. If teacher hidden is revisited, prefer:
   - short-horizon interface alignment only
   - normalized / cosine hidden alignment instead of raw MSE
   - or a factorized / parallel trajectory head rather than pure LM-head continuation
4. Keep the main blocker framed as `Stage A trajectory-body stabilization`, not reasoning KD.
