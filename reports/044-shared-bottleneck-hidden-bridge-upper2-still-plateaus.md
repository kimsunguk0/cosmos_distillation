## Summary

We added a shared-bottleneck trajectory hidden bridge to address the `teacher 4096 -> student 2048` manifold mismatch without reusing the old raw `2048 -> 4096 + MSE` path.

- Student hidden: project into a `512`-d bridge space
- Teacher hidden: project into the same `512`-d bridge space
- Loss: cosine-heavy token alignment on body-only traj positions
- Curriculum: first `16` traj body tokens only
- Trainable backbone scope: upper language blocks `26-27` + final norm

This fixed the earlier bridge checkpoint/inference compatibility issues and kept training stable, but **did not change pure LM trajectory body generation**.

## What Changed

- New shared bridge modules:
  - [student_wrapper.py](/workspace/cosmos_distillation/src/model/student_wrapper.py)
  - [checkpoint_io.py](/workspace/cosmos_distillation/src/model/checkpoint_io.py)
- New cosine-heavy bridge loss:
  - [losses.py](/workspace/cosmos_distillation/src/training/losses.py)
  - [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- New optimization/config plumbing:
  - [09_train_distill.py](/workspace/cosmos_distillation/scripts/09_train_distill.py)
  - [stage_t1_hiddenbridge_upper2_prefix16_from_format.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_hiddenbridge_upper2_prefix16_from_format.yaml)
- Inference bridge-load compatibility:
  - [15_infer_student_smoke.py](/workspace/cosmos_distillation/scripts/15_infer_student_smoke.py)

## Stability

Smoke training succeeded and the bridge modules were checkpointed:

- [smoke_t1_hiddenbridge_upper2_prefix16_from_format.json](/workspace/cosmos_distillation/outputs/reports/smoke_t1_hiddenbridge_upper2_prefix16_from_format.json)

Main probe:

- [subset_t1_hiddenbridge_upper2_prefix16_from_format_train4_step12.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenbridge_upper2_prefix16_from_format_train4_step12.json)

Observed:

- training remained bounded
- no aux-style gradient explosion
- `teacher_traj_hidden_align_loss` could run through the bridge path

## Trainable Set Audit

Audit artifact:

- [subset_t1_hiddenbridge_upper2_prefix16_trainable_audit.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenbridge_upper2_prefix16_trainable_audit.json)

The actual trainable set was:

- upper language blocks `26-27`
- final language norm
- `traj_hidden_bridge_student`
- `traj_hidden_bridge_teacher`

Notably, this run did **not** open full tied embeddings or the full LM head.

## LM-only Decode Result

Artifacts:

- train: [subset_t1_hiddenbridge_upper2_prefix16_from_format_train4_step12_infer_train_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenbridge_upper2_prefix16_from_format_train4_step12_infer_train_lm.json)
- val: [subset_t1_hiddenbridge_upper2_prefix16_from_format_train4_step12_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenbridge_upper2_prefix16_from_format_train4_step12_infer_val10_lm.json)
- comparison: [subset_t1_hiddenbridge_upper2_prefix16_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenbridge_upper2_prefix16_compare.json)

Metrics were unchanged versus the previous `upper2` probe:

- train `avg_unique_traj_ids = 1.0`
- train `avg_max_same_token_run = 128.0`
- val `avg_unique_traj_ids = 1.0`
- val `avg_max_same_token_run = 128.0`

Every sample still emitted an `<i1499>` plateau for the full 128-token body.

## Interpretation

This result is important because it separates two hypotheses:

1. `4096 vs 2048` mismatch exists and needed a cleaner bridge.
2. Fixing that mismatch in a normalized shared space, plus opening the top `2` blocks, might be enough to repair pure LM body generation.

The probe supports `1`, but rejects `2`.

The hidden bridge made the hidden-alignment path cleaner and trainable, but **pure LM readout still stayed locked in the same plateau attractor**.

## Current Takeaway

The bottleneck is still not just "teacher hidden dim mismatch".

At this point:

- hidden manifold mismatch exists
- LM readout/interface mismatch also exists
- repairing the manifold alone, at least with a `512`-d shared bridge plus upper-2 unfreeze, is insufficient

## Next Best Step

The next likely experiment should target the readout path more directly, for example:

- keep the hidden bridge, but open a broader upper stack (`22-27`) and/or tied token readout rows
- or move to a mixed objective that explicitly absorbs the hybrid decode contract back into the LM body path
