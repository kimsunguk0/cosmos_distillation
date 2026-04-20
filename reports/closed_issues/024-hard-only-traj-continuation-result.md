# 024 Hard-Only Traj Continuation Result

- status: in_progress
- scope:
  - add separate hard-view vs teacher-view trajectory weighting
  - run a true hard-only trajectory continuation from `stage_a2_v3_2/final`
  - compare the result against `Stage A2` and `Stage A3`

## What changed

The training path now supports an explicit teacher-side trajectory weight.

Key changes:

- `src/training/losses.py`
  - added optional `teacher_traj_ce` / `teacher_traj_loss`
  - legacy configs still keep the previous mixed-trajectory behavior

- `src/training/collator.py`
  - teacher-view trajectory weights now read `teacher_traj_loss` when present
  - otherwise they fall back to the legacy `traj_loss`

- `src/training/trainer.py`
  - legacy path:
    - keep `traj_loss * mean(hard_traj, teacher_traj)`
  - new explicit path:
    - `traj_loss * hard_traj + teacher_traj_loss * teacher_traj`

## Validation

Unit tests:

- `18 passed`
  - includes the new trainer-path test for hard-only trajectory weighting

Artifact:

- `tests/unit/test_trainer.py`

## Hard-only continuation run

Config:

- `configs/train/stage_a4_hard_traj_only.yaml`

Run artifacts:

- summary:
  - `outputs/reports/stage_a4_v3_2.json`
- checkpoint:
  - `outputs/checkpoints/stage_a4_v3_2/final`
- constrained held-out decode:
  - `outputs/reports/stage_a4_v3_2_infer_val10_constrained.json`

Start checkpoint:

- `outputs/checkpoints/stage_a2_v3_2/final`

## Main result

Validation metrics improved again:

- `Stage A2` val `traj_token_acc`
  - `0.0299`
- `Stage A3` val `traj_token_acc`
  - `0.0797`
- `Stage A4` val `traj_token_acc`
  - `0.08125`

But held-out constrained decoding still did not recover rich sample-specific trajectory bodies.

Comparison artifact:

- `outputs/reports/stage_a2_a3_a4_traj_decode_compare.json`

### Stage A2

- average generated unique traj ids per sample:
  - `3.0`
- average longest repeated-token run:
  - `105.1`
- average set Jaccard vs target ids:
  - `0.0370`

### Stage A3

- average generated unique traj ids per sample:
  - `2.4`
- average longest repeated-token run:
  - `78.8`
- average set Jaccard vs target ids:
  - `0.0226`

### Stage A4

- average generated unique traj ids per sample:
  - `2.4`
- average longest repeated-token run:
  - `68.4`
- average set Jaccard vs target ids:
  - `0.0227`

## Interpretation

- separating teacher-side trajectory weighting did work mechanically
- the model now repeats slightly less aggressively than `Stage A3`
- but it still collapses into the narrow `1495/1496` band
- compared with `Stage A2`, the held-out decode is still less diverse and less target-aligned

So the new split was necessary for control, but it was **not sufficient** to solve trajectory-body generalization.

## Current decision

- `Stage B` remains blocked

Reason:

- output contract is alive
- validation token metrics are improving
- but held-out trajectory-body generation is still too collapsed to justify KD-heavy training

## Recommended next step

Stop selecting continuation checkpoints by token-level loss alone.

Instead:

- add a held-out constrained-decode selection metric
- score checkpoints by:
  - per-sample unique traj ids
  - longest repeated-token run
  - target-id set overlap

If needed, save and compare intermediate checkpoints within the same continuation rather than trusting the final epoch endpoint.
