# 023 Stage A3 Traj Audit and Continuation

- status: in_progress
- scope:
  - audit whether the held-out trajectory collapse is caused by corpus quantization or by the student
  - run one additional `Stage A` continuation from `stage_a2_v3_2/final` with teacher CoT/KD effectively disabled and trajectory weight increased
  - compare held-out constrained decoding before and after the continuation

## Bottom line

- the trajectory collapse is **not** primarily a data-distribution problem
- the corpus itself has rich trajectory-token diversity on both `train` and `val`
- the `Stage A3` continuation improved aggregate validation `traj_token_acc`
- but the actual held-out constrained decode still collapsed, and in this run it became **narrower** than `Stage A2`
- because of that, `Stage B` should still remain blocked

## Data audit result

Artifact:

- `outputs/reports/traj_token_audit_v3_2.json`

Key findings:

- `train`
  - `244` records
  - `1449` unique trajectory tokens overall
  - average per-sequence unique token count: `79.76`
  - average max same-token run: `1.28`

- `val`
  - `40` records
  - `669` unique trajectory tokens overall
  - average per-sequence unique token count: `80.88`
  - average max same-token run: `1.40`

- train/val divergence is not extreme
  - Jensen-Shannon divergence: `0.0948 bits`

Implication:

- the GT trajectories are not naturally collapsing to `1-3` ids
- the observed held-out collapse is a model/generalization issue, not a simple corpus pathology

## Stage A3 continuation setup

New config:

- `configs/train/stage_a3_traj_focus.yaml`

Settings:

- start checkpoint:
  - `outputs/checkpoints/stage_a2_v3_2/final`
- learning rate:
  - `3e-5`
- weights:
  - `gt_cot_loss: 0.75`
  - `teacher_cot_loss: 0.0`
  - `teacher_topk_kd_loss: 0.0`
  - `traj_loss: 1.5`
  - `output_format_loss: 0.75`

Training artifact:

- `outputs/reports/stage_a3_v3_2.json`
- `outputs/checkpoints/stage_a3_v3_2/final`

## What improved

Compared to `Stage A2` validation summary:

- `traj_token_acc`
  - `0.0299 -> 0.0797`

- `traj_loss`
  - `5.7590 -> 4.9527`

So the continuation did improve aggregate token-level validation metrics on the supervised objective.

## What did not improve

Held-out constrained decode artifact:

- `outputs/reports/stage_a3_v3_2_infer_val10_constrained.json`

Direct comparison:

- `outputs/reports/stage_a2_vs_a3_traj_decode_compare.json`

For the same `10` held-out validation samples:

- average generated unique traj ids per sample
  - `Stage A2: 3.0`
  - `Stage A3: 2.4`

- average set Jaccard against target traj ids
  - `Stage A2: 0.0370`
  - `Stage A3: 0.0226`

- global generated unique ids across all 10 samples
  - `Stage A2: 4`
  - `Stage A3: 3`

- average longest repeated-token run
  - `Stage A2: 105.1`
  - `Stage A3: 78.8`

Interpretation:

- `Stage A3` reduced the worst repetition run length somewhat
- but it also collapsed more aggressively into the narrow `1495/1496` region
- so the decode became slightly smoother but less sample-specific

## Important implementation note

One subtle but important detail is that `traj_loss` is not purely a hard-view loss.

- `src/training/trainer.py`
  - `traj_ce` is computed as the mean of:
    - `hard_traj_ce`
    - `teacher_traj_ce` when a teacher view exists

- `src/training/collator.py`
  - teacher-enabled samples always populate teacher-view trajectory supervision
  - teacher trajectory weights currently reuse the same `traj_ce` weight path

This means:

- even when `teacher_cot_loss=0.0` and `teacher_topk_kd_loss=0.0`
- the trajectory objective still partially includes teacher-view conditioning on teacher-allowed records

So `Stage A3` was not a completely pure “hard-only” trajectory continuation.

## Current assessment

- the corpus is healthy enough to support rich trajectory generation
- the student can now reliably satisfy the boundary contract
- the remaining blocker is sample-specific trajectory-body generation under held-out decode
- the latest continuation improved token metrics but did not solve decode collapse

## Recommended next step

Before `Stage B`, add one more control to the training path:

- separate hard-view and teacher-view trajectory weighting

Then run a true hard-view-only trajectory continuation from:

- `outputs/checkpoints/stage_a2_v3_2/final`
  or
- `outputs/checkpoints/stage_a3_v3_2/final`

and select checkpoints by held-out constrained-decode quality, not by `traj_token_acc` alone.
