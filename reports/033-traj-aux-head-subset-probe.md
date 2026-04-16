# 033. Training-Only Traj Auxiliary Head Subset Probe

## status

- completed
- result: **partially promising, but not yet usable**

## goal

Test whether a **training-only parallel trajectory interface head** can reduce the multi-sample plateau collapse that survived:

- GT trajectory CE
- teacher trajectory token / top-k
- teacher trajectory hidden + projector

The intent was to avoid routing all trajectory supervision through the LM next-token head.

## implementation

Added a tiny auxiliary trajectory head on top of student hidden states:

- hidden -> `2` channels
  - channel `0`: accel-like control tokens
  - channel `1`: curvature-like control tokens
- loss uses `traj_token_mask` positions only
- parity selects which channel is supervised at each trajectory token position
- target is the **decoded continuous control value** implied by the GT discrete token

This head is **training-only auxiliary supervision**. The main inference path is still the existing LM generation path.

### files changed

- [student_wrapper.py](/workspace/cosmos_distillation/src/model/student_wrapper.py)
- [checkpoint_io.py](/workspace/cosmos_distillation/src/model/checkpoint_io.py)
- [losses.py](/workspace/cosmos_distillation/src/training/losses.py)
- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- [09_train_distill.py](/workspace/cosmos_distillation/scripts/09_train_distill.py)
- [stage_a0_traj_aux_head.yaml](/workspace/cosmos_distillation/configs/train/stage_a0_traj_aux_head.yaml)

### validation

- smoke summary: [smoke_a0_traj_aux_head.json](/workspace/cosmos_distillation/outputs/reports/smoke_a0_traj_aux_head.json)
- smoke metrics: [metrics.jsonl](/workspace/cosmos_distillation/outputs/checkpoints/smoke_a0_traj_aux_head/metrics.jsonl)
- unit tests: `6 passed`

The smoke run confirmed the new loss is active:

- `traj_aux_loss = 7.62` at step 1

## experiment

We reran the standard subset family:

- `train1`
- `train4`
- `train16`

with:

- `prompt_mode: traj_only`
- `target_mode: traj_only`
- `traj_loss: 1.0`
- `traj_aux_loss: 1.0`
- existing decoded-geometry GT losses still on
- `20` optimizer steps
- mid-run eval/save disabled for apples-to-apples decode comparison

### artifacts

Train summaries:

- [subset_a0_trajaux_train1_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train1_step20.json)
- [subset_a0_trajaux_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train4_step20.json)
- [subset_a0_trajaux_train16_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train16_step20.json)

Inference summaries:

- [subset_a0_trajaux_train1_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train1_step20_infer_train.json)
- [subset_a0_trajaux_train1_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train1_step20_infer_val.json)
- [subset_a0_trajaux_train4_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train4_step20_infer_train.json)
- [subset_a0_trajaux_train4_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train4_step20_infer_val.json)
- [subset_a0_trajaux_train16_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train16_step20_infer_train.json)
- [subset_a0_trajaux_train16_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_train16_step20_infer_val.json)

Comparison JSON:

- [subset_a0_trajaux_1_4_16_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_trajaux_1_4_16_compare.json)

## results

### 1. collapse metrics improved a lot

Compared against the current clean baseline `a0_fix`:

- `train4 val`
  - anti-collapse: `0.0482 -> 0.3686`
  - avg unique ids: `3.0 -> 13.4`
  - avg max same-token run: `119.3 -> 3.6`

- `train16 val`
  - anti-collapse: `0.0985 -> 0.3243`
  - avg unique ids: `2.0 -> 8.8`
  - avg max same-token run: `98.3 -> 4.2`

This is the strongest evidence so far that **the interface matters**. Once supervision is allowed to bypass the LM trajectory-token path, the pure plateau failure mode weakens substantially.

### 2. but alignment quality got worse

The improvement above is not the full story.

For the same `train4 val` and `train16 val` runs:

- `avg_set_jaccard` dropped to approximately `0.0`
- `coarse_motion_agreement_rate` dropped to `0.0`

So the head did **not** produce “better GT-aligned trajectories”.
It produced **less-collapsed but semantically drifting trajectories**.

### 3. this is not just a trivial win over `a0_fix`

Compared against the older `a0_geom` baseline:

- `train4 val`
  - `a0_geom anti-collapse = 0.2900`
  - `a0_trajaux anti-collapse = 0.3686`
  - but `jaccard: 0.0287 -> 0.0`
  - and `motion: 0.1 -> 0.0`

- `train16 val`
  - `a0_geom anti-collapse = 0.1055`
  - `a0_trajaux anti-collapse = 0.3243`
  - but `jaccard: 0.0108 -> 0.0`
  - and `motion: 0.1 -> 0.0`

So the auxiliary head is not simply “strictly better”.
It is better at **breaking collapse**, but worse at **preserving target semantics**.

## interpretation

This run gives two important signals:

1. the old conclusion still holds:
   - pure LM next-token trajectory learning is a major part of the collapse problem

2. but a new conclusion is now stronger:
   - a separate trajectory interface can break plateau collapse
   - however, our first version currently behaves like a **high-entropy but weakly grounded trajectory prior**

In practice, the generated trajectories often moved into very different token bands such as:

- `~140`
- sometimes `~2960-2990`

rather than matching GT token sets or GT motion class.

That means the new head is acting more like a **de-collapse regularizer** than a true planner-alignment head.

## verdict

This is the first experiment that gives a credible reason to keep pursuing a **separate trajectory interface/head**.

But this exact version is **not ready** to unblock `Stage B`, because:

- it breaks collapse
- but it does not yet preserve GT-aligned planning semantics

## recommended next step

Do **not** discard the auxiliary-head direction.

Instead, the next version should explicitly anchor semantics:

- keep the separate traj interface/head
- add short-horizon control / motion anchoring on that head
- likely factorize further by horizon and/or control group
- avoid using anti-collapse alone as the success criterion

### concrete next move

Most likely next experiment:

- `traj auxiliary head v2`
- plus short-horizon semantic/control anchor
- and evaluate:
  - collapse metrics
  - GT overlap
  - coarse motion agreement

## takeaway

The training-only traj auxiliary head did **not** solve the problem end-to-end.

But it did show something important:

**the plateau collapse is at least partly an interface problem, not just a supervision problem.**
