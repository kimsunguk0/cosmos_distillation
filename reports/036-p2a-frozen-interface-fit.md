# 036. P2a Frozen Interface Fit

## Goal

Stabilize the paired-interface trajectory head before re-coupling it to the backbone/LM path.

- base checkpoint: `subset_t1_teacherpair_train4_step20/final`
- stage config: `configs/train/stage_t1_paired_interface_v2a_frozen.yaml`
- trainable modules: `traj_aux_head` only
- targets: teacher-paired control only
- GT prefix/final anchors: off
- aux loss: normalized target + `tanh` bounded output + Huber

## Implementation

- Added `TrajectoryAuxInterfaceConfig` with:
  - per-bucket/channel target means/stds
  - bounded `tanh` scale
  - Huber delta
- Added teacher-paired target-stat builder in `scripts/09_train_distill.py`
- Added optimization policy `freeze_all_but_traj_aux_head`
- Reconfigured the aux head after checkpoint load when bucket count mismatched the stage config

## Result

Run:

- summary: `outputs/reports/subset_t1_pairifacev2a_frozen_train4_step8_v2.json`
- metrics: `outputs/checkpoints/subset_t1_pairifacev2a_frozen_train4_step8_v2/metrics.jsonl`

Observed behavior:

- `traj_aux_num_buckets=4` persisted into the saved checkpoint
- train `traj_aux_loss` stayed bounded:
  - first: `2.24`
  - min: `1.90`
  - max: `2.81`
  - last: `2.58`
- val `traj_aux_loss` was also bounded:
  - best: `2.43`
- `grad_norm` no longer exploded into the `1e4~1e5` range:
  - max: `368.36`

This is a clear improvement over report 035, where the naive end-to-end paired-interface head caused aux loss explosion and broke LM formatting.

## Remaining issue

Some later steps hit `grad_norm=0.0`, which suggests the bounded head is now stable but can saturate on certain samples/buckets.

Interpretation:

- the immediate failure mode in 035 was optimization coupling / scale explosion
- P2a removes that failure mode
- the next problem is no longer explosion, but making the frozen interface fit informative enough before reintroducing GT prefix anchors or mixed LM/interface training

## Next step

Move to `P2b`:

- keep the frozen or near-frozen setup
- add only a weak `GT prefix@8` anchor
- keep `GT final` off
- continue monitoring:
  - `traj_aux_loss`
  - `grad_norm`
  - `traj_aux_abs_max`
