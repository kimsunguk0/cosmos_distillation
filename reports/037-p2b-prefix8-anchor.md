# 037. P2b Prefix-8 Anchor

## Goal

Test whether a weak GT prefix anchor can be added on top of the stable frozen paired-interface fit without reintroducing optimization blow-up.

- base checkpoint: `subset_t1_teacherpair_train4_step20/final`
- stage config: `configs/train/stage_t1_paired_interface_v2b_prefix8.yaml`
- trainable modules: `traj_aux_head` only
- teacher control target: on
- GT prefix anchor: on, first 8 steps only
- GT final anchor: off
- aux target space: normalized + bounded `tanh` + Huber

## Result

Run:

- summary: `outputs/reports/subset_t1_pairifacev2b_prefix8_train4_step8.json`
- metrics: `outputs/checkpoints/subset_t1_pairifacev2b_prefix8_train4_step8/metrics.jsonl`

Observed behavior:

- training remained stable
- `traj_aux_loss` stayed in the same bounded range as P2a
- no `1e4~1e5` grad explosion reappeared
- checkpoint manifest again recorded `traj_aux_num_buckets=4`

Key comparison against P2a:

- P2a best val total: `2.4327`
- P2b best val total: `2.4349`
- P2a val `traj_aux_xyz_loss`: `0.6614`
- P2b val `traj_aux_xyz_loss`: `0.0220`

Interpretation:

- adding a weak `GT prefix@8` anchor does **not** destabilize the frozen interface fit
- the prefix term is being satisfied much better
- but the overall teacher-control fitting objective is essentially unchanged at this scale

## Conclusion

P2b is a successful stability check, not yet a semantic breakthrough.

What it shows:

- the immediate optimization problem from report 035 is still solved
- weak prefix anchors can be introduced safely
- the next meaningful step is no longer "make it stop exploding"
- it is now either:
  - increase the usefulness of the prefix anchor slightly, or
  - move to a mixed curriculum where LM contract batches and interface batches coexist

## Recommended next step

`P2c mixed`:

- `70~80%` pair-LM sanity batches
- `20~30%` frozen or near-frozen interface batches
- keep `GT prefix@8`
- keep `GT final` off
