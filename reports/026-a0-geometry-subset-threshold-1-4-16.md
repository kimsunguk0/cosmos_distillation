# 026 A0 Geometry Subset Threshold 1-4-16

- status: completed
- scope:
  - re-run the `1 / 4 / 16` subset collapse probe after adding decoded-geometry trajectory regularization
  - check whether `A0 traj-only` delays or weakens the collapse onset seen in [025](./025-subset-collapse-threshold-1-4-16.md)

## Experiment setup

New configuration:

- `configs/train/stage_a0_traj_only.yaml`
- prompt mode: `traj_only`
- target mode: `traj_only`
- teacher CoT / teacher top-k / action aux: off
- new traj losses:
  - `traj_ce_freq`
  - `traj_xyz_loss`
  - `traj_delta_loss`
  - `traj_final_loss`

Runtime settings:

- `max_steps=20`
- `batch_size=1`
- `multi_gpu=off`
- constrained trajectory decoding enabled at inference

Artifacts:

- comparison json:
  - `outputs/reports/subset_a0_geom_1_4_16_compare.json`

- train `1`
  - `outputs/reports/subset_a0_geom_train1_step20.json`
  - `outputs/reports/subset_a0_geom_train1_step20_infer_train.json`
  - `outputs/reports/subset_a0_geom_train1_step20_infer_val.json`

- train `4`
  - `outputs/reports/subset_a0_geom_train4_step20.json`
  - `outputs/reports/subset_a0_geom_train4_step20_infer_train.json`
  - `outputs/reports/subset_a0_geom_train4_step20_infer_val.json`

- train `16`
  - `outputs/reports/subset_a0_geom_train16_step20.json`
  - `outputs/reports/subset_a0_geom_train16_step20_infer_train.json`
  - `outputs/reports/subset_a0_geom_train16_step20_infer_val.json`

## Important caveat

This is not an equal-training-budget apples-to-apples rerun of [025](./025-subset-collapse-threshold-1-4-16.md).

- baseline `025`:
  - `stage_a_contract`
  - `max_steps=100`

- new probe `026`:
  - `stage_a0_traj_only`
  - `max_steps=20`

So the question here is not “does this beat the old setup on every metric already?”

The real question is:

- does decoded-geometry regularization weaken the repeated-token collapse attractor early enough to justify continuing this curriculum?

## Main result

Yes, but only partially.

- `train4` improved a lot:
  - the old probe already collapsed into a tiny repeated-token band
  - the new probe still collapses, but into a much wider plateau pattern instead of near-pure repetition

- `train16` also improved, but not enough:
  - the old probe was basically single-token collapse
  - the new probe is no longer pure single-token collapse
  - however, very long repeated plateaus still dominate

- `train1` got worse on exact-set fit:
  - this is expected under the shorter `20-step` `traj-only` warmup
  - the model is no longer just memorizing one full target trajectory the way the old `100-step` run did

## Quantitative summary

### Seen-train decode

- `train1`
  - old:
    - unique traj ids avg: `60.0`
    - max same-token run avg: `1.0`
    - set Jaccard: `1.0000`
  - new:
    - unique traj ids avg: `8.0`
    - max same-token run avg: `25.0`
    - set Jaccard: `0.0303`
    - anti-collapse score: `0.2837`

- `train4`
  - old:
    - unique traj ids avg: `2.25`
    - max same-token run avg: `119.25`
    - set Jaccard: `0.0262`
  - new:
    - unique traj ids avg: `10.0`
    - max same-token run avg: `38.25`
    - set Jaccard: `0.0058`
    - anti-collapse score: `0.2705`
  - interpretation:
    - collapse onset is clearly delayed
    - the model now emits a broader trajectory body instead of an almost constant token stream

- `train16`
  - old:
    - unique traj ids avg: `1.0`
    - max same-token run avg: `128.0`
    - set Jaccard: `0.0130`
  - new:
    - unique traj ids avg: `5.4375`
    - max same-token run avg: `108.25`
    - set Jaccard: `0.0174`
    - coarse motion agreement: `0.125`
    - anti-collapse score: `0.1189`
  - interpretation:
    - the new loss breaks the pure single-token mode
    - but long repeated plateaus still dominate

### Held-out val decode

- `train1`
  - old:
    - unique traj ids avg: `60.0`
    - max same-token run avg: `1.0`
    - set Jaccard: `0.0999`
  - new:
    - unique traj ids avg: `7.3`
    - max same-token run avg: `33.6`
    - set Jaccard: `0.0319`
    - anti-collapse score: `0.2608`

- `train4`
  - old:
    - unique traj ids avg: `2.3`
    - max same-token run avg: `119.5`
    - set Jaccard: `0.0212`
  - new:
    - unique traj ids avg: `8.7`
    - max same-token run avg: `35.2`
    - set Jaccard: `0.0287`
    - coarse motion agreement: `0.10`
    - anti-collapse score: `0.2900`
  - interpretation:
    - the held-out collapse is much weaker than before
    - this is the strongest sign that decoded-geometry loss is helping the right failure mode

- `train16`
  - old:
    - unique traj ids avg: `1.0`
    - max same-token run avg: `128.0`
    - set Jaccard: `0.0143`
  - new:
    - unique traj ids avg: `4.9`
    - max same-token run avg: `109.1`
    - set Jaccard: `0.0108`
    - coarse motion agreement: `0.10`
    - anti-collapse score: `0.1055`
  - interpretation:
    - still too collapsed for us to trust this as a stable Stage-B entry point

## What changed qualitatively

The failure mode shifted.

- before:
  - almost pure `1499 1499 1499 ...` style collapse by `4` samples

- now:
  - the model tends to emit several plateau bands instead of one dominant token
  - typical `train4` / `val` generations contain repeated segments like:
    - `1532 / 1541 / 1527 / 1519 / 1485 / 1353 / 266`

- `train16` still collapses, but the collapse is no longer completely degenerate:
  - common bands now look like:
    - `267 / 1135 / 1526 / 1597 / 1493`

This is still not good enough, but it is a real change in the attractor.

## Practical interpretation

The decoded-geometry regularizer appears to help with the exact failure mode we cared about:

- it reduces the cheapest repeated-token shortcut
- it increases token diversity substantially at `4` samples
- it lowers the longest-run metric by a large margin

However, it does **not** yet solve the multi-sample trajectory problem.

- by `16` samples:
  - repeated plateau collapse is still strong
  - set overlap remains weak
  - motion agreement is only marginal

## Conclusion

This is a real step forward, not a full fix.

- good news:
  - `A0 traj-only + decoded geometry regularization` is worth keeping
  - it materially weakens the early collapse regime

- bad news:
  - it is still not enough to justify moving to `Stage B`
  - `train16` remains too collapsed

## Recommended next step

Stay in `Stage A`.

The most sensible next move is:

- continue `A0` with a longer training budget on `train4` and `train16`
- increase emphasis on short-horizon decoded geometry
- only then test whether `A1 = GT CoT + GT traj` can be reintroduced without re-triggering collapse

Do **not** move to teacher KD yet.
