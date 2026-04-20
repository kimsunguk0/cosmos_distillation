# 028 Short-Horizon Geometry Regression

- status: completed
- scope:
  - test whether stronger short-horizon decoded-geometry pressure improves `A0 traj-only` collapse behavior on `train4` and `train16`
  - compare against the previous decoded-geometry baseline from [026](./026-a0-geometry-subset-threshold-1-4-16.md)

## Setup

Candidate configuration:

- `configs/train/stage_a0_traj_only_short_horizon.yaml`
- key changes vs the previous geometry baseline:
  - `traj_xyz_loss: 0.30`
  - `traj_delta_loss: 0.12`
  - `traj_final_loss: 0.05`
  - `short_horizon_steps: 20`
  - `short_horizon_weight: 3.0`

Runs:

- `outputs/reports/subset_a0_short_train4_step60.json`
- `outputs/reports/subset_a0_short_train4_step60_infer_train.json`
- `outputs/reports/subset_a0_short_train4_step60_infer_val.json`
- `outputs/reports/subset_a0_short_train16_step60.json`
- `outputs/reports/subset_a0_short_train16_step60_infer_train.json`
- `outputs/reports/subset_a0_short_train16_step60_infer_val.json`

Comparison:

- `outputs/reports/subset_a0_short_train4_train16_compare.json`

## Main result

This change regressed sharply.

- `train4` collapsed back toward a near-single-token mode
- `train16` collapsed even more strongly into a pure single-token mode

The practical reading is:

- increasing short-horizon geometry pressure in this form did **not** counter the central-band AR shortcut
- it appears to have made that shortcut more attractive

## Quantitative summary

### `train4`

Seen-train decode:

- baseline geometry probe:
  - unique traj ids avg: `10.0`
  - max same-token run avg: `38.25`
  - anti-collapse score: `0.2705`

- short-horizon candidate:
  - unique traj ids avg: `1.25`
  - max same-token run avg: `122.25`
  - anti-collapse score: `0.1729`

Held-out val decode:

- baseline geometry probe:
  - unique traj ids avg: `8.7`
  - max same-token run avg: `35.2`
  - anti-collapse score: `0.2900`

- short-horizon candidate:
  - unique traj ids avg: `1.5`
  - max same-token run avg: `113.3`
  - anti-collapse score: `0.0653`

Qualitatively:

- generated sequences collapsed mostly into the `1485` band

### `train16`

Seen-train decode:

- baseline geometry probe:
  - unique traj ids avg: `5.4375`
  - max same-token run avg: `108.25`
  - anti-collapse score: `0.1189`

- short-horizon candidate:
  - unique traj ids avg: `1.0`
  - max same-token run avg: `128.0`
  - anti-collapse score: `0.0609`

Held-out val decode:

- baseline geometry probe:
  - unique traj ids avg: `4.9`
  - max same-token run avg: `109.1`
  - anti-collapse score: `0.1055`

- short-horizon candidate:
  - unique traj ids avg: `1.0`
  - max same-token run avg: `128.0`
  - anti-collapse score: `0.0304`

Qualitatively:

- generated sequences collapsed almost entirely into the `1475` band

## Interpretation

This does **not** mean decoded geometry is useless.

The previous geometry-aware baseline in [026](./026-a0-geometry-subset-threshold-1-4-16.md) was still directionally helpful.

What this experiment says is narrower:

- aggressively increasing short-horizon geometry pressure
- while keeping the same AR token objective

can strengthen the already-dense central occupancy shortcut instead of breaking it.

In other words:

- the model is not learning richer motion structure
- it is finding an even cheaper central-band token that satisfies the stronger local geometry objective “well enough”

## Conclusion

This tuning should **not** become the new default.

- keep the earlier geometry-aware `A0` baseline from [026](./026-a0-geometry-subset-threshold-1-4-16.md)
- discard this stronger short-horizon candidate as a regression

## Recommended next step

Do not increase short-horizon geometry pressure further in the same way.

Instead, the next attempt should change the objective shape rather than just turn the same knobs harder, for example:

- add a collapse-aware diversity or occupancy penalty
- change how token reweighting interacts with the central band
- or revisit the decoded-geometry target without making the local shortcut even cheaper
