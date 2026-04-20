# 027 Current Trajectory Collapse Status And Next Actions

- status: in_progress
- scope:
  - consolidate the latest trajectory-collapse findings after `A0 traj-only`, subset probes, and decoded-geometry regularization
  - record what is now ruled out, what is still broken, and what should happen before `Stage B`

## Bottom line

The current blocker is no longer mysterious.

- `GT traj token` supervision is not corrupted
- tokenizer / flatten / round-trip behavior does not look broken
- the real issue is that the current discrete future-token distribution is already fairly concentrated in a central band
- under small-data autoregressive next-token CE, that creates a very cheap repeated-token shortcut
- decoded-geometry regularization helps weaken that shortcut
- but `train16` is still too collapsed to justify entering `Stage B`

So the current plan remains:

- keep working inside `Stage A`
- stabilize `GT traj` learning first
- only then reintroduce joint `GT CoT + GT traj`
- only after that consider `teacher traj hidden` or `teacher traj top-k`

## Issue 1. GT trajectory supervision itself does not look broken

symptom we needed to rule out:
- maybe the `GT traj token` files were already corrupted
- maybe tokenizer flatten order was wrong
- maybe round-trip quantize/decode quality was too poor to support learning

current evidence:
- earlier local audit already showed healthy per-sample diversity in GT trajectory tokens
- reference tokenizer-audit summary also says:
  - round-trip `xy_rmse_mean ~= 0.023 m`
  - `xy_rmse_p95 ~= 0.089 m`
  - `final_xy_err_mean ~= 0.050 m`
  - `yaw_mae_mean ~= 0.032 deg`
  - edge saturation is negligible
  - flatten order is `step-major interleaved`

interpretation:
- the discrete trajectory representation is usable
- this is not primarily a tokenizer implementation bug

current status:
- ruled out as the main blocker

## Issue 2. Token occupancy is biased toward the central band

symptom:
- the model repeatedly collapses toward a narrow future-token band during generation

current evidence:
- reference tokenizer-audit summary says:
  - used bins: `1464 / 3000`
  - median token: `1497`
  - `q10 / q50 / q90 = 1329 / 1497 / 1641`
  - `1400~1600` band accounts for about `70.9%`
  - top occupied tokens include:
    - `1498`
    - `1499`
    - `1493`

root cause interpretation:
- this is not “already broken GT”
- but it is a distribution that makes AR token CE collapse much easier
- once multiple samples compete, the model can cheaply fall into the dense middle band

current status:
- confirmed as a real risk factor
- this is now the leading explanation for the repeated-token attractor

## Issue 3. Plain multi-sample AR CE collapses even without teacher pressure

symptom:
- the model can overfit a single sample
- but as soon as the train set grows, trajectory-body generation collapses

evidence:
- see [025-subset-collapse-threshold-1-4-16.md](./025-subset-collapse-threshold-1-4-16.md)
- baseline results showed:
  - `train1`:
    - seen-train unique traj ids avg: `60.0`
  - `train4`:
    - seen-train unique traj ids avg: `2.25`
    - max same-token run avg: `119.25`
  - `train16`:
    - seen-train unique traj ids avg: `1.0`
    - max same-token run avg: `128.0`

interpretation:
- this is stronger than “held-out generalization is weak”
- collapse already happens on seen-train once we leave the single-sample regime
- therefore the primary issue is the current `GT traj CE` objective, not missing teacher signal

current status:
- confirmed
- this is the reason `Stage B` remains blocked

## Issue 4. A0 traj-only is necessary, but not sufficient by itself

symptom:
- even after removing:
  - `GT CoT`
  - `teacher CoT`
  - `teacher top-k`
- the trajectory body still showed strong collapse in quick smoke runs

what changed:
- `A0` now uses:
  - prompt mode: `traj_only`
  - target mode: `traj_only`
  - teacher off

interpretation:
- taking `CoT` out of the loop was the right move
- but removing text competition alone does not solve the collapse
- the actual loss on `GT traj` still needed to be changed

current status:
- still part of the right curriculum
- not enough as a standalone fix

## Issue 5. Decoded-geometry regularization helps, but does not finish the job

what we changed:
- kept token CE as the anchor
- added decoded-geometry regularization:
  - `traj_xyz_loss`
  - `traj_delta_loss`
  - `traj_final_loss`

artifacts:
- [026-a0-geometry-subset-threshold-1-4-16.md](./026-a0-geometry-subset-threshold-1-4-16.md)
- [subset_a0_geom_1_4_16_compare.json](/workspace/cosmos_distillation/outputs/reports/subset_a0_geom_1_4_16_compare.json)

main result:
- `train4` improved materially
  - seen-train unique ids: `2.25 -> 10.0`
  - seen-train max run: `119.25 -> 38.25`
  - val unique ids: `2.3 -> 8.7`
  - val max run: `119.5 -> 35.2`

- `train16` improved only partially
  - seen-train unique ids: `1.0 -> 5.4375`
  - seen-train max run: `128.0 -> 108.25`
  - val unique ids: `1.0 -> 4.9`
  - val max run: `128.0 -> 109.1`

interpretation:
- this is a real improvement
- the cheapest “single token forever” mode is weaker
- but long plateau collapse still dominates by `16` samples

current status:
- keep this loss family
- do not treat it as a full fix yet

## Issue 6. Teacher traj hidden / top-k is not the first lever

question:
- should we add teacher `traj` top-k or hidden signals now?

current answer:
- not yet as the primary fix

why:
- current evidence says the first failure already happens under `GT traj` supervision alone
- if we add teacher traj signals too early, we risk:
  - hiding the real objective problem
  - sharpening the wrong trajectory mode
  - making the collapse harder to diagnose

where it will likely be used later:
- after `A0` and then `A1` are stable
- inside `Stage B`
- on `traj span` only
- likely short-horizon first
- `hidden align` before `top-k logits`

current status:
- deferred
- still a valid later tool, not the immediate fix

## Current overall diagnosis

The most accurate current reading is:

- `GT traj token` representation:
  - healthy enough

- tokenizer / flatten / round-trip:
  - healthy enough

- occupancy distribution:
  - centrally biased enough to create shortcut pressure

- present AR CE objective:
  - too happy to collapse into that shortcut under multi-sample training

- decoded-geometry auxiliary loss:
  - directionally correct and worth keeping
  - not yet sufficient for `Stage B`

## Current gate for Stage B

`Stage B` should remain paused until at least the following are true:

- `train16` no longer shows severe plateau collapse on seen-train decode
- held-out val decode no longer has very long repeated-token runs
- `A1 = GT CoT + GT traj` can be added without immediately re-triggering the collapse

Right now those conditions are not met.

## Recommended next step

Stay in `Stage A` and keep the order strict:

1. continue `A0 traj-only`
2. keep `CE + decoded geometry + delta/final`
3. bias more toward short-horizon geometry if needed
4. re-check `train4 / train16` collapse
5. only then reintroduce `A1 = GT CoT + GT traj`
6. only after that consider `teacher traj hidden`
7. only later consider `teacher traj top-k`

## Practical takeaway

This is no longer a vague “something about distillation is broken” problem.

It is now a much narrower issue:

- the project is trying to teach a valid discrete future-token scaffold
- but the current autoregressive token objective is too willing to collapse into the dense middle-band occupancy mode
- the fix is to keep rebuilding `Stage A` around better `GT traj` learning before we bring teacher trajectory signals back in
