# H1d / H1e Listwise Smokes From H0d

Base hidden checkpoint:
- `full_h0d_softrel_prefix16_from_h0b_step128`

Artifacts:
- single-batch sanity: [sanity_h1c_singlebatch_overfit.json](/workspace/cosmos_distillation/outputs/reports/sanity_h1c_singlebatch_overfit.json)
- H1d smoke: [smoke_h1d_listwise_from_h0d_train128_step32_fixkl.json](/workspace/cosmos_distillation/outputs/reports/smoke_h1d_listwise_from_h0d_train128_step32_fixkl.json)
- H1e smoke: [smoke_h1e_listwise_from_h0d_train128_step32.json](/workspace/cosmos_distillation/outputs/reports/smoke_h1e_listwise_from_h0d_train128_step32.json)
- compare: [h1d_h1e_listwise_compare.json](/workspace/cosmos_distillation/outputs/reports/h1d_h1e_listwise_compare.json)

## Single-Batch Sanity

`H1c`-style dense readout on one train sample overfits cleanly.

- baseline train/val=`same sample` traj-vocab rank: `432.0`
- final traj-vocab rank: `1.0`
- final target-vs-1499 margin: `+20.11`

Interpretation:
- current `traj-vocab CE + hardneg` plumbing is not fundamentally broken
- target local index / trajectory-vocab gather ordering are consistent enough to fit one sample
- the remaining problem is not “no gradient,” but generalization / objective bias under the full central-band prior

## H1d: Baseline-Preserving Listwise Readout

Trainable:
- trajectory row adapter
- final norm
- last `1` language layer LoRA

Loss:
- baseline-improvement listwise rank
- dynamic top-k rank
- trust-region negative KL
- tiny `1499` margin
- warmup-to-tiny balanced traj-vocab CE
- H0d hidden anchor

Result:
- H0d baseline val traj-vocab rank: `174.85`
- H1d best val traj-vocab rank: `177.54` at step `1`
- H1d final val traj-vocab rank: `255.93`
- H1d final target-vs-1499 margin: `-1.40`
- H1d final top traj tokens: `1500`, `1482`

Interpretation:
- after stabilizing the trust KL, the run is numerically valid
- but it still fails the core criterion “beat the H0d baseline rank”
- the readout again shifts mass within the central-band substitute set instead of improving GT rank

## H1e: Rows Frozen, Hidden-Side Readout Coupling

Trainable:
- final norm
- last `1` language layer LoRA

Frozen:
- trajectory row adapter

Result:
- H0d baseline val traj-vocab rank: `174.85`
- H1e best val traj-vocab rank: `174.85` at step `1`
- H1e final val traj-vocab rank: `174.85`
- H1e final target-vs-1499 margin: `-3.50`
- val top traj token stays `1499`

Interpretation:
- hidden-side-only listwise coupling does not move validation readout at all in this smoke
- rows must move if we want the current H0d hidden to alter trajectory-token preference
- but rows moving alone or with tiny last-layer freedom still drift into substitute central-band tokens

## Decision

- `H0d-A` stays the main hidden checkpoint
- `H1d` does not currently beat the H0d baseline and is not ready as the main branch
- `H1e` is weaker: it preserves the baseline instead of improving it
- combined with the single-batch overfit sanity, the evidence now points to:
  - implementation is capable of fitting the local readout task
  - but current readout objectives do not generalize under the strong central-band prior
  - the next step should move back toward a readout-aware hidden stage (`H0e`) rather than continuing to push narrow H1 probes
