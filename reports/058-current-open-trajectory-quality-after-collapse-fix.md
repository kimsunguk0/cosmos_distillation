# 058 Current Open Issue: Trajectory Quality After Collapse Fix

- status: open
- scope:
  - record the current post-collapse state
  - separate the solved trajectory-emission problem from the remaining geometry-quality problem
  - define the next experiments after the Claude-derived SFT baseline

## Current State

The old pure-LM trajectory body collapse is no longer the primary blocker.

The Claude-derived recipe showed that the student can emit complete 128-token trajectory bodies when trained with:

- all-layer LoRA
- 4016 custom trajectory/control token rows trainable through PEFT `trainable_token_indices`
- GT trajectory token CE as the main objective
- weak GT CoT loss
- small output-format loss
- teacher KD, teacher hidden alignment, aux heads, and hybrid-policy losses disabled
- full 959-sample corpus and enough steps

This means the question has changed from:

```text
Can the model emit trajectory tokens at all?
```

to:

```text
Can the emitted 128-token trajectory match GT/teacher geometry well enough?
```

## Latest Confirmed Metrics

Validation decode on 204 samples:

| Run | ADE | FDE | Avg Unique Traj IDs | Avg Max Same-Token Run | Token Match |
|---|---:|---:|---:|---:|---:|
| reweight on | 5.11 m | 14.47 m | 17.03 | 2.88 | 1.68% |
| reweight off | 4.72 m | 13.76 m | 21.25 | 3.36 | 1.98% |

Both runs generated 128/128 trajectory body tokens on all 204 validation samples.
Neither run reproduced the old `<i1499>` full-body plateau.

## What Was Solved

- The model can now emit the required trajectory body span.
- The old average max-run `128` collapse is gone.
- `reweight off` is viable and currently slightly better than `reweight on` on ADE/FDE.
- The token-row/readout path is capable of learning the trajectory contract when the recipe is clean and long enough.

## What Is Still Open

The generated path is still not teacher-quality.

Remaining gaps:

- ADE/FDE are still high.
- Exact token match is still around 2%.
- The model still overuses central-band trajectory tokens, just not as a single-token plateau.
- It is not yet clear whether more clean SFT, teacher top-k KD, geometry losses, or hidden/KV distillation gives the best next improvement.

## Corrected Diagnosis

The first-order collapse was mostly a training-path and recipe issue:

```text
custom token rows + all-layer LoRA + clean GT SFT + enough data/steps
```

Earlier hidden-manifold and readout probes were useful but over-escalated too early.
Many of them were short, prefix-only, low-LR, teacher-pair, aux/hybrid, or hidden-first experiments rather than a clean full-data GT SFT baseline.

Hidden/KD work may still matter later, but it should now be used to improve geometry, not to explain the old emission collapse.

## Next Recommended Experiments

1. **Promote `reweight off` to the main SFT candidate for now.**
   Validate on broader splits and overlays before fully committing.

2. **Continue clean GT SFT longer.**
   Run from the `reweight off` checkpoint to 8k-10k total steps, saving periodic checkpoints.

3. **Select by validation geometry, not final step.**
   Evaluate ADE/FDE, unique-token count, max-run, top-token histogram, and overlays every checkpoint interval.

4. **Reintroduce teacher trajectory top-k KD after SFT stabilizes.**
   Start with low LR and keep GT traj CE active so KD refines the recovered LM contract instead of replacing it.

5. **Run geometry regularization as a controlled branch.**
   Add weak `traj_xyz`, `traj_delta`, or `traj_final` losses only after the clean SFT continuation baseline is measured.

6. **Defer hidden/KV distillation.**
   Bring it back only if SFT + teacher top-k KD fails to reduce ADE/FDE enough.

## Active Decision

Do not reopen the old collapse-debugging stack as the main path.

The active path is now:

```text
clean GT SFT continuation
-> best-ADE checkpoint selection
-> teacher traj top-k KD refinement
-> optional geometry loss
-> hidden/KV distill only if geometry plateaus
```

## Related Documents

- Current root summary: [../README.md](../README.md)
- Design README: [../DESIGN_README.md](../DESIGN_README.md)
- Closed issue archive: [closed_issues/README.md](./closed_issues/README.md)
