# Cosmos Distillation Status Report

This README is the current high-level report for the trajectory-token collapse work.
It is intentionally placed at the repository root so GitHub shows the current project state first.

The original design README has been preserved as [DESIGN_README.md](./DESIGN_README.md).
Older issue reports have been moved to [reports/closed_issues](./reports/closed_issues/).

## Current Verdict

The original pure-LM trajectory body collapse is no longer the main blocker.

With the Claude-derived recipe:

- all-layer LoRA enabled
- 4016 custom token rows trainable through PEFT `trainable_token_indices`
- GT trajectory token CE as the main objective
- weak GT CoT loss
- output-format loss kept small
- teacher KD, teacher hidden distill, aux heads, and hybrid policy losses off
- full 959-sample corpus and enough steps

the student emits full 128-token trajectory bodies on validation instead of collapsing to a single repeated token.

The remaining problem is trajectory quality, not trajectory emission.

## Latest Confirmed Result

Full-corpus Phase 1 SFT was run for both reweighted and non-reweighted trajectory CE.

| Run | Val Samples | ADE | FDE | Avg Unique Traj IDs | Avg Max Same-Token Run | Token Match |
|---|---:|---:|---:|---:|---:|---:|
| `reweight on` | 204 | 5.11 m | 14.47 m | 17.03 | 2.88 | 1.68% |
| `reweight off` | 204 | 4.72 m | 13.76 m | 21.25 | 3.36 | 1.98% |

Both runs generated 128/128 trajectory body tokens on all 204 validation samples.
Neither run reproduced the old `<i1499>` single-token plateau.

The important correction is that reweighting is not required for non-collapse under the full recipe. It may still be a useful training-shaping knob, but the `reweight off` ablation was actually better on ADE/FDE in this run.

## What We Were Trying To Solve

The target was to distill Alpamayo-style VLM trajectory generation into a Cosmos-Reason2-2B student.

The desired output contract is:

```text
<|cot_start|> ... <|cot_end|><|traj_future_start|><i...> x128 <|traj_future_end|>
```

Early training could learn some text and formatting, but trajectory body generation collapsed into repeated central-band tokens such as `<i1499>`.
This happened even when GT trajectory token supervision was present.

## What We Tried

### Stage A Contract Repair

We first found real mechanical issues:

- prompt/full label masking was misaligned
- newly added special/traj token rows were frozen under LoRA
- boundary/format tokens were under-weighted

Those were fixed in the Stage A contract work. The important file-level fix was adding `distill_trainable_token_ids()` and passing those ids into PEFT `trainable_token_indices`, so the custom trajectory/control token rows could move during LoRA training.

### GT Trajectory CE And Geometry Regularization

After the mechanical fixes, GT trajectory CE still collapsed in small multi-sample probes.

Observed pattern:

- 1 sample could be memorized
- 4 samples already showed collapse
- 16 samples collapsed almost completely

We interpreted this as a central-band token prior problem. Geometry losses helped somewhat, but did not solve the collapse at train16/full scale.

### Teacher Top-k, Hidden, Aux, And Hybrid Experiments

We then tried several heavier approaches:

- teacher trajectory top-k KD
- teacher trajectory hidden projection
- trajectory auxiliary heads
- paired-interface and hybrid decode
- prefix pseudo-CE absorption
- hidden manifold recovery with frozen teacher latent targets
- H1 readout row probes and listwise losses

These experiments taught useful things:

- hybrid decode can avoid single-token body collapse
- student hidden has some weak trajectory signal
- hidden geometry can be improved with frozen latent targets
- row-space/readout can be moved diagnostically

But they did not produce a clean pure-LM body solution before the Claude recipe was checked properly.

## What We Missed

The main mistake was not that hidden manifold mismatch was imaginary.
It was that we escalated to hidden/readout complexity before locking down the simplest viable GT SFT baseline.

In particular:

- We had already identified trainable custom token rows in report 022, but later probes often reintroduced `freeze_all_parameters`, short runs, low LR, prefix-only objectives, teacher-pair targets, or aux/hybrid losses.
- We over-interpreted `train4` runs with 8-20 steps as evidence that LM-only trajectory generation was fundamentally broken.
- We treated "lm_head/embed opened but still plateau" as stronger evidence than it deserved, because those runs were short, mixed-objective, and not the clean full-data GT SFT recipe.
- We spent too long trying to repair hidden manifolds before asking whether the token-row adapter path plus all-layer LoRA could simply learn the GT trajectory body with enough data and steps.
- We treated trajectory token reweighting as probably essential, but the full `reweight off` ablation showed that enough clean SFT can break collapse without it.

The corrected diagnosis is:

```text
The first-order collapse was mostly a recipe/training-path issue:
custom token rows + all-layer LoRA + clean GT SFT + enough steps/data.

Hidden/KD methods may still matter later for quality,
but they were not the first bottleneck to solve.
```

## Claude-Derived Resolution

The useful Claude-side insight was to simplify the recipe instead of adding more machinery:

- remove `optimization.freeze_all_parameters`
- allow default LoRA to apply across the language stack
- pass the 4016 custom token ids to PEFT `trainable_token_indices`
- train with GT trajectory CE as the main loss
- keep GT CoT weak
- keep teacher KD and hidden distill off
- use enough samples and enough steps

This moved the project from:

```text
"Why does the model only emit one repeated trajectory token?"
```

to:

```text
"How do we make the emitted 128-token trajectory geometrically better?"
```

That is real progress.

## Current Best Interpretation

The student can now learn the pure-LM trajectory body contract.

However, the generated trajectories are still far from teacher/GT quality:

- ADE is around 4.7-5.1 m
- FDE is around 13.8-14.5 m
- exact token match is still around 2%

So the next phase should focus on improving trajectory geometry and teacher alignment, not on collapse debugging.

## Recommended Next Steps

1. Treat `reweight off` as the current main SFT candidate unless broader eval overturns it.
2. Continue the clean GT SFT run longer, selecting by validation ADE/FDE rather than only final step.
3. Reintroduce teacher trajectory top-k KD only after the SFT baseline is fixed.
4. Keep GT trajectory CE active when adding KD, so teacher signals refine rather than replace the recovered LM contract.
5. Add geometry regularization as a controlled ablation, not as the primary collapse fix.
6. Defer hidden/KV manifold distillation until SFT + teacher top-k KD stops improving geometry.

## Archive

The old numbered issue reports are archived under:

- [reports/closed_issues](./reports/closed_issues/)

These reports are not deleted. They are closed as historical debugging context.
