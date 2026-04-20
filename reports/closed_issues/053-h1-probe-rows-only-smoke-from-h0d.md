# 053 - H1-probe rows-only smoke from H0d

## Setup
- Base checkpoint: `full_h0d_softrel_prefix16_from_h0b_step128/final`
- Probe script: `scripts/23_probe_h1_readout.py`
- Trainable params:
  - trajectory/special token row adapter only (`embed_tokens.token_adapter.trainable_tokens_delta.default`)
  - final norm frozen
- Prefix: `16`
- Train/val: `64 / 40`
- Steps: `32`
- Loss:
  - GT-vs-1499 margin
  - GT-vs-traj-hardneg margin
  - no CE

## Result
Rows-only readout coupling moves the output distribution immediately, but too narrowly.

Validation changes:
- target-vs-1499 margin: `-3.50 -> +0.39`
- traj-top1-1499 ratio: `1.00 -> 0.00`
- hardneg margin: `-2.64 -> -2.54`
- traj-vocab target rank: `174.85 -> 391.99` (worse)
- all-vocab target rank: `100076.78 -> 147078.25` (worse)

Top-1 trajectory token moved away from `<i1499>`, but mostly collapsed onto other wrong central-band tokens (`<i1165>` dominant on val).

## Interpretation
This confirms the next bottleneck more sharply.

- H0d hidden is already readout-usable enough for a narrow rows-only probe to suppress the old `<i1499>` attractor.
- But pairwise `1499` suppression alone is not enough to raise the GT row against the broader trajectory-token row-space.
- The next probe should keep the rows-only narrow setup, but add broader trajectory-row competition:
  - stronger hard-negative pressure,
  - and/or tiny traj-vocab sampled CE after short warmup.

## Decision
- Keep `full_h0d_softrel_prefix16_from_h0b_step128` as the main hidden-first checkpoint.
- Keep `rows-only H1-probe` as the correct readout direction.
- Do not jump to full H1 yet.
- Next step: `rows-only H1-probe v2` with stronger broad traj-vocab competition.
