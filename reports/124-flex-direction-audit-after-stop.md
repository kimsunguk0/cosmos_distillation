# 124 - FLEX Direction Audit After Stop

## Current State

No FLEX training/eval process is running.

The last stopped run was:

- `flex_f63_selector_passthrough_lora4_f62_heldout256_trajonly_20260608`
- normal heldout256 finished: ADE `3.3348`, FDE `10.7773`
- camera_shuffle was stopped partway through after the user interrupted the direction

B0 comparison available in the same heldout256 traj-only setting:

- B0 normal traj-only: ADE `3.1807`, FDE `10.4378`

So F62 selector/passthrough normal is close but worse than B0 by about:

- ADE `+0.1541`
- FDE `+0.3395`

## What Was Wrong

The F62/F63 direction should not be treated as final FLEX.

F62 used:

- selected original Qwen image features
- selected original Qwen DeepStack features
- LoRA adaptation on the language model

That means it still relies on original visual feature extraction and bypasses
the learned FLEX scene encoder.  It is useful as a diagnostic upper-bound /
selector baseline, but it is not paper-style learned FLEX compression.

The overstatement was: calling selector-style K1792 the "viable FLEX path" without
qualifying that it is diagnostic, not final.

## What To Keep

These changes are structurally relevant for real FLEX and should be kept unless
a cleaner implementation replaces them:

1. Preserve official Qwen MRoPE position ids before/after image-token compression.
2. Gather rank-3 `position_ids` in `compress_batch_for_flex()`.
3. Support uniform image-token selection as a diagnostic/control.
4. Support DeepStack visual hooks for compressed FLEX, because Qwen3-VL uses
   layer-specific visual injections.
5. Keep pre-norm hidden capture, because hidden parity metrics otherwise compare
   the wrong representation.

## What To Quarantine

These are diagnostic-only and should not be used for final claims:

1. `--flex-passthrough-image-slots`
2. `--flex-residual-image-slots`
3. `--flex-dummy-image-slots`
4. F62 selector/passthrough checkpoints as "FLEX"
5. F63 heldout selector evaluation as proof of learned compression

They can remain behind explicit diagnostic flags, but training reports must mark
them as non-final.

## Real Next Gate

The next valid FLEX gate is not another selector run.

A valid gate is:

1. learned FLEX scene encoder only
2. no selected original image-feature passthrough
3. official MRoPE preserved
4. learned DeepStack projector enabled
5. 16-sample feature + free-run overfit must pass before any heldout run

F61 showed the current learned query compressor is insufficient:

- image feature cosine only `0.817` after 1000 steps
- B0 parity ADE `5.437`

Therefore the likely next implementation should change the learned compressor
architecture or initialization:

- selector-initialized / identity-biased compressor
- residual learning around selected uniform features only for initialization, not final passthrough
- stronger direct image-feature parity objective before CE/free-run objectives

One-line status:

`Current FLEX is not solved; selector/passthrough is only a diagnostic baseline, and learned FLEX needs a better compressor before more heldout evaluation.`
