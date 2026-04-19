**Summary**
`H0c token-align` was tested as a follow-up to `H0a full128`:

- base: `full_h0a_frozenlatent_prefix16_step128`
- train subset: `64`
- steps: `16`
- projector: trainable at `0.05x lr`
- added latent InfoNCE-style contrastive alignment

Artifacts:
- config: [stage_h0c_tokenalign_prefix16_projectorlowlr.yaml](/workspace/cosmos_distillation/configs/train/stage_h0c_tokenalign_prefix16_projectorlowlr.yaml)
- train summary: [smoke_h0c_tokenalign_prefix16_projectorlowlr.json](/workspace/cosmos_distillation/outputs/reports/smoke_h0c_tokenalign_prefix16_projectorlowlr.json)
- probe: [smoke_h0c_tokenalign_prefix16_projectorlowlr_hiddenprobe_val40_rawproj.json](/workspace/cosmos_distillation/outputs/reports/smoke_h0c_tokenalign_prefix16_projectorlowlr_hiddenprobe_val40_rawproj.json)
- comparison: [h0c_tokenalign_compare.json](/workspace/cosmos_distillation/outputs/reports/h0c_tokenalign_compare.json)

**Result**
This branch did **not** beat `H0b-A`.

Compared to `H0b-A`:

Raw hidden:
- rank: `3.86 -> 3.38`
- offdiag cosine: `0.9814 -> 0.9833`
- centered Gram corr: `0.420 -> 0.388`

Projected latent:
- offdiag cosine: `0.521 -> 0.893`
- Gram corr: `0.363 -> 0.230`
- token cosine: `0.042 -> 0.062`

Logit probe:
- target-vs-1499 margin: `-30.04 -> -31.44`
- target rank: `5647.8 -> 5391.2`

**Interpretation**
The contrastive/token-alignment branch slightly recovered projected token identity relative to `H0b-A`, but it paid for that with a large collapse in projected geometry and weaker raw-hidden recovery.

So the current pattern is:
- `H0a`: projected manifold recovery
- `H0b-A`: best raw-coupled geometry recovery so far
- `H0c`: naive token-contrastive coupling is too aggressive and destroys the geometry gains

The target-vs-`<i1499>` margin also did not improve, so this branch does not yet justify an H1 transition.

**Decision**
`H0c token-align` is a useful negative result, but it should **not** replace `H0b-A` as the main line.

The better next move is to keep `H0b-A` as the base and add a more conservative identity signal, rather than using this direct contrastive version as-is.
