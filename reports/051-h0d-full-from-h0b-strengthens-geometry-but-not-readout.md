**Summary**
`H0d-A` (`soft relational KL + residual diagonal alignment`, projector frozen) was extended from the earlier smoke run to a larger run:

- base: `smoke_h0b_rawcoupled_prefix16_projectorfreeze`
- train set: `754`
- val set: `204`
- steps: `128`
- prefix mask: `16`
- LM traj CE/top-k: `0`

Artifacts:
- train summary: [full_h0d_softrel_prefix16_from_h0b_step128.json](/workspace/cosmos_distillation/outputs/reports/full_h0d_softrel_prefix16_from_h0b_step128.json)
- hidden probe: [full_h0d_softrel_prefix16_from_h0b_step128_hiddenprobe_val40_rawproj.json](/workspace/cosmos_distillation/outputs/reports/full_h0d_softrel_prefix16_from_h0b_step128_hiddenprobe_val40_rawproj.json)
- LM sanity: [full_h0d_softrel_prefix16_from_h0b_step128_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/full_h0d_softrel_prefix16_from_h0b_step128_infer_val10_lm.json)
- comparison: [h0d_full_compare.json](/workspace/cosmos_distillation/outputs/reports/h0d_full_compare.json)

**Results**
Against the main earlier checkpoints, the large `H0d-A` run continued to improve hidden geometry.

Raw final hidden:
- baseline rank: `2.20`
- `H0a full128`: `3.21`
- `H0b-A smoke`: `3.86`
- `H0d-A smoke`: `4.33`
- `H0d-A full128`: `5.34`

Raw structure:
- raw offdiag cosine: `0.9846 -> 0.9815`
- raw centered Gram corr: `0.420 -> 0.649`
- raw spectral entropy: `1.346 -> 1.672`

Projected latent:
- projected offdiag cosine: `0.521 -> 0.483`
- projected Gram corr: `0.363 -> 0.612`
- projected token cosine: `0.042 -> 0.058`

Logit probe:
- target-vs-`<i1499>` margin: `-30.04 -> -27.80`
- target rank: `5647.8 -> 5941.0`

LM-only decode sanity stayed alive:
- `CoT -> <|traj_future_start|> -> 128 body -> <|traj_future_end|>` survived on val 10
- but body collapse remained unchanged:
  - avg unique traj ids: `1.0`
  - avg max same-token run: `128.0`

**Interpretation**
This is the strongest hidden-first branch so far.

What improved:
- raw final hidden is no longer only barely moving; geometry recovery is now clearly visible in raw rank, centered Gram correlation, and spectral entropy
- projected latent geometry improved further without the severe regression seen in `H0c`
- token-wise latent identity recovered a little relative to `H0b-A`
- the average target-vs-`<i1499>` margin improved slightly

What did not improve:
- free LM decoding is still full `<i1499>` plateau
- token-discriminative readout remains weak
- target rank actually worsened, so hidden improvement is still not turning into usable LM token preference

The current picture is:
- `H0a`: projected latent recovery
- `H0b-A`: raw-coupled manifold recovery starts
- `H0d-A`: raw-coupled manifold recovery continues and remains stable at larger scale
- but LM readout is still not repaired

**Decision**
`H0d-A` should replace `H0b-A` as the hidden-first main branch.

However, this result is still **not** enough to justify a full `H1` transition.
The next step should be a **small readout-coupling probe** rather than full LM contract training:

- keep the `H0d-A` hidden losses alive
- open only a narrow readout path at very low LR
- test whether logit margin and target rank move before expecting free LM body to improve
