**Summary**
`H0b-A` (`projector freeze`) and `H0b-B` (`projector low-lr=0.1x`) were both run from the same base checkpoint:

- base: `full_h0a_frozenlatent_prefix16_step128`
- train subset: `64`
- steps: `16`
- prefix mask: `16`
- LM traj CE/top-k: `0`

The comparison JSON is:
- [h0b_ab_compare.json](/workspace/cosmos_distillation/outputs/reports/h0b_ab_compare.json)

**Results**
Compared to `H0a full128`, both H0b branches improved raw and projected hidden geometry. The difference between A and B is where the gain concentrated.

Raw hidden:
- `H0a full128`: rank `3.21`, offdiag `0.9831`
- `H0b-A`: rank `3.86`, offdiag `0.9814`, centered Gram `0.420`, common ratio `0.9814`
- `H0b-B`: rank `3.79`, offdiag `0.9821`, centered Gram `0.412`, common ratio `0.9822`

Projected latent:
- `H0a full128`: offdiag `0.7298`, token cosine `0.1374`, Gram `0.3320`
- `H0b-A`: offdiag `0.5210`, token cosine `0.0424`, Gram `0.3629`
- `H0b-B`: offdiag `0.6178`, token cosine `0.0599`, Gram `0.3759`

Logit probe:
- `H0b-A`: target-vs-1499 margin `-30.04`, target rank `5647.8`
- `H0b-B`: target-vs-1499 margin `-30.97`, target rank `5539.8`

Both branches still show a severely negative target-vs-`<i1499>` margin, so H0b has not yet repaired the LM readout contract.

**Interpretation**
`H0b-A` is better if the objective is **raw-coupled manifold recovery**:
- better raw rank
- lower raw offdiag cosine
- higher raw centered Gram correlation
- lower common-mode ratio

`H0b-B` is slightly better if the objective is **projected interface smoothness**:
- slightly higher projected rank
- slightly better projected token cosine
- slightly better projected Gram correlation

But H0b-B gives back too much of the raw-hidden improvement, which was the main purpose of H0b.

**Decision**
For the stated H0b goal, `H0b-A` should remain the main branch.

The current interpretation is:
- `H0a` recovered projected latent structure
- `H0b-A` starts pulling that structure back into raw final hidden
- `H0b-B` allows the projector to absorb too much of the correction again

So the next main step should continue from `H0b-A`, not `H0b-B`.
