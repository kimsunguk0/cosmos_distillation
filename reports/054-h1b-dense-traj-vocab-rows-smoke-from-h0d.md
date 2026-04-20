# H1b Dense Trajectory-Vocab Rows-Only Smoke From H0d

Base checkpoint:
- `full_h0d_softrel_prefix16_from_h0b_step128`

Goal:
- Replace narrow pairwise `<i1499>` suppression with dense trajectory-vocab competition.
- Train only trajectory/special token row adapter.
- Keep backbone, projector, norm, and LoRA frozen.

Config highlights:
- `traj-vocab CE (4000-way)`: `0.20`
- `dynamic top-k hard negative rank loss`: `0.20`, `k=128`
- `1499 margin`: `0.03`
- `row delta l2`: `1e-4`

Artifacts:
- Summary: [smoke_h1b_dense_rows_from_h0d.json](/workspace/cosmos_distillation/outputs/reports/smoke_h1b_dense_rows_from_h0d.json)
- Checkpoint: [/workspace/cosmos_distillation/outputs/checkpoints/smoke_h1b_dense_rows_from_h0d](/workspace/cosmos_distillation/outputs/checkpoints/smoke_h1b_dense_rows_from_h0d)

Result:
- `traj-vocab target rank`: `174.85 -> 241.55`
- `target-vs-1499 margin`: `-3.50 -> -4.00`
- `traj-top1-1499 ratio`: `1.00 -> 0.00`
- New dominant wrong rows: `1486`, `1484`, `1498`

Interpretation:
- Dense trajectory-vocab supervision is better than the earlier rows-only pairwise-margin probe at avoiding a single `<i1499>` sink.
- But it still does **not** improve correct token ranking over the H0d baseline.
- Probability mass shifts from `<i1499>` into nearby central-band trajectory rows, so this branch is still a diagnostic, not a generation candidate.

Verdict:
- `rows-only pairwise`: diagnostic only
- `rows-only dense traj-vocab`: still diagnostic only
- Next escalation should test whether readout needs a tiny amount of final transformation freedom beyond row deltas.
