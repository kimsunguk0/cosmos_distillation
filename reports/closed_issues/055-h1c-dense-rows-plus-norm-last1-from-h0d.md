# H1c Dense Rows + Final Norm + Last1 LoRA Smoke From H0d

Base checkpoint:
- `full_h0d_softrel_prefix16_from_h0b_step128`

Goal:
- Test whether rows-only failure is caused by a last-stage transformation mismatch.
- Train:
  - trajectory/special token row adapter
  - final norm
  - last `1` language layer LoRA
- Keep the rest frozen.

Implementation note:
- Added optional `--train-last-lora-layers` and `--lora-learning-rate` to [23_probe_h1_readout.py](/workspace/cosmos_distillation/scripts/23_probe_h1_readout.py).
- Enabled gradient checkpointing plus `use_cache=False` for backbone-grad probes to fit in `16GB`.

Artifacts:
- Summary: [smoke_h1c_dense_rows_norm_last1_from_h0d.json](/workspace/cosmos_distillation/outputs/reports/smoke_h1c_dense_rows_norm_last1_from_h0d.json)
- Checkpoint: [/workspace/cosmos_distillation/outputs/checkpoints/smoke_h1c_dense_rows_norm_last1_from_h0d](/workspace/cosmos_distillation/outputs/checkpoints/smoke_h1c_dense_rows_norm_last1_from_h0d)
- Compare: [h1_readout_probe_compare.json](/workspace/cosmos_distillation/outputs/reports/h1_readout_probe_compare.json)

Final metrics:
- `traj-vocab target rank`: `174.85 -> 213.63`
- `all-vocab target rank`: `100076.78 -> 116716.47`
- `target-vs-1499 margin`: `-3.50 -> -3.00`
- `traj-top1-1499 ratio`: `1.00 -> 0.089`
- Dominant wrong traj rows: `1484`, `1478`, `1486`, `1479`

Interpretation:
- This is **better than H1b rows-only dense** on ranking:
  - `traj-vocab target rank`: `241.55 -> 213.63`
  - `target-vs-1499 margin`: `-4.00 -> -3.00`
- But it is **still worse than the H0d baseline** on `traj-vocab target rank` (`174.85`).
- So final norm + last1 LoRA gives some useful readout flexibility, but still does not make the LM reliably pick the GT trajectory token.
- Collapse has become a broader central-band readout problem, not a single-token problem.

Verdict:
- `H0d-A` remains the main hidden checkpoint.
- `H1c` shows rows-only is not enough, and tiny last-stage freedom helps a bit.
- But current H1c smoke still fails the criterion “traj-vocab target rank should beat H0d baseline”.
