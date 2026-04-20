# 045. Hidden-First Hybrid Body Still Has Poor Geometry

## Summary

The hidden-first `top-k + shared bridge` checkpoint does produce a non-collapsed hybrid body, but its decoded geometry is still far from both GT and teacher discrete trajectories.

- Hybrid val body metrics:
  - `avg_unique_traj_ids = 12.2`
  - `avg_max_same_token_run = 1.0`
  - `avg_edge_frac = 0.0`
- Geometry vs GT / teacher remains poor:
  - `avg_gen_vs_gt_ade_m = 23.30`
  - `avg_gen_vs_gt_fde_m = 45.82`
  - `avg_gen_vs_teacher_ade_m = 23.88`
  - `avg_gen_vs_teacher_fde_m = 47.22`

Teacher discrete remains much better aligned to GT:

- `avg_teacher_vs_gt_ade_m = 1.05`
- `avg_teacher_vs_gt_fde_m = 3.06`

## Artifacts

- Hybrid val decode:
  - [subset_t1_hiddenfirst_topk_bridge_prefix16_from_format_train4_step12_infer_val10_auxblend_clip.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_hiddenfirst_topk_bridge_prefix16_from_format_train4_step12_infer_val10_auxblend_clip.json)
- Geometry compare:
  - [hiddenfirst_hybrid_traj_decode_vs_gt_teacher_compare.json](/workspace/cosmos_distillation/outputs/reports/hiddenfirst_hybrid_traj_decode_vs_gt_teacher_compare.json)

## Interpretation

This narrows the earlier interpretation of hybrid success.

What hybrid proves:

- the current student hidden contains enough trajectory/affordance signal to avoid the old single-token plateau
- decode policy can use that signal to emit a full 128-token body

What hybrid does **not** prove:

- that the student hidden already contains a good 6.4-second planning trajectory
- that the hidden is already teacher-like in geometric calibration

The new hidden-first bridge checkpoint therefore supports:

1. there is weak trajectory structure in student hidden
2. hidden manifold mismatch is still real
3. non-collapse alone is not the same as good planning geometry

## Takeaway

The bottleneck is still two-part:

- hidden manifold mismatch
- LM readout / interface mismatch

The hidden-first bridge helps with the first part in a limited way, but it does not yet make the emitted body geometrically correct.
