# 078 Action Expert Cleanup Manifest

Cleanup time: 2026-06-01 07:07:42

Kept:

- `outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531`

Deleted first-level output directories/files:

- `outputs/action_expert/ae28_fixed_seed_overfit16_20260521_020204` (6.7G)
- `outputs/action_expert/b0_recipe_validation` (8.5G)
- `outputs/action_expert/b0_self_recon_sanity` (13G)
- `outputs/action_expert/fm_collapse_diagnostics` (216K)
- `outputs/action_expert/g0_random_fm_mlp_seed42_b1` (16K)
- `outputs/action_expert/g0_random_fm_mlp_seed42_b16` (16K)
- `outputs/action_expert/g0_random_fm_mlp_seed42_b1_s5000` (16K)
- `outputs/action_expert/g0_random_fm_mlp_seed42_b256` (16K)
- `outputs/action_expert/h1_ablation_ladder_seed42_draw16` (52K)
- `outputs/action_expert/h2_noattn_seed42_draw16` (16K)
- `outputs/action_expert/h3_lr_grid_seed42_draw1` (24K)
- `outputs/action_expert/h3_lr_grid_seed42_draw16` (40K)
- `outputs/action_expert/prefill_only_ae_paths` (20K)
- `outputs/action_expert/sanity_target_gt_check` (5.3G)
- `outputs/action_expert/sanity_target_teacher_check` (5.3G)
- `outputs/action_expert/stage1_heldout_sanity_128_s100_seed42` (5.3G)
- `outputs/action_expert/stage1_heldout_sanity_128_s200_seed42_full444k` (2.7G)
- `outputs/action_expert/stage1_heldout_sanity_128_s50_seed42_full444k` (2.7G)
- `outputs/action_expert/student_ae28` (95G)
- `outputs/action_expert/student_ae28_cached_timesamples` (27G)
- `outputs/action_expert/student_ae28_official` (96G)
- `outputs/action_expert/student_ae28_official_cached_overfit` (34G)
- `outputs/action_expert/student_ae28_streaming_kv` (57G)
- `outputs/action_expert/student_kv_adapter_ae36` (69G)
- `outputs/action_expert/sweep` (380K)
- `outputs/action_expert/teacher8b_scratch_ae36` (17G)
- `outputs/action_expert/teacher_ae36_self_reconstruction` (144K)
- `outputs/action_expert/teacher_kv36_scratch_ae36` (26G)
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_fixed_seed42` (40K)
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_seed42` (40K)
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_teacher_action_in_seed42` (40K)
- `outputs/action_expert/v2_e3_actual_bundle_no_kv_teacher_projections_seed42` (40K)
- `outputs/action_expert/verify_v1_projection_trainable_seed42` (5.3G)
- `outputs/action_expert/w1_e3_actual_bundle_recipe_seed42_draw16` (60K)
- `outputs/action_expert/w4_e3_actual_bundle_recipe_seed42_draw8` (40K)
- `outputs/action_expert/x1_x2_sampling_best_step2750_full32` (424K)
- `outputs/action_expert/x1_x2_sampling_best_step2750_full32_steps10_20_40_80_seed5_best8` (424K)
- `outputs/action_expert/y1_temperature_single_full32` (316K)
- `outputs/action_expert/y2_temperature_selection_full32` (1.1M)
- `outputs/action_expert/y2_temperature_selection_n16_full32` (484K)

Total before: `485G`
Also deleted redundant current-run checkpoints: `step_001000.pt`, `step_002000.pt`, `step_003000.pt` (kept `best.pt` and `step_004000.pt`).

Total after: `14G`
