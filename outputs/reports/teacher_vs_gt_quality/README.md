# Teacher vs GT Quality Table

CSV: `/workspace/cosmos_distillation/outputs/reports/teacher_vs_gt_quality/teacher_vs_gt_quality_table.csv`
JSONL: `/workspace/cosmos_distillation/outputs/reports/teacher_vs_gt_quality/teacher_vs_gt_quality_table.jsonl`

## Group Policy

- `group` is now the teacher absolute geometry bin, also duplicated as `teacher_abs_bin`.
- `A_good`: teacher ADE <= 2.0 and teacher FDE <= 6.0. Starting `kd_weight=1.0`.
- `B_ok`: teacher ADE <= 3.5 and teacher FDE <= 9.0. Starting `kd_weight=0.5`.
- `C_bad`: teacher ADE > 5.0 or teacher FDE > 12.0. Starting `kd_weight=0.0`.
- `D_middle`: all remaining samples. Starting `kd_weight=0.25`.
- The older B0-relative trust split is preserved as `relative_kd_group` / `relative_kd_weight`.
- `prefix_teacher_good` is true when teacher early ADE <= 1.0 and early FDE <= 2.0.

## Summary

### overall

- samples: `958`
- teacher_abs_bin counts: `{'A_good': 707, 'B_ok': 127, 'D_middle': 53, 'C_bad': 71}`
- relative_kd_group counts: `{'group2_mixed_gt_primary': 285, 'group3_teacher_downweight_or_exclude': 328, 'group1_teacher_trusted': 345}`
- teacher failure tags: `{'ok_or_low_error': 656, 'E_long_horizon_divergence': 265, 'C_speed_scale_failure': 111, 'B_curvature_or_lateral_failure': 50, 'A_stop_or_decel_failure': 26, 'B_curvature_or_turn_direction_failure': 8, 'F_repetition_or_local_band_oscillation': 6, 'D_initial_prefix_failure': 5}`
- prefix_teacher_good: `945`
- mean teacher ADE/FDE: `1.4744` / `4.1829`
- mean student ADE/FDE: `2.0639` / `6.3721`
- mean teacher-student ADE/FDE delta: `-0.5895` / `-2.1892`
- mean teacher token match: `0.0427`

### split=train

- samples: `754`
- teacher_abs_bin counts: `{'A_good': 555, 'B_ok': 100, 'D_middle': 42, 'C_bad': 57}`
- relative_kd_group counts: `{'group2_mixed_gt_primary': 239, 'group3_teacher_downweight_or_exclude': 311, 'group1_teacher_trusted': 204}`
- teacher failure tags: `{'ok_or_low_error': 512, 'E_long_horizon_divergence': 210, 'C_speed_scale_failure': 97, 'B_curvature_or_lateral_failure': 38, 'A_stop_or_decel_failure': 21, 'F_repetition_or_local_band_oscillation': 6, 'B_curvature_or_turn_direction_failure': 5, 'D_initial_prefix_failure': 3}`
- prefix_teacher_good: `745`
- mean teacher ADE/FDE: `1.4712` / `4.1813`
- mean student ADE/FDE: `1.3592` / `4.4787`
- mean teacher-student ADE/FDE delta: `0.1120` / `-0.2974`
- mean teacher token match: `0.0441`

### split=val

- samples: `204`
- teacher_abs_bin counts: `{'B_ok': 27, 'C_bad': 14, 'A_good': 152, 'D_middle': 11}`
- relative_kd_group counts: `{'group2_mixed_gt_primary': 46, 'group3_teacher_downweight_or_exclude': 17, 'group1_teacher_trusted': 141}`
- teacher failure tags: `{'ok_or_low_error': 144, 'E_long_horizon_divergence': 55, 'C_speed_scale_failure': 14, 'B_curvature_or_lateral_failure': 12, 'A_stop_or_decel_failure': 5, 'B_curvature_or_turn_direction_failure': 3, 'D_initial_prefix_failure': 2}`
- prefix_teacher_good: `200`
- mean teacher ADE/FDE: `1.4862` / `4.1889`
- mean student ADE/FDE: `4.6685` / `13.3703`
- mean teacher-student ADE/FDE delta: `-3.1824` / `-9.1815`
- mean teacher token match: `0.0376`

## By Teacher Abs Bin

### A_good

- samples: `707`
- mean teacher ADE/FDE: `0.7786` / `2.0328`
- mean student ADE/FDE: `1.9446` / `6.0476`
- mean teacher-student ADE/FDE delta: `-1.1660` / `-4.0149`
- teacher failure tags: `{'ok_or_low_error': 656, 'C_speed_scale_failure': 30, 'E_long_horizon_divergence': 15, 'A_stop_or_decel_failure': 8, 'F_repetition_or_local_band_oscillation': 5, 'B_curvature_or_turn_direction_failure': 1}`
- prefix_teacher_good: `706`

### B_ok

- samples: `127`
- mean teacher ADE/FDE: `2.3271` / `6.4752`
- mean student ADE/FDE: `2.4248` / `7.4481`
- mean teacher-student ADE/FDE delta: `-0.0977` / `-0.9729`
- teacher failure tags: `{'E_long_horizon_divergence': 127, 'B_curvature_or_lateral_failure': 22, 'C_speed_scale_failure': 20, 'B_curvature_or_turn_direction_failure': 4, 'A_stop_or_decel_failure': 3}`
- prefix_teacher_good: `125`

### C_bad

- samples: `71`
- mean teacher ADE/FDE: `5.5307` / `17.2080`
- mean student ADE/FDE: `2.5064` / `7.5533`
- mean teacher-student ADE/FDE delta: `3.0243` / `9.6547`
- teacher failure tags: `{'E_long_horizon_divergence': 70, 'C_speed_scale_failure': 38, 'B_curvature_or_lateral_failure': 21, 'A_stop_or_decel_failure': 10, 'D_initial_prefix_failure': 5, 'B_curvature_or_turn_direction_failure': 2}`
- prefix_teacher_good: `62`

### D_middle

- samples: `53`
- mean teacher ADE/FDE: `3.2782` / `9.9229`
- mean student ADE/FDE: `2.1970` / `6.5397`
- mean teacher-student ADE/FDE delta: `1.0812` / `3.3832`
- teacher failure tags: `{'E_long_horizon_divergence': 53, 'C_speed_scale_failure': 23, 'B_curvature_or_lateral_failure': 7, 'A_stop_or_decel_failure': 5, 'B_curvature_or_turn_direction_failure': 1, 'F_repetition_or_local_band_oscillation': 1}`
- prefix_teacher_good: `52`

## By B0-Relative KD Group

### group1_teacher_trusted

- samples: `345`
- mean teacher ADE/FDE: `0.7106` / `1.8770`
- mean student ADE/FDE: `3.8132` / `11.6609`
- mean teacher-student ADE/FDE delta: `-3.1026` / `-9.7839`
- teacher failure tags: `{'ok_or_low_error': 332, 'C_speed_scale_failure': 7, 'E_long_horizon_divergence': 5, 'A_stop_or_decel_failure': 4}`
- prefix_teacher_good: `344`

### group2_mixed_gt_primary

- samples: `285`
- mean teacher ADE/FDE: `1.1165` / `2.9986`
- mean student ADE/FDE: `1.5607` / `4.9360`
- mean teacher-student ADE/FDE delta: `-0.4441` / `-1.9374`
- teacher failure tags: `{'ok_or_low_error': 192, 'E_long_horizon_divergence': 79, 'C_speed_scale_failure': 24, 'B_curvature_or_lateral_failure': 13, 'F_repetition_or_local_band_oscillation': 5, 'B_curvature_or_turn_direction_failure': 2, 'A_stop_or_decel_failure': 1}`
- prefix_teacher_good: `284`

### group3_teacher_downweight_or_exclude

- samples: `328`
- mean teacher ADE/FDE: `2.5887` / `7.6373`
- mean student ADE/FDE: `0.6612` / `2.0570`
- mean teacher-student ADE/FDE delta: `1.9274` / `5.5803`
- teacher failure tags: `{'E_long_horizon_divergence': 181, 'ok_or_low_error': 132, 'C_speed_scale_failure': 80, 'B_curvature_or_lateral_failure': 37, 'A_stop_or_decel_failure': 21, 'B_curvature_or_turn_direction_failure': 6, 'D_initial_prefix_failure': 5, 'F_repetition_or_local_band_oscillation': 1}`
- prefix_teacher_good: `317`
