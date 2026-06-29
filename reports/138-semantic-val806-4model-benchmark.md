# Report 138: Semantic Val806 4-Model Benchmark

Date: 2026-06-12

## Scope

- Benchmark public Alpamayo 10B, no-FLEX 2B+AE28, FLEX K512+AE28, and FLEX K512+AE14.
- All models generate their own CoT/prefix before trajectory inference.
- Metrics are computed on the same semantic validation benchmark set.

## Dataset

- Corpus: `/home/pm97/workspace/sukim/distillation/cosmos_distillation/data/corpus/benchmark_semantic_val_cap50_seed42.jsonl`
- Selected samples: `806`
- Category counts:
  - `curve`: 50
  - `cut_in_merge_yield`: 50
  - `green_light_go_straight`: 50
  - `intersection_other`: 50
  - `keep_lane_straight`: 50
  - `lane_change`: 50
  - `lead_vehicle_follow`: 50
  - `left_turn_no_light`: 50
  - `other`: 50
  - `parked_stopped_obstacle_nudge`: 50
  - `pedestrian_crosswalk`: 50
  - `red_light_stop`: 38
  - `right_turn_no_light`: 50
  - `slow_decel_other`: 50
  - `stop_sign`: 50
  - `traffic_left_turn`: 18
  - `traffic_right_turn`: 50

## Eval Settings

- `eval_num_paths`: `6`
- `eval_temperature`: `0.85`
- `eval_selection_method`: `mean_traj`
- `default_inference_steps`: `10`
- `ae14_inference_steps`: `4`
- `batch_size`: `4`
- `student_batch_size`: `8`
- `attn_implementation`: `flash_attention_2`
- `dtype`: `bfloat16`
- `seed`: `42`

## Results

| Model | N | ADE GT | FDE GT | minADE6 GT | minFDE6 GT | ADE vs 10B | minADE6 vs 10B | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alpamayo-1.5-10B | 806 | 1.6742 | 4.8004 | 0.9280 | 2.7102 | NA | NA | 1917.3970 |
| Student-2B-NoFLEX-AE28 | 806 | 2.7227 | 8.1559 | 1.6835 | 5.0521 | 2.3113 | 1.3832 | 616.2138 |
| Student-2B-FLEXK512-AE28 | 806 | 3.0818 | 9.3282 | 2.0721 | 6.2832 | 2.6308 | 1.6985 | 525.1725 |
| Student-2B-FLEXK512-AE14 | 806 | 3.1970 | 9.6055 | 2.5478 | 7.6595 | 2.8218 | 2.1868 | 493.0510 |

## Artifacts

- Combined summary: `outputs/benchmarks/semantic_val806_4models_20260612/summary.json`
- Prediction NPZ root: `outputs/benchmarks/semantic_val806_4models_20260612/predictions`
- Visualizations: `outputs/benchmarks/semantic_val806_4models_20260612/visualizations`
- `teacher10b` rows: `outputs/benchmarks/semantic_val806_4models_20260612/teacher10b/rows.jsonl`
- `student_noflex_ae28` rows: `outputs/benchmarks/semantic_val806_4models_20260612/student_noflex_ae28/rows.jsonl`
- `student_flex_ae28` rows: `outputs/benchmarks/semantic_val806_4models_20260612/student_flex_ae28/rows.jsonl`
- `student_flex_ae14` rows: `outputs/benchmarks/semantic_val806_4models_20260612/student_flex_ae14/rows.jsonl`

## Notes

- `ADE GT` is the deployable selected trajectory using `eval_selection_method`.
- `minADE6 GT` is oracle best-of-6 against GT for diagnostic comparison.
- Student `vs 10B` metrics compare student trajectories against the 10B selected trajectory on the same sample.
- AE14 is evaluated with the configured AE14 denoising step count, currently 4 steps for deployment-oriented latency.
