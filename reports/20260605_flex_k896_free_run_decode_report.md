# FLEX K896 Backbone Free-Run Decode Eval

Date: 2026-06-05

## Setup

- Checkpoint: `outputs/checkpoints/flex_k896_camtime_top4_bestbase20k_s3000_b2_20260605/final`
- Eval script: `scripts/70_eval_checkpoint_free_run.py`
- Corpus: `data/corpus/vis_4per_category_val.jsonl`
- Split: `val`
- Samples: 68, category-balanced 17 x 4
- Decode path: backbone autoregressive generation of 128 trajectory discrete tokens, then trajectory tokenizer XYZ decode
- Geometry reference: `teacher_discrete`
- Summary JSON: `outputs/reports/flex_k896_final_vis68_free_run_decode_summary.json`

Command:

```bash
.venv/bin/python -u scripts/70_eval_checkpoint_free_run.py \
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl \
  --checkpoint-dir outputs/checkpoints/flex_k896_camtime_top4_bestbase20k_s3000_b2_20260605/final \
  --num-samples 68 \
  --split val \
  --max-new-tokens 192 \
  --summary-json outputs/reports/flex_k896_final_vis68_free_run_decode_summary.json \
  --device cuda:0
```

## Overall Result

| Metric | Value |
|---|---:|
| Token count match rate | 1.000 |
| ADE vs teacher discrete | 4.430 m |
| FDE vs teacher discrete | 14.043 m |
| Median ADE | 2.792 m |
| P90 ADE | 9.462 m |
| Max ADE | 22.671 m |
| Bad geometry rate | 0.176 |
| Coarse motion agreement | 0.515 |
| Anti-collapse score | 0.502 |
| Avg unique trajectory token ids | 15.191 |
| Samples with <=2 unique trajectory ids | 30 / 68 |

## Category Breakdown

| Category | n | ADE | FDE | Motion Match | Bad <=2 Unique |
|---|---:|---:|---:|---:|---:|
| curve | 4 | 5.503 | 16.479 | 0.50 | 2 |
| cut_in_merge_yield | 4 | 3.817 | 10.865 | 0.50 | 1 |
| green_light_go_straight | 4 | 2.869 | 9.382 | 0.25 | 0 |
| intersection_other | 4 | 3.211 | 10.570 | 0.50 | 2 |
| keep_lane_straight | 4 | 4.043 | 12.886 | 0.25 | 2 |
| lane_change | 4 | 3.093 | 9.364 | 0.75 | 3 |
| lead_vehicle_follow | 4 | 2.746 | 9.199 | 0.75 | 0 |
| left_turn_no_light | 4 | 6.492 | 20.624 | 1.00 | 2 |
| other | 4 | 5.987 | 19.337 | 0.50 | 1 |
| parked_stopped_obstacle_nudge | 4 | 1.931 | 6.389 | 0.50 | 3 |
| pedestrian_crosswalk | 4 | 2.178 | 6.112 | 1.00 | 2 |
| red_light_stop | 4 | 2.899 | 9.819 | 0.50 | 3 |
| right_turn_no_light | 4 | 9.557 | 29.685 | 0.25 | 2 |
| slow_decel_other | 4 | 2.542 | 8.591 | 0.75 | 2 |
| stop_sign | 4 | 3.828 | 14.934 | 0.00 | 3 |
| traffic_left_turn | 4 | 8.905 | 27.596 | 0.25 | 1 |
| traffic_right_turn | 4 | 5.702 | 16.893 | 0.50 | 1 |

## Failure Pattern

- The model emits exactly 128 trajectory tokens, so the format path is working.
- Geometry quality is still weak: 4.43 m ADE / 14.04 m FDE versus teacher discrete.
- 30 / 68 samples collapse to two or fewer unique trajectory token ids. This is a discrete-token generation collapse, not a parsing failure.
- Worst categories are turn-heavy: `right_turn_no_light`, `traffic_left_turn`, `left_turn_no_light`, `traffic_right_turn`, and `curve`.
- The FLEX pilot did not yet prove backbone trajectory generation improvement. It only proved the FLEX module can be trained and loaded into the free-run decode path.

## Conclusion

FLEX K896 pilot status: **not a usable backbone baseline yet**.

The current bottleneck is backbone discrete trajectory generation quality. Next runs should evaluate on free-run 128-token decode during training, not only training loss, because training loss did not predict the repeated-token geometry failures.
