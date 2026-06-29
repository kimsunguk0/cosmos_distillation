# 108 - FLEX DeepStack Fix and Long 16-Sample Overfit

Date: 2026-06-07

## Question

The FLEX wrapper had to pass the no-op/residual parity gate first, then show whether compressed FLEX can overfit the 16-sample B0 free-run target set when trained long enough.

Teacher/B0:

`outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`

16-sample target:

`outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json`

## Fix

Root cause found in F37:

- Qwen3-VL does not use only the masked-scattered image embeddings.
- It also injects visual information through DeepStack hooks in the language model.
- The FLEX wrapper had replaced/compressed image placeholders but did not preserve or replace `deepstack_visual_embeds`.
- This made even residual scale-0/no-op FLEX diverge from B0 during generation.

Code changes:

- `src/model/student_wrapper.py`
  - `_qwen_visual_features()` now returns both image embeds and DeepStack embeds.
  - residual/no-op FLEX forwards original `visual_pos_masks` and `deepstack_visual_embeds`.
  - compressed FLEX can optionally inject scene tokens into DeepStack via `flex_scene_deepstack`.
- `src/inference/checkpoint_eval.py`
  - manual FLEX decode now preserves Qwen mRoPE cache offset via `rope_deltas`.
- `scripts/25_decode_checkpoint_overlays.py`
  - added `--flex-scene-deepstack`.
- `scripts/104_eval_flex_teacher_parity.py`, `scripts/105_train_flex_teacher_parity.py`
  - added `--flex-scene-deepstack`.

No-op validation:

| Run | B0 token match | B0 ADE | B0 FDE |
|---|---:|---:|---:|
| F37 residual scale 0 + DeepStack + rope fix | 1.000 | 0.000 | 0.000 |

Artifact:

`outputs/reports/flex_f37_residual_scale0_deepstack_rope_smoke16_20260607_b0_trajonly_parity_summary.json`

## Compressed 16-Sample Overfit Results

All rows compare generated FLEX trajectory tokens against the fixed B0 free-run trajectory tokens on the same 16 samples.

| Run | Checkpoint | Token match | B0 ADE m | B0 FDE m | Unique tokens |
|---|---|---:|---:|---:|---:|
| F38 CE only | `flex_f38.../final` | 0.551 | 2.341 | 7.617 | 15.75 |
| F40 CE + traj-state | `flex_f40.../final` | 0.533 | 0.768 | 2.494 | 15.88 |
| F41 state-heavy | `flex_f41.../final` | 0.693 | 0.523 | 2.035 | 16.44 |
| F42 long state | `flex_f42.../step_002000` | 0.778 | 0.380 | 1.513 | 21.94 |
| F43 continue, +2k | `flex_f43.../step_002000` | 0.720 | 0.814 | 3.107 | 21.25 |
| F43 continue, +4k | `flex_f43.../step_004000` | 0.760 | 0.569 | 2.159 | 18.69 |
| F43 continue, +6k | `flex_f43.../final` | 0.754 | 0.396 | 1.575 | 21.63 |

Best so far:

`outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`

Best report:

`outputs/reports/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607_step002000_b0_trajonly_parity_summary.json`

## Interpretation

1. The old no-op failure was an implementation bug, not FLEX capacity: DeepStack visual injection was missing.
2. Target-context token CE alone learns token logits but does not preserve free-run geometry.
3. Trajectory hidden-state alignment is necessary. It reduced B0 ADE from 2.34 m to 0.77 m even when token match barely changed.
4. Longer state-heavy overfit improves further, but only up to a point. F42 step 2000 is the current best; continuing from there degraded free-run parity.
5. Current compressed FLEX still does not exactly reproduce B0 on 16 samples. The remaining gap is likely structural: repeated same scene tokens for all DeepStack layers are a crude approximation of Qwen3-VL's layer-specific visual injections.

## Next Action

Do not scale FLEX to full train yet.

Next diagnostic should be structural:

- Add layer-specific DeepStack projections for compressed scene tokens instead of repeating the same scene embedding at every DeepStack injection layer.
- Re-run the same 16-sample B0 parity gate.
- Only after 16-sample parity is near zero should FLEX-only 512 train/val and full-val compression parity be trusted.

Current one-line status:

`FLEX no-op fixed: Y. Compressed FLEX best 16-sample B0 parity: 0.380 m ADE / 1.513 m FDE. Full-scale FLEX: not ready.`
