# 109 - FLEX Actual DeepStack Projector Check

Date: 2026-06-07

## Question

Check whether compressed FLEX can safely use the real Qwen3-VL DeepStack visual injection path.

This corrects the previous F38-F43 interpretation: those runs passed `--flex-scene-deepstack`, but the wrapper was checking DeepStack attributes on the language model. For this backbone the active hooks live under:

`conditional.visual.deepstack_visual_indexes = [5, 11, 17]`

So F42's best result was effectively compressed FLEX without actual compressed DeepStack injection.

## Code Fix

Files changed:

- `src/model/student_wrapper.py`
  - Added `FlexDeepStackProjector`.
  - Reads DeepStack layer count from `conditional.visual.deepstack_visual_indexes` or `conditional.visual.deepstack_merger_list` first.
  - Supports layer-specific low-rank projector outputs for compressed scene tokens.
  - Zero-initializes projector output weights so enabling the projector starts from no-compressed-DeepStack behavior, not the harmful repeated-scene baseline.
- `src/model/checkpoint_io.py`
  - Saves and loads `flex_deepstack_projector.pt`.
  - Stores `flex_deepstack_projector_config` in checkpoint manifests.
- `scripts/105_train_flex_teacher_parity.py`
  - Adds `--flex-deepstack-projector-rank`, `--flex-deepstack-projector-dropout`, and `--train-flex-deepstack-projector`.
  - Infers projector layer count from the real visual DeepStack hooks.

Compile check:

`.venv/bin/python -m py_compile src/model/student_wrapper.py src/model/checkpoint_io.py scripts/105_train_flex_teacher_parity.py scripts/25_decode_checkpoint_overlays.py src/inference/checkpoint_eval.py scripts/104_eval_flex_teacher_parity.py scripts/110_debug_flex_generation_parity.py`

Result: pass.

## 16-Sample B0 Parity Results

All rows compare FLEX free-run trajectory tokens against the fixed no-FLEX B0 free-run target on the same 16 samples.

| Run | Actual compressed DeepStack | Token match | B0 ADE m | B0 FDE m | Unique tokens | Max same-token run |
|---|---|---:|---:|---:|---:|---:|
| F42 best, no actual DeepStack | Off | 0.778 | 0.380 | 1.513 | 21.94 | 1.25 |
| F42 redecoded with repeated scene DeepStack | On, repeated scene tokens | 0.161 | 9.575 | 28.977 | 19.81 | 14.69 |
| F44b projector-only step 1000 | On, rank64 projector | 0.752 | 0.534 | 1.952 | 21.19 | 1.12 |
| F44b projector-only step 2000 | On, rank64 projector | 0.737 | 0.762 | 2.909 | 19.00 | 1.12 |
| F44b projector-only final/step 3000 | On, rank64 projector | 0.724 | 0.787 | 2.996 | 21.31 | 1.12 |

Artifacts:

- F42 no actual DeepStack best: `outputs/reports/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607_step002000_b0_trajonly_parity_summary.json`
- F42 actual repeated-scene DeepStack failure: `outputs/reports/flex_f42_step002000_actual_scene_deepstack_redecode_20260607_b0_trajonly_parity_summary.json`
- F44b step 1000: `outputs/reports/flex_f44b_deepstack_projector_only_from_f42best_overfit16_s3000_lr1e5_20260607_step001000_b0_trajonly_parity_summary.json`
- F44b step 2000: `outputs/reports/flex_f44b_deepstack_projector_only_from_f42best_overfit16_s3000_lr1e5_20260607_step002000_b0_trajonly_parity_summary.json`
- F44b final: `outputs/reports/flex_f44b_deepstack_projector_only_from_f42best_overfit16_s3000_lr1e5_20260607_overfit16_b0_trajonly_parity_summary.json`

## F44b Setup

F44b starts from F42 step 2000 and trains only the DeepStack projector:

- Student init: `outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`
- Trainable params: `790,528`
- Trainable group: `flex_deepstack_projector`
- Projector: hidden `2048`, layers `3`, rank `64`, dropout `0.0`
- Steps: `3000`
- Batch: `1`
- LR: `1e-5`
- Losses: free-run token CE + trajectory state alignment

Training loss did not translate into better free-run geometry. The best F44b checkpoint is step 1000 at `0.534 m` ADE, still worse than F42 no-actual-DeepStack at `0.380 m` ADE.

## Interpretation

1. The real compressed DeepStack path is not ready.
2. Repeating the same compressed scene embeddings into all three Qwen3-VL DeepStack hooks is catastrophic.
3. A zero-initialized layer-specific rank64 projector prevents the catastrophic failure, but projector-only training still degrades B0 free-run parity versus leaving compressed DeepStack off.
4. The current target-context CE and trajectory-state alignment losses are not sufficient to make the DeepStack projector useful in free-run.

## Decision

Mainline FLEX should keep actual compressed DeepStack off for now.

Current best usable FLEX checkpoint remains:

`outputs/checkpoints/flex_f42_scene_deepstack_long_state_from_f41_overfit16_s8000_lr2e7_20260607/step_002000`

Use it without `--flex-scene-deepstack` for deployment/eval until a stronger DeepStack objective passes the 16-sample B0 parity gate.

Next structural options:

- Train full FLEX + projector jointly from the zero-output projector start, not projector-only.
- Add explicit no-FLEX teacher DeepStack hidden parity per hook/layer.
- Add position-preserving diagnostic with actual DeepStack projector to isolate content compression from token-position shift.
- Only after the 16-sample B0 parity gate improves below F42's `0.380 m` should this branch go to 512 train/val.

One-line status:

`Actual compressed DeepStack projector pass: N. Mainline FLEX = F42 no actual DeepStack, 0.380 m ADE / 1.513 m FDE on 16-sample B0 parity.`
