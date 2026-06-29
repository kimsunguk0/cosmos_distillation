# FLEX F5 Sequence-CE Camera-Contrast Run

Date: 2026-06-06

## Why

F4b showed that hidden/logprob pairwise parity is not enough:

- F4b internal pairwise overfit passed: action_pre delta norm ratio `0.98`.
- F4b teacher-forced parity stayed strong: action_pre cosine `0.994`, traj KL `0.015`.
- But free-run camera_shuffle sensitivity still failed: shuffle gap only `+0.053 / +0.128`, while B0 no-FLEX gap is `+1.064 / +2.740`.

F5 tested a stronger generation-level objective: train FLEX to reproduce B0 free-run trajectory tokens separately for `normal` and `camera_shuffle`.

## Code Change

`src/training/collator.py` now supports deterministic image ablations:

- `normal`
- `black`
- `gray`
- `noise`
- `camera_shuffle`

`scripts/105_train_flex_teacher_parity.py` now supports:

- `--free-run-token-targets mode=decode_summary.json`
- `--free-run-token-ce-weight`
- `--free-run-token-force-context`

This lets training replace trajectory context/labels with B0 free-run generated tokens for each ablation mode.

## Run

- run name: `flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606`
- init: `outputs/checkpoints/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606/final`
- teacher: B0 no-FLEX checkpoint
- base samples: `68` vis hard samples
- train rows: `136` = `normal + camera_shuffle`
- steps: `3000`
- batch size: `2`
- LR: `5e-6`
- trainable params: `66,513,920`
- trainable groups:
  - FLEX scene encoder: `31,378,432`
  - multimodal projector: `25,174,016`
  - last-4 language LoRA: `9,961,472`
- free-run token targets:
  - normal: `outputs/reports/b0_step006250_vis68_decode_normal_summary.json`
  - camera_shuffle: `outputs/reports/b0_step006250_vis68_decode_camera_shuffle_summary.json`
- losses:
  - free-run token CE: `5.0`
  - text KL: `0.05`
  - format KL: `0.05`
  - boundary cosine/norm: `0.01 / 0.02`
  - pairwise boundary delta cosine/norm: `0.02 / 0.02`
  - traj KL: `0.0`

Artifacts:

- checkpoint: `outputs/checkpoints/flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606/final`
- train summary: `outputs/reports/flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606_train_summary.json`
- parity summary: `outputs/reports/flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606_eval_vis68_summary.json`
- decode summaries:
  - normal: `outputs/reports/flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606_vis68_decode_normal_summary.json`
  - camera_shuffle: `outputs/reports/flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606_vis68_decode_camera_shuffle_summary.json`
  - black: `outputs/reports/flex_f5_seqce_camera_contrast_vis68_from_f4b_s3000_lr5e6_20260606_vis68_decode_black_summary.json`

## Train Result

F5 did learn the explicit B0 free-run token targets:

| Step | Loss | Free-run Token CE | Free-run Token Acc | Action Pre Cos | Pair Delta Norm Ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | `5.334` | `1.063` | `0.910` | `0.987` | `1.306` |
| 2700 | `1.794` | `0.352` | `0.941` | `0.904` | `1.148` |
| 3000 | `2.039` | `0.401` | `0.937` | `0.904` | `1.169` |

But the stronger token objective damaged B0 teacher-forced parity:

| Metric | F4b | F5 |
|---|---:|---:|
| action_pre cosine mean | `0.994` | `0.852` |
| action_pre norm ratio mean | near `1.0` | `0.555` |
| cot_end cosine mean | `0.988` | `0.832` |
| traj KL mean | `0.015` | `0.528` |
| traj top1 agreement | `0.895` | `0.714` |
| teacher top1 in student top5 | `0.998` | `0.964` |
| TF argmax ADE delta | `0.0006m` | `0.0109m` |

## Free-Run Result

| Model | Normal ADE/FDE | Shuffle ADE/FDE | Black ADE/FDE | Shuffle Gap | Black Gap | Unique Normal |
|---|---:|---:|---:|---:|---:|---:|
| B0 no-FLEX | `3.101 / 10.011` | `4.165 / 12.751` | `4.296 / 13.536` | `+1.064 / +2.740` | `+1.195 / +3.525` | `27.69` |
| F3 ablation parity | `3.423 / 11.181` | `3.373 / 10.661` | `4.299 / 13.644` | `-0.051 / -0.520` | `+0.876 / +2.463` | `20.74` |
| F4b pair contrast | `3.482 / 11.118` | `3.535 / 11.246` | `4.259 / 13.483` | `+0.053 / +0.128` | `+0.777 / +2.365` | `22.71` |
| F5 sequence CE | `3.700 / 11.886` | `4.024 / 12.907` | `4.559 / 14.149` | `+0.324 / +1.021` | `+0.860 / +2.264` | `20.68` |

F5 recovered some camera_shuffle sensitivity, but not enough:

- shuffle ADE gap improved over F4b: `+0.053 -> +0.324`
- shuffle FDE gap improved over F4b: `+0.128 -> +1.021`
- still far below B0: B0 shuffle gap is `+1.064 / +2.740`
- normal quality regressed: B0 `3.101 / 10.011`, F4b `3.482 / 11.118`, F5 `3.700 / 11.886`

Sample-level token sensitivity:

| Model | Ablation | Token Match Mean | Token Match P50 | Exact Same | ADE Delta Mean | ADE Delta P50 |
|---|---|---:|---:|---:|---:|---:|
| B0 | camera_shuffle | `0.276` | `0.051` | `5/68` | `+1.064` | `+0.410` |
| F3 | camera_shuffle | `0.682` | `0.707` | `25/68` | `-0.051` | `0.000` |
| F4b | camera_shuffle | `0.520` | `0.508` | `14/68` | `+0.053` | `0.000` |
| F5 | camera_shuffle | `0.472` | `0.500` | `16/68` | `+0.320` | `0.000` |
| B0 | black | `0.229` | `0.008` | `3/68` | `+1.195` | `+0.069` |
| F3 | black | `0.276` | `0.152` | `2/68` | `+0.876` | `+0.248` |
| F4b | black | `0.260` | `0.031` | `3/68` | `+0.777` | `+0.147` |
| F5 | black | `0.275` | `0.008` | `8/68` | `+0.862` | `0.000` |

## Conclusion

F5 sequence-level camera-contrast overfit is a partial but insufficient recovery.

Evidence:

- Direct token-target learning works: token CE `1.063 -> 0.401`, token acc `0.910 -> 0.937`.
- camera_shuffle gap improves over F4b: `+0.053 -> +0.324` ADE.
- But normal free-run regresses: `3.482 -> 3.700` ADE vs F4b, and `3.101 -> 3.700` vs B0.
- Teacher-forced parity is materially damaged: action_pre cosine `0.994 -> 0.852`, traj KL `0.015 -> 0.528`.
- The recovered shuffle gap is still only about 30% of B0's ADE gap.

Interpretation:

- FLEX can carry some visual/camera signal when forced hard enough.
- The current compressed FLEX path does not preserve B0 camera-order / camera-indexed geometry well enough for autoregressive free-run.
- More loss on the same compressed representation trades off normal behavior rather than fully restoring camera sensitivity.

Next action:

1. Run position-preserving FLEX diagnostic: keep downstream text/history/action token positions close to B0 while inserting FLEX tokens, to separate position shift from content compression.
2. If position-preserving passes but compressed mode fails, fix RoPE/position handling or use gradual compression.
3. If position-preserving also fails, change FLEX structure to preserve camera/time factorization explicitly, e.g. per-camera/per-frame summary tokens instead of a fully mixed scene token sequence.

One-line status: F5 generation-level overfit PARTIAL, deployable FLEX camera_shuffle sensitivity still FAIL.
