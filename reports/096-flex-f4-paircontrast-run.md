# FLEX F4 Pairwise Camera-Contrast Run

Date: 2026-06-06

## Why

F3 ablation-augmented parity passed teacher-forced metrics and restored some black-image sensitivity, but camera_shuffle still failed:

- B0 normal vs camera_shuffle token match mean: `0.276`, exact same: `5/68`
- F3 normal vs camera_shuffle token match mean: `0.682`, exact same: `25/68`
- F3 normal/shuffle ADE gap: `-0.051`, so shuffle was not worse.

Conclusion: independent parity on normal/shuffle examples is insufficient. The objective must preserve the pairwise B0 difference between the same sample under normal and camera_shuffle.

## Code Change

`scripts/105_train_flex_teacher_parity.py` now supports:

- `--paired-ablation camera_shuffle`
- batch rows forced to `[normal, camera_shuffle]` for the same sample
- pairwise action_pre delta cosine/norm matching
- pairwise trajectory log-prob delta matching

Existing behavior is unchanged unless `--paired-ablation` or pairwise weights are set.

Smoke:

- run: `outputs/checkpoints/flex_f4_paircontrast_smoke2_s3_20260606`
- summary: `outputs/reports/flex_f4_paircontrast_smoke2_s3_20260606_train_summary.json`
- result: passed 2 base samples x normal/shuffle x 3 steps.
- initial failure signature in smoke:
  - teacher action_pre delta norm: `419-535`
  - student action_pre delta norm: `101-126`
  - student/teacher delta norm ratio: `0.24-0.26`
  - pairwise loss was active and dominated the loss as intended.

## Run

- tmux train session: `flex_f4_paircontrast_vis68`
- tmux post-eval watcher: `flex_f4_paircontrast_eval_wait`
- run name: `flex_f4_paircontrast_vis68_from_f3_s6000_lr1e6_20260606`
- init: `outputs/checkpoints/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606/final`
- teacher: B0 no-FLEX checkpoint
- base samples: 68 vis hard samples
- training units: 68 paired batches, each `[normal, camera_shuffle]`
- steps: 6000
- batch size: 2
- LR: `1e-6`
- trainable: FLEX + multimodal projector + last 4 LoRA layers
- base parity losses: same as F3
- pairwise losses:
  - action_pre delta cosine: `0.05`
  - action_pre delta norm: `0.05`
  - trajectory log-prob delta: `0.10`

## Gate

After final checkpoint, watcher runs vis68:

1. teacher-forced parity
2. normal free-run
3. camera_shuffle free-run
4. black free-run

Pass condition:

- normal ADE should not regress badly from F3 (`3.423`).
- camera_shuffle gap must move toward B0 (`+1.064 / +2.740`), not stay near zero.
- sample-level normal-vs-shuffle token match should drop from F3 `0.682` toward B0 `0.276`.

One-line status: F4 pairwise camera-contrast launched, final pass/fail pending.

## F4 Early Stop

The first F4 run was stopped at step 200:

- run: `flex_f4_paircontrast_vis68_from_f3_s6000_lr1e6_20260606`
- issue: pairwise signal did not improve and training was slow due on-the-fly batch2 collation.
- step 100: pair action_pre delta norm ratio `0.123`, pairwise loss `0.153`
- step 200: pair action_pre delta norm ratio `0.125`, pairwise loss `0.154`

This is not worth waiting for 6000 steps.

## F4b

Code update:

- added paired batch collated-cache support for `--paired-ablation`.

Smoke:

- run: `outputs/checkpoints/flex_f4b_paircontrast_cache_smoke2_s3_20260606`
- summary: `outputs/reports/flex_f4b_paircontrast_cache_smoke2_s3_20260606_train_summary.json`
- result: passed paired collated-cache + teacher cache + 3 train steps.

F4b run:

- run name: `flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606`
- changes vs F4:
  - `--cache-collated-batches`
  - LR `1e-6 -> 5e-6`
  - grad clip `1.0 -> 5.0`
  - steps `6000 -> 3000` first gate

## F4b Final

F4b training completed:

- final checkpoint: `outputs/checkpoints/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606/final`
- train summary: `outputs/reports/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606_train_summary.json`
- parity summary: `outputs/reports/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606_eval_vis68_summary.json`
- decode summaries:
  - normal: `outputs/reports/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606_vis68_decode_normal_summary.json`
  - camera_shuffle: `outputs/reports/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606_vis68_decode_camera_shuffle_summary.json`
  - black: `outputs/reports/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606_vis68_decode_black_summary.json`

Final train metrics at step 3000:

- loss: `0.0460`
- action_pre cosine: `0.9946`
- traj KL: `0.0146`
- traj top1 agreement: `0.8924`
- teacher top1 in student top5: `0.9972`
- pair action_pre delta cosine: `0.8095`
- pair action_pre delta norm ratio: `0.9788`
- pairwise loss: `0.0195`
- student/teacher traj logprob delta L1: `0.1620 / 0.3349`

Teacher-forced parity on vis68:

- action_pre cosine mean: `0.9943`
- cot_end cosine mean: `0.9884`
- traj KL mean: `0.0150`
- traj top1 agreement mean: `0.8953`
- teacher top1 in student top5 mean: `0.9980`
- student minus teacher TF argmax ADE mean: `0.0006m`

Free-run results:

| Model | Normal ADE/FDE | Shuffle ADE/FDE | Black ADE/FDE | Shuffle Gap | Black Gap | Unique Normal |
|---|---:|---:|---:|---:|---:|---:|
| B0 no-FLEX | `3.101 / 10.011` | `4.165 / 12.751` | `4.296 / 13.536` | `+1.064 / +2.740` | `+1.195 / +3.525` | `27.69` |
| F3 ablation parity | `3.423 / 11.181` | `3.373 / 10.661` | `4.299 / 13.644` | `-0.051 / -0.520` | `+0.876 / +2.463` | `20.74` |
| F4b pair contrast | `3.482 / 11.118` | `3.535 / 11.246` | `4.259 / 13.483` | `+0.053 / +0.128` | `+0.777 / +2.365` | `22.71` |

Sample-level normal-vs-ablation token sensitivity:

| Model | Ablation | Token Match Mean | Token Match P50 | Exact Same | ADE Delta Mean | ADE Delta P50 |
|---|---|---:|---:|---:|---:|---:|
| B0 | camera_shuffle | `0.276` | `0.051` | `5/68` | `+1.064` | `+0.410` |
| F3 | camera_shuffle | `0.682` | `0.707` | `25/68` | `-0.051` | `0.000` |
| F4b | camera_shuffle | `0.520` | `0.508` | `14/68` | `+0.053` | `0.000` |
| B0 | black | `0.229` | `0.008` | `3/68` | `+1.195` | `+0.069` |
| F3 | black | `0.276` | `0.152` | `2/68` | `+0.876` | `+0.248` |
| F4b | black | `0.260` | `0.031` | `3/68` | `+0.777` | `+0.147` |

## Conclusion

F4b pairwise camera-contrast overfit succeeded internally but failed the deployable free-run gate.

Evidence:

- Pairwise hidden delta magnitude recovered: student/teacher action_pre delta norm ratio `0.24 -> 0.98`.
- Teacher-forced parity remained strong: traj KL `0.015`, action_pre cosine `0.994`, TF ADE delta `0.0006m`.
- Free-run camera_shuffle gap remained near zero: F4b `+0.053 / +0.128`, while B0 is `+1.064 / +2.740`.
- Black-image gap remained present: F4b `+0.777 / +2.365`.

Interpretation:

- FLEX is not ignoring vision entirely.
- FLEX still loses camera-order / camera-indexed geometry in autoregressive free-run.
- Hidden/logprob pairwise parity is insufficient because small teacher-forced differences do not reliably survive token decoding.

Next action:

- Move from hidden/logprob parity to generation-level camera-contrast training.
- Candidate objective: same sample normal vs camera_shuffle sequence-level KD / preference loss, where the student is explicitly trained to reproduce B0's free-run token divergence under camera_shuffle.
- Also test a position-preserving diagnostic to separate camera-order information loss from compressed-token position shift.

One-line status: F4b internal pairwise overfit PASS, deployable camera_shuffle sensitivity FAIL.
