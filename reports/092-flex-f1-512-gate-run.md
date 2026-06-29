# FLEX F1 512-Sample Gate Run

Date: 2026-06-06

## Purpose

Run the next FLEX-only gate after the successful 16-sample parity overfit.

The previous patched FLEX F1 checkpoint fixed the pre-norm hidden-state bug and passed 16-sample teacher-forced parity, but failed deployable free-run vision sensitivity. This run checks whether the failure is simply because FLEX was trained on only 16 visual-val samples.

## Run

- tmux session: `flex_f1_512`
- post-eval tmux watcher: `flex_f1_512_eval_wait`
- log: `outputs/logs/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606.log`
- post-eval log: `outputs/logs/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_posteval.log`
- output checkpoint dir: `outputs/checkpoints/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606`
- summary JSON: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_train_summary.json`
- teacher: `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`
- student init: `outputs/checkpoints/flex_f0_untrained_k896_camtime_from_step006250_20260605`
- corpus: `data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl`
- split: `train`
- samples: `512`
- steps: `4096`
- batch size: `1`
- seed: `42`
- trainable: FLEX scene encoder only
- LR: `2e-5`
- FLEX K: `896`

## Loss

This first 512 gate uses:

- `traj_kl_weight=1.0`
- `boundary_cos_weight=0.05`
- `boundary_norm_weight=0.10`
- `text_kl_weight=0.0`
- `format_kl_weight=0.0`

Reason: text/format full-vocab KL cache is too large at 512 samples. This gate tests the action-relevant FLEX path first: trajectory logits plus boundary hidden parity.

## Code Fix Applied Before Run

`scripts/105_train_flex_teacher_parity.py` teacher cache now:

- skips text/format logits when the corresponding loss weight is `0`
- stores cached teacher tensors on CPU instead of GPU

This prevents 512-sample cache from consuming excessive VRAM.

## Current Status

Training completed and final checkpoint was written.

- collated cache completed
- teacher cache completed
- training reached step `4096 / 4096`
- final checkpoint exists: `outputs/checkpoints/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606/final`
- train summary exists: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_train_summary.json`
- post-eval watcher started and is running parity eval first
- GPU memory stable with Stage2 process plus FLEX post-eval process

Observed training metrics:

| Step | Loss | Traj KL | Traj Top1 | Traj Top5 | Action-Pre Cos |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0666 | 0.0650 | 0.781 | 0.977 | 0.985 |
| 100 | 0.0522 | 0.0505 | 0.846 | 0.989 | 0.971 |
| 500 | 0.0352 | 0.0336 | 0.868 | 0.991 | 0.976 |
| 1000 | 0.0341 | 0.0326 | 0.873 | 0.992 | 0.977 |
| 2000 | 0.0288 | 0.0275 | 0.879 | 0.995 | 0.980 |
| 3000 | 0.0260 | 0.0249 | 0.884 | 0.994 | 0.982 |
| 3400 | 0.0245 | 0.0237 | 0.884 | 0.994 | 0.988 |
| 3900 | 0.0235 | 0.0226 | 0.886 | 0.995 | 0.987 |
| 4096 | 0.0254 | 0.0244 | 0.886 | 0.995 | 0.985 |

Current read: no collapse; FLEX-only 512-sample parity training passes in the
teacher-forced loss/logit/hidden sense. Final pass/fail still depends on held-out
parity and deployable free-run vision-sensitivity.

## Next Checks

`flex_f1_512_eval_wait` is running:

1. Run `scripts/104_eval_flex_teacher_parity.py` on 512 train samples.
2. Run the same parity eval on at least 512 held-out val samples.
3. Re-run 68-sample free-run decode for normal, camera_shuffle, and black.
4. Pass condition: parity improves materially and camera_shuffle/black gaps move toward B0 instead of remaining collapsed.

One-line status: FLEX F1 512 train pass, post-eval pass/fail pending.
