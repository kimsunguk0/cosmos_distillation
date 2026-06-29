# FLEX F1 512 Post-Eval Results

Date: 2026-06-06

## Scope

Evaluate patched FLEX-only K896 after 512-sample teacher-parity training.

Run:

- checkpoint: `outputs/checkpoints/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606/final`
- train summary: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_train_summary.json`
- train parity: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_eval_train512_summary.json`
- val parity: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_eval_val512_summary.json`
- vis68 parity: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_eval_vis68_summary.json`
- vis68 normal: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_vis68_decode_normal_summary.json`
- vis68 camera_shuffle: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_vis68_decode_camera_shuffle_summary.json`
- vis68 black: `outputs/reports/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606_vis68_decode_black_summary.json`

Training used FLEX scene encoder only, K896, 512 train samples, 4096 steps, LR `2e-5`, `traj_kl=1.0`, `boundary_cos=0.05`, `boundary_norm=0.10`, text/format losses disabled.

## Teacher-Forced Parity

| Split | Traj KL | Traj Top1 | Top5 | Action-Pre Cos | Action-Pre Norm | TF ADE Delta |
|---|---:|---:|---:|---:|---:|---:|
| train512 | 0.0224 | 0.886 | 0.995 | 0.986 | 1.010 | 0.011m |
| val512 | 0.0319 | 0.867 | 0.994 | 0.986 | 1.011 | 0.012m |
| vis68 | 0.0313 | 0.861 | 0.993 | 0.992 | 0.999 | 0.021m |

Interpretation: FLEX-only 512 training generalizes in teacher-forced parity, including the hard vis68 set. The hidden-scale bug is not the remaining blocker.

## Free-Run Decode

| Model | Condition | ADE | FDE | Unique IDs | Max Same-Token Run |
|---|---|---:|---:|---:|---:|
| B0 no-FLEX | normal | 3.101 | 10.011 | 27.691 | 1.265 |
| B0 no-FLEX | camera_shuffle | 4.165 | 12.751 | 31.838 | 5.088 |
| B0 no-FLEX | black | 4.296 | 13.536 | 14.044 | 1.059 |
| F1 16-sample | normal | 3.660 | 11.377 | 21.118 | 3.779 |
| F1 16-sample | camera_shuffle | 3.775 | 11.606 | 21.147 | 4.824 |
| F1 16-sample | black | 4.371 | 13.620 | 16.000 | 4.500 |
| F1 512-sample | normal | 3.984 | 12.357 | 19.353 | 1.059 |
| F1 512-sample | camera_shuffle | 3.932 | 12.201 | 17.485 | 1.029 |
| F1 512-sample | black | 4.180 | 13.036 | 19.985 | 1.191 |

## Vision-Sensitivity Gaps

Gaps are relative to each model's normal condition.

| Model | camera_shuffle ADE/FDE Gap | black ADE/FDE Gap |
|---|---:|---:|
| B0 no-FLEX | +1.064 / +2.740 | +1.195 / +3.525 |
| F1 16-sample | +0.116 / +0.229 | +0.711 / +2.243 |
| F1 512-sample | -0.052 / -0.156 | +0.196 / +0.679 |

## Verdict

FLEX-only K896 is not sufficient.

The failure is now specific:

- teacher-forced trajectory logits and boundary hidden parity pass on train, val, and vis68
- deployable free-run quality is worse than B0
- camera_shuffle sensitivity is gone; shuffled cameras are not worse than normal
- black sensitivity is much weaker than B0

One-line status: FLEX F1 512 teacher-forced parity pass Y, deployable free-run vision-sensitivity pass N.

## Next Action

Run a fresh patched-code F2 gate. The old F2 smoke/overfit artifacts were produced before the pre-norm hidden fix and should not be used for the current decision.

Recommended F2 gate:

- init: `outputs/checkpoints/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606/final`
- teacher: B0 no-FLEX checkpoint
- data: `data/corpus/vis_4per_category_val.jsonl`
- trainable: FLEX + multimodal projector + last 4 LoRA layers
- samples: 68 vis hard samples
- steps: 1000-2000
- LR: `1e-6` to `2e-6` for all opened params first
- losses: restore text/format KL because 68 samples is small enough; keep trajectory KL and boundary losses
- gate: vis68 teacher parity plus normal/camera_shuffle/black free-run

If F2 overfits vis68 teacher-forced parity but still fails free-run camera sensitivity, then the next fix is not more parity loss; it needs generation-level KD or explicit visual-ablation sensitivity loss.
