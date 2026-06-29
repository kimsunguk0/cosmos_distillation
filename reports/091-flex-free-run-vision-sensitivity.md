# FLEX Free-Run Vision Sensitivity Check

Date: 2026-06-06

## Scope

Compare no-FLEX baseline (`B0`) against patched FLEX-only overfit checkpoint (`F1`, K=896) on the same 68-sample visual category validation set.

This check uses free-run decode, not teacher-forced parity. The goal is to verify whether FLEX preserves deployable behavior and visual/camera sensitivity after the pre-norm hidden-state fix.

## Artifacts

- B0 checkpoint: `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`
- FLEX checkpoint: `outputs/checkpoints/flex_f1_parity_overfit16_prenormfix_s2000_k896_20260606/final`
- Dataset: `data/corpus/vis_4per_category_val.jsonl`
- B0 normal: `outputs/reports/b0_step006250_vis68_decode_normal_summary.json`
- B0 camera_shuffle: `outputs/reports/b0_step006250_vis68_decode_camera_shuffle_summary.json`
- B0 black: `outputs/reports/b0_step006250_vis68_decode_black_summary.json`
- FLEX normal: `outputs/reports/flex_f1_prenormfix_k896_vis68_decode_normal_summary.json`
- FLEX camera_shuffle: `outputs/reports/flex_f1_prenormfix_k896_vis68_decode_camera_shuffle_summary.json`
- FLEX black: `outputs/reports/flex_f1_prenormfix_k896_vis68_decode_black_summary.json`

## Results

| Model | Condition | ADE | FDE | Unique Traj IDs | Max Same-Token Run |
|---|---:|---:|---:|---:|---:|
| B0 no-FLEX | normal | 3.101 | 10.011 | 27.691 | 1.265 |
| B0 no-FLEX | camera_shuffle | 4.165 | 12.751 | 31.838 | 5.088 |
| B0 no-FLEX | black | 4.296 | 13.536 | 14.044 | 1.059 |
| FLEX F1 K896 | normal | 3.660 | 11.377 | 21.118 | 3.779 |
| FLEX F1 K896 | camera_shuffle | 3.775 | 11.606 | 21.147 | 4.824 |
| FLEX F1 K896 | black | 4.371 | 13.620 | 16.000 | 4.500 |

## Ablation Gaps

| Model | camera_shuffle ADE/FDE Gap | black ADE/FDE Gap |
|---|---:|---:|
| B0 no-FLEX | +1.064 / +2.740 | +1.195 / +3.525 |
| FLEX F1 K896 | +0.116 / +0.229 | +0.711 / +2.243 |

## Interpretation

- The pre-norm hidden-state bug is fixed: FLEX teacher-forced parity reached `action_pre_cosine=0.9969`, `traj_kl=0.0043`, and teacher-forced ADE parity with the no-FLEX teacher.
- Free-run normal quality is still behind B0: FLEX is worse by `+0.559 ADE` and `+1.366 FDE`.
- FLEX camera_shuffle sensitivity is too weak: B0 degrades by `+1.064 ADE`, while FLEX degrades by only `+0.116 ADE`.
- FLEX black sensitivity exists but is still not equivalent to B0, and token diversity/repetition worsens under FLEX normal.

## Current Verdict

FLEX-only K896 with the pre-norm fix passes the 16-sample parity/overfit gate, but it does not yet pass deployable free-run vision-sensitivity.

Root cause at this point is not the hidden-state bug anymore. The remaining issue is that the current FLEX-only training is too narrow: it preserves teacher behavior on a tiny parity set but does not yet preserve camera/order-dependent visual evidence in free-run.

## Next Gate

Run patched FLEX-only training beyond the 16-sample overfit:

1. 512 train / 512 val teacher-parity training with backbone frozen.
2. Evaluate normal, camera_shuffle, and black on the same 68-sample visual category set.
3. If 512 parity passes but camera_shuffle gap remains collapsed, add visual-sensitivity training or open projector/LoRA. If 512 parity fails, fix FLEX capacity/loss/position handling before LoRA.

One-line status: FLEX structure fixed Y, deployable FLEX vision-sensitivity pass N.
