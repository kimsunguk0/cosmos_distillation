# FLEX F3 Ablation-Augmented Run

Date: 2026-06-06

## Why

F1 and F2 both passed teacher-forced parity but failed deployable camera sensitivity.

Key prior result:

- F1-512 normal/shuffle gap: `-0.052 / -0.156`
- F2 LoRA4/projector normal/shuffle gap: `+0.035 / +0.117`
- B0 no-FLEX normal/shuffle gap: `+1.064 / +2.740`

F2 recovered part of normal free-run quality but not camera-order semantics. Therefore this run trains not only on normal images but also on B0 teacher targets under `camera_shuffle` and `black` image conditions.

## Code Change

- `src/training/collator.py`: added `apply_image_ablation()` and per-sample `_image_ablation` support.
- `scripts/105_train_flex_teacher_parity.py`: added `--image-ablations` and expands rows into `sample_id::ablation` training units.
- smoke passed with 2 base samples x 3 ablations x 3 steps:
  - summary: `outputs/reports/flex_f3_ablation_aug_smoke2x3_s3_20260606_train_summary.json`

## Run

- tmux train session: `flex_f3_ablation_aug_vis68`
- tmux post-eval watcher: `flex_f3_ablation_aug_eval_wait`
- log: `outputs/logs/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606.log`
- post-eval log: `outputs/logs/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606_posteval.log`
- output checkpoint: `outputs/checkpoints/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606`
- summary: `outputs/reports/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606_train_summary.json`

Config:

- init: `outputs/checkpoints/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606/final`
- teacher: B0 no-FLEX checkpoint
- base samples: 68 vis hard samples
- image ablations: `normal,camera_shuffle,black`
- expanded training units: 204
- steps: 3000
- batch size: 1
- LR: `1e-6`
- trainable: FLEX + multimodal projector + last 4 LoRA layers
- losses: `traj_kl=1.0`, `text_kl=0.2`, `format_kl=0.05`, `boundary_cos=0.05`, `boundary_norm=0.10`

## Final Train

- final checkpoint: `outputs/checkpoints/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606/final`
- step 3000:
  - loss: `0.0209`
  - traj KL: `0.0122`
  - traj top1 agreement: `0.9095`
  - teacher top1 in student top5: `0.9980`
  - action_pre cosine: `0.9955`
  - cot_end cosine: `0.9908`

## Teacher-Forced Parity

Vis68, same 68 samples.

| Model | traj KL | top1 | top5 | action_pre cos | TF ADE delta |
|---|---:|---:|---:|---:|---:|
| F1-512 | 0.0313 | 0.8609 | 0.9932 | 0.9916 | 0.0211 m |
| F2 LoRA4/projector | 0.0189 | 0.8824 | 0.9963 | 0.9923 | 0.0046 m |
| F3 ablation-aug | 0.0171 | 0.8851 | 0.9969 | 0.9924 | 0.0011 m |

Teacher-forced parity improved slightly over F2. This means the FLEX/LoRA/projector path can imitate B0 logits/hidden states on these inputs.

## Free-Run Results

Vis68, `samples-per-row=1`, normal/shuffle/black evaluated on the same rows.

| Model | Condition | ADE | FDE | unique | max same-token run |
|---|---|---:|---:|---:|---:|
| B0 no-FLEX | normal | 3.101 | 10.011 | 27.691 | 1.265 |
| B0 no-FLEX | camera_shuffle | 4.165 | 12.751 | 31.838 | 5.088 |
| B0 no-FLEX | black | 4.296 | 13.536 | 14.044 | 1.059 |
| F1-512 | normal | 3.984 | 12.357 | 19.353 | 1.059 |
| F1-512 | camera_shuffle | 3.932 | 12.201 | 17.485 | 1.029 |
| F1-512 | black | 4.180 | 13.036 | 19.985 | 1.191 |
| F2 LoRA4/projector | normal | 3.530 | 11.175 | 18.618 | 1.176 |
| F2 LoRA4/projector | camera_shuffle | 3.565 | 11.292 | 22.544 | 1.176 |
| F2 LoRA4/projector | black | 3.975 | 12.178 | 19.779 | 3.103 |
| F3 ablation-aug | normal | 3.423 | 11.181 | 20.735 | 1.176 |
| F3 ablation-aug | camera_shuffle | 3.373 | 10.661 | 24.559 | 1.206 |
| F3 ablation-aug | black | 4.299 | 13.644 | 16.338 | 1.118 |

Gaps relative to normal:

| Model | shuffle ADE/FDE gap | black ADE/FDE gap |
|---|---:|---:|
| B0 no-FLEX | +1.064 / +2.740 | +1.195 / +3.525 |
| F1-512 | -0.052 / -0.156 | +0.196 / +0.679 |
| F2 LoRA4/projector | +0.035 / +0.117 | +0.445 / +1.003 |
| F3 ablation-aug | -0.051 / -0.520 | +0.876 / +2.463 |

## Interpretation

F3 did not solve camera-order sensitivity. It improved teacher-forced parity and slightly improved normal free-run ADE versus F2 (`3.530 -> 3.423`), but `camera_shuffle` still does not degrade; it is slightly better than normal. Therefore the deployable generation path is still not using camera-order visual information in the B0-like way.

The black-image gap did increase (`+0.876 / +2.463`), so the model has some visual-content dependence. The missing piece is specifically camera-ordered visual semantics, not simply "vision ignored."

Sample-level token comparison confirms this:

| Model | Comparison | token match mean | exact same traj | ADE delta mean | ADE delta p50 |
|---|---|---:|---:|---:|---:|
| B0 no-FLEX | normal vs camera_shuffle | 0.276 | 5/68 | +1.064 | +0.410 |
| B0 no-FLEX | normal vs black | 0.229 | 3/68 | +1.195 | +0.069 |
| F1-512 | normal vs camera_shuffle | 0.728 | 27/68 | -0.052 | +0.000 |
| F1-512 | normal vs black | 0.504 | 12/68 | +0.196 | +0.000 |
| F2 LoRA4/projector | normal vs camera_shuffle | 0.724 | 30/68 | +0.035 | +0.000 |
| F2 LoRA4/projector | normal vs black | 0.423 | 8/68 | +0.445 | +0.012 |
| F3 ablation-aug | normal vs camera_shuffle | 0.682 | 25/68 | -0.051 | +0.000 |
| F3 ablation-aug | normal vs black | 0.276 | 2/68 | +0.876 | +0.248 |

This is the clearest failure signature: F3 has B0-like black-image sensitivity, but camera_shuffle still produces highly overlapping trajectories. It is not enough to include shuffled images as independent teacher-parity examples; the objective does not explicitly force normal and camera_shuffle outputs apart.

Current diagnosis:

- FLEX structure and trainability: OK.
- Teacher-forced hidden/logit parity: OK.
- Normal-only LoRA/projector adaptation: insufficient.
- Ablation-augmented parity: improves black sensitivity, fails camera_shuffle sensitivity.
- Remaining failure mode: free-run generation is insensitive to camera ordering even when teacher-forced parity is strong.

Next practical direction:

1. Add generation-level KD or sequence-level free-run loss, because token/hidden parity alone is not preserving autoregressive camera sensitivity.
2. Oversample camera-order hard pairs: same row normal vs camera_shuffle with explicit contrastive loss on `action_pre` or trajectory logits.
3. Keep B0 as the behavior teacher and measure success by free-run normal/shuffle/black gaps, not train loss.

One-line status: F3 = partial. Parity passed, black sensitivity improved, camera_shuffle sensitivity still failed.
