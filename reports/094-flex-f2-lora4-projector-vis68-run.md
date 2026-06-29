# FLEX F2 LoRA4 Projector Vis68 Run

Date: 2026-06-06

## Why

F1-512 established that FLEX-only K896 can match the no-FLEX teacher in teacher-forced parity on train512, val512, and vis68, but still fails deployable free-run and camera sensitivity.

F1-512 key result:

- vis68 teacher-forced parity: pass
- normal free-run: `3.984 / 12.357`
- camera_shuffle free-run: `3.932 / 12.201`
- shuffle gap: `-0.052 / -0.156`

This means the remaining issue is not hidden scale or teacher-forced trajectory parity. The next check is whether a small adaptation surface can make the compressed visual prefix usable in free-run.

## Run

- tmux train session: `flex_f2_lora4_vis68`
- tmux post-eval watcher: `flex_f2_vis68_eval_wait`
- log: `outputs/logs/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606.log`
- post-eval log: `outputs/logs/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606_posteval.log`
- output checkpoint: `outputs/checkpoints/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606`
- summary: `outputs/reports/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606_train_summary.json`

Config:

- init: `outputs/checkpoints/flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606/final`
- teacher: `outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`
- data: `data/corpus/vis_4per_category_val.jsonl`
- samples: 68
- steps: 2000
- batch size: 1
- LR: `1e-6`
- seed: 42
- trainable: FLEX + multimodal projector + last 4 LoRA layers
- losses: `traj_kl=1.0`, `text_kl=0.2`, `format_kl=0.05`, `boundary_cos=0.05`, `boundary_norm=0.10`

Trainable params from launch:

| Group | Params |
|---|---:|
| flex_scene_encoder | 31,378,432 |
| language_lora | 9,961,472 |
| multimodal_projector | 25,174,016 |
| total trainable | 66,513,920 |

## Final Training Metrics

| Step | Loss | Traj KL | Text KL | Traj Top1 | Top5 | Action-Pre Cos | Action-Pre Norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0900 | 0.0221 | 0.335 | 0.914 | 1.000 | 0.989 | 0.920 |
| 100 | 0.0873 | 0.0297 | 0.284 | 0.866 | 0.993 | 0.992 | 1.010 |
| 500 | 0.0666 | 0.0285 | 0.185 | 0.866 | 0.994 | 0.991 | 1.037 |
| 1000 | 0.0546 | 0.0242 | 0.148 | 0.871 | 0.995 | 0.992 | 1.045 |
| 1500 | 0.0454 | 0.0223 | 0.111 | 0.877 | 0.995 | 0.991 | 1.034 |
| 2000 | 0.0338 | 0.0188 | 0.071 | 0.885 | 0.996 | 0.992 | 1.026 |

## Teacher-Forced Eval

Artifact: `outputs/reports/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606_eval_vis68_summary.json`

| Metric | F1-512 | F2 |
|---|---:|---:|
| Traj KL | 0.0313 | 0.0189 |
| Text KL | 0.3530 | 0.0774 |
| Traj Top1 | 0.861 | 0.882 |
| Top5 | 0.993 | 0.996 |
| Action-Pre Cos | 0.992 | 0.992 |
| Action-Pre Norm | 0.999 | 1.026 |
| TF ADE Delta | 0.021m | 0.0046m |

Interpretation: opening last-4 LoRA plus multimodal projector improves teacher-forced parity, especially text KL and TF trajectory geometry.

## Free-Run Eval

Artifacts:

- normal: `outputs/reports/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606_vis68_decode_normal_summary.json`
- camera_shuffle: `outputs/reports/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606_vis68_decode_camera_shuffle_summary.json`
- black: `outputs/reports/flex_f2_lora4_projector_vis68_from_f1_512_s2000_lr1e6_20260606_vis68_decode_black_summary.json`

| Model | Condition | ADE | FDE | Unique IDs | Max Same-Token Run |
|---|---|---:|---:|---:|---:|
| B0 no-FLEX | normal | 3.101 | 10.011 | 27.691 | 1.265 |
| B0 no-FLEX | camera_shuffle | 4.165 | 12.751 | 31.838 | 5.088 |
| B0 no-FLEX | black | 4.296 | 13.536 | 14.044 | 1.059 |
| F1-512 FLEX-only | normal | 3.984 | 12.357 | 19.353 | 1.059 |
| F1-512 FLEX-only | camera_shuffle | 3.932 | 12.201 | 17.485 | 1.029 |
| F1-512 FLEX-only | black | 4.180 | 13.036 | 19.985 | 1.191 |
| F2 LoRA4/projector | normal | 3.530 | 11.175 | 18.618 | 1.176 |
| F2 LoRA4/projector | camera_shuffle | 3.565 | 11.292 | 22.544 | 1.176 |
| F2 LoRA4/projector | black | 3.975 | 12.178 | 19.779 | 3.103 |

## Vision-Sensitivity Gaps

| Model | camera_shuffle ADE/FDE Gap | black ADE/FDE Gap |
|---|---:|---:|
| B0 no-FLEX | +1.064 / +2.740 | +1.195 / +3.525 |
| F1-512 FLEX-only | -0.052 / -0.156 | +0.196 / +0.679 |
| F2 LoRA4/projector | +0.035 / +0.117 | +0.445 / +1.003 |

## Verdict

F2 LoRA4/projector partially helps normal free-run quality but does not restore camera-order sensitivity.

- Normal ADE improves from F1-512 `3.984` to F2 `3.530`.
- F2 is still worse than B0 `3.101`.
- camera_shuffle gap remains near zero: `+0.035 ADE`, far from B0 `+1.064`.
- black gap improves over F1-512 but is still weak: `+0.445 ADE`, far from B0 `+1.195`.

The remaining failure is no longer teacher-forced parity or hidden-state scale. It is deployable generation behavior and visual sensitivity. More local parity loss is unlikely to fix it by itself.

## Next Gate

The next experiment should explicitly train the compressed model to preserve no-FLEX behavior under visual perturbations or generated-prefix drift.

Recommended options:

1. Generation-level KD on the first trajectory tokens: compare no-FLEX teacher free-run distribution/sequence against FLEX student, not only teacher-forced labels.
2. Visual-ablation sensitivity loss: for normal vs camera_shuffle/black, preserve the teacher's degradation gap or hidden/logit delta.
3. Open more adaptation only after adding one of the above losses; opening LoRA alone recovered normal ADE but not camera semantics.

One-line status: F2 improves normal ADE but fails camera sensitivity; next fix must target free-run/ablation behavior, not just teacher-forced parity.
