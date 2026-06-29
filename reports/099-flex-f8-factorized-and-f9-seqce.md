# FLEX F8/F9 Factorized Diagnostics

Date: 2026-06-06

## Question

Can FLEX preserve the no-FLEX baseline behavior after replacing the 16-image visual stream with compressed scene tokens?

The earlier global FLEX variants could match teacher-forced parity partially, but failed free-run normal quality and lost camera-shuffle sensitivity. F8 tests whether per-image/factorized compression fixes the structural issue.

## F8: factorized per-image FLEX

Config:

- Init: `outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606`
- Compression: `compression_mode=per_image`, 16 images, 56 tokens/image, total K=896
- Train set: first 16 val samples expanded to normal/camera_shuffle/black = 48 views
- Trainable: `flex_scene_encoder` only, 31.38M params
- Steps/LR: 1500 steps, LR `2e-5`
- Loss: teacher-forced traj/text/format KL + boundary cosine/norm

Teacher-forced parity converged:

| metric | F8 final |
|---|---:|
| train loss | 0.0175 |
| train traj KL | 0.0117 |
| train action_pre cosine | 0.9962 |
| eval action_pre cosine | 0.9948 |
| eval cot_end cosine | 0.9853 |
| eval traj KL | 0.0184 |
| eval traj top1 agreement | 0.8911 |
| eval student TF argmax ADE/FDE | 0.104 / 0.207 |

Free-run result on the same first 16 samples:

| model | normal ADE/FDE | shuffle ADE/FDE | black ADE/FDE | shuffle gap | black gap |
|---|---:|---:|---:|---:|---:|
| B0 no-FLEX | 2.688 / 8.092 | 5.198 / 15.796 | 4.618 / 14.538 | +2.509 / +7.704 | +1.929 / +6.446 |
| F3 global ablation | 3.943 / 12.262 | 4.023 / 11.866 | 5.003 / 16.092 | +0.080 / -0.396 | +1.060 / +3.830 |
| F4b global pair | 3.802 / 11.902 | 3.444 / 10.103 | 4.496 / 14.209 | -0.358 / -1.799 | +0.695 / +2.307 |
| F5 global seqCE | 4.141 / 11.786 | 4.297 / 12.650 | 4.688 / 14.495 | +0.156 / +0.864 | +0.547 / +2.708 |
| F8 factorized | 3.997 / 11.346 | 3.675 / 10.851 | 4.557 / 14.148 | -0.321 / -0.495 | +0.561 / +2.802 |

Conclusion:

- Per-image factorization fixes neither normal free-run quality nor camera-shuffle sensitivity.
- F8 proves FLEX can match teacher-forced boundary/logit parity, but autoregressive free-run still drifts.
- The remaining blocker is not only global image mixing. It is teacher-forced parity vs free-run exposure mismatch under compressed visual tokens.

## F9: factorized FLEX + B0 free-run token CE

Launched run:

- Script: `scripts/tmp_run_flex_f9_factorized_seqce_ablation16.sh`
- Run: `flex_f9_factorized_seqce_ablation16_s3000_lr2e5_ce5_20260606`
- Init: F8 factorized untrained checkpoint
- Train set: same 16 samples expanded to normal/camera_shuffle/black = 48 views
- Targets: B0 generated trajectory tokens from normal/camera_shuffle/black summaries
- Trainable: `flex_scene_encoder` only
- Steps/LR: 3000 steps, LR `2e-5`
- Loss: free-run token CE weight `5.0`, text/format KL, small boundary loss, no traj KL

Early training:

| step | free_run_token_acc | free_run_token_ce | action_pre_cos | action_pre_norm_ratio |
|---:|---:|---:|---:|---:|
| 1 | 0.781 | 1.074 | 0.935 | 0.910 |
| 50 | 0.889 | 0.625 | 0.969 | 0.940 |
| 100 | 0.888 | 0.547 | 0.964 | 0.803 |
| 200 | 0.888 | 0.501 | 0.962 | 0.778 |
| 300 | 0.906 | 0.460 | 0.946 | 0.705 |

Final training:

| step | free_run_token_acc | free_run_token_ce | action_pre_cos | action_pre_norm_ratio |
|---:|---:|---:|---:|---:|
| 3000 | 0.973 | 0.310 | 0.624 | 0.347 |

Post-train parity:

| metric | F9 final |
|---|---:|
| action_pre cosine | 0.590 |
| action_pre norm ratio | 0.298 |
| cot_end cosine | 0.838 |
| traj KL | 0.716 |
| traj top1 agreement | 0.671 |
| student TF argmax ADE/FDE | 0.121 / 0.248 |

Normal free-run:

| model | normal ADE/FDE |
|---|---:|
| B0 no-FLEX, first 16 | 2.688 / 8.092 |
| F8 factorized, first 16 | 3.997 / 11.346 |
| F9 factorized seqCE, first 16 | 6.379 / 18.942 |

Interpretation so far:

- The sequence CE path is active and improving token accuracy.
- It also pushes the action-prefix hidden away from the B0 teacher, especially in norm.
- The normal free-run result is worse than both B0 and F8. FLEX-only sequence CE is not a fix.

## F10: factorized FLEX + projector + last-4 LoRA

Launched run:

- Script: `scripts/tmp_run_flex_f10_factorized_seqce_lora4_ablation16.sh`
- Run: `flex_f10_factorized_seqce_lora4_ablation16_s3000_lr5e6_ce5_20260606`
- Init: F8 factorized untrained checkpoint
- Train set/targets: same as F9
- Trainable: `flex_scene_encoder` + `multimodal_projector` + last-4 language LoRA
- Trainable params: 66.51M
- Steps/LR: 3000 steps, LR `5e-6`

Purpose:

- If F10 overfits free-run without destroying normal quality, FLEX needs limited backbone adaptation.
- If F10 also fails, the current FLEX placement/objective is structurally wrong, not just FLEX-only capacity.

## Current verdict

FLEX is not deployable yet. F8 failed free-run and camera sensitivity despite strong teacher-forced parity. F9 showed that FLEX-only sequence CE can fit many B0 tokens but destroys hidden/logit parity and worsens normal free-run. F10 is the active test for whether limited backbone adaptation fixes that.
