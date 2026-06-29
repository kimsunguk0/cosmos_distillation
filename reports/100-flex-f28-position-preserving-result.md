# FLEX F28 Position-Preserving Result

## Purpose

F28 tested whether the heldout256 FLEX failure was mainly caused by compressed-token position shift.

- Base: B0 no-FLEX `step_006250`
- Student: per-image FLEX K896 with camera/time embeddings
- Stage A: preserve-position teacher-parity warmup, 2000 steps, lr 2e-6
- Stage B: preserve-position CE-free pair margin, 3000 steps, lr 1e-6
- Eval: heldout256 normal free-run decode, `--preserve-flex-positions`

## Result

| Run | Normal ADE/FDE | TF action_pre | TF cot_end | TF traj KL | Pair delta cos | Unique traj ids | Max same-token run |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 no-FLEX | 3.091 / 10.173 | n/a | n/a | n/a | n/a | 20.6 | 2.5 |
| F27 per-image compressed | 3.642 / 11.935 | 0.9841 | 0.9854 | 0.0334 | 0.682 final | 14.5 | 1.6 |
| F28 per-image preserve-position | 5.556 / 16.978 | 0.9835 | 0.9848 | 0.0358 | 0.670 final | 6.1 | 37.5 |

F28 exceeded the normal ADE gate 3.35, so shuffle/black decodes were skipped.

## Judgment

F28 rejects "RoPE/absolute position shift is the main heldout256 failure" as a sufficient explanation. Position preservation did not recover free-run quality and made repetition much worse.

Current blocker: FLEX preserves teacher-forced hidden/logit behavior, but not deployable autoregressive free-run trajectory generation. The next experiment should anchor normal free-run rollout behavior directly before adding camera-shuffle sensitivity objectives.

## Artifacts

- Train summary: `outputs/reports/flex_f28b_perimage_preservepos_margin_from_f28a_heldout256_s3000_lr1e6_20260607_train_summary.json`
- Parity eval: `outputs/reports/flex_f28b_perimage_preservepos_margin_from_f28a_heldout256_s3000_lr1e6_20260607_eval_heldout256_summary.json`
- Normal decode: `outputs/reports/flex_f28b_perimage_preservepos_margin_from_f28a_heldout256_s3000_lr1e6_20260607_heldout256_decode_normal_summary.json`
- Chain log: `outputs/logs/flex_f28b_perimage_preservepos_margin_from_f28a_heldout256_s3000_lr1e6_20260607_chain.log`
