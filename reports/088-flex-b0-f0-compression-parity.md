# FLEX B0/F0 Compression-Parity Diagnostics

Date: 2026-06-05

## Purpose

Evaluate FLEX as a compression module, not as an immediate performance booster.

- B0: no-FLEX latest backbone checkpoint.
- F0: B0 weights plus untrained K896 camera/time FLEX.
- F2 pilot: previously trained `flex_scene_encoder + language_lora top4`.

The eval path is backbone autoregressive generation of 128 discrete trajectory tokens, then trajectory-tokenizer XYZ decode.

## Artifacts

- B0 free-run summary: `outputs/reports/b0_noflex_step006250_vis68_free_run_decode_summary.json`
- F0 checkpoint: `outputs/checkpoints/flex_f0_untrained_k896_camtime_from_step006250_20260605`
- F0 free-run summary: `outputs/reports/f0_untrained_k896_vis68_free_run_decode_summary.json`
- F0 B0-teacher parity summary: `outputs/reports/f0_untrained_k896_vis68_teacher_parity_summary.json`
- F2 pilot free-run summary: `outputs/reports/flex_k896_final_vis68_free_run_decode_summary.json`
- F0 creation script: `scripts/103_make_flex_untrained_checkpoint.py`
- Parity eval script: `scripts/104_eval_flex_teacher_parity.py`
- F1 FLEX-only config: `configs/train/stage_flex_f1_4cam4frame_k896_camtime_flex_only_parity.yaml`

## Free-Run Decode

All rows use the same 68 sample category-balanced val set.

| Model | FLEX | Trainable During Its Run | ADE | FDE | Bad Geometry | Avg Unique Traj IDs | <=2 Unique Collapse | Motion Match |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 no-FLEX | no | n/a | 3.159 | 10.159 | 0.088 | 25.779 | 18 / 68 | 0.632 |
| F0 untrained K896 | yes | none | 4.980 | 15.798 | 0.221 | 20.588 | 25 / 68 | 0.515 |
| F2 pilot K896 | yes | FLEX + LoRA top4 | 4.430 | 14.043 | 0.176 | 15.191 | 30 / 68 | 0.515 |

Interpretation:

- F0 compressed mode damages free-run geometry relative to B0.
- F2 training recovers some ADE/FDE versus F0, but worsens token diversity/collapse.
- Therefore the previous F2 pilot is not a valid FLEX success signal.

## F0 Teacher-Forced B0-Parity

Teacher: B0 no-FLEX, frozen.

Student: F0 untrained K896 FLEX, frozen except no training occurred.

| Metric | Mean |
|---|---:|
| Traj teacher-student KL | 0.112 |
| Traj top1 agreement | 0.736 |
| Teacher top1 in student top5 | 0.969 |
| Text teacher-student KL | 0.436 |
| Text top1 agreement | 0.779 |
| Format top1 agreement | 1.000 |
| Teacher TF argmax ADE | 0.092 |
| Student TF argmax ADE | 0.179 |
| Student - teacher TF argmax ADE | 0.086 |
| Teacher TF unique traj ids | 64.485 |
| Student TF unique traj ids | 66.103 |

Boundary hidden parity:

| Boundary | Cosine | Norm Ratio |
|---|---:|---:|
| cot_end | 0.910 | 0.106 |
| traj_start | 0.681 | 0.0128 |
| action_pre | 0.681 | 0.0128 |

Interpretation:

- Teacher-forced trajectory logits are much more preserved than free-run ADE suggests.
- F0 does not destroy trajectory-token information under teacher forcing.
- The major internal failure is hidden scale: `traj_start/action_pre` hidden norm is only about 1.3% of B0.
- This can explain why autoregressive free-run and AE/KV consumers would be fragile even when teacher-forced argmax looks acceptable.

## Current Diagnosis

FLEX compression is not obviously information-empty. The first failure mode is:

**compressed-position FLEX causes boundary hidden scale collapse and free-run autoregressive instability.**

Next step should be F1 FLEX-only training against B0 behavior parity, not another F2 LoRA run.

F1 success criteria:

- B0-vs-F1 traj KL lower than F0.
- `traj_start/action_pre` norm ratio recovers toward 1.0, or at least stops being near zero.
- Free-run ADE/FDE approaches B0.
- Unique/repetition metrics do not degrade.
- Shuffle/black sensitivity must be checked after parity improves.
