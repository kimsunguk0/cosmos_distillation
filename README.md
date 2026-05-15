# Cosmos Distillation

Current focus: Alpamayo 1.5 no-nav teacher-pair distillation into a Cosmos Reason2 2B student backbone.

This README is a compact status board. The longer reasoning/history is in [NO_NAV_DISTILL_ISSUE_DECISION_LOG.md](./NO_NAV_DISTILL_ISSUE_DECISION_LOG.md).

## Current Status

As of 2026-05-15, the active branch is the official Alpamayo 4V input-contract run.

The student input format now matches the public Alpamayo 4-camera contract:

- camera labels are explicit text tokens
- order is front-left, front, front-right, front-telephoto
- 4 frames per camera, 16 image placeholders total
- ego history is fused as Alpamayo-style trajectory-history tokens
- prompt text is the official Alpamayo prompt
- assistant prefix starts at `<|cot_start|>`

Current active run:

```text
run_id:
  no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509

config:
  configs/train/stage_bp3_no_nav_camera_labeled_gc_decode_eval.yaml

init checkpoint:
  outputs/checkpoints/no_nav_camera_labeled_official_20k/

active output:
  outputs/checkpoints/no_nav_camera_labeled_official_200k/no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509

active log:
  logs/no_nav_distill/no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509.train.log
```

Latest observed training progress:

```text
step:        about 6.7k / 12.5k
epoch:       about 0.53 / 1.00
train set:   200,000 samples
val set:     2,048 samples
decode eval: 64 val samples per eval
batch:       16
```

## Current Performance

All ADE/FDE below are student free-run trajectory vs teacher discrete decoded trajectory, using the run's decode-eval setting.

### Official 20k Init

This is the completed official-input warmup run used as the current 200k init.

| Step | Val Loss | CoT Acc | Traj Acc | ADE m | FDE m | Bad Rate | Unique IDs | Motion Agree | Token Count OK |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 312 | 1.5367 | 0.8800 | 0.4774 | 4.5438 | 12.3202 | 18.75% | 18.28 | 59.38% | 100% |
| 624 | 1.5293 | 0.8829 | 0.4787 | 4.1751 | 11.7736 | 12.50% | 15.88 | 50.00% | 100% |
| 936 | 1.5249 | 0.8835 | 0.4790 | 4.2226 | 11.9612 | 15.63% | 18.05 | 53.13% | 100% |
| 1248 | 1.5230 | 0.8830 | 0.4797 | 4.0592 | 11.7664 | 12.50% | 12.78 | 45.31% | 100% |

Best decode score:

```text
step 1248: free_run_geometry_score = -7.6258
```

### Official 200k Continuation

This is the currently running main experiment.

| Step | Val Loss | Decode Score | ADE m | FDE m | Bad Rate | Unique IDs | Motion Agree |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3125 | 1.5041 | -6.6298 | 3.4699 | 10.7648 | 9.38% | 13.13 | 40.63% |
| 6250 | 1.5006 | -6.0505 | 3.2152 | 10.0912 | 6.25% | 13.72 | 50.00% |

Current interpretation:

- 200k continuation is clearly better than the 20k init on free-run ADE/FDE.
- Bad-geometry rate improved from 12.5% at the 20k best point to 6.25% at step 6250.
- Teacher-forced train trajectory accuracy is still only around 0.45-0.55, so exact token prediction is not solved.
- Free-run geometry is improving, which is the more important readiness signal.

## Latest Training Metrics

Recent training window around step 6.6k:

```text
CoT token acc:    roughly 0.80 - 0.95
Traj token acc:   roughly 0.42 - 0.56
Total loss:       roughly 1.3 - 1.7
Output format:    mostly stable
Scheduled sample: off
```

Important: these are teacher-forced training metrics. They are useful for diagnosis, but model selection should use free-run geometry and malformed-output checks.

## Input Contract Check

Run this before trusting a new training/eval result:

```bash
.venv/bin/python scripts/81_check_camera_prompt_contract.py \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_300chunks.jsonl \
  --sample-index 0
```

Expected:

```text
image placeholders: 16
camera labels:      Front left, Front, Front right, Front telephoto
frame order:        0,1,2,3 per camera
cam3 mapping:       original camera index 6 / front telephoto
assistant prefix:   <|cot_start|>
chat-template:      teacher helper and student collator match
```

## What To Watch Next

The next useful checkpoint/eval should answer:

1. Does ADE/FDE keep dropping after step 6250?
2. Does bad-geometry rate stay below 6.25% or improve?
3. Does token diversity stay healthy without long repeated-token runs?
4. Does `<traj_future_start>` appear reliably in free-run?
5. Are curve/stop/turn buckets improving, not only straight driving?
6. Does normal-image eval beat black/shuffled-image ablations?
7. Is the backbone stable enough to start a student-compatible action/FM head smoke?

## Key Files

```text
README:
  README.md

full issue and decision log:
  NO_NAV_DISTILL_ISSUE_DECISION_LOG.md

active training config:
  configs/train/stage_bp3_no_nav_camera_labeled_gc_decode_eval.yaml

prompt contract test:
  scripts/81_check_camera_prompt_contract.py

training entrypoint:
  scripts/09_train_distill.py

decode / overlay eval:
  scripts/25_decode_checkpoint_overlays.py

checkpoint export:
  scripts/27_export_student_weights_for_trt.py
```

## Notes

- The current README intentionally avoids old experiment genealogy.
- Older experiments remain useful for debugging, but current decisions should be based on the official-input run.
- Generated logs, checkpoints, reports, and exports are git-ignored.
