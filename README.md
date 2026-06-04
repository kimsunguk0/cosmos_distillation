# Cosmos Distillation

This repository is the local workspace for distilling Alpamayo 1.5 no-navigation
driving behavior into a Cosmos Reason2 2B student.  The current active work is
the student-compatible action expert: a 28-layer flow-matching action head that
uses the student VLM backbone as its conditioning source.

Generated checkpoints, ad-hoc reports, and large eval artifacts live under
`outputs/` and `reports/`. Many of those paths are local run artifacts and are
not part of normal commits.

## Current Status

As of 2026-06-04, the VLM backbone checkpoint used by the action-expert runs is:

```text
outputs/checkpoints/no_nav_camera_labeled_official_full444k/
  no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/
  step_006250
```

`best_decode` in that run also resolves to step 6250.  The action expert is not
using the student's greedy trajectory-token decode as its output path; it uses
the student backbone/KV conditioning and learns a continuous action-space flow.

## Student VLM Backbone Reference

Full-val free-run discrete trajectory decode for `step_006250`:

| Metric | Value |
|---|---:|
| Greedy ADE/FDE vs teacher discrete trajectory | 2.557 / 8.365 m |
| Greedy ADE/FDE vs GT | 3.011 / 9.822 m |
| Teacher ADE/FDE vs GT | 1.739 / 5.175 m |
| Oracle minADE@4/minFDE@4 vs teacher | 1.376 / 3.956 m |
| Oracle minADE@4/minFDE@4 vs GT | 1.702 / 4.914 m |

Interpretation: the backbone can produce useful trajectories under sampling,
but greedy token decoding is mode-collapsed.  Treat greedy VLM decode as a
diagnostic baseline, not as the deployable action-expert path.

## Action Expert Recipe

Current action-expert recipe:

```text
prefix_mode:         student_free
ae_init_mode:        student_backbone_init
target_source:       teacher
expert_lr:           1e-4
proj_lr:             1e-4
num_time_samples:    16
grad_clip_norm:      5.0
optimizer:           AdamW, no norm/bias decay, fused when enabled
attention:           flash_attention_2 when available
inference:           temperature 0.85, 16 paths, mean_traj selection
```

The `expert_lr=1e-4` value matches the public `alpamayo-recipes` SFT Stage 2
learning rate.  `num_time_samples=16` is a local stabilization value: official
recipes use one flow-matching draw per sample, but our colder student-compatible
expert collapsed with single-draw training.

## Root-Cause Findings

The important resolved issues:

| Issue | Finding | Fix / Status |
|---|---|---|
| Projection freeze | `action_in_proj` and `action_out_proj` inherited `requires_grad=False` after deepcopy/reset | Explicitly re-enable grad and verify optimizer groups |
| Random FM collapse | Fixed `(x0,t)` could overfit, random `(x0,t)` collapsed | Caused by under-training of the 28-layer expert path |
| Independent FM task | A small MLP learned the same random FM task | FM target/sampler/formula were not the root problem |
| Official hyperparams | Public SFT uses LR `1e-4`, Beta(1.5,1.0), single draw | Our LR aligns; multi-draw is our stability addition |
| Long-horizon gap | Short horizon is good, full 6.4s remains hard | Remaining problem is long-horizon sampling/selection/generalization |

What is not currently the main suspect:

- action-space `accel_mean/std` and `curvature_mean/std`: train and eval use the
  same teacher action space, and normalized target action magnitudes are O(1).
- hidden/KV absence as the original collapse cause: oracle/no-KV/fixed/random
  diagnostics isolated the FM optimization path first.
- overfit as the current Stage 2 issue: train and val ADE are currently very
  close.

## Stage 1 Held-Out Baseline

Stage 1 used 20k train samples and 2k held-out validation samples with zero
train/val overlap.

| Step | Val ADE | Val h1.6 | Val h3.2 | Oracle minADE@16 | Train ADE | Gap |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 2.712 | 0.168 | 0.674 | 1.802 | 2.400 | 0.312 |
| 7000 | 2.512 | 0.146 | 0.612 | 1.539 | 2.431 | 0.081 |
| 9000 | 2.503 | 0.162 | 0.645 | 1.311 | 2.282 | 0.221 |
| 10000 | 2.517 | 0.144 | 0.606 | 1.288 | 2.293 | 0.224 |

Conclusion: Stage 1 fixed the random-FM amplitude collapse, but held-out
6.4-second ADE plateaued around 2.5 m.  The train/val gap was small, so the
failure was not just memorization.

## Q2 / Q3 Follow-Up

Q2 continued the 20k run to additional epochs with the same core recipe.

| Run | Best checkpoint | Val ADE | Train ADE | Oracle minADE@16 | Interpretation |
|---|---|---:|---:|---:|---|
| Q2 constant-LR continuation | `q2.../best.pt` at step 26000 | 2.1268 | 1.8181 | 1.0204 | Better than Stage 1, but still long-horizon limited |
| Q3 short cosine cooldown | `q3.../best.pt` at local step 2000 | 2.1049 | 1.7511 | 0.9567 | Schedule tweak helped slightly, but did not remove the long-horizon blocker |

Separate E2E metric smoke on Q3 `best.pt` with the Stage 1 held-out eval set:

| Setting | ADE@1 | FDE@1 | minADE@6 | minFDE@6 |
|---|---:|---:|---:|---:|
| temp 0.85, N=6, seed 42 / eval base 1042 | 2.512 | 7.301 | 1.290 | 3.847 |

The gap between ADE@1 and minADE@6/minADE@16 shows that good trajectories exist
inside the sample set, but the deployable selection rule still leaves substantial
long-horizon error.

## Stage 2 200k Run

Stage 2 is the current clean 200k run from `student_backbone_init`, not a warm
start from Q2.

```text
launcher:
  scripts/launch_stage2_ae28_200k.sh

output:
  outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603

split cache:
  outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/
  split_cache_200k_10k_seed42.json

train samples:      200000
held-out val:       10000
batch size:         8
total steps:        25000
epoch count:        1.0
eval cadence:       every 2500 steps
eval samples:       1024 val, 512 train
```

Latest parsed status at README update time:

```text
latest train step: 16900 / 25000
last completed eval: step 15000
```

Stage 2 eval curve so far:

| Step | Val ADE | Val FDE | Val p50 ADE | Val minADE@16 | Train ADE | Train minADE@16 | Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2500 | 2.986 | 8.623 | 2.282 | 1.447 | 3.098 | 1.529 | -0.112 |
| 5000 | 3.049 | 8.851 | 2.340 | 1.328 | 2.763 | 1.247 | 0.286 |
| 7500 | 2.538 | 7.281 | 1.936 | 1.106 | 2.405 | 1.087 | 0.133 |
| 10000 | 2.840 | 8.290 | 2.256 | 1.258 | 2.715 | 1.274 | 0.125 |
| 12500 | 2.492 | 7.437 | 1.884 | 1.098 | 2.437 | 1.099 | 0.054 |
| 15000 | 2.221 | 6.545 | 1.651 | 1.002 | 2.192 | 1.008 | 0.029 |

Current interpretation:

- Stage 2 is still improving; it has not completed one epoch.
- Train/val gap is tiny at the latest eval, so the current issue is not
  overfitting.
- `pred_v_abs_mean / target_v_abs_mean` is close to matched in recent train
  windows, so the original FM collapse is not back.
- The major remaining gap is deployable long-horizon selection: val ADE is
  2.221 m while oracle minADE@16 is 1.002 m at step 15000.

## Action Space

The action space is the official Alpamayo 1.5 unicycle acceleration/curvature
space:

```text
action shape:        [64, 2]
accel_mean:          0.02902694707164455
accel_std:           0.6810426736454882
curvature_mean:      0.0002692167976330542
curvature_std:       0.026148280660833106
accel_bounds:        [-9.8, 9.8]
curvature_bounds:    [-0.33, 0.33]
dt:                  0.1
```

These values are fixed teacher config constants, not re-estimated from the 20k
or 200k local splits.  They normalize physical acceleration and curvature before
FM training and denormalize predicted actions during `action_to_traj`.

## Monitoring Commands

Check live Stage 2 progress:

```bash
tail -n 40 outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/train_log.jsonl
```

Parse Stage 2 eval summaries:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/train_log.jsonl")
for line in p.read_text().splitlines():
    row = json.loads(line)
    if row.get("event") in {"val_eval", "train_eval"}:
        print(
            row["event"], row["step"],
            "ADE", row["ade_mean_m"],
            "FDE", row["fde_mean_m"],
            "minADE@N", row.get("ade_best_of_n_mean_m"),
            "paths", row.get("eval_num_paths"),
        )
PY
```

Check the exact Stage 2 command:

```bash
tr '\0' ' ' < /proc/907/cmdline
```

If PID 907 has changed, inspect the process from `nvidia-smi` or by matching the
output directory in `/proc/*/cmdline`.

## Key Files

```text
main AE training script:
  scripts/84_train_student_ae28_official.py

Stage 2 launcher, local run helper:
  scripts/launch_stage2_ae28_200k.sh

seed/minADE eval helper, local run helper:
  scripts/101_eval_ae28_seed_sweep.py

diagnostic reports:
  reports/066-fm-collapse-objective-diagnostics.md
  reports/070-ae-ablation-ladder-random-fm.md
  reports/076-official-alpamayo-recipes-sft-fm-hparams.md
  reports/080-stage1-heldout-training-results.md
  reports/083-q3-short-schedule-and-stage2-setup.md

student VLM backbone run:
  outputs/checkpoints/no_nav_camera_labeled_official_full444k/
  no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/

current Stage 2 run:
  outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603
```

## Decision Rules

Use the next Stage 2 evals to decide the next move:

1. If val ADE and minADE@16 both keep falling, continue training or extend past
   one epoch.
2. If minADE@16 improves but deployable ADE stalls, prioritize non-oracle
   selection/scoring over LR or action-space changes.
3. If train ADE becomes much lower than val ADE, data/generalization is the
   issue.
4. If `pred_v_abs_mean / target_v_abs_mean` collapses again, revisit FM
   optimization before interpreting ADE.
5. Do not change action-space mean/std unless running a dedicated ablation from
   scratch; train/eval currently use a consistent teacher action space.
