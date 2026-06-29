# 074 - Stage 1 Held-Out Split and Sanity

## Code Changes

`scripts/84_train_student_ae28_official.py` now separates train and validation items in `main()` instead of evaluating on the same `items` used for training.

Key properties:

- deterministic split by stable hash
- split key is the scene/group id: `sample_id` before `__sg_`
- no train/val overlap at sample id or scene/group level
- `val_eval` is used for best checkpoint selection
- optional `train_eval` is logged via `--eval-train-samples`
- deployable inference recipe is exposed through:
  - `--eval-temperature`
  - `--eval-num-paths`
  - `--eval-selection-method`

Note: split selection avoids expensive per-row `exists()` checks and stores remapped paths. Actual file validity is still exercised when batches are built.

## 20k/2k Split Validation

Command-level validation used the Stage 1 intended split size:

- train samples: `20000`
- val samples: `2000`
- split seed: `42`
- val fraction: `0.1`
- source corpus split: `train`
- group key: `sample_id before __sg_`

Result:

| metric | value |
|---|---:|
| scanned rows | 22704 |
| eligible rows | 22208 |
| train selected | 20000 |
| val selected | 2000 |
| train groups | 2501 |
| val groups | 250 |
| sample id overlap | 0 |
| scene/group overlap | 0 |

This fixes the previous Stage 0 issue where eval samples were effectively in-distribution.

## Sanity Run

Correct W2/Y3 student checkpoint:

`outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250`

Output:

`outputs/action_expert/stage1_heldout_sanity_128_s200_seed42_full444k`

Settings:

- train samples `128`
- val samples `16`
- eval train samples `16`
- steps `200`
- batch size `2`
- effective FM batch `32`
- `expert_lr=1e-4`
- `proj_lr=1e-4`
- `num_time_samples=16`
- `grad_clip_norm=5.0`
- `no_norm_bias_decay=True`
- `eval_temperature=0.85`
- `eval_num_paths=16`
- `eval_selection_method=mean_traj`

Split sanity:

| metric | value |
|---|---:|
| train selected | 128 |
| val selected | 16 |
| train groups | 16 |
| val groups | 2 |
| sample id overlap | 0 |
| scene/group overlap | 0 |

ADE curve:

| split | step | full ADE | p50 | h1.6 ADE | h3.2 ADE | oracle best | mean over paths | path ADE std | single-path mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0 | 8.9290 | 7.5643 | 0.7853 | 2.7049 | 5.7576 | 9.5129 | 2.7677 | 8.8200 |
| train | 0 | 9.5323 | 6.5680 | 0.6488 | 2.5964 | 3.4253 | 9.9925 | 4.1534 | 7.9356 |
| val | 100 | 9.4672 | 9.8614 | 0.7214 | 2.6739 | 5.8799 | 10.1520 | 2.9660 | 10.8989 |
| train | 100 | 3.1755 | 2.2725 | 0.3254 | 1.0002 | 2.1327 | 5.6027 | 2.9899 | 5.7722 |
| val | 200 | 11.4969 | 10.1969 | 0.8502 | 3.2137 | 6.9646 | 11.8698 | 3.0299 | 13.1964 |
| train | 200 | 5.4473 | 5.0690 | 0.4290 | 1.5442 | 1.9935 | 6.4881 | 3.4165 | 7.5567 |

Sanity interpretation:

- train split learns quickly by step 100, so gradients, projections, split plumbing, and deployable eval path are working.
- val has only 2 scene groups in this tiny sanity run and is too noisy for a generalization decision.
- val did not improve in this tiny run, so the actual 20k/2k Stage 1 run is required before making any Stage 2 decision.

## Stage 1 Run

Launched full Stage 1 as a detached run:

`outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531`

PID file:

`outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/run.pid`

Command:

```bash
outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/run_command.sh
```

Launch status note:

- the first detached launch was not actually training: the Python child stayed before GPU/model progress, `stdout.log` was empty, and no Stage 1 process appeared in `nvidia-smi`
- stale launch processes were killed and Stage 1 was relaunched
- split selection was cached to avoid the startup stall:
  `outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/split_cache_20k_2k_seed42.json`
- current relaunched process:
  - bash PID `11077`
  - Python PID `11082`
  - GPU memory observed increasing from about `5.0GB` to `12.3GB`
- startup log confirms:
  - split cache loaded
  - train/val sample overlap `0`
  - train/val scene-group overlap `0`
  - student loaded on `cuda:0`
  - action projections are trainable and included in the optimizer at LR `1e-4`

Important settings:

- train `20000`
- held-out val `2000`
- train eval `256`
- steps `10000`
- batch size `2`
- eval batch size `4`
- eval every `1000`
- save every `1000`
- `expert_lr=1e-4`
- `proj_lr=1e-4`
- `num_time_samples=16`
- `grad_clip_norm=5.0`
- `no_norm_bias_decay=True`
- `eval_temperature=0.85`
- `eval_num_paths=16`
- `eval_selection_method=mean_traj`

## Current Conclusion

held-out baseline = pending; Stage 1 full run is active after relaunch. Stage 2 갈 자격 = TBD until the 20k/2k validation curve is available.

## Train/Val Gap Readout

Stage 1 must be judged by train ADE and held-out val ADE together, not val alone. The full run logs `val_eval` on 2000 held-out samples and `train_eval` on 256 train samples at the same eval points via `--eval-train-samples 256`.

Decision rubric:

| train ADE | val ADE | interpretation | next action |
|---:|---:|---|---|
| ~0.5m | ~0.5m | generalizes | Stage 2 is justified |
| ~0.5m | ~2.0m | memorizes / large generalization gap | increase data, regularize, or revisit split/distribution |
| ~2.0m | ~2.2m | undertrained but gap is small | train longer / more data / optimize throughput |

Primary diagnostic is the gap: `val_full_ADE - train_full_ADE`, alongside the absolute val full ADE and horizon breakdown.
