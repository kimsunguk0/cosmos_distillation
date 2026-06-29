# 077 Stage 1 Speedup Prep

Prepared while the original Stage 1 run continues.

## Current bottleneck

- Normal training speed is about 3.0s/step.
- Each 1000-step eval has been adding roughly 1.6-2.3h.
- Main causes:
  - `eval_samples=2000`
  - `eval_num_paths=16`
  - legacy eval loops over paths in Python
  - per-path/per-step `gc.collect()` and `torch.cuda.empty_cache()`
  - large per-sample `rows` emitted into JSON logs

## Script updates

Updated `scripts/84_train_student_ae28_official.py` with opt-in fast-path flags:

- `--resume-ae-checkpoint`: load a saved AE bundle checkpoint.
- `--start-step`: continue absolute step numbering from a checkpoint.
- `--eval-vectorize-paths`: batch N diffusion paths into path chunks.
- `--eval-path-batch-size`: chunk size for vectorized paths.
- `--eval-log-rows`: keep eval JSON compact; `0` logs aggregates only.
- `--cleanup-every`: control train-loop `gc/empty_cache`; `0` disables.
- `--eval-cleanup-every`: control eval-loop `gc/empty_cache`; `0` disables.
- `--eval-only`: load a checkpoint and run val/train eval once without training.
- `--attn-implementation flash_attention_2`: now propagates to the AE expert instead of being coerced to SDPA.

Defaults preserve the legacy behavior unless these flags are explicitly used.

## Ready-to-run template

Prepared:

`outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/run_fast_resume_template.sh`

Default use after `step_005000.pt` exists:

```bash
nohup outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/run_fast_resume_template.sh 5000 \
  > fast_resume_5000.launch.log 2>&1 &
```

Conservative defaults:

- resume from `step_005000.pt`
- keep `batch_size=2`
- keep `num_time_samples=16`
- keep `eval_num_paths=16`
- reduce online val eval to `512`
- keep train eval at `256`
- vectorize eval paths in chunks of `4`
- log aggregate eval metrics only
- disable repeated cache cleanup
- use `flash_attention_2` by default in the fast-resume template

Tunable environment overrides:

```bash
BATCH_SIZE=4 EVAL_SAMPLES=512 EVAL_PATH_BATCH_SIZE=4 \
  nohup outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/run_fast_resume_template.sh 5000 \
  > fast_resume_5000_b4.launch.log 2>&1 &
```

Fallback to SDPA if FA2 errors:

```bash
ATTN_IMPL=sdpa \
  nohup outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/run_fast_resume_template.sh 5000 \
  > fast_resume_5000_sdpa.launch.log 2>&1 &
```

## Validation

- `python3 -m py_compile scripts/84_train_student_ae28_official.py`: pass.
- `bash -n run_fast_resume_template.sh`: pass.
- `flash_attn` import: pass (`flash_attn 2.8.3`, H200 NVL, compute capability 9.0).
- CUDA kernel smoke: pass (`flash_attn_func`, bf16, no NaNs).
- End-to-end `84` FA2 smoke: pass.
  - Command used `--attn-implementation flash_attention_2 --eval-only --resume-ae-checkpoint step_004000.pt`.
  - Output: `outputs/action_expert/fa2_smoke_eval_step4000_fixed_20260601_060358`.
  - Log contained `Casting fp32 inputs back to torch.bfloat16 for flash-attn compatibility.`
  - `summary.json` status: `ok`.
