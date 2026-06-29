# 082 Stage 1 Train Speed Flags

Date: 2026-06-02

## Changes

Implemented two opt-in speed flags in `scripts/84_train_student_ae28_official.py`.

### `--allow-train-cache-mutation`

Skips `copy.deepcopy(prompt_cache)` in `train_step()` before `batch_repeat_interleave()` when `num_time_samples > 1`.

Why:
- Q2 uses `num_time_samples=16`, so the prompt KV cache is repeated every step.
- The deepcopy is only required if the same batch cache is reused later for diagnostics.
- Q2 uses `--train-ade-every 0`, so the same batch cache is not reused after `train_step()`.

Guard:
- The script raises if `--allow-train-cache-mutation` is combined with `--train-ade-every > 0`.

Log:
- Train rows now include `train_cache_deepcopy`: `0.0` when the copy is skipped, `1.0` when preserved.

### `--fused-adamw`

Uses `torch.optim.AdamW(..., fused=True)`.

Guard:
- Requires `--device` to be CUDA.

Log:
- Startup emits `{"event": "optimizer_created", "optimizer": "AdamW", "fused": true/false}`.

## Verification

- `python -m py_compile scripts/84_train_student_ae28_official.py`: pass.
- `--help` exposes both new flags.

Note:
- These changes do not affect the currently running Q2 process PID 17058. They apply after restart or to future runs.

## Suggested Next Command Additions

For the next Q2/Q3-style launch, add:

```bash
--allow-train-cache-mutation \
--fused-adamw
```

Keep `--train-ade-every 0` when using `--allow-train-cache-mutation`.
