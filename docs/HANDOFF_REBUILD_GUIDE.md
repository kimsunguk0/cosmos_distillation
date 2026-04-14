# Handoff And Rebuild Guide

This document is for moving the `cosmos_distillation` work to another server when the current machine may become unavailable.

## What to copy before losing access

If you can only save a few things, save these first.

### 1. The repository itself

- clone target:
  - `git@github.com:kimsunguk0/cosmos_distillation.git`
- current important commit at handoff time:
  - check `git rev-parse HEAD` on the source machine

### 2. Model weights

These are critical if you want to continue teacher probing or student training without re-downloading everything from scratch.

- teacher VLM weights:
  - `/home/pm97/workspace/sukim/weights/alpamayo15_vlm_weights`
- student base weights:
  - `/home/pm97/workspace/sukim/weights/cosmos-reason-2b`

If you cannot copy these, prepare an alternative plan to restore them first on the new server.

### 3. Small processed manifests that define the exact sample subsets

These are tiny compared with raw dataset downloads and are worth preserving.

- `data/processed/strict_download_subset_manifest.parquet`
  - the strict human CoC subset
- `data/processed/teacher_262_manifest.parquet`
  - the current `262` usable-sample set we were actively using
- `data/processed/materialize_new_40_manifest.parquet`
  - the additional `40` samples we queued after the first `224`
- `outputs/reports/priority_next2_40_run.json`
  - the exact `next2_40` chunk selection for resumed download

### 4. Optional but highly valuable cached artifacts

If storage allows, copy these too.

- `data/teacher_cache/`
  - expensive teacher outputs and signal caches
- `outputs/checkpoints/`
  - train checkpoints
- `outputs/reports/`
  - experiment summaries, evals, and timing profiles

### 5. Useful reference repos

These are not required to run the current pipeline, but they are useful when debugging teacher behavior.

- `/home/pm97/workspace/sukim/alpamayo1.5`
- `/home/pm97/workspace/sukim/alpamayo_base`

## What can be recreated later

These are reproducible and do not need to be backed up if transfer is expensive.

- raw `PhysicalAI-Autonomous-Vehicles` chunk zips
- canonical samples
- supervision records
- rebuilt corpus files
- smoke/eval reports

## Minimal restore checklist on a new server

### 1. Clone the repo

```bash
git clone git@github.com:kimsunguk0/cosmos_distillation.git
cd cosmos_distillation
```

### 2. Restore the small manifests

Copy these files from the old server into the same relative paths if possible.

- `data/processed/strict_download_subset_manifest.parquet`
- `data/processed/teacher_262_manifest.parquet`
- `data/processed/materialize_new_40_manifest.parquet`
- `outputs/reports/priority_next2_40_run.json`

If these are missing, exact reproduction of the current working subset will be harder.

### 3. Prepare the Python environment

Minimum practical requirements:

- `torch`
- `transformers`
- `accelerate`
- `peft`
- `huggingface_hub`
- `pandas`
- `pyarrow`
- `numpy`
- `pillow`

Optional but useful:

- `av`
- `pytest`

If you want the same training path as the current server, also make sure these work:

- `flash_attn` if supported by the GPU stack
- otherwise fall back to `attn_implementation="eager"` or `"sdpa"` only if the model supports it
- `tmux`

If you already have a working venv, point the scripts to it. On the old machine we used:

- `/home/pm97/workspace/kjhong/alpamayo_100/ar1_venv/bin/python`

### 4. Redownload metadata only

```bash
python scripts/01_download_metadata.py
```

This is lightweight and should finish before any raw chunk downloads.

### 5. Resume raw chunk download from the preserved chunk plan

If you copied `outputs/reports/priority_next2_40_run.json`, you can resume the same target selection instead of recomputing a different `40`-chunk batch.

Single-run resume:

```bash
python scripts/03_download_priority_chunks.py \
  --manifest-path data/processed/strict_download_subset_manifest.parquet \
  --selected-chunks-json outputs/reports/priority_next2_40_run.json \
  --summary-json outputs/reports/priority_next2_40_run.json \
  --max-workers 8
```

Retry + watcher mode, matching the latest safer setup:

```bash
tmux new-session -d -s priority_next2_40_retry 'scripts/17_run_priority_next2_40_retry.sh'
tmux new-session -d -s priority_next2_40_watch 'scripts/18_watch_priority_next2_40.sh'
```

Current behavior of the retry/watcher setup:

- retries on broken chunked downloads
- keeps `priority_next2_40_progress.json` fresh
- avoids trusting stale PIDs by checking process cmdline

Important note:

- raw AV data is effectively downloaded in `chunk zip` units, not sample units
- preserving the selected-chunk JSON is therefore much more valuable than trying to reconstruct the exact same next batch by memory

### 6. Re-materialize canonical samples

For the main teacher set:

```bash
python scripts/03_materialize_canonical_samples.py \
  --manifest-path data/processed/teacher_262_manifest.parquet \
  --summary-json outputs/reports/canonical_materialization_teacher262_v2.json
```

### 7. Rebuild supervision

```bash
python scripts/04_build_supervision_records.py
```

### 8. Teacher work order

Do not jump straight into full distillation again.

Current safest order:

1. probe honest teacher routes first
2. confirm what the Alpamayo teacher can truly emit
3. only then run full teacher overwrite/cache generation

Important current understanding:

- safe primary teacher:
  - `teacher_long_cot`
- not yet fully trusted:
  - direct `teacher_answer`
  - direct `teacher_meta_action`
- currently disabled for honest training:
  - `structured_json`

Current honest interpretation:

- the public `alpamayo1.5` helper path clearly supports `cot`
- `answer` and `meta_action` direct slots are still not strong enough to trust as primary teacher targets
- `alpamayo_base` shows a chain-style `cot -> meta_action` design, but our first probe on the local `alpamayo15_vlm_weights` did not yet recover a usable direct `meta_action`
- until that changes, prefer `teacher_long_cot` as the primary soft target and treat other fields as optional or derived

### 9. Rebuild corpus and train

Once teacher cache is rebuilt:

```bash
python scripts/08_build_multitask_corpus.py
python scripts/09_train_distill.py
```

## Current design caveat to remember

The biggest lesson from this server’s work is:

- the distillation pipeline itself is not the main problem
- teacher contract honesty is the main problem

Use this rule on the new server:

- if Alpamayo only gives trustworthy `long_cot`, then train with `long_cot` as the primary teacher target
- do not pretend `answer/meta_action/structured_json` are strong direct teacher slots unless smoke probes prove they are

## Current recommended backup priority

If you are in a hurry, copy in this order:

1. the repo itself
2. `/home/pm97/workspace/sukim/weights/alpamayo15_vlm_weights`
3. `/home/pm97/workspace/sukim/weights/cosmos-reason-2b`
4. `data/processed/teacher_262_manifest.parquet`
5. `data/processed/strict_download_subset_manifest.parquet`
6. `outputs/reports/priority_next2_40_run.json`
7. `data/teacher_cache/` if possible
8. `outputs/checkpoints/` if possible

These few files make recovery much faster than re-deriving the same sample/chunk selections from scratch.
