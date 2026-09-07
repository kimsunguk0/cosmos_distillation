# Code Cleanup Audit

Date: 2026-09-07

This audit records the current cleanup surface for `cosmos_distillation`.
It intentionally does not move or delete files. The repository already has a
large dirty worktree, so cleanup should be split into small commits with clear
scope and rollback points.

## Snapshot

| Area | Observed state | Cleanup concern |
|---|---:|---|
| Branch | `main` | Worktree is not clean. Avoid broad staging. |
| Modified tracked files | `56` | Existing source/config/test edits need separate review. |
| Untracked files | `84` | Mostly experiment scripts/configs/reports. |
| `scripts/` files | `262` | Too many top-level entrypoints and one-off launchers. |
| Top-level Python scripts | `168` | Several scripts contain framework-sized logic. |
| Top-level shell launchers | `92` | `run_*`, `launch_*`, `watch_*` jobs are mixed together. |
| `configs/train/*.yaml` | `127` | Canonical recipes and sweep configs are not separated. |
| Test files | `13` | Enough for gated refactors, but no project lint config exists. |

## Disk-Heavy Artifact Roots

These are not code changes, but they dominate repository maintenance.
Generated artifacts are already meant to stay out of normal commits.

| Path | Size | Recommendation |
|---|---:|---|
| `outputs/action_expert/` | `322G` | Prune periodic `step_*` checkpoints after keeping best/final/metrics. |
| `outputs/checkpoints/` | `206G` | Remove repeated old `student_state.pt` copies after run selection. |
| `outputs/ae28_teacher_dumps/` | `186G` | Archive or delete if not referenced by the current pipeline. |
| `outputs/kv_distill_pipeline/` | `21G` | Compare `run_20260527` vs `run_20260527_v2`; keep one if equivalent. |
| `outputs/trt_export/` | `17G` | Keep only deployment targets that are still used. |
| `outputs/exports/` | `16G` | Keep only reproducible/final export directories. |
| `outputs/reports/` | `2.8G` | Preserve durable reports; archive generated dashboards. |
| `data/corpus/` | `7.1G` | Treat old JSONL corpora as archive candidates if the external distill dataset is canonical. |

## Code Cleanup Priorities

### 1. Script directory split

`scripts/` should stop being a flat experiment log. Move-only refactors should
come first, with compatibility wrappers when a filename is likely referenced by
older runbooks.

Suggested layout:

```text
scripts/
  train/       long-lived training entrypoints
  eval/        benchmark and checkpoint evaluation entrypoints
  audit/       diagnostics, gates, metric audits
  launch/      tmux/nohup/scheduler shell launchers
  export/      checkpoint/export/packaging scripts
  legacy/      old one-off experiments kept for provenance
```

Initial classification:

| Class | Examples | Action |
|---|---|---|
| Keep as compatibility wrappers | `09_train_distill.py`, `25_decode_checkpoint_overlays.py`, `84_train_student_ae28_official.py` | Keep filenames stable until downstream callers are updated. |
| Move to `scripts/launch/` | `launch_*`, `run_*`, `watch_*`, `monitor_*`, `stop_*` | Move after checking docs/runbooks. |
| Move to `scripts/audit/` | `audit_*`, `check_*`, `measure_*`, `validate_*`, `score_*`, `summarize_*` | Low behavior risk if imports are path-safe. |
| Move to `scripts/eval/` | `eval_*`, `benchmark_*`, `bench_*`, `*_eval_*.py` | Update references from reports/runbooks. |
| Move to `scripts/legacy/` | April/May pipeline bootstrap scripts and broken old-path launchers | Keep for provenance unless explicitly deleted. |

### 2. Split oversized entrypoints

Several `scripts/` files are large enough to be library modules:

| File | Lines | Refactor target |
|---|---:|---|
| `scripts/09_train_distill.py` | `3049` | Move scheduler, checkpoint pruning, eval config, distributed setup into `src/training/`. |
| `scripts/84_train_student_ae28_official.py` | `2976` | Move AE dataset/model/runtime helpers into `src/training/` or `src/inference/`. |
| `scripts/105_train_flex_teacher_parity.py` | `2689` | Move FLEX parity utilities into `src/model/` and `src/inference/`. |
| `scripts/25_decode_checkpoint_overlays.py` | `1572` | Move decoding/render helpers into `src/inference/`. |
| `scripts/117_backbone_discrete_on_ae_val1024.py` | `1473` | Move reusable discrete-token metrics to `src/inference/` or `src/data/`. |

Do not combine these into one broad refactor. Each file should be split behind
tests that already cover nearby contracts, especially `collator`, `losses`,
`trainer`, `tokenizer_ext`, and `checkpoint_eval`.

### 3. Normalize runtime paths

Hardcoded local paths are still widespread:

| Area | Files with local path/job-control hits |
|---|---:|
| `scripts/` | `99` |
| `src/` | `2` |
| `configs/` | `1` |
| `tests/` | `0` |

The canonical direction should be:

- Prefer `src/utils/runtime_paths.py` for local dataset/model path discovery.
- Keep shell launchers environment-driven: `ROOT`, `COSMOS_DATA_ROOT`,
  `ALPAMAYO_MODEL_PATH`, `ALPAMAYO_SRC`, `CUDA_VISIBLE_DEVICES`.
- Replace stale paths such as `/home/pm97/workspace/sukim/cosmos_distillation`
  with repo-relative discovery before moving those launchers.
- Do not hide machine-specific paths in Python defaults when a CLI flag or
  environment variable is enough.

### 4. Config cleanup

`configs/train/` currently mixes current recipes, ablations, sweeps, and older
stage configs.

Suggested layout:

```text
configs/train/
  active/      current canonical recipes
  sweeps/      learning-rate/model-size/token-topk sweeps
  legacy/      old Stage A/B/FLEX/Q2 historical configs
  smoke/       tiny overfit or smoke-test configs
```

First-pass grouping candidates:

| Group | Count | Notes |
|---|---:|---|
| `stage_bp*` | `19` | No-nav backbone pipeline family. |
| `stage_flex*` / `stage_ml*` / `stage_mlflex*` | `15` | FLEX and ML-FLEX experiments. |
| `stepb*` | `17` | Currently untracked StepB configs; review before commit. |
| `stage_b0*` | `4` | Older B0 FP8 recipes. |
| `stage_a0*` / `stage_t*` | `29` | Early teacher/student interface configs. |
| Other yaml | `43` | Needs manual classification. |

### 5. Commit hygiene

Current untracked files are mostly durable experiment products:

| Area | Untracked count | Recommendation |
|---|---:|---|
| `scripts/` | `61` | Classify before committing; many are launch/eval/audit one-offs. |
| `configs/` | `17` | Commit only if tied to a retained experiment report. |
| `reports/` | `6` | Likely durable; review and commit separately from code moves. |

Do not stage all changes in this repository until the existing dirty worktree is
split by topic. A safe order is:

1. Commit this audit document.
2. Commit durable reports, if they are final.
3. Commit active configs with their matching reports.
4. Move one script category at a time, preserving wrappers for known entrypoints.
5. Split oversized entrypoints behind targeted tests.
6. Prune generated artifacts only from an explicit artifact manifest.

## Suggested Test Gates

Before and after each real code refactor:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_collator.py tests/unit/test_losses.py tests/unit/test_tokenizer_ext.py tests/unit/test_trainer.py
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_decoding.py tests/unit/test_teacher_cache.py
```

Use broader tests only when touching training/inference contracts:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/unit tests/integration
```

