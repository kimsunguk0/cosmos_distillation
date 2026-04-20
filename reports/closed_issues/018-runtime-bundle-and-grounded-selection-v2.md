# 018 Runtime Bundle And Grounded Selection V2

- status: resolved
- symptom:
  - teacher request bundles contained fields that could allow future-label leakage
  - teacher selection favored syntax/completeness more than grounded agreement
- root cause:
  - runtime and diagnostics payloads were not separated
  - selection score only weakly reflected human/weak-GT agreement and internal consistency
- resolution:
  - runtime bundle now contains only frames, timestamps, ego history, prompt families, and policy metadata
  - diagnostics bundle now carries human and weak-derived references
  - validator rejects forbidden runtime keys
  - teacher selection now includes groundedness terms: human agreement, weak GT agreement, hallucination penalty, and internal consistency
  - prompt families now carry per-family generation configs
- key files or artifacts:
  - `src/data/teacher_cache.py`
  - `scripts/05_generate_teacher_text_cache.py`
  - `scripts/05_run_teacher_text_inference.py`
  - `outputs/reports/teacher_text_cache_summary_teacher262_v2.json`
- follow-up:
  - revisit grounded selection weights once more full-sample teacher outputs are available

