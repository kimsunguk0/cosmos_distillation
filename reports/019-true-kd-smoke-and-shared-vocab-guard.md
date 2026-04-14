# 019 True KD Smoke And Shared Vocab Guard

- status: mitigated
- symptom:
  - initial train smoke failed even after sparse top-k teacher signals were present
  - first failure was sequence-length mismatch between sparse teacher tokens and student labels
  - second failure was CUDA gather out-of-bounds from teacher token ids outside student vocab
- root cause:
  - sparse signal cache stores only target-token positions, while loss initially assumed full-sequence alignment
  - teacher and student vocabularies are not perfectly aligned, so raw teacher top-k ids cannot be blindly gathered from student logits
- resolution:
  - `logit_kd_loss()` now aligns teacher sparse tokens against the student's valid target-token positions sample-by-sample
  - out-of-range teacher token ids are filtered before KD is computed
  - 10-sample smoke validated nonzero `seq_kd`, nonzero `logit_kd`, nonzero `self_cons`, and checkpoint saving
- key files or artifacts:
  - `src/training/losses.py`
  - `scripts/05_extract_teacher_signal_cache.py`
  - `outputs/reports/teacher_signal_cache_teacher262_smoke10.json`
  - `outputs/reports/train_stage_b_teacher262_smoke.json`
  - `outputs/reports/smoke_gate_summary_teacher262_smoke10.json`
- follow-up:
  - add a tokenizer-aware teacher-to-student token mapping if exact field-level logit KD becomes a priority

