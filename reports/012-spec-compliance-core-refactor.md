# 012 Spec Compliance Core Refactor

status: resolved

symptom:
- The original training path was text-only in practice.
- Canonical sample images and ego history existed on disk, but the student never consumed them.
- Teacher cache only preserved coarse `cot` text and did not expose action-class diagnostics or prompt-selection metadata.
- Consistency and corpus wiring were partially stubbed, so `teacher_text__gt_path`, `teacher_text__human_reasoning`, and `consistency_score` were either missing or stale in downstream training.

root cause:
- The first bootstrap pipeline optimized for getting a runnable path quickly and left several spec-level contracts as placeholders.
- The student collator used tokenizer-only prompt assembly.
- Teacher inference stayed on a single prompt path even though request bundles already declared multiple prompt families.
- Corpus/eval were being rebuilt in parallel with consistency generation, so downstream artifacts could observe stale diagnostics.

resolution:
- Reworked the student path to be genuinely multimodal with `Qwen3VLProcessor`, bounded pixel budgets, canonical JPG loading, and serialized ego-history text blocks.
- Switched the default student path to the local `Cosmos-Reason2-2B` checkpoint and forced local loading when a local path exists.
- Added `load_student_processor(...)` and routed `pixel_values` plus `image_grid_thw` through the collator, trainer, and student forward path.
- Expanded teacher normalization to derive:
  - `teacher_action_class`
  - `teacher_action_confidence`
  - `teacher_selection_score`
  - `teacher_parse_status`
  - `teacher_hallucination_flags`
- Extended consistency generation to populate:
  - `teacher_text__gt_path`
  - `teacher_text__human_reasoning`
  - numeric `consistency_score`
- Rebuilt the safe corpus so `seq_kd`, `rank`, and `consistency_score` reflect the latest teacher diagnostics.
- Fixed the joint-cache scaffold so already-generated teacher text samples with status `ok` remain eligible for joint request bundles.

key files or artifacts:
- `src/model/student_wrapper.py`
- `src/training/collator.py`
- `src/training/trainer.py`
- `src/training/losses.py`
- `scripts/05_generate_teacher_text_cache.py`
- `scripts/05_run_teacher_text_inference.py`
- `scripts/06_generate_teacher_joint_cache.py`
- `scripts/07_run_consistency_gate.py`
- `scripts/08_build_multitask_corpus.py`
- `scripts/09_train_distill.py`
- `scripts/10_eval_distill.py`
- `outputs/reports/teacher_text_cache_summary_v2.json`
- `outputs/reports/consistency_summary_v2.md`
- `outputs/reports/corpus_summary_v2_refreshed.json`
- `outputs/reports/eval_summary_v2_pretrain_rerun.json`

follow-up:
- `teacher_logit_cache_path` and `teacher_hidden_path` are still unpopulated, so `logit_kd` and `feat_align` stay inactive.
- Alpamayo structured prompting still mostly falls back to heuristic recovery from `cot`, so `json_parseability` remains low.
- A dedicated post-training eval path for the student checkpoint is still needed; current eval remains teacher/corpus-centric.
