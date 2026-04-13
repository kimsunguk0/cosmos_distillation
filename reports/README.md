# Cosmos Distillation Issue Log

This folder tracks implementation issues, root causes, fixes, and follow-up work.

Current convention:

- One Markdown file per issue.
- File names stay stable and sortable.
- Each file records:
  - status
  - symptom
  - root cause
  - resolution
  - key files or artifacts
  - follow-up

Current issue list:

- [001-strict-human-coc-split-gap.md](./001-strict-human-coc-split-gap.md)
- [002-download-scope-and-priority-strategy.md](./002-download-scope-and-priority-strategy.md)
- [003-priority-download-redundant-pattern-bug.md](./003-priority-download-redundant-pattern-bug.md)
- [004-hf-xet-and-resume-failures.md](./004-hf-xet-and-resume-failures.md)
- [005-missing-front-wide-chunk-1086.md](./005-missing-front-wide-chunk-1086.md)
- [006-pyav-vendor-path-for-materialization.md](./006-pyav-vendor-path-for-materialization.md)
- [007-alpamayo-vlm-only-loading-for-text-generation.md](./007-alpamayo-vlm-only-loading-for-text-generation.md)
- [008-teacher-cache-refresh-overwrite.md](./008-teacher-cache-refresh-overwrite.md)
- [009-qwen3vl-student-wrapper-compat.md](./009-qwen3vl-student-wrapper-compat.md)
- [010-detached-train-launch-instability.md](./010-detached-train-launch-instability.md)
- [011-teacher-text-fields-mostly-cot-only.md](./011-teacher-text-fields-mostly-cot-only.md)
- [012-spec-compliance-core-refactor.md](./012-spec-compliance-core-refactor.md)
- [013-stage-b-retune-after-consistency-refresh.md](./013-stage-b-retune-after-consistency-refresh.md)
- [014-teacher-slot-prompt-recovery.md](./014-teacher-slot-prompt-recovery.md)
- [015-teacher-cache-provenance-and-quality-weighting.md](./015-teacher-cache-provenance-and-quality-weighting.md)

Latest known high-level state:

- Strict human CoC subset is `577 train / 105 val / 50 test`.
- Teacher index currently has `106` recovered/generated `ok` samples and `118` additional `ready_request_bundle` samples.
- Teacher text cache is being re-applied with slot-specific prompts to improve `meta_action` and `answer` quality.
- Materialized train samples currently used for multimodal distillation: `195`.
- Refreshed consistency-aware Stage B and tuned Stage B runs have completed on the current local sample budget.
