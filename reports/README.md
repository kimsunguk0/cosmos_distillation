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
- [016-priority-next-40-progress-reporting-gap.md](./016-priority-next-40-progress-reporting-gap.md)
- [017-t0-anchor-localization-bug.md](./017-t0-anchor-localization-bug.md)
- [018-runtime-bundle-and-grounded-selection-v2.md](./018-runtime-bundle-and-grounded-selection-v2.md)
- [019-true-kd-smoke-and-shared-vocab-guard.md](./019-true-kd-smoke-and-shared-vocab-guard.md)
- [020-spec-design-issues-so-far.md](./020-spec-design-issues-so-far.md)
- [021-alpamayo-base-vs-1p5-teacher-contract.md](./021-alpamayo-base-vs-1p5-teacher-contract.md)
- [022-v3-2-stage-a-contract-recovery-issues.md](./022-v3-2-stage-a-contract-recovery-issues.md)
- [023-stage-a3-traj-audit-and-continuation.md](./023-stage-a3-traj-audit-and-continuation.md)
- [024-hard-only-traj-continuation-result.md](./024-hard-only-traj-continuation-result.md)
- [025-subset-collapse-threshold-1-4-16.md](./025-subset-collapse-threshold-1-4-16.md)
- [026-a0-geometry-subset-threshold-1-4-16.md](./026-a0-geometry-subset-threshold-1-4-16.md)
- [027-current-traj-collapse-status-and-next-actions.md](./027-current-traj-collapse-status-and-next-actions.md)
- [028-short-horizon-geometry-regression.md](./028-short-horizon-geometry-regression.md)
- [029-teacher-traj-cache-design.md](./029-teacher-traj-cache-design.md)
- [030-trainer-traj-bug-fix-and-a0-rerun.md](./030-trainer-traj-bug-fix-and-a0-rerun.md)
- [031-teacher-traj-token-topk-subset-probe.md](./031-teacher-traj-token-topk-subset-probe.md)

Latest known high-level state:

- Teacher v2 pipeline now targets the `262` usable canonical samples.
- Runtime bundle split, explicit `t0` anchoring, weak GT semantics wiring, and grounded teacher selection are implemented.
- Smoke validation has confirmed:
  - `t0` anchor gate pass
  - runtime bundle forbidden-key gate pass
  - nonzero `logit_kd`
  - nonzero `self_cons`
  - checkpoint saving
- Full `teacher262` background pipeline is currently running from `teacher overwrite -> signal cache -> consistency -> corpus -> train -> eval`.
- Consolidated design/spec issue summary:
  - [020-spec-design-issues-so-far.md](./020-spec-design-issues-so-far.md)
- Current distillation status:
  - `Stage B` remains paused
  - `Stage A` is being rebuilt around `traj-only` warmup and decoded-geometry GT trajectory loss
  - latest trajectory-collapse summary:
    - [027-current-traj-collapse-status-and-next-actions.md](./027-current-traj-collapse-status-and-next-actions.md)
  - latest teacher trajectory extraction design:
    - [029-teacher-traj-cache-design.md](./029-teacher-traj-cache-design.md)
  - latest post-bugfix A0 rerun:
    - [030-trainer-traj-bug-fix-and-a0-rerun.md](./030-trainer-traj-bug-fix-and-a0-rerun.md)
  - latest teacher trajectory token/top-k subset probe:
    - [031-teacher-traj-token-topk-subset-probe.md](./031-teacher-traj-token-topk-subset-probe.md)
