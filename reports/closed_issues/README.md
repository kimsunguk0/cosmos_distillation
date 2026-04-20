# Closed Issue Reports

This folder archives the numbered reports that led up to the current collapse-resolution summary in [../README.md](../README.md).

The reports are kept for traceability, but the active high-level interpretation has changed:

- reports `001-021`: data, teacher-cache, runtime, and early spec issues
- reports `022-030`: Stage A contract repair, GT trajectory collapse, and geometry-regression attempts
- reports `031-042`: teacher trajectory, aux/interface, hybrid decode, and absorption probes
- reports `043-047`: upper-layer/readout/all-layer LoRA probes that were too short or too mixed-objective to settle the clean SFT question
- reports `048-057`: hidden-manifold and H1 readout probes, useful diagnostically but no longer the first path forward

Important note: some old reports contain relative links written for the previous flat `reports/` layout. Treat them as historical notes; the current summary and next-step direction live in the parent README.

## Issue Index

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
- [032-teacher-traj-hidden-projector-subset-probe.md](./032-teacher-traj-hidden-projector-subset-probe.md)
- [033-traj-aux-head-subset-probe.md](./033-traj-aux-head-subset-probe.md)
- [034-teacher-pair-t1-bootstrap.md](./034-teacher-pair-t1-bootstrap.md)
- [035-p0-manifest-and-p2-paired-interface-v2.md](./035-p0-manifest-and-p2-paired-interface-v2.md)
- [036-p2a-frozen-interface-fit.md](./036-p2a-frozen-interface-fit.md)
- [037-p2b-prefix8-anchor.md](./037-p2b-prefix8-anchor.md)
- [038-p2c-mixed-curriculum.md](./038-p2c-mixed-curriculum.md)
- [039-mixed-from-format-restores-traj-emission.md](./039-mixed-from-format-restores-traj-emission.md)
- [040-hybrid-traj-decode-breaks-body-collapse.md](./040-hybrid-traj-decode-breaks-body-collapse.md)
- [041-prefix16-pseudoce-partially-absorbs-hybrid-body.md](./041-prefix16-pseudoce-partially-absorbs-hybrid-body.md)
- [042-followup-absorption-tests-regressed-to-plateau.md](./042-followup-absorption-tests-regressed-to-plateau.md)
- [043-purelm-upper-block-unfreeze-probes-still-plateau.md](./043-purelm-upper-block-unfreeze-probes-still-plateau.md)
- [044-shared-bottleneck-hidden-bridge-upper2-still-plateaus.md](./044-shared-bottleneck-hidden-bridge-upper2-still-plateaus.md)
- [045-hiddenfirst-hybrid-geometry-still-far-from-teacher.md](./045-hiddenfirst-hybrid-geometry-still-far-from-teacher.md)
- [046-readout-unfreeze-upper2-upper4-still-plateaus.md](./046-readout-unfreeze-upper2-upper4-still-plateaus.md)
- [047-alllayer-readout-contract-prefix16-still-plateaus.md](./047-alllayer-readout-contract-prefix16-still-plateaus.md)
- [048-h0a-frozen-latent-958-hidden-recovery.md](./048-h0a-frozen-latent-958-hidden-recovery.md)
- [049-h0b-a-vs-b-projector-freeze-vs-lowlr.md](./049-h0b-a-vs-b-projector-freeze-vs-lowlr.md)
- [050-h0c-tokenalign-smoke-regresses-geometry.md](./050-h0c-tokenalign-smoke-regresses-geometry.md)
- [051-h0d-full-from-h0b-strengthens-geometry-but-not-readout.md](./051-h0d-full-from-h0b-strengthens-geometry-but-not-readout.md)
- [053-h1-probe-rows-only-smoke-from-h0d.md](./053-h1-probe-rows-only-smoke-from-h0d.md)
- [054-h1b-dense-traj-vocab-rows-smoke-from-h0d.md](./054-h1b-dense-traj-vocab-rows-smoke-from-h0d.md)
- [055-h1c-dense-rows-plus-norm-last1-from-h0d.md](./055-h1c-dense-rows-plus-norm-last1-from-h0d.md)
- [056-teacher-pair-upperbound-sft-from-h0d-step128.md](./056-teacher-pair-upperbound-sft-from-h0d-step128.md)
- [057-h1d-h1e-listwise-smokes-from-h0d.md](./057-h1d-h1e-listwise-smokes-from-h0d.md)
