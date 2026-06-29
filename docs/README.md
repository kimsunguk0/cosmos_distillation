# Documentation Index

Stable architecture and operating notes for the Cosmos distillation workspace.
For experiment-by-experiment history, read `../reports/` instead.

## Current Architecture

- [`WORKSPACE_ARCHITECTURE.md`](WORKSPACE_ARCHITECTURE.md): local workspace,
  data layout, ownership boundaries, and artifact policy.
- [`DISTILLATION_ARCHITECTURE.md`](DISTILLATION_ARCHITECTURE.md):
  teacher-cache, student backbone, AE, FLEX, Q2 VQA, deployment, and eval loop.

## Existing Project Notes

- [`DATA_PIPELINE.md`](DATA_PIPELINE.md): original data pipeline contract.
- [`EVAL_PLAN.md`](EVAL_PLAN.md): original evaluation checklist.
- [`TRAINING_PLAN.md`](TRAINING_PLAN.md): training-plan notes.
- [`SUPERVISION_POLICY.md`](SUPERVISION_POLICY.md): supervision separation
  policy.
- [`HANDOFF_REBUILD_GUIDE.md`](HANDOFF_REBUILD_GUIDE.md): rebuild/handoff notes.
- [`LEGAL_NOTES.md`](LEGAL_NOTES.md): model and data license notes.

## Reading Order

1. Start with the root [`README.md`](../README.md).
2. Read [`WORKSPACE_ARCHITECTURE.md`](WORKSPACE_ARCHITECTURE.md) to understand
   where local data and generated artifacts live.
3. Read [`DISTILLATION_ARCHITECTURE.md`](DISTILLATION_ARCHITECTURE.md) to
   understand the teacher-to-student system design.
4. Read [`../reports/139-alpamayo-distillation-retrospective.md`](../reports/139-alpamayo-distillation-retrospective.md)
   for the full experiment narrative and current decisions.
