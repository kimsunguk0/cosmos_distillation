# Cosmos Distillation

This repository is the student-side workspace for distilling Alpamayo-1.5
driving behavior into a Cosmos-Reason2-2B student. The project covers the full
loop from teacher data construction to student VLM distillation, action-expert
flow matching, FLEX visual-token compression, Q2 VQA grounding, deployment
export, and benchmark reporting.

Large checkpoints, teacher dumps, eval outputs, and visualizations are local
artifacts. Normal commits should focus on source, configs, and durable reports.

## Current Snapshot

As of 2026-06-29:

- Quality reference: no-FLEX Cosmos-Reason2-2B backbone `step_006250` plus
  student-compatible AE28.
- Deployment branch: ML-FLEX K512 plus AE28 or AE14. It reduces latency, but it
  is not yet no-FLEX-equivalent on the semantic benchmark.
- Grounding curriculum: official Q2 VQA is the main language-grounding target.
  Raw Q1 / 1A prompts are auxiliary only after strict filtering.
- Selector work: AE-path ranking has real oracle headroom, but the first
  path-only reranker did not beat `mean_traj`.
- QAT/export: the high-level INT4/AWQ direction is right, but the repo-side QAT
  save/load and module-scope contracts need hardening before production use.

The compact project history is in
[`reports/139-alpamayo-distillation-retrospective.md`](reports/139-alpamayo-distillation-retrospective.md).

Detailed notes:
[`docs/DISTILLATION_ARCHITECTURE.md`](docs/DISTILLATION_ARCHITECTURE.md).

## Current Best Results

The cleanest global comparison is the semantic val806 benchmark. All models
generate their own CoT/prefix before action inference.

| Model | N | ADE GT | FDE GT | minADE6 GT | minFDE6 GT | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Alpamayo-1.5-10B | 806 | `1.6742` | `4.8004` | `0.9280` | `2.7102` | `1917 ms` |
| Student-2B-NoFLEX-AE28 | 806 | `2.7227` | `8.1559` | `1.6835` | `5.0521` | `616 ms` |
| Student-2B-FLEXK512-AE28 | 806 | `3.0818` | `9.3282` | `2.0721` | `6.2832` | `525 ms` |
| Student-2B-FLEXK512-AE14 | 806 | `3.1970` | `9.6055` | `2.5478` | `7.6595` | `493 ms` |

Source:
[`reports/138-semantic-val806-4model-benchmark.md`](reports/138-semantic-val806-4model-benchmark.md).

## Key Artifacts

```text
No-FLEX quality reference:
  outputs/checkpoints/no_nav_camera_labeled_official_full444k/
    no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/
    step_006250

No-FLEX AE28:
  outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/best.pt

FLEX K512 backbone:
  outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final

FLEX AE28:
  outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt

AE14:
  outputs/action_expert/ae14_from_ae28_10step/best.pt

Semantic benchmark:
  outputs/benchmarks/semantic_val806_4models_20260612/

Presentation visualizations:
  outputs/benchmarks/semantic_val806_4models_20260612/visualizations/
  ../../visualization/
```

## Current Decisions

- Keep no-FLEX 2B+AE28 as the quality reference.
- Treat semantic val806 as the default benchmark for cross-model decisions.
- Keep official Q2 as the main VQA grounding target; raw Q1 / 1A labels are
  auxiliary only after strict visual filtering.
- Evaluate FLEX changes on semantic val806, not only on small val512 or 68-sample
  debug sets.
- Do not trust path-only AE reranking as a selector solution. The next selector
  needs prefix confidence, token entropy/margin, diffusion likelihood, or
  teacher/value-head supervision.
- Harden QAT contracts before launching expensive QAT runs.

## Repository Map

```text
configs/train/    Training configs for backbone, FLEX, Q2, QAT, and AE runs
scripts/          Launchers, eval scripts, export scripts, and diagnostics
src/model/        Student wrapper, checkpoint IO, FLEX scene encoder
src/training/     Collators, trainers, FLEX batches, QAT integration
src/vqa/          Q2 VQA grounding data/eval utilities
docs/             Stable design and architecture notes
reports/          Durable experiment reports and decision records
outputs/          Local generated artifacts; not normal commit material
```

## Reports To Read First

- [`reports/139-alpamayo-distillation-retrospective.md`](reports/139-alpamayo-distillation-retrospective.md):
  full project history and current operating decisions.
- [`reports/138-semantic-val806-4model-benchmark.md`](reports/138-semantic-val806-4model-benchmark.md):
  current benchmark table.
- [`reports/140-ae-path-reranker-bootstrap.md`](reports/140-ae-path-reranker-bootstrap.md):
  AE-path selector/reranker result.
- [`docs/WORKSPACE_ARCHITECTURE.md`](docs/WORKSPACE_ARCHITECTURE.md):
  local workspace and data layout.
- [`docs/DISTILLATION_ARCHITECTURE.md`](docs/DISTILLATION_ARCHITECTURE.md):
  teacher/student/FLEX/AE/Q2 architecture.

## Artifact And Git Policy

Do not commit large generated artifacts by default:

```text
outputs/*
data/raw/*
data/processed/*
data/corpus/*
data/eval/*
logs/*
```

Before pushing, check for untracked generated directories. In the current local
workspace, newer artifact roots such as `outputs/benchmarks`, `outputs/eval`,
`outputs/reranker`, and `data/vqa_q2_stepa*` may need explicit ignore rules or
manual exclusion from commits.
