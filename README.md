# Cosmos Distillation

This workspace prepares a v1 distillation pipeline from `nvidia/Alpamayo-1.5-10B`
into `nvidia/Cosmos-Reason2-2B` using the PhysicalAI AV reasoning subset.

The project follows the spec in:

- `../codex_alpamayo15_reason2_2b_distillation_spec_v2.md`

## Fixed v1 rules

- Teacher outputs are soft targets, not ground truth.
- Human reasoning and observed future path are hard data.
- Teacher reasoning must not be paired with GT path by default.
- Human reasoning must not be paired with teacher trajectory by default.
- v1 is text-reasoning backbone distillation only.
- Trajectory and planning distillation stay off by default until v2.

## Planned workflow

1. Environment and access smoke test
2. Metadata-only download
3. Clip manifest build
4. Canonical sample materialization
5. Provenance-separated supervision records
6. Teacher text cache
7. Consistency gate
8. Safe multitask corpus
9. Student wrapper and tokenizer extension
10. v1 training, evaluation, and reports

## Directory map

- `docs/`: working notes aligned to the spec
- `configs/`: YAML configs for data, model, and train stages
- `scripts/`: entrypoints for each work package
- `src/`: reusable library code
- `inputs/`: user-provided clip selections and generated manifests
- `data/`: local pipeline artifacts
- `outputs/`: checkpoints, reports, and debug bundles

## Immediate next steps

1. Fill in `scripts/00_env_check.sh`
2. Implement metadata fetch in `src/data/hf_download.py`
3. Build the first clip manifest in `scripts/02_build_clip_manifest.py`
4. Keep provenance separation enforced in the corpus builder

## Known Limitation

- v1 currently follows the spec's clip-level anchor design: one `clip_uuid__anchor0`
  sample per clip, even when the clip contains multiple parsed reasoning events.
- This is spec-compliant, but it can weaken input-target alignment on multi-event clips
  because the canonical visual window is centered on one representative `t0`.
- We keep `num_events`, `parsed_events_json`, and `keyframe_timestamps_us_json` in the
  manifest so event-level ablations can be added later without losing the original
  event structure.
