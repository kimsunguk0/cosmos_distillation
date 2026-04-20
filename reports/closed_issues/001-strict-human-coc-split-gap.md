# Strict Human CoC Split Gap

Status: resolved

## Symptom

The original target split of `750 train / 200 val / 50 test` could not be built under a strict human chain-of-causation policy.

## Root Cause

The available official reasoning metadata only exposed `ood_reasoning.parquet`, and the number of clips with actual strict human CoC was lower than the requested target.

## Resolution

The strict subset was rebuilt to the maximum valid size:

- `577 train`
- `105 val`
- `50 test`

This keeps the total at `732` while preserving the strict human CoC requirement.

## Key Artifacts

- `data/processed/strict_download_subset_manifest.parquet`
- `outputs/reports/strict_download_subset_summary.json`

## Follow-up

If a larger split is required later, we need either:

- a non-strict policy with additional weak or teacher-only supervision, or
- new human reasoning data not present in the current official metadata.
