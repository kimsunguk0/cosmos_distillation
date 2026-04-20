# Download Scope And Priority Strategy

Status: resolved

## Symptom

A full strict-subset raw download implied multi-terabyte transfer and delayed teacher-ready samples too much.

## Root Cause

The initial plan optimized for eventual completeness, not for early chunk completion and fast sample usability.

## Resolution

The download strategy was changed in stages:

1. stop broad all-chunk style downloading
2. prioritize chunk completion over feature-by-feature growth
3. finish a `priority 100 chunk` batch first
4. launch an additional `next 40 chunk` batch in parallel

This raised teacher-ready samples much faster than broad downloading.

## Key Artifacts

- `scripts/03_download_priority_chunks.py`
- `outputs/reports/priority_download_plan_100_run.json`
- `outputs/reports/priority_next_40_run.json`
- `outputs/logs/priority_next_40.log`

## Follow-up

Keep using chunk-completion-first planning when immediate teacher generation matters more than raw total bytes downloaded.
