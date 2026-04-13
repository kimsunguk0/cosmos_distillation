# Teacher Cache Refresh Overwrite

Status: resolved

## Symptom

After teacher outputs had already been generated, rerunning the teacher request-bundle builder could overwrite the index and drop existing `ok` statuses.

## Root Cause

`scripts/05_generate_teacher_text_cache.py` rebuilt `index.jsonl` from scratch without merging previously generated outputs back in.

## Resolution

The script was patched to recover generated outputs from either:

- the existing teacher index, or
- `data/teacher_cache/text/outputs/*.teacher_raw.json`

Recovered entries preserve `ok` status and their normalized output fields.

## Key Files

- `scripts/05_generate_teacher_text_cache.py`
- `data/teacher_cache/text/index.jsonl`
- `data/teacher_cache/text/outputs/*.teacher_raw.json`

## Follow-up

Any future cache-refresh step should be merge-safe by default.
