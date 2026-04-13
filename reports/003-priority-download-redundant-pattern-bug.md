# Priority Download Redundant Pattern Bug

Status: resolved

## Symptom

The priority downloader kept scheduling files that already existed locally, especially on `front_wide`.

## Root Cause

`remaining_patterns` was being populated without first checking whether the destination zip already existed or was already linked locally.

## Resolution

The downloader was patched to skip a pattern when the destination zip already exists locally.

## Key Files

- `scripts/03_download_priority_chunks.py`

## Outcome

This removed redundant requests and made the remaining download estimates much closer to reality.

## Follow-up

Keep local-existence checks at the point where remaining download patterns are computed, not only at transfer time.
