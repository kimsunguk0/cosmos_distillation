# Missing Front Wide Chunk 1086

Status: resolved

## Symptom

The original `priority 100 chunk` target was stuck at `99 / 100` completed chunks because one `front_wide` chunk was still missing.

## Root Cause

`camera_front_wide_120fov.chunk_1086.zip` was not available as a completed final zip even though the underlying partial download had already finished.

## Resolution

The partial file was manually validated:

- confirmed expected byte size
- confirmed zip integrity
- promoted the `.zip.part` artifact into the final zip

## Outcome

This closed the original `priority 100 chunk` target and unlocked the full `214 / 214` sample potential for that batch.

## Follow-up

When a single missing chunk is blocking a whole ready subset, inspect partial artifacts before scheduling another long download.
