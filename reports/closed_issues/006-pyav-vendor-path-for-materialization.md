# PyAV Vendor Path For Materialization

Status: resolved

## Symptom

Canonical sample materialization created sample directories, but frame decoding failed and many samples had no JPG outputs.

## Root Cause

`PyAV` had been vendored into `.vendor`, but the active materialization runtime was not loading that path, so frame extraction behaved as if `PyAV` was missing.

## Resolution

Materialization was rerun with the vendor path exposed through `PYTHONPATH`, allowing the runtime to import `av` correctly.

## Key Files

- `.vendor/av`
- `scripts/03_materialize_canonical_samples.py`

## Outcome

Decoded-frame sample count started increasing from the old smoke subset into the larger ready subset.

## Follow-up

If this pipeline is used from multiple shells or launchers, standardize the vendor path in the launcher environment rather than relying on ad hoc command prefixes.
