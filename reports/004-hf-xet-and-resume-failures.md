# HF Xet And Resume Failures

Status: resolved

## Symptom

Some download runs failed mid-transfer with Hugging Face Xet refresh or remote disconnect errors.

## Root Cause

The transfer path using Xet was unstable for the active session and long-running dataset pull.

## Resolution

The resume strategy was changed:

- fall back away from the unstable Xet path when needed
- restart from the remaining subset instead of restarting the full plan
- use narrower resumes such as `egomotion-only` when that was the real blocker

## Key Artifacts

- `outputs/logs/priority_resume_no_xet.log`
- `outputs/logs/priority_resume_egomotion_only.log`

## Follow-up

If long downloads continue to be unstable, keep splitting recovery runs by missing feature type instead of retrying the whole batch.
