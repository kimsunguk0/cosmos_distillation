# Teacher Text Fields Mostly Cot Only

Status: open

## Symptom

Teacher text generation currently produces strong `cot` text, but `meta_action` and `answer` are often empty in the normalized output.

## Root Cause

The current generation path uses the chain-of-thought style prompt path first, which is good for reasoning text but not yet optimized for structured `meta_action` and `answer` extraction.

## Current Mitigation

The pipeline normalizes:

- `teacher_long_cot <- cot`
- `teacher_short_reason <- cot or answer`

This keeps the distillation corpus usable even when `meta_action` and `answer` are blank.

## Evidence

- `outputs/reports/eval_summary_77.json`

## Follow-up

Add a second-pass teacher generation path dedicated to structured answer extraction, likely using a VQA-style or JSON-specific prompt and a separate normalization rule.
