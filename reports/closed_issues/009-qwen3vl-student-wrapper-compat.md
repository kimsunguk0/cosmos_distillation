# Qwen3VL Student Wrapper Compat

Status: resolved

## Symptom

The student model could not be loaded through the original wrapper because local `Cosmos-Reason2-2B` weights are not a plain `AutoModelForCausalLM` target.

## Root Cause

The local student checkpoint is a `Qwen3VL` family model, so the old wrapper assumed the wrong class and the wrong output access pattern.

## Resolution

The student wrapper was patched to:

- inspect the model config first
- route `qwen3_vl` models through `Qwen3VLForConditionalGeneration` or `AutoModelForVision2Seq`
- read logits and hidden states from the correct nested outputs when needed

## Key Files

- `src/model/student_wrapper.py`

## Outcome

Smoke training now works against the local `Cosmos-Reason2-2B` checkpoint.
