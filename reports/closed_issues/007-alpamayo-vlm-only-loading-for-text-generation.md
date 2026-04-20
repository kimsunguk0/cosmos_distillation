# Alpamayo VLM Only Loading For Text Generation

Status: resolved with limitation

## Symptom

Initial Alpamayo text-generation attempts failed when loading from the local weight directory.

## Root Cause

Two separate issues were involved:

- loading against the plain `config.json` path did not provide the full `Alpamayo1_5` action-space config
- `sdpa` attention was not supported for this architecture in the active `transformers` path

## Resolution

Text generation was made to work by:

- explicitly loading `alpamayo_1.5_config.json`
- using `attn_implementation=\"eager\"`
- accepting the local weight pack as a VLM-only text-generation path

## Key Files

- `scripts/05_run_teacher_text_inference.py`
- `/home/pm97/workspace/sukim/weights/alpamayo15_vlm_weights/alpamayo_1.5_config.json`

## Limitation

The local weight pack reports many uninitialized expert-side weights, so this path should be treated as suitable for text generation, not for full trajectory-side inference.
