# Teacher Slot Prompt Recovery

Status: in_progress

## Symptom

- Alpamayo teacher raw outputs frequently contained only `cot`.
- `meta_action` was often blank.
- `answer` was often blank or fell back to the same string as `long_cot`.
- Existing cache statistics before the fix:
  - `ok=106`
  - `meta_action missing=50`
  - `answer missing=0` in normalized index, but mostly due to fallback
  - `short_reason == long_cot` for all `106`

## Root Cause

- Most cached teacher outputs were legacy `cot`-only generations.
- The previous structured prompts leaned on JSON or multi-slot generation that the current local Alpamayo VLM checkpoint did not follow reliably.
- The model strongly preferred writing free-form reasoning into the `cot` channel, even for answer-oriented prompts.

## Resolution

- Replaced request bundle prompts with slot-focused prompts:
  - `long_cot_v1`
  - `short_reason_only_v2`
  - `meta_action_only_v2`
  - `answer_only_v2`
- Updated teacher inference to:
  - rebuild messages without the old default “future trajectory” tail
  - use dedicated slot prompts
  - read useful fallback text from `cot` when the model ignores the nominal slot
  - normalize `meta_action` back to the allowed action-label set
  - keep `teacher_hidden_path` and `teacher_logit_cache_path` when overwriting cache rows
- Added a tie-break so `short_reason` can diverge from `long_cot` when a shorter answer is available.

## Validation

- Single-sample overwrite test on `000ba013-9eb4-45ca-8e86-93fdc68c37e2__anchor0` now yields:
  - `teacher_long_cot = "Stop to yield to cross traffic"`
  - `teacher_short_reason = "Stop to yield to cross-traffic"`
  - `teacher_meta_action = "yield"`
  - `teacher_answer = "Stop to yield to cross-traffic"`
- This is still not perfect, but it is materially better than the previous “all fields collapse to the same legacy CoT or remain blank” behavior.

## Follow-up

- The full overwrite pass for current ready samples is running.
- If Alpamayo continues to route almost everything through `cot`, the next refinement is prompt-specific decoding or direct token-level generation analysis rather than further JSON prompting.
