# Teacher Cache Provenance And Quality Weighting

Status: in_progress

## Symptom

- Normalized teacher outputs were usable for training, but there was no reliable way to tell:
  - whether a field came from direct generation or postprocess recovery
  - whether slot prompts collapsed back into the `cot` channel
  - how aggressively a sample should be downweighted during distillation

## Root Cause

- The original cache schema only stored the final normalized text fields.
- Downstream code had no provenance-aware weighting mechanism.
- Hidden/logit cache paths could silently become stale after teacher text regeneration.

## Resolution

- Added per-field provenance fields to teacher cache:
  - `teacher_long_cot_source`
  - `teacher_long_cot_direct`
  - `teacher_short_reason_source`
  - `teacher_short_reason_direct`
  - `teacher_meta_action_source`
  - `teacher_meta_action_direct`
  - `teacher_answer_source`
  - `teacher_answer_direct`
- Added slot-channel diagnostics:
  - `slot_channel_behavior`
  - `channel_collapse_detected`
  - `all_text_from_same_channel`
- Added quality annotations:
  - `teacher_text_quality`
  - `teacher_structured_quality`
  - `teacher_direct_slot_reliability`
  - `teacher_answer_short_reason_overlap`
  - `teacher_quality_multiplier`
- Added signal-cache provenance:
  - `teacher_signal_target_field`
  - `teacher_signal_target_source`
  - `teacher_signal_cache_stale`
- Updated corpus building so seq/logit/feat/rank weights are multiplied by `teacher_quality_multiplier`.

## Current Outcome

- Legacy recovered outputs now carry explicit low-reliability provenance instead of silently looking “clean”.
- Corpus records now preserve the soft-target provenance and quality metadata needed for ablations and error analysis.
- Teacher text overwrite now invalidates stale hidden/logit cache pointers when the normalized target text changes.

## Follow-up

- Let the full teacher overwrite finish on all currently ready samples.
- Re-run teacher signal cache extraction on stale records.
- Rebuild corpus and re-run refreshed distillation once the updated teacher cache is stable.
