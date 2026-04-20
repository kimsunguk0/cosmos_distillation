# 020 Spec Design Issues So Far

- status: in_progress
- scope:
  - summary of the main issues discovered while implementing the `codex_alpamayo15_reason2_2b_distillation_spec_v2` design
  - focuses on what failed, why it failed, what was fixed, and what remains open

## Bottom line

- the overall direction is still valid:
  - `Cosmos-Reason2-2B` backbone distillation before expert-stage work is a reasonable `v1` goal
- the main implementation mistake was not the existence of distillation itself
- the main implementation mistake was assuming a richer teacher contract than the current Alpamayo text path can honestly support

## What did not work

- the original strict split target `750 / 200 / 50` was not achievable
  - only `732` strict human CoC samples were actually available from the released reasoning subset
  - this forced a smaller strict subset

- canonical local-frame construction was wrong
  - local coordinates were anchored to the last future pose instead of true `t0`
  - this could distort history/future semantics, weak GT path labels, and consistency evaluation

- teacher runtime bundles were too permissive
  - future-path and human-derived information lived too close to the runtime payload
  - inference code did not leak at execution time, but the storage layout left a future leakage footgun

- weak GT semantics existed only partially
  - path semantics extraction logic existed, but supervision records did not consistently carry the derived weak GT labels needed for grounded selection and metrics

- teacher selection was too syntax-heavy
  - early selection favored completeness-like properties such as “field exists”, parseability, and slot count
  - grounded agreement with human CoC, weak GT path, and internal consistency was too weak in candidate selection

- the first distillation runs were not actually using the intended KD signals
  - stale corpus and stale signal wiring caused `seq_kd`, `logit_kd`, and `self_cons` to be zero or near-zero in early runs
  - one early full run also failed to save reusable checkpoints

- the biggest teacher-side issue is still unresolved:
  - direct `meta_action` and reliable direct `answer` generation are not yet proven with the current Alpamayo text route
  - many teacher samples collapsed into `long_cot_only_fallback`

## Why it failed

- the implementation over-assumed native teacher output richness
  - our initial teacher design treated `short_reason`, `answer`, `meta_action`, and `structured_json` as if they were all stable native teacher channels
  - the current Alpamayo local weights and official text routes do not support that assumption robustly

- the official Alpamayo text path is narrower than we initially treated it
  - `create_message()` is the native reasoning path through `<|cot_start|>`
  - `create_vqa_message()` is the native VQA path through `<|answer_start|>`
  - in practice, direct `cot` is the only consistently trustworthy text teacher signal seen so far
  - `meta_action` direct-slot behavior is still unproven in our current local setup
  - direct `answer` remains unstable enough that it cannot yet be treated as a primary teacher target

- prompt-level fallback hid the real problem for too long
  - once direct slots failed, normalization and fallback logic were synthesizing plausible-looking derived fields
  - this protected the pipeline from crashing, but also made weak teacher samples look richer than they really were

- the first pipeline passes mixed correctness work with contract work
  - we fixed data correctness, KD wiring, checkpoint saving, and teacher contract issues in overlapping waves
  - that made it harder to tell whether failures were due to data, training, or teacher signal quality

## What was fixed

- strict-subset reality was acknowledged and documented
  - the pipeline now works against the actually available strict human CoC subset rather than the original idealized count

- `t0` localization was fixed
  - canonicalization now uses an explicit `t0` anchor
  - canonical samples store both world-space and `local@t0` trajectories
  - regression tests were added

- runtime and diagnostics bundles were split
  - runtime bundle now contains only safe inference-time inputs
  - diagnostics carry future-path and human/weak-derived reference data
  - runtime validators reject forbidden keys

- weak GT semantics were wired into supervision and consistency
  - GT-path-derived action/turn/stop semantics now feed downstream gating and reporting

- grounded teacher selection was strengthened
  - groundedness terms now include human agreement, weak GT agreement, hallucination penalties, and internal consistency

- sparse true-KD wiring was repaired
  - top-k teacher signal extraction now aligns against actual target token positions
  - out-of-range teacher token ids are filtered
  - `seq_kd`, `logit_kd`, and `self_cons` were verified nonzero in smoke runs

- checkpoint saving was added
  - train runs now save restorable artifacts instead of metrics only

## What remains open

- the honest teacher contract is still narrower than the original spec intent
  - right now the trustworthy contract is:
    - primary: `teacher_long_cot`
    - auxiliary: derived `teacher_meta_action`
    - probe-only / not yet trusted as primary: direct `teacher_answer`
    - unsupported for now: `structured_json`

- teacher quality is still the main bottleneck
  - many samples remain `long_cot_only_fallback`
  - direct slot reliability is still low
  - this means the current pipeline is mechanically correct, but the resulting student quality is still limited by teacher quality

- feature KD remains intentionally off
  - teacher `8B` and student `2B` hidden-space alignment still lacks a clear projection/layer contract
  - hidden caches can be extracted, but `feat_align` should remain disabled until that contract is defined

- clip-level manifest remains a known limitation
  - this is a spec choice, not an implementation bug
  - multi-event clips still collapse to one canonical anchor
  - event-level ablation remains future work

## Current honest assessment of distillation

- “is distillation the right approach?”:
  - yes, in principle
  - but only if the teacher contract matches what Alpamayo can actually and reliably emit

- “is the current multi-slot teacher fully working?”:
  - no
  - the current implementation has become more honest, but the teacher is still not rich enough to justify strong `answer/meta_action/structured` supervision at full weight

- “what is the safe path right now?”:
  - use `teacher_long_cot` as the primary backbone teacher signal
  - keep `teacher_meta_action` as a derived auxiliary
  - only promote `teacher_answer` if direct-answer probes succeed on a meaningful smoke subset

## Key files or artifacts

- `src/data/canonicalize.py`
- `src/data/local_dataset.py`
- `src/data/path_semantics.py`
- `src/data/supervision.py`
- `src/data/teacher_cache.py`
- `src/data/teacher_prompts.py`
- `src/training/collator.py`
- `scripts/05_generate_teacher_text_cache.py`
- `scripts/05_run_teacher_text_inference.py`
- `scripts/05_extract_teacher_signal_cache.py`
- `scripts/08_build_multitask_corpus.py`
- `scripts/09_train_distill.py`

## Follow-up

- finish the honest Alpamayo teacher probe over a smoke subset
- decide whether direct `answer` survives enough to keep as a real auxiliary target
- if not, collapse the contract fully to `long_cot primary + derived meta_action auxiliary`
- only rerun larger distillation after the teacher contract is proven on smoke samples
