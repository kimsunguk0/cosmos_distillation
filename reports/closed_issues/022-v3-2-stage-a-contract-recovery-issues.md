# 022 V3.2 Stage A Contract Recovery Issues

- status: in_progress
- scope:
  - summarize the main issues encountered while bringing the `cosmos_distillation_spec_v3_2` pipeline to a usable `Stage A` baseline
  - focus on the concrete symptoms, verified root causes, fixes, and remaining gaps
  - cover the period from the first broken `Stage A` reruns through the current `A1 -> A2` warmup/refine flow

## Bottom line

- the main `Stage A` failure was not a single bug
- it was a stack of issues:
  - corrupted supervision labels from prompt/full template mismatch
  - frozen custom token rows under LoRA
  - no explicit pressure on boundary-format tokens
  - weak held-out trajectory decoding despite improved train loss
- the pipeline is now mechanically much healthier:
  - `Stage A` loss curves are real
  - `<|cot_end|>`, `<|traj_future_start|>`, and `<|traj_future_end|>` are now emitted reliably
  - full-data held-out trajectory generation still collapses too much, so `Stage B` has not started yet

## Issue 1. Prompt/full chat template mismatch corrupted supervision labels

symptom:
- the first `Stage A` run showed improving loss, but generated outputs never emitted:
  - `<|cot_end|>`
  - `<|traj_future_start|>`
  - trajectory body tokens
- direct inspection showed that the supervised completion span was not aligned with the intended assistant continuation

root cause:
- `src/training/collator.py` encoded prompt-only and prompt+completion messages with different effective chat-template prefixes
- the prompt batch closed the assistant turn with `<|im_end|>`
- the full batch inserted completion text before that close token
- `_labels_from_prompt_and_full(...)` masked labels using the prompt length, which cut into the true completion span

resolution:
- updated prompt/full message encoding to use `continue_final_message=True` consistently
- added explicit prefix-equality validation before label masking
- added regression tests for:
  - completion tokens remaining trainable
  - prompt/full mismatch raising an error

key files:
- `src/training/collator.py`
- `tests/unit/test_collator.py`

current status:
- resolved
- supervised labels now include the intended `cot_end + traj span` contract

## Issue 2. New special/traj token rows were frozen under LoRA

symptom:
- after the label-alignment fix, free-text CoT quality improved, but the model still refused to emit:
  - `<|cot_end|>`
  - `<|traj_future_start|>`
  - `<i...>` trajectory tokens
- small smoke runs kept producing plain text or `<tool_call>`-like junk instead of structured spans

root cause:
- the tokenizer had been expanded with:
  - custom boundary tokens
  - `4000` discrete trajectory tokens
- but LoRA training only updated adapter layers and did not update the newly added embedding/output rows
- this meant the student could consume those tokens in labels but had almost no trainable path to reliably emit them

resolution:
- added `distill_trainable_token_ids(...)` to collect control-token ids plus the full trajectory-token range
- passed those ids into PEFT `trainable_token_indices`
- verified that LoRA now creates trainable token adapters for the expanded vocabulary

key files:
- `src/model/tokenizer_ext.py`
- `src/model/peft_setup.py`
- `scripts/09_train_distill.py`
- `scripts/15_infer_student_smoke.py`
- `scripts/16_profile_prefill_generation.py`

current status:
- resolved
- custom token rows are now trainable during LoRA runs

## Issue 3. Compact LoRA checkpoints were not being saved correctly

symptom:
- earlier checkpoints became unexpectedly large
- a LoRA run produced roughly `8 GB` student checkpoints even though the base weights were much smaller and only adapters were intended to change

root cause:
- the training path was saving full `state_dict`s in `fp32`
- this preserved the whole backbone instead of saving adapter-only artifacts plus the auxiliary head

resolution:
- added compact checkpoint IO helpers
- LoRA runs now save:
  - adapter weights
  - `meta_action_head`
  - manifest metadata
- inference/profile scripts were updated to load the new format

key files:
- `src/model/checkpoint_io.py`
- `scripts/09_train_distill.py`
- `scripts/15_infer_student_smoke.py`
- `scripts/16_profile_prefill_generation.py`

current status:
- resolved
- compact adapter checkpoints are now the default LoRA artifact format

## Issue 4. `Stage A` still learned text before it learned the output contract

symptom:
- once label alignment and trainable token rows were fixed, `Stage A` loss improved significantly
- however held-out inference still preferred:
  - plain-text CoT only
  - or malformed boundary-token repetition
- the model was learning the natural-language part of the answer faster than the structural contract

root cause:
- the original `Stage A` weighting did not provide enough explicit pressure on:
  - `<|cot_end|>`
  - `<|traj_future_start|>`
  - `<|traj_future_end|>`
- boundary tokens were too weak relative to the rest of the sequence objective

resolution:
- added `output_format_loss` / `format_ce`
- created a dedicated mask for structure tokens inside the supervised completion span
- updated `Stage A` configs to put explicit weight on the output-format contract

key files:
- `src/training/collator.py`
- `src/training/losses.py`
- `src/training/trainer.py`
- `configs/train/stage_a.yaml`
- `configs/train/stage_a_contract.yaml`

current status:
- partially resolved
- full-data `Stage A` now emits all required boundary tags on held-out samples
- trajectory body quality is still not good enough

## Issue 5. CLI learning rate override masked stage-specific schedule tuning

symptom:
- some experiments appeared less effective than expected even after config changes
- stage configs declared different intended learning rates, but actual runs kept behaving as if they used the same global default

root cause:
- `scripts/09_train_distill.py` always supplied a non-`None` CLI default learning rate
- this effectively overrode YAML stage-config learning rates even when the user did not intend to override them

resolution:
- changed CLI `--learning-rate` default to `None`
- only override the stage config when the CLI flag is explicitly passed
- set stage-specific learning rates in:
  - `stage_a.yaml`
  - `stage_b.yaml`
  - `stage_c.yaml`
  - `stage_a1_warmup.yaml`
  - `stage_a2_refine.yaml`

key files:
- `scripts/09_train_distill.py`
- `configs/train/stage_a.yaml`
- `configs/train/stage_b.yaml`
- `configs/train/stage_c.yaml`
- `configs/train/stage_a1_warmup.yaml`
- `configs/train/stage_a2_refine.yaml`

current status:
- resolved

## Issue 6. Full-data `Stage A` recovered the tags but not the held-out trajectory body

symptom:
- after the format-loss and LR fixes, full-data `Stage A` improved strongly:
  - val `total_loss` dropped sharply
  - val `output_format_loss` dropped near zero
- held-out generation now emitted:
  - `<|cot_end|>`
  - `<|traj_future_start|>`
  - `<|traj_future_end|>`
- but the actual `<i...>` trajectory body either stayed empty or collapsed into a tiny repetitive pattern

root cause:
- optimization was still not strong enough to generalize trajectory-body emission from the full dataset in a single pass
- overfit probes proved the model family and current pipeline *can* emit the full structured sequence, so this was not a fundamental architecture block
- the remaining problem was curriculum/generalization, not token availability

resolution:
- introduced a two-stage `Stage A` recovery flow:
  - `A1 warmup`: `traj + format` heavy, teacher losses off
  - `A2 refine`: reintroduce modest teacher pressure after the contract is stabilized
- added dedicated configs and updated the run script to follow:
  - `Stage A1 -> Stage A2 -> Stage B`

key files:
- `configs/train/stage_a1_warmup.yaml`
- `configs/train/stage_a2_refine.yaml`
- `scripts/17_run_distill_v3_2.sh`

artifacts:
- `outputs/reports/stage_a1_v3_2.json`
- `outputs/reports/stage_a2_v3_2.json`

current status:
- in progress
- this flow improved held-out trajectory metrics:
  - `Stage A1` val `traj_token_acc ~= 0.0158`
  - `Stage A2` val `traj_token_acc ~= 0.0299`
- however held-out trajectory tokens still collapse too much to call `Stage A` complete

## Issue 7. Held-out decoding needed contract-aware constraints

symptom:
- unconstrained greedy decoding on held-out data often produced:
  - repeated boundary tokens
  - premature `traj_end`
  - malformed spans
- even when the model had partially learned the contract, the raw decoder did not enforce the sequence shape

root cause:
- generation had no knowledge of the intended post-`cot_end` contract:
  - first emit `traj_start`
  - then emit exactly `128` trajectory tokens
  - then emit `traj_end`

resolution:
- added constrained decoding helpers:
  - `TrajSpanLogitsProcessor`
  - `StopOnTrajEndCriteria`
- the decoder now forces:
  - `traj_start` immediately after `cot_end`
  - only `<i...>` tokens during the trajectory body
  - `traj_end` only after the requested token count is reached

key files:
- `src/inference/decoding.py`
- `scripts/15_infer_student_smoke.py`
- `tests/unit/test_decoding.py`

current status:
- resolved as a decoding-side guard
- this does not solve model collapse by itself, but it turns malformed boundary spam into usable fixed-length trajectory spans and makes held-out diagnosis much more honest

## Current best factual state

- best current `Stage A` checkpoint:
  - `outputs/checkpoints/stage_a2_v3_2/final`
- strongest full-data held-out metrics so far:
  - `outputs/reports/stage_a2_v3_2.json`
- strongest constrained held-out decode artifact so far:
  - `outputs/reports/stage_a2_v3_2_infer_val10_constrained.json`

What is already true now:

- the student reliably emits:
  - CoT text
  - `<|cot_end|>`
  - `<|traj_future_start|>`
  - fixed-length trajectory-token spans
  - `<|traj_future_end|>`

What is not yet true:

- held-out trajectory-body diversity is still too low
- constrained decoding currently reveals a narrow trajectory-token band rather than a rich sample-specific path
- because of that, `Stage B` has not been launched yet

## Recommended next step

- continue from `stage_a2_v3_2/final`
- run one more trajectory-focused continuation before enabling `Stage B`
- judge success on:
  - held-out `traj_token_acc`
  - held-out trajectory-token diversity under constrained decoding
  - non-collapsed sample-specific trajectory body patterns

## Key files and artifacts

- `src/training/collator.py`
- `src/training/losses.py`
- `src/training/trainer.py`
- `src/model/peft_setup.py`
- `src/model/tokenizer_ext.py`
- `src/inference/decoding.py`
- `scripts/09_train_distill.py`
- `scripts/15_infer_student_smoke.py`
- `scripts/17_run_distill_v3_2.sh`
- `configs/train/stage_a.yaml`
- `configs/train/stage_a1_warmup.yaml`
- `configs/train/stage_a2_refine.yaml`
- `outputs/reports/stage_a_v3_2_format_lr.json`
- `outputs/reports/stage_a1_v3_2.json`
- `outputs/reports/stage_a2_v3_2.json`
- `outputs/reports/stage_a1_v3_2_infer_val10_constrained.json`
- `outputs/reports/stage_a2_v3_2_infer_val10_constrained.json`
