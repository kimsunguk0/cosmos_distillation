# 029 Teacher Traj Cache Design

## Status

- proposed
- not implemented yet

## Problem

We want to try a `teacher-primary` trajectory stage because the current `GT traj` objective still collapses in multi-sample training.

The important catch is that the current teacher path in this repo is **not** a real trajectory-teacher path yet:

- [scripts/05_extract_teacher_signal_cache.py](/workspace/cosmos_distillation/scripts/05_extract_teacher_signal_cache.py:129) builds a text-only teacher completion of the form `<|cot_start|>...<|cot_end|>`.
- [scripts/08_build_multitask_corpus.py](/workspace/cosmos_distillation/scripts/08_build_multitask_corpus.py:233) only wires `teacher_long_cot` text fields plus `topk_logits_path` / `pooled_hidden_path`.
- [src/training/collator.py](/workspace/cosmos_distillation/src/training/collator.py:398) pads teacher top-k against `cot_content_positions`, not trajectory positions.
- [src/training/trainer.py](/workspace/cosmos_distillation/src/training/trainer.py:168) computes `teacher_traj_ce`, but today that is still CE on the teacher-view prompt with GT trajectory labels, not CE against teacher-generated trajectory tokens.

## Critical Constraint

The currently used `alpamayo1.5` inference path does **not** directly expose trajectory token logits after `<|traj_future_start|>`.

- [alpamayo1_5/helper.py](/workspace/alpamayo_repos/alpamayo1.5/src/alpamayo1_5/helper.py:77) builds a prompt that asks for CoT and then future trajectory.
- But [alpamayo1_5/models/alpamayo1_5.py](/workspace/alpamayo_repos/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py:258) stops the VLM rollout at `<|traj_future_start|>`.
- After that, the action expert / diffusion path takes over at [alpamayo1_5/models/alpamayo1_5.py](/workspace/alpamayo_repos/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py:327).
- `return_extra=True` currently only returns extracted text tokens, not raw trajectory token logits, at [alpamayo1_5/models/alpamayo1_5.py](/workspace/alpamayo_repos/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py:392).

So:

- `teacher CoT top-k / hidden`: directly available from the current teacher stack
- `teacher traj token / top-k / hidden`: **not directly available from the current 1.5 rollout**

## Recommended Teacher Strategy

Use **split teachers by signal type**.

### Teacher A: reasoning teacher

Keep the current `alpamayo1.5` teacher for:

- `teacher CoT text`
- `teacher CoT top-k logits`
- `teacher CoT hidden`

This path is already aligned with the current cache and corpus format.

### Teacher B: trajectory teacher

Use a **Stage-1 discrete AR Alpamayo teacher** for real trajectory-token supervision.

Reason:

- [alpamayo_base SFT base model](/workspace/alpamayo_repos/alpamayo_base/finetune/sft/models/sft_base_model.py:425) does full autoregressive generation.
- It actually emits discrete trajectory tokens and extracts them with [extract_traj_tokens](/workspace/alpamayo_repos/alpamayo_base/finetune/sft/models/sft_base_model.py:441).
- The extraction logic for the trajectory span already exists at [alpamayo token_utils](/workspace/alpamayo_repos/alpamayo1.5/src/alpamayo1_5/models/token_utils.py:29).

This is the clean way to get:

- teacher trajectory token sequence
- teacher trajectory top-k logits
- teacher trajectory span hidden states

## Core Design

Use a **2-pass teacher trajectory cache pipeline**.

### Pass 1: rollout and candidate selection

New script:

- `scripts/06_extract_teacher_traj_signal_cache.py`

Inputs:

- materialized sample images
- ego history
- route if present
- Stage-1 discrete trajectory teacher weights

Procedure:

1. Build the same driving prompt contract as Alpamayo.
2. Run teacher autoregressive generation with:
   - `output_logits=True`
   - `return_dict_in_generate=True`
3. Save raw generated completion tokens.
4. Extract:
   - `teacher_cot_text`
   - `teacher_traj_token_ids`
   - decoded `teacher_future_xyz`
5. If multiple samples are generated, choose **one canonical teacher candidate** per sample.

Candidate selection should be offline and GT-aware:

- exact trajectory token count must match expected count
- reject obvious plateau collapse
- prefer coarse motion agreement with GT / human label
- then prefer best short-horizon geometry agreement

Recommended first-pass selection priority:

1. correct token count
2. coarse motion agreement
3. lowest short-horizon ADE/L1
4. lower repeated-token run

### Pass 2: teacher-forced aligned cache extraction

After selecting the canonical teacher trajectory sample, rerun a **teacher-forced forward** on:

- prompt
- `teacher_cot_text`
- `teacher_traj_token_ids`

This pass is for alignment, not generation.

Reason:

- generation outputs are awkward to align to exact student training positions
- a teacher-forced forward gives stable per-position logits and hidden states for the chosen completion

Full completion should be serialized as:

`<|cot_start|>{teacher_cot_text}<|cot_end|><|traj_future_start|><i...><|traj_future_end|>`

using the same discrete token string contract as [src/utils/traj_tokens.py](/workspace/cosmos_distillation/src/utils/traj_tokens.py:8).

Outputs from pass 2:

- per-token `teacher_traj_topk_indices`
- per-token `teacher_traj_topk_logprobs`
- per-token `teacher_traj_hidden`
- optional pooled trajectory hidden
- canonical `teacher_traj_token_ids`

## Cache Schema

Per sample under `data/teacher_cache/traj/`:

- `teacher_traj_tokens.npy`
  - canonical generated trajectory token ids
- `teacher_traj_topk.npz`
  - `topk_indices`
  - `topk_logits`
  - `target_token_count`
  - `positions` or `traj_token_count`
- `teacher_traj_hidden.npy` or `teacher_traj_hidden.npz`
  - either full `[T, H]` span hidden
  - or pooled `[H]`
- `teacher_traj_xyz.npy`
  - decoded continuous teacher future
- `teacher_traj_quality.json`
  - candidate selection metrics
  - short-horizon geometry score
  - coarse motion class
  - anti-collapse stats

Index JSONL should add fields like:

- `teacher_traj_tokens_path`
- `teacher_traj_topk_path`
- `teacher_traj_hidden_path`
- `teacher_traj_xyz_path`
- `teacher_traj_signal_ready`
- `teacher_traj_quality_path`
- `teacher_traj_source`

## Corpus Wiring

[scripts/08_build_multitask_corpus.py](/workspace/cosmos_distillation/scripts/08_build_multitask_corpus.py:210) should be extended so `teacher_target` has separate text and trajectory teacher slots.

Recommended `teacher_target` split:

- `cot_text`
- `cot_topk_logits_path`
- `cot_pooled_hidden_path`
- `traj_token_ids_path`
- `traj_topk_logits_path`
- `traj_hidden_path`
- `traj_xyz_path`
- `traj_signal_ready`
- `teacher_view_allowed`
- `teacher_traj_allowed`
- `teacher_quality_multiplier`

Do **not** overload the current CoT paths for trajectory data.

## Collator Wiring

[src/training/collator.py](/workspace/cosmos_distillation/src/training/collator.py:398) should gain a second padding branch for trajectory-teacher fields.

Add batch fields:

- `teacher_traj_token_ids`
- `teacher_traj_token_mask`
- `teacher_traj_topk_indices`
- `teacher_traj_topk_logprobs`
- `teacher_traj_topk_mask`
- `teacher_traj_hidden`
- `teacher_traj_hidden_mask`

Important:

- trajectory teacher signals must align against `traj_token_positions`, not `cot_content_positions`
- CoT teacher and trajectory teacher should stay separately addressable in the batch

## Trainer Wiring

[src/training/trainer.py](/workspace/cosmos_distillation/src/training/trainer.py:150) should use a new order of trajectory-teacher losses.

Recommended order:

1. `teacher_traj_hidden_align_loss`
2. `teacher_traj_topk_kd_loss`
3. optional `teacher_traj_seq_ce`

Reason:

- hidden align is the safest way to transfer planning structure
- top-k KD transfers local token uncertainty without going fully one-hot
- hard teacher trajectory CE is the most brittle and should be last

## Recommended First Pilot

Run a clean `A0-teacher-traj-only` pilot.

Training signal:

- `teacher_traj_hidden_align_loss`: on
- `teacher_traj_topk_kd_loss`: on, weak
- `teacher_traj_seq_ce`: off or very weak
- `GT traj CE`: off
- `GT xyz/final`: off

Use GT only for:

- teacher candidate gating
- offline evaluation

This gives the cleanest answer to:

“Can teacher trajectory distribution break the current GT-collapse mode?”

## Fallback If Stage-1 Discrete Teacher Is Unavailable

If we only have the public `alpamayo1.5` rollout teacher, then we should be honest:

- we can get `teacher CoT` signals directly
- we can get **continuous** teacher future trajectories from the diffusion expert
- we can quantize that continuous future into pseudo trajectory tokens
- we can possibly expose expert hidden states with a custom patch
- but we still do **not** get native trajectory-token top-k logits from the public 1.5 rollout path

So in that case the fallback design is:

- `teacher continuous traj xyz`
- quantized pseudo `teacher_traj_token_ids`
- optional `expert hidden`
- no real `teacher traj top-k`

That fallback is usable, but it is weaker than a true Stage-1 discrete trajectory teacher.

## Bottom Line

If we want real:

- `teacher traj token`
- `teacher traj top-k`
- `teacher traj hidden`

the best design is:

1. keep current `alpamayo1.5` teacher for CoT signals
2. add a Stage-1 discrete AR Alpamayo teacher for trajectory signals
3. extract trajectory teacher cache in two passes:
   - rollout + candidate selection
   - teacher-forced aligned cache extraction

That is the cleanest path to a real `teacher-primary` trajectory stage without pretending the current 1.5 rollout already exposes trajectory-token KD signals.
