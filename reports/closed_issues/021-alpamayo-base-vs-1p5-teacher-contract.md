# 021 Alpamayo Base vs 1.5 Teacher Contract

- status: in_progress
- purpose:
  - compare `alpamayo_base` and `alpamayo1.5` text-generation structures
  - decide what teacher contract is honest for backbone distillation

## Key comparison

### 1. `alpamayo_base` clearly contains a multi-component conversation system

- `alpamayo_base/src/alpamayo_r1/chat_template/conversation.py`
  - defines assistant-side components for:
    - `cot`
    - `meta_action`
    - `traj_future`
  - builds assistant output with explicit component boundaries
  - `construct_meta_action()` uses:
    - `<|meta_action_start|>`
    - `<|meta_action_end|>`

- `alpamayo_base/src/alpamayo_r1/models/base_model.py`
  - `SPECIAL_TOKENS_KEYS` explicitly includes:
    - `meta_action_start`
    - `meta_action_end`
    - `question_start`
    - `question_end`
    - `answer_start`
    - `answer_end`

- `alpamayo_base/src/alpamayo_r1/models/token_utils.py`
  - explicitly documents output of the form:
    - `...<|cot_end|><|meta_action_start|>...<|meta_action_end|><|traj_future_start|>...`
  - extracts `cot`, `meta_action`, `answer`

### 2. `alpamayo1.5` public helper path is narrower

- `alpamayo1.5/src/alpamayo1_5/helper.py`
  - `create_message()` exposes the official reasoning route through `<|cot_start|>`
  - `create_vqa_message()` exposes the official VQA route through `<|question_start|>...<|question_end|>` and `<|answer_start|>`

- `alpamayo1.5/src/alpamayo1_5/models/base_model.py`
  - public `SPECIAL_TOKENS_KEYS` includes:
    - `question_start`
    - `question_end`
    - `answer_start`
    - `answer_end`
  - but unlike `alpamayo_base`, it does **not** list `meta_action_start/end`

- `alpamayo1.5/src/alpamayo1_5/models/token_utils.py`
  - still extracts `cot`, `meta_action`, `answer`
  - this creates a source-level inconsistency:
    - extractor expects `meta_action`
    - public helper path does not expose a direct `meta_action` route

### 3. local 1.5 tokenizer files still contain meta-action tokens

- local weight tokenizer file:
  - `weights/alpamayo15_vlm_weights/tokenizer.json`
- confirmed token strings present:
  - `<|meta_action_start|>`
  - `<|meta_action_end|>`

This matters because it means:

- `meta_action` is not impossible in principle for the local 1.5 weights
- but it is not honestly justified by the public `helper.py` route alone

## What this changes

Our earlier conclusion was too strong in one direction:

- old conclusion:
  - “direct meta_action is unsupported”

More accurate conclusion after comparing both repos:

- corrected conclusion:
  - “direct meta_action is unsupported by the public 1.5 helper path, but not ruled out by the tokenizer or by the older Alpamayo conversation design”

So the real uncertainty is not:

- whether the tokenizer can represent `meta_action`

It is:

- whether the current 1.5 local weights still **behaviorally support** chained multi-component generation well enough to use as a teacher contract

## Important structural difference

`alpamayo_base` conversation design is chain-structured.

- `build_conversation()` only asks the model to generate the **last** assistant component in `components_order` when `generation_mode=true`
- earlier assistant components must already be present in `data`

That means:

- direct `meta_action` in the base design is likely a **second-stage generation**
- not necessarily a one-shot “ask for everything at once” route

This is consistent with the code:

- if `meta_action` is not the final component, `construct_meta_action()` expects `meta_action_strings` already in data
- if `cot` is not the final component, `construct_cot()` expects `cot` already in data

## Practical implication for our teacher design

We should not treat `alpamayo1.5` teacher probing as a single-route question anymore.

We need at least three tiers:

### Tier A: official 1.5 reasoning route

- route:
  - `helper.create_message()`
- target:
  - `teacher_long_cot`
- trust:
  - high

### Tier B: official 1.5 VQA route

- route:
  - `helper.create_vqa_message()`
- target:
  - `teacher_answer`
- trust:
  - probe only until direct answer success rate is verified

### Tier C: base-style chained component route on 1.5 weights

- route:
  - use base conversation logic or an equivalent reproduction
  - stage 1: generate `cot`
  - stage 2: feed generated `cot` back and ask for `meta_action`
- target:
  - `teacher_meta_action`
- trust:
  - unproven, but now worth testing

## Revised teacher contract recommendation

Until Tier C is validated, the honest contract remains:

- primary:
  - `teacher_long_cot`
- auxiliary:
  - derived `teacher_meta_action`
- probe-only:
  - direct `teacher_answer`
- disabled:
  - `structured_json`

If Tier C succeeds on smoke samples, we can reopen:

- direct `teacher_meta_action` as a real auxiliary target

## Recommended next validation

Run a small comparison matrix on the same smoke samples:

1. official 1.5 `cot` route
2. official 1.5 `vqa answer` route
3. base-style chained `cot -> meta_action` route on 1.5 weights

For each route, record:

- direct field non-empty rate
- conflict rate against `teacher_long_cot`
- weak GT agreement
- human agreement
- collapse rate

## Current assessment

- the previous teacher design was too ambitious
- but the new comparison shows we should not prematurely conclude that `meta_action` is impossible
- the correct next step is not blind promotion of `meta_action`
- the correct next step is a staged probe:
  - official route first
  - chained base-style route second

## Key files

- `alpamayo_base/src/alpamayo_r1/chat_template/conversation.py`
- `alpamayo_base/src/alpamayo_r1/models/base_model.py`
- `alpamayo_base/src/alpamayo_r1/models/token_utils.py`
- `alpamayo_base/src/alpamayo_r1/processor/qwen_processor.py`
- `alpamayo1.5/src/alpamayo1_5/helper.py`
- `alpamayo1.5/src/alpamayo1_5/models/base_model.py`
- `alpamayo1.5/src/alpamayo1_5/models/token_utils.py`
- `weights/alpamayo15_vlm_weights/tokenizer.json`
