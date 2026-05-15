# No-Nav Distillation Issue And Decision Log

Updated: 2026-05-15

This document summarizes the main issues, hypotheses, checks, and decisions from the Alpamayo 1.5 to Cosmos Reason2 2B no-nav distillation work. It is intentionally written as a decision log, not only as a result table.

## Scope

Current target:

- Teacher: Alpamayo 1.5 10B
- Student backbone: Cosmos Reason2 2B
- Dataset: non-human OOD materialized into Alpamayo-style 4V x 4-frame inputs
- Primary mode: no navigation
- Later mode: selected navigation-conditioned samples
- Main downstream goal: produce a 2B backbone state that can support a lightweight action expert or flow-matching action head

Important paths:

```text
distill dataset:
  /home/pm97/workspace/dataset/distill_dataset

no-nav teacher cache:
  /home/pm97/workspace/dataset/distill_dataset/teacher_cache/no_nav

distillation repo:
  /home/pm97/workspace/sukim/distillation/cosmos_distillation

current main corpus:
  data/corpus/no_nav_teacher_pair_300chunks.jsonl

current official-input 20k init:
  outputs/checkpoints/no_nav_camera_labeled_official_20k/

current official-input 200k run:
  outputs/checkpoints/no_nav_camera_labeled_official_200k/
```

## Short Version

The biggest lesson so far is that the problem was not a single "loss is weak" issue.

We found several separate layers of risk:

- Teacher-cache completeness matters: raw action trajectory success is not enough if CoT, text top-k, traj top-k, hidden, or token artifacts are missing.
- Teacher-forced token accuracy is necessary but not sufficient. Free-run 128-token trajectory generation can still collapse.
- Oracle CoT did not fix free-run trajectory, so CoT text alone was not the main bottleneck.
- Traj-only training can hurt representation quality and collapse the prefill/interface state.
- The old Alpamayo action expert cannot be treated as plug-and-play with a 2B student KV cache because layer count, hidden size, and KV distribution differ.
- The original student input formatting had a serious mismatch with the official Alpamayo input contract.
- The most important recent fix is official Alpamayo-style input formatting:
  - explicit camera labels
  - correct 4V camera order
  - ego history fused as special tokens
  - official prompt text
  - assistant prefix starting at `<|cot_start|>`

Because of the last point, older checkpoints are useful for debugging and comparison, but the current official-input retraining is the branch that should be trusted going forward.

## Dataset And Teacher Cache Decisions

The source OOD dataset was converted into Alpamayo-style materialized samples:

- 4 cameras
- 4 temporal images per camera
- 1.6s ego history
- 6.4s future horizon
- about 8 anchors per clip
- about 444k materialized input samples

The clean dataset root policy became:

```text
distill_dataset/
  materialized/
  requests/
  manifests/
  reports/
  logs/
  teacher_cache/
    no_nav/
      text/
      traj/
    nav/
      text/
      traj/
```

Large array artifacts should be stored by path, not inline inside a giant manifest row.

Important no-nav teacher cache fields:

- `teacher_long_cot`
- `teacher_cot_token_ids`
- `teacher_text_topk_ids_path`
- `teacher_text_topk_logprobs_path`
- `teacher_text_entropy_path`
- `teacher_text_top1_margin_path`
- `teacher_cot_end_hidden_path`
- `teacher_traj_start_hidden_path`
- `teacher_action_pre_hidden_path`
- `teacher_future_token_ids`
- `teacher_traj_topk_ids_path`
- `teacher_traj_topk_logprobs_path`
- `teacher_traj_entropy_path`
- `teacher_traj_hidden_path`
- `teacher_discrete_decoded_traj_xyz_path`
- `teacher_action_expert_traj_xyz_path`
- prompt, request, input, and output hashes
- trajectory frame metadata:
  - `ego_at_sample_time`
  - `x_forward_y_left_z_up`
  - meters
  - 6.4s horizon

### Why text top-k and boundary hidden were added

At first, the cache design focused on CoT text, future tokens, trajectory top-k, and action expert output. Later we realized that this is incomplete for teacher-pair distillation:

- CoT hard CE alone only stores one sampled teacher text path.
- Text top-k lets the student see local teacher uncertainty over CoT tokens.
- Boundary hidden is small but valuable because action generation depends on the interface state after CoT and before trajectory/action prediction.

Decision:

- Store text top-k for CoT positions plus boundary tokens.
- Store boundary hidden for:
  - `cot_end`
  - `traj_future_start`
  - `action_expert_pre`
- Do not store full per-token CoT hidden for every sample by default.

## Teacher Inference And Cache QC Issues

### Issue: raw inference success was too loosely defined

We saw cases where raw Alpamayo inference produced an action expert trajectory but CoT was empty or missing.

Old interpretation:

- "action trajectory exists" means inference success.

Correct distillation interpretation:

- sample is usable only when the required teacher-pair artifacts are present.

Ready criteria should include:

- non-empty CoT
- valid CoT token ids
- valid text top-k, if required by the training stage
- valid boundary hidden, if required
- 128 future discrete tokens
- valid trajectory top-k
- valid trajectory hidden
- valid action expert trajectory
- valid output hashes and artifact paths

Decision:

- Treat empty CoT as a distillation failure even if action trajectory exists.
- Keep failure reason counts in reports.

### Issue: text top-k replay added another pass

At one point, the pipeline had effectively three passes:

1. raw teacher inference
2. text top-k teacher-forced replay
3. trajectory token/top-k extraction

This was too slow and created extra failure modes.

Better direction:

1. raw + text top-k/boundary hidden during generation when possible
2. trajectory generation-capture pass for discrete traj token/top-k/hidden

Decision:

- New cache ranges should avoid unnecessary replay.
- Older incomplete ranges can be recovered with replay if needed.

## Early Backbone Distillation

The first backbone distillation stages were:

- BP1: CoT CE + trajectory token CE
- BP2: BP1 + text top-k KD
- BP3: BP2 + trajectory top-k KD

The initial BP3 baseline looked acceptable under teacher forcing:

- CoT token accuracy roughly around the high 0.8 range
- trajectory token accuracy roughly around the 0.45 range

But free-run trajectory was weak:

- ADE vs teacher was around 4m
- FDE vs teacher was around 13m
- many samples had low unique trajectory-token diversity
- token match rate was low

Decision:

- Teacher-forced accuracy is not enough.
- Free-run ADE/FDE, token diversity, malformed rate, and hidden/interface QC must be part of model readiness.

## Teacher-Forced vs Free-Run

Teacher-forced evaluation means:

- The model sees the teacher prefix.
- For trajectory token position `i`, previous teacher trajectory tokens are provided.
- The model predicts the next teacher token.

This answers:

- "Can the student understand the teacher path when kept on the teacher trajectory-token manifold?"

It does not answer:

- "Can the student generate a stable 128-token future by itself?"

Free-run evaluation means:

- The student generates its own CoT and/or trajectory prefix.
- Its own mistakes become the next context.

This answers:

- "Does the deployed autoregressive path stay stable?"

Decision:

- Use teacher-forced metrics for learning signal and diagnosis.
- Use free-run metrics for actual behavior readiness.

## CoT Hypothesis

Hypothesis:

- Maybe trajectory fails because the student cannot generate the right CoT.

Test:

- Give oracle teacher CoT prefix, then let trajectory tokens free-run.

Observed:

- Oracle CoT did not materially improve trajectory free-run.
- Joint free-run and oracle-CoT trajectory-only free-run remained similarly weak.

Conclusion:

- CoT generation quality matters, but it was not the primary cause of the 128-token trajectory collapse.
- The larger issue was trajectory autoregression and planning/interface state.

Decision:

- Do not solve this only by increasing CoT CE.
- Keep CoT training, but do not let it hide trajectory/interface failure.

## Trajectory-Only Hypothesis

Hypothesis:

- Maybe CoT loss is competing with trajectory loss. Train only trajectory tokens.

Observed:

- Straight-ish samples improved visually in some cases.
- Curves, stops, and speed-sensitive cases remained weak.
- Some samples where CoT said stop/slow still generated overly long paths when CoT was removed.
- Hidden/interface checks showed representation collapse risk.

Important hidden QC observation:

- Traj-only could improve some trajectory-body alignment while damaging prefill/interface representation.
- Effective rank collapsed badly in some checks.

Conclusion:

- Traj-only is not the correct main route.
- It may overfit token ids without preserving a usable planning state.

Decision:

- Keep joint CoT + trajectory objectives.
- Watch hidden/interface distribution, not only token CE.

## Action Expert And KV Contract

The Alpamayo action expert consumes VLM KV/interface state. This creates a contract problem:

Teacher VLM:

- larger hidden size
- more layers
- Alpamayo-specific KV distribution

Student VLM:

- Cosmos Reason2 2B
- different hidden size
- different layer count
- different KV distribution

### Experiment: use prefill KV without full CoT generation

We tested action expert behavior using prompt/image/ego prefill KV compared to full generation KV.

Observation:

- Action expert trajectory did not change as much as expected.

Interpretation:

- The action expert heavily uses image/prompt/ego information already encoded into the prefill KV.
- Generated CoT tokens can still affect KV, but in tested samples the prefill state already contained strong driving information.

Important clarification:

- This does not mean the LLM is unused.
- Image and text tokens still pass through the LLM layers to create KV.
- The action expert reads the resulting VLM state.

### Experiment: compress 36-layer teacher KV to 28-layer expert input

Hypothesis:

- Maybe we can adapt the existing Alpamayo action expert by compressing/selecting 36 teacher layers into 28 layers.

Observed:

- Simple 36-to-28 layer select/mix did not reproduce the original 36-layer action expert well enough.

Conclusion:

- Layer compression is not just a shape conversion.
- The action expert expects a learned distribution over layer-wise KV states.

Decision:

- Do not rely on the original 36-layer Alpamayo action expert as a direct plug-in for the 2B student.
- Long-term direction should be a student-compatible lightweight action expert or flow-matching head trained against teacher action expert trajectories.

## Hidden-To-Action Probe

Question:

- Does the student backbone hidden state contain enough information to recover teacher action trajectories?

Probe setup:

- Freeze backbone.
- Extract boundary hidden features such as:
  - `h_cot_end`
  - `h_traj_start`
  - `mean(last16 prefix hidden)`
- Train a small probe to predict teacher action expert trajectory or action representation.
- Compare against:
  - BP3 init
  - 20만 final
  - BP5
  - ego-only baseline
  - visual ablations

Purpose:

- This is not the final action expert.
- It is a diagnostic to check whether the backbone has action-relevant information.

Observed direction:

- 20만 final hidden probes improved over BP3 init.
- Ego-only remained a strong baseline, so visual contribution must be checked with ablations.
- Normal vs black/shuffled image ablations are needed to avoid fooling ourselves with ego priors.

Decision:

- Keep probe results as evidence, but do not overinterpret them as final driving performance.
- For final inference, student-free prefix probe matters more than teacher-prefix probe.

## Visual Grounding And Camera Input Contract

This became one of the most important discoveries.

We initially suspected that student distillation might be weak because the student input contract did not match Alpamayo's official input format.

Important questions:

1. Are camera labels explicitly included?
2. Is 4V camera order exactly Alpamayo official order?
3. Is ego history fused as special trajectory-history tokens?
4. Does the prompt text match official Alpamayo?
5. Does the assistant prefix begin with `<|cot_start|>`?
6. Does the `apply_chat_template` token order match teacher inference?

### Camera labels

The official 4-camera setup uses explicit text labels such as:

```text
Front left camera:
Front camera:
Front right camera:
Front telephoto camera:
```

These labels are text tokens, not vision tokens. They become part of the prompt context before each image block.

Why this matters:

- Without labels, the model may only see image placeholders in a compact order.
- If camera identity is not represented the same way as the teacher, vision grounding can be degraded.

### 4V vs 7V confusion

A related GitHub issue raised the question of fixed slots vs compact image lists.

Our conclusion:

- Alpamayo paper and public setup are 4-camera oriented.
- Public code/weights do not appear to require filling missing rear-camera slots for a 7V fixed layout.
- For this public 4V setup, the correct path is to provide the 4 expected camera views with correct labels/order.

Decision:

- Distillation should match the public Alpamayo 4V input contract.
- Do not invent 7V missing-slot handling unless using a model explicitly trained for 7V.

### Ego history formatting

We also found the student path had formatting mismatch risk around ego history.

Correct direction:

- Ego history should be fused into the prompt as Alpamayo-compatible special trajectory-history tokens.
- It should not be treated as plain numeric text if the teacher uses tokenized trajectory history.

Decision:

- The collator now supports official Alpamayo prompt text and fused history tokens.
- Prompt contract checks were added to compare teacher helper output vs student collator output.

Required unit/contract checks:

- image placeholder count is 16
- camera label order is:
  - Front left
  - Front
  - Front right
  - Front telephoto
- frame order is 0,1,2,3 per camera
- materialized `cam3` maps to original camera index 6, front telephoto
- prompt text follows image content
- assistant prefix is `<|cot_start|>`
- chat-template token order matches teacher helper

Decision:

- Older checkpoints before this fix are diagnostic only.
- Official-input retraining is the branch to trust.

## Current Official-Input Retraining

We created an official-input 20k checkpoint and started a 200k continuation.

Why:

- The previous 200k/BP3 results were trained under a less faithful prompt/input contract.
- If the input contract was wrong, more loss tuning on old data would not answer the real question.

Current training direction:

- official camera labels
- official prompt text
- fused ego-history tokens
- CoT CE
- text top-k KD
- trajectory token CE
- trajectory top-k KD
- decode/free-run eval checkpoints

What to watch:

- CoT token accuracy
- trajectory token accuracy
- free-run ADE/FDE vs teacher
- bad sample rate
- unique future-token count
- max same-token run
- malformed output rate
- `<traj_future_start>` hit rate
- first16/curvature token metrics
- hidden/interface QC

Early signs:

- The 20k official-input checkpoint improved over the mismatched-format baseline but was not enough by itself.
- The 200k official-input run is intended to test whether scale plus correct input contract gives a real free-run improvement.

## Navigation Samples

Navigation work was investigated separately.

Findings:

- Raw nav categories include straight, lane changes, turns, curves, sharp turns, and exits.
- Curve/sharp/exit labels need more careful review and visualization.
- Straight and turn samples are easier to make reliable first.

Current nav strategy:

- Keep no-nav as the main bulk teacher cache.
- Add selected nav samples later.
- Prefer one or a few representative nav samples per clip instead of adding near-duplicate nav text to every anchor.
- For event-based nav text, select anchor points before the event and compute distance-to-event, such as "turn left in Xm".
- Balance straight and turn classes.

Decision:

- Nav is useful, but it should not distract from fixing no-nav backbone input contract and action interface first.

## Inference And Quantization Notes

Profiling suggested the main inference bottleneck is VLM prefill/decode, heavily affected by vision-token count.

Important points:

- Downsampling image resolution can help, but the number of vision tokens entering the LLM is often more important.
- Alpamayo/Qwen-VL preprocessing resizes raw camera images to roughly `320 x 576` under pixel constraints.
- FP8 attempts were sensitive, especially around activations/KV cache.
- int4 AWQ and eventually Thor-friendly NVFP4 are more promising deployment directions.

Decision:

- For speed reports, compare teacher vs student with identical input format.
- Separate timing into:
  - visual preprocessing/ViT
  - LLM prefill
  - decode
  - action-pre state
  - action expert/head

## Why The Earlier Results Were Confusing

Several things made results look contradictory:

1. Different sample counts were being discussed:
   - 444k materialized samples
   - 233k teacher-pair JSONL
   - 200k training slice
   - 4,760 full-val
   - smaller val64/val204 probes

2. Different evaluation modes were mixed:
   - teacher-forced token acc
   - free-run decode
   - oracle-CoT free-run
   - teacher-prefix hidden probe
   - student-free hidden probe

3. Different targets were mixed:
   - GT future
   - teacher discrete decoded trajectory
   - teacher action expert trajectory

Current rule:

- For distillation model selection, compare student vs teacher unless explicitly doing GT quality analysis.
- Always name the exact corpus, checkpoint, eval mode, and target.

## Current Best Interpretation

The best current explanation is:

1. The model can learn teacher-forced local token prediction.
2. Free-run trajectory still fails when its own generated trajectory-token prefix drifts.
3. CoT is not the sole cause, because oracle CoT did not rescue free-run trajectory.
4. Traj-only training can make token-side metrics look better while damaging the broader interface state.
5. Action expert reuse is blocked by KV contract mismatch.
6. The student must first match the official Alpamayo input contract.
7. After the input-contract fix, the next meaningful result is the official 200k run with free-run and hidden/interface eval.

## Practical Lessons

Do:

- Keep official Alpamayo prompt/input parity tests.
- Treat CoT, traj token, top-k, hidden, and action expert outputs as separate artifacts.
- Use teacher-forced metrics for diagnosis.
- Use free-run ADE/FDE and malformed rate for readiness.
- Track hidden/interface QC.
- Compare to ego-only and visual ablation baselines.
- Use teacher action expert trajectory as the main distillation target for action-head readiness.

Do not:

- Trust raw action trajectory success as full cache success.
- Judge backbone readiness only by teacher-forced token accuracy.
- Assume oracle CoT fixes trajectory generation.
- Push trajectory-only CE as the main training path without hidden/interface checks.
- Directly plug the 36-layer teacher action expert into the 28-layer/2048-dim student KV.
- Mix GT-vs-teacher metrics without naming the target.
- Train on a prompt format that does not match the teacher.

## Next Decisions To Make

1. Finish or checkpoint the official-input 200k run.
2. Evaluate it with:
   - teacher-forced token metrics
   - free-run ADE/FDE vs teacher
   - `<traj_future_start>` contract metrics
   - unique token diversity
   - malformed output rate
   - hidden/interface QC
   - visual ablation
3. Decide whether official-input BP3 is strong enough for an action-head smoke test.
4. If yes, train a student-compatible small action/FM head against teacher action expert trajectory.
5. If no, improve backbone with:
   - scheduled sampling
   - stronger but calibrated trajectory top-k KD
   - first16/curvature-focused weighting
   - hidden/interface alignment
   - hard bucket oversampling
6. Resume nav teacher-cache work only after no-nav backbone direction is stable.

## Current Standing Recommendation

The correct branch is no longer "keep tuning old BP3." The correct branch is:

```text
official Alpamayo 4V input contract
  -> CoT + text KD + traj CE + traj KD
  -> free-run and hidden/interface eval
  -> student-compatible action/FM head
```

Older BP1/BP2/BP3/BP5 results remain useful as debugging evidence, but the input-contract fix changes the baseline. The next high-value answer is whether the official-input 200k run improves free-run geometry and interface stability enough to justify moving to an action head.
