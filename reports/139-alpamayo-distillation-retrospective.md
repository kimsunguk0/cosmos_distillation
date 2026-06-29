# 139 - Alpamayo Distillation Retrospective

**Date:** 2026-06-12
**Updated:** 2026-06-29
**Scope:** dataset → teacher cache → Cosmos-Reason2-2B student backbone → discrete trajectory distillation → decoding/minADE → Action Expert → FLEX/QAT/deployment → VQA grounding / Q2 curriculum
**Main source set:** `reports/closed_issues/*`, `reports/058` onward, `reports/2026-05-27_multimodal_input_backbone_ae_summary.md`, `reports/075`, `reports/080`-`085`, `reports/125`-`140`, latest `outputs/reports/*` audits, `alpamayo_repo/alpamayo1.5/reports/vqa_q1_grounding_stability_20260621.md`, and training/corpus/export scripts.

---

## 0. One-Line History

The project moved from "make the 2B student emit valid 128 trajectory tokens" to "make those tokens geometrically good", then to "train a student-compatible action expert", and finally to "compress the Qwen3-VL visual interface with DeepStack-aware FLEX for Jetson Thor 100 ms / 10 Hz deployment."

The strongest current non-FLEX student reference is:

```text
outputs/checkpoints/no_nav_camera_labeled_official_full444k/
  no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250
```

The strongest current FLEX deployment branch is:

```text
backbone:
outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final

AE28:
outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt

AE14:
outputs/action_expert/ae14_from_ae28_10step/best.pt
```

The current high-level conclusion is strict:

```text
No-FLEX 2B remains the quality reference.
FLEX K512 is functional and deployability-relevant, but it is not yet B0-equivalent on the harder semantic benchmark.
QAT/export path is conceptually correct, but the repo-side QAT implementation still needs contract fixes before it should be trusted.
Official Q2-style VQA grounding is the current stable language-supervision path; raw Q1 / 1A scene-grounding prompts are not stable enough for an ungated main teacher dump.
AE-path reranking has real oracle headroom, but the first path-only reranker did not beat mean-trajectory selection.
```

---

## 0.1 2026-06-29 Current Delta

This update adds the post-retrospective VQA grounding work from
`alpamayo_repo/alpamayo1.5` and the AE-path reranker bootstrap from report 140.

### VQA / Q2 grounding state

Latest Alpamayo-side report:

```text
alpamayo_repo/alpamayo1.5/reports/vqa_q1_grounding_stability_20260621.md
```

The stable result is not Q1. On the 4cam x 1frame visual audit:

| Label source | Audit size | Usable | Verdict profile |
|---|---:|---:|---|
| Official Q1: `Describe the scene.` | 1000 | `264/1000` | bad `736`, partial `246`, ok `18` |
| Official Q2 traffic-elements / driving-behavior prompt | 1000 | `830/1000` | partial `610`, bad `170`, ok `220` |
| Best raw Q1 candidate `G_scene_context_no_counts` | 100 | `50/100` | ok `4`, partial `48`, bad `41`, missing `7` |
| Best Q2-derived Q1 source `D_q2_sanitized_visible_elements` | 100 | `55/100` | ok `25`, partial `30`, bad `45` |

Interpretation:

1. Official Q1 is too hallucination-prone for a main label dump.
2. Removing the behavior/action wording from Q2 hurts stability rather than
   preserving it.
3. Q2-derived Q1 labels are better than raw Q1 prompts, but still below the
   official Q2 baseline on the same samples.
4. Q1 / 1A labels should be filtered auxiliary data only; Q2 remains the main
   teacher-label path.

The latest output-side smoke data confirms the pipeline wiring, not a final
training result:

```text
alpamayo_repo/alpamayo1.5/output/cosmos2b_vqa_q2_lora_smoke_data_20260622/split_summary.json
total_q2_rows: 22284
train_rows: 8
val_rows: 2
model_path: base_weights/cosmos-reason-2b
```

The 2026-06-22 1A context checks are too small to change the decision. One
guarded Alpamayo 1A sample failed visual judging (`0/1` usable), while a
separate 10B-vs-8B gate had Alpamayo 10B at `1/1` partial and Cosmos-Reason2-8B
at `0/1`. Treat this as prompt/debug evidence only, not a stable dataset metric.

### AE path reranker state

Report 140 moved selector work to the deployable AE path:

```text
student self-generated CoT/prefix -> AE diffusion N paths -> selected path
```

The first reranker used GT-free path geometry features only. It failed:

| Method, semantic val806 external test | ADE | FDE |
|---|---:|---:|
| first path | `3.1990` | `9.5897` |
| mean_traj | `2.7227` | `8.1559` |
| medoid | `2.8139` | `8.4370` |
| oracle best | `1.6835` | `5.0521` |
| learned argmax | `3.4770` | `10.3411` |
| learned weighted | `2.7529` | `8.2555` |

The important point is the gap, not the failed model:

```text
semantic val806 B0 AE28 mean_traj ADE: 2.7227
semantic val806 B0 AE28 oracle-best ADE: 1.6835
recoverable selected-vs-oracle gap: ~1.04 m ADE
```

Conclusion: selection/ranking is still worth doing, but not from path geometry
alone. A viable selector probably needs prefix confidence, token entropy/margin,
diffusion likelihood/probability features, or teacher/value-head supervision.

### Pairwise diversity and reasoning audit

The latest no-FLEX AE28 audit shows the model is not simply collapsed:

```text
outputs/reports/student_noflex_ae28_pairwise_reasoning_20260622/report.md
```

Key numbers on semantic val806:

| Metric | Value |
|---|---:|
| pairwise mean ADE | `1.7538 m` |
| collapse rate, pairwise mean < 0.25 m | `2.3573%` |
| low-diversity rate, pairwise mean < 1.0 m | `23.9454%` |
| CoC causal score | `0.9814` |
| teacher agreement score | `0.6072` |
| teacher exact action match | `53.8462%` |
| teacher direction conflict | `6.0794%` |

This supports the current curriculum rule: fluent causal text is not the same as
teacher-grounded driving intent. Direct CoC/action injection into the backbone
should stay gated behind grounding checks.

---

## 1. Dataset Preparation

### 1.1 Raw materialization contract

The distillation data is built around materialized autonomous-driving samples:

- 4 cameras
- 4 frames per camera
- 16 image blocks per sample
- ego history tokens
- no navigation instruction
- 6.4 s future trajectory represented by 128 discrete trajectory tokens

The camera order and prompt labels were fixed to match the public Alpamayo helper contract:

| Materialized cam | Original camera | Prompt label |
|---|---|---|
| cam0 | `camera_cross_left_120fov` | Front left camera |
| cam1 | `camera_front_wide_120fov` | Front camera |
| cam2 | `camera_cross_right_120fov` | Front right camera |
| cam3 | `camera_front_tele_30fov` | Front telephoto camera |

The prompt is a single Qwen-style decoder-only multimodal stream:

```text
system text
camera label text
image placeholder spans
trajectory history tokens
request text
assistant <|cot_start|>
```

Important: camera labels such as `Front camera` are text tokens. The actual image tokens are inserted into the sequence at the image placeholder spans after ViT/merger processing.

Primary reference:

- `reports/2026-05-27_multimodal_input_backbone_ae_summary.md`

### 1.2 Teacher-pair corpus construction

The main corpus builder is:

```text
scripts/42_build_no_nav_teacher_pair_corpus.py
```

It reads teacher inference manifest parquet files and emits JSONL rows with:

- `sample_id`, `clip_id`, split
- materialized sample path
- 16 expected image names: `cam{0..3}_f{0..3}.png`
- no-nav question text
- teacher long CoT
- 128 future trajectory token ids or token-id path
- text top-k ids/logprobs paths
- trajectory top-k ids/logprobs paths
- trajectory hidden path
- boundary hidden paths: `cot_end`, `traj_start`, `action_pre`
- provenance hashes for request/output JSON

Filtering is intentionally strict:

- teacher inference status must be ready
- CoT must be non-empty
- future trajectory tokens must be non-empty
- future token count must be exactly 128
- invalid future token ids above the trajectory vocab range are rejected
- text top-k and trajectory top-k artifacts are required unless explicitly allowed

The output row has separate target namespaces:

```text
hard_target:
  GT/teacher-provided CoT text and 128 future token ids

teacher_target:
  teacher CoT text + text top-k artifacts

teacher_traj_target:
  future token path + trajectory top-k + hidden artifacts

teacher_cache:
  raw teacher JSON/request JSON + boundary hidden paths
```

This split became important later because text-CoT distillation, future-token CE, top-k KD, hidden alignment, and Action Expert conditioning are not the same supervision problem.

### 1.3 Semantic-balanced train set

The semantic-balanced corpus builder is:

```text
scripts/85_build_semantic_scene_balanced_corpus.py
```

It classifies rows into 17 scene buckets using teacher CoT text:

```text
traffic_right_turn
traffic_left_turn
right_turn_no_light
left_turn_no_light
red_light_stop
stop_sign
pedestrian_crosswalk
cut_in_merge_yield
lead_vehicle_follow
parked_stopped_obstacle_nudge
lane_change
curve
green_light_go_straight
intersection_other
slow_decel_other
keep_lane_straight
other
```

The purpose was not cosmetic. Early full-val breakdown showed the model was much worse on turns, curves, intersections, and slow/decel scenes than on straight lane keeping. Balancing forced hard categories to appear in training instead of letting easy straight driving dominate.

Important dataset artifacts:

```text
data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl
data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl
data/corpus/benchmark_semantic_val_cap50_seed42.jsonl
```

The 806-sample benchmark in report 138 uses up to 50 samples per semantic bucket, producing a harder and more stable comparison set than ad hoc val512 splits.

---

## 2. Teacher Data Preparation

### 2.1 Teacher model role

The teacher is the public Alpamayo 1.5 10B stack:

```text
/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B/
```

It supplies:

- long driving CoT
- 128 discrete future trajectory tokens
- text top-k distributions
- trajectory top-k distributions
- selected hidden states and boundary hidden states
- Action Expert reference behavior for later AE work

The main student target is not "copy every internal teacher tensor." The practical target evolved into:

```text
1. reproduce the Alpamayo input/output contract
2. learn valid CoT + 128 trajectory-token emission
3. improve geometry under free-run decode
4. train an Action Expert that works on student KV
```

### 2.2 Critical teacher-cache lesson

Closed issue 029 documented a key trap: the public Alpamayo rollout path stops at `<|traj_future_start|>` and then the Action Expert/diffusion path takes over. Therefore raw LM trajectory-token logits after the trajectory start are not automatically available from the ordinary public trajectory generation path.

This forced a split in how teacher signals were treated:

- Alpamayo 1.5 is reliable for CoT/reasoning prefix and AE trajectory behavior.
- Discrete trajectory token supervision must be explicitly cached/constructed as a trajectory-token target.
- Teacher top-k/hidden artifacts need exact position contracts; otherwise "KD" may be applied to the wrong positions.

This was one reason early hidden/KD-heavy attempts failed: the labels existed, but the semantic meaning of each target channel was not always aligned to the student training positions.

---

## 3. Why Cosmos-Reason2-2B Was the Student Backbone

The student base is:

```text
/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b/
```

The practical reasons:

1. It is a Qwen3-VL-family vision-language model, so the Alpamayo-style camera/image/text/history prompt can be represented with the same kind of decoder-only multimodal stream.
2. It is small enough to be a realistic student for embedded deployment, unlike the public 10B teacher.
3. It preserves the important Qwen3-VL interface pieces: image placeholders, MRoPE/position handling, DeepStack visual injection, and a causal LLM prefix cache that an Action Expert can condition on.
4. The target is Jetson Thor 10 Hz, so the project needs a 2B-class backbone plus compression/quantization rather than a teacher-scale model.

The compatibility work is recorded in:

```text
reports/closed_issues/009-qwen3vl-student-wrapper-compat.md
src/model/student_wrapper.py
```

The wrapper had to route `qwen3_vl` models through Qwen3-VL-compatible classes and expose the right output patterns for hidden states, image embeddings, DeepStack, and generation.

---

## 4. Early Backbone Distillation: What Failed

### 4.1 The first blocker: trajectory emission collapse

Early attempts could make losses move while the model still failed to emit a valid trajectory body. Common symptoms:

- incomplete trajectory body
- repeated token plateau such as `<i1499>`
- low token diversity
- good-looking teacher-forced loss but bad free-run geometry

Closed issues 030-057 document the debug path:

- trainer/collator trajectory label bugs
- teacher-pair target ambiguity
- teacher trajectory top-k probes
- hidden projector probes
- auxiliary trajectory heads
- hybrid LM + aux decode
- frozen latent/bridge experiments
- readout unfreezing
- dense trajectory-token row experiments

The most important conclusion from report 058:

```text
The old "cannot emit trajectory body" collapse was mostly a recipe/path issue.
The model could emit 128/128 trajectory tokens once the clean token path was used.
```

The recovery recipe was:

- all-layer LoRA
- trainable custom trajectory/control token rows
- GT trajectory-token CE as the main objective
- weak GT CoT loss
- small output-format CE
- teacher KD/hidden alignment/aux/hybrid objectives disabled or kept secondary
- enough data/steps

This changed the core question from:

```text
Can the student emit trajectory tokens?
```

to:

```text
Can the emitted trajectory-token sequence produce good 6.4 s geometry?
```

### 4.2 Why mixed objectives were dangerous

The repeated failure pattern was a "policy soup":

- GT CE
- teacher CE/KD
- hidden alignment
- boundary hidden alignment
- boundary-action XYZ
- auxiliary trajectory heads
- teacher-pair SFT

Several of these could improve loss while making decoded ADE/FDE worse. The project therefore shifted to selecting checkpoints by:

- ADE/FDE
- token diversity
- max same-token run
- token histogram concentration
- scene failure tags
- later, minADE@N

Not by total loss alone.

### 4.3 Scheduled sampling / DAgger attempts

Scheduled sampling and DAgger-like token training were tried because teacher-forced training did not expose the model to its own generated-prefix errors.

But the tried versions did not improve the main free-run geometry:

| Run | Result |
|---|---|
| scheduled sampling/top-k, sched16 p20 | full free-run ADE/FDE `2.8378 / 9.4037`, slightly worse than official 200k baseline `2.8178 / 9.1748` |
| token DAgger 10k prefix32 | val64 ADE/FDE `8.24 / 24.96`, bad |

Interpretation:

```text
The idea was valid, but the implementation pushed the model into off-manifold generated-prefix states without a strong enough corrective signal.
Scheduled sampling/DAgger did not become the main path.
```

Primary reference:

- `reports/2026-05-27_multimodal_input_backbone_ae_summary.md`

---

## 5. Current Best No-FLEX Student Backbone

### 5.1 Training recipe

Main checkpoint:

```text
outputs/checkpoints/no_nav_camera_labeled_official_full444k/
  no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250
```

Core recipe:

```text
base model: Cosmos-Reason2-2B
corpus: no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl
train cap: 200,000 samples from ~209,007-row balanced corpus
max length: 4096
precision: bf16
LoRA: all-layer rank 64, alpha 128, dropout 0.05
trainable: trajectory/control token rows
trainable side modules: traj_hidden_bridge, multimodal_projector
input contract: official Alpamayo camera-labeled prompt
```

Loss mix from the stage config:

| Loss | Weight |
|---|---:|
| GT CoT CE | 0.08 |
| text top-k KD | 0.08 |
| trajectory CE | 0.85 |
| trajectory top-k KD | 0.12 |
| trajectory hidden alignment | 0.08 |
| text boundary hidden alignment | 0.05 |
| output format CE | 0.20 |
| geometry losses / aux losses | 0.0 |

This is intentionally token-path-first. Hidden alignment is present, but it is not the dominant objective.

Config reference:

```text
configs/train/stage_bp3_no_nav_camera_labeled_full444k_balanced_hidden_gc.yaml
```

### 5.2 Why `step_006250`

`step_006250` is the stable backbone reference because decode geometry selected it better than later lower-loss checkpoints.

Full-val greedy decode on 4,760 validation samples:

| Metric | 200k baseline | `step_006250` |
|---|---:|---:|
| Greedy ADE/FDE vs teacher | `2.818 / 9.175` | `2.557 / 8.365` |
| Avg unique traj ids | `13.9` | `17.0` |

GT comparison:

| Model | ADE GT | FDE GT |
|---|---:|---:|
| Teacher | `1.739` | `5.175` |
| Student greedy | `3.011` | `9.822` |

Vision ablation showed the student really uses images:

| Input | ADE/FDE |
|---|---:|
| normal | `2.64 / 8.58` |
| camera shuffle | `3.36 / 10.30` |
| black | `3.50 / 10.82` |
| gray | `3.57 / 10.92` |
| noise | `3.37 / 10.35` |

Interpretation:

```text
The student is not blind.
But camera-specific grounding is weaker than the teacher, and long-horizon drift remains the main discrete-token weakness.
```

---

## 6. Greedy Decode vs Sampling / minADE

A major evaluation correction happened after `step_006250`.

Greedy argmax looked weak because it collapses token diversity:

| Decode | Avg unique trajectory tokens |
|---|---:|
| greedy argmax | ~17 / 128 |
| sampling, `temperature=1.0`, `top_p=0.95` | ~75 / 128 |

Best-of-4 oracle on full 4,760 val:

| Decode | ADE GT | FDE GT |
|---|---:|---:|
| Teacher | `1.739` | `5.175` |
| Student greedy | `3.011` | `9.822` |
| Student best-of-4 oracle | `1.702` | `4.914` |

Best-of-4 vs teacher:

| Decode | ADE vs teacher | FDE vs teacher |
|---|---:|---:|
| greedy | `2.557` | `8.365` |
| best-of-4 oracle | `1.376` | `3.956` |

This is why later reports use `minADE6@6.4s`:

- single selected trajectory = deployable quality
- minADE@6 = whether the model distribution contains good modes
- mean/medoid/selection methods = how much of that oracle can be recovered without GT

The correction is not "minADE makes the model better." It reveals that greedy under-reports model capacity and over-punishes multimodal futures.

---

## 7. Action Expert Phase

### 7.1 First AE failure: student KV mismatch

Report 060 captured the key failure:

| Pairing | Result |
|---|---:|
| Teacher AE + Teacher KV | ~`1.5 m` |
| Teacher AE + Student KV | `8.82 m` |
| Student AE + Student KV, early best | ~`3.63 m` |

This proved that the token-distilled student hidden/KV space was not automatically compatible with the teacher Action Expert.

Root causes:

1. Hidden alignment was too weak relative to token CE.
2. Teacher and student bridges were both trainable, so the teacher distribution could move toward the student instead of anchoring the student to the teacher.
3. Cosine-heavy alignment matched direction more than norm/scale.
4. Only selected final hidden/boundary states were aligned, while the AE attends to layerwise KV across the prefix.
5. Vision-side distillation was weak or absent.
6. Token-level ADE did not imply AE-compatible hidden states.

The project then stopped assuming that "good discrete-token backbone" means "teacher AE can be reused."

### 7.2 Official FM recipe audit

Report 075 checked the public Alpamayo SFT/FM code and found:

| Item | Official public SFT path | Our working student AE path |
|---|---|---|
| expert LR | `1e-4` | `1e-4` |
| warmup | 100 | often 0 or explicit schedule experiments |
| LR schedule | cosine, min LR `1e-6` | varied |
| timestep sampler | Beta(1.5, 1.0) | same |
| training target | velocity `x - noise` | same |
| inference | Euler, 10 steps | same default |
| FM draws per sample | 1 | `num_time_samples=16` |

The larger `num_time_samples=16` is not official, but it stabilized our colder student-AE training because we were not merely fine-tuning an already well-aligned official AE.

### 7.3 Stage1 / Q1 / Q2 / Q3 results

Held-out Stage1 setup:

```text
train: 20,000
val: 2,000
prefix: student_free
target: teacher
AE init: student_backbone_init
expert/proj LR: 1e-4
num_time_samples: 16
eval: temp 0.85, N paths, mean_traj / oracle diagnostics
```

Report 080 showed the Stage1 run reached:

```text
best held-out val ADE: 2.503 at step 9000
```

Report 081 showed inference sweeps did not solve the remaining gap:

| Eval | ADE |
|---|---:|
| single temp 1.0 | `3.081` |
| single temp 0.5 | `2.550` |
| mean_traj N16 temp 0.85 | `2.525` |
| oracle best N16 temp 0.85 | `1.316` |

Interpretation:

```text
The sampled distribution contains much better trajectories than the deployable selector extracts.
Temperature and mean_traj help, but do not fully solve selection/ranking.
```

Report 085 best paper-style numbers:

| Run | Eval | ADE | minADE@6 |
|---|---|---:|---:|
| Q3 temp 1.0 | Stage1 val512 | `2.631` | `1.243` |
| Q3 temp 0.85 | Stage1 val512 | `2.506` | `1.276` |
| Stage2 200k best temp 0.85 | Stage2 val1024 | `2.768` | `1.350` |
| Stage2 final temp 0.85 | Stage2 val1024 | `2.814` | `1.339` |

Important metric hygiene:

```text
Older Q2/Q3 numbers around ADE 2.1 and minADE 1.0 were often N16 mean_traj/minADE@16.
They are not directly paper-style minADE@6.
```

### 7.4 Current No-FLEX AE reference

Current registry reference:

```text
AE28 No-FLEX:
outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/best.pt

Q3 comparison AE:
outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt
```

The checkpoint used depends on the eval split/report:

- report 085: Q3 is still strongest on the older Stage1 paper-style N6 table
- report 132: B0 Q3 is the same-split comparator against FLEX K512 AE
- report 138: No-FLEX 2B+AE28 is benchmarked on the semantic val806 benchmark

Do not compare numbers across these splits without naming the split.

---

## 8. FLEX Phase

### 8.1 Why FLEX was introduced

The current Alpamayo-style input into the LLM is token-heavy:

```text
~3086 total prefix tokens
~2880 vision tokens
```

For Jetson Thor 100 ms / 10 Hz, the visual-token count is one of the major latency drivers. FLEX was introduced to reduce the LLM visual interface while keeping enough scene information for planning.

The target was not "paper reproduction for its own sake." The target is:

```text
reduce LLM vision tokens enough for embedded deployment
without destroying trajectory quality or Action Expert conditioning.
```

### 8.2 FLEX v1 failure

Reports 086-124 document many FLEX attempts. The useful summary is report 125:

| Attempt family | Result |
|---|---|
| FLEX encoder only / weak LoRA | failed or worse than B0 |
| per-image factorized FLEX | failed |
| position preservation / dummy slots / trajectory alignment | failed |
| DeepStack OFF tiny-sample success | did not scale |
| K1792 passthrough/selectors | useful diagnostic, not learned FLEX |

Core v1 root causes:

1. It compressed final ViT output but did not coherently compress DeepStack intermediate visual features.
2. LLM adaptation was too weak for a 2880 → 512 token distribution shift.
3. Greedy ADE was overused as the gate, even though B0 itself needs sampling/minADE for a fair view.
4. Some "FLEX" successes were actually selected original-feature passthrough, not learned scene-token compression.

### 8.3 DeepStack became non-negotiable

Step 0 ablation in report 125 tested DeepStack ON vs OFF.

| Model | Decode path | DS ON ADE/FDE | DS OFF ADE/FDE | ADE hit |
|---|---|---:|---:|---:|
| public 10B | discrete | `1.810 / 5.099` | `2.838 / 7.759` | `+56.8%` |
| public 10B | AE | `1.843 / 5.325` | `2.709 / 7.735` | `+47.0%` |
| B0 2B | discrete | `3.486 / 11.534` | `3.734 / 12.081` | `+7.1%` |
| B0 2B | AE | `2.713 / 7.712` | `4.226 / 12.315` | `+55.8%` |

The important asymmetry:

```text
B0 discrete token path can partly survive DeepStack OFF.
The Action Expert path cannot.
```

Since deployment uses the AE path, FLEX must be DeepStack-aware.

### 8.4 ML-FLEX design

The redesign is Multi-Level FLEX:

```text
ViT layer 5 merged output   -> FLEX level 0 -> 512 DeepStack tokens
ViT layer 11 merged output  -> FLEX level 1 -> 512 DeepStack tokens
ViT layer 17 merged output  -> FLEX level 2 -> 512 DeepStack tokens
ViT final merged output     -> FLEX level 3 -> 512 LLM input image tokens
```

For the 2B student:

```text
images per sample: 16
tokens per image: 32
scene tokens: 512
compression: 2880 -> 512
slot layout: camera x frame x local slot
DeepStack: active
```

The design principle:

```text
The final visual tokens and all three DeepStack streams must refer to the same compressed slot order.
```

Otherwise the LLM input sees one visual reality while early residual DeepStack injections add a different one.

Primary implementation:

```text
src/model/flex_scene_encoder.py
src/model/student_wrapper.py
scripts/09_train_distill.py
configs/train/stage_mlflex_k512_bp3_hidden_gc_20k_e3.yaml
```

### 8.5 ML-FLEX training and result

Initial K512 run:

```text
run: outputs/checkpoints/mlflex_k512_bp3_20k_e3_b16_20260608
corpus: no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl
train rows: 20,000
epochs: 3
steps: 3,750
batch: 16
LoRA: all-layer rank 64
FLEX: K512, 4-level, DeepStack active
base dense weights: frozen
```

Loss mix matched the B0 token recipe:

| Loss | Weight |
|---|---:|
| traj CE | 0.85 |
| CoT CE | 0.08 |
| text top-k KD | 0.08 |
| trajectory top-k KD | 0.12 |
| trajectory hidden alignment | 0.08 |
| text boundary hidden alignment | 0.05 |
| format CE | 0.20 |

K512 first result on val512:

| Model | ADE@6.4s | minADE6@6.4s |
|---|---:|---:|
| B0 step_006250 | `3.2304` | `1.6886` |
| FLEX K512 final | `3.7973` | `1.9947` |

K512 was learning but not equivalent. The minADE gap was smaller than the greedy gap, meaning useful modes existed but readout/stability was worse.

K1024 was tested as a capacity ablation, then K512 continuation was run. The current registry's main FLEX backbone is:

```text
outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final
```

Training summary:

```text
20K samples
6 epochs total: initial K512 3ep + K512 continuation 3ep
K1024 was a separate capacity ablation, not the direct source of the K512 weights
LR: 1e-5 on continuation
FLEX scene LR scale: 20x in registry for the final branch
```

### 8.6 FLEX AE retraining

The user correctly rejected evaluating FLEX by plugging a barely trained FLEX backbone into an old AE. The AE must be retrained because:

```text
No-FLEX KV cache: 2880 vision-token interface
FLEX KV cache: 512 compressed visual-token interface
```

FLEX AE28 training:

```text
backbone: mlflex_k512.../final
AE: 28 layers
train: 18,000 samples
steps: 10,000
batch: 16
num_time_samples: 16
prefix: student_free
target: teacher
expert LR: 1e-4
attention: official_none
best: step 7500
```

Evaluation and deployment-facing comparisons should use student-generated
prefix/CoT. Report 138 explicitly uses self-generated CoT/prefix for all four
models on semantic val806.

Critical bug fixes before/around FLEX AE:

1. `rope_deltas` offset used pre-compression sequence length while KV used post-compression length.
2. Missing `--preserve-flex-positions` could crash or mis-handle position deltas.
3. Data prefetch was added to overlap image/token/FLEX batch building with GPU AE training, roughly 2x speedup.

Same val512 comparison from report 132:

| Model | ADE@6.4s | minADE6@6.4s |
|---|---:|---:|
| B0 Q3 best AE | `3.0542` | `1.6450` |
| FLEX K512 AE best | `3.1811` | `1.7566` |
| Gap | `+0.1269` | `+0.1116` |

This was encouraging on that split. It did not mean FLEX was globally B0-equivalent.

### 8.7 Harder semantic benchmark

Report 138 is the current clean comparison because all models run on the same semantic val806 benchmark and generate their own CoT/prefix.

| Model | N | ADE GT | FDE GT | minADE6 GT | minFDE6 GT | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| Alpamayo-1.5-10B | 806 | `1.6742` | `4.8004` | `0.9280` | `2.7102` | `1917.3970` |
| Student-2B-NoFLEX-AE28 | 806 | `2.7227` | `8.1559` | `1.6835` | `5.0521` | `616.2138` |
| Student-2B-FLEXK512-AE28 | 806 | `3.0818` | `9.3282` | `2.0721` | `6.2832` | `525.1725` |
| Student-2B-FLEXK512-AE14 | 806 | `3.1970` | `9.6055` | `2.5478` | `7.6595` | `493.0510` |

Interpretation:

```text
FLEX K512 reduces latency but currently costs quality on the harder balanced benchmark.
AE14 4-step is faster but its minADE6 drop is substantial on val806.
The val512 result was useful, but val806 is now the better decision benchmark.
```

---

## 9. AE14 / Export / Deployment State

### 9.1 AE14

AE14 was made by selecting 14 layers from the 28-layer FLEX AE:

```text
[0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27]
```

Best checkpoint:

```text
outputs/action_expert/ae14_from_ae28_10step/best.pt
best step: 7500
```

Training val metrics:

| Step | ADE | FDE |
|---:|---:|---:|
| 2500 | `2.9029` | `8.6751` |
| 5000 | `2.6976` | `7.9893` |
| 7500 | `2.5665` | `7.5872` |
| 10000 | `2.6948` | `8.0342` |

Small 68-sample step ablation:

| Denoise steps | ADE mean | FDE mean | latency mean |
|---:|---:|---:|---:|
| 10 | `3.4499` | `9.5739` | `130.8 ms` |
| 4 | `3.4449` | `9.5529` | `52.7 ms` |

That 68-sample result justified testing 4-step for latency, but report 138 showed AE14 4-step is weaker on the real semantic val806 benchmark.

### 9.2 ONNX export

FLEX ONNX was corrected in report 136.

Old broken contract:

```text
final_visual -> scene_embeds
```

Correct contract:

```text
inputs:
  ds_level0       [B, n_vis, 2048]
  ds_level1       [B, n_vis, 2048]
  ds_level2       [B, n_vis, 2048]
  final_visual    [B, n_vis, 2048]
  camera_ids      [B, n_vis]
  relative_times  [B, n_vis, 1]

outputs:
  scene_embeds       [B, 512, 2048]
  deepstack_scene_0  [B, 512, 2048]
  deepstack_scene_1  [B, 512, 2048]
  deepstack_scene_2  [B, 512, 2048]
```

Current export artifacts:

```text
outputs/trt_export/flex_k512_fp16/visual/model.onnx
outputs/trt_export/flex_k512_fp16/flex/flex_encoder.onnx
outputs/trt_export/flex_k512_fp16/llm/model.onnx
outputs/trt_export/flex_k512_fp16/ae28/ae28_single_step.onnx
outputs/trt_export/flex_k512_fp16/ae14/ae14_single_step.onnx
```

Remaining deployment wiring:

```text
visual.deepstack_features_0/1/2 + visual.output
  -> FLEX 4-level inputs

FLEX scene_embeds
  -> LLM compressed image placeholder positions

FLEX deepstack_scene_0/1/2
  -> scatter into full-sequence LLM DeepStack tensors
     at the same compressed image positions

AE14 runtime
  -> gather selected 14 KV layers from the 28-layer LLM cache
```

---

## 10. QAT / INT4 Plan

Target deployment plan:

```text
Jetson Thor
TensorRT-Edge-LLM
LLM INT4 AWQ
Q/K/V/O, softmax, RMSNorm/LayerNorm FP16 as needed
ViT/FLEX/AE initially FP16 unless separately optimized
```

The high-level QAT plan in report 133 is sensible:

```text
1. Load base Cosmos-Reason2-2B
2. Load existing LoRA as trainable
3. Load FLEX scene encoder
4. Apply deployment-format AWQ fake quantization
5. Fine-tune LoRA/FLEX to absorb quantization error
6. Merge/export
7. Retrain AE on the quantized backbone representation
8. Export to TensorRT-Edge-LLM
```

But report 133's follow-up verification found the repo path was not ready:

1. The launch wrapper path was broken.
2. The reported command used `--config` but `09_train_distill.py` expects `--stage-config`.
3. Quantization was applied to `raw_model.backbone`, which may include ViT/merger, not only the intended LLM language submodule.
4. QAT checkpoint save/load did not clearly preserve ModelOpt quantizer state.
5. Calibration counted failed forwards as progress.
6. TensorRT-Edge-LLM CLI tools were not present in the current environment, so target export commands were not locally verified.

Current judgment:

```text
QAT direction: right.
Current implementation: do not launch as trusted production QAT until the save/load and module-scope contracts are fixed.
```

---

## 11. Current Best Numbers by Context

### 11.1 Discrete-token backbone, no AE

| Model | Eval | ADE | minADE |
|---|---|---:|---:|
| B0 `step_006250` greedy | full val 4760, GT | `3.011` | - |
| B0 `step_006250` best-of-4 oracle | full val 4760, GT | `1.702` | best-of-4 |
| FLEX K512 first 3ep | val512, GT | `3.7973` | `1.9947` |
| B0 same val512 comparator | val512, GT | `3.2304` | `1.6886` |

### 11.2 Action Expert, older val512/Stage1 contexts

| Model | Eval | ADE | minADE@6 |
|---|---|---:|---:|
| Q3 temp 1.0 | Stage1 val512 | `2.631` | `1.243` |
| Q3 temp 0.85 | Stage1 val512 | `2.506` | `1.276` |
| B0 Q3 best AE | FLEX val512 split | `3.0542` | `1.6450` |
| FLEX K512 AE best | same FLEX val512 split | `3.1811` | `1.7566` |

### 11.3 Current semantic val806 benchmark

| Model | ADE GT | minADE6 GT | Latency |
|---|---:|---:|---:|
| 10B teacher | `1.6742` | `0.9280` | `1917 ms` |
| 2B NoFLEX AE28 | `2.7227` | `1.6835` | `616 ms` |
| 2B FLEX K512 AE28 | `3.0818` | `2.0721` | `525 ms` |
| 2B FLEX K512 AE14 4-step | `3.1970` | `2.5478` | `493 ms` |

This table is the best current global comparison because it uses the same semantic benchmark and self-generated CoT/prefix for all models.

---

## 12. What We Actually Learned

### 12.1 Dataset / teacher

- The teacher data must be treated as a structured cache, not one blob of labels.
- CoT, discrete trajectory tokens, top-k KD, hidden states, boundary states, and AE behavior are different channels.
- Teacher manifest provenance and position contracts matter. Misaligned KD/hidden labels can make losses improve while behavior worsens.
- Semantic balancing was necessary because easy straight-driving samples masked hard turns/intersections.
- VQA labels have the same structured-label problem. Official Q2 is currently the stable teacher prompt shape; raw Q1 / 1A grounding prompts are too weak unless strictly filtered.
- Q2-derived Q1 labels are acceptable as auxiliary grounding data, but they should not replace Q2 as the main distillation target.

### 12.2 Backbone

- The clean token path is the backbone success story.
- All-layer LoRA rank 64 plus trainable special token rows was the key recovery.
- Hidden/action objectives were useful diagnostically but harmful when they dominated or were ambiguously aligned.
- `step_006250` is strong because decode geometry selected it, not because it had the lowest val loss.
- Greedy decode is a weak diagnostic for a multimodal trajectory distribution.

### 12.3 Scheduled sampling / DAgger

- The motivation was correct: reduce train/inference prefix mismatch.
- The tried implementation failed because generated-prefix states were too off-manifold and the corrective target was not strong enough.
- It should not be the default next move unless redesigned around a stronger selector/ranker or better off-policy correction.

### 12.4 Action Expert

- AE training is not just "attach a head."
- The AE consumes backbone KV/cache distribution. If the student backbone was trained mainly for token readout, the hidden/KV distribution can still be bad for AE.
- Official FM details matter: Beta sampler, velocity target, Euler steps, LR, warmup/schedule.
- Our `num_time_samples=16` is a practical stabilizer for a colder student AE, not a public official setting.
- The deployable gap is partly selection/ranking: oracle trajectories exist but are not always selected.
- Path-only AE reranking did not beat `mean_traj`; the next selector needs richer prefix/token/diffusion/value features.

### 12.5 FLEX

- Paper-style single-level FLEX is not directly compatible with our current Qwen3-VL DeepStack student.
- DeepStack OFF is not viable for the AE path.
- The correct adaptation is ML-FLEX: compress final visual stream and all three DeepStack streams into the same 512-slot layout.
- FLEX K512 works, but the harder val806 benchmark shows a real quality cost.
- FLEX is still deployment-relevant because it reduces sequence length and latency, but it is not yet a free lunch.

### 12.6 Deployment

- FP16 ONNX modular export is now structurally closer: visual, FLEX, LLM, AE.
- Runtime wiring remains non-trivial because DeepStack tensors must be scattered into full sequence positions.
- QAT is needed if INT4 AWQ quality loss is non-negligible, but current QAT code needs hardening before serious training.
- The 100 ms target likely requires multiple levers together:
  - FLEX or stronger visual-token reduction
  - INT4 LLM
  - AE14 or step-reduced AE
  - TensorRT engine build
  - careful runtime tensor wiring and KV reuse

---

## 13. Recommended Next Direction

1. Treat report 138 semantic val806 as the default benchmark.
2. Keep B0 NoFLEX AE28 as the quality reference.
3. Keep official Q2 as the main VQA/grounding distillation source; use Q1 / 1A only as strict-filter auxiliary data.
4. Continue FLEX only if every change is judged on semantic val806, not just val512.
5. Fix QAT implementation contracts before launching expensive QAT:
   - module scope
   - calibration success accounting
   - quantizer save/load
   - AE loader using the quantized representation
6. For latency, do not trust 68-sample AE14 step ablation alone; AE14/4-step needs category-wise val806 analysis.
7. If FLEX quality remains too costly, compare against processor-level visual-token budget reduction and FocusUI-style token selection as a separate baseline.
8. Add a deployable selector/ranker study because both discrete and AE paths repeatedly show a large oracle-vs-selected gap.
9. Add a grounded-language audit metric beside ADE/minADE: Q2 visual support rate, unsupported-motion/action rate, teacher-intent agreement, and direction-conflict rate.

---

## 14. Source Map

Core reports:

```text
reports/closed_issues/009-qwen3vl-student-wrapper-compat.md
reports/closed_issues/020-spec-design-issues-so-far.md
reports/closed_issues/029-teacher-traj-cache-design.md
reports/closed_issues/035-p0-manifest-and-p2-paired-interface-v2.md
reports/closed_issues/039-mixed-from-format-restores-traj-emission.md
reports/closed_issues/040-hybrid-traj-decode-breaks-body-collapse.md
reports/closed_issues/048-h0a-frozen-latent-958-hidden-recovery.md
reports/058-current-open-trajectory-quality-after-collapse-fix.md
reports/059-claude-findings-and-h200-transition.md
reports/060-student-ae-pairing-4m-ceiling-root-cause.md
reports/072-long-horizon-sampling-diagnostics.md
reports/073-temperature-selection-deployable-fix.md
reports/074-stage1-heldout-split-and-sanity.md
reports/075-official-alpamayo-sft-fm-hparams.md
reports/080-stage1-heldout-training-results.md
reports/081-q1-inference-sweep-results.md
reports/085-current-metrics-summary.md
reports/125-flex-v2-redesign-with-deepstack.md
reports/126-mlflex-implementation-smoke.md
reports/129-mlflex-k512-results-and-k1024-launch.md
reports/132-b0-vs-flex-k512-ae-val512.md
reports/133-qat-pipeline-prep-and-ae-results.md
reports/135-checkpoint-registry.md
reports/136-flex-onnx-deepstack-export-fix.md
reports/137-ae14-step-ablation-and-onnx-export.md
reports/138-semantic-val806-4model-benchmark.md
reports/140-ae-path-reranker-bootstrap.md
reports/2026-05-27_multimodal_input_backbone_ae_summary.md
outputs/reports/student_noflex_ae28_pairwise_reasoning_20260622/report.md
../../alpamayo_repo/alpamayo1.5/reports/vqa_q1_grounding_stability_20260621.md
```

Core code/config:

```text
scripts/42_build_no_nav_teacher_pair_corpus.py
scripts/85_build_semantic_scene_balanced_corpus.py
scripts/09_train_distill.py
scripts/84_train_student_ae28_official.py
scripts/export_flex_ae_onnx.py
src/model/student_wrapper.py
src/model/flex_scene_encoder.py
src/training/losses.py
src/training/trainer.py
configs/train/stage_bp3_no_nav_camera_labeled_full444k_balanced_hidden_gc.yaml
configs/train/stage_mlflex_k512_bp3_hidden_gc_20k_e3.yaml
configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml
```
