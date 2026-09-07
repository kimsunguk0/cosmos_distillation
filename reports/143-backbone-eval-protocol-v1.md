# Backbone Evaluation Protocol v1

Date: 2026-07-16

Status: FROZEN v1 for protocol semantics. Existing numeric anchors with incomplete
stamps are corrected here and remain `TBD` where the required frozen reference
has not been generated yet. Any later change requires a v2 document with a diff
and reason.

Scope: VLM backbone discrete-path evaluation. Action Expert and deployable
evaluation remain separated in section 7.

## 0. Principles

Every reported number must carry this 5-tuple stamp:

```text
(split, reference, decode config, checkpoint step, commit hash)
```

Numbers without this stamp are not valid for meetings, Notion, checkpoint
selection, or method decisions.

Rules:

- Do not compare numbers across splits arithmetically.
- Use one geometry scoring code path for ladder `decode_eval`, frozen evaluation,
  and benchmark reports.
- Freeze every reference after generation. Do not regenerate references inside
  evaluation runs.
- Store the reference generation config: model, revision/path, precision,
  do_sample, temperature, top_p, top_k, seed, prompt version, date, and commit.
- Do not use free-run token ID match as an evaluation metric. The 10B control
  only reached 11.55% against cached-prefix teacher top-1, so this metric is
  invalid under self-prefix rollout.
- Use geometry space (ADE/FDE/minADE) and matched-conditioning distribution
  diagnostics instead.

## 1. Evaluation Sets And Reference Registry

| Set | Size | Purpose | Status |
|---|---:|---|---|
| `frozen_val512` | 512 | Final arm/recipe decision and target anchors | Decision set; avoid repeated probing |
| `decode_val256` | 256 | Training monitor, checkpoint selection, early stop | Monitor set; fixed seed within lineage |
| `val806_semantic` | 806 | Deployable / AE benchmark, report-138 protocol | Dev set, not test |
| `TEST` | TBD | Final one-time decision | Build from unused chunks; sealed after teacher inference |

Reference registry, per frozen split:

| Reference ID | Content | Generation | Use |
|---|---|---|---|
| `gt_v1` | Dataset ego future 6.4s | Dataset | Product scoreboard |
| `teacher_greedy_ref_v1` | 10B VLM discrete only, `do_sample=false`, no AE | Existing val512 artifact must be stamp-promoted; decode_val256 TBD | Main imitation reference |
| `teacher_sampled_t06_p098_n6_v1` | 10B VLM discrete only, `temperature=0.6`, `top_p=0.98`, `n=6` | Existing artifact | Historical sampled distribution anchor; not P-S6 |
| `teacher_cache_v1` | Training cache sampled target, generated with `temperature=0.6`, `top_p=0.98` | Existing training cache | Legacy target-quality and noise-floor reference only |

Do not use `teacher_cache_v1` as the main geometric reference. Its current
measured disagreement against the current 10B discrete greedy reference is
1.211m ADE, so absolute vs-cache values have a cache-sampling noise floor.

## 2. Metric Definitions

Trajectory body: 64 waypoints at 10Hz over 6.4s, decoded by the existing
detokenizer. The scoring plane is the xy plane unless the unified scoring module
explicitly says otherwise. M6 must write this into code and report metadata.

- `ADE@6.4s`: mean Euclidean distance over all 64 waypoints.
- `FDE@6.4s`: Euclidean distance at waypoint 64.
- `minADE6@6.4s`: minimum full-horizon ADE among six sampled paths.
- `selected-ADE`: ADE of a single selector-chosen path. Current selector may be
  `mean_traj`; future rerankers must be stamped.
- `body_completion_rate`: valid trajectory body length equals 128/128.
- `bad_geom_rate`: use the current checkpoint-eval definition, but M6 must write
  the exact predicate into the shared scoring module and this report lineage.
- `norm-acc`: teacher-forced token accuracy divided by the same-set sampled-target
  ceiling. Monitor only; not a decision metric.
- `matched_argmax@k`: under cached-prefix teacher forcing, whether student argmax
  is in the teacher top-k set. Diagnostic only.

## 3. Decode Protocols

### P-G: Greedy Single Path

Required config:

```text
do_sample=false
top_k=0
temperature=None or ignored by generate
CoT + 128 trajectory tokens full free-run
max_new_tokens >= CoT p99 + 128 + margin
```

Outputs:

- ADE/FDE vs `teacher_greedy_ref_v1` as the main imitation metric.
- ADE/FDE vs `gt_v1` as the scoreboard metric.
- Body completion rate.

Do not allow implicit generation defaults. The config must explicitly stamp
`do_sample=false`.

### P-S6: Sampled Six-Path Distribution

Required config:

```text
CoT greedy once
trajectory span sampled six independent times
temperature=1.0
top_p=1.0
top_k=0
seed = int(sha1(f"{sample_id}:{path_idx}")) % 2**32
```

Outputs:

- minADE6 vs `gt_v1`.
- oracle-min vs `teacher_greedy_ref_v1`.
- six-path pairwise spread.

Rationale:

- It matches the deployable pattern of one prefill plus several path candidates.
- Student CE targets came from teacher `T=0.6/top_p=0.98` draws. Sampling the
  student again at 0.6 can double-sharpen the distribution, so v1 uses T=1.0 for
  distribution evaluation.

### P-CAL: One-Time Sensitivity Appendix

Run once for calibration only:

- teacher at temperatures `{0.6, 1.0}`
- champion at temperatures `{0.6, 0.85, 1.0}`

This is not a frozen decision protocol. It is a sensitivity appendix.

### Training-Time Decode Eval

Use P-G on `decode_val256` every 0.2 epoch. The reference must be
`teacher_greedy_ref_v1` for `decode_val256`.

Checkpoint selection and early stop use greedy ADE-vs-`teacher_greedy_ref_v1`.
Do not select checkpoints by training loss or token accuracy.

## 4. Anchors And Targets

### 4.1 Current Anchors With Corrected Provenance

These are the currently verified artifacts. Names below supersede earlier loose
or ambiguous labels. The key correction is that the VLM discrete `n=1` artifact
was stamped with `temperature=0.6/top_p=0.98`, but the generating script sets
`do_sample = samples_per_row > 1`; with `samples_per_row=1` it is greedy.

| Anchor | Value | Status |
|---|---:|---|
| `teacher_greedy_ref_v1` vs `gt_v1` | ADE 2.115 / FDE 6.511 | Existing val512 VLM discrete greedy artifact; stamp promotion required |
| `teacher_cache_v1` vs `gt_v1` | ADE 1.999 / FDE 5.954 | Existing training-cache target quality |
| current `teacher_greedy_ref_v1` vs `teacher_cache_v1` | ADE 1.211 / FDE 3.832 | Existing cache-sampling noise-floor audit |
| `teacher_ae_official_v1` vs `gt_v1` | ADE 2.058 / minADE6 0.985 | AE/deployable anchor; not a backbone discrete reference |
| `teacher_discrete_t06_p098_n6` vs `gt_v1` | minADE6 1.041 | Existing sampled n=6 artifact, **not P-S6** |
| `teacher_greedy_ref_v1` on `decode_val256` | TBD | Must be generated with P-G, no AE |
| `teacher_discrete_PS6_v1` vs `gt_v1` | TBD | Must be generated with P-S6 |

Important corrections:

- The earlier confusion came from the summary carrying `temperature=0.6` and
  `top_p=0.98`. For `samples_per_row=1`, `eval_10b_backbone_discrete.py` sets
  `do_sample=false`; therefore `2.115 / 6.511` is the VLM discrete greedy
  reference for val512 once stamped.
- The earlier value `2.058 / 0.985` belongs to the Action Expert path and stays
  in section 7.
- `teacher_greedy_ref_v1` vs `teacher_cache_v1` remains a noise-floor audit, not
  a main reference metric.

### 4.2 Current Student Snapshot

Current champion: FullFT 3e-5, 200K, best-decode checkpoint.

| Metric | Value | Use |
|---|---:|---|
| P-G-like greedy vs GT | ADE 3.157 | Scoreboard, existing artifact |
| sampled n=6 minADE6 vs GT | 1.645 | Existing artifact; remeasure under P-S6 |
| oracle-min vs sampled 10B reference | 1.049 | Indicates useful modes in the distribution |
| matched argmax@1 / @5 / @10 | 67.0 / 94.6 / 98.2% | Diagnostic: support learned, top-1 ranking weak |
| sampled-target token acc | 47.63% | Monitor only |
| sampled-target ceiling | 53.85% | Monitor denominator only |
| clean bucket student-vs-reference ADE | 0.916 | Drift exists even in low-noise scenes |

Student-vs-`teacher_greedy_ref_v1` is valid only when the frozen reference rows
are explicitly loaded and stamped. Legacy vs-cache rows are not equivalent.

### 4.3 Targets

Targets are directionally frozen, but numeric thresholds depending on
`teacher_greedy_ref_v1` must be filled after M2.

| Milestone | Metric | Target |
|---|---|---|
| M1: KD sign-check + 444K | P-G ADE vs `teacher_greedy_ref_v1` | TBD baseline to improve |
| M1 | matched argmax@1 | 67% -> at least 75% |
| M1 | norm-acc | 88.4% -> at least 92% |
| M1 | clean-bucket vs `teacher_greedy_ref_v1` | TBD baseline to improve |
| M1 | P-S6 minADE6 vs GT | no degradation from P-S6 baseline |
| M2: reranker | selected-ADE vs GT, discrete six-path | recover at least one third of oracle-selection gap |
| M3: student + AE-v2 + selector | selected ADE / minADE6 on val806 report-138 protocol | within 15% of teacher AE |

## 5. Decision Mapping

| Decision | Metrics | Set |
|---|---|---|
| Checkpoint selection / early stop | P-G ADE vs `teacher_greedy_ref_v1` | `decode_val256` |
| Arm / recipe decision | P-G vs teacher, P-G vs GT, P-S6 minADE6, paired bootstrap CI | `frozen_val512` |
| KD sign-check | matched argmax@1 up, clean-bucket ADE down, P-G vs teacher down, P-S6 not worse | `frozen_val512` |
| Deployable decision | report-138 protocol, selected and oracle | `val806_semantic` |
| Final one-time decision | all relevant frozen metrics | sealed `TEST` |

Forbidden decision inputs:

- training loss
- free-run token ID match
- unstamped numbers
- cross-split arithmetic
- token accuracy as a final quality metric

## 6. Reporting Rules

- Every A/B claim needs per-sample paired bootstrap 95% CI.
- CI overlap means "tie" unless a preregistered tie-breaker applies.
- If a number is corrected later, keep the old line with a correction note rather
  than silently deleting it.
- Buckets with `n < 30` are anecdotal only.

## 7. Action Expert / Deployable Path

AE/deployable evaluation keeps the existing val806 report-138 protocol:

```text
N=6
temperature=0.85
mean_traj / oracle
self-generated prefix
```

Backbone discrete-path metrics do not replace AE metrics. If the backbone
changes, the AE must be retrained or at least treated as out-of-distribution with
that backbone. One backbone experiment equals backbone plus AE-transfer
evaluation.

## 8. Migration Actions

M0 report-level correction is complete in this document. Artifact-level
backfilling is still required before automated tooling can enforce the same
names and generation stamps.

- [x] M0a: Report-level provenance audit and canonical rename.
  - Promote `2.115 / 6.511` as val512 `teacher_greedy_ref_v1` only after adding
    the missing `do_sample=false` stamp.
  - Keep the `1.211` audit as `teacher_greedy_ref_v1` vs `teacher_cache_v1`
    noise-floor evidence, not as a main score.
  - Move AE `2.058 / 0.985` out of backbone references and keep it in section 7.
  - Mark decode_val256 `teacher_greedy_ref_v1` as TBD until P-G no-AE generation is complete.
- [ ] M0b: Backfill corrected reference names and 5-tuple stamps into summary
  JSON/manifests so tooling cannot reintroduce the old labels.
- [ ] M1: Replace `checkpoint_eval.py` reference from `hard_target`/cache to
  frozen `teacher_greedy_ref_v1`. Keep vs-cache only as a legacy auxiliary
  column with explicit reference stamp.
- [ ] M2: Generate `teacher_greedy_ref_v1` once for `frozen_val512` and
  `decode_val256`; write stamps and commit hash.
- [ ] M3: Remeasure teacher discrete and current champion with P-S6; replace all
  daggered `minADE6` placeholders.
- [ ] M4: Run P-CAL once and store it as a non-decision appendix.
- [ ] M5: Audit `max_new_tokens` against CoT p99 + 128 + margin. Explicitly set
  `do_sample=false` in all P-G configs.
- [ ] M6: Unify scoring modules across ladder, frozen eval, and benchmarks.
  Document bad-geometry predicate and xy/xyz scoring plane.
- [ ] M7: Add a common utility that enforces the 5-tuple stamp in all evaluation
  summary JSON files.
- [ ] M8: Build sealed TEST from reserved chunks, materialize it, and allow only
  teacher-reference inference before sealing.

## 9. Artifact Evidence Used For v1 Corrections

| Artifact | Fact |
|---|---|
| `outputs/benchmarks/val512_full_metrics_10b_fullft_lora_20260715/teacher10b_vlm_discrete_n1_summary.json` | `samples_per_row=1`; code path sets `do_sample=false`; ADE 2.115 / FDE 6.511 |
| `outputs/benchmarks/val512_teacher_cache_offline_audit_20260715/summary.json` | greedy-ref-vs-cache ADE 1.211, cache-vs-GT ADE 1.999 |
| `outputs/benchmarks/val512_full_metrics_10b_fullft_lora_20260715/combined_val512_ade_minade6_summary.json` | AE anchor 2.058 / 0.985 and student summary values |
| `outputs/benchmarks/val512_matched_conditioning_argmax_top1_20260716/final/summary.json` | matched argmax@1/@5/@10 diagnostics |
