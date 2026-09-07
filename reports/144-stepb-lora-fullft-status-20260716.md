# Report 144: Step-B LoRA vs FullFT Status and Current Direction

Date: 2026-07-16
Status: active
Scope: Cosmos-Reason2-2B Step-B backbone distillation, no-FLEX discrete VLM path

## 0. One-line summary

지금까지의 LoRA vs FullFT 실험은 "loss가 낮은 모델"이 아니라 "free-run geometry와 self-prefix rollout이 좋은 모델"을 골라야 한다는 쪽으로 정리됐다. 200K 기준 FullFT-3e-5가 평균 geometry는 LoRA-2e-4보다 좋지만, frozen val512 paired CI는 아직 0을 포함하므로 통계적으로 확정 champion이라고 부르지는 않는다. 현재는 FullFT 계열에서 trajectory KD가 실제로 도움이 되는지 확인하는 R1 sign-check를 돌리는 중이다.

## 1. Current running job

tmux session:

```text
stepb_r1_signcheck_20260716
```

Run id:

```text
r1_signcheck_20260716_0726
```

Launcher:

```text
scripts/launch_r1_signcheck_baseline_then_train.sh
```

Current queue:

1. `r0f20k` baseline P-G vs `teacher_greedy_ref_v1` - done
2. `fullft200k` baseline P-G vs `teacher_greedy_ref_v1` - running at last check
3. `lora200k` baseline P-G
4. `r0f20k` matched argmax
5. clean-bucket audit
6. P-S6 minADE6
7. R1 FullFT KD train

Completed baseline artifact:

```text
outputs/reports/stepb_r1_signcheck/r1_signcheck_20260716_0726/baseline/r0f20k_pg_teacher_ref_summary.json
```

`r0f20k` new-reference baseline:

| metric | value |
|---|---:|
| n | 512 |
| reference | `teacher_greedy_ref_v1` |
| ADE vs teacher-greedy-ref | 3.1217 |
| FDE vs teacher-greedy-ref | 9.7413 |
| bad geometry rate | 0.0957 |
| avg unique traj ids | 14.1992 |
| missing / excluded refs | 0 / 0 |
| truncation count | 0 |

This number is the correct comparison line for the 20K R1 sign-check. It must not be compared against the 200K champion baseline as a KD decision.

## 2. Why this experiment exists

The original Step-B question was whether the 2B backbone should be trained with LoRA or FullFT for CoC + 128 discrete trajectory-token imitation.

The important constraints are:

- Vision tower is frozen.
- Multimodal projector / merger remains trainable.
- LoRA arms apply LoRA to the LLM path, with trainable new action-token rows.
- FullFT arms disable LoRA and train the language backbone plus trainable token embeddings / LM head, while still freezing ViT.
- Target is teacher CoT plus teacher 128 trajectory tokens.
- Main checkpoint selection should use free-run geometry, not teacher-forced loss.

The current conclusion is not "FullFT always wins". The more precise conclusion is:

> LoRA can lower teacher-forced CE loss aggressively, but FullFT has shown better free-run geometry at the same scale, which suggests better self-prefix / KV / rollout behavior. The gap is directionally consistent but not yet statistically sealed on frozen val512.

## 3. 20K ladder and follow-up

Initial 20K ladder showed that LR mattered heavily. Low-LR FullFT underfit, and LoRA at higher LR improved sharply. Follow-up probes then compared FullFT-3e-5 and LoRA-2e-4.

Reference note: the decode metrics in this table are from the old ladder monitor path, before the current frozen teacher-greedy reference protocol. They are useful for within-run trend reading, not as final absolute metrics.

| run | train size | LR | train mode | best step | val loss | monitor ADE | monitor FDE | bad geom | unique |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| R0-C soup LoRA | 20K | 2e-5 | LoRA r64 | 5000 | 2.058 | 5.535 | 15.679 | 0.254 | 3.23 |
| R0-L CE LoRA | 20K | 2e-5 | LoRA r64 | 7500 | 1.886 | 5.923 | 17.120 | 0.266 | 3.59 |
| R0-L' CE LoRA | 20K | 1e-4 | LoRA r64 | 7500 | 1.455 | 3.923 | 12.437 | 0.156 | 10.93 |
| FullFT | 20K | 5e-6 | FullFT | 7500 | 2.566 | 5.594 | 15.836 | 0.250 | 3.52 |
| FullFT | 20K | 1e-5 | FullFT | 7500 | 1.763 | 4.708 | 14.006 | 0.227 | 6.47 |
| FullFT | 20K | 2e-5 | FullFT | 7500 | -- | -- | -- | -- | -- |
| FullFT follow-up | 20K | 3e-5 | FullFT | 7500 | 1.463 | 3.526 | 11.334 | 0.125 | 13.66 |
| LoRA follow-up | 20K | 2e-4 | LoRA r64 | 7500 | 1.433 | 3.611 | 11.376 | 0.137 | 13.93 |

Interpretation:

- `2e-5 -> 1e-4` was the first major LoRA unlock.
- FullFT also improved monotonically with LR, so earlier FullFT failures were not decisive.
- At 20K follow-up, FullFT-3e-5 and LoRA-2e-4 are close; FullFT is slightly better in geometry, LoRA slightly lower in val loss.
- This already shows that val loss alone is not a safe checkpoint-selection metric.

## 4. 200K double promotion

Both practical finalists were promoted to 200K balanced, 1 epoch.

Reference note: these are still the old monitor decode metrics. The new protocol uses frozen `teacher_greedy_ref_v1`; current R1 baseline package is rebuilding the comparable numbers.

| run | train size | LR | train mode | best step | val loss | monitor ADE | monitor FDE | bad geom | unique |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| FullFT-3e-5 | 200K | 3e-5 | FullFT | 25000 | 1.349 | 2.356 | 7.602 | 0.043 | 22.30 |
| LoRA-2e-4 | 200K | 2e-4 | LoRA r64 | 25000 | 1.347 | 2.469 | 8.122 | 0.063 | 19.98 |

Interpretation:

- LoRA still has slightly lower validation loss.
- FullFT is better on free-run geometry: lower ADE, lower FDE, lower bad geometry, higher token diversity.
- This is the main evidence for the "FullFT improves rollout/KV behavior better than LoRA" hypothesis.

## 5. Frozen val512 scoreboard

Frozen val512 uses GT as product scoreboard reference. This table should not be mixed with old ladder monitor numbers.

Artifact:

```text
outputs/benchmarks/val512_full_metrics_10b_fullft_lora_20260715/combined_val512_ade_minade6_summary.json
```

| model / path | ADE vs GT | FDE vs GT | minADE6 vs GT | minFDE6 / FDE of minADE path |
|---|---:|---:|---:|---:|
| Alpamayo-1.5-10B VLM discrete | 2.115 | 6.511 | 1.041 | 3.053 |
| Alpamayo-1.5-10B + Action Expert | 2.058 | 6.133 | 0.985 | 2.803 |
| FullFT-3e-5-200K | 3.157 | 10.150 | 1.645 | 4.879 |
| LoRA-2e-4-200K | 3.221 | 10.199 | 1.710 | 4.996 |

Interpretation:

- FullFT is directionally better than LoRA on both greedy and minADE6.
- Both student backbones remain far behind the 10B discrete single path on greedy ADE.
- Both students have much better minADE6 than greedy, meaning the distribution contains useful paths but selection is still weak.
- The real deployable path is expected to use AE / selector, so greedy LM-head decoding is a diagnostic path, not the final deployment path.

## 6. FullFT vs LoRA paired CI

Artifact:

```text
outputs/reports/stepb_200k_double_promotion/double200k_20260711_181751/val512_fullft3e5_vs_lora2e4_ci.json
```

Delta is `FullFT - LoRA`; negative is better for ADE/FDE/bad geometry.

| metric | delta | 95% CI | read |
|---|---:|---:|---|
| ADE | -0.064 | [-0.213, 0.083] | FullFT mean better, not statistically sealed |
| FDE | -0.049 | [-0.577, 0.463] | tied |
| bad geometry | -0.002 | [-0.021, 0.018] | tied |
| unique ids | +1.047 | [-0.668, 2.826] | FullFT mean more diverse, not sealed |

Conclusion:

FullFT-3e-5-200K is the practical candidate, but not a statistically final champion yet. The correct wording is "directionally better and operationally preferred for the next sign-check", not "proven winner".

## 7. Teacher/cache audit and why the reference changed

Artifact:

```text
outputs/benchmarks/val512_teacher_cache_offline_audit_20260715/summary.json
```

| audit | value |
|---|---:|
| current 10B greedy vs cached sampled teacher ADE | 1.211 |
| current 10B greedy vs cached sampled teacher FDE | 3.832 |
| token match | 9.81% |
| cache target top-1 rate | 53.85% |
| cache target in top-k32 | 99.62% |
| top-k32 probability mass mean | 0.9939 |
| teacher cache vs GT ADE | 1.999 |
| teacher cache vs GT FDE | 5.954 |

Interpretation:

- The cached trajectory target was generated with sampling, so it is not the same object as current 10B greedy.
- vs-cache metrics have an inherent 1.21m teacher sampling/regeneration floor.
- Therefore absolute vs-cache ADE is not a valid final monitor.
- New evaluations should use `teacher_greedy_ref_v1` as the frozen imitation reference, while cache remains only a legacy/training-target artifact.

## 8. Token diagnostics

Matched-conditioning trajectory argmax artifact:

```text
outputs/benchmarks/val512_matched_conditioning_argmax_top1_20260716/final/summary.json
```

| metric | FullFT-3e-5-200K | LoRA-2e-4-200K |
|---|---:|---:|
| student argmax == teacher top-1 | 66.98% | 66.73% |
| student argmax in teacher top-5 | 94.64% | 94.37% |
| student argmax in teacher top-10 | 98.18% | 98.10% |
| student argmax == cached sampled target | 47.63% | 47.70% |

Text top-8 ceiling:

```text
outputs/reports/stepb_200k_double_promotion/double200k_20260711_181751/text_top8_ceiling_val512_20260716.json
```

Summary:

- CoT sampled-target top-1 ceiling is roughly 92-94%.
- Existing CoT CE accuracy around 90.5% is already close to that ceiling.
- Text KD is therefore not the first lever.
- Trajectory ranking / mode selection is the more relevant KD target.

Interpretation:

- The student usually lands inside the teacher's top-k support.
- The weak part is top-1 ranking / mode selection, not simply "does it know the candidate set".
- This is why R1 tests trajectory top-k KD rather than adding more plain CE or text KD.

## 9. R1 KD sign-check

Correct R1 config:

```text
configs/train/stepb_ladder_r1_tailkl.yaml
```

Required properties:

- base lineage: FullFT-3e-5 recipe, not stale LoRA L' config
- `--disable-lora`
- `learning_rate: 3e-5`
- loss:
  - CoT CE: 0.1
  - trajectory CE: 1.0
  - trajectory top-k KD: 0.5
- KD temperature: 1.0
- tail bucket: enabled
- text KD: disabled
- ViT: frozen
- LLM and multimodal projector: trainable
- decode eval:
  - `geometry_reference_id: teacher_greedy_ref_v1`
  - `do_sample: false`
  - `max_new_tokens: 320`

Why R1 is 20K first:

- KD in this exact regime has not been validated yet.
- Old "soup" KD evidence is not reusable because it used old support-restricted KL and a different LoRA/LR setting.
- A wrong 200K KD run costs about a day; a 20K sign-check is the cheap way to get the sign.

Decision rule:

| result | action |
|---|---|
| strong positive: R1 improves greedy vs teacher-ref with CI excluding 0 | run 200K + KD, then use that as 444K recipe |
| weak positive: 3/4 registered metrics move correctly but CI includes 0 | run 200K + KD with flag |
| negative / reversed | stop KD, go CE-only toward 444K |

The R1 baseline must compare against R0-F-20K, not the 200K champion. Otherwise KD effect and data-scale effect are mixed.

## 10. Current direction

Near-term:

1. Finish the R1 baseline package now running in tmux.
2. Run R1 FullFT trajectory-KD 20K sign-check.
3. Evaluate R1 vs R0-F-20K on frozen val512 with paired CI.
4. If positive, promote to 200K+KD and then 444K.
5. If negative, close KD and continue CE-only scaling.

Parallel / next track:

- AE smoke pair: compare AE behavior on teacher-KV vs champion-student-KV under matched budget.
- Selector/reranker: minADE6 is much better than greedy, so deployable quality depends heavily on path selection.
- Frozen TEST split remains unbuilt; final claims should not use val806/val512 as final test.

Important practical point:

The final deployed path is expected to go through Action Expert / selector, not pure LM-head greedy token decoding. Therefore greedy discrete ADE is mainly a backbone diagnostic and training-monitor metric. However, it still matters because AE consumes the backbone KV/state, and poor self-prefix/rollout behavior can indicate weak conditioning even when teacher-forced loss looks good.

## 11. Workspace hygiene done today

Cleanup completed before this report:

- Removed cache directories and temporary smoke/sanity outputs.
- Reduced `outputs/kv_distill_pipeline` from about 718G to about 21G by deleting intermediate `step_*.pt` while preserving finals and logs.
- Reduced `outputs/checkpoints/stepb_ladder_20k` from about 165G to about 16G by preserving `best_decode` and deleting intermediate step checkpoints.
- Removed small smoke/verify action-expert outputs.
- Left large but still relevant artifacts intact:
  - `outputs/action_expert`
  - `outputs/ae28_teacher_dumps`
  - `outputs/trt_export`
  - `outputs/exports`

Do not delete the remaining AE/export artifacts without an explicit follow-up decision; they may be needed for AE transfer, deployment export, or reproducibility.

## 12. Main caveats

- Old ladder monitor numbers and new frozen-reference numbers are not directly comparable.
- The 200K FullFT vs LoRA result is directionally favorable to FullFT, but paired CI still crosses zero.
- CoT CE accuracy is near sampled-target ceiling, so extra CoT CE/KD is unlikely to be the first-order lever.
- Cached teacher targets are sampled; teacher greedy is a different reference object.
- Token ID free-run matching is not a valid model-quality metric here. Geometry and matched-conditioning distribution diagnostics are the valid spaces.

