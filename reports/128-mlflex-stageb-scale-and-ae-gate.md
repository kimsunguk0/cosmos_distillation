# 128 - ML-FLEX Stage B Scale and Action Expert Gate

**Date:** 2026-06-08  
**Status:** Stage B 16-sample checkpoint passes wiring, but is not expected to pass the existing action expert without further adaptation  

## Objective

Continue from report 127 and check whether the 16-sample Stage B checkpoint generalizes beyond the tiny gate.

Important interpretation: this is not a final action-expert quality test. The existing action expert was trained on dense B0 hidden/KV distributions. A lightly trained ML-FLEX backbone presents a different compressed hidden/KV distribution, so poor AE quality is expected. The purpose of this report is to verify that the FLEX path reaches the action expert correctly and to measure the size of the mismatch before deciding whether to scale Stage B or move to action-expert adaptation.

The evaluated checkpoint is:

```text
outputs/checkpoints/mlflex_stageb_task_gate16_s500_20260608/final
```

## Artifacts

Teacher-parity 256:

```text
outputs/reports/mlflex_stageb_task_gate16_s500_parity256_summary.json
outputs/logs/mlflex_stageb_task_gate16_s500_parity256.log
```

Action expert eval support:

```text
scripts/launch_mlflex_stageb_ae64_compare.sh
outputs/logs/mlflex_stageb_ae_smoke4_20260608_retry.log
outputs/logs/b0_ae_smoke4_20260608.log
outputs/logs/mlflex_stageb_ae64_compare_20260608.log
outputs/reports/b0_ae64_20260608_summary.json
outputs/reports/mlflex_stageb_ae64_20260608_summary.json
```

Code support added:

- `scripts/84_train_student_ae28_official.py`: FLEX-aware teacher-forced AE prefill path.
- `scripts/85_eval_ae28_best_of_n.py`: optional `--eval-summary-json`.

## Teacher-Parity Scale Check

Same dense B0 teacher, same Stage B ML-FLEX checkpoint.

| Metric, mean | 16 samples | 256 samples | Direction |
|---|---:|---:|---|
| traj teacher-student KL | 0.0099 | 0.0502 | worse |
| traj top1 agreement | 0.9136 | 0.8546 | worse |
| teacher top1 in student top5 | 0.9985 | 0.9914 | worse |
| text teacher-student KL | 0.0606 | 0.3904 | worse |
| text top1 agreement | 0.9507 | 0.7986 | worse |
| action-pre hidden cosine | 0.9905 | 0.9817 | worse |
| action-pre norm ratio | 1.0073 | 1.0426 | drift |
| student TF argmax ADE | 0.1261 m | 0.1205 m | similar |
| student - teacher TF argmax ADE | 0.0175 m | 0.0397 m | worse |

Interpretation: the 16-sample Stage B run is a valid wiring/adaptation gate, but it is not a general checkpoint. The token/logit interface regresses substantially on 256 heldout samples, especially text KL and trajectory top1 agreement.

## Action Expert Path

The action expert eval path was patched to handle ML-FLEX teacher-forced prefix correctly:

1. compress image placeholders before student prefill,
2. preserve Qwen MRoPE position ids,
3. preserve `rope_deltas` for expert position construction,
4. call the `StudentWrapper` for FLEX checkpoints instead of bypassing FLEX through `student.backbone`.

The current eval is intentionally teacher-forced prefix, not student-free CoT generation. This isolates the action-expert hidden/KV path from CoT generation errors.

Settings:

```text
corpus: data/corpus/flex_heldout256_stage2val_seed42.jsonl
samples: first 64 val rows
AE checkpoint: outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt
prefix_mode: teacher_forced
target_source: gt
eval_num_paths: 1
eval_selection_method: single
ML-FLEX flags: preserve-flex-positions, uniform selection, scene DeepStack on
```

Smoke-4 result:

| Model | ADE m | FDE m |
|---|---:|---:|
| B0 dense | 1.532 | 5.083 |
| ML-FLEX Stage B | 2.258 | 6.638 |

64-sample result:

| Model | ADE m | FDE m | h1.6 ADE | h3.2 ADE |
|---|---:|---:|---:|---:|
| B0 dense | 3.261 | 9.726 | 0.217 | 0.847 |
| ML-FLEX Stage B | 7.808 | 22.912 | 0.487 | 1.929 |
| Delta | +4.548 | +13.186 | +0.271 | +1.082 |
| Relative | +139.5% | +135.6% | +125.0% | +127.7% |

Interpretation: the action expert path is much more damaged than the teacher-forced token-path metric suggests. This is expected because the existing action expert has not been trained on FLEX-conditioned hidden/KV. The result is still useful: it confirms that the old action expert cannot be treated as plug-compatible with a lightly adapted FLEX backbone.

## Conclusion

Current checkpoint status:

```text
ML-FLEX Stage B 16-sample wiring gate: pass
ML-FLEX Stage B 256-sample teacher parity: not ready
Existing B0 action expert on ML-FLEX hidden/KV: expected mismatch, not a final-quality failure
```

This does not invalidate the ML-FLEX design. It says the current training scale is too small to make the compressed backbone interface general, and the existing B0 action expert should not be expected to work unchanged on a lightly adapted FLEX hidden/KV distribution.

## Next Step

Do not fine-tune the action expert yet from this checkpoint. First scale Stage B itself:

```text
Stage B-256:
  init: outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608/final
  train samples: 256
  train: ML-FLEX + all LoRA
  keep LR split: FLEX 5e-5, LoRA 1e-6
  monitor: 256 teacher parity, AE64 teacher-forced, free-run discrete decode
```

If Stage B-256 recovers teacher-parity but AE64 remains bad, then Stage C is justified:

```text
Stage C:
  freeze ML-FLEX + LoRA
  fine-tune or adapt action expert on FLEX-conditioned hidden/KV
  evaluate minADE@6 and B0-relative degradation
```

If Stage B-256 still fails teacher-parity, the issue is before the action expert: the compressed backbone interface is not generalizing and needs a stronger Stage B objective or larger K.
