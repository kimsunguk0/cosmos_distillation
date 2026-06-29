# FLEX F1 Parity Overfit Results

Date: 2026-06-05

## Setup

- Teacher: B0 no-FLEX checkpoint, frozen.
- Student init: F0 K896 camera/time FLEX checkpoint.
- Trainable params: `flex_scene_encoder` only, 31.38M / 2.264B total.
- Data: first 16 rows from `data/corpus/vis_4per_category_val.jsonl`.
- Training: 200 steps, batch 1, LR `2e-5`, cached B0 teacher sliced logits/hidden.
- Trainer: `scripts/105_train_flex_teacher_parity.py`.

Artifacts:

- F1 checkpoint: `outputs/checkpoints/flex_f1_parity_overfit16_k896_20260605/final`
- Train log: `outputs/reports/flex_f1_parity_overfit16_k896_20260605_train.log`
- Train summary: `outputs/reports/flex_f1_parity_overfit16_k896_20260605_train_summary.json`
- F0 parity baseline: `outputs/reports/f0_untrained_k896_vis16_teacher_parity_summary.json`
- F1 parity eval: `outputs/reports/flex_f1_overfit16_k896_vis16_teacher_parity_summary.json`
- F1 free-run eval: `outputs/reports/flex_f1_overfit16_k896_vis68_free_run_decode_summary.json`
- F0 position-preserving parity diagnostic: `outputs/reports/f0_untrained_k896_vis16_teacher_parity_position_preserved_summary.json`

## Train Curve

| Step | Loss | Traj KL | Text KL | Traj Top1 | Top1 in Top5 | action_pre Norm Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.465 | 0.132 | 0.250 | 0.758 | 0.969 | 0.0240 |
| 100 | 0.409 | 0.030 | 0.183 | 0.874 | 0.992 | 0.0109 |
| 200 | 0.366 | 0.025 | 0.083 | 0.873 | 0.990 | 0.0147 |

FLEX-only training learns teacher-forced logits, but it does not recover boundary hidden scale.

## Exact 16-Sample B0-Parity

| Metric | F0 Untrained | F1 FLEX-Only | Delta |
|---|---:|---:|---:|
| Traj teacher-student KL | 0.100 | 0.026 | -0.074 |
| Traj top1 agreement | 0.760 | 0.870 | +0.110 |
| Teacher top1 in student top5 | 0.974 | 0.993 | +0.019 |
| Text teacher-student KL | 0.279 | 0.100 | -0.180 |
| Text top1 agreement | 0.877 | 0.956 | +0.080 |
| Student TF argmax ADE | 0.097 | 0.075 | -0.022 |
| cot_end norm ratio | 0.108 | 0.107 | -0.000 |
| traj_start norm ratio | 0.0130 | 0.0127 | -0.0003 |
| action_pre norm ratio | 0.0130 | 0.0127 | -0.0003 |

Interpretation: FLEX-only can imitate B0 local logits under teacher forcing, but the downstream boundary hidden vectors remain near-zero scale relative to B0.

## Free-Run Decode On Vis68

| Model | ADE | FDE | Bad Geometry | Avg Unique IDs | Motion Match | Anti-Collapse |
|---|---:|---:|---:|---:|---:|---:|
| B0 no-FLEX | 3.159 | 10.159 | 0.088 | 25.779 | 0.632 | 0.628 |
| F0 untrained K896 | 4.980 | 15.798 | 0.221 | 20.588 | 0.515 | 0.548 |
| F1 FLEX-only parity16 | 3.768 | 12.253 | 0.088 | 17.574 | 0.544 | 0.527 |
| F2 LoRA-top4 pilot | 4.430 | 14.043 | 0.176 | 15.191 | 0.515 | 0.502 |

F1 is better than F0/F2 on ADE/FDE and bad-geometry, but still behind B0 and worse on diversity. The remaining gap matches the unresolved boundary hidden scale/position problem.

## Conclusion

FLEX-only is not enough. The bottleneck is not only raw visual information loss; it is compressed visual-prefix adaptation of the frozen language backbone boundary states.

## Position-Preserving Diagnostic

Patch:

- `src/training/flex_batch.py` can now emit original B0-style `position_ids` for compressed FLEX batches.
- `src/model/student_wrapper.py` can now pass a `position_ids` override through FLEX prefill.
- `scripts/104_eval_flex_teacher_parity.py` exposes `--preserve-flex-positions`.

Result on F0 untrained, same 16 rows:

| Metric | Normal Compressed | Position-Preserved | Direction |
|---|---:|---:|---|
| Traj teacher-student KL | 0.100 | 0.368 | worse |
| Traj top1 agreement | 0.760 | 0.675 | worse |
| Text teacher-student KL | 0.279 | 0.271 | flat |
| Student TF argmax ADE | 0.097 | 0.144 | worse |
| traj_start norm ratio | 0.0130 | 0.0130 | unchanged |
| action_pre norm ratio | 0.0130 | 0.0130 | unchanged |

Interpretation: simple RoPE/position-id preservation does not restore boundary hidden scale. The failure is not just downstream token position shift; it is the frozen backbone's response to the compressed visual-prefix content/length itself.

Next fix:

1. Stop spending runs on FLEX-only as the main path.
2. Train LoRA-open F2 with B0 parity loss, not the previous GT/CE-heavy pilot recipe.
3. Open only the minimum adaptation surface first: FLEX + multimodal projector + last 2-4 LM LoRA layers, low LR.
4. Keep the same parity table as the gate: logits KL/top-k, boundary norm ratio, free-run ADE/FDE, unique/repetition.
