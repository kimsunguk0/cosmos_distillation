# Report 137: AE14 Step Ablation and ONNX Export

Date: 2026-06-12

## Scope

- Confirm AE14 best checkpoint from the completed run.
- Compare AE14 inference with 10 denoising steps vs 4 denoising steps on a small balanced visualization/eval subset.
- Export AE14 single-step model to ONNX for TensorRT Edge LLM integration work.

## Checkpoints

- AE14 run: `outputs/action_expert/ae14_from_ae28_10step`
- Best checkpoint: `outputs/action_expert/ae14_from_ae28_10step/best.pt`
- Best step: `7500`
- Source AE28 checkpoint used for compressed init/export:
  `outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt`
- FLEX K512 backbone:
  `outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final`

AE14 uses 14 selected KV/action-expert layers from the AE28 source:

```text
[0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27]
```

## Training Checkpoint Summary

Validation metrics recorded during AE14 training:

| step | ADE | FDE |
|---:|---:|---:|
| 2500 | 2.902859 | 8.675071 |
| 5000 | 2.697595 | 7.989290 |
| 7500 | 2.566465 | 7.587199 |
| 10000 | 2.694770 | 8.034215 |

The best checkpoint is step 7500, not the final step.

## 10-step vs 4-step Ablation

Dataset:

- `data/corpus/vis_4per_category_val.jsonl`
- 68 samples total
- 17 categories, 4 samples per category

Condition:

- AE14 `best.pt` step 7500
- FLEX K512 backbone
- `student_free` CoT prefix
- GT trajectory target
- Temperature `0.85`
- Same seed per sample for 10-step and 4-step

Output artifacts:

- `outputs/action_expert/ae14_from_ae28_10step_step_ablation_68/rows.jsonl`
- `outputs/action_expert/ae14_from_ae28_10step_step_ablation_68/summary.json`

Aggregate result:

| denoise steps | ADE mean | ADE p50 | FDE mean | FDE p50 | latency mean | latency p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 3.4499 m | 2.5143 m | 9.5739 m | 7.5492 m | 130.8 ms | 124.5 ms |
| 4 | 3.4449 m | 2.5103 m | 9.5529 m | 7.6541 m | 52.7 ms | 50.8 ms |

Delta, 4-step minus 10-step:

- ADE mean: `-0.0050 m`
- ADE p50: `-0.0321 m`
- FDE mean: `-0.0210 m`
- FDE p50: `-0.0974 m`

Interpretation:

- On this 68-sample balanced subset, 4-step did not degrade aggregate ADE/FDE versus 10-step.
- Latency dropped from about `130.8 ms` to `52.7 ms`, roughly `2.48x` faster for AE denoising.
- This is still a small subset. Treat it as a deployment-oriented smoke/ablation result, not a final quality benchmark.

Largest 4-step degradation by ADE:

| sample | category | ADE 10 | ADE 4 | delta |
|---|---|---:|---:|---:|
| `9d511503-161f-4efe-aca9-7e3443a1526d__sg_05__t0_9600000` | `intersection_other` | 4.1852 | 5.8387 | +1.6535 |
| `60968314-d463-463b-9f14-7ff2c6190c8b__sg_03__t0_6400000` | `traffic_left_turn` | 5.5270 | 6.4594 | +0.9325 |
| `9613c363-7d42-4680-ac8e-7341bca8aeaa__sg_02__t0_4800000` | `intersection_other` | 6.4766 | 7.0864 | +0.6098 |

Largest 4-step improvement by ADE:

| sample | category | ADE 10 | ADE 4 | delta |
|---|---|---:|---:|---:|
| `9d888f9b-6deb-4328-86a8-98e1605ff565__sg_02__t0_4800000` | `curve` | 8.0078 | 7.4415 | -0.5662 |
| `d4190223-c9fd-4671-a0f1-cc7424171971__sg_06__t0_11200000` | `curve` | 2.2727 | 1.7899 | -0.4828 |
| `9d888f9b-6deb-4328-86a8-98e1605ff565__sg_03__t0_6400000` | `curve` | 9.4609 | 9.0061 | -0.4548 |

## ONNX Export

Script:

- `scripts/export_flex_ae_onnx.py`

Required export options added/used for AE14:

- `--ae-checkpoint`
- `--compressed-layers 14`
- `--mapping linspace_round`
- `--ae-init-mode ae_checkpoint_compressed`
- `--init-ae-source-checkpoint outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt`
- `--ae-output-name ae14_single_step.onnx`

Export command:

```bash
.venv/bin/python scripts/export_flex_ae_onnx.py \
  --student-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
  --ae-checkpoint outputs/action_expert/ae14_from_ae28_10step/best.pt \
  --output-dir outputs/trt_export/flex_k512_fp16/ae14 \
  --device cuda:0 \
  --skip-flex \
  --compressed-layers 14 \
  --mapping linspace_round \
  --ae-init-mode ae_checkpoint_compressed \
  --init-ae-source-checkpoint outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt \
  --ae-output-name ae14_single_step.onnx
```

Exported artifact:

- `outputs/trt_export/flex_k512_fp16/ae14/ae14_single_step.onnx`
- Size: `1530.2 MB`

ONNX checker result:

- `onnx.checker.check_model`: passed

ONNX signature:

```text
inputs:
  noisy_action: [batch, 64, 2]
  timestep: [batch, 1, 1]
  position_ids: [3, batch, 64]
  past_keys: [14, batch, 8, kv_seq_len, 128]
  past_values: [14, batch, 8, kv_seq_len, 128]

outputs:
  velocity: [batch, *, *]
```

Deployment caveat:

- Runtime must gather the selected 14 KV layers from the 28-layer LLM cache in this exact order:
  `[0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27]`.
- AE14 ONNX is a single denoising step. 4-step or 10-step inference is controlled by the runtime loop around this ONNX graph.
