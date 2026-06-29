# 126 - ML-FLEX Implementation Smoke

**Date:** 2026-06-08  
**Status:** Forward contract smoke passed; Stage A prealignment smoke and 16-sample run passed  

## Objective

Start the DeepStack-aware ML-FLEX implementation from report 125 and verify two minimum contracts:

1. Compressed FLEX image positions match Qwen3-VL `visual_pos_masks`.
2. Compressed DeepStack tensors match the Qwen3-VL language-model hook shape.
3. Stage A feature/interface prealignment loss can run an optimizer step.

## Artifacts

ML-FLEX F0 checkpoint:

```text
outputs/checkpoints/mlflex_f0_k512_camtime_from_b0_20260608_smoke
```

Forward smoke summary:

```text
outputs/reports/mlflex_f0_k512_forward_smoke_summary.json
```

Stage A 1-step smoke summary:

```text
outputs/reports/mlflex_stagea_prealign_smoke1_summary.json
```

Reusable Stage A launch script:

```text
scripts/launch_mlflex_stagea_prealign_16.sh
```

Stage A 16-sample run:

```text
outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608
outputs/reports/mlflex_stagea_prealign16_s500_20260608_summary.json
outputs/logs/mlflex_stagea_prealign16_s500_20260608.log
```

Final checkpoint forward smoke:

```text
outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608/final
outputs/reports/mlflex_stagea_prealign16_s500_forward_smoke_summary.json
```

## Forward Smoke Result

Command:

```bash
.venv/bin/python -u scripts/111_mlflex_forward_smoke.py \
  --checkpoint-dir outputs/checkpoints/mlflex_f0_k512_camtime_from_b0_20260608_smoke \
  --corpus-jsonl data/corpus/flex_heldout256_stage2val_seed42.jsonl \
  --split val \
  --sample-index 0 \
  --summary-json outputs/reports/mlflex_f0_k512_forward_smoke_summary.json
```

Result:

```json
{
  "contract_ok": true,
  "sample_id": "0a948f59-0a06-41a2-8e20-ac3a39ff4d61__sg_00__t0_1600000",
  "original_seq_len": 3223,
  "compressed_seq_len": 855,
  "original_image_tokens": 2880,
  "compressed_image_tokens": 512,
  "visual_pos_masks_shape": [1, 855],
  "visual_pos_masks_sum": 512,
  "deepstack_visual_embeds_shapes": [[512, 2048], [512, 2048], [512, 2048]],
  "logits_shape": [1, 855, 155685],
  "hidden_shape": [1, 855, 2048]
}
```

Conclusion: ML-FLEX now satisfies the Qwen3-VL DeepStack injection contract for a real sample.

## Stage A Loss Smoke Result

Command:

```bash
.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl data/corpus/flex_heldout256_stage2val_seed42.jsonl \
  --teacher-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --student-checkpoint-dir outputs/checkpoints/mlflex_f0_k512_camtime_from_b0_20260608_smoke \
  --output-dir outputs/checkpoints/mlflex_stagea_prealign_smoke1_20260608 \
  --split val \
  --max-train-samples 1 \
  --max-steps 1 \
  --cache-teacher-targets \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack \
  --image-feature-tokens-per-image 32 \
  --image-feature-mse-weight 1.0 \
  --image-feature-cos-weight 0.1 \
  --image-feature-norm-weight 0.05 \
  --deepstack-feature-tokens-per-image 32 \
  --deepstack-feature-mse-weight 1.0 \
  --deepstack-feature-cos-weight 0.1 \
  --deepstack-feature-norm-weight 0.05 \
  --train-flex \
  --no-save-final \
  --summary-json outputs/reports/mlflex_stagea_prealign_smoke1_summary.json
```

Observed step-1 metrics:

| Metric | Value |
|---|---:|
| trainable params | 68,819,968 |
| image_feature_loss | 0.3687 |
| image_feature_usable | 512 |
| deepstack_feature_loss | 1.0501 |
| deepstack_feature_usable | 512 |
| total loss | 1.4188 |
| grad_norm | 2.3429 |

Conclusion: Stage A feature/interface prealignment is wired and trainable for ML-FLEX. The 1-step result is not a quality result; it only verifies that target extraction, compressed final stream, compressed DeepStack stream, loss, backward, and optimizer step all execute.

## Stage A 16-Sample Prealignment Result

Command:

```bash
bash scripts/launch_mlflex_stagea_prealign_16.sh
```

Default settings:

```text
RUN_NAME=mlflex_stagea_prealign16_s500_20260608
MAX_TRAIN_SAMPLES=16
MAX_STEPS=500
```

This run trained only `flex_scene_encoder` against dense B0 image and DeepStack feature anchors.

Optimization:

```text
trainable params: 68,819,968
trainable group: flex_scene_encoder
lr: 1e-4
frozen backbone / LoRA / heads: yes
samples: 16
steps: 500
```

Trend:

| Step | Loss | Image cos | Image MSE | DeepStack cos | DeepStack MSE | DS L0 cos | DS L1 cos | DS L2 cos |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.4188 | -0.0162 | 0.2576 | -0.0114 | 0.2398 | -0.0168 | -0.0115 | -0.0060 |
| 100 | 0.4392 | 0.6861 | 0.0729 | 0.5777 | 0.0633 | 0.6818 | 0.4171 | 0.6342 |
| 200 | 0.3772 | 0.6906 | 0.0705 | 0.6662 | 0.0539 | 0.8617 | 0.4930 | 0.6439 |
| 300 | 0.3504 | 0.7001 | 0.0685 | 0.7020 | 0.0496 | 0.8863 | 0.5462 | 0.6734 |
| 400 | 0.3321 | 0.7210 | 0.0652 | 0.7204 | 0.0473 | 0.8952 | 0.5763 | 0.6898 |
| 500 | 0.3201 | 0.7288 | 0.0664 | 0.7323 | 0.0446 | 0.8948 | 0.6025 | 0.6997 |

Saved checkpoints:

```text
step_000100
step_000200
step_000300
step_000400
step_000500
final
```

Conclusion: Stage A is not saturated or collapsed in the first 500 steps. The weakest alignment is still DeepStack level 1, but it moved from negative cosine to 0.6025. This is enough to justify moving to the next gate: task-loss adaptation with this initialized ML-FLEX interface.

## Final Forward Smoke Result

Command:

```bash
.venv/bin/python -u scripts/111_mlflex_forward_smoke.py \
  --checkpoint-dir outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608/final \
  --corpus-jsonl data/corpus/flex_heldout256_stage2val_seed42.jsonl \
  --split val \
  --sample-index 0 \
  --summary-json outputs/reports/mlflex_stagea_prealign16_s500_forward_smoke_summary.json
```

Result:

```json
{
  "contract_ok": true,
  "original_seq_len": 3223,
  "compressed_seq_len": 855,
  "original_image_tokens": 2880,
  "compressed_image_tokens": 512,
  "visual_pos_masks_shape": [1, 855],
  "visual_pos_masks_sum": 512,
  "deepstack_visual_embeds_shapes": [[512, 2048], [512, 2048], [512, 2048]],
  "logits_shape": [1, 855, 155685],
  "hidden_shape": [1, 855, 2048]
}
```

Conclusion: The trained Stage A checkpoint still satisfies the Qwen3-VL DeepStack forward contract.
