# 127 - ML-FLEX Stage B Task Gate

**Date:** 2026-06-08  
**Status:** 16-sample task-adaptation gate passed  

## Objective

Check whether the DeepStack-aware ML-FLEX interface from report 126 can be adapted with task loss without collapsing the B0 task behavior.

This gate starts from the Stage A feature-prealigned ML-FLEX checkpoint and trains:

- `flex_scene_encoder` at LR `5e-5`
- all language LoRA weights at LR `1e-6`
- dense B0 backbone as teacher, frozen
- task/interface losses from `scripts/105_train_flex_teacher_parity.py`

The point of this run is not final quality. It answers the narrower question: can FLEX plus all LoRA move toward the dense B0 teacher after replacing 2880 image tokens with 512 ML-FLEX tokens while keeping Qwen3-VL DeepStack tensors aligned?

## Artifacts

Launch script:

```text
scripts/launch_mlflex_stageb_task_gate_16.sh
```

Stage B checkpoint and logs:

```text
outputs/checkpoints/mlflex_stageb_task_gate16_s500_20260608
outputs/checkpoints/mlflex_stageb_task_gate16_s500_20260608/final
outputs/logs/mlflex_stageb_task_gate16_s500_20260608.log
outputs/reports/mlflex_stageb_task_gate16_s500_20260608_summary.json
```

Evaluation summaries:

```text
outputs/reports/mlflex_stageb_task_gate16_s500_forward_smoke_summary.json
outputs/reports/mlflex_stageb_task_gate16_s500_parity16_summary.json
outputs/reports/mlflex_stagea_prealign16_s500_parity16_summary.json
```

## Training Setup

Command:

```bash
bash scripts/launch_mlflex_stageb_task_gate_16.sh
```

Defaults:

```text
teacher: outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250
init: outputs/checkpoints/mlflex_stagea_prealign16_s500_20260608/final
samples: 16
steps: 500
batch size: 1
FLEX tokens: 512 = 16 images x 32 tokens/image
DeepStack levels: 3
trainable params: 138,550,272
  flex_scene_encoder: 68,819,968 at 5e-5
  language_lora: 69,730,304 at 1e-6
```

Losses:

```text
traj KL: 1.0
text KL: 0.05
format KL: 0.05
boundary cos/norm: 0.02 / 0.02
image feature MSE/cos/norm: 0.2 / 0.02 / 0.01
DeepStack feature MSE/cos/norm: 0.2 / 0.02 / 0.01
```

## Training Trend

| Step | Loss | Traj KL | Traj top1 | Text KL | Image cos | DeepStack cos |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1606 | 0.0467 | 0.9453 | 0.9183 | 0.7337 | 0.7338 |
| 100 | 0.1409 | 0.0448 | 0.8867 | 0.2266 | 0.6176 | 0.6472 |
| 200 | 0.0933 | 0.0143 | 0.9117 | 0.0816 | 0.6488 | 0.6911 |
| 300 | 0.0828 | 0.0084 | 0.9219 | 0.0629 | 0.6692 | 0.7042 |
| 400 | 0.0902 | 0.0135 | 0.9023 | 0.0852 | 0.6602 | 0.6995 |
| 500 | 0.0891 | 0.0101 | 0.9250 | 0.0689 | 0.6575 | 0.6842 |

Interpretation: early feature cosine dipped after LoRA adaptation started, but it recovered and stabilized. No collapse was observed in the first 500 steps.

## Forward Contract

Final checkpoint smoke:

```bash
.venv/bin/python -u scripts/111_mlflex_forward_smoke.py \
  --checkpoint-dir outputs/checkpoints/mlflex_stageb_task_gate16_s500_20260608/final \
  --corpus-jsonl data/corpus/flex_heldout256_stage2val_seed42.jsonl \
  --split val \
  --sample-index 0 \
  --summary-json outputs/reports/mlflex_stageb_task_gate16_s500_forward_smoke_summary.json
```

Result:

```json
{
  "contract_ok": true,
  "original_seq_len": 3223,
  "compressed_seq_len": 855,
  "original_image_tokens": 2880,
  "compressed_image_tokens": 512,
  "scene_tokens": 512,
  "num_deepstack_levels": 3,
  "deepstack_visual_embeds_shapes": [[512, 2048], [512, 2048], [512, 2048]]
}
```

## Teacher-Parity Result

Both checkpoints were evaluated against dense B0 teacher on the same 16 validation samples.

| Metric, mean | Stage A prealign | Stage B task-adapted | Delta |
|---|---:|---:|---:|
| traj teacher-student KL | 0.0533 | 0.0099 | -0.0433 |
| traj top1 agreement | 0.8433 | 0.9136 | +0.0703 |
| teacher top1 in student top5 | 0.9883 | 0.9985 | +0.0103 |
| text teacher-student KL | 0.4108 | 0.0606 | -0.3502 |
| text top1 agreement | 0.7878 | 0.9507 | +0.1630 |
| format teacher-student KL | 0.0028 | 0.0004 | -0.0024 |
| action-pre hidden cosine | 0.9783 | 0.9905 | +0.0122 |
| student TF argmax ADE | 0.1590 m | 0.1261 m | -0.0328 m |
| student - teacher TF argmax ADE | 0.0504 m | 0.0175 m | -0.0328 m |

## Conclusion

This gate supports the LR-split joint-adaptation direction:

1. FLEX-only Stage A is not enough; it improves feature alignment but leaves a large task-interface gap.
2. Joint training of ML-FLEX plus all LoRA at low LoRA LR closes most of that gap on the 16-sample gate.
3. The DeepStack-aware contract remains intact after task adaptation.

The next stage should not be a frozen-backbone FLEX-only run. The next practical run should scale this recipe to more data, while keeping the same monitoring:

- forward contract smoke
- teacher-parity on a fixed heldout slice
- decode/free-run ADE through the action expert
- DeepStack level-wise feature cosine, especially level 1

## Remaining Risk

This is still a 16-sample gate. It proves the interface can adapt; it does not prove generalization or deployable closed-loop quality. The next decision point needs at least 256-sample heldout parity plus action-expert decode/free-run ADE.
