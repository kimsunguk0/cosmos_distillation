#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f8_factorized_per_image_ablation16_s1500_lr2e5_20260606"
LOG_PATH="outputs/logs/${RUN_NAME}.log"
mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl \
  --teacher-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --student-checkpoint-dir outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606 \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 16 \
  --image-ablations normal,camera_shuffle,black \
  --max-steps 1500 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --grad-clip-norm 5.0 \
  --log-every 50 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.2 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.05 \
  --boundary-norm-weight 0.10 \
  --cache-teacher-targets \
  --cache-collated-batches \
  --train-flex \
  --unfreeze-lora-last-n-layers 0 \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"
