#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606"
LOG_PATH="outputs/logs/${RUN_NAME}.log"
mkdir -p outputs/logs

exec > >(tee -a "${LOG_PATH}") 2>&1

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl \
  --teacher-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --student-checkpoint-dir outputs/checkpoints/flex_f3_ablation_aug_vis68_from_f2_s3000_lr1e6_20260606/final \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 68 \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 5e-6 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 1500 \
  --seed 42 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.2 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.05 \
  --boundary-norm-weight 0.10 \
  --cache-teacher-targets \
  --cache-collated-batches \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --paired-ablation camera_shuffle \
  --pairwise-boundary-delta-cos-weight 0.05 \
  --pairwise-boundary-delta-norm-weight 0.05 \
  --pairwise-traj-logprob-delta-weight 0.10 \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"
