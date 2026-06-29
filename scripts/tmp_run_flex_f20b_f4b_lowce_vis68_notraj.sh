#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f20b_f4b_lowce_vis68_s3000_lr1e6_ce05_notraj_20260606"
LOG_PATH="outputs/logs/${RUN_NAME}.log"
mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl data/corpus/vis_4per_category_val.jsonl \
  --teacher-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --student-checkpoint-dir outputs/checkpoints/flex_f4b_paircontrast_cache_vis68_from_f3_s3000_lr5e6_gc5_20260606/final \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 68 \
  --paired-ablation camera_shuffle \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 1e-6 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 1500 \
  --seed 42 \
  --traj-kl-weight 0.0 \
  --text-kl-weight 0.10 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.10 \
  --boundary-norm-weight 0.20 \
  --prefix-token-ce-weight 0.05 \
  --pairwise-boundary-delta-cos-weight 0.05 \
  --pairwise-boundary-delta-norm-weight 0.05 \
  --pairwise-traj-logprob-delta-weight 0.0 \
  --cache-teacher-targets \
  --cache-collated-batches \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --free-run-token-targets normal=outputs/reports/b0_step006250_vis68_decode_normal_summary.json,camera_shuffle=outputs/reports/b0_step006250_vis68_decode_camera_shuffle_summary.json \
  --free-run-token-ce-weight 0.50 \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"
