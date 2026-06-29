#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

RUN_NAME="flex_f1_parity512_prenormfix_trajonly_s4096_k896_20260606"
LOG_PATH="outputs/logs/${RUN_NAME}.log"
mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"run_script_start\",\"run_name\":\"${RUN_NAME}\",\"log_path\":\"${LOG_PATH}\"}"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_200k.jsonl \
  --teacher-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --student-checkpoint-dir outputs/checkpoints/flex_f0_untrained_k896_camtime_from_step006250_20260605 \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split train \
  --max-train-samples 512 \
  --max-steps 4096 \
  --batch-size 1 \
  --learning-rate 2e-5 \
  --grad-clip-norm 1.0 \
  --log-every 100 \
  --save-every 1024 \
  --seed 42 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.0 \
  --format-kl-weight 0.0 \
  --boundary-cos-weight 0.05 \
  --boundary-norm-weight 0.10 \
  --boundary-mse-weight 0.0 \
  --cache-teacher-targets \
  --cache-collated-batches \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"
