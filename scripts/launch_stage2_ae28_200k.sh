#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603"
SPLIT_CACHE="outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json"
LOG_FILE="${OUT_DIR}/launch.log"

mkdir -p "$OUT_DIR" "$(dirname "$SPLIT_CACHE")"

{
  echo "===== STAGE2_200K_AE28 START $(date -Is) ====="
  echo "out_dir=${OUT_DIR}"
  echo "split_cache=${SPLIT_CACHE}"

  .venv/bin/python -u scripts/84_train_student_ae28_official.py \
    --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
    --num-samples 200000 \
    --val-samples 10000 \
    --split-cache-json "$SPLIT_CACHE" \
    --split-scan-all \
    --eval-samples 1024 \
    --eval-train-samples 512 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --eval-every 2500 \
    --log-every 100 \
    --train-ade-every 0 \
    --prefix-mode student_free \
    --ae-init-mode student_backbone_init \
    --target-source teacher \
    --expert-lr 1e-4 \
    --proj-lr 1e-4 \
    --num-time-samples 16 \
    --grad-clip-norm 5.0 \
    --no-norm-bias-decay \
    --eval-temperature 0.85 \
    --eval-num-paths 16 \
    --eval-selection-method mean_traj \
    --eval-seed-mode fixed \
    --eval-vectorize-paths \
    --eval-path-batch-size 8 \
    --eval-log-rows 0 \
    --cleanup-every 0 \
    --eval-cleanup-every 0 \
    --attn-implementation flash_attention_2 \
    --save-every 2500 \
    --skip-initial-eval \
    --seed 42 \
    --steps 25000 \
    --lr-warmup-steps 0 \
    --min-lr 1e-6 \
    --allow-train-cache-mutation \
    --fused-adamw \
    --output-dir "$OUT_DIR"

  echo "===== STAGE2_200K_AE28 END $(date -Is) ====="
} 2>&1 | tee -a "$LOG_FILE"
