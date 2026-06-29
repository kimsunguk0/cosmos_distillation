#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="outputs/action_expert/q2_q3_seed_sweep_eval512_n16_20260603"
mkdir -p "$OUT_DIR"

.venv/bin/python -u scripts/101_eval_ae28_seed_sweep.py \
  --ckpt q2=outputs/action_expert/q2_continue_s10000_to_s30000_b8pb8_20260602_0220/best.pt \
  --ckpt q3=outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt \
  --seeds 42,43,44,45,46 \
  --seed-sweep-output-dir "$OUT_DIR" \
  --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --num-samples 20000 \
  --val-samples 2000 \
  --split-cache-json outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/split_cache_20k_2k_seed42.json \
  --eval-samples 512 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --prefix-mode student_free \
  --ae-init-mode student_backbone_init \
  --target-source teacher \
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
  --seed 42
