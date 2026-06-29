#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="outputs/action_expert/q3_e2e_val512_n6_temp1_seed42_evalbase1042_20260604"
mkdir -p "$OUT_DIR"
exec >> "$OUT_DIR/run.log" 2>&1

echo "{\"event\":\"q3_e2e_n6_temp1_launch\",\"time\":\"$(date -Is)\",\"out_dir\":\"$OUT_DIR\",\"eval_samples\":512,\"eval_num_paths\":6,\"seed\":42,\"eval_seed_base\":1042,\"eval_temperature\":1.0}"

.venv/bin/python -u scripts/101_eval_ae28_seed_sweep.py \
  --ckpt q3=outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt \
  --seeds 42 \
  --seed-sweep-output-dir "$OUT_DIR" \
  --student-checkpoint-dir outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250 \
  --num-samples 20000 \
  --val-samples 2000 \
  --split-cache-json outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/split_cache_20k_2k_seed42.json \
  --eval-samples 512 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --prefix-mode student_free \
  --ae-init-mode student_backbone_init \
  --target-source teacher \
  --eval-temperature 1.0 \
  --eval-num-paths 6 \
  --eval-selection-method single \
  --eval-seed-mode fixed \
  --eval-log-rows 0 \
  --cleanup-every 0 \
  --eval-cleanup-every 0 \
  --attn-implementation flash_attention_2 \
  --seed 42
