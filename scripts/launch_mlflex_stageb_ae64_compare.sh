#!/usr/bin/env bash
set -euo pipefail

CORPUS="${CORPUS:-data/corpus/flex_heldout256_stage2val_seed42.jsonl}"
AE_CKPT="${AE_CKPT:-outputs/action_expert/q3_cosine_cooldown_from_q2best_s26000_2k_20260603_0505/best.pt}"
B0_CKPT="${B0_CKPT:-outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250}"
MLFLEX_CKPT="${MLFLEX_CKPT:-outputs/checkpoints/mlflex_stageb_task_gate16_s500_20260608/final}"
RUN_TAG="${RUN_TAG:-20260608}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"

mkdir -p outputs/logs outputs/reports outputs/action_expert

COMMON_ARGS=(
  --ckpt-path "${AE_CKPT}"
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples "${NUM_SAMPLES}"
  --eval-samples "${NUM_SAMPLES}"
  --eval-batch-size 1
  --eval-num-paths 1
  --eval-temperature 1.0
  --eval-selection-method single
  --eval-log-rows 0
  --prefix-mode teacher_forced
  --ae-init-mode student_backbone_init
  --target-source gt
  --device cuda:0
  --teacher-load-device cpu
  --attn-implementation sdpa
)

echo "{\"event\":\"ae64_b0_start\",\"samples\":${NUM_SAMPLES}}"
.venv/bin/python -u scripts/85_eval_ae28_best_of_n.py \
  "${COMMON_ARGS[@]}" \
  --student-checkpoint-dir "${B0_CKPT}" \
  --output-dir "outputs/action_expert/b0_ae${NUM_SAMPLES}_${RUN_TAG}" \
  --eval-summary-json "outputs/reports/b0_ae${NUM_SAMPLES}_${RUN_TAG}_summary.json"

echo "{\"event\":\"ae64_mlflex_start\",\"samples\":${NUM_SAMPLES}}"
.venv/bin/python -u scripts/85_eval_ae28_best_of_n.py \
  "${COMMON_ARGS[@]}" \
  --student-checkpoint-dir "${MLFLEX_CKPT}" \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack \
  --output-dir "outputs/action_expert/mlflex_stageb_ae${NUM_SAMPLES}_${RUN_TAG}" \
  --eval-summary-json "outputs/reports/mlflex_stageb_ae${NUM_SAMPLES}_${RUN_TAG}_summary.json"

echo "{\"event\":\"ae64_compare_done\",\"b0_summary\":\"outputs/reports/b0_ae${NUM_SAMPLES}_${RUN_TAG}_summary.json\",\"mlflex_summary\":\"outputs/reports/mlflex_stageb_ae${NUM_SAMPLES}_${RUN_TAG}_summary.json\"}"
