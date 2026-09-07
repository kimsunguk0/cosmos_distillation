#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"; cd "${ROOT}"
PY=".venv/bin/python"
SPLIT="outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json"
BASE_OUT="outputs/action_expert"
STAMP="ae3way_20260720"
DLOG="${BASE_OUT}/${STAMP}_driver.log"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1" >> "${DLOG}"; }

run_one(){ # label ckpt
  local label="$1" ckpt="$2"
  local out="${BASE_OUT}/${STAMP}_${label}_teacherforced"
  if [[ -s "${out}/summary.json" ]]; then say "skip ${label} (summary exists)"; return; fi
  say "START ${label}  ckpt=${ckpt}"
  "${PY}" -u scripts/84_train_student_ae28_official.py \
    --student-checkpoint-dir "${ckpt}" \
    --num-samples 200000 --val-samples 10000 --split-cache-json "${SPLIT}" --split-scan-all \
    --batch-size 8 --eval-batch-size 8 \
    --steps 12500 --eval-every 2500 --save-every 2500 --log-every 100 --skip-initial-eval \
    --prefix-mode teacher_forced --ae-init-mode teacher_compressed --target-source teacher \
    --num-time-samples 8 \
    --expert-lr 1e-4 --proj-lr 1e-4 --grad-clip-norm 1.0 --lr-warmup-steps 0 --min-lr 1e-6 \
    --no-norm-bias-decay --fused-adamw \
    --eval-samples 1024 --eval-num-paths 6 --eval-selection-method mean_traj \
    --eval-temperature 0.85 --eval-seed-mode fixed --eval-vectorize-paths \
    --teacher-load-device cpu --device cuda:0 --attn-implementation flash_attention_2 --seed 42 \
    --output-dir "${out}" >> "${out}.log" 2>&1
  say "DONE ${label} (exit $?)"
}

say "AE 3-way START (teacher_forced, backbone frozen, AE init teacher_compressed, batch8/effFM128, lr1e-4, 25000 steps)"
run_one ce_fullft   outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode
run_one r1kd_fullft outputs/checkpoints/stepb_r1kd_200k/r1kd_200k_20260717/fullft_lr3e5_r1kd_200k_e1/best_decode
run_one lora_ce     outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode
say "ALL DONE"; touch "${BASE_OUT}/${STAMP}_COMPLETE"
