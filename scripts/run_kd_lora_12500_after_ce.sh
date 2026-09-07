#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"; cd "${ROOT}"
PY=".venv/bin/python"
SPLIT="outputs/action_expert/stage2_heldout200k_val10k_seed42_20260603/split_cache_200k_10k_seed42.json"
BASE_OUT="outputs/action_expert"; STAMP="ae3way_20260720"
DLOG="${BASE_OUT}/${STAMP}_driver.log"
CE_LOG="${BASE_OUT}/${STAMP}_ce_fullft_teacherforced.log"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1" >> "${DLOG}"; }

run_one(){ # label ckpt
  local label="$1" ckpt="$2"
  local out="${BASE_OUT}/${STAMP}_${label}_teacherforced"
  say "START ${label} (12500)  ckpt=${ckpt}"
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

# 1) wait for CE to finish its step-12500 eval (proof it reached 0.5 epoch + checkpoint saved)
say "orchestrator: waiting for CE val_eval@12500"
while true; do
  if grep -q '"event": "val_eval", "step": 12500' "${CE_LOG}" 2>/dev/null; then say "CE reached 12500 eval"; break; fi
  if ! pgrep -f 'ce_fullft_teacherforced' >/dev/null 2>&1; then say "CE proc gone before 12500 (proceeding)"; break; fi
  sleep 60
done

# 2) stop the old 25000-driver + the CE python so GPU frees and no 25000 KD is launched
say "stopping old driver + CE python"
pkill -f 'run_ae3way_train.sh' 2>/dev/null
for p in $(pgrep -f 'ae3way_20260720_ce_fullft_teacherforced'); do kill -9 "$p" 2>/dev/null; done
sleep 10

# 3) KD then LoRA at 12500
run_one r1kd_fullft outputs/checkpoints/stepb_r1kd_200k/r1kd_200k_20260717/fullft_lr3e5_r1kd_200k_e1/best_decode
run_one lora_ce     outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode
say "ALL DONE (12500)"; touch "${BASE_OUT}/${STAMP}_COMPLETE"
