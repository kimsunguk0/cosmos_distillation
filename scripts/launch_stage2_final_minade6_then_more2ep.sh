#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STAGE2_DIR="${STAGE2_DIR:-outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603}"
FINAL_CKPT="${FINAL_CKPT:-${STAGE2_DIR}/final.pt}"
FINAL_EVAL_OUT="${FINAL_EVAL_OUT:-outputs/action_expert/stage2_200k_final_minade6_eval_${RUN_TAG}}"
MORE2EP_OUT="${MORE2EP_OUT:-outputs/action_expert/stage2_200k_more2ep_b8_nt16_minade6_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-outputs/action_expert/stage2_final_eval_then_more2ep_${RUN_TAG}}"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/chain.log"

{
  echo "{\"event\":\"chain_start\",\"time\":\"$(date -Is)\",\"run_tag\":\"${RUN_TAG}\",\"final_ckpt\":\"${FINAL_CKPT}\",\"final_eval_out\":\"${FINAL_EVAL_OUT}\",\"more2ep_out\":\"${MORE2EP_OUT}\"}"

  if [ ! -f "$FINAL_CKPT" ]; then
    echo "{\"event\":\"missing_final_ckpt\",\"time\":\"$(date -Is)\",\"path\":\"${FINAL_CKPT}\"}"
    exit 1
  fi

  echo "{\"event\":\"final_minade6_eval_start\",\"time\":\"$(date -Is)\"}"
  CKPT="$FINAL_CKPT" \
  OUT_DIR="$FINAL_EVAL_OUT" \
  RUN_TAG="${RUN_TAG}_final" \
  bash scripts/launch_stage2_200k_best_minade6_eval.sh
  echo "{\"event\":\"final_minade6_eval_done\",\"time\":\"$(date -Is)\",\"out_dir\":\"${FINAL_EVAL_OUT}\"}"

  echo "{\"event\":\"more2ep_train_start\",\"time\":\"$(date -Is)\"}"
  RESUME_CKPT="$FINAL_CKPT" \
  OUT_DIR="$MORE2EP_OUT" \
  RUN_TAG="${RUN_TAG}_more2ep" \
  START_STEP=25000 \
  END_STEP=75000 \
  EVAL_TEMPERATURE=1.0 \
  EVAL_SELECTION_METHOD=single \
  bash scripts/launch_stage2_ae28_200k_more2ep_minade6.sh
  echo "{\"event\":\"more2ep_train_done\",\"time\":\"$(date -Is)\",\"out_dir\":\"${MORE2EP_OUT}\"}"

  echo "{\"event\":\"chain_done\",\"time\":\"$(date -Is)\"}"
} >> "$LOG_PATH" 2>&1
