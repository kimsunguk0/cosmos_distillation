#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export COSMOS_DATA_ROOT="${COSMOS_DATA_ROOT:-/home/pm97/workspace/dataset/distill_dataset}"
source "$ROOT_DIR/scripts/31_local_env.sh"

CHAIN_ID="${CHAIN_ID:-no_nav_chain_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BP2_MAX_STEPS="${BP2_MAX_STEPS:-5000}"
BP3_MAX_STEPS="${BP3_MAX_STEPS:-5000}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-50}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-0.02}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-999}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
VAL_EVAL_SAMPLES="${VAL_EVAL_SAMPLES:-204}"
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-4}"
COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-2}"
COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}"
COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}"
COSMOS_SKIP_ASSET_CHECK="${COSMOS_SKIP_ASSET_CHECK:-1}"

REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/outputs/reports/no_nav_distill}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT_DIR/outputs/checkpoints}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/no_nav_distill}"
mkdir -p "$REPORT_DIR" "$CHECKPOINT_ROOT" "$LOG_DIR"

chain_log="$LOG_DIR/${CHAIN_ID}.chain.log"
touch "$chain_log"

log_msg() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$chain_log" >&2
}

CHAIN_LOCK_NAME="${CHAIN_LOCK_NAME:-no_nav_backbone_chain_${WAIT_FOR_PID:-standalone}}"
LOCK_DIR="$LOG_DIR/${CHAIN_LOCK_NAME}.lock"
if [[ -d "$LOCK_DIR" ]]; then
  lock_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" >/dev/null 2>&1; then
    log_msg "another chain watcher is already active lock=$LOCK_DIR pid=$lock_pid"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
fi
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log_msg "another chain watcher acquired lock=$LOCK_DIR"
  exit 0
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"
printf '%s\n' "$CHAIN_ID" > "$LOCK_DIR/chain_id"
trap 'rm -rf "$LOCK_DIR"' EXIT

wait_for_existing_stage() {
  local wait_pid="${WAIT_FOR_PID:-}"
  local wait_summary="${WAIT_FOR_SUMMARY:-}"
  local wait_checkpoint="${WAIT_FOR_CHECKPOINT_DIR:-}"
  if [[ -n "$wait_pid" ]]; then
    log_msg "waiting for existing stage pid=$wait_pid"
    while kill -0 "$wait_pid" >/dev/null 2>&1; do
      sleep 60
      log_msg "still waiting for pid=$wait_pid"
    done
    log_msg "existing stage pid=$wait_pid exited"
  fi
  if [[ -n "$wait_summary" && ! -s "$wait_summary" ]]; then
    log_msg "waiting for summary=$wait_summary"
    while [[ ! -s "$wait_summary" ]]; do
      sleep 30
    done
  fi
  if [[ -n "$wait_checkpoint" ]]; then
    log_msg "waiting for checkpoint=$wait_checkpoint"
    while [[ ! -d "$wait_checkpoint" ]]; do
      sleep 30
    done
    echo "$wait_checkpoint"
    return 0
  fi
  echo "${INIT_CHECKPOINT_DIR:-}"
}

run_stage() {
  local stage_key="$1"
  local stage_config="$2"
  local init_checkpoint="$3"
  local max_steps="$4"
  local output_dir="$CHECKPOINT_ROOT/no_nav_${stage_key}_b${BATCH_SIZE}/${CHAIN_ID}"
  local stage_summary_json="$REPORT_DIR/${CHAIN_ID}_${stage_key}_summary.json"
  local stage_log="$LOG_DIR/${CHAIN_ID}_${stage_key}.log"
  mkdir -p "$(dirname "$output_dir")"

  log_msg "starting $stage_key config=$stage_config max_steps=$max_steps init=${init_checkpoint:-none}"
  local args=(
    env
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    STAGE_CONFIG="$stage_config"
    BATCH_SIZE="$BATCH_SIZE"
    MAX_STEPS="$max_steps"
    COSMOS_DATALOADER_NUM_WORKERS="$COSMOS_DATALOADER_NUM_WORKERS"
    COSMOS_DATALOADER_PREFETCH_FACTOR="$COSMOS_DATALOADER_PREFETCH_FACTOR"
    COSMOS_DATALOADER_PIN_MEMORY="$COSMOS_DATALOADER_PIN_MEMORY"
    COSMOS_DATALOADER_PERSISTENT_WORKERS="$COSMOS_DATALOADER_PERSISTENT_WORKERS"
    COSMOS_SKIP_ASSET_CHECK="$COSMOS_SKIP_ASSET_CHECK"
    EVAL_EVERY_EPOCHS="$EVAL_EVERY_EPOCHS"
    SAVE_EVERY_EPOCHS="$SAVE_EVERY_EPOCHS"
    OUTPUT_DIR="$output_dir"
    SUMMARY_JSON="$stage_summary_json"
  )
  if [[ -n "$init_checkpoint" ]]; then
    args+=(INIT_CHECKPOINT_DIR="$init_checkpoint")
  fi

  "${args[@]}" bash scripts/43_train_no_nav_backbone_pilot.sh \
    --multi-gpu off \
    --log-every-steps "$LOG_EVERY_STEPS" \
    > "$stage_log" 2>&1

  log_msg "finished $stage_key log=$stage_log summary=$stage_summary_json"
  if [[ ! -d "$output_dir/final" ]]; then
    log_msg "ERROR: missing final checkpoint for $stage_key at $output_dir/final"
    exit 1
  fi
  echo "$output_dir/final"
}

log_msg "chain_start id=$CHAIN_ID batch=$BATCH_SIZE cuda=$CUDA_VISIBLE_DEVICES"
previous_checkpoint="$(wait_for_existing_stage)"
if [[ -n "$previous_checkpoint" ]]; then
  log_msg "initial checkpoint ready: $previous_checkpoint"
else
  log_msg "no initial checkpoint provided; this chain will start from BP2 without resume"
fi

bp2_final="$(run_stage "bp2_text_topk" "configs/train/stage_bp2_no_nav_text_topk_kd.yaml" "$previous_checkpoint" "$BP2_MAX_STEPS")"
bp3_final="$(run_stage "bp3_traj_topk" "configs/train/stage_bp3_no_nav_traj_topk_kd.yaml" "$bp2_final" "$BP3_MAX_STEPS")"

if [[ "$RUN_FINAL_EVAL" == "1" ]]; then
  eval_log="$LOG_DIR/${CHAIN_ID}_bp3_eval.log"
  report_prefix="$REPORT_DIR/${CHAIN_ID}_bp3"
  log_msg "starting final eval checkpoint=$bp3_final"
  env \
    CHECKPOINT_DIR="$bp3_final" \
    CORPUS_JSONL="$ROOT_DIR/data/corpus/no_nav_teacher_pair_300chunks.jsonl" \
    REPORT_PREFIX="$report_prefix" \
    VAL_EVAL_SAMPLES="$VAL_EVAL_SAMPLES" \
    TRAIN_EVAL_SAMPLES="$TRAIN_EVAL_SAMPLES" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    bash scripts/39_eval_h200_checkpoint.sh > "$eval_log" 2>&1
  log_msg "finished final eval log=$eval_log prefix=$report_prefix"
fi

log_msg "chain_done final_checkpoint=$bp3_final"
