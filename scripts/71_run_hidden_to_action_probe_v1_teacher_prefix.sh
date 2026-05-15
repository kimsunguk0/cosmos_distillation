#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source scripts/31_local_env.sh >/dev/null

CORPUS_JSONL="${CORPUS_JSONL:-data/corpus/no_nav_teacher_pair_300chunks.jsonl}"
SPLIT_IDS_JSON="${SPLIT_IDS_JSON:-data/splits/hidden_to_action_probe_v1/hidden_to_action_probe_v1.sample_ids.json}"
FEATURE_ROOT="${FEATURE_ROOT:-outputs/probe_cache/hidden_to_action_v1}"
PROBE_ROOT="${PROBE_ROOT:-outputs/checkpoints/hidden_to_action_probe_v1}"
PREFIX_TYPE="${PREFIX_TYPE:-teacher_prefix}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-4}"
FEATURE_SHARD_SIZE="${FEATURE_SHARD_SIZE:-512}"
PROBE_EPOCHS="${PROBE_EPOCHS:-30}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
PROBE_HIDDEN_DIM="${PROBE_HIDDEN_DIM:-1024}"
PROBE_LR="${PROBE_LR:-1e-3}"

declare -a NAMES=(
  "bp3_init"
  "bp3_200k_final"
  "bp5"
)

declare -a CHECKPOINT_DIRS=(
  "outputs/checkpoints/no_nav_bp3_h200fast_b4/no_nav_bp3_h200fast_b4_from_step2288_20260504_053208/final"
  "outputs/checkpoints/no_nav_bp3_200k_epoch/no_nav_bp3_200k_epoch_gc_b16_eval64_from_bp3final_20260508_072958/final"
  "outputs/checkpoints/no_nav_bp5_hidden_interface/no_nav_bp5_hidden_interface_sanity_nogc_b8_nodecode_from_step_012500_20260509_164252/final"
)

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  ckpt="${CHECKPOINT_DIRS[$index]}"
  echo "[$(date -Is)] feature extraction start: ${name} ${PREFIX_TYPE}"
  "$VENV_PYTHON" scripts/69_extract_hidden_to_action_features.py \
    --corpus-jsonl "$CORPUS_JSONL" \
    --split-sample-ids-json "$SPLIT_IDS_JSON" \
    --checkpoint-name "$name" \
    --checkpoint-dir "$ckpt" \
    --student-model "$COSMOS_STUDENT_MODEL" \
    --prefix-type "$PREFIX_TYPE" \
    --output-dir "$FEATURE_ROOT" \
    --splits probe_train probe_val probe_test \
    --batch-size "$FEATURE_BATCH_SIZE" \
    --shard-size "$FEATURE_SHARD_SIZE"

  echo "[$(date -Is)] probe train start: ${name} ${PREFIX_TYPE}"
  if [[ -f "$PROBE_ROOT/${name}_${PREFIX_TYPE}/summary.json" ]]; then
    echo "[$(date -Is)] probe train skip existing: ${name} ${PREFIX_TYPE}"
    echo "[$(date -Is)] done: ${name} ${PREFIX_TYPE}"
    continue
  fi
  "$VENV_PYTHON" scripts/70_train_hidden_to_action_probe.py \
    --feature-root "$FEATURE_ROOT" \
    --checkpoint-name "$name" \
    --prefix-type "$PREFIX_TYPE" \
    --output-dir "$PROBE_ROOT/${name}_${PREFIX_TYPE}" \
    --student-model "$COSMOS_STUDENT_MODEL" \
    --epochs "$PROBE_EPOCHS" \
    --batch-size "$PROBE_BATCH_SIZE" \
    --hidden-dim "$PROBE_HIDDEN_DIM" \
    --lr "$PROBE_LR" \
    --device cuda
  echo "[$(date -Is)] done: ${name} ${PREFIX_TYPE}"
done

echo "[$(date -Is)] all hidden-to-action teacher-prefix probes finished"
