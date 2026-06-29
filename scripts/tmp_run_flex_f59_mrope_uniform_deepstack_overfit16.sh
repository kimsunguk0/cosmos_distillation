#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
B0_TRAJONLY="outputs/reports/b0_step006250_flexheldout16_decode_trajonly_summary.json"
F0_K1792="outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607"
RUN_NAME="flex_f59_mrope_uniform_deepstack_from_f0_overfit16_s3000_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f59_chain_start\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\",\"target\":\"${B0_TRAJONLY}\"}"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F0_K1792}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 16 \
  --prompt-mode-override joint \
  --target-mode-override traj_only \
  --image-ablations normal \
  --paired-ablation none \
  --max-steps 3000 \
  --batch-size 1 \
  --learning-rate 5e-7 \
  --flex-lr 5e-7 \
  --lora-lr 5e-7 \
  --multimodal-projector-lr 5e-7 \
  --deepstack-projector-lr 1e-4 \
  --weight-decay 0.0 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 500 \
  --seed 42 \
  --traj-kl-weight 0.0 \
  --text-kl-weight 0.0 \
  --format-kl-weight 0.0 \
  --boundary-cos-weight 0.0 \
  --boundary-norm-weight 0.0 \
  --boundary-mse-weight 0.0 \
  --pairwise-boundary-delta-cos-weight 0.0 \
  --pairwise-boundary-delta-norm-weight 0.0 \
  --pairwise-traj-logprob-delta-weight 0.0 \
  --pairwise-free-run-margin-weight 0.0 \
  --free-run-token-targets normal="${B0_TRAJONLY}" \
  --free-run-token-ce-weight 1.0 \
  --free-run-token-ce-modes normal \
  --free-run-end-token-ce-weight 0.05 \
  --prefix-token-ce-weight 0.0 \
  --traj-state-cos-weight 1.0 \
  --traj-state-norm-weight 0.10 \
  --traj-state-mse-weight 0.001 \
  --deepstack-feature-tokens-per-image 112 \
  --deepstack-feature-cos-weight 0.0 \
  --deepstack-feature-norm-weight 0.0 \
  --deepstack-feature-mse-weight 1.0 \
  --free-run-token-force-context \
  --free-run-token-context-source target \
  --cache-teacher-targets \
  --cache-collated-batches \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --flex-scene-deepstack \
  --flex-deepstack-projector-rank 64 \
  --train-flex-deepstack-projector \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 16
  --max-new-tokens 160
  --prompt-mode joint
  --target-mode traj_only
  --image-prompt-style camera_labeled
  --prompt-text-style official_alpamayo
  --fuse-history-tokens
  --geometry-reference teacher
  --batch-size 1
  --samples-per-row 1
  --skip-overlays
  --disable-failure-tags
  --preserve-flex-positions
  --flex-selection-strategy uniform
  --flex-scene-deepstack
)

for CKPT_NAME in step_000500 step_001000 step_001500 step_002000 step_002500 step_003000 final; do
  CKPT_DIR="outputs/checkpoints/${RUN_NAME}/${CKPT_NAME}"
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "{\"event\":\"f59_decode_skip_missing\",\"checkpoint\":\"${CKPT_DIR}\"}"
    continue
  fi
  SAFE_NAME="${CKPT_NAME//_/-}"
  DECODE_SUMMARY="outputs/reports/${RUN_NAME}_${SAFE_NAME}_decode_trajonly_mrope_uniform_deepstack_summary.json"
  PARITY_SUMMARY="outputs/reports/${RUN_NAME}_${SAFE_NAME}_b0_trajonly_parity_summary.json"
  echo "{\"event\":\"f59_decode_start\",\"checkpoint\":\"${CKPT_DIR}\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --output-dir "outputs/reports/${RUN_NAME}_${SAFE_NAME}_decode_trajonly_mrope_uniform_deepstack" \
    --summary-json "${DECODE_SUMMARY}"

  .venv/bin/python -u scripts/109_compare_decode_to_free_run_targets.py \
    --decode-summary "${DECODE_SUMMARY}" \
    --target-summary "${B0_TRAJONLY}" \
    --corpus-jsonl "${CORPUS}" \
    --split val \
    --num-samples 16 \
    --summary-json "${PARITY_SUMMARY}"
done

.venv/bin/python - <<'PY'
import json
from pathlib import Path

run_name = "flex_f59_mrope_uniform_deepstack_from_f0_overfit16_s3000_20260607"
rows = []
for summary_path in sorted(Path("outputs/reports").glob(f"{run_name}_*_b0_trajonly_parity_summary.json")):
    data = json.loads(summary_path.read_text())
    label = summary_path.name.removeprefix(f"{run_name}_").removesuffix("_b0_trajonly_parity_summary.json")
    rows.append(
        {
            "label": label,
            "token": float(data.get("avg_target_token_match_rate", float("nan"))),
            "ade": float(data.get("avg_target_ade_m", float("nan"))),
            "fde": float(data.get("avg_target_fde_m", float("nan"))),
            "unique": float(data.get("avg_generated_unique_token_count", float("nan"))),
            "max_run": float(data.get("avg_generated_max_same_token_run", float("nan"))),
            "path": str(summary_path),
        }
    )
best = min(rows, key=lambda row: row["ade"]) if rows else None
print(json.dumps({"event": "f59_done", "best": best, "rows": rows}, ensure_ascii=True))
PY
