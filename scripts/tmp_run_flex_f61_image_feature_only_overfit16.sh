#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
STUDENT_CKPT="${STUDENT_CKPT:-outputs/checkpoints/flex_f54_f0_perimage_k1792_camtime_from_b0_20260607}"
RUN_NAME="${RUN_NAME:-flex_f61_image_feature_only_from_f0_overfit16_s1000_lr1e4_20260607}"
MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-1e-4}"
FLEX_LR="${FLEX_LR:-${LR}}"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"
export RUN_NAME

mkdir -p outputs/logs outputs/reports
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f61_chain_start\",\"run_name\":\"${RUN_NAME}\",\"max_steps\":${MAX_STEPS},\"lr\":\"${LR}\",\"flex_lr\":\"${FLEX_LR}\",\"student_ckpt\":\"${STUDENT_CKPT}\",\"corpus\":\"${CORPUS}\"}"

.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${STUDENT_CKPT}" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 16 \
  --prompt-mode-override joint \
  --target-mode-override traj_only \
  --image-ablations normal \
  --paired-ablation none \
  --max-steps "${MAX_STEPS}" \
  --batch-size 1 \
  --learning-rate "${LR}" \
  --flex-lr "${FLEX_LR}" \
  --weight-decay 0.0 \
  --grad-clip-norm 5.0 \
  --log-every 50 \
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
  --free-run-token-ce-weight 0.0 \
  --free-run-end-token-ce-weight 0.0 \
  --prefix-token-ce-weight 0.0 \
  --traj-state-cos-weight 0.0 \
  --traj-state-norm-weight 0.0 \
  --traj-state-mse-weight 0.0 \
  --image-feature-tokens-per-image 112 \
  --image-feature-cos-weight 1.0 \
  --image-feature-norm-weight 0.1 \
  --image-feature-mse-weight 1.0 \
  --cache-teacher-targets \
  --cache-collated-batches \
  --preserve-flex-positions \
  --flex-selection-strategy uniform \
  --train-flex \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"

.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

run_name = os.environ["RUN_NAME"]
summary_path = Path("outputs/reports") / f"{run_name}_train_summary.json"
payload = json.loads(summary_path.read_text(encoding="utf-8"))
history = payload.get("history") or []
best = None
last = None
for row in history:
    metrics = row.get("metrics") or {}
    if "image_feature_cos" not in metrics:
        continue
    current = {
        "step": row.get("step"),
        "image_feature_cos": metrics.get("image_feature_cos"),
        "image_feature_mse": metrics.get("image_feature_mse"),
        "image_feature_norm_ratio": metrics.get("image_feature_norm_ratio"),
        "loss": metrics.get("loss"),
        "grad_norm": metrics.get("grad_norm"),
    }
    last = current
    if best is None or float(current["image_feature_cos"]) > float(best["image_feature_cos"]):
        best = current
print(json.dumps({"event": "f61_feature_only_summary", "best": best, "last": last}, ensure_ascii=True), flush=True)
PY
