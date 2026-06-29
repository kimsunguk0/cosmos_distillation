#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

CORPUS="data/corpus/flex_heldout256_stage2val_seed42.jsonl"
B0_CKPT="outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250"
PERIMAGE_F0="outputs/checkpoints/flex_f8_factorized_per_image_k896_camtime_untrained_from_b0_20260606"
B0_PREFIX="outputs/reports/b0_step006250_flexheldout256"
WARM_RUN="flex_f28a_perimage_preservepos_parity_heldout256_s2000_lr2e6_20260607"
RUN_NAME="flex_f28b_perimage_preservepos_margin_from_f28a_heldout256_s3000_lr1e6_20260607"
LOG_PATH="outputs/logs/${RUN_NAME}_chain.log"

mkdir -p outputs/logs
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "{\"event\":\"f28_chain_start\",\"warm_run\":\"${WARM_RUN}\",\"run_name\":\"${RUN_NAME}\",\"corpus\":\"${CORPUS}\",\"preserve_flex_positions\":true}"

COMMON_DECODE_ARGS=(
  --corpus-jsonl "${CORPUS}"
  --split val
  --num-samples 256
  --max-new-tokens 192
  --prompt-mode joint
  --target-mode joint
  --image-prompt-style camera_labeled
  --prompt-text-style official_alpamayo
  --fuse-history-tokens
  --geometry-reference teacher
  --batch-size 1
  --samples-per-row 1
  --skip-overlays
  --disable-failure-tags
)

if [[ ! -f "${B0_PREFIX}_decode_normal_summary.json" ]]; then
  echo "{\"event\":\"f28_b0_decode_start\",\"mode\":\"normal\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${B0_CKPT}" \
    --output-dir "${B0_PREFIX}_decode_normal" \
    --summary-json "${B0_PREFIX}_decode_normal_summary.json"
  echo "{\"event\":\"f28_b0_decode_done\",\"mode\":\"normal\"}"
fi

if [[ ! -f "${B0_PREFIX}_decode_camera_shuffle_summary.json" ]]; then
  echo "{\"event\":\"f28_b0_decode_start\",\"mode\":\"camera_shuffle\"}"
  .venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
    "${COMMON_DECODE_ARGS[@]}" \
    --checkpoint-dir "${B0_CKPT}" \
    --image-ablation camera_shuffle \
    --output-dir "${B0_PREFIX}_decode_camera_shuffle" \
    --summary-json "${B0_PREFIX}_decode_camera_shuffle_summary.json"
  echo "{\"event\":\"f28_b0_decode_done\",\"mode\":\"camera_shuffle\"}"
fi

echo "{\"event\":\"f28a_warmup_start\"}"
.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${PERIMAGE_F0}" \
  --output-dir "outputs/checkpoints/${WARM_RUN}" \
  --split val \
  --max-train-samples 256 \
  --paired-ablation camera_shuffle \
  --max-steps 2000 \
  --batch-size 2 \
  --learning-rate 2e-6 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 1000 \
  --seed 42 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.20 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.05 \
  --boundary-norm-weight 0.10 \
  --pairwise-boundary-delta-cos-weight 0.0 \
  --pairwise-boundary-delta-norm-weight 0.0 \
  --pairwise-traj-logprob-delta-weight 0.0 \
  --pairwise-free-run-margin-weight 0.0 \
  --no-free-run-token-force-context \
  --cache-teacher-targets \
  --cache-collated-batches \
  --preserve-flex-positions \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${WARM_RUN}_train_summary.json"
echo "{\"event\":\"f28a_warmup_done\"}"

echo "{\"event\":\"f28b_margin_start\"}"
.venv/bin/python -u scripts/105_train_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "outputs/checkpoints/${WARM_RUN}/final" \
  --output-dir "outputs/checkpoints/${RUN_NAME}" \
  --split val \
  --max-train-samples 256 \
  --paired-ablation camera_shuffle \
  --max-steps 3000 \
  --batch-size 2 \
  --learning-rate 1e-6 \
  --grad-clip-norm 5.0 \
  --log-every 100 \
  --save-every 1500 \
  --seed 42 \
  --traj-kl-weight 1.0 \
  --text-kl-weight 0.20 \
  --format-kl-weight 0.05 \
  --boundary-cos-weight 0.05 \
  --boundary-norm-weight 0.10 \
  --pairwise-boundary-delta-cos-weight 0.05 \
  --pairwise-boundary-delta-norm-weight 0.05 \
  --pairwise-traj-logprob-delta-weight 0.10 \
  --pairwise-free-run-margin-weight 0.05 \
  --pairwise-free-run-margin 0.10 \
  --free-run-token-targets normal="${B0_PREFIX}_decode_normal_summary.json",camera_shuffle="${B0_PREFIX}_decode_camera_shuffle_summary.json" \
  --free-run-token-ce-weight 0.0 \
  --no-free-run-token-force-context \
  --cache-teacher-targets \
  --cache-collated-batches \
  --preserve-flex-positions \
  --train-flex \
  --unfreeze-lora-last-n-layers 4 \
  --unfreeze-multimodal-projector \
  --summary-json "outputs/reports/${RUN_NAME}_train_summary.json"
echo "{\"event\":\"f28b_margin_done\"}"

F28_CKPT="outputs/checkpoints/${RUN_NAME}/final"

.venv/bin/python -u scripts/104_eval_flex_teacher_parity.py \
  --corpus-jsonl "${CORPUS}" \
  --teacher-checkpoint-dir "${B0_CKPT}" \
  --student-checkpoint-dir "${F28_CKPT}" \
  --split val \
  --num-samples 256 \
  --batch-size 1 \
  --preserve-flex-positions \
  --summary-json "outputs/reports/${RUN_NAME}_eval_heldout256_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F28_CKPT}" \
  --preserve-flex-positions \
  --output-dir "outputs/reports/${RUN_NAME}_heldout256_decode_normal" \
  --summary-json "outputs/reports/${RUN_NAME}_heldout256_decode_normal_summary.json"

NORMAL_ADE="$(.venv/bin/python - <<PY
import json
p = "outputs/reports/${RUN_NAME}_heldout256_decode_normal_summary.json"
print(json.load(open(p)).get("avg_ade_m", 999.0))
PY
)"
echo "{\"event\":\"f28_normal_gate\",\"avg_ade_m\":${NORMAL_ADE}}"

.venv/bin/python - <<PY
import sys
ade = float("${NORMAL_ADE}")
if ade > 3.35:
    print(f"position-preserving normal ADE {ade:.3f} exceeds gate 3.35; skip shuffle/black decodes", file=sys.stderr)
    raise SystemExit(10)
PY

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F28_CKPT}" \
  --preserve-flex-positions \
  --image-ablation camera_shuffle \
  --output-dir "outputs/reports/${RUN_NAME}_heldout256_decode_camera_shuffle" \
  --summary-json "outputs/reports/${RUN_NAME}_heldout256_decode_camera_shuffle_summary.json"

.venv/bin/python -u scripts/25_decode_checkpoint_overlays.py \
  "${COMMON_DECODE_ARGS[@]}" \
  --checkpoint-dir "${F28_CKPT}" \
  --preserve-flex-positions \
  --image-ablation black \
  --output-dir "outputs/reports/${RUN_NAME}_heldout256_decode_black" \
  --summary-json "outputs/reports/${RUN_NAME}_heldout256_decode_black_summary.json"

echo "{\"event\":\"f28_chain_done\",\"run_name\":\"${RUN_NAME}\"}"
