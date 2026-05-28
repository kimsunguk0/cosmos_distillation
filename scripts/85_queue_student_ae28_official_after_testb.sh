#!/usr/bin/env bash
set -euo pipefail

cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

TESTB_SUMMARY="${TESTB_SUMMARY:-outputs/reports/no_nav_distill/test_b_teacher_forced_ar_sched_rowscale_best_decode_20260518/best_decode_full_val_summary.json}"
RUN_ROOT="${RUN_ROOT:-outputs/action_expert/student_ae28_official/queued_$(date +%Y%m%d_%H%M%S)}"
STUDENT_CKPT="${STUDENT_CKPT:-outputs/checkpoints/no_nav_camera_labeled_official_200k/no_nav_official12500_topk_sched16_ar_ramp_p20_rowscale_evalfix_20260517/best_decode}"
TEACHER_CKPT="${TEACHER_CKPT:-/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B}"
CORPUS="${CORPUS:-data/corpus/no_nav_teacher_pair_300chunks.jsonl}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "$RUN_ROOT"
echo "$(date -Iseconds) queue_start run_root=$RUN_ROOT"
echo "$(date -Iseconds) waiting_for_testb_summary=$TESTB_SUMMARY"
while [ ! -s "$TESTB_SUMMARY" ]; do
  sleep 60
done
echo "$(date -Iseconds) testb_done_detected"

COMMON_ARGS=(
  --corpus-jsonl "$CORPUS"
  --split train
  --student-checkpoint-dir "$STUDENT_CKPT"
  --teacher-checkpoint-path "$TEACHER_CKPT"
  --teacher-load-device cpu
  --device "$DEVICE"
  --student-dtype bfloat16
  --ae-dtype bfloat16
  --attn-implementation sdpa
  --max-new-tokens 192
  --num-time-samples 1
)

echo "$(date -Iseconds) stage0_overfit16_start"
.venv/bin/python -u scripts/84_train_student_ae28_official.py \
  "${COMMON_ARGS[@]}" \
  --num-samples 16 \
  --steps 500 \
  --batch-size 2 \
  --eval-samples 16 \
  --eval-batch-size 2 \
  --eval-every 50 \
  --log-every 10 \
  --expert-lr 2e-5 \
  --proj-lr 6e-5 \
  --output-dir "$RUN_ROOT/stage0_overfit16"
echo "$(date -Iseconds) stage0_overfit16_done"

echo "$(date -Iseconds) stage1_pilot1k_start"
.venv/bin/python -u scripts/84_train_student_ae28_official.py \
  "${COMMON_ARGS[@]}" \
  --num-samples 1000 \
  --steps 1000 \
  --batch-size 4 \
  --eval-samples 128 \
  --eval-batch-size 4 \
  --eval-every 250 \
  --log-every 20 \
  --expert-lr 1e-5 \
  --proj-lr 3e-5 \
  --output-dir "$RUN_ROOT/stage1_pilot1k"
echo "$(date -Iseconds) stage1_pilot1k_done"

echo "$(date -Iseconds) stage2_pilot30k_start"
.venv/bin/python -u scripts/84_train_student_ae28_official.py \
  "${COMMON_ARGS[@]}" \
  --num-samples 30000 \
  --steps 5000 \
  --batch-size 4 \
  --eval-samples 512 \
  --eval-batch-size 4 \
  --eval-every 500 \
  --log-every 25 \
  --expert-lr 1e-5 \
  --proj-lr 3e-5 \
  --output-dir "$RUN_ROOT/stage2_pilot30k"
echo "$(date -Iseconds) stage2_pilot30k_done"

echo "$(date -Iseconds) queue_done run_root=$RUN_ROOT"
