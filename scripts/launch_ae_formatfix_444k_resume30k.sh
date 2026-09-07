#!/usr/bin/env bash
# Resume the format-fix 444K student_free AE28 run after the 2026-07-30 02:13 CUDA OOM.
#
# The original run died at step 39,300; the newest checkpoint is step_030000.pt, so we
# restart from 30,000. Differences from the original launch:
#   --resume-ae-checkpoint / --start-step  restore the step-30,000 bundle weights
#   --save-every 5000 (was 10000)          cap the loss from another crash at ~9 h
#   --skip-initial-eval dropped            eval at 30,000 must reproduce ADE 2.2516
# Optimizer state is not stored in the checkpoint, so Adam moments restart from zero.
#
# VRAM: this box is shared. The default --cleanup-every 1 calls torch.cuda.empty_cache()
# every train step, handing cached VRAM back to the driver; a co-tenant job takes it and
# we OOM on the next allocation. Both cleanups are disabled so the caching allocator keeps
# its high-water mark, and --reserve-vram-gib claims the footprint up front. Disabling
# empty_cache trades driver-level churn for intra-process fragmentation, which is what
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True below is for.
set -euo pipefail
cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

OUT=outputs/action_expert/ae_formatfix_444k_studentfree_20260727/main
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -u scripts/84_train_student_ae28_official.py \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k.jsonl \
  --num-samples 391586 \
  --steps 50000 \
  --batch-size 8 \
  --eval-samples 1024 \
  --eval-num-paths 6 \
  --eval-temperature 0.85 \
  --eval-selection-method mean_traj \
  --eval-vectorize-paths \
  --eval-batch-size 8 \
  --eval-every 10000 \
  --log-every 50 \
  --val-samples 10000 \
  --split-scan-all \
  --split-cache-json outputs/action_expert/split_cache_444k_10k_seed42.json \
  --student-checkpoint-dir outputs/checkpoints/stepb_ceonly_444k_formatfix_20260726/formatfix_e0/best_decode \
  --output-dir "${OUT}" \
  --attn-implementation flash_attention_2 \
  --num-time-samples 8 \
  --lr-warmup-steps 100 \
  --no-norm-bias-decay \
  --fused-adamw \
  --seed 42 \
  --eval-seed-mode fixed \
  --save-every 5000 \
  --cleanup-every 0 \
  --eval-cleanup-every 0 \
  --reserve-vram-gib 60 \
  --resume-ae-checkpoint "${OUT}/step_030000.pt" \
  --start-step 30000 \
  2>&1 | tee -a "${OUT}/train.log"
