#!/usr/bin/env bash
# Flow-matching temperature sweep on the best AE28 checkpoint.
#
# The 444K student_free run peaked at step 30,000 (mean_traj ADE 2.2516); 40k and 50k are
# both worse, and best.pt/final.pt point at those, so the checkpoint is named explicitly.
# In flow matching, temperature only scales the initial noise x0 (alpamayo1_5/diffusion/
# flow_matching.py:172) and the euler integration is deterministic, so T=0 collapses to a
# single deterministic path — hence n=1 for that entry.
#
# Each eval logs ade_m (the selection method), ade_single_m (path 0) and ade_best_of_n_m
# (oracle minADE6) per row, so one eval per temperature yields all three metrics.
set -euo pipefail
cd /home/pm97/workspace/sukim/distillation/cosmos_distillation

SRC=outputs/action_expert/ae_formatfix_444k_studentfree_20260727/main
OUT=outputs/action_expert/ae_formatfix_444k_studentfree_20260727/tsweep_step30000
mkdir -p "${OUT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python -u scripts/84_train_student_ae28_official.py \
  --eval-only \
  --eval-sweep-json "@scripts/ae_formatfix_step30000_tsweep.json" \
  --resume-ae-checkpoint "${SRC}/step_030000.pt" \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k.jsonl \
  --num-samples 391586 \
  --steps 50000 \
  --batch-size 8 \
  --eval-samples 1024 \
  --eval-vectorize-paths \
  --eval-batch-size 8 \
  --log-every 50 \
  --val-samples 10000 \
  --split-scan-all \
  --split-cache-json outputs/action_expert/split_cache_444k_10k_seed42.json \
  --student-checkpoint-dir outputs/checkpoints/stepb_ceonly_444k_formatfix_20260726/formatfix_e0/best_decode \
  --output-dir "${OUT}" \
  --attn-implementation flash_attention_2 \
  --num-time-samples 8 \
  --no-norm-bias-decay \
  --fused-adamw \
  --seed 42 \
  --eval-seed-mode fixed \
  --cleanup-every 0 \
  --eval-cleanup-every 0 \
  --reserve-vram-gib 60 \
  2>&1 | tee -a "${OUT}/tsweep.log"
