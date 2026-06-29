#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# QAT (Quantization-Aware Training) for ML-FLEX K512 backbone
#
# Uses 09_train_distill.py directly with --qat-quantization flag.
# ModelOpt INT4 AWQ fake-quant is applied to the LLM language_model only.
# ViT, FLEX, LoRA adapters stay FP16.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="outputs/logs/qat_mlflex_k512_int4awq_${TIMESTAMP}.log"
mkdir -p outputs/logs

exec .venv/bin/python scripts/09_train_distill.py \
  --stage-config configs/train/stage_qat_mlflex_k512_int4awq_20k.yaml \
  --corpus-jsonl data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl \
  --init-checkpoint-dir outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final \
  --output-dir outputs/checkpoints/qat_mlflex_k512_int4awq_20k_e3 \
  --qat-quantization int4_awq \
  --qat-calib-samples 512 \
  --max-val-samples 512 \
  --num-workers 8 \
  --pin-memory \
  --persistent-workers \
  --prefetch-factor 2 \
  --log-every-steps 50 \
  2>&1 | tee "$LOG"
