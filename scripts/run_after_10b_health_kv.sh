#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"; cd "${ROOT}"
PY=".venv/bin/python"
CORPUS="data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
CKPT444="outputs/checkpoints/stepb_ceonly_444k/ceonly_444k_20260718/fullft_lr3e5_ceonly_444k_e1/best_decode"
OUT="outputs/reports/stepb_ceonly_444k/ceonly_444k_20260718/health_kv"; mkdir -p "${OUT}"
DLOG="${OUT}/driver.log"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1" >> "${DLOG}"; }

# ---- wait until the 10B on-policy run frees the GPU ----
say "waiting for 10B run to finish"
while pgrep -f 113_eval_10b_onpolicy >/dev/null; do sleep 60; done
say "10B done -> GPU free, starting health + KV"

# ---- ② black-delta (image ablation): 444K, black vs normal, same cfg as ps6_t1 ----
say "black-delta: 444K minADE6 with --image-ablation black (T=1, 6-sample)"
"${PY}" -u scripts/25_decode_checkpoint_overlays.py --corpus-jsonl "${CORPUS}" --checkpoint-dir "${CKPT444}" \
  --split val --num-samples 512 --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo --fuse-history-tokens --geometry-reference gt \
  --batch-size 4 --samples-per-row 6 --temperature 1.0 --top-p 1.0 --max-new-tokens 320 --seed 42 \
  --image-ablation black --device cuda \
  --output-dir "${OUT}/444k_ps6_black" --summary-json "${OUT}/444k_ps6_black_summary.json" --skip-overlays \
  >> "${OUT}/black.log" 2>&1 && say "  black done" || say "  black FAIL"

# ---- Ⓑ+Ⓒ hidden/KV representation compare: 3x 200K ckpts + teacher anchor ----
say "hidden/KV compare (CE / CE+KD / LoRA) via script 112"
"${PY}" scripts/112_extract_hidden_kv_repr_compare.py \
  --checkpoint-dir CE=outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode \
  --checkpoint-dir CE_KD=outputs/checkpoints/stepb_r1kd_200k/r1kd_200k_20260717/fullft_lr3e5_r1kd_200k_e1/best_decode \
  --checkpoint-dir LoRA=outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode \
  --ce-baseline-dir CE --num-samples 512 --batch-size 2 --device cuda --compute-hidden-action-r2 \
  --output-json "${OUT}/hidden_kv_repr_compare_val512.json" >> "${OUT}/kv.log" 2>&1 && say "  kv done" || say "  kv FAIL"

say "ALL DONE"; touch "${OUT}/_COMPLETE"
