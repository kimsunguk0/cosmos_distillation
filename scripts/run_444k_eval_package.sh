#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"; cd "${ROOT}"
PY=".venv/bin/python"
CORPUS="data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
REF="outputs/references/backbone_eval_v1/frozen_val512_teacher_greedy_ref_v1/rows.jsonl"
CKPT="outputs/checkpoints/stepb_ceonly_444k/ceonly_444k_20260718/fullft_lr3e5_ceonly_444k_e1/best_decode"
OUT="outputs/reports/stepb_ceonly_444k/ceonly_444k_20260718/eval"; mkdir -p "${OUT}"
DLOG="${OUT}/driver.log"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1" >> "${DLOG}"; }

say "START 444K eval package"

# ① greedy P-G vs teacher_ref (scaling-curve greedy point + A: greedy ADE/FDE)
say "1 greedy P-G vs teacher_ref"
"${PY}" -u scripts/70_eval_checkpoint_free_run.py --corpus-jsonl "${CORPUS}" --checkpoint-dir "${CKPT}" \
  --split val --num-samples 512 --max-new-tokens 320 --no-do-sample --temperature 1.0 --top-p 1.0 \
  --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled --prompt-text-style official_alpamayo \
  --fuse-history-tokens --reference-id teacher_greedy_ref_v1 --reference-jsonl "${REF}" \
  --reference-token-field selected_traj_tokens --reference-xyz-field selected_xyz \
  --summary-json "${OUT}/444k_pg_teacher_ref_summary.json" --device cuda >> "${OUT}/pg.log" 2>&1 && say " done" || say " FAIL"

# A + scaling-curve minADE6: T=1.0/top-p1.0 (vs GT, 6 samples)
say "2 minADE6 T=1.0 top-p1.0"
"${PY}" -u scripts/25_decode_checkpoint_overlays.py --corpus-jsonl "${CORPUS}" --checkpoint-dir "${CKPT}" \
  --split val --num-samples 512 --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo --fuse-history-tokens --geometry-reference gt \
  --batch-size 4 --samples-per-row 6 --temperature 1.0 --top-p 1.0 --max-new-tokens 320 --seed 42 --device cuda \
  --output-dir "${OUT}/444k_ps6_t1" --summary-json "${OUT}/444k_ps6_t1_summary.json" --skip-overlays >> "${OUT}/ps6_t1.log" 2>&1 && say " done" || say " FAIL"

# A: minADE6 T=0.6/top-p0.98 (deployment-like sampling)
say "3 minADE6 T=0.6 top-p0.98"
"${PY}" -u scripts/25_decode_checkpoint_overlays.py --corpus-jsonl "${CORPUS}" --checkpoint-dir "${CKPT}" \
  --split val --num-samples 512 --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled \
  --prompt-text-style official_alpamayo --fuse-history-tokens --geometry-reference gt \
  --batch-size 4 --samples-per-row 6 --temperature 0.6 --top-p 0.98 --max-new-tokens 320 --seed 42 --device cuda \
  --output-dir "${OUT}/444k_ps6_t06" --summary-json "${OUT}/444k_ps6_t06_summary.json" --skip-overlays >> "${OUT}/ps6_t06.log" 2>&1 && say " done" || say " FAIL"

# ① matched argmax: teacher-forced samples + audit
say "4 teacher-forced testb"
"${PY}" -u scripts/82_eval_test_b_teacher_forced.py --corpus-jsonl "${CORPUS}" --checkpoint-dir "${CKPT}" \
  --checkpoint-name ceonly_444k --split val --num-samples 512 --batch-size 4 \
  --image-prompt-style camera_labeled --prompt-text-style official_alpamayo --fuse-history-tokens \
  --summary-json "${OUT}/444k_testb_summary.json" --samples-jsonl "${OUT}/444k_testb_samples.jsonl" \
  --save-token-sequences --device cuda >> "${OUT}/testb.log" 2>&1 && say " done" || say " FAIL"

say "5 matched argmax audit"
"${PY}" -u scripts/audit_val512_matched_argmax_vs_teacher_top1.py --corpus-jsonl "${CORPUS}" \
  --model-samples ceonly_444k "${OUT}/444k_testb_samples.jsonl" --output-dir "${OUT}/444k_matched_argmax" \
  >> "${OUT}/matched.log" 2>&1 && say " done" || say " FAIL"

# CIs: 444K vs 200K CE (greedy)
say "6 paired CI greedy 444K vs 200K-CE"
"${PY}" scripts/stepb_compare_decode_summaries.py \
  --baseline-summary outputs/reports/stepb_r1_signcheck/r1_signcheck_20260716_0726/baseline/fullft200k_pg_teacher_ref_summary.json \
  --candidate-summary "${OUT}/444k_pg_teacher_ref_summary.json" \
  --baseline-name ce200k --candidate-name ce444k --bootstrap 5000 --seed 42 \
  --output-json "${OUT}/ci_greedy_444k_vs_200kce.json" --output-md "${OUT}/ci_greedy_444k_vs_200kce.md" >> "${DLOG}" 2>&1 && say " done" || say " FAIL"

say "ALL DONE"; touch "${OUT}/_COMPLETE"
