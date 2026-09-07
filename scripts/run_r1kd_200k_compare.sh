#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"
cd "${ROOT}"
PY=".venv/bin/python"

CORPUS_VAL="data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
REF_FROZEN="outputs/references/backbone_eval_v1/frozen_val512_teacher_greedy_ref_v1/rows.jsonl"
CAND="outputs/checkpoints/stepb_r1kd_200k/r1kd_200k_20260717/fullft_lr3e5_r1kd_200k_e1/best_decode"
BASE="outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode"

OUT="outputs/reports/stepb_r1kd_200k/r1kd_200k_20260717/compare"
mkdir -p "${OUT}"
DLOG="${OUT}/driver.log"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1" >> "${DLOG}"; }

# existing baseline greedy P-G (CE-only FullFT-200K) already at:
BASE_PG="outputs/reports/stepb_r1_signcheck/r1_signcheck_20260716_0726/baseline/fullft200k_pg_teacher_ref_summary.json"

pg(){ # ckpt summary
  "${PY}" -u scripts/70_eval_checkpoint_free_run.py \
    --corpus-jsonl "${CORPUS_VAL}" --checkpoint-dir "$1" --split val --num-samples 512 \
    --max-new-tokens 320 --no-do-sample --temperature 1.0 --top-p 1.0 \
    --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo --fuse-history-tokens \
    --reference-id teacher_greedy_ref_v1 --reference-jsonl "${REF_FROZEN}" \
    --reference-token-field selected_traj_tokens --reference-xyz-field selected_xyz \
    --summary-json "$2" --device cuda >> "${OUT}/pg.log" 2>&1
}
testb(){ # ckpt name summary samples
  "${PY}" -u scripts/82_eval_test_b_teacher_forced.py \
    --corpus-jsonl "${CORPUS_VAL}" --checkpoint-dir "$1" --checkpoint-name "$2" \
    --split val --num-samples 512 --batch-size 4 \
    --image-prompt-style camera_labeled --prompt-text-style official_alpamayo --fuse-history-tokens \
    --summary-json "$3" --samples-jsonl "$4" --save-token-sequences --device cuda >> "${OUT}/testb.log" 2>&1
}
ps6(){ # ckpt outdir summary
  "${PY}" -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${CORPUS_VAL}" --checkpoint-dir "$1" --split val --num-samples 512 \
    --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo --fuse-history-tokens --geometry-reference gt \
    --batch-size 4 --samples-per-row 6 --temperature 1.0 --top-p 1.0 --max-new-tokens 320 --seed 42 \
    --device cuda --output-dir "$2" --summary-json "$3" --skip-overlays >> "${OUT}/ps6.log" 2>&1
}

say "START compare candidate(200K+KD) vs baseline(CE-only 200K)"

say "1/5 candidate greedy P-G val512"
pg "${CAND}" "${OUT}/cand_pg_teacher_ref_summary.json" && say "  done" || say "  FAILED"

say "2/5 candidate teacher-forced dist"
testb "${CAND}" r1kd_200k "${OUT}/cand_testb_summary.json" "${OUT}/cand_testb_samples.jsonl" && say "  done" || say "  FAILED"

say "3/5 baseline teacher-forced dist"
testb "${BASE}" fullft_ceonly_200k "${OUT}/base_testb_summary.json" "${OUT}/base_testb_samples.jsonl" && say "  done" || say "  FAILED"

say "4/5 candidate minADE6"
ps6 "${CAND}" "${OUT}/cand_ps6" "${OUT}/cand_ps6_summary.json" && say "  done" || say "  FAILED"

say "5/5 baseline minADE6"
ps6 "${BASE}" "${OUT}/base_ps6" "${OUT}/base_ps6_summary.json" && say "  done" || say "  FAILED"

say "computing paired CIs"
# greedy P-G CI (candidate vs existing baseline P-G)
"${PY}" scripts/stepb_compare_decode_summaries.py \
  --baseline-summary "${BASE_PG}" --candidate-summary "${OUT}/cand_pg_teacher_ref_summary.json" \
  --baseline-name ceonly200k --candidate-name kd200k \
  --bootstrap 5000 --seed 42 \
  --output-json "${OUT}/ci_greedy_pg.json" --output-md "${OUT}/ci_greedy_pg.md" >> "${DLOG}" 2>&1 && say "  greedy CI done" || say "  greedy CI FAILED"

say "ALL DONE"
touch "${OUT}/_COMPLETE"
