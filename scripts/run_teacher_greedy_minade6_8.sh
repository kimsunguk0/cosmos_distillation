#!/usr/bin/env bash
set -uo pipefail
ROOT="/home/pm97/workspace/sukim/distillation/cosmos_distillation"; cd "${ROOT}"
PY=".venv/bin/python"
REF="outputs/references/backbone_eval_v1/frozen_val512_teacher_greedy_ref_v1/rows.jsonl"
COR="data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
OUT="outputs/reports/teacher_greedy_minade6_20260721"; mkdir -p "${OUT}"
DLOG="${OUT}/driver.log"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$1" >> "${DLOG}"; }

run(){ # name ckpt extra_flags...
  local name="$1" ckpt="$2"; shift 2
  local sj="${OUT}/${name}_summary.json"
  if [[ -s "${sj}" ]]; then say "skip ${name}"; return; fi
  say "START ${name}"
  "${PY}" -u scripts/25_decode_checkpoint_overlays.py \
    --corpus-jsonl "${COR}" --checkpoint-dir "${ckpt}" --split val --num-samples 512 \
    --prompt-mode joint --target-mode joint --image-prompt-style camera_labeled \
    --prompt-text-style official_alpamayo --fuse-history-tokens \
    --geometry-reference teacher_greedy --reference-id teacher_greedy_ref_v1 \
    --reference-jsonl "${REF}" --reference-token-field selected_traj_tokens \
    --batch-size 2 --samples-per-row 6 --max-new-tokens 320 --seed 42 \
    --device cuda:0 --output-dir "${OUT}/${name}" --summary-json "${sj}" --skip-overlays \
    "$@" >> "${OUT}/${name}.log" 2>&1
  say "DONE ${name} (exit $?)"
}

C20K=outputs/checkpoints/stepb_followup/followup_20260711_023134/r0_f_fullft_lr3e5_20k/best_decode
C20KKD=outputs/checkpoints/stepb_r1_signcheck/r1_signcheck_20260716_0726/r1_fullft_lr3e5_tailkl_tau1_20k/best_decode
C200CE=outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/fullft_lr3e5_200k_e1/best_decode
C200KD=outputs/checkpoints/stepb_r1kd_200k/r1kd_200k_20260717/fullft_lr3e5_r1kd_200k_e1/best_decode
C200LORA=outputs/checkpoints/stepb_200k_double_promotion/double200k_20260711_181751/lprime_lora_lr2e4_200k_e1/best_decode
C444CE=outputs/checkpoints/stepb_ceonly_444k/ceonly_444k_20260718/fullft_lr3e5_ceonly_444k_e1/best_decode

say "teacher_greedy minADE6 8-run START (ref=teacher_greedy_ref_v1, samples6, batch2)"
run 20K_CE      "${C20K}"    --temperature 1.0 --top-p 1.0
run 20K_R1KD    "${C20KKD}"  --temperature 1.0 --top-p 1.0
run 200K_CE     "${C200CE}"  --temperature 1.0 --top-p 1.0
run 200K_KD     "${C200KD}"  --temperature 1.0 --top-p 1.0
run 200K_LoRA   "${C200LORA}" --temperature 1.0 --top-p 1.0
run 444K_CE_t1  "${C444CE}"  --temperature 1.0 --top-p 1.0
run 444K_CE_t06 "${C444CE}"  --temperature 0.6 --top-p 0.98
run 444K_CE_black "${C444CE}" --temperature 1.0 --top-p 1.0 --image-ablation black
say "ALL DONE"; touch "${OUT}/_COMPLETE"
