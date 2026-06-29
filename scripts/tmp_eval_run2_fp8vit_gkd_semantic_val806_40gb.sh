#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT_DIR}/outputs/tmp/triton_cache}"

RUN_ROOT="outputs/benchmarks/run2_fp8vit_gkd_semantic_val806_20260619"
LOG_DIR="logs/b0_fp8_vit_step006250_20260618"
LOG="${LOG_DIR}/run2_fp8vit_gkd_semantic_val806_40gb.log"
CORPUS="data/corpus/benchmark_semantic_val_cap50_seed42.jsonl"
CKPT="outputs/checkpoints/b0_fp8_vit_step006250_20260618/run2_20k_fp8vit_gkd_from_step006250_val512_b8/final"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${TRITON_CACHE_DIR}"

run_decode() {
  local samples_per_row="$1"
  local tag="$2"
  local out_dir="${RUN_ROOT}/${tag}"
  local summary="${out_dir}/summary.json"
  mkdir -p "${out_dir}"

  if [[ -s "${summary}" ]]; then
    printf '%s skip_existing tag=%s summary=%s\n' "$(date -Is)" "${tag}" "${summary}" | tee -a "${LOG}"
    return
  fi

  printf '%s start tag=%s samples_per_row=%s checkpoint=%s\n' "$(date -Is)" "${tag}" "${samples_per_row}" "${CKPT}" | tee -a "${LOG}"

  .venv/bin/python -u - "${samples_per_row}" "${out_dir}" "${summary}" <<'PY' 2>&1 | tee -a "${LOG}"
import runpy
import sys

import torch

samples_per_row = sys.argv[1]
out_dir = sys.argv[2]
summary = sys.argv[3]

limit_gb = 40
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    frac = min(0.95, (limit_gb * (1024**3)) / total)
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    print({"event": "memory_fraction_set", "limit_gb": limit_gb, "fraction": frac}, flush=True)

sys.argv = [
    "scripts/25_decode_checkpoint_overlays.py",
    "--corpus-jsonl", "data/corpus/benchmark_semantic_val_cap50_seed42.jsonl",
    "--checkpoint-dir", "outputs/checkpoints/b0_fp8_vit_step006250_20260618/run2_20k_fp8vit_gkd_from_step006250_val512_b8/final",
    "--split", "val",
    "--num-samples", "0",
    "--prompt-mode", "joint",
    "--target-mode", "joint",
    "--image-prompt-style", "camera_labeled",
    "--prompt-text-style", "official_alpamayo",
    "--fuse-history-tokens",
    "--geometry-reference", "gt",
    "--batch-size", "4",
    "--samples-per-row", str(samples_per_row),
    "--temperature", "0.6",
    "--top-p", "0.98",
    "--seed", "42",
    "--max-new-tokens", "256",
    "--qat-quantization", "fp8_pcpt_vit",
    "--qat-calib-samples", "128",
    "--qat-calib-batch-size", "1",
    "--skip-overlays",
    "--output-dir", out_dir,
    "--summary-json", summary,
]
runpy.run_path("scripts/25_decode_checkpoint_overlays.py", run_name="__main__")
PY

  printf '%s done tag=%s summary=%s\n' "$(date -Is)" "${tag}" "${summary}" | tee -a "${LOG}"
}

printf '%s eval_start run_root=%s vram_limit_gb=40\n' "$(date -Is)" "${RUN_ROOT}" | tee -a "${LOG}"
run_decode 1 "backbone_run2_fp8vit_gkd_semantic_val806_official_t06_topp098_n1"
run_decode 6 "backbone_run2_fp8vit_gkd_semantic_val806_official_t06_topp098_n6"
printf '%s eval_done run_root=%s\n' "$(date -Is)" "${RUN_ROOT}" | tee -a "${LOG}"
