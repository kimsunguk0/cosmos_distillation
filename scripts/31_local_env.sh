#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export COSMOS_DATA_ROOT="${COSMOS_DATA_ROOT:-/home/pm97/workspace/dataset/human_coc_dataset}"
export COSMOS_STUDENT_MODEL="${COSMOS_STUDENT_MODEL:-/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b}"
export ALPAMAYO_MODEL_PATH="${ALPAMAYO_MODEL_PATH:-/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B}"
export ALPAMAYO_SRC="${ALPAMAYO_SRC:-/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src}"
export VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COSMOS_DATALOADER_NUM_WORKERS="${COSMOS_DATALOADER_NUM_WORKERS:-4}"
export COSMOS_DATALOADER_PREFETCH_FACTOR="${COSMOS_DATALOADER_PREFETCH_FACTOR:-2}"
export COSMOS_DATALOADER_PIN_MEMORY="${COSMOS_DATALOADER_PIN_MEMORY:-1}"
export COSMOS_DATALOADER_PERSISTENT_WORKERS="${COSMOS_DATALOADER_PERSISTENT_WORKERS:-1}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[missing] shared venv not found at $VENV_PYTHON" >&2
  exit 1
fi

export PATH="$ROOT_DIR/.venv/bin:$PATH"
export PYTHONPATH="$ALPAMAYO_SRC${PYTHONPATH:+:$PYTHONPATH}"

echo "[env] COSMOS_DATA_ROOT=$COSMOS_DATA_ROOT" >&2
echo "[env] COSMOS_STUDENT_MODEL=$COSMOS_STUDENT_MODEL" >&2
echo "[env] ALPAMAYO_MODEL_PATH=$ALPAMAYO_MODEL_PATH" >&2
echo "[env] ALPAMAYO_SRC=$ALPAMAYO_SRC" >&2
echo "[env] python=$VENV_PYTHON" >&2
echo "[env] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" >&2
echo "[env] COSMOS_DATALOADER_NUM_WORKERS=$COSMOS_DATALOADER_NUM_WORKERS" >&2
