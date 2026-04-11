#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
KNOWN_CACHE_ROOT="/home/pm97/workspace/kjhong/alpamayo_100/.physical_ai_av_cache/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/2ae73f49ffd2b5db43b404201beb7b92889f7afc"

echo "[env] project_root: $PROJECT_ROOT"
echo "[env] python3: $(command -v python3 || true)"
echo "[env] pip3: $(command -v pip3 || true)"
echo "[env] huggingface-cli: $(command -v huggingface-cli || true)"
echo "[env] hf_token_cache: $HOME/.cache/huggingface/token"
if [[ -f "$HOME/.cache/huggingface/token" ]]; then
  echo "[ok] huggingface token present"
else
  echo "[missing] huggingface token"
fi

if [[ -x "$VENV_PYTHON" ]]; then
  echo "[ok] venv python: $VENV_PYTHON"
  PYTHON_BIN="$VENV_PYTHON"
else
  echo "[warn] venv python missing, falling back to system python3"
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -d "$KNOWN_CACHE_ROOT" ]]; then
  echo "[ok] known PhysicalAI AV cache root exists"
else
  echo "[warn] known PhysicalAI AV cache root missing"
fi

"$PYTHON_BIN" - <<'PY'
import importlib
from pathlib import Path

packages = [
    "yaml",
    "pandas",
    "pyarrow",
    "numpy",
    "huggingface_hub",
    "pytest",
]

for name in packages:
    try:
        importlib.import_module(name)
        print(f"[ok] {name}")
    except Exception as exc:
        print(f"[missing] {name}: {exc}")

known_cache_root = Path(
    "/home/pm97/workspace/kjhong/alpamayo_100/.physical_ai_av_cache/"
    "datasets--nvidia--PhysicalAI-Autonomous-Vehicles/snapshots/"
    "2ae73f49ffd2b5db43b404201beb7b92889f7afc"
)
for rel in [
    "features.csv",
    "clip_index.parquet",
    "reasoning/ood_reasoning.parquet",
    "metadata/feature_presence.parquet",
    "metadata/sensor_presence.parquet",
    "metadata/data_collection.parquet",
]:
    path = known_cache_root / rel
    print(f"[cache] {rel}: {'present' if path.exists() else 'missing'}")
PY
