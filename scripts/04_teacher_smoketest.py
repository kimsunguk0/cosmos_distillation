#!/usr/bin/env python3
"""WP0/WP5 entrypoint: teacher text smoke-test and readiness audit."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPAMAYO15_SRC = PROJECT_ROOT.parent / "alpamayo1.5" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ALPAMAYO15_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO15_SRC))

import pandas as pd

from src.data.teacher_cache import inspect_teacher_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "canonical_samples",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_smoketest.json",
    )
    return parser.parse_args()


def import_status(module_name: str) -> dict[str, str | bool]:
    """Return import status for a module without crashing the smoke test."""
    try:
        importlib.import_module(module_name)
        return {"module": module_name, "ok": True, "error": ""}
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        return {"module": module_name, "ok": False, "error": str(exc)}


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)

    imports = [
        import_status("torch"),
        import_status("transformers"),
        import_status("hydra"),
        import_status("alpamayo1_5"),
        import_status("av"),
    ]

    ready_sample = None
    blocked_sample = None
    for _, row in manifest.iterrows():
        readiness = inspect_teacher_sample(str(row["sample_id"]), args.canonical_root)
        candidate = {
            "sample_id": readiness.sample_id,
            "status": readiness.status,
            "blockers": readiness.blockers,
            "sample_dir": str(readiness.sample_dir) if readiness.sample_dir else None,
        }
        if readiness.status == "ready" and ready_sample is None:
            ready_sample = candidate
        if readiness.status != "ready" and blocked_sample is None:
            blocked_sample = candidate
        if ready_sample and blocked_sample:
            break

    summary = {
        "manifest_path": str(args.manifest_path),
        "canonical_root": str(args.canonical_root),
        "imports": imports,
        "teacher_model_load_attempted": False,
        "first_ready_sample": ready_sample,
        "first_blocked_sample": blocked_sample,
        "notes": [
            "This smoke test only audits dependencies and sample readiness.",
            "Actual Alpamayo generation stays deferred until image frames are materialized.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
