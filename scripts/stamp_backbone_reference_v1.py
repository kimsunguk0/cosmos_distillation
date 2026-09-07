#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--source-summary-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--model", default="Alpamayo-1.5-10B")
    parser.add_argument("--prompt-version", default="official_alpamayo/camera_labeled/fuse_history_tokens")
    parser.add_argument("--script", default="scripts/eval_10b_backbone_discrete.py")
    args = parser.parse_args()

    source: dict[str, Any] = json.loads(args.source_summary_json.read_text(encoding="utf-8"))
    manifest = {
        "reference_id": str(args.reference_id),
        "split_id": str(args.split_id),
        "rows_jsonl": str(args.rows_jsonl),
        "source_summary_json": str(args.source_summary_json),
        "num_samples": source.get("num_samples") or source.get("count"),
        "metrics": source.get("metrics"),
        "generation_stamp": {
            "model": source.get("checkpoint_path") or str(args.model),
            "precision": source.get("dtype") or "bfloat16",
            "do_sample": False,
            "temperature": None,
            "top_p": 1.0,
            "top_k": 0,
            "seed": source.get("seed"),
            "prompt_version": str(args.prompt_version),
            "script": str(args.script),
            "script_behavior": "generation_config.do_sample = samples_per_row > 1; samples_per_row=1 is greedy",
            "date_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "commit_hash": git_commit(),
        },
        "legacy_source_fields": {
            "samples_per_row": source.get("samples_per_row"),
            "temperature": source.get("temperature"),
            "top_p": source.get("top_p"),
            "top_k": source.get("top_k"),
            "note": "temperature/top_p are inactive for samples_per_row=1 because do_sample=false",
        },
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "reference_manifest_written", "path": str(args.output_manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
