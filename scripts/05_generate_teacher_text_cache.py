#!/usr/bin/env python3
"""WP5 entrypoint: teacher text cache request generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.teacher_cache import (
    dump_json,
    inspect_teacher_sample,
    load_jsonl_by_key,
    prompt_bundle_for_sample,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--supervision-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "supervision_records" / "records.jsonl",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "canonical_samples",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text",
    )
    parser.add_argument(
        "--question",
        default="Explain the chain of causation for the ego vehicle.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_text_cache_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    supervision_map = load_jsonl_by_key(args.supervision_jsonl)

    requests_dir = args.cache_root / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)

    prompt_names = ["long_cot_v1", "concise_json_v1", "strict_schema_v1"]
    records: list[dict] = []
    status_counts: dict[str, int] = {}

    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        readiness = inspect_teacher_sample(sample_id, args.canonical_root)
        supervision = supervision_map.get(sample_id)

        blockers = list(readiness.blockers)
        status = readiness.status
        if supervision is None:
            blockers.append("supervision_record_missing")
            status = "blocked"

        bundle_path = requests_dir / f"{sample_id}.request.json"
        bundle = prompt_bundle_for_sample(
            sample_id=sample_id,
            canonical_sample_path=args.canonical_root / sample_id,
            prompt_names=prompt_names,
            question=args.question,
        )
        dump_json(bundle_path, bundle)

        record = {
            "sample_id": sample_id,
            "clip_uuid": str(row["clip_uuid"]),
            "split": str(row["subset_split"] if "subset_split" in row else row["split"]),
            "status": "ready_request_bundle" if status == "ready" else status,
            "blockers": blockers,
            "request_bundle_path": str(bundle_path),
            "canonical_sample_path": str(args.canonical_root / sample_id),
            "output": {
                "teacher_long_cot": None,
                "teacher_short_reason": None,
                "teacher_meta_action": None,
                "teacher_answer": None,
                "teacher_structured_json_path": None,
                "teacher_logit_cache_path": None,
            },
            "provenance": {
                "soft": "teacher_text",
                "source": "request_bundle_only",
                "teacher_is_gt": False,
            },
        }
        records.append(record)
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    index_path = args.cache_root / "index.jsonl"
    write_jsonl(index_path, records)

    summary = {
        "manifest_path": str(args.manifest_path),
        "supervision_jsonl": str(args.supervision_jsonl),
        "canonical_root": str(args.canonical_root),
        "cache_root": str(args.cache_root),
        "index_path": str(index_path),
        "request_bundle_count": len(records),
        "status_counts": status_counts,
        "ready_for_teacher_generation": status_counts.get("ready_request_bundle", 0),
        "notes": [
            "This step scaffolds teacher requests without generating teacher outputs.",
            "Alpamayo inference is deferred until image frames exist for the canonical sample.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
