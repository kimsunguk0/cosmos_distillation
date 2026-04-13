#!/usr/bin/env python3
"""WP6 entrypoint: optional teacher joint-cache request generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.teacher_cache import load_jsonl_by_key, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--teacher-text-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "canonical_samples",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "joint" / "index.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_joint_cache_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    teacher_text_index = load_jsonl_by_key(args.teacher_text_index)

    records: list[dict] = []
    status_counts: dict[str, int] = {}
    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        teacher_text_record = teacher_text_index.get(sample_id)
        canonical_dir = args.canonical_root / sample_id
        future_path = canonical_dir / "ego_future_xyz.npy"

        blockers: list[str] = []
        if not canonical_dir.exists():
            blockers.append("canonical_sample_missing")
        if not future_path.exists():
            blockers.append("gt_future_path_missing")
        if teacher_text_record is None:
            blockers.append("teacher_text_index_missing")
        elif teacher_text_record.get("status") not in {"ready_request_bundle", "ok"}:
            blockers.append(f"teacher_text_not_ready:{teacher_text_record.get('status')}")

        status = "ready_joint_request_bundle" if not blockers else "blocked"
        record = {
            "sample_id": sample_id,
            "clip_uuid": str(row["clip_uuid"]),
            "split": str(row["subset_split"] if "subset_split" in row else row["split"]),
            "status": status,
            "blockers": blockers,
            "teacher_text_index_path": str(args.teacher_text_index),
            "teacher_text_request_bundle_path": (
                teacher_text_record.get("request_bundle_path") if teacher_text_record else None
            ),
            "canonical_sample_path": str(canonical_dir),
            "gt_future_path": str(future_path),
            "provenance": {
                "source": "joint_cache_request_bundle_only",
                "mixed_task_allowed": False,
                "teacher_is_gt": False,
            },
        }
        records.append(record)
        status_counts[status] = status_counts.get(status, 0) + 1

    write_jsonl(args.output_jsonl, records)
    summary = {
        "manifest_path": str(args.manifest_path),
        "teacher_text_index": str(args.teacher_text_index),
        "output_jsonl": str(args.output_jsonl),
        "status_counts": status_counts,
        "notes": [
            "Joint cache generation is scaffolded only.",
            "These records are for future v2 trajectory-reasoning alignment, not v1 training.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
