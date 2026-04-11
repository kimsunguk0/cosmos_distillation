#!/usr/bin/env python3
"""Build WP4 supervision records from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.supervision import build_supervision_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "strict_download_subset_manifest.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "supervision_records",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "supervision_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_parquet(args.manifest_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    action_counts: dict[str, int] = {}
    missing_human_coc = 0
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for _, row in manifest.iterrows():
            record = build_supervision_record(row)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            meta_action = record["weak_derived"]["meta_action_from_human"]
            if meta_action is None:
                missing_human_coc += 1
            else:
                action = meta_action["value"]
                action_counts[action] = action_counts.get(action, 0) + 1

    summary = {
        "manifest_path": str(args.manifest_path),
        "output_jsonl": str(args.output_dir / "records.jsonl"),
        "num_records": int(len(manifest)),
        "missing_human_coc_records": missing_human_coc,
        "meta_action_counts": action_counts,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
