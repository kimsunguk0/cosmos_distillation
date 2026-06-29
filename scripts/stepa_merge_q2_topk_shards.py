#!/usr/bin/env python3
"""Merge Step A Q2 top-k shard outputs."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def expand_roots(values: list[str]) -> list[Path]:
    roots: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value)]
        roots.extend(matches or [Path(value)])
    deduped: dict[str, Path] = {str(root): root for root in roots}
    return [deduped[key] for key in sorted(deduped)]


def sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    try:
        clip_index = int(row.get("clip_index", 10**9))
    except Exception:  # noqa: BLE001
        clip_index = 10**9
    slot = str(row.get("slot", ""))
    slot_rank = {"early": "0", "middle": "1", "late": "2"}.get(slot, slot)
    return (clip_index, str(row.get("clip_id", "")), slot_rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, help="Top-k shard dir or glob. Repeatable.")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = expand_roots(args.input_root)
    rows_by_id: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    duplicates: list[str] = []
    for root in roots:
        counters["input_roots"] += 1
        for row in load_jsonl(root / "records_with_topk.jsonl") or []:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                counters["missing_sample_id"] += 1
                continue
            if sample_id in rows_by_id:
                counters["duplicate_sample_id"] += 1
                duplicates.append(sample_id)
                continue
            if not row.get("teacher_topk_ready") or not row.get("teacher_topk_path"):
                counters["skip_not_ready"] += 1
                continue
            rows_by_id[sample_id] = row
            counters["rows_seen"] += 1

    rows = sorted(rows_by_id.values(), key=sort_key)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    all_path = output_root / "records_with_topk.jsonl"
    write_jsonl(all_path, rows)

    split_paths: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split in sorted({str(row.get("split", "unknown")) for row in rows} | {"train", "val", "test"}):
        split_rows = [row for row in rows if str(row.get("split", "unknown")) == split]
        path = output_root / f"records_with_topk_{split}.jsonl"
        write_jsonl(path, split_rows)
        split_paths[split] = str(path)
        split_counts[split] = len(split_rows)

    summary = {
        "created_at": utc_now(),
        "input_roots": [str(root) for root in roots],
        "output_root": str(output_root),
        "records_with_topk": str(all_path),
        "row_count": len(rows),
        "split_counts": split_counts,
        "split_paths": split_paths,
        "duplicate_sample_ids": duplicates[:50],
        "duplicate_sample_id_count": len(duplicates),
        "counters": dict(sorted(counters.items())),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
