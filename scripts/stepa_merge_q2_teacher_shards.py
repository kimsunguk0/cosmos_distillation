#!/usr/bin/env python3
"""Merge parallel Step A Q2 teacher shard outputs into one teacher directory."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "teacher_q2_t0p60_parallel_merged"


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
    deduped: dict[str, Path] = {}
    for root in roots:
        deduped[str(root)] = root
    return [deduped[key] for key in sorted(deduped)]


def sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    clip_index = row.get("clip_index")
    try:
        clip_index_int = int(clip_index)
    except Exception:  # noqa: BLE001
        clip_index_int = 10**9
    slot = str(row.get("slot", ""))
    slot_rank = {"early": "0", "middle": "1", "late": "2"}.get(slot, slot)
    return (clip_index_int, str(row.get("clip_id", "")), slot_rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, help="Teacher shard dir or glob. Repeatable.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = expand_roots(args.input_root)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    records_by_id: dict[str, dict[str, Any]] = {}
    hard_accept_ids: set[str] = set()
    hard_reject_ids: set[str] = set()
    hard_reject_by_id: dict[str, dict[str, Any]] = {}
    text_judge_by_id: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    duplicate_ids: list[str] = []

    for root in roots:
        counters["input_roots"] += 1
        records_path = root / "teacher_records.jsonl"
        accept_path = root / "q2_hard_gate_accept.jsonl"
        reject_path = root / "q2_hard_gate_reject.jsonl"
        text_path = root / "q2_text_judge_input.jsonl"
        for row in load_jsonl(records_path) or []:
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                counters["record_missing_sample_id"] += 1
                continue
            if sample_id in records_by_id:
                duplicate_ids.append(sample_id)
                counters["duplicate_record_sample_id"] += 1
                continue
            records_by_id[sample_id] = row
            counters["records_seen"] += 1
        for row in load_jsonl(accept_path) or []:
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                hard_accept_ids.add(sample_id)
        for row in load_jsonl(reject_path) or []:
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                hard_reject_ids.add(sample_id)
                hard_reject_by_id.setdefault(sample_id, row)
        for row in load_jsonl(text_path) or []:
            sample_id = str(row.get("sample_id", ""))
            if sample_id and sample_id not in text_judge_by_id:
                text_judge_by_id[sample_id] = row

    records = sorted(records_by_id.values(), key=sort_key)
    accept_rows: list[dict[str, Any]] = []
    reject_rows: list[dict[str, Any]] = []
    reject_row_ids: set[str] = set()
    for row in records:
        sample_id = str(row["sample_id"])
        hard_reject = bool((row.get("teacher") or {}).get("hard_reject"))
        if hard_reject or sample_id in hard_reject_ids:
            reject_rows.append(row)
            reject_row_ids.add(sample_id)
        elif sample_id in hard_accept_ids or sample_id in text_judge_by_id:
            accept_rows.append(row)
        else:
            counters["record_without_gate_file"] += 1
            if hard_reject:
                reject_rows.append(row)
                reject_row_ids.add(sample_id)
            else:
                accept_rows.append(row)

    failed_reject_rows = [
        row
        for sample_id, row in hard_reject_by_id.items()
        if sample_id not in records_by_id and sample_id not in reject_row_ids
    ]
    if failed_reject_rows:
        counters["failed_generation_reject_rows"] += len(failed_reject_rows)
        reject_rows.extend(failed_reject_rows)
        reject_rows = sorted(reject_rows, key=sort_key)

    text_rows = [
        text_judge_by_id[str(row["sample_id"])]
        for row in accept_rows
        if str(row["sample_id"]) in text_judge_by_id
    ]
    write_jsonl(output_root / "teacher_records.jsonl", records)
    write_jsonl(output_root / "q2_hard_gate_accept.jsonl", accept_rows)
    write_jsonl(output_root / "q2_hard_gate_reject.jsonl", reject_rows)
    write_jsonl(output_root / "q2_text_judge_input.jsonl", text_rows)

    summary = {
        "created_at": utc_now(),
        "input_roots": [str(root) for root in roots],
        "output_root": str(output_root),
        "records": len(records),
        "hard_accept": len(accept_rows),
        "hard_reject": len(reject_rows),
        "text_judge_inputs": len(text_rows),
        "hard_accept_rate": len(accept_rows) / len(records) if records else 0.0,
        "duplicate_sample_ids": duplicate_ids[:50],
        "duplicate_sample_id_count": len(duplicate_ids),
        "counters": dict(sorted(counters.items())),
    }
    write_json(output_root / "teacher_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
