#!/usr/bin/env python3
"""Build Q2-only text-judge shards from Step A teacher output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "teacher_q2_t0p60"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_jsonl(path: Path):
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


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    teacher = row.get("teacher") or {}
    text_flags = row.get("text_flags") or {}
    return {
        "sample_id": str(row["sample_id"]),
        "family": "Q2",
        "candidate_id": str(row.get("candidate_id") or teacher.get("temperature_label") or "t0p60"),
        "answer": str(row.get("answer") or ""),
        "text_flags": {
            "hard_reject": bool(text_flags.get("hard_reject") or teacher.get("hard_reject")),
            "quality_flags": text_flags.get("quality_flags") or teacher.get("quality_flags", []),
            "has_coordinate": bool(text_flags.get("has_coordinate") or teacher.get("has_coordinate")),
            "has_velocity_word": bool(text_flags.get("has_velocity_word") or teacher.get("has_velocity_word")),
            "has_future_language": bool(text_flags.get("has_future_language") or teacher.get("has_future_language")),
            "has_action_language": bool(text_flags.get("has_action_language") or teacher.get("has_action_language")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--judge-input-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shard-size", type=int, default=250)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.judge_input_jsonl or args.teacher_dir / "q2_text_judge_input.jsonl"
    out_dir = args.output_dir or args.teacher_dir / "text_judge_gpt55_medium"
    shard_dir = out_dir / "shards"
    result_dir = out_dir / "judge_results"

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in load_jsonl(input_path):
        item = normalize_row(row)
        sample_id = item["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        rows.append(item)
        if args.limit is not None and len(rows) >= int(args.limit):
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    compact_path = out_dir / "q2_judge_input_compact.jsonl"
    write_jsonl(compact_path, rows)

    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(rows), max(1, int(args.shard_size)))):
        shard_rows = rows[start : start + int(args.shard_size)]
        shard_path = shard_dir / f"judge_shard_{shard_index:05d}.jsonl"
        output_path = result_dir / f"judge_results_{shard_index:05d}.jsonl"
        write_jsonl(shard_path, shard_rows)
        shards.append(
            {
                "shard_index": int(shard_index),
                "input_path": str(shard_path),
                "output_path": str(output_path),
                "count": len(shard_rows),
                "q2_count": len(shard_rows),
            }
        )

    summary = {
        "created_at": utc_now(),
        "teacher_dir": str(args.teacher_dir),
        "input_jsonl": str(input_path),
        "output_dir": str(out_dir),
        "compact_input": str(compact_path),
        "manifest": str(out_dir / "judge_manifest.json"),
        "shard_size": int(args.shard_size),
        "total_count": len(rows),
        "shard_count": len(shards),
    }
    (out_dir / "judge_manifest.json").write_text(
        json.dumps({"summary": summary, "shards": shards}, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
