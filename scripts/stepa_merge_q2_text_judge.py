#!/usr/bin/env python3
"""Merge Q2 text-judge decisions back into Step A teacher records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_judge_results(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for item in sorted(path.glob("*.jsonl")):
            rows.extend(load_jsonl(item))
        return rows
    return list(load_jsonl(path))


def with_text_judge(row: dict[str, Any], judge: dict[str, Any], *, selected_by: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    teacher = out.setdefault("teacher", {})
    teacher["selected_by"] = selected_by
    teacher["text_judge_decision"] = judge.get("decision")
    teacher["text_judge_reason"] = judge.get("reason", "")
    teacher["text_judge_flags"] = judge.get("flags", [])
    teacher["text_judge_model"] = judge.get("judge_model", "gpt-5.5")
    teacher["text_judge_reasoning_effort"] = judge.get("judge_reasoning_effort", "medium")
    out.setdefault("source", {})
    out["source"]["text_judged_at"] = utc_now()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--teacher-records", type=Path)
    parser.add_argument("--judge-results", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selected-by", default="gpt-5.5-medium-text-llm-judge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_records_path = args.teacher_records or args.teacher_dir / "teacher_records.jsonl"
    judge_results_path = args.judge_results or args.teacher_dir / "text_judge_gpt55_medium" / "judge_results"
    out_dir = args.output_dir or args.teacher_dir / "text_judge_gpt55_medium" / "merged"

    teacher_lookup: dict[str, dict[str, Any]] = {}
    duplicate_teacher_ids: list[str] = []
    for row in load_jsonl(teacher_records_path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in teacher_lookup:
            duplicate_teacher_ids.append(sample_id)
            continue
        teacher_lookup[sample_id] = row

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    issues: list[str] = []
    counters: Counter[str] = Counter()
    judge_rows = load_judge_results(judge_results_path)
    seen_judges: set[str] = set()
    for judge in judge_rows:
        sample_id = str(judge.get("sample_id", ""))
        if sample_id in seen_judges:
            issues.append(f"duplicate judge result: {sample_id}")
            continue
        seen_judges.add(sample_id)
        decision = str(judge.get("decision", ""))
        counters[f"judge_{decision}"] += 1
        for flag in judge.get("flags") or []:
            counters[f"flag_{flag}"] += 1

        row = teacher_lookup.get(sample_id)
        if row is None:
            issues.append(f"missing teacher row: {sample_id}")
            rejected.append({**judge, "reject_source": "missing_teacher_row"})
            continue
        if row.get("teacher", {}).get("hard_reject"):
            counters["reject_hard_reject_teacher"] += 1
            rejected.append({**judge, "reject_source": "hard_reject_teacher"})
            continue
        if decision != "accept":
            rejected.append({**judge, "reject_source": "text_judge_reject"})
            continue

        out = with_text_judge(row, judge, selected_by=str(args.selected_by))
        accepted.append(out)
        counters[f"accepted_{out.get('split', 'unknown')}"] += 1

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in accepted:
        split_rows.setdefault(str(row.get("split", "unknown")), []).append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "q2_text_judged_all.jsonl"
    write_jsonl(all_path, accepted)
    split_paths: dict[str, str] = {}
    for split, rows in sorted(split_rows.items()):
        path = out_dir / f"q2_text_judged_{split}.jsonl"
        write_jsonl(path, rows)
        split_paths[split] = str(path)
    rejected_path = out_dir / "q2_text_judged_rejected.jsonl"
    write_jsonl(rejected_path, rejected)

    summary = {
        "created_at": utc_now(),
        "teacher_records": str(teacher_records_path),
        "judge_results": str(judge_results_path),
        "output_dir": str(out_dir),
        "selected_by": str(args.selected_by),
        "teacher_rows": len(teacher_lookup),
        "judge_rows": len(judge_rows),
        "accepted_total": len(accepted),
        "rejected_total": len(rejected),
        "accepted_by_split": {split: len(rows) for split, rows in sorted(split_rows.items())},
        "duplicate_teacher_ids": duplicate_teacher_ids[:20],
        "duplicate_teacher_id_count": len(duplicate_teacher_ids),
        "issues": issues[:100],
        "issue_count": len(issues),
        "counters": dict(sorted(counters.items())),
        "outputs": {
            "all": str(all_path),
            "rejected": str(rejected_path),
            **split_paths,
        },
    }
    write_json(out_dir / "q2_text_judged_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
