#!/usr/bin/env python3
"""Validate blind T3/T4 judge JSONL before unblinding with answer_key.json."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_WINNERS = {"A", "B", "tie", "both_bad"}
ALLOWED_JUDGMENT_KEYS = {"case_id", "task_id", "winner", "rationale", "failure_modes"}
ALLOWED_TASKS = {"T3", "T4"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                row = {"_parse_error": str(exc), "_raw": line.rstrip("\n")}
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def normalize_winner(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in {"A", "B", "tie", "both_bad"}:
        return text
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    pairs_path = args.judge_dir / "pairs.jsonl"
    judgments_path = args.judge_dir / "judgments_blind.jsonl"
    pairs = read_jsonl(pairs_path)
    judgments = read_jsonl(judgments_path)
    pair_by_id = {row.get("case_id"): row for row in pairs if isinstance(row.get("case_id"), str)}

    errors: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for row in judgments:
        line_no = row.get("_line_no")
        case_id = row.get("case_id")
        if row.get("_parse_error"):
            errors.append({"line_no": line_no, "error": "json_parse_error", "detail": row.get("_parse_error")})
            continue
        if not isinstance(case_id, str):
            errors.append({"line_no": line_no, "error": "missing_case_id"})
            continue
        extra_keys = sorted(set(row) - ALLOWED_JUDGMENT_KEYS - {"_line_no"})
        if extra_keys:
            errors.append({"line_no": line_no, "case_id": case_id, "error": "extra_keys", "keys": extra_keys})
        seen[case_id] += 1
        pair = pair_by_id.get(case_id)
        if pair is None:
            errors.append({"line_no": line_no, "case_id": case_id, "error": "unknown_case_id"})
            continue
        task_id = row.get("task_id")
        if task_id not in ALLOWED_TASKS:
            errors.append({"line_no": line_no, "case_id": case_id, "error": "invalid_task_id", "task_id": task_id})
        if task_id != pair.get("task_id"):
            errors.append(
                {
                    "line_no": line_no,
                    "case_id": case_id,
                    "error": "task_id_mismatch",
                    "judgment_task": row.get("task_id"),
                    "pair_task": pair.get("task_id"),
                }
            )
        winner = normalize_winner(row.get("winner"))
        if winner not in ALLOWED_WINNERS:
            errors.append(
                {
                    "line_no": line_no,
                    "case_id": case_id,
                    "error": "invalid_winner",
                    "winner": row.get("winner"),
                }
            )
        if not isinstance(row.get("rationale"), str) or not row.get("rationale", "").strip():
            errors.append({"line_no": line_no, "case_id": case_id, "error": "missing_rationale"})
        if "failure_modes" in row and not isinstance(row.get("failure_modes"), list):
            errors.append({"line_no": line_no, "case_id": case_id, "error": "failure_modes_not_list"})

    missing = sorted(set(pair_by_id) - set(seen))
    duplicates = sorted(case_id for case_id, count in seen.items() if count > 1)
    if missing:
        errors.append({"error": "missing_case_ids", "count": len(missing), "case_ids": missing[:50]})
    if duplicates:
        errors.append({"error": "duplicate_case_ids", "count": len(duplicates), "case_ids": duplicates[:50]})

    payload = {
        "ok": not errors,
        "judge_dir": str(args.judge_dir),
        "pairs": len(pair_by_id),
        "judgments": len(judgments),
        "errors": errors,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
