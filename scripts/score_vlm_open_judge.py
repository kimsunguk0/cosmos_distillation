#!/usr/bin/env python3
"""Score blind T3/T4 open-judge results for the CR2 capacity-gap eval."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_line_no": line_no, "_parse_error": str(exc), "_raw": line.rstrip("\n")})
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def normalize_winner(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in {"A", "B", "tie", "both_bad"}:
        return text
    return None


def increment(counter: Counter[str], key: str) -> None:
    counter[key] += 1
    counter["total_cases"] += 1


def score(judge_dir: Path) -> dict[str, Any]:
    pairs_path = judge_dir / "pairs.jsonl"
    answer_key_path = judge_dir / "answer_key.json"
    judgments_path = judge_dir / "judgments_blind.jsonl"
    answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    pairs = {row["case_id"]: row for row in read_jsonl(pairs_path) if "case_id" in row}
    judgments = read_jsonl(judgments_path)

    overall: Counter[str] = Counter()
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    case_results: list[dict[str, Any]] = []
    missing_case_ids = sorted(set(pairs) - {row.get("case_id") for row in judgments if isinstance(row, dict)})
    duplicate_case_ids: list[str] = []
    seen: set[str] = set()

    for row in judgments:
        case_id = row.get("case_id")
        task_id = row.get("task_id") or pairs.get(case_id, {}).get("task_id")
        raw_winner = row.get("winner")
        blind_winner = normalize_winner(raw_winner)
        parse_error = row.get("_parse_error")

        if case_id in seen:
            duplicate_case_ids.append(str(case_id))
        if isinstance(case_id, str):
            seen.add(case_id)

        key = answer_key.get(case_id) if isinstance(case_id, str) else None
        winning_model = None
        result = blind_winner or "invalid"
        if parse_error:
            result = "invalid"
        elif blind_winner in {"A", "B"}:
            winning_model = key.get(blind_winner) if isinstance(key, dict) else None
            result = winning_model if winning_model in {"2b", "8b"} else "invalid"
        elif blind_winner in {"tie", "both_bad"}:
            result = blind_winner

        increment(overall, result)
        if isinstance(task_id, str):
            increment(by_task[task_id], result)

        case_results.append(
            {
                "case_id": case_id,
                "task_id": task_id,
                "A": key.get("A") if isinstance(key, dict) else None,
                "B": key.get("B") if isinstance(key, dict) else None,
                "blind_winner": blind_winner,
                "raw_winner": raw_winner,
                "winning_model": winning_model,
                "scored_result": result,
                "parse_error": parse_error,
                "line_no": row.get("_line_no"),
            }
        )

    def serialize_counts(counter: Counter[str]) -> dict[str, Any]:
        total = int(counter.get("total_cases", 0))
        out = {
            "2b": int(counter.get("2b", 0)),
            "8b": int(counter.get("8b", 0)),
            "tie": int(counter.get("tie", 0)),
            "both_bad": int(counter.get("both_bad", 0)),
            "invalid": int(counter.get("invalid", 0)),
            "total_cases": total,
        }
        decisive = out["2b"] + out["8b"]
        out["2b_winrate_decisive"] = (out["2b"] / decisive) if decisive else None
        out["8b_winrate_decisive"] = (out["8b"] / decisive) if decisive else None
        out["2b_winrate_all"] = (out["2b"] / total) if total else None
        out["8b_winrate_all"] = (out["8b"] / total) if total else None
        return out

    return {
        "created_at": utc_now(),
        "judge_dir": str(judge_dir),
        "input_pairs": str(pairs_path),
        "answer_key": str(answer_key_path),
        "judgments": str(judgments_path),
        "overall": serialize_counts(overall),
        "by_task": {task_id: serialize_counts(counter) for task_id, counter in sorted(by_task.items())},
        "case_results": case_results,
        "missing_case_ids": missing_case_ids,
        "duplicate_case_ids": duplicate_case_ids,
    }


def append_report(report_path: Path, summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    lines = [
        "## Blind LLM Judge",
        "",
        f"- Created: `{summary['created_at']}`",
        f"- Judge dir: `{summary['judge_dir']}`",
        f"- Overall: 2B={overall['2b']}, 8B={overall['8b']}, tie={overall['tie']}, both_bad={overall['both_bad']}, invalid={overall['invalid']}, total={overall['total_cases']}",
        f"- Decisive winrate: 2B={overall['2b_winrate_decisive']}, 8B={overall['8b_winrate_decisive']}",
        "",
        "| task | 2B | 8B | tie | both_bad | invalid | total | 2B decisive | 8B decisive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task_id, counts in summary["by_task"].items():
        lines.append(
            f"| {task_id} | {counts['2b']} | {counts['8b']} | {counts['tie']} | {counts['both_bad']} | "
            f"{counts['invalid']} | {counts['total_cases']} | {counts['2b_winrate_decisive']} | {counts['8b_winrate_decisive']} |"
        )
    if summary.get("missing_case_ids"):
        lines.append("")
        lines.append(f"- Missing judgments: `{len(summary['missing_case_ids'])}`")
    if summary.get("duplicate_case_ids"):
        lines.append(f"- Duplicate judgments: `{len(summary['duplicate_case_ids'])}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "\n## Blind LLM Judge\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip()
    else:
        existing = existing.rstrip()
    section = "\n".join(lines)
    text = (existing + "\n\n" + section + "\n") if existing else (section + "\n")
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--append-report", default="")
    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)
    summary = score(judge_dir)
    out_path = Path(args.out) if args.out else judge_dir / "judge_summary.json"
    write_json(out_path, summary)
    if args.append_report:
        append_report(Path(args.append_report), summary)
    print(json.dumps({"judge_summary": str(out_path), "overall": summary["overall"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
