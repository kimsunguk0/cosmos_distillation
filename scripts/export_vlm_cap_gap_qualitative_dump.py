#!/usr/bin/env python3
"""Export side-by-side qualitative examples for the CR2 capacity-gap eval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact_text(value: Any, *, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def answer_json(row: dict[str, Any] | None) -> str:
    if not row:
        return "-"
    answer = row.get("answer")
    if answer is None:
        return "-"
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


def choose_cases(
    manifest: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    *,
    tasks: list[str],
    limit: int,
) -> list[tuple[dict[str, Any], str]]:
    selected: list[tuple[dict[str, Any], str]] = []
    split_counts: dict[str, int] = defaultdict(int)
    for item in sorted(manifest, key=lambda row: int(row.get("eval_index", 0))):
        split = str(item.get("split_tag") or "unknown")
        for task_id in tasks:
            if split_counts[split] >= max(1, limit // 2) and len(selected) < limit:
                continue
            if all((item["clip_id"], task_id, model) in rows_by_key for model in ["2b", "8b", "32b"]):
                selected.append((item, task_id))
                split_counts[split] += 1
                break
        if len(selected) >= limit:
            break
    if len(selected) >= limit:
        return selected[:limit]
    seen = {(row["clip_id"], task_id) for row, task_id in selected}
    for item in sorted(manifest, key=lambda row: int(row.get("eval_index", 0))):
        for task_id in tasks:
            key = (item["clip_id"], task_id)
            if key in seen:
                continue
            if all((item["clip_id"], task_id, model) in rows_by_key for model in ["2b", "8b", "32b"]):
                selected.append((item, task_id))
                seen.add(key)
                break
        if len(selected) >= limit:
            break
    return selected[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="2b,8b,32b")
    parser.add_argument("--tasks", default="T1_pos,T1_neg,T2,T3,T4")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    manifest = read_jsonl(output_dir / "manifest.jsonl")

    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model in models:
        for row in read_jsonl(output_dir / "runs" / model / "predictions.jsonl"):
            if int(row.get("sample_idx", -1)) != 0:
                continue
            rows_by_key[(row["clip_id"], row["task_id"], model)] = row

    cases = choose_cases(manifest, rows_by_key, tasks=tasks, limit=int(args.limit))
    out_path = Path(args.out) if args.out else output_dir / "qualitative_examples.md"
    lines: list[str] = []
    lines.append("# Qualitative Examples")
    lines.append("")
    lines.append("Greedy sample_idx=0 outputs. Answers are side-by-side for sanity checking; use the JSON metrics and blind judge for decisions.")
    lines.append("")
    for index, (manifest_row, task_id) in enumerate(cases, start=1):
        clip_id = manifest_row["clip_id"]
        row0 = rows_by_key.get((clip_id, task_id, models[0]))
        lines.append(f"## {index}. eval_index={manifest_row.get('eval_index')} task={task_id}")
        lines.append("")
        lines.append(f"- clip_id: `{clip_id}`")
        lines.append(f"- split: `{manifest_row.get('split_tag')}`")
        lines.append(f"- event_cluster: `{manifest_row.get('event_cluster')}`")
        lines.append(f"- video: `{row0.get('video_path') if row0 else '-'}`")
        lines.append(f"- question: {compact_text(row0.get('question') if row0 else '', limit=500)}")
        lines.append("")
        lines.append("| model | parse_error | answer | think excerpt |")
        lines.append("|---|---|---|---|")
        for model in models:
            row = rows_by_key.get((clip_id, task_id, model))
            parse_error = row.get("parse_error") if row else "missing"
            lines.append(
                "| "
                + model.upper()
                + " | "
                + compact_text(parse_error or "", limit=120).replace("|", "\\|")
                + " | "
                + compact_text(answer_json(row), limit=700).replace("|", "\\|")
                + " | "
                + compact_text(row.get("think") if row else "", limit=700).replace("|", "\\|")
                + " |"
            )
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"qualitative_examples": str(out_path), "cases": len(cases)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
