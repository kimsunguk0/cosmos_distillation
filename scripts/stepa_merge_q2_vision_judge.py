#!/usr/bin/env python3
"""Merge Q2 vision-judge decisions into text-judged Step A records."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_JUDGED = (
    PROJECT_ROOT
    / "data"
    / "vqa_q2_stepa_pilot50k"
    / "teacher_q2_t0p60_parallel_merged"
    / "text_judge_gpt55_medium"
    / "merged"
    / "q2_text_judged_all.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "q2_final_judged"


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


def expand_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        p = Path(value)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
            continue
        matches = [Path(item) for item in glob.glob(value)]
        paths.extend(matches or [p])
    deduped: dict[str, Path] = {str(path): path for path in paths}
    return [deduped[key] for key in sorted(deduped)]


def load_vision_results(values: list[str]) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    lookup: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    for path in expand_paths(values):
        if not path.exists():
            counters["missing_vision_path"] += 1
            continue
        for row in load_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                counters["vision_missing_sample_id"] += 1
                continue
            if sample_id in lookup:
                counters["duplicate_vision_sample_id"] += 1
                continue
            lookup[sample_id] = row
            counters["vision_rows"] += 1
            counters[f"vision_verdict_{row.get('visual_verdict', 'unknown')}"] += 1
            counters[f"vision_usable_{bool(row.get('usable'))}"] += 1
    return lookup, counters


def attach_vision(row: dict[str, Any], vision: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    teacher = out.setdefault("teacher", {})
    teacher["vision_judge_verdict"] = vision.get("visual_verdict")
    teacher["vision_judge_usable"] = bool(vision.get("usable"))
    teacher["vision_judge_supported"] = vision.get("supported", [])
    teacher["vision_judge_unsupported"] = vision.get("unsupported", [])
    teacher["vision_judge_notes"] = vision.get("notes", "")
    teacher["vision_judge_model"] = vision.get("vision_judge_model")
    teacher["vision_judge_effort"] = vision.get("vision_judge_effort")
    out.setdefault("source", {})
    out["source"]["vision_judged_at"] = utc_now()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-judged-jsonl", type=Path, default=DEFAULT_TEXT_JUDGED)
    parser.add_argument("--vision-results", action="append", required=True, help="Vision result JSONL, dir, or glob. Repeatable.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--require-usable", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vision_lookup, counters = load_vision_results(args.vision_results)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for row in load_jsonl(args.text_judged_jsonl):
        sample_id = str(row.get("sample_id", ""))
        counters["text_judged_seen"] += 1
        vision = vision_lookup.get(sample_id)
        if vision is None:
            counters["missing_vision_for_text_row"] += 1
            missing.append({"sample_id": sample_id, "reject_source": "missing_vision_result"})
            continue
        out = attach_vision(row, vision)
        if args.require_usable and not bool(vision.get("usable")):
            counters["reject_not_vision_usable"] += 1
            rejected.append(out)
            continue
        accepted.append(out)
        counters[f"accepted_{out.get('split', 'unknown')}"] += 1

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in accepted:
        split_rows.setdefault(str(row.get("split", "unknown")), []).append(row)

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    all_path = out_root / "q2_text_vision_judged_all.jsonl"
    rejected_path = out_root / "q2_text_vision_judged_rejected.jsonl"
    missing_path = out_root / "q2_text_vision_judged_missing_vision.jsonl"
    write_jsonl(all_path, accepted)
    write_jsonl(rejected_path, rejected)
    write_jsonl(missing_path, missing)
    split_paths: dict[str, str] = {}
    for split, rows in sorted(split_rows.items()):
        path = out_root / f"q2_text_vision_judged_{split}.jsonl"
        write_jsonl(path, rows)
        split_paths[split] = str(path)

    summary = {
        "created_at": utc_now(),
        "text_judged_jsonl": str(args.text_judged_jsonl),
        "vision_results": args.vision_results,
        "output_root": str(out_root),
        "require_usable": bool(args.require_usable),
        "accepted_total": len(accepted),
        "rejected_total": len(rejected),
        "missing_vision_total": len(missing),
        "accepted_by_split": {split: len(rows) for split, rows in sorted(split_rows.items())},
        "counters": dict(sorted(counters.items())),
        "outputs": {
            "all": str(all_path),
            "rejected": str(rejected_path),
            "missing_vision": str(missing_path),
            **split_paths,
        },
    }
    write_json(out_root / "q2_text_vision_judged_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
