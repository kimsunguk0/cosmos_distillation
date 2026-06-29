#!/usr/bin/env python3
"""Build a Step A Q2 dataset with supported-only repaired partial rows.

The strict vision gate rejects a row when the original Alpamayo answer contains
unsupported claims. Many rejected-partial rows still contain useful supported
evidence from the vision judge. This script keeps the strict accepted rows and
adds repaired targets built only from ``teacher.vision_judge_supported``.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCEPTED = (
    PROJECT_ROOT
    / "data"
    / "vqa_q2_stepa_pilot50k"
    / "q2_final_judged"
    / "q2_text_vision_judged_all.jsonl"
)
DEFAULT_REJECTED = (
    PROJECT_ROOT
    / "data"
    / "vqa_q2_stepa_pilot50k"
    / "q2_final_judged"
    / "q2_text_vision_judged_rejected.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "q2_repaired_supported_v1"
)

WHITESPACE_RE = re.compile(r"\s+")
COORDINATE_RE = re.compile(
    r"(\[[^\]]*\d[^\]]*\]|\([^\)]*\d[,\s]+[^\)]*\d[^\)]*\)|"
    r"\b\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\b)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def clean_supported_item(text: str) -> str:
    text = str(text or "").strip()
    text = COORDINATE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip(" ;,.")
    text = re.sub(r"^(the answer is grounded in|visible evidence includes)\s+", "", text, flags=re.I)
    text = re.sub(r"\bthe ego vehicle\b", "the current driving scene", text, flags=re.I)
    text = re.sub(r"\bour vehicle\b", "the current driving scene", text, flags=re.I)
    if not text:
        return ""
    return text[0].upper() + text[1:]


def join_supported_items(items: list[str], *, max_items: int, max_words: int) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = clean_supported_item(item)
        key = text.lower()
        if not text or key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
        if len(cleaned) >= max_items:
            break
    if not cleaned:
        return ""

    prefix = "Visible evidence includes "
    body = "; ".join(text.rstrip(".") for text in cleaned)
    answer = f"{prefix}{body}. These visible elements are relevant to the current driving judgment."
    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words]).rstrip(" ,;:.") + "."
    return answer


def sort_key(row: dict[str, Any]) -> tuple[int, str, int, str]:
    try:
        clip_index = int(row.get("clip_index", 10**9))
    except Exception:  # noqa: BLE001
        clip_index = 10**9
    slot_rank = {"early": 0, "middle": 1, "late": 2}.get(str(row.get("slot", "")), 9)
    return (clip_index, str(row.get("clip_id", "")), slot_rank, str(row.get("sample_id", "")))


def mark_strict_row(row: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(row)
    policy = copy.deepcopy(out.get("target_policy") or {})
    policy["label_source"] = "strict_text_vision_accepted"
    policy["hard_target"] = "teacher_answer_short"
    out["target_policy"] = policy
    out["repair"] = {
        "mode": "none",
        "source": "strict_text_vision_accepted",
    }
    return out


def build_repaired_row(row: dict[str, Any], *, max_items: int, max_words: int) -> dict[str, Any] | None:
    teacher = row.get("teacher") or {}
    verdict = str(teacher.get("vision_judge_verdict") or "")
    usable = bool(teacher.get("vision_judge_usable"))
    supported = [str(item) for item in (teacher.get("vision_judge_supported") or [])]
    if verdict != "partial" or usable:
        return None
    if len(supported) < 2:
        return None
    answer = join_supported_items(supported, max_items=max_items, max_words=max_words)
    if len(answer.split()) < 8:
        return None

    out = copy.deepcopy(row)
    raw_answer = str(out.get("answer") or "")
    old_short = str(out.get("teacher_answer_short") or "")
    out["answer_raw_before_repair"] = raw_answer
    out["teacher_answer_short_before_repair"] = old_short
    out["teacher_answer_short"] = answer
    out["answer_repaired"] = answer
    out["answer"] = raw_answer
    out["repair"] = {
        "mode": "vision_supported_only",
        "source": "rejected_partial_false",
        "supported_count": len(supported),
        "used_supported_count": min(len(supported), max_items),
        "unsupported_count": len(teacher.get("vision_judge_unsupported") or []),
        "created_at": utc_now(),
    }
    policy = copy.deepcopy(out.get("target_policy") or {})
    policy["hard_target"] = "teacher_answer_short"
    policy["label_source"] = "vision_supported_repair"
    policy["repair_source"] = "teacher.vision_judge_supported"
    policy["soft_target"] = "alpamayo_teacher_forced_topk32_after_repaired_answer_start"
    out["target_policy"] = policy
    teacher = copy.deepcopy(teacher)
    teacher["selected_by"] = "vision-supported-repair"
    teacher["repair_label_source"] = "vision_judge_supported"
    out["teacher"] = teacher
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-jsonl", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--rejected-jsonl", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-supported-items", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=56)
    parser.add_argument(
        "--repair-splits",
        default="train",
        help="Comma-separated splits where repaired rows are allowed. Default: train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted_rows = read_jsonl(args.accepted_jsonl)
    rejected_rows = read_jsonl(args.rejected_jsonl)
    repair_splits = {item.strip() for item in str(args.repair_splits).split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in accepted_rows:
        rows.append(mark_strict_row(row))
        counters[f"strict_{row.get('split', 'unknown')}"] += 1

    for row in rejected_rows:
        split = str(row.get("split", "unknown"))
        teacher = row.get("teacher") or {}
        counters[f"rejected_{teacher.get('vision_judge_verdict', 'unknown')}"] += 1
        if split not in repair_splits:
            counters[f"skip_repair_split_{split}"] += 1
            continue
        repaired = build_repaired_row(
            row,
            max_items=max(1, int(args.max_supported_items)),
            max_words=max(12, int(args.max_words)),
        )
        if repaired is None:
            counters["skip_repair_not_eligible"] += 1
            continue
        rows.append(repaired)
        counters[f"repaired_{split}"] += 1

    seen: set[str] = set()
    duplicates: list[str] = []
    deduped: list[dict[str, Any]] = []
    for row in sorted(rows, key=sort_key):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in seen:
            duplicates.append(sample_id)
            counters["duplicate_sample_id"] += 1
            continue
        seen.add(sample_id)
        deduped.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    all_path = args.output_root / "q2_supported_repaired_all.jsonl"
    write_jsonl(all_path, deduped)

    split_paths: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split in sorted({str(row.get("split", "unknown")) for row in deduped} | {"train", "val", "test"}):
        split_rows = [row for row in deduped if str(row.get("split", "unknown")) == split]
        path = args.output_root / f"q2_supported_repaired_{split}.jsonl"
        write_jsonl(path, split_rows)
        split_paths[split] = str(path)
        split_counts[split] = len(split_rows)

    repair_rows = [row for row in deduped if (row.get("repair") or {}).get("mode") == "vision_supported_only"]
    summary = {
        "created_at": utc_now(),
        "accepted_jsonl": str(args.accepted_jsonl),
        "rejected_jsonl": str(args.rejected_jsonl),
        "output_root": str(args.output_root),
        "all": str(all_path),
        "split_paths": split_paths,
        "split_counts": split_counts,
        "row_count": len(deduped),
        "repair_row_count": len(repair_rows),
        "strict_row_count": len(deduped) - len(repair_rows),
        "repair_splits": sorted(repair_splits),
        "max_supported_items": int(args.max_supported_items),
        "max_words": int(args.max_words),
        "duplicate_sample_ids": duplicates[:50],
        "duplicate_sample_id_count": len(duplicates),
        "counters": dict(sorted(counters.items())),
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
