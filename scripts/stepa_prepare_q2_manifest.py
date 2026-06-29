#!/usr/bin/env python3
"""Prepare Step A Q2-only VQA manifests from judged Alpamayo teacher rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vqa.q2_stepa import Q2_OFFICIAL, read_jsonl, shorten_q2_answer, write_jsonl


DEFAULT_INPUT = Path(
    "/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/output/"
    "vqa_4cam1_distill_10k_20260619/llm_judge_gpt55_medium/"
    "train_q2_4cam1_gpt55_medium_judged.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vqa_q2_stepa"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_bucket(value: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(modulo)


def build_record(row: dict[str, Any], *, max_words: int) -> dict[str, Any]:
    short, flags = shorten_q2_answer(str(row.get("answer") or ""), max_words=max_words)
    teacher = dict(row.get("teacher") or {})
    return {
        "sample_id": str(row["sample_id"]),
        "base_sample_id": str(row.get("base_sample_id") or row["sample_id"]),
        "task": "stepa_q2_vqa_distill",
        "family": "Q2",
        "stage": "1B-pre",
        "image_profile": "4cam_x1",
        "question": str(row.get("question") or Q2_OFFICIAL),
        "teacher_answer_short": short,
        "teacher_answer_raw": str(row.get("answer") or ""),
        "teacher_full_trace": {
            "raw_answer": str(row.get("answer") or ""),
            "judge_reason": teacher.get("judge_reason"),
            "judge_flags": teacher.get("judge_flags", []),
            "quality_flags": teacher.get("quality_flags", []),
        },
        "target_policy": {
            "hard_target": "teacher_answer_short",
            "soft_target": "alpamayo_teacher_forced_topk32_after_answer_start",
            "shorten_flags": flags,
        },
        "teacher": teacher,
        "dataset_root": row["dataset_root"],
        "clip_id": row["clip_id"],
        "clip_index": int(row["clip_index"]),
        "chunk": int(row["chunk"]),
        "slot": row["slot"],
        "t0_us": int(row["t0_us"]),
        "camera_aliases": row.get("camera_aliases") or ["cross_left", "front_wide", "cross_right", "front_tele"],
        "camera_indices": [int(v) for v in row.get("camera_indices") or [0, 1, 2, 6]],
        "frames_per_camera": int(row.get("frames_per_camera") or 1),
        "frame_offsets_us": [int(v) for v in row.get("frame_offsets_us") or [0]],
        "frame_plan": row["frame_plan"],
        "source": {
            "judged_q2_jsonl": str(DEFAULT_INPUT),
            "source_sample_id": str(row["sample_id"]),
            "created_at": utc_now(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.03)
    parser.add_argument("--max-target-words", type=int, default=56)
    parser.add_argument("--require-judge-accept", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    val_cutoff = int(round(float(args.val_fraction) * 10_000))
    for row in rows:
        counts["seen"] += 1
        if str(row.get("family")) != "Q2":
            counts["skip_non_q2"] += 1
            continue
        teacher = row.get("teacher") or {}
        if args.require_judge_accept and str(teacher.get("judge_decision", "accept")).lower() != "accept":
            counts["skip_not_judge_accept"] += 1
            continue
        record = build_record(row, max_words=int(args.max_target_words))
        if stable_bucket(record["base_sample_id"]) < val_cutoff:
            record["split"] = "val"
            val.append(record)
        else:
            record["split"] = "train"
            train.append(record)
        counts["accepted"] += 1
        if (record["target_policy"]["shorten_flags"] or {}).get("had_action_or_future_language"):
            counts["shortener_saw_action_or_future"] += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_path = args.output_root / "train_q2_stepa.jsonl"
    val_path = args.output_root / "val_q2_stepa.jsonl"
    all_path = args.output_root / "all_q2_stepa.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(all_path, train + val)
    summary = {
        "created_at": utc_now(),
        "input_jsonl": str(args.input_jsonl),
        "output_root": str(args.output_root),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "all_jsonl": str(all_path),
        "val_fraction": float(args.val_fraction),
        "max_target_words": int(args.max_target_words),
        "counts": dict(counts),
    }
    summary_path = args.output_root / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

