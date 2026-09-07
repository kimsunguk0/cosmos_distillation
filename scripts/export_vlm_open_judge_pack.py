#!/usr/bin/env python3
"""Build blind T3/T4 pairwise judge inputs for the CR2 capacity-gap eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def make_contact_sheet(video_path: Path, out_path: Path, frames: int = 8, width: int = 320) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not decode {video_path}")
    indices = np.linspace(0, max(0, total - 1), frames).round().astype(int)
    imgs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        new_h = int(round(h * (width / float(w))))
        frame = cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)
        cv2.putText(
            frame,
            f"f{int(idx):02d}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        imgs.append(frame)
    cap.release()
    if not imgs:
        raise RuntimeError(f"No frames decoded from {video_path}")
    while len(imgs) < frames:
        imgs.append(imgs[-1].copy())
    row1 = np.concatenate(imgs[: frames // 2], axis=1)
    row2 = np.concatenate(imgs[frames // 2 :], axis=1)
    sheet = np.concatenate([row1, row2], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def compact_answer(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    answer = row.get("answer")
    return {
        "answer": answer,
        "think": row.get("think") or "",
        "parse_error": row.get("parse_error"),
        "raw": row.get("raw") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="2b,8b,32b")
    parser.add_argument("--tasks", default="T3,T4")
    parser.add_argument("--out-name", default="open_judge_codex55_xhigh")
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_rows = read_jsonl(output_dir / "manifest.jsonl")
    manifest_by_clip = {row["clip_id"]: row for row in manifest_rows}
    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    tasks = {item.strip() for item in args.tasks.split(",") if item.strip()}
    judge_dir = output_dir / args.out_name
    if (judge_dir / "judgments_blind.jsonl").exists():
        print(
            json.dumps(
                {
                    "judge_dir": str(judge_dir),
                    "skipped": True,
                    "reason": "judgments_blind.jsonl exists; refusing to overwrite blind judge inputs",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    rows_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for model_key in model_keys:
        for row in read_jsonl(output_dir / "runs" / model_key / "predictions.jsonl"):
            if int(row.get("sample_idx", -1)) != 0:
                continue
            if row.get("task_id") not in tasks:
                continue
            rows_by_key[(row["clip_id"], row["task_id"], 0, model_key)] = row

    image_dir = judge_dir / "contact_sheets"
    candidates: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    for clip_id in sorted(manifest_by_clip):
        manifest = manifest_by_clip[clip_id]
        for task_id in sorted(tasks):
            row_2b = rows_by_key.get((clip_id, task_id, 0, "2b"))
            row_8b = rows_by_key.get((clip_id, task_id, 0, "8b"))
            if not row_2b or not row_8b:
                continue
            candidates.append(
                {
                    "clip_id": clip_id,
                    "task_id": task_id,
                    "manifest": manifest,
                    "row_2b": row_2b,
                    "row_8b": row_8b,
                }
            )

    if int(args.limit_per_task) > 0:
        selected_candidates: list[dict[str, Any]] = []
        for task_id in sorted(tasks):
            task_cases = [case for case in candidates if case["task_id"] == task_id]
            rng.shuffle(task_cases)
            selected_candidates.extend(task_cases[: int(args.limit_per_task)])
        candidates = selected_candidates
    candidates = sorted(candidates, key=lambda item: (item["manifest"]["eval_index"], item["task_id"]))

    pairs: list[dict[str, Any]] = []
    answer_key: dict[str, dict[str, str]] = {}
    for case in candidates:
        clip_id = case["clip_id"]
        task_id = case["task_id"]
        manifest = case["manifest"]
        row_2b = case["row_2b"]
        row_8b = case["row_8b"]
        video_path = Path(row_2b.get("video_path") or row_8b.get("video_path"))
        if not video_path.is_absolute():
            video_path = output_dir.parent.parent.parent / video_path
            if not video_path.exists():
                video_path = Path(row_2b.get("video_path") or row_8b.get("video_path"))
        sheet_path = image_dir / f"{manifest['eval_index']:05d}_{clip_id}_{task_id}.png"
        make_contact_sheet(video_path, sheet_path)

        swap = bool(rng.integers(0, 2))
        a_model, b_model = ("8b", "2b") if swap else ("2b", "8b")
        a_row = row_8b if swap else row_2b
        b_row = row_2b if swap else row_8b
        case_id = f"{manifest['eval_index']:05d}_{task_id}"
        answer_key[case_id] = {"A": a_model, "B": b_model}
        pairs.append(
            {
                "case_id": case_id,
                "clip_id": clip_id,
                "task_id": task_id,
                "question": row_2b.get("question"),
                "contact_sheet": str(sheet_path),
                "answer_a": compact_answer(a_row),
                "answer_b": compact_answer(b_row),
            }
        )

    write_jsonl(judge_dir / "pairs.jsonl", pairs)
    (judge_dir / "answer_key.json").write_text(
        json.dumps(answer_key, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    (judge_dir / "README.md").write_text(
        "\n".join(
            [
                "# Codex 5.5 xhigh T3/T4 Judge Pack",
                "",
                "Judge each `pairs.jsonl` case blindly: choose A, B, tie, or both_bad.",
                "Use the contact sheet as visual evidence, not model identity or metadata labels.",
                "`pairs.jsonl` intentionally excludes 32B answers, event clusters, clear/ambiguous tags, and metadata reasoning hints.",
                "`answer_key.json` is for scoring only; do not use it while judging.",
                "Prefer the answer that is more visibly grounded, better calibrated, and less speculative.",
                "Return JSONL with: case_id, task_id, winner, rationale, failure_modes.",
                "",
                "After `judgments_blind.jsonl` is written, score it with:",
                "`python scripts/score_vlm_open_judge.py --judge-dir <this_dir> --append-report <output_dir>/report.md`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pairs": len(pairs), "judge_dir": str(judge_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
