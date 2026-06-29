#!/usr/bin/env python3
"""Run Codex gpt-5.5 medium as the Q2-only text judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGE_DIR = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "teacher_q2_t0p60" / "text_judge_gpt55_medium"
ALLOWED_FLAGS = {
    "coordinate_leakage",
    "bbox_or_object_id",
    "velocity_value",
    "generic",
    "unsupported_future",
    "action_overreach",
    "incoherent",
    "too_short",
    "no_concrete_visible_element",
    "ok",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def codex_bin() -> str:
    preferred = Path("/home/pm97/.npm-global/bin/codex")
    return str(preferred) if preferred.exists() else "codex"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def compact_for_prompt(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, row in enumerate(rows):
        flags = row.get("text_flags", {})
        item = {
            "i": i,
            "sample_id": row["sample_id"],
            "family": "Q2",
            "candidate_id": row["candidate_id"],
            "answer": row["answer"],
            "flags": {
                "coord": bool(flags.get("has_coordinate")),
                "vel": bool(flags.get("has_velocity_word")),
                "future": bool(flags.get("has_future_language")),
                "action": bool(flags.get("has_action_language")),
            },
        }
        lines.append(json.dumps(item, ensure_ascii=True, sort_keys=True))
    return "\n".join(lines)


def build_prompt(rows: list[dict[str, Any]], *, model: str, effort: str) -> str:
    return f"""You are a text-only LLM judge for autonomous-driving VQA distillation labels.

Do not use tools, shell commands, markdown, or code fences.
Return only JSONL, exactly one JSON object per input row, in the same order.
This is text-only validation, not image verification.

Output object keys, in this exact order:
sample_id, family, decision, selected_candidate_id, reason, flags, judge_model, judge_reasoning_effort

Rules:
- decision must be "accept" or "reject_all".
- If accept, selected_candidate_id must equal the input candidate_id.
- If reject_all, selected_candidate_id must be null.
- judge_model must be "{model}".
- judge_reasoning_effort must be "{effort}".
- Keep reason under 20 words.
- flags must be a short list chosen from:
  coordinate_leakage, bbox_or_object_id, velocity_value, generic, unsupported_future,
  action_overreach, incoherent, too_short, no_concrete_visible_element, ok
- Use [] for clean accepted rows.

Q2 accept:
- names concrete visible traffic elements, road users, road geometry, traffic controls, obstacles, construction, blocked areas, or visibility conditions
- gives grounded driving-behavior relevance or decision-support implication
- mild caution/action wording is allowed when anchored to concrete visible elements
- short but concrete answers are acceptable

Q2 reject:
- coordinates, bounding boxes, object IDs, raw numeric locations, or velocity values
- mostly generic advice with no concrete visible element
- unsupported hidden-object or future speculation
- incoherent, empty, too short, or action-only label without evidence

INPUT_JSONL:
{compact_for_prompt(rows)}
"""


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|jsonl)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def parse_and_validate(raw_text: str, source_rows: list[dict[str, Any]], *, model: str, effort: str) -> list[dict[str, Any]]:
    text = strip_code_fence(raw_text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    expected_keys = [
        "sample_id",
        "family",
        "decision",
        "selected_candidate_id",
        "reason",
        "flags",
        "judge_model",
        "judge_reasoning_effort",
    ]
    if len(lines) != len(source_rows):
        raise ValueError(f"Expected {len(source_rows)} output lines, got {len(lines)}")
    rows: list[dict[str, Any]] = []
    for i, (line, source) in enumerate(zip(lines, source_rows, strict=True), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at output line {i}: {exc}: {line[:200]}") from exc
        if list(row.keys()) != expected_keys:
            raise ValueError(f"Invalid key order at line {i}: {list(row.keys())}")
        if row["sample_id"] != source["sample_id"]:
            raise ValueError(f"sample_id mismatch at line {i}: {row['sample_id']} != {source['sample_id']}")
        if row["family"] != "Q2":
            raise ValueError(f"invalid family at line {i}: {row['family']}")
        if row["decision"] not in {"accept", "reject_all"}:
            raise ValueError(f"invalid decision at line {i}: {row['decision']}")
        if row["decision"] == "accept" and row["selected_candidate_id"] != source["candidate_id"]:
            raise ValueError(f"selected_candidate_id mismatch at line {i}")
        if row["decision"] == "reject_all" and row["selected_candidate_id"] is not None:
            raise ValueError(f"reject_all selected_candidate_id must be null at line {i}")
        if row["judge_model"] != model:
            raise ValueError(f"invalid judge_model at line {i}: {row['judge_model']}")
        if row["judge_reasoning_effort"] != effort:
            raise ValueError(f"invalid judge_reasoning_effort at line {i}: {row['judge_reasoning_effort']}")
        if not isinstance(row["flags"], list):
            raise ValueError(f"flags must be list at line {i}")
        unknown_flags = set(row["flags"]) - ALLOWED_FLAGS
        if unknown_flags:
            raise ValueError(f"unknown flags at line {i}: {sorted(unknown_flags)}")
        rows.append(row)
    return rows


def run_codex(
    *,
    prompt: str,
    raw_output_path: Path,
    log_path: Path,
    timeout_s: int,
    model: str,
    effort: str,
) -> None:
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        codex_bin(),
        "exec",
        "-m",
        model,
        "-c",
        f"model_reasoning_effort='{effort}'",
        "-c",
        "approval_policy='never'",
        "-s",
        "read-only",
        "-C",
        str(PROJECT_ROOT),
        "--ephemeral",
        "--output-last-message",
        str(raw_output_path),
        "-",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"codex exec failed with code {proc.returncode}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.judge_dir / "judge_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = [
        shard
        for shard in payload["shards"]
        if int(shard["shard_index"]) >= int(args.start_shard)
    ]
    if args.max_shards is not None:
        shards = shards[: int(args.max_shards)]

    summary_path = args.judge_dir / "codex_judge_run_summary.jsonl"
    for shard in shards:
        shard_index = int(shard["shard_index"])
        input_path = Path(shard["input_path"])
        output_path = Path(shard["output_path"])
        raw_output_path = args.judge_dir / "raw_outputs" / f"raw_{shard_index:05d}.txt"
        log_path = args.judge_dir / "logs" / f"codex_judge_{shard_index:05d}.log"
        if output_path.exists() and not args.force:
            print(f"[{shard_index:05d}] skip existing {output_path}", flush=True)
            continue

        rows = load_jsonl(input_path)
        prompt = build_prompt(rows, model=str(args.model), effort=str(args.effort))
        last_error: Exception | None = None
        started_at = utc_now()
        for attempt in range(int(args.retries) + 1):
            try:
                print(f"[{shard_index:05d}] run rows={len(rows)} attempt={attempt + 1}", flush=True)
                run_codex(
                    prompt=prompt,
                    raw_output_path=raw_output_path,
                    log_path=log_path,
                    timeout_s=int(args.timeout_s),
                    model=str(args.model),
                    effort=str(args.effort),
                )
                judged = parse_and_validate(
                    raw_output_path.read_text(encoding="utf-8"),
                    rows,
                    model=str(args.model),
                    effort=str(args.effort),
                )
                write_jsonl(output_path, judged)
                accepted = sum(1 for row in judged if row["decision"] == "accept")
                record = {
                    "time": utc_now(),
                    "shard_index": shard_index,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "raw_output_path": str(raw_output_path),
                    "log_path": str(log_path),
                    "rows": len(judged),
                    "accepted": accepted,
                    "rejected": len(judged) - accepted,
                    "started_at": started_at,
                    "status": "ok",
                }
                with summary_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
                print(f"[{shard_index:05d}] ok rows={len(judged)} accept={accepted} reject={len(judged) - accepted}", flush=True)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[{shard_index:05d}] failed attempt={attempt + 1}: {exc}", flush=True)
                if attempt < int(args.retries):
                    time.sleep(2)
        if last_error is not None:
            record = {
                "time": utc_now(),
                "shard_index": shard_index,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "raw_output_path": str(raw_output_path),
                "log_path": str(log_path),
                "rows": len(rows),
                "started_at": started_at,
                "status": "failed",
                "error": repr(last_error),
            }
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            raise SystemExit(f"Shard {shard_index} failed: {last_error}")


if __name__ == "__main__":
    main()
