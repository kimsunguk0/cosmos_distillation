#!/usr/bin/env python3
"""Run a Codex vision judge over rendered Q2-only Step A contact sheets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs" / "stepa_q2_vision_audit"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_EFFORT = "medium"

PROMPT_HEADER = """You are judging Q2-only visual grounding for PhysicalAI AV VQA distillation.

Each attached image is a 2x2 contact sheet from the same driving sample:
- front-left, front-wide, front-right, and front-telephoto cameras
- one frame per camera
- camera labels are printed at the top of each cell

Judge whether the candidate Q2 answer is supported by visible evidence in the contact sheet.

Strict rejection rules:
- Mark bad if the answer mentions objects, signs, lights, pedestrians, vehicles, lane states, weather, or scene details that are not visible.
- Mark bad if it uses coordinates, bounding boxes, object ids, exact distances, or exact velocities.
- Mark bad if it asserts unsupported future motion, hidden hazards, collision predictions, intent, or temporal change not visible from the single frame.
- Mark bad if it is only generic action advice without grounded visible traffic elements.
- Mark partial if some visible elements are supported but important claims are unsupported or over-specific.
- Mark ok only when the answer's traffic elements and driving-behavior relevance are grounded in the visible scene.
- Grounded driving-behavior relevance is allowed when it follows from visible evidence, for example a visible stop sign, pedestrian, red light, parked cars, lane blockage, or nearby traffic.

Return JSON only. For each sample, emit exactly:
{
  "sample_id": "...",
  "qid": "...",
  "visual_verdict": "ok|partial|bad|missing",
  "supported": ["..."],
  "unsupported": ["..."],
  "usable": true|false,
  "notes": "...",
  "vision_judge_model": "...",
  "vision_judge_effort": "..."
}

If multiple samples are provided, return a JSON array in the same order. Do not include markdown.
"""


def codex_bin() -> str:
    preferred = Path("/home/pm97/.npm-global/bin/codex")
    return str(preferred) if preferred.exists() else "codex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        qid = row.get("qid")
        if sample_id and qid:
            keys.add((str(sample_id), str(qid)))
    return keys


def resolve_image_path(audit_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return audit_dir / path


def compact_sample(row: dict[str, Any], image_number: int) -> dict[str, Any]:
    candidate = (row.get("candidates") or [{}])[0]
    return {
        "image_number": image_number,
        "sample_id": row.get("sample_id"),
        "qid": row.get("qid"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "teacher_answer_short": row.get("teacher_answer_short"),
        "split": row.get("split"),
        "clip_id": row.get("clip_id"),
        "slot": row.get("slot"),
        "image_profile": row.get("image_profile"),
        "candidate_flags": {
            "hard_reject": candidate.get("hard_reject"),
            "has_coordinate": candidate.get("has_coordinate"),
            "has_future_language": candidate.get("has_future_language"),
            "has_action_language": candidate.get("has_action_language"),
            "quality_flags": candidate.get("quality_flags"),
        },
    }


def build_prompt(rows: list[dict[str, Any]], *, model: str, effort: str) -> str:
    samples = [compact_sample(row, image_number=index + 1) for index, row in enumerate(rows)]
    payload = {
        "vision_judge_model": model,
        "vision_judge_effort": effort,
        "samples": samples,
    }
    return f"{PROMPT_HEADER}\nSamples to judge:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n"


def parse_json_response(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty Codex response")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start_candidates = [pos for pos in (stripped.find("["), stripped.find("{")) if pos >= 0]
    if not start_candidates:
        raise ValueError("could not find JSON payload in Codex response")
    start = min(start_candidates)
    end = max(stripped.rfind("]"), stripped.rfind("}"))
    if end < start:
        raise ValueError("could not find JSON payload in Codex response")
    return json.loads(stripped[start : end + 1])


def normalize_result(result: dict[str, Any], source: dict[str, Any], *, model: str, effort: str) -> dict[str, Any]:
    verdict = str(result.get("visual_verdict") or "missing").lower()
    if verdict not in {"ok", "partial", "bad", "missing"}:
        verdict = "missing"
    supported = result.get("supported")
    unsupported = result.get("unsupported")
    if not isinstance(supported, list):
        supported = [] if supported is None else [str(supported)]
    if not isinstance(unsupported, list):
        unsupported = [] if unsupported is None else [str(unsupported)]
    usable = result.get("usable")
    if usable is None:
        usable = verdict == "ok"
    return {
        "sample_id": str(result.get("sample_id") or source.get("sample_id")),
        "qid": str(result.get("qid") or source.get("qid")),
        "visual_verdict": verdict,
        "supported": [str(v) for v in supported],
        "unsupported": [str(v) for v in unsupported],
        "usable": bool(usable),
        "notes": str(result.get("notes") or ""),
        "vision_judge_model": str(result.get("vision_judge_model") or model),
        "vision_judge_effort": str(result.get("vision_judge_effort") or effort),
    }


def failure_result(source: dict[str, Any], *, model: str, effort: str, error: str) -> dict[str, Any]:
    return {
        "sample_id": str(source.get("sample_id")),
        "qid": str(source.get("qid")),
        "visual_verdict": "missing",
        "supported": [],
        "unsupported": [],
        "usable": False,
        "notes": f"vision judge failed: {error}",
        "vision_judge_model": model,
        "vision_judge_effort": effort,
    }


def run_codex_batch(
    rows: list[dict[str, Any]],
    *,
    audit_dir: Path,
    model: str,
    effort: str,
    timeout_s: int,
) -> list[dict[str, Any]]:
    prompt = build_prompt(rows, model=model, effort=effort)
    image_paths = [resolve_image_path(audit_dir, str(row["image_path"])) for row in rows]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(image_path)

    with tempfile.TemporaryDirectory(prefix="q2_vision_judge_") as tmpdir:
        output_path = Path(tmpdir) / "last_message.txt"
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
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
        ]
        for image_path in image_paths:
            cmd.extend(["--image", str(image_path)])
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=int(timeout_s),
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        response_text = output_path.read_text(encoding="utf-8") if output_path.exists() else proc.stdout
        if proc.returncode != 0:
            raise RuntimeError(f"codex exit {proc.returncode}: {proc.stderr.strip()[-1000:]}")
        parsed = parse_json_response(response_text)
        parsed_rows = parsed if isinstance(parsed, list) else [parsed]
        if len(parsed_rows) != len(rows):
            raise ValueError(f"expected {len(rows)} results, got {len(parsed_rows)}")
        return [
            normalize_result(result, source, model=model, effort=effort)
            for result, source in zip(parsed_rows, rows, strict=True)
        ]


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace, output_jsonl: Path) -> list[dict[str, Any]]:
    completed = set() if args.force else done_keys(output_jsonl)
    selected: list[dict[str, Any]] = []
    start_index = int(args.start_index)
    end_index = None if args.max_samples is None else start_index + int(args.max_samples)
    for row in rows:
        audit_index = int(row.get("audit_index", -1))
        if audit_index < start_index:
            continue
        if end_index is not None and audit_index >= end_index:
            continue
        key = (str(row.get("sample_id")), str(row.get("qid")))
        if key in completed:
            continue
        selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    audit_dir = args.audit_dir
    manifest_path = args.manifest or audit_dir / "manifest.jsonl"
    output_jsonl = args.output_jsonl or audit_dir / "vision_judge_results.jsonl"
    summary_json = args.summary_json or audit_dir / "vision_judge_summary.json"
    if args.force:
        output_jsonl.write_text("", encoding="utf-8")

    rows = read_jsonl(manifest_path)
    selected = select_rows(rows, args, output_jsonl)
    counts: Counter[str] = Counter()
    processed = 0
    failed = 0

    for offset in range(0, len(selected), int(args.batch_size)):
        batch = selected[offset : offset + int(args.batch_size)]
        last_error: Exception | None = None
        results: list[dict[str, Any]] | None = None
        for attempt in range(int(args.retries) + 1):
            try:
                results = run_codex_batch(
                    batch,
                    audit_dir=audit_dir,
                    model=str(args.model),
                    effort=str(args.effort),
                    timeout_s=int(args.timeout_s),
                )
                break
            except Exception as exc:  # noqa: BLE001 - preserve failure in JSONL for resumability.
                last_error = exc
                if attempt < int(args.retries):
                    time.sleep(2.0 * (attempt + 1))
        if results is None:
            failed += len(batch)
            results = [
                failure_result(row, model=str(args.model), effort=str(args.effort), error=str(last_error))
                for row in batch
            ]
        for result in results:
            append_jsonl(output_jsonl, result)
            counts[str(result.get("visual_verdict", "missing"))] += 1
            processed += 1
        print(json.dumps({"processed": processed, "failed": failed, "counts": dict(counts)}, sort_keys=True))

    summary = {
        "audit_dir": str(audit_dir),
        "manifest_path": str(manifest_path),
        "output_jsonl": str(output_jsonl),
        "selected": len(selected),
        "processed": processed,
        "failed": failed,
        "counts": dict(counts),
        "model": str(args.model),
        "effort": str(args.effort),
        "batch_size": int(args.batch_size),
        "start_index": int(args.start_index),
        "max_samples": args.max_samples,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
