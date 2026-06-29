#!/usr/bin/env python3
"""Run a paired vision judge for 1A scene-context answers."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_HEADER = """You are judging autonomous-driving VQA answers for a 1A perception/scene-context task.

Each attached image is a 2x2 contact sheet from one driving sample:
- front-left, front-wide, front-right, and front-telephoto cameras
- one frame per camera
- camera labels are printed at the top of each cell

Question:
What kind of driving situation is shown? Describe the overall scene context in one or two sentences based only on clearly visible evidence. Focus on road type, environment, and notable conditions, not on counting.

Compare Answer A and Answer B using only visible evidence in the contact sheet.

Reward:
- Correct road type and environment: urban street, intersection, highway, residential street, night/day, wet road, work zone, etc.
- Clearly visible notable conditions: traffic light, stop sign, construction, parked vehicles, occlusion, open road, crosswalk, lane markings.
- Concise scene-context description, not a list of imagined rules.

Penalize:
- Hallucinated objects, workers, bicyclists, signs, lane counts, road markings, weather, or vehicles not visible.
- Exact counts or over-specific lane geometry when not clearly supported.
- Driving action advice, future prediction, hidden hazards, intent, or generic safety text.
- Repetition, truncated output, or incoherent text.

Return JSON only, no markdown:
{
  "sample_id": "...",
  "better": "A|B|tie|neither",
  "score_a": 0|1|2|3,
  "score_b": 0|1|2|3,
  "unsupported_a": ["..."],
  "unsupported_b": ["..."],
  "reason": "...",
  "judge_model": "...",
  "judge_effort": "..."
}

Scoring:
3 = mostly correct and grounded scene context
2 = partially correct with minor unsupported detail
1 = weak/generic or several unsupported claims
0 = mostly wrong, hallucinated, repetitive, or unusable
"""


def codex_bin() -> str:
    preferred = Path("/home/pm97/.npm-global/bin/codex")
    return str(preferred) if preferred.exists() else "codex"


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


def done_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id")) for row in read_jsonl(path) if row.get("sample_id")}


def resolve_image(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def build_prompt(row: dict[str, Any], *, model: str, effort: str) -> str:
    payload = {
        "sample_id": row.get("sample_id"),
        "answer_a": row.get("answer_a"),
        "answer_b": row.get("answer_b"),
        "judge_model": model,
        "judge_effort": effort,
    }
    return PROMPT_HEADER + "\nSample:\n" + json.dumps(payload, ensure_ascii=True, indent=2) + "\n"


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(stripped[start : end + 1])


def normalize_result(raw: dict[str, Any], source: dict[str, Any], *, model: str, effort: str) -> dict[str, Any]:
    better = str(raw.get("better") or "neither")
    if better not in {"A", "B", "tie", "neither"}:
        better = "neither"
    score_a = int(raw.get("score_a", 0))
    score_b = int(raw.get("score_b", 0))
    score_a = max(0, min(3, score_a))
    score_b = max(0, min(3, score_b))
    winner_model = None
    if better == "A":
        winner_model = source.get("answer_a_model")
    elif better == "B":
        winner_model = source.get("answer_b_model")
    return {
        "sample_id": str(source.get("sample_id")),
        "better": better,
        "winner_model": winner_model,
        "score_a": score_a,
        "score_b": score_b,
        "score_ft": score_a if source.get("answer_a_model") == "ft_step3488" else score_b,
        "score_base": score_a if source.get("answer_a_model") == "base2b" else score_b,
        "unsupported_a": [str(item) for item in (raw.get("unsupported_a") or [])],
        "unsupported_b": [str(item) for item in (raw.get("unsupported_b") or [])],
        "reason": str(raw.get("reason") or ""),
        "answer_a_model": source.get("answer_a_model"),
        "answer_b_model": source.get("answer_b_model"),
        "answer_a": source.get("answer_a"),
        "answer_b": source.get("answer_b"),
        "image_path": source.get("image_path"),
        "judge_model": str(raw.get("judge_model") or model),
        "judge_effort": str(raw.get("judge_effort") or effort),
    }


def run_one(row: dict[str, Any], *, audit_dir: Path, model: str, effort: str, timeout_s: int) -> dict[str, Any]:
    image_path = resolve_image(audit_dir, str(row["image_path"]))
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    prompt = build_prompt(row, model=model, effort=effort)
    with tempfile.TemporaryDirectory(prefix="1a_pair_judge_") as tmpdir:
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
            "workspace-write",
            "-C",
            str(PROJECT_ROOT),
            "--ephemeral",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
            "--image",
            str(image_path),
        ]
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
        return normalize_result(parse_json_response(response_text), row, model=model, effort=effort)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    winners = Counter(str(row.get("winner_model") or row.get("better")) for row in rows)
    better = Counter(str(row.get("better")) for row in rows)
    ft_scores = [int(row["score_ft"]) for row in rows]
    base_scores = [int(row["score_base"]) for row in rows]
    return {
        "n": n,
        "better_counts": dict(better),
        "winner_model_counts": dict(winners),
        "ft_win_rate": winners.get("ft_step3488", 0) / max(1, n),
        "base_win_rate": winners.get("base2b", 0) / max(1, n),
        "tie_or_neither_rate": (better.get("tie", 0) + better.get("neither", 0)) / max(1, n),
        "ft_score_mean": sum(ft_scores) / max(1, len(ft_scores)),
        "base_score_mean": sum(base_scores) / max(1, len(base_scores)),
        "score_delta_ft_minus_base": (sum(ft_scores) / max(1, len(ft_scores))) - (sum(base_scores) / max(1, len(base_scores))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.audit_dir / "paired_manifest.jsonl"
    output_jsonl = args.output_jsonl or args.audit_dir / "vision_pair_judge_results.jsonl"
    summary_json = args.summary_json or args.audit_dir / "vision_pair_judge_summary.json"
    rows = read_jsonl(manifest)
    if args.start_index:
        rows = [row for row in rows if int(row.get("audit_index", -1)) >= int(args.start_index)]
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    completed = set() if args.force else done_sample_ids(output_jsonl)
    if args.force and output_jsonl.exists():
        output_jsonl.unlink()
    started = time.time()
    for index, row in enumerate(rows, start=1):
        if str(row.get("sample_id")) in completed:
            continue
        last_error: Exception | None = None
        for attempt in range(int(args.retries) + 1):
            try:
                result = run_one(
                    row,
                    audit_dir=args.audit_dir,
                    model=str(args.model),
                    effort=str(args.effort),
                    timeout_s=int(args.timeout_s),
                )
                append_jsonl(output_jsonl, result)
                print(
                    json.dumps(
                        {
                            "event": "judged",
                            "index": index,
                            "sample_id": row.get("sample_id"),
                            "better": result["better"],
                            "winner_model": result["winner_model"],
                            "score_ft": result["score_ft"],
                            "score_base": result["score_base"],
                            "elapsed_sec": round(time.time() - started, 1),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < int(args.retries):
                    time.sleep(2)
        if last_error is not None:
            raise RuntimeError(f"Failed sample {row.get('sample_id')}: {last_error}") from last_error
    judged = read_jsonl(output_jsonl) if output_jsonl.exists() else []
    summary = {
        "audit_dir": str(args.audit_dir),
        "manifest": str(manifest),
        "output_jsonl": str(output_jsonl),
        "model": str(args.model),
        "effort": str(args.effort),
        "elapsed_sec": round(time.time() - started, 3),
        **summarize(judged),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
