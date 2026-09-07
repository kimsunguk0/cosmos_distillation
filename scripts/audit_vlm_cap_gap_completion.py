#!/usr/bin/env python3
"""Audit completion of the vanilla CR2 human-OOD VLM capacity-gap eval."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_MODELS = ("2b", "8b", "32b")
REQUIRED_TASKS = ("T1_neg", "T1_pos", "T2", "T3", "T4")
FORBIDDEN_JUDGE_KEYS = {"reference_32b", "event_cluster", "split_tag", "event_cot_hint"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_PATHS = {
    "2b": "/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b",
    "8b": "/home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-8B",
    "32b": "/home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-32B",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def file_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def file_nonempty_any(path_value: Any) -> bool:
    path = Path(str(path_value or ""))
    if file_nonempty(path):
        return True
    if not path.is_absolute() and file_nonempty(PROJECT_ROOT / path):
        return True
    return False


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})

    @property
    def ok(self) -> bool:
        return all(item["ok"] for item in self.checks)


def audit_predictions(
    audit: Audit,
    out_dir: Path,
    expected_rows: int,
    expected_clips: int,
    expected_samples_per_task: int,
) -> None:
    expected_task_rows = expected_clips * expected_samples_per_task
    expected_sample_rows = expected_clips * len(REQUIRED_TASKS)
    for model in REQUIRED_MODELS:
        pred_path = out_dir / "runs" / model / "predictions.jsonl"
        audit.add(f"{model}.predictions.exists", file_nonempty(pred_path), str(pred_path))
        if not file_nonempty(pred_path):
            continue
        rows = read_jsonl(pred_path)
        task_counts = Counter(str(row.get("task_id")) for row in rows)
        sample_counts = Counter(str(row.get("sample_idx")) for row in rows)
        keys = [(row.get("clip_id"), row.get("task_id"), row.get("sample_idx")) for row in rows]
        parse_errors = sum(1 for row in rows if row.get("parse_error"))
        runtime_errors = sum(1 for row in rows if row.get("error"))
        unexpected_tasks = sorted(set(task_counts) - set(REQUIRED_TASKS))
        expected_sample_keys = {str(idx) for idx in range(expected_samples_per_task)}
        unexpected_samples = sorted(set(sample_counts) - expected_sample_keys)
        wrong_model_paths = sorted({str(row.get("model_path")) for row in rows if str(row.get("model_path")) != EXPECTED_MODEL_PATHS[model]})
        bad_temps = []
        for row in rows:
            sample_idx = safe_int(row.get("sample_idx"))
            temperature = safe_float(row.get("temperature"))
            ok_temp = (
                sample_idx == 0
                and temperature == 0.0
            ) or (
                sample_idx is not None
                and sample_idx > 0
                and temperature is not None
                and abs(temperature - 0.7) < 1e-9
            )
            if not ok_temp:
                bad_temps.append(
                    {
                        "line_no": row.get("_line_no"),
                        "sample_idx": row.get("sample_idx"),
                        "temperature": row.get("temperature"),
                    }
                )
        missing_questions = [
            {"line_no": row.get("_line_no"), "task_id": row.get("task_id")}
            for row in rows
            if row.get("task_id") in REQUIRED_TASKS and not isinstance(row.get("question"), str)
        ]
        missing_videos = [
            {"line_no": row.get("_line_no"), "video_path": row.get("video_path")}
            for row in rows
            if safe_int(row.get("sample_idx")) == 0 and not file_nonempty_any(row.get("video_path"))
        ]
        audit.add(f"{model}.row_count", len(rows) == expected_rows, {"actual": len(rows), "expected": expected_rows})
        audit.add(f"{model}.unique_keys", len(set(keys)) == len(keys), {"rows": len(rows), "unique": len(set(keys))})
        audit.add(f"{model}.model_path", not wrong_model_paths, {"unexpected": wrong_model_paths})
        audit.add(f"{model}.temperature_mapping", not bad_temps, {"bad_count": len(bad_temps), "examples": bad_temps[:20]})
        audit.add(f"{model}.questions_present", not missing_questions, {"missing_count": len(missing_questions), "examples": missing_questions[:20]})
        audit.add(f"{model}.video_cache_exists", not missing_videos, {"missing_count": len(missing_videos), "examples": missing_videos[:20]})
        audit.add(f"{model}.no_unexpected_tasks", not unexpected_tasks, {"unexpected": unexpected_tasks})
        audit.add(f"{model}.no_unexpected_samples", not unexpected_samples, {"unexpected": unexpected_samples})
        audit.add(
            f"{model}.task_distribution",
            all(task_counts[task] == expected_task_rows for task in REQUIRED_TASKS),
            {"actual": dict(task_counts), "expected_each": expected_task_rows},
        )
        audit.add(
            f"{model}.sample_distribution",
            all(sample_counts[str(idx)] == expected_sample_rows for idx in range(expected_samples_per_task)),
            {"actual": dict(sample_counts), "expected_each": expected_sample_rows},
        )
        audit.add(f"{model}.runtime_errors", runtime_errors == 0, {"runtime_errors": runtime_errors, "parse_errors": parse_errors})


def audit_judge_pack(audit: Audit, out_dir: Path, judge_name: str, expected_pairs: int, require_judge: bool) -> None:
    judge_dir = out_dir / judge_name
    pairs_path = judge_dir / "pairs.jsonl"
    key_path = judge_dir / "answer_key.json"
    judgments_path = judge_dir / "judgments_blind.jsonl"
    validation_path = judge_dir / "judgments_validation.json"
    summary_path = judge_dir / "judge_summary.json"

    audit.add("judge.pairs.exists", file_nonempty(pairs_path), str(pairs_path))
    audit.add("judge.answer_key.exists", file_nonempty(key_path), str(key_path))
    if not file_nonempty(pairs_path):
        if require_judge:
            audit.add("judge.judgments.exists", False, str(judgments_path))
            audit.add("judge.summary.exists", False, str(summary_path))
        return

    pairs = read_jsonl(pairs_path)
    forbidden_present = sorted({key for row in pairs for key in FORBIDDEN_JUDGE_KEYS if key in row})
    contact_missing = [row.get("case_id") for row in pairs if not file_nonempty_any(row.get("contact_sheet"))]
    audit.add("judge.pair_count", len(pairs) == expected_pairs, {"actual": len(pairs), "expected": expected_pairs})
    audit.add("judge.no_forbidden_metadata", not forbidden_present, forbidden_present)
    audit.add("judge.contact_sheets_exist", not contact_missing, {"missing": contact_missing[:20], "missing_count": len(contact_missing)})

    if not require_judge:
        return

    audit.add("judge.judgments.exists", file_nonempty(judgments_path), str(judgments_path))
    audit.add("judge.validation.exists", file_nonempty(validation_path), str(validation_path))
    audit.add("judge.summary.exists", file_nonempty(summary_path), str(summary_path))
    if not file_nonempty(judgments_path) or not file_nonempty(summary_path):
        return
    if file_nonempty(validation_path):
        validation = read_json(validation_path)
        audit.add("judge.validation.ok", validation.get("ok") is True, validation)
    judgments = read_jsonl(judgments_path)
    summary = read_json(summary_path)
    missing = summary.get("missing_case_ids") or []
    duplicates = summary.get("duplicate_case_ids") or []
    total_cases = ((summary.get("overall") or {}).get("total_cases"))
    audit.add("judge.judgment_count", len(judgments) == len(pairs), {"judgments": len(judgments), "pairs": len(pairs)})
    audit.add("judge.summary_total", total_cases == len(pairs), {"summary_total": total_cases, "pairs": len(pairs)})
    audit.add("judge.no_missing_cases", not missing, {"missing_count": len(missing), "missing": missing[:20]})
    audit.add("judge.no_duplicate_cases", not duplicates, {"duplicate_count": len(duplicates), "duplicates": duplicates[:20]})


def audit_report(audit: Audit, out_dir: Path, require_judge: bool, require_qualitative: bool) -> None:
    score_path = out_dir / "score_summary.json"
    report_path = out_dir / "report.md"
    qual_path = out_dir / "qualitative_examples.md"
    decision_path = out_dir / "decision_brief.md"
    audit.add("score_summary.exists", file_nonempty(score_path), str(score_path))
    audit.add("report.exists", file_nonempty(report_path), str(report_path))
    if require_qualitative:
        audit.add("qualitative_examples.exists", file_nonempty(qual_path), str(qual_path))
    if require_judge:
        audit.add("decision_brief.exists", file_nonempty(decision_path), str(decision_path))
    if file_nonempty(score_path):
        score = read_json(score_path)
        audit.add("score.models", set(score.get("models", [])) == set(REQUIRED_MODELS), score.get("models"))
        summaries = score.get("summaries") or {}
        audit.add("score.has_summaries", all(model in summaries for model in REQUIRED_MODELS), sorted(summaries))
    if file_nonempty(report_path):
        text = report_path.read_text(encoding="utf-8")
        required_sections = ["## Data", "## Metrics", "## ECE Lite", "## Pairwise Proxy", "## Recommendations"]
        if require_judge:
            required_sections.append("## Blind LLM Judge")
        missing_sections = [section for section in required_sections if section not in text]
        audit.add("report.required_sections", not missing_sections, missing_sections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-clips", type=int, default=300)
    parser.add_argument("--expected-rows-per-model", type=int, default=9000)
    parser.add_argument("--expected-samples-per-task", type=int, default=6)
    parser.add_argument("--judge-name", default="open_judge_codex55_xhigh_full300")
    parser.add_argument("--expected-judge-pairs", type=int, default=100)
    parser.add_argument("--require-judge", action="store_true")
    parser.add_argument("--allow-missing-qualitative", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    audit = Audit()

    manifest_path = out_dir / "manifest.jsonl"
    manifest_summary_path = out_dir / "manifest_summary.json"
    audit.add("manifest.exists", file_nonempty(manifest_path), str(manifest_path))
    audit.add("manifest_summary.exists", file_nonempty(manifest_summary_path), str(manifest_summary_path))
    if file_nonempty(manifest_path):
        manifest_rows = read_jsonl(manifest_path)
        audit.add("manifest.clip_count", len(manifest_rows) == args.expected_clips, {"actual": len(manifest_rows), "expected": args.expected_clips})
    if file_nonempty(manifest_summary_path):
        manifest_summary = read_json(manifest_summary_path)
        audit.add(
            "manifest_summary.selected",
            int(manifest_summary.get("selected", -1)) == args.expected_clips,
            manifest_summary,
        )

    audit_predictions(
        audit,
        out_dir,
        args.expected_rows_per_model,
        args.expected_clips,
        args.expected_samples_per_task,
    )
    audit_report(audit, out_dir, args.require_judge, not args.allow_missing_qualitative)
    audit_judge_pack(audit, out_dir, args.judge_name, args.expected_judge_pairs, args.require_judge)

    payload = {"ok": audit.ok, "output_dir": str(out_dir), "checks": audit.checks}
    print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if audit.ok else 1


if __name__ == "__main__":
    sys.exit(main())
