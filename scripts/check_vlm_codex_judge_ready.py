#!/usr/bin/env python3
"""Check whether the full300 blind judge pack is ready for Codex judging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PAIR_KEYS = {"reference_32b", "event_cluster", "split_tag", "event_cot_hint"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--judge-name", default="open_judge_codex55_xhigh_full300")
    parser.add_argument("--expected-pairs", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    judge_dir = args.output_dir / args.judge_name
    pairs_path = judge_dir / "pairs.jsonl"
    key_path = judge_dir / "answer_key.json"
    judgments_path = judge_dir / "judgments_blind.jsonl"
    validation_path = judge_dir / "judgments_validation.json"
    summary_path = judge_dir / "judge_summary.json"
    prompt_path = PROJECT_ROOT / "scripts" / "vlm_cap_gap_codex_judge_prompt.md"

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("pairs.exists", file_nonempty(pairs_path), str(pairs_path))
    add("answer_key.exists", file_nonempty(key_path), str(key_path))
    add("prompt.exists", file_nonempty(prompt_path), str(prompt_path))

    pairs: list[dict[str, Any]] = []
    if file_nonempty(pairs_path):
        pairs = read_jsonl(pairs_path)
        forbidden = sorted({key for row in pairs for key in FORBIDDEN_PAIR_KEYS if key in row})
        missing_contact = [
            row.get("case_id")
            for row in pairs
            if not file_nonempty_any(row.get("contact_sheet"))
        ]
        add("pairs.count", len(pairs) == args.expected_pairs, {"actual": len(pairs), "expected": args.expected_pairs})
        add("pairs.no_forbidden_metadata", not forbidden, forbidden)
        add("pairs.contact_sheets_exist", not missing_contact, {"missing_count": len(missing_contact), "missing": missing_contact[:20]})

    ready_for_blind_judge = (
        all(item["ok"] for item in checks)
        and not judgments_path.exists()
    )
    needs_scoring = file_nonempty(judgments_path) and not file_nonempty(summary_path)
    complete = file_nonempty(summary_path)

    payload = {
        "judge_dir": str(judge_dir),
        "prompt": str(prompt_path),
        "pairs": len(pairs),
        "ready_for_blind_judge": ready_for_blind_judge,
        "needs_scoring": needs_scoring,
        "complete": complete,
        "judgments": str(judgments_path),
        "validation": str(validation_path),
        "summary": str(summary_path),
        "checks": checks,
        "next_action": (
            "spawn_codex_5_5_xhigh_with_prompt"
            if ready_for_blind_judge
            else "run_validate_and_score"
            if needs_scoring
            else "done"
            if complete
            else "wait_for_postprocess"
        ),
    }

    text = json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if (ready_for_blind_judge or needs_scoring or complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
