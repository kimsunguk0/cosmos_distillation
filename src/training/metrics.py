"""Metric names used by evaluation and reporting."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "json_parseability",
    "meta_action_f1",
    "human_coc_overlap",
    "category_qa_accuracy",
    "hallucination_rate",
    "teacher_human_disagreement_rate",
    "consistency_gate_score",
    "rationale_answer_consistency",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def exact_match_rate(pairs: list[tuple[str, str]]) -> float:
    """Compute case-insensitive exact match rate."""
    if not pairs:
        return 0.0
    hits = 0
    for left, right in pairs:
        if str(left).strip().lower() == str(right).strip().lower():
            hits += 1
    return hits / len(pairs)


def jaccard_overlap(left: str, right: str) -> float:
    """Simple lexical-overlap score for quick reporting."""
    left_tokens = {token for token in str(left).lower().split() if token}
    right_tokens = {token for token in str(right).lower().split() if token}
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def summarize_levels(values: list[str]) -> dict[str, int]:
    """Count level strings."""
    return dict(Counter(values))
