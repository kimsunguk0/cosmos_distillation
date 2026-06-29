#!/usr/bin/env python3
"""Compare two Step A Q2 VQA eval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = [
    "teacher_short_token_f1_mean",
    "supported_claim_token_f1_mean",
    "hard_bad_output_rate",
    "action_or_future_language_rate",
    "mean_word_count",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_row(summary_a: dict[str, Any], summary_b: dict[str, Any], scope: str, metric: str) -> dict[str, Any]:
    group_a = summary_a["overall"] if scope == "overall" else summary_a["by_split"][scope]
    group_b = summary_b["overall"] if scope == "overall" else summary_b["by_split"][scope]
    a = group_a.get(metric)
    b = group_b.get(metric)
    delta = None if a is None or b is None else float(a) - float(b)
    return {"scope": scope, "metric": metric, "a": a, "b": b, "delta_a_minus_b": delta}


def build_comparison(summary_a: dict[str, Any], summary_b: dict[str, Any], *, label_a: str, label_b: str) -> dict[str, Any]:
    scopes = ["overall"]
    scopes.extend(sorted(set(summary_a.get("by_split", {})) & set(summary_b.get("by_split", {}))))
    rows = [metric_row(summary_a, summary_b, scope, metric) for scope in scopes for metric in METRICS]
    return {
        "label_a": label_a,
        "label_b": label_b,
        "summary_a": summary_a.get("output_dir"),
        "summary_b": summary_b.get("output_dir"),
        "rows": rows,
    }


def markdown_table(comparison: dict[str, Any]) -> str:
    label_a = comparison["label_a"]
    label_b = comparison["label_b"]
    lines = [
        "# Step A Q2 VQA Eval Comparison",
        "",
        f"- A: `{label_a}`",
        f"- B: `{label_b}`",
        "",
        f"| scope | metric | {label_a} | {label_b} | delta A-B |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        lines.append(
            "| {scope} | {metric} | {a} | {b} | {delta} |".format(
                scope=row["scope"],
                metric=row["metric"],
                a=fmt(row["a"]),
                b=fmt(row["b"]),
                delta=fmt(row["delta_a_minus_b"]),
            )
        )
    lines.append("")
    lines.append("Positive delta is better for overlap metrics and worse for bad-output rates/word count.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-a", type=Path, required=True)
    parser.add_argument("--summary-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = build_comparison(
        load_json(args.summary_a),
        load_json(args.summary_b),
        label_a=str(args.label_a),
        label_b=str(args.label_b),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, ensure_ascii=True), encoding="utf-8")
    args.output_md.write_text(markdown_table(comparison), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
