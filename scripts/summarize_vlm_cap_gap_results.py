#!/usr/bin/env python3
"""Write a compact decision brief for the CR2 VLM capacity-gap eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ("2b", "8b", "32b")
LABELS = {"2b": "2B", "8b": "8B", "32b": "32B"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(summary: dict[str, Any] | None, key: str) -> float | None:
    if not summary:
        return None
    item = summary.get(key)
    if isinstance(item, dict):
        mean = item.get("mean")
        return float(mean) if isinstance(mean, (float, int)) else None
    if isinstance(item, (float, int)):
        return float(item)
    return None


def fmt_metric(item: Any) -> str:
    if item is None:
        return "-"
    if isinstance(item, dict):
        mean = item.get("mean")
        if mean is None:
            return "-"
        n = item.get("n")
        lo = item.get("ci95_low")
        hi = item.get("ci95_high")
        ci = f", 95% CI {lo:.3f}-{hi:.3f}" if isinstance(lo, (float, int)) and isinstance(hi, (float, int)) else ""
        return f"{float(mean):.3f} (n={n}{ci})"
    if isinstance(item, (float, int)):
        return f"{float(item):.3f}"
    return str(item)


def ece(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    item = summary.get("ECE_lite")
    if isinstance(item, dict) and isinstance(item.get("ece"), (float, int)):
        return float(item["ece"])
    return None


def recommend(metric: str, summaries: dict[str, Any]) -> str:
    s2 = summaries.get("2b")
    s8 = summaries.get("8b")
    s32 = summaries.get("32b")
    if not s2 or not s8:
        return "insufficient"
    if metric == "T1":
        a2 = value(s2, "T1_accuracy_present")
        a8 = value(s8, "T1_accuracy_present")
        a32 = value(s32, "T1_accuracy_present")
        h2 = value(s2, "T1_hallucination_rate_negative")
        h8 = value(s8, "T1_hallucination_rate_negative")
        h32 = value(s32, "T1_hallucination_rate_negative")
        e2 = ece(s2)
        if a8 is not None and a32 is not None and a8 < 0.65 and a32 < 0.65:
            return "capacity ceiling/data issue"
        if h8 is not None and h32 is not None and h8 > 0.20 and h32 > 0.20:
            return "capacity ceiling/data issue"
        if a2 is not None and a2 >= 0.75 and e2 is not None and e2 > 0.20:
            return "target pre-stage"
        if a2 is not None and a8 is not None and (a8 - a2) < 0.05 and (h2 is not None and h2 < 0.10):
            return "skip"
        if a2 is not None and a8 is not None and (a8 - a2) >= 0.10:
            return "target pre-stage"
    if metric == "T2":
        q2 = value(s2, "T2_abstention_quality")
        q8 = value(s8, "T2_abstention_quality")
        q32 = value(s32, "T2_abstention_quality")
        amb2 = value(s2, "T2_abstain_ambiguous")
        clear2 = value(s2, "T2_abstain_clear")
        if q2 is not None and q8 is not None:
            if q32 is not None and q8 < 0.55 and q32 < 0.55:
                return "capacity ceiling/data issue"
            if amb2 is not None and amb2 < 0.70:
                return "target pre-stage"
            if clear2 is not None and clear2 > 0.20:
                return "target pre-stage"
            if q2 >= 0.80 and q8 >= 0.80 and abs(q8 - q2) < 0.05:
                return "skip"
            if (q8 - q2) >= 0.10:
                return "target pre-stage"
    if metric == "T4":
        r2 = value(s2, "T4_ref_agreement_32b")
        r8 = value(s8, "T4_ref_agreement_32b")
        if r2 is not None and r8 is not None:
            if r2 >= 0.80 and (r8 - r2) < 0.05:
                return "skip"
            if (r8 - r2) >= 0.10:
                return "target pre-stage"
            if r8 < 0.55:
                return "capacity ceiling/data issue"
    return "target pre-stage"


def metric_table(score: dict[str, Any]) -> list[str]:
    summaries = score.get("summaries") or {}
    metrics = [
        ("parsed_json_rate", "JSON parse"),
        ("T1_accuracy_present", "T1 presence"),
        ("T1_hallucination_rate_negative", "T1 neg hallucination"),
        ("T2_abstention_quality", "T2 abstention quality"),
        ("consistency", "Consistency"),
        ("T3_ref_agreement_32b", "T3 ref agreement"),
        ("T4_ref_agreement_32b", "T4 ref agreement"),
    ]
    lines = [
        "| metric | 2B | 8B | 32B | delta 8B-2B |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in metrics:
        vals = [value(summaries.get(model), key) for model in MODELS]
        delta = "-" if vals[0] is None or vals[1] is None else f"{(vals[1] - vals[0]):.3f}"
        cells = [fmt_metric((summaries.get(model) or {}).get(key)) for model in MODELS]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {delta} |")
    return lines


def write_brief(out_dir: Path, output: Path, judge_name: str) -> None:
    score = read_json(out_dir / "score_summary.json")
    summaries = score.get("summaries") or {}
    manifest = score.get("manifest_summary") or {}
    judge_path = out_dir / judge_name / "judge_summary.json"
    audit_path = out_dir / "completion_audit_final.json"
    judge = read_json(judge_path) if judge_path.exists() else None
    audit = read_json(audit_path) if audit_path.exists() else None

    lines: list[str] = []
    lines.append("# CR2 VLM Capacity Gap Decision Brief")
    lines.append("")
    lines.append(f"- Output root: `{out_dir}`")
    lines.append(f"- Completion audit: `{(audit or {}).get('ok')}`")
    lines.append(f"- Selected clips: `{manifest.get('selected')}` from candidate human-OOD `{manifest.get('candidate_human_ood')}`")
    lines.append(f"- Split tags: `{manifest.get('split_tag_counts')}`")
    lines.append(f"- Event clusters: `{manifest.get('event_cluster_counts')}`")
    lines.append("- Gold limitation: local NCore cuboid GT was not present; T1/T2 use OOD metadata/heuristics, T3/T4 use 32B proxy plus blind pairwise judge.")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.extend(metric_table(score))
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append("| ability | decision | basis |")
    lines.append("|---|---|---|")
    lines.append(
        "| T1 perception/hallucination | "
        f"{recommend('T1', summaries)} | "
        f"2B presence {fmt_metric((summaries.get('2b') or {}).get('T1_accuracy_present'))}; "
        f"2B negative hallucination {fmt_metric((summaries.get('2b') or {}).get('T1_hallucination_rate_negative'))}; "
        f"2B ECE {ece(summaries.get('2b'))} |"
    )
    lines.append(
        "| T2 calibration/abstention | "
        f"{recommend('T2', summaries)} | "
        f"2B quality {fmt_metric((summaries.get('2b') or {}).get('T2_abstention_quality'))}; "
        f"8B quality {fmt_metric((summaries.get('8b') or {}).get('T2_abstention_quality'))}; "
        f"32B quality {fmt_metric((summaries.get('32b') or {}).get('T2_abstention_quality'))} |"
    )
    lines.append(
        "| T4 causal decision | "
        f"{recommend('T4', summaries)} | "
        f"2B ref agreement {fmt_metric((summaries.get('2b') or {}).get('T4_ref_agreement_32b'))}; "
        f"8B ref agreement {fmt_metric((summaries.get('8b') or {}).get('T4_ref_agreement_32b'))}; "
        "see blind judge below |"
    )
    lines.append("")
    lines.append("## Blind Judge")
    lines.append("")
    if judge:
        overall = judge.get("overall") or {}
        lines.append(
            f"- Overall: 2B={overall.get('2b')}, 8B={overall.get('8b')}, tie={overall.get('tie')}, "
            f"both_bad={overall.get('both_bad')}, invalid={overall.get('invalid')}, total={overall.get('total_cases')}"
        )
        lines.append(f"- Decisive winrate: 2B={overall.get('2b_winrate_decisive')}, 8B={overall.get('8b_winrate_decisive')}")
        lines.append("")
        lines.append("| task | 2B | 8B | tie | both_bad | invalid | total |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for task_id, counts in sorted((judge.get("by_task") or {}).items()):
            lines.append(
                f"| {task_id} | {counts.get('2b')} | {counts.get('8b')} | {counts.get('tie')} | "
                f"{counts.get('both_bad')} | {counts.get('invalid')} | {counts.get('total_cases')} |"
            )
    else:
        lines.append("- Not available yet.")
    lines.append("")
    lines.append("## Operational Recommendation")
    lines.append("")
    lines.append("- Use these vanilla-backbone numbers to decide the pre-stage target; do not interpret them as post-distillation VLA performance.")
    lines.append("- If T2 remains weak while T1 is acceptable, prioritize calibration/abstention or decision-support VQA before action-token distillation.")
    lines.append("- If T4 is weak for 2B but 8B/32B are strong, target causal decision pre-stage; if 8B/32B are also weak, treat it as a data/task ceiling signal.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--judge-name", default="open_judge_codex55_xhigh_full300")
    args = parser.parse_args()
    output = args.out or (args.output_dir / "decision_brief.md")
    write_brief(args.output_dir, output, args.judge_name)
    print(json.dumps({"decision_brief": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
