#!/usr/bin/env python3
"""Post-hoc pairwise diversity and CoC reasoning audit for AE rows.jsonl outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_pairwise_reasoning_from_decode_summary import (  # noqa: E402
    agreement,
    coc_causal_score,
    mean,
    normalize_text,
    pairwise_distances,
    scene_bucket_from_text,
    stats,
)


DEFAULT_ROWS = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "student_noflex_ae28_semantic_val806_official_t06_n6_20260615"
    / "student_noflex_ae28"
    / "rows.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "student_noflex_ae28_semantic_val806_official_t06_n6_20260615"
    / "student_noflex_ae28"
    / "summary.json"
)
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "benchmark_semantic_val_cap50_seed42.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "student_noflex_ae28_pairwise_reasoning_20260622"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collapse-threshold-m", type=float, default=0.25)
    parser.add_argument("--low-diversity-threshold-m", type=float, default=1.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_records(records: list[dict[str, Any]], collapse_threshold_m: float, low_diversity_threshold_m: float) -> dict[str, Any]:
    pairwise_mean = [float(row["pairwise_mean_ade_m"]) for row in records if row.get("pairwise_mean_ade_m") is not None]
    pairwise_min = [float(row["pairwise_min_ade_m"]) for row in records if row.get("pairwise_min_ade_m") is not None]
    pairwise_fde = [float(row["pairwise_mean_fde_m"]) for row in records if row.get("pairwise_mean_fde_m") is not None]
    causal = [float(row["coc_causal_score"]) for row in records]
    teacher_agree = [float(row["teacher_agreement_score"]) for row in records]
    return {
        "count": int(len(records)),
        "pairwise_mean_ade_m": stats(pairwise_mean),
        "pairwise_min_ade_m": stats(pairwise_min),
        "pairwise_mean_fde_m": stats(pairwise_fde),
        "collapse_rate_pairwise_mean_lt_threshold": float(
            sum(float(v) < collapse_threshold_m for v in pairwise_mean) / max(len(pairwise_mean), 1)
        ),
        "low_diversity_rate_pairwise_mean_lt_threshold": float(
            sum(float(v) < low_diversity_threshold_m for v in pairwise_mean) / max(len(pairwise_mean), 1)
        ),
        "coc_causal_score": stats(causal),
        "teacher_agreement_score": stats(teacher_agree),
        "teacher_exact_action_match_rate": float(
            sum(bool(row["teacher_exact_action_match"]) for row in records) / max(len(records), 1)
        ),
        "teacher_family_match_rate": float(
            sum(bool(row["teacher_family_match"]) for row in records) / max(len(records), 1)
        ),
        "teacher_scene_bucket_match_rate": float(
            sum(bool(row["teacher_scene_bucket_match"]) for row in records) / max(len(records), 1)
        ),
        "teacher_direction_conflict_rate": float(
            sum(bool(row["teacher_direction_conflict"]) for row in records) / max(len(records), 1)
        ),
    }


def by_category(records: list[dict[str, Any]], collapse_threshold_m: float, low_diversity_threshold_m: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("category") or "unknown")].append(row)
    return {
        category: summarize_records(rows, collapse_threshold_m, low_diversity_threshold_m)
        for category, rows in sorted(grouped.items())
    }


def maybe_load_paths(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    if "paths" not in payload.files:
        return None
    paths = np.asarray(payload["paths"], dtype=np.float32)
    if paths.ndim != 3 or paths.shape[0] < 2 or paths.shape[-1] < 2:
        return None
    return paths


def make_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    collapse_threshold = summary["collapse_threshold_m"]
    low_diversity_threshold = summary["low_diversity_threshold_m"]
    lines = [
        "# Student 2B Backbone + AE28 Pairwise Diversity + CoC Reasoning Audit",
        "",
        f"- Rows: `{summary['input_rows_jsonl']}`",
        f"- Corpus: `{summary['input_corpus_jsonl']}`",
        f"- Source summary: `{summary['input_summary_json']}`",
        f"- Samples: `{summary['num_samples']}`; candidates per sample: `{summary['samples_per_row']}`",
        f"- Student checkpoint: `{summary['checkpoint'].get('student_checkpoint')}`",
        f"- AE checkpoint: `{summary['checkpoint'].get('ae_checkpoint')}`",
        f"- Sampling: temperature `{summary['settings'].get('eval_temperature')}`, top_p `{summary['settings'].get('teacher_top_p')}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| pairwise mean ADE, mean | {overall['pairwise_mean_ade_m']['mean']:.4f} m |",
        f"| pairwise mean ADE, p50 | {overall['pairwise_mean_ade_m']['p50']:.4f} m |",
        f"| pairwise min ADE, p50 | {overall['pairwise_min_ade_m']['p50']:.4f} m |",
        f"| pairwise mean FDE, mean | {overall['pairwise_mean_fde_m']['mean']:.4f} m |",
        f"| collapse rate, pairwise mean < {collapse_threshold:.2f} m | {overall['collapse_rate_pairwise_mean_lt_threshold']:.4%} |",
        f"| low diversity rate, pairwise mean < {low_diversity_threshold:.1f} m | {overall['low_diversity_rate_pairwise_mean_lt_threshold']:.4%} |",
        f"| CoC causal score, mean | {overall['coc_causal_score']['mean']:.4f} |",
        f"| teacher agreement score, mean | {overall['teacher_agreement_score']['mean']:.4f} |",
        f"| teacher exact action match | {overall['teacher_exact_action_match_rate']:.4%} |",
        f"| teacher family match | {overall['teacher_family_match_rate']:.4%} |",
        f"| teacher scene bucket match | {overall['teacher_scene_bucket_match_rate']:.4%} |",
        f"| teacher direction conflict | {overall['teacher_direction_conflict_rate']:.4%} |",
        "",
        "## Category Highlights",
        "",
        "| Category | Count | Pairwise ADE mean | Collapse | Low-div | Causal | Teacher agree | Direction conflict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, payload in summary["by_category"].items():
        lines.append(
            "| "
            + f"{category} | {payload['count']} | {payload['pairwise_mean_ade_m']['mean']:.3f} | "
            + f"{payload['collapse_rate_pairwise_mean_lt_threshold']:.2%} | "
            + f"{payload['low_diversity_rate_pairwise_mean_lt_threshold']:.2%} | "
            + f"{payload['coc_causal_score']['mean']:.3f} | "
            + f"{payload['teacher_agreement_score']['mean']:.3f} | "
            + f"{payload['teacher_direction_conflict_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Worst Teacher-Agreement Examples",
            "",
            "| Sample | Category | Student | Teacher | Teacher score | Pairwise ADE |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in summary["worst_teacher_agreement_examples"]:
        student = str(row["student_cot"]).replace("|", " ")[:120]
        teacher = str(row["teacher_cot"]).replace("|", " ")[:120]
        lines.append(
            f"| `{row['sample_id']}` | {row['category']} | {student} | {teacher} | "
            f"{row['teacher_agreement_score']:.2f} | {row['pairwise_mean_ade_m']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.rows_jsonl)
    corpus = {str(row.get("sample_id")): row for row in read_jsonl(args.corpus_jsonl)}
    source_summary = json.loads(args.summary_json.read_text(encoding="utf-8")) if args.summary_json.exists() else {}

    per_sample: list[dict[str, Any]] = []
    missing_npz: list[str] = []
    missing_corpus: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        corpus_row = corpus.get(sample_id)
        if corpus_row is None:
            missing_corpus.append(sample_id)
            continue
        pred_path = PROJECT_ROOT / str(row["prediction_npz"])
        paths = maybe_load_paths(pred_path)
        if paths is None:
            missing_npz.append(sample_id)
            continue
        pair_ade, pair_fde = pairwise_distances([paths[idx] for idx in range(paths.shape[0])])
        teacher_cot = normalize_text(
            ((corpus_row.get("teacher_target") or {}).get("cot_text"))
            or ((corpus_row.get("hard_target") or {}).get("cot_text"))
        )
        student_cot = normalize_text(row.get("cot_preview") or row.get("generated_text"))
        category = str(
            row.get("category")
            or (corpus_row.get("metadata") or {}).get("semantic_scene_category")
            or scene_bucket_from_text(teacher_cot)
        )
        causal = coc_causal_score(student_cot)
        teacher = agreement(student_cot, teacher_cot)
        per_sample.append(
            {
                "sample_id": sample_id,
                "category": category,
                "candidate_count": int(paths.shape[0]),
                "pairwise_pair_count": int(len(pair_ade)),
                "pairwise_mean_ade_m": mean(pair_ade),
                "pairwise_min_ade_m": min(pair_ade) if pair_ade else None,
                "pairwise_max_ade_m": max(pair_ade) if pair_ade else None,
                "pairwise_mean_fde_m": mean(pair_fde),
                "pairwise_min_fde_m": min(pair_fde) if pair_fde else None,
                "pairwise_max_fde_m": max(pair_fde) if pair_fde else None,
                "selected_ade_gt_m": row.get("ade_gt_m"),
                "selected_fde_gt_m": row.get("fde_gt_m"),
                "oracle_minade6_gt_m": row.get("minade6_gt_m"),
                "best_path_idx_gt": row.get("best_path_idx_gt"),
                "student_cot": student_cot,
                "teacher_cot": teacher_cot,
                "coc_causal_score": causal["score"],
                "coc_causal_details": causal,
                "teacher_agreement_score": teacher["score"],
                "teacher_exact_action_match": teacher["exact_action_match"],
                "teacher_family_match": teacher["family_match"],
                "teacher_scene_bucket_match": teacher["scene_bucket_match"],
                "teacher_direction_conflict": teacher["direction_conflict"],
                "teacher_agreement_details": teacher,
            }
        )

    overall = summarize_records(per_sample, args.collapse_threshold_m, args.low_diversity_threshold_m)
    worst_teacher = sorted(
        per_sample,
        key=lambda item: (
            float(item["teacher_agreement_score"]),
            -(float(item["pairwise_mean_ade_m"]) if item.get("pairwise_mean_ade_m") is not None else -1.0),
        ),
    )[:20]
    best_teacher = sorted(
        per_sample,
        key=lambda item: (
            -float(item["teacher_agreement_score"]),
            float(item["pairwise_mean_ade_m"]) if item.get("pairwise_mean_ade_m") is not None else 1e9,
        ),
    )[:20]

    output = {
        "input_rows_jsonl": str(args.rows_jsonl),
        "input_summary_json": str(args.summary_json),
        "input_corpus_jsonl": str(args.corpus_jsonl),
        "num_samples": int(len(per_sample)),
        "missing_npz": missing_npz,
        "missing_corpus_rows": missing_corpus,
        "samples_per_row": int(source_summary.get("settings", {}).get("eval_num_paths") or 0),
        "settings": source_summary.get("settings") or {},
        "checkpoint": source_summary.get("checkpoint") or {},
        "collapse_threshold_m": float(args.collapse_threshold_m),
        "low_diversity_threshold_m": float(args.low_diversity_threshold_m),
        "overall": overall,
        "by_category": by_category(per_sample, args.collapse_threshold_m, args.low_diversity_threshold_m),
        "student_action_counts": dict(Counter(item["coc_causal_details"]["action"] for item in per_sample).most_common()),
        "teacher_action_counts": dict(Counter(item["teacher_agreement_details"]["reference_action"] for item in per_sample).most_common()),
        "student_bucket_counts": dict(Counter(item["teacher_agreement_details"]["student_bucket"] for item in per_sample).most_common()),
        "teacher_bucket_counts": dict(Counter(item["teacher_agreement_details"]["reference_bucket"] for item in per_sample).most_common()),
        "worst_teacher_agreement_examples": worst_teacher,
        "best_teacher_agreement_examples": best_teacher,
        "samples": per_sample,
    }
    out_json = args.output_dir / "summary.json"
    out_md = args.output_dir / "report.md"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(make_report(output), encoding="utf-8")
    print(json.dumps({"summary_json": str(out_json), "report_md": str(out_md), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
