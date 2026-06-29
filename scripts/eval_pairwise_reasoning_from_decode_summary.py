#!/usr/bin/env python3
"""Post-hoc pairwise diversity and CoC reasoning audit from decode summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import TrajectoryTokenDecoder, load_ego_history_rot  # noqa: E402
from src.training.collator import load_ego_history_xyz  # noqa: E402


DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "benchmarks"
    / "run2_fp8vit_gkd_semantic_val806_20260619"
    / "backbone_run2_fp8vit_gkd_semantic_val806_official_t06_topp098_n6"
    / "summary.json"
)
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "benchmark_semantic_val_cap50_seed42.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "run2_pairwise_reasoning_20260622"

SEMANTIC_CATEGORIES = [
    "traffic_right_turn",
    "traffic_left_turn",
    "right_turn_no_light",
    "left_turn_no_light",
    "red_light_stop",
    "stop_sign",
    "pedestrian_crosswalk",
    "cut_in_merge_yield",
    "lead_vehicle_follow",
    "parked_stopped_obstacle_nudge",
    "lane_change",
    "curve",
    "green_light_go_straight",
    "intersection_other",
    "slow_decel_other",
    "keep_lane_straight",
    "other",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "due",
    "for",
    "from",
    "in",
    "is",
    "it",
    "lane",
    "of",
    "our",
    "since",
    "the",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--corpus-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collapse-threshold-m", type=float, default=0.25)
    parser.add_argument("--low-diversity-threshold-m", type=float, default=1.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return None
    return float(np.percentile(finite, q))


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
    }


def normalize_text(text: str | None) -> str:
    text = str(text or "")
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"<i\d+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def has(text: str, *subs: str) -> bool:
    lowered = text.lower()
    return any(sub in lowered for sub in subs)


def scene_bucket_from_text(text: str | None) -> str:
    t = normalize_text(text).lower()
    priority = [
        ("traffic_right_turn", lambda x: has(x, "traffic light", "green light", "red light") and has(x, "turn right", "right turn")),
        ("traffic_left_turn", lambda x: has(x, "traffic light", "green light", "red light") and has(x, "turn left", "left turn")),
        ("right_turn_no_light", lambda x: has(x, "turn right", "right turn")),
        ("left_turn_no_light", lambda x: has(x, "turn left", "left turn")),
        ("red_light_stop", lambda x: has(x, "red light", "light is red", "traffic light is red")),
        ("stop_sign", lambda x: has(x, "stop sign", "all-way stop")),
        ("pedestrian_crosswalk", lambda x: has(x, "pedestrian", "crosswalk")),
        ("cut_in_merge_yield", lambda x: has(x, "cut-in", "cut in", "merge", "merges into our lane")),
        ("lead_vehicle_follow", lambda x: has(x, "lead vehicle", "directly ahead in our lane", "vehicle ahead", "follow the vehicle")),
        ("parked_stopped_obstacle_nudge", lambda x: has(x, "nudge", "parked car", "parked vehicle", "parked cars", "stopped vehicle", "blocking")),
        ("lane_change", lambda x: has(x, "lane change", "change lane", "change lanes")),
        ("curve", lambda x: has(x, "curve", "curvature", "bends left", "bends right", "bend left", "bend right")),
        ("green_light_go_straight", lambda x: has(x, "green light", "light is green", "traffic light is green")),
        ("intersection_other", lambda x: has(x, "intersection")),
        ("slow_decel_other", lambda x: has(x, "slow down", "decelerate", "deceleration", "slow", "adapt speed")),
        ("keep_lane_straight", lambda x: has(x, "keep lane", "lane is clear", "keep speed", "straight")),
    ]
    for name, fn in priority:
        if fn(t):
            return name
    return "other"


def action_from_text(text: str | None) -> str:
    t = normalize_text(text).lower()
    if has(t, "right curve", "bends right", "bend right"):
        return "curve_right"
    if has(t, "left curve", "bends left", "bend left"):
        return "curve_left"
    if has(t, "lane change to the left", "change to the left", "change lanes to the left", "merge left"):
        return "change_lane_left"
    if has(t, "lane change to the right", "change to the right", "change lanes to the right", "merge right"):
        return "change_lane_right"
    if has(t, "nudge left", "move left", "shift left", "veer left"):
        return "nudge_left"
    if has(t, "nudge right", "move right", "shift right", "veer right"):
        return "nudge_right"
    if has(t, "turn left", "left turn"):
        return "left_turn"
    if has(t, "turn right", "right turn"):
        return "right_turn"
    if has(t, "stop sign", "red light", "stop for", "stop due", "stop to", "stop since", "full stop"):
        return "stop"
    if has(t, "yield"):
        return "yield"
    if has(t, "creep"):
        return "creep"
    if has(t, "lead vehicle", "vehicle ahead", "directly ahead in our lane", "keep distance"):
        return "follow_lead"
    if has(t, "slow down", "decelerate", "deceleration", "reduce speed", "adapt speed"):
        return "slow_down"
    if has(t, "keep lane", "lane is clear", "continue", "proceed", "accelerate", "keep speed"):
        return "lane_keep"
    return "unknown"


def action_family(action: str) -> str:
    if action in {"curve_left", "curve_right"}:
        return "curve"
    if action in {"change_lane_left", "change_lane_right", "nudge_left", "nudge_right"}:
        return "lateral_shift"
    if action in {"left_turn", "right_turn"}:
        return "turn"
    if action in {"slow_down", "yield", "creep", "stop"}:
        return "slow_stop_yield"
    if action in {"follow_lead"}:
        return "lead_vehicle_follow"
    if action in {"lane_keep"}:
        return "lane_keep"
    return "unknown"


def direction_conflict(left: str, right: str) -> bool:
    directional = (
        ("left", "right"),
        ("right", "left"),
    )
    for a, b in directional:
        if a in left and b in right:
            return True
    return False


def content_tokens(text: str | None) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9_-]+", normalize_text(text).lower())
    return {word for word in words if len(word) >= 3 and word not in STOPWORDS}


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    return float(len(left & right) / max(len(left | right), 1))


def coc_causal_score(text: str | None) -> dict[str, Any]:
    clean = normalize_text(text)
    lowered = clean.lower()
    words = re.findall(r"[a-z0-9_'-]+", lowered)
    action = action_from_text(clean)
    has_connector = bool(re.search(r"\b(since|because|due to|as|for|so|therefore|to|when|after|while)\b", lowered))
    evidence_keywords = (
        "vehicle",
        "lead",
        "pedestrian",
        "cyclist",
        "cone",
        "construction",
        "worker",
        "traffic",
        "red light",
        "green light",
        "stop sign",
        "crosswalk",
        "lane",
        "curve",
        "bends",
        "blocked",
        "gap",
        "roundabout",
        "intersection",
        "speed bump",
    )
    has_evidence = has(lowered, *evidence_keywords)
    clean_format = not bool(re.search(r"<\|[^>]+?\|>|<i\d+>", str(text or "")))
    score = (
        0.20 * float(len(words) >= 4)
        + 0.25 * float(action != "unknown")
        + 0.25 * float(has_connector)
        + 0.25 * float(has_evidence)
        + 0.05 * float(clean_format)
    )
    return {
        "score": float(score),
        "word_count": int(len(words)),
        "action": action,
        "has_connector": bool(has_connector),
        "has_evidence": bool(has_evidence),
        "clean_format": bool(clean_format),
    }


def agreement(student: str | None, reference: str | None) -> dict[str, Any]:
    student_action = action_from_text(student)
    reference_action = action_from_text(reference)
    student_family = action_family(student_action)
    reference_family = action_family(reference_action)
    student_bucket = scene_bucket_from_text(student)
    reference_bucket = scene_bucket_from_text(reference)
    exact = student_action != "unknown" and student_action == reference_action
    family = student_family != "unknown" and student_family == reference_family
    bucket = student_bucket == reference_bucket
    conflict = direction_conflict(student_action, reference_action)
    token_jaccard = jaccard(content_tokens(student), content_tokens(reference))
    if exact:
        score = 1.0
    elif family and not conflict:
        score = 0.75
    elif bucket and not conflict:
        score = 0.6
    elif conflict:
        score = 0.0
    else:
        score = 0.25 if (token_jaccard or 0.0) >= 0.2 else 0.0
    return {
        "score": float(score),
        "exact_action_match": bool(exact),
        "family_match": bool(family and not conflict),
        "scene_bucket_match": bool(bucket),
        "direction_conflict": bool(conflict),
        "student_action": student_action,
        "reference_action": reference_action,
        "student_family": student_family,
        "reference_family": reference_family,
        "student_bucket": student_bucket,
        "reference_bucket": reference_bucket,
        "content_jaccard": token_jaccard,
    }


def pairwise_distances(paths: list[np.ndarray]) -> tuple[list[float], list[float]]:
    ade_values: list[float] = []
    fde_values: list[float] = []
    for left_idx in range(len(paths)):
        for right_idx in range(left_idx + 1, len(paths)):
            left = paths[left_idx]
            right = paths[right_idx]
            n = min(int(left.shape[0]), int(right.shape[0]))
            if n <= 0:
                continue
            dist = np.linalg.norm(left[:n, :2] - right[:n, :2], axis=-1)
            ade_values.append(float(dist.mean()))
            fde_values.append(float(dist[-1]))
    return ade_values, fde_values


def by_category_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("category") or "unknown")].append(record)
    out: dict[str, Any] = {}
    for category in sorted(grouped):
        rows = grouped[category]
        out[category] = summarize_records(rows)
    return out


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairwise_mean = [float(row["pairwise_mean_ade_m"]) for row in records if row.get("pairwise_mean_ade_m") is not None]
    pairwise_min = [float(row["pairwise_min_ade_m"]) for row in records if row.get("pairwise_min_ade_m") is not None]
    pairwise_fde = [float(row["pairwise_mean_fde_m"]) for row in records if row.get("pairwise_mean_fde_m") is not None]
    causal = [float(row["coc_causal_score"]) for row in records]
    teacher_agree = [float(row["teacher_agreement_score"]) for row in records]
    human_agree = [float(row["human_agreement_score"]) for row in records]
    return {
        "count": int(len(records)),
        "pairwise_mean_ade_m": stats(pairwise_mean),
        "pairwise_min_ade_m": stats(pairwise_min),
        "pairwise_mean_fde_m": stats(pairwise_fde),
        "collapse_rate_pairwise_mean_lt_0p25": float(sum(float(v) < 0.25 for v in pairwise_mean) / max(len(pairwise_mean), 1)),
        "low_diversity_rate_pairwise_mean_lt_1m": float(sum(float(v) < 1.0 for v in pairwise_mean) / max(len(pairwise_mean), 1)),
        "coc_causal_score": stats(causal),
        "teacher_agreement_score": stats(teacher_agree),
        "human_agreement_score": stats(human_agree),
        "teacher_exact_action_match_rate": float(sum(bool(row["teacher_exact_action_match"]) for row in records) / max(len(records), 1)),
        "teacher_family_match_rate": float(sum(bool(row["teacher_family_match"]) for row in records) / max(len(records), 1)),
        "teacher_scene_bucket_match_rate": float(sum(bool(row["teacher_scene_bucket_match"]) for row in records) / max(len(records), 1)),
        "teacher_direction_conflict_rate": float(sum(bool(row["teacher_direction_conflict"]) for row in records) / max(len(records), 1)),
    }


def make_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# RUN2 Pairwise Diversity + CoC Reasoning Audit",
        "",
        f"- Decode summary: `{summary['input_summary_json']}`",
        f"- Corpus: `{summary['input_corpus_jsonl']}`",
        f"- Model checkpoint: `{summary['checkpoint_dir']}`",
        f"- Samples: `{summary['num_samples']}`; candidates per sample: `{summary['samples_per_row']}`",
        f"- Sampling: temperature `{summary['temperature']}`, top_p `{summary['top_p']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| pairwise mean ADE, mean | {overall['pairwise_mean_ade_m']['mean']:.4f} m |",
        f"| pairwise mean ADE, p50 | {overall['pairwise_mean_ade_m']['p50']:.4f} m |",
        f"| pairwise min ADE, p50 | {overall['pairwise_min_ade_m']['p50']:.4f} m |",
        f"| pairwise mean FDE, mean | {overall['pairwise_mean_fde_m']['mean']:.4f} m |",
        f"| collapse rate, pairwise mean < 0.25 m | {overall['collapse_rate_pairwise_mean_lt_0p25']:.4%} |",
        f"| low diversity rate, pairwise mean < 1.0 m | {overall['low_diversity_rate_pairwise_mean_lt_1m']:.4%} |",
        f"| CoC causal score, mean | {overall['coc_causal_score']['mean']:.4f} |",
        f"| teacher agreement score, mean | {overall['teacher_agreement_score']['mean']:.4f} |",
        f"| human agreement score, mean | {overall['human_agreement_score']['mean']:.4f} |",
        f"| teacher exact action match | {overall['teacher_exact_action_match_rate']:.4%} |",
        f"| teacher family match | {overall['teacher_family_match_rate']:.4%} |",
        f"| teacher scene bucket match | {overall['teacher_scene_bucket_match_rate']:.4%} |",
        f"| teacher direction conflict | {overall['teacher_direction_conflict_rate']:.4%} |",
        "",
        "## Category Highlights",
        "",
        "| Category | Count | Pairwise ADE mean | Collapse <0.25m | Causal | Teacher agree | Direction conflict |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, payload in summary["by_category"].items():
        lines.append(
            "| "
            + f"{category} | {payload['count']} | {payload['pairwise_mean_ade_m']['mean']:.3f} | "
            + f"{payload['collapse_rate_pairwise_mean_lt_0p25']:.2%} | "
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

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    corpus_rows = read_jsonl(args.corpus_jsonl)
    corpus_by_id = {str(row.get("sample_id")): row for row in corpus_rows}
    decoder = TrajectoryTokenDecoder(config_path=Path(summary["traj_tokenizer_config"]))

    per_sample: list[dict[str, Any]] = []
    missing_rows: list[str] = []
    for sample_record in summary["samples"]:
        sample_id = str(sample_record["sample_id"])
        corpus_row = corpus_by_id.get(sample_id)
        if corpus_row is None:
            missing_rows.append(sample_id)
            continue
        history_xyz = load_ego_history_xyz(corpus_row, PROJECT_ROOT)
        history_rot = load_ego_history_rot(corpus_row, PROJECT_ROOT)
        decoded_paths = []
        token_candidates = sample_record.get("student_free_run_candidate_records") or []
        for candidate in token_candidates:
            tokens = list(candidate.get("student_free_run_traj_tokens") or [])
            if len(tokens) != decoder.n_waypoints * 2:
                continue
            xyz = decoder.decode(history_xyz, history_rot, tokens)
            if xyz is not None and int(xyz.shape[0]) > 0:
                decoded_paths.append(xyz)

        pair_ade, pair_fde = pairwise_distances(decoded_paths)
        category = str((corpus_row.get("metadata") or {}).get("semantic_scene_category") or scene_bucket_from_text(sample_record.get("teacher_cot")))
        student_cot = normalize_text(sample_record.get("student_cot"))
        teacher_cot = normalize_text(sample_record.get("teacher_cot"))
        human_coc = normalize_text(sample_record.get("human_coc"))
        causal = coc_causal_score(student_cot)
        teacher = agreement(student_cot, teacher_cot)
        human = agreement(student_cot, human_coc)
        row = {
            "sample_id": sample_id,
            "category": category,
            "candidate_count": int(len(token_candidates)),
            "decoded_candidate_count": int(len(decoded_paths)),
            "pairwise_pair_count": int(len(pair_ade)),
            "pairwise_mean_ade_m": mean(pair_ade),
            "pairwise_min_ade_m": min(pair_ade) if pair_ade else None,
            "pairwise_max_ade_m": max(pair_ade) if pair_ade else None,
            "pairwise_mean_fde_m": mean(pair_fde),
            "pairwise_min_fde_m": min(pair_fde) if pair_fde else None,
            "pairwise_max_fde_m": max(pair_fde) if pair_fde else None,
            "student_cot": student_cot,
            "teacher_cot": teacher_cot,
            "human_coc": human_coc,
            "coc_causal_score": causal["score"],
            "coc_causal_details": causal,
            "teacher_agreement_score": teacher["score"],
            "teacher_exact_action_match": teacher["exact_action_match"],
            "teacher_family_match": teacher["family_match"],
            "teacher_scene_bucket_match": teacher["scene_bucket_match"],
            "teacher_direction_conflict": teacher["direction_conflict"],
            "teacher_agreement_details": teacher,
            "human_agreement_score": human["score"],
            "human_agreement_details": human,
        }
        per_sample.append(row)

    overall = summarize_records(per_sample)
    worst_teacher = sorted(
        per_sample,
        key=lambda row: (
            float(row["teacher_agreement_score"]),
            -(float(row["pairwise_mean_ade_m"]) if row.get("pairwise_mean_ade_m") is not None else -1.0),
        ),
    )[:20]
    best_teacher = sorted(
        per_sample,
        key=lambda row: (
            -float(row["teacher_agreement_score"]),
            float(row["pairwise_mean_ade_m"]) if row.get("pairwise_mean_ade_m") is not None else 1e9,
        ),
    )[:20]
    output = {
        "input_summary_json": str(args.summary_json),
        "input_corpus_jsonl": str(args.corpus_jsonl),
        "checkpoint_dir": summary.get("checkpoint_dir"),
        "num_samples": int(len(per_sample)),
        "missing_corpus_rows": missing_rows,
        "samples_per_row": int(summary.get("samples_per_row") or 0),
        "temperature": summary.get("temperature"),
        "top_p": summary.get("top_p"),
        "collapse_threshold_m": float(args.collapse_threshold_m),
        "low_diversity_threshold_m": float(args.low_diversity_threshold_m),
        "overall": overall,
        "by_category": by_category_summary(per_sample),
        "student_action_counts": dict(Counter(row["coc_causal_details"]["action"] for row in per_sample).most_common()),
        "teacher_action_counts": dict(Counter(row["teacher_agreement_details"]["reference_action"] for row in per_sample).most_common()),
        "student_bucket_counts": dict(Counter(row["teacher_agreement_details"]["student_bucket"] for row in per_sample).most_common()),
        "teacher_bucket_counts": dict(Counter(row["teacher_agreement_details"]["reference_bucket"] for row in per_sample).most_common()),
        "worst_teacher_agreement_examples": worst_teacher,
        "best_teacher_agreement_examples": best_teacher,
        "samples": per_sample,
    }
    out_json = args.output_dir / "summary.json"
    out_md = args.output_dir / "report.md"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(make_markdown(output), encoding="utf-8")
    print(json.dumps({"summary_json": str(out_json), "report_md": str(out_md), "overall": overall}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
