#!/usr/bin/env python3
"""Build teacher-vs-GT quality tables for KD weighting.

The table answers one practical question: for each sample, how much should we
trust the trajectory teacher for KD relative to the GT trajectory target?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import (  # noqa: E402
    TrajectoryTokenDecoder,
    load_ego_history_rot,
    resolve_traj_tokenizer_config_path,
)
from src.training.collator import load_ego_future_xyz, load_ego_history_xyz  # noqa: E402
from src.utils.runtime_paths import remap_external_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2_959.jsonl",
    )
    parser.add_argument(
        "--student-summary-json",
        type=Path,
        action="append",
        default=[],
        help="B0 decode summary JSON. Can be passed multiple times for train/val.",
    )
    parser.add_argument(
        "--teacher-traj-tokens-dir",
        type=Path,
        default=Path("/data/teacher_cache/traj15/tokens"),
    )
    parser.add_argument(
        "--teacher-traj-manifest-dir",
        type=Path,
        default=Path("/data/teacher_cache/traj15/manifest"),
    )
    parser.add_argument("--traj-tokenizer-config", type=Path, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_vs_gt_quality" / "teacher_vs_gt_quality_table.csv",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_vs_gt_quality" / "teacher_vs_gt_quality_table.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_vs_gt_quality" / "teacher_vs_gt_quality_summary.json",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_vs_gt_quality" / "README.md",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_path(raw_path: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def _teacher_token_path(sample_id: str, tokens_dir: Path) -> Path:
    return tokens_dir / f"{sample_id}.teacher_traj15.tokens.npy"


def _load_teacher_manifest_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for item in path.glob("*.manifest.json"):
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        sample_id = payload.get("sample_id")
        if sample_id:
            out[str(sample_id)] = payload
    return out


def _load_student_samples(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            print(json.dumps({"event": "missing_student_summary", "path": str(path)}))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for sample in payload.get("samples") or []:
            sample_id = str(sample.get("sample_id") or "")
            if sample_id:
                out[sample_id] = sample
    return out


def _ade_fde(pred: np.ndarray | None, target: np.ndarray | None) -> tuple[float, float]:
    if pred is None or target is None or pred.shape[0] == 0 or target.shape[0] == 0:
        return float("nan"), float("nan")
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def _path_len(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=-1).sum())


def _final_speed(xyz: np.ndarray, dt: float = 0.1) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(xyz[-1, :2] - xyz[-2, :2]) / dt)


def _direction_cosine(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return float("nan")
    pred_vec = pred[-1, :2] - pred[0, :2]
    gt_vec = gt[-1, :2] - gt[0, :2]
    denom = float(np.linalg.norm(pred_vec) * np.linalg.norm(gt_vec))
    if denom < 1e-6:
        return float("nan")
    return float(np.dot(pred_vec, gt_vec) / denom)


def _max_same_token_run(token_ids: list[int]) -> int:
    if not token_ids:
        return 0
    best = current = 1
    for left, right in zip(token_ids, token_ids[1:]):
        if left == right:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _token_repetition_stats(tokens: list[int]) -> dict[str, Any]:
    counter = Counter(tokens)
    total = max(len(tokens), 1)
    top = counter.most_common(2)
    return {
        "unique": len(counter),
        "max_same_run": _max_same_token_run(tokens),
        "top2_mass": float(sum(count for _, count in top) / total) if top else 0.0,
    }


def _teacher_time_metrics(teacher_xyz: np.ndarray | None, gt_xyz: np.ndarray) -> dict[str, float]:
    if teacher_xyz is None or teacher_xyz.shape[0] == 0 or gt_xyz.shape[0] == 0:
        return {
            "teacher_ADE_to_GT": float("nan"),
            "teacher_FDE_to_GT": float("nan"),
            "teacher_early_ADE": float("nan"),
            "teacher_early_FDE": float("nan"),
            "teacher_late_ADE": float("nan"),
            "teacher_path_length_ratio": float("nan"),
            "gt_final_speed_mps": float("nan"),
            "teacher_final_speed_mps": float("nan"),
            "gt_final_y_m": float("nan"),
            "teacher_final_y_m": float("nan"),
            "teacher_final_lateral_error_m": float("nan"),
            "teacher_direction_cosine": float("nan"),
        }
    n = min(int(teacher_xyz.shape[0]), int(gt_xyz.shape[0]))
    teacher = teacher_xyz[:n]
    gt = gt_xyz[:n]
    ade, fde = _ade_fde(teacher, gt)
    early_n = min(20, n)
    late_start = min(20, max(n - 1, 0))
    early_ade, early_fde = _ade_fde(teacher[:early_n], gt[:early_n])
    late_ade, _ = _ade_fde(teacher[late_start:n], gt[late_start:n])
    return {
        "teacher_ADE_to_GT": ade,
        "teacher_FDE_to_GT": fde,
        "teacher_early_ADE": early_ade,
        "teacher_early_FDE": early_fde,
        "teacher_late_ADE": late_ade,
        "teacher_path_length_ratio": float(_path_len(teacher) / max(_path_len(gt), 1e-6)),
        "gt_final_speed_mps": _final_speed(gt),
        "teacher_final_speed_mps": _final_speed(teacher),
        "gt_final_y_m": float(gt[-1, 1]),
        "teacher_final_y_m": float(teacher[-1, 1]),
        "teacher_final_lateral_error_m": float(abs(teacher[-1, 1] - gt[-1, 1])),
        "teacher_direction_cosine": _direction_cosine(teacher, gt),
    }


def _teacher_failure_tags(
    *,
    sample: dict[str, Any],
    teacher_tokens: list[int],
    teacher_metrics: dict[str, float],
) -> list[str]:
    """Apply the same lightweight geometry triage rules to teacher-vs-GT."""
    tags: list[str] = []
    invalid_count = sum(1 for token in teacher_tokens if token < 0 or token >= 3000)
    rep = _token_repetition_stats(teacher_tokens)
    gt_motion = str((sample.get("derived") or {}).get("gt_motion_class") or "").lower()

    if len(teacher_tokens) != 128:
        tags.append("invalid_token_count")
    if invalid_count > 0:
        tags.append("invalid_future_token_i3000_plus")
    if rep["max_same_run"] >= 8 or rep["unique"] <= 8 or rep["top2_mass"] >= 0.85:
        tags.append("F_repetition_or_local_band_oscillation")

    ade = teacher_metrics["teacher_ADE_to_GT"]
    fde = teacher_metrics["teacher_FDE_to_GT"]
    if not math.isfinite(ade) or not math.isfinite(fde):
        tags.append("no_decoded_geometry")
        return tags

    early_ade = teacher_metrics["teacher_early_ADE"]
    early_fde = teacher_metrics["teacher_early_FDE"]
    late_ade = teacher_metrics["teacher_late_ADE"]
    ratio = teacher_metrics["teacher_path_length_ratio"]
    direction_cosine = teacher_metrics["teacher_direction_cosine"]
    gt_final_speed = teacher_metrics["gt_final_speed_mps"]
    teacher_final_speed = teacher_metrics["teacher_final_speed_mps"]
    gt_final_y = teacher_metrics["gt_final_y_m"]
    teacher_final_y = teacher_metrics["teacher_final_y_m"]
    lateral_error = teacher_metrics["teacher_final_lateral_error_m"]

    stop_like_gt = gt_motion in {"stop", "stopping", "decelerate", "slow"} or gt_final_speed < 0.75
    if stop_like_gt and teacher_final_speed > 1.5 and ratio > 1.25:
        tags.append("A_stop_or_decel_failure")
    if abs(gt_final_y) > 1.0 and abs(teacher_final_y) > 1.0 and gt_final_y * teacher_final_y < 0.0:
        tags.append("B_curvature_or_turn_direction_failure")
    elif lateral_error > 2.5 and ade > 2.0:
        tags.append("B_curvature_or_lateral_failure")
    if math.isfinite(direction_cosine) and direction_cosine > 0.75 and (ratio > 1.35 or ratio < 0.65):
        tags.append("C_speed_scale_failure")
    if early_ade > 2.0 or early_fde > 3.0:
        tags.append("D_initial_prefix_failure")
    if early_ade <= 2.0 and (late_ade > max(2.5, early_ade * 2.0) or fde > 6.0):
        tags.append("E_long_horizon_divergence")
    if not tags and (ade > 2.0 or fde > 6.0):
        tags.append("unclassified_geometry_error")
    if not tags:
        tags.append("ok_or_low_error")
    return tags


def _valid_token_match(teacher_tokens: list[int], gt_tokens: list[int]) -> float:
    n = min(len(teacher_tokens), len(gt_tokens))
    if n <= 0:
        return float("nan")
    valid_positions = [
        idx
        for idx in range(n)
        if 0 <= int(teacher_tokens[idx]) < 3000 and 0 <= int(gt_tokens[idx]) < 3000
    ]
    if not valid_positions:
        return float("nan")
    matches = sum(1 for idx in valid_positions if int(teacher_tokens[idx]) == int(gt_tokens[idx]))
    return float(matches / len(valid_positions))


def _teacher_abs_bin(*, teacher_ade: float, teacher_fde: float) -> str:
    """Bucket teacher quality using only teacher-vs-GT geometry."""
    if not (math.isfinite(teacher_ade) and math.isfinite(teacher_fde)):
        return "C_bad"
    if teacher_ade <= 2.0 and teacher_fde <= 6.0:
        return "A_good"
    if teacher_ade <= 3.5 and teacher_fde <= 9.0:
        return "B_ok"
    if teacher_ade > 5.0 or teacher_fde > 12.0:
        return "C_bad"
    return "D_middle"


def _abs_bin_kd_weight(teacher_bin: str) -> float:
    """Conservative starting weights for teacher trajectory KD by absolute quality."""
    return {
        "A_good": 1.0,
        "B_ok": 0.5,
        "D_middle": 0.25,
        "C_bad": 0.0,
    }.get(teacher_bin, 0.0)


def _relative_group_and_weight(
    *,
    teacher_ade: float,
    teacher_fde: float,
    student_ade: float,
    student_fde: float,
    teacher_invalid_rate: float,
) -> tuple[str, float]:
    finite_teacher = math.isfinite(teacher_ade) and math.isfinite(teacher_fde)
    finite_student = math.isfinite(student_ade) and math.isfinite(student_fde)
    if teacher_invalid_rate > 0.0 or not finite_teacher:
        return "group3_teacher_downweight_or_exclude", 0.0
    if (
        teacher_ade > 5.0
        or teacher_fde > 12.0
        or (
            finite_student
            and teacher_ade >= student_ade + 0.5
            and teacher_fde >= student_fde + 1.5
        )
    ):
        return "group3_teacher_downweight_or_exclude", 0.0
    if (
        finite_student
        and teacher_ade <= 2.0
        and teacher_fde <= 6.0
        and (
            teacher_ade <= student_ade - 0.75
            or teacher_ade <= 0.80 * student_ade
            or teacher_fde <= student_fde - 2.0
        )
    ):
        return "group1_teacher_trusted", 1.0
    return "group2_mixed_gt_primary", 0.4


def _mean(values: list[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relative_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_split[str(rec["split"])].append(rec)
        by_group[str(rec["group"])].append(rec)
        by_relative_group[str(rec["relative_kd_group"])].append(rec)

    def pack(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(row["group"]) for row in rows)
        relative_counts = Counter(str(row["relative_kd_group"]) for row in rows)
        tag_counts: Counter[str] = Counter()
        for row in rows:
            tag_counts.update(tag for tag in str(row.get("teacher_failure_tags") or "").split(";") if tag)
        prefix_count = sum(1 for row in rows if bool(row["prefix_teacher_good"]))
        return {
            "num_samples": len(rows),
            "group_counts": dict(counts),
            "relative_kd_group_counts": dict(relative_counts),
            "teacher_failure_tag_counts": dict(tag_counts.most_common()),
            "prefix_teacher_good_count": prefix_count,
            "mean_teacher_ADE_to_GT": _mean([row["teacher_ADE_to_GT"] for row in rows]),
            "mean_teacher_FDE_to_GT": _mean([row["teacher_FDE_to_GT"] for row in rows]),
            "mean_student_ADE_to_GT": _mean([row["student_ADE_to_GT"] for row in rows]),
            "mean_student_FDE_to_GT": _mean([row["student_FDE_to_GT"] for row in rows]),
            "mean_teacher_minus_student_ADE": _mean([row["teacher_minus_student_ADE"] for row in rows]),
            "mean_teacher_minus_student_FDE": _mean([row["teacher_minus_student_FDE"] for row in rows]),
            "mean_teacher_token_match_3000": _mean([row["teacher_token_match_3000"] for row in rows]),
        }

    return {
        "overall": pack(records),
        "by_split": {split: pack(rows) for split, rows in sorted(by_split.items())},
        "by_group": {group: pack(rows) for group, rows in sorted(by_group.items())},
        "by_relative_kd_group": {
            group: pack(rows) for group, rows in sorted(by_relative_group.items())
        },
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "teacher_ADE_to_GT",
        "teacher_FDE_to_GT",
        "student_ADE_to_GT",
        "student_FDE_to_GT",
        "student_vs_teacher_ADE",
        "student_vs_teacher_FDE",
        "teacher_minus_student_ADE",
        "teacher_minus_student_FDE",
        "teacher_token_match_3000",
        "teacher_invalid_i3000plus_rate",
        "teacher_early_ADE",
        "teacher_early_FDE",
        "teacher_late_ADE",
        "teacher_path_length_ratio",
        "gt_final_speed_mps",
        "teacher_final_speed_mps",
        "gt_final_y_m",
        "teacher_final_y_m",
        "teacher_final_lateral_error_m",
        "teacher_direction_cosine",
        "teacher_failure_tags",
        "teacher_abs_bin",
        "group",
        "kd_weight",
        "relative_kd_group",
        "relative_kd_weight",
        "prefix_teacher_good",
        "teacher_quality_bucket",
        "teacher_unique_ids",
        "teacher_max_same_token_run",
        "student_summary_available",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({name: rec.get(name) for name in fieldnames})


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, summary: dict[str, Any], *, csv_path: Path, jsonl_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Teacher vs GT Quality Table",
        "",
        f"CSV: `{csv_path}`",
        f"JSONL: `{jsonl_path}`",
        "",
        "## Group Policy",
        "",
        "- `group` is now the teacher absolute geometry bin, also duplicated as `teacher_abs_bin`.",
        "- `A_good`: teacher ADE <= 2.0 and teacher FDE <= 6.0. Starting `kd_weight=1.0`.",
        "- `B_ok`: teacher ADE <= 3.5 and teacher FDE <= 9.0. Starting `kd_weight=0.5`.",
        "- `C_bad`: teacher ADE > 5.0 or teacher FDE > 12.0. Starting `kd_weight=0.0`.",
        "- `D_middle`: all remaining samples. Starting `kd_weight=0.25`.",
        "- The older B0-relative trust split is preserved as `relative_kd_group` / `relative_kd_weight`.",
        "- `prefix_teacher_good` is true when teacher early ADE <= 1.0 and early FDE <= 2.0.",
        "",
        "## Summary",
        "",
    ]
    for title, block in (("overall", summary["overall"]), *[(f"split={k}", v) for k, v in summary["by_split"].items()]):
        lines.extend(
            [
                f"### {title}",
                "",
                f"- samples: `{block['num_samples']}`",
                f"- teacher_abs_bin counts: `{block['group_counts']}`",
                f"- relative_kd_group counts: `{block['relative_kd_group_counts']}`",
                f"- teacher failure tags: `{block['teacher_failure_tag_counts']}`",
                f"- prefix_teacher_good: `{block['prefix_teacher_good_count']}`",
                f"- mean teacher ADE/FDE: `{fmt(block['mean_teacher_ADE_to_GT'])}` / `{fmt(block['mean_teacher_FDE_to_GT'])}`",
                f"- mean student ADE/FDE: `{fmt(block['mean_student_ADE_to_GT'])}` / `{fmt(block['mean_student_FDE_to_GT'])}`",
                f"- mean teacher-student ADE/FDE delta: `{fmt(block['mean_teacher_minus_student_ADE'])}` / `{fmt(block['mean_teacher_minus_student_FDE'])}`",
                f"- mean teacher token match: `{fmt(block['mean_teacher_token_match_3000'])}`",
                "",
            ]
        )
    lines.extend(["## By Teacher Abs Bin", ""])
    for group, block in summary["by_group"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                f"- samples: `{block['num_samples']}`",
                f"- mean teacher ADE/FDE: `{fmt(block['mean_teacher_ADE_to_GT'])}` / `{fmt(block['mean_teacher_FDE_to_GT'])}`",
                f"- mean student ADE/FDE: `{fmt(block['mean_student_ADE_to_GT'])}` / `{fmt(block['mean_student_FDE_to_GT'])}`",
                f"- mean teacher-student ADE/FDE delta: `{fmt(block['mean_teacher_minus_student_ADE'])}` / `{fmt(block['mean_teacher_minus_student_FDE'])}`",
                f"- teacher failure tags: `{block['teacher_failure_tag_counts']}`",
                f"- prefix_teacher_good: `{block['prefix_teacher_good_count']}`",
                "",
            ]
        )
    lines.extend(["## By B0-Relative KD Group", ""])
    for group, block in summary["by_relative_kd_group"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                f"- samples: `{block['num_samples']}`",
                f"- mean teacher ADE/FDE: `{fmt(block['mean_teacher_ADE_to_GT'])}` / `{fmt(block['mean_teacher_FDE_to_GT'])}`",
                f"- mean student ADE/FDE: `{fmt(block['mean_student_ADE_to_GT'])}` / `{fmt(block['mean_student_FDE_to_GT'])}`",
                f"- mean teacher-student ADE/FDE delta: `{fmt(block['mean_teacher_minus_student_ADE'])}` / `{fmt(block['mean_teacher_minus_student_FDE'])}`",
                f"- teacher failure tags: `{block['teacher_failure_tag_counts']}`",
                f"- prefix_teacher_good: `{block['prefix_teacher_good_count']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = _load_jsonl(args.corpus_jsonl)
    student_by_id = _load_student_samples(args.student_summary_json)
    manifest_by_id = _load_teacher_manifest_map(args.teacher_traj_manifest_dir)
    config_path = args.traj_tokenizer_config or resolve_traj_tokenizer_config_path(None)
    if config_path is None:
        raise SystemExit("Could not resolve Alpamayo trajectory tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=config_path)

    records: list[dict[str, Any]] = []
    missing_teacher = 0
    missing_student = 0
    for sample in rows:
        sample_id = str(sample.get("sample_id") or "")
        split = str(sample.get("split") or "")
        if not sample_id:
            continue
        teacher_path = _teacher_token_path(sample_id, args.teacher_traj_tokens_dir)
        if not teacher_path.exists():
            missing_teacher += 1
            continue
        teacher_tokens = [int(value) for value in np.load(teacher_path).reshape(-1).tolist()]
        gt_tokens = [int(value) for value in (sample.get("hard_target") or {}).get("traj_future_token_ids") or []]
        teacher_invalid_count = sum(1 for token in teacher_tokens if token < 0 or token >= 3000)
        teacher_invalid_rate = float(teacher_invalid_count / max(len(teacher_tokens), 1))

        try:
            history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
            history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
            gt_xyz = load_ego_future_xyz(sample, PROJECT_ROOT)
            teacher_xyz = decoder.decode(history_xyz, history_rot, teacher_tokens)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"event": "sample_decode_failed", "sample_id": sample_id, "error": str(exc)}))
            missing_teacher += 1
            continue

        t_metrics = _teacher_time_metrics(teacher_xyz, gt_xyz)
        student_summary = student_by_id.get(sample_id)
        student_summary_available = student_summary is not None
        if not student_summary_available:
            missing_student += 1
        student_ade = float(student_summary.get("ade_m")) if student_summary and student_summary.get("ade_m") is not None else float("nan")
        student_fde = float(student_summary.get("fde_m")) if student_summary and student_summary.get("fde_m") is not None else float("nan")
        student_tokens = [int(tok) for tok in (student_summary or {}).get("generated_traj_tokens") or []]
        student_xyz = decoder.decode(history_xyz, history_rot, student_tokens) if len(student_tokens) == 128 else None
        student_vs_teacher_ade, student_vs_teacher_fde = _ade_fde(student_xyz, teacher_xyz)

        teacher_ade = float(t_metrics["teacher_ADE_to_GT"])
        teacher_fde = float(t_metrics["teacher_FDE_to_GT"])
        teacher_failure_tags = _teacher_failure_tags(
            sample=sample,
            teacher_tokens=teacher_tokens,
            teacher_metrics=t_metrics,
        )
        teacher_bin = _teacher_abs_bin(teacher_ade=teacher_ade, teacher_fde=teacher_fde)
        kd_weight = _abs_bin_kd_weight(teacher_bin)
        relative_group, relative_kd_weight = _relative_group_and_weight(
            teacher_ade=teacher_ade,
            teacher_fde=teacher_fde,
            student_ade=student_ade,
            student_fde=student_fde,
            teacher_invalid_rate=teacher_invalid_rate,
        )
        manifest = manifest_by_id.get(sample_id) or {}
        record = {
            "sample_id": sample_id,
            "split": split,
            **t_metrics,
            "student_ADE_to_GT": student_ade,
            "student_FDE_to_GT": student_fde,
            "student_vs_teacher_ADE": student_vs_teacher_ade,
            "student_vs_teacher_FDE": student_vs_teacher_fde,
            "teacher_minus_student_ADE": teacher_ade - student_ade if math.isfinite(student_ade) else float("nan"),
            "teacher_minus_student_FDE": teacher_fde - student_fde if math.isfinite(student_fde) else float("nan"),
            "teacher_token_match_3000": _valid_token_match(teacher_tokens, gt_tokens),
            "teacher_invalid_i3000plus_rate": teacher_invalid_rate,
            "teacher_failure_tags": ";".join(teacher_failure_tags),
            "teacher_abs_bin": teacher_bin,
            "group": teacher_bin,
            "kd_weight": kd_weight,
            "relative_kd_group": relative_group,
            "relative_kd_weight": relative_kd_weight,
            "prefix_teacher_good": bool(t_metrics["teacher_early_ADE"] <= 1.0 and t_metrics["teacher_early_FDE"] <= 2.0),
            "teacher_quality_bucket": manifest.get("quality_bucket"),
            "teacher_unique_ids": manifest.get("unique_ids"),
            "teacher_max_same_token_run": manifest.get("max_same_token_run"),
            "student_summary_available": student_summary_available,
        }
        records.append(record)

    records.sort(key=lambda row: (str(row["split"]), str(row["sample_id"])))
    summary = _summarize(records)
    summary["inputs"] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "student_summary_json": [str(path) for path in args.student_summary_json],
        "teacher_traj_tokens_dir": str(args.teacher_traj_tokens_dir),
        "teacher_traj_manifest_dir": str(args.teacher_traj_manifest_dir),
        "traj_tokenizer_config": str(config_path),
    }
    summary["missing"] = {
        "missing_teacher_samples": missing_teacher,
        "missing_student_summaries": missing_student,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_csv, records)
    _write_jsonl(args.output_jsonl, records)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.summary_md, summary, csv_path=args.output_csv, jsonl_path=args.output_jsonl)
    print(json.dumps({"event": "done", "records": len(records), "summary": str(args.summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
