#!/usr/bin/env python3
"""Compare a decode summary against external free-run trajectory token targets."""

from __future__ import annotations

import argparse
import json
import math
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
from src.training.collator import load_ego_history_xyz  # noqa: E402
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-summary", type=Path, required=True)
    parser.add_argument("--target-summary", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path, split: str, num_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("split") or "") == split:
                rows.append(row)
            if num_samples > 0 and len(rows) >= num_samples:
                break
    return rows


def sample_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for sample in summary.get("samples") or []:
        sample_id = sample.get("sample_id")
        if sample_id:
            out[str(sample_id)] = sample
    return out


def finite_mean(values: list[float]) -> float | None:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def ade_fde(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def max_same_token_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    best = current = 1
    for left, right in zip(tokens, tokens[1:]):
        if int(left) == int(right):
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def extract_tokens(sample: dict[str, Any]) -> list[int]:
    tokens = sample.get("generated_traj_tokens")
    if tokens is None:
        tokens = sample.get("student_free_run_traj_tokens")
    if tokens is None:
        records = sample.get("student_free_run_candidate_records") or []
        if records:
            tokens = records[0].get("student_free_run_traj_tokens")
    return [int(token) for token in (tokens or [])]


def main() -> None:
    args = parse_args()
    decode_summary = load_json(args.decode_summary)
    target_summary = load_json(args.target_summary)
    decode_by_id = sample_map(decode_summary)
    target_by_id = sample_map(target_summary)
    rows = load_rows(args.corpus_jsonl, args.split, int(args.num_samples))

    decoder_path = resolve_traj_tokenizer_config_path(str(args.student_model))
    if decoder_path is None:
        raise SystemExit("Could not resolve trajectory tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=decoder_path)

    per_sample: list[dict[str, Any]] = []
    token_matches: list[float] = []
    ades: list[float] = []
    fdes: list[float] = []
    unique_counts: list[float] = []
    max_runs: list[float] = []
    exact_128 = 0
    compared = 0

    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        decoded = decode_by_id.get(sample_id)
        target = target_by_id.get(sample_id)
        if decoded is None or target is None:
            per_sample.append({"sample_id": sample_id, "status": "missing"})
            continue
        pred_tokens = extract_tokens(decoded)
        target_tokens = extract_tokens(target)
        usable = min(len(pred_tokens), len(target_tokens), decoder.n_waypoints * 2)
        if usable <= 0:
            per_sample.append({"sample_id": sample_id, "status": "empty_tokens"})
            continue
        match = sum(
            1 for left, right in zip(pred_tokens[:usable], target_tokens[:usable]) if int(left) == int(right)
        ) / float(max(usable, 1))
        compared += 1
        token_matches.append(float(match))
        unique_counts.append(float(len(set(pred_tokens))))
        max_runs.append(float(max_same_token_run(pred_tokens)))
        if len(pred_tokens) == decoder.n_waypoints * 2:
            exact_128 += 1

        target_ade = float("nan")
        target_fde = float("nan")
        if len(pred_tokens) == decoder.n_waypoints * 2 and len(target_tokens) == decoder.n_waypoints * 2:
            history_xyz = load_ego_history_xyz(row, PROJECT_ROOT)
            history_rot = load_ego_history_rot(row, PROJECT_ROOT)
            pred_xyz = decoder.decode(history_xyz, history_rot, pred_tokens)
            target_xyz = decoder.decode(history_xyz, history_rot, target_tokens)
            target_ade, target_fde = ade_fde(pred_xyz, target_xyz)
            ades.append(target_ade)
            fdes.append(target_fde)

        per_sample.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                "target_token_match_rate": float(match),
                "target_ade_m": target_ade,
                "target_fde_m": target_fde,
                "generated_token_count": len(pred_tokens),
                "target_token_count": len(target_tokens),
                "generated_unique_token_count": len(set(pred_tokens)),
                "generated_max_same_token_run": max_same_token_run(pred_tokens),
            }
        )

    summary = {
        "decode_summary": str(args.decode_summary),
        "target_summary": str(args.target_summary),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": str(args.split),
        "requested_samples": int(args.num_samples),
        "selected_samples": len(rows),
        "compared_samples": compared,
        "exact_128_rate": float(exact_128 / max(compared, 1)),
        "avg_target_token_match_rate": finite_mean(token_matches),
        "avg_target_ade_m": finite_mean(ades),
        "avg_target_fde_m": finite_mean(fdes),
        "avg_generated_unique_token_count": finite_mean(unique_counts),
        "avg_generated_max_same_token_run": finite_mean(max_runs),
        "samples": per_sample,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, ensure_ascii=True))


if __name__ == "__main__":
    main()
