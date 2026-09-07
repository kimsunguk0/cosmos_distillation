#!/usr/bin/env python3
"""Audit cached teacher text top-k targets.

This is the text analogue of the trajectory cache ceiling audit:
measure how often the sampled CE target is also the cached teacher top-1
under the same cached-prefix distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def rate(num: int, den: int) -> float:
    return float(num / den) if den else float("nan")


def read_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def load_text_arrays(target: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_path = Path(str(target.get("topk_ids_path") or ""))
    logprobs_path = Path(str(target.get("topk_logprobs_path") or ""))
    tokens_path = Path(str(target.get("target_token_ids_path") or ""))
    if not ids_path.exists() or not logprobs_path.exists() or not tokens_path.exists():
        raise FileNotFoundError("missing text top-k/token path")
    topk_ids = np.asarray(np.load(ids_path), dtype=np.int64)
    topk_logprobs = np.asarray(np.load(logprobs_path), dtype=np.float64)
    target_ids = np.asarray(np.load(tokens_path), dtype=np.int64).reshape(-1)
    if topk_ids.ndim != 2 or topk_logprobs.ndim != 2:
        raise ValueError("text top-k arrays must be rank-2")
    n = min(topk_ids.shape[0], topk_logprobs.shape[0], target_ids.shape[0])
    return topk_ids[:n], topk_logprobs[:n], target_ids[:n]


class Bucket:
    def __init__(self) -> None:
        self.rows = 0
        self.positions = 0
        self.top1 = 0
        self.in_topk = 0
        self.missing_topk = 0
        self.top1_probs: list[float] = []
        self.target_probs: list[float] = []
        self.topk_masses: list[float] = []
        self.tail_masses: list[float] = []
        self.token_counts: list[float] = []

    def add(
        self,
        *,
        target_ids: np.ndarray,
        topk_ids: np.ndarray,
        topk_logprobs: np.ndarray,
    ) -> None:
        self.rows += 1
        self.token_counts.append(float(target_ids.shape[0]))
        valid = np.isfinite(topk_logprobs) & (topk_logprobs > -1.0e8)
        probs = np.where(valid, np.exp(topk_logprobs), 0.0)
        masses = probs.sum(axis=1)
        self.topk_masses.extend(float(v) for v in masses.tolist())
        self.tail_masses.extend(float(max(0.0, 1.0 - v)) for v in masses.tolist())
        self.top1_probs.extend(float(v) for v in probs[:, 0].tolist())

        for i, target in enumerate(target_ids):
            self.positions += 1
            matches = np.flatnonzero(topk_ids[i] == int(target))
            if matches.size == 0:
                self.missing_topk += 1
                continue
            rank0 = int(matches[0])
            self.in_topk += 1
            self.target_probs.append(float(probs[i, rank0]))
            if rank0 == 0:
                self.top1 += 1

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "positions": self.positions,
            "target_is_top1_rate": rate(self.top1, self.positions),
            "target_in_topk_rate": rate(self.in_topk, self.positions),
            "target_missing_topk_rate": rate(self.missing_topk, self.positions),
            "top1_prob": quantiles(self.top1_probs),
            "target_prob_when_in_topk": quantiles(self.target_probs),
            "topk_prob_mass": quantiles(self.topk_masses),
            "tail_mass_1_minus_topk_sum": quantiles(self.tail_masses),
            "target_token_count": quantiles(self.token_counts),
        }


def main() -> int:
    args = parse_args()
    rows = read_rows(args.corpus_jsonl, args.max_rows)

    overall = Bucket()
    by_split: dict[str, Bucket] = defaultdict(Bucket)
    by_source: dict[str, Bucket] = defaultdict(Bucket)
    by_pos_from_start: dict[int, Bucket] = defaultdict(Bucket)
    by_pos_from_end: dict[int, Bucket] = defaultdict(Bucket)
    token_count_hist = Counter()
    missing_rows = 0
    invalid_rows = 0
    k_hist = Counter()

    for row in rows:
        target = row.get("teacher_target") or {}
        if not target.get("teacher_text_topk_valid", False):
            missing_rows += 1
            continue
        try:
            topk_ids, topk_logprobs, target_ids = load_text_arrays(target)
        except Exception:
            invalid_rows += 1
            continue
        if target_ids.size == 0:
            invalid_rows += 1
            continue
        k_hist[int(topk_ids.shape[1])] += 1
        split = str(row.get("split") or "unknown")
        source = str(target.get("teacher_text_topk_source") or "unknown")
        token_count_hist[int(target_ids.size)] += 1

        overall.add(target_ids=target_ids, topk_ids=topk_ids, topk_logprobs=topk_logprobs)
        by_split[split].add(target_ids=target_ids, topk_ids=topk_ids, topk_logprobs=topk_logprobs)
        by_source[source].add(target_ids=target_ids, topk_ids=topk_ids, topk_logprobs=topk_logprobs)

        for pos in range(target_ids.size):
            one_target = target_ids[pos : pos + 1]
            one_ids = topk_ids[pos : pos + 1]
            one_lp = topk_logprobs[pos : pos + 1]
            if pos < 32:
                by_pos_from_start[pos + 1].add(target_ids=one_target, topk_ids=one_ids, topk_logprobs=one_lp)
            rev = target_ids.size - pos
            if rev <= 8:
                by_pos_from_end[rev].add(target_ids=one_target, topk_ids=one_ids, topk_logprobs=one_lp)

    summary: dict[str, Any] = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "max_rows": int(args.max_rows),
        "rows_read": len(rows),
        "missing_rows": missing_rows,
        "invalid_rows": invalid_rows,
        "topk_k_histogram": dict(sorted(k_hist.items())),
        "token_count_histogram_top20": dict(token_count_hist.most_common(20)),
        "overall": overall.summary(),
        "by_split": {k: v.summary() for k, v in sorted(by_split.items())},
        "by_source": {k: v.summary() for k, v in sorted(by_source.items())},
        "position_from_start_1_to_32": {str(k): v.summary() for k, v in sorted(by_pos_from_start.items())},
        "position_from_end_1_to_8": {str(k): v.summary() for k, v in sorted(by_pos_from_end.items())},
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def pct(v: float) -> str:
        return "nan" if not math.isfinite(v) else f"{100.0 * v:.2f}%"

    lines = [
        "# Text Top-8 CE Ceiling Audit",
        "",
        f"Corpus: `{args.corpus_jsonl}`",
        "",
        f"Rows read: `{len(rows)}`; missing rows: `{missing_rows}`; invalid rows: `{invalid_rows}`",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| positions | {overall.positions} |",
        f"| draw target is teacher top-1 | {pct(summary['overall']['target_is_top1_rate'])} |",
        f"| draw target is in top-8 | {pct(summary['overall']['target_in_topk_rate'])} |",
        f"| mean teacher top-1 prob | {summary['overall']['top1_prob'].get('mean', float('nan')):.4f} |",
        f"| mean target prob when in top-8 | {summary['overall']['target_prob_when_in_topk'].get('mean', float('nan')):.4f} |",
        f"| mean top-8 mass | {summary['overall']['topk_prob_mass'].get('mean', float('nan')):.4f} |",
        f"| mean tail mass | {summary['overall']['tail_mass_1_minus_topk_sum'].get('mean', float('nan')):.4f} |",
        f"| target token count p50 / p95 | {summary['overall']['target_token_count'].get('p50', float('nan')):.0f} / {summary['overall']['target_token_count'].get('p95', float('nan')):.0f} |",
        "",
        "## By Split",
        "",
        "| split | rows | positions | target top-1 | in top-8 | top-8 mass | tail mass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in summary["by_split"].items():
        lines.append(
            f"| {split} | {item['rows']} | {item['positions']} | {pct(item['target_is_top1_rate'])} | "
            f"{pct(item['target_in_topk_rate'])} | {item['topk_prob_mass'].get('mean', float('nan')):.4f} | "
            f"{item['tail_mass_1_minus_topk_sum'].get('mean', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## By Top-K Source",
            "",
            "| source | rows | positions | target top-1 | in top-8 | top-8 mass | tail mass |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, item in summary["by_source"].items():
        lines.append(
            f"| {source} | {item['rows']} | {item['positions']} | {pct(item['target_is_top1_rate'])} | "
            f"{pct(item['target_in_topk_rate'])} | {item['topk_prob_mass'].get('mean', float('nan')):.4f} | "
            f"{item['tail_mass_1_minus_topk_sum'].get('mean', float('nan')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Last Positions",
            "",
            "Position from end `1` is the final cached text/boundary token.",
            "",
            "| pos from end | positions | target top-1 | in top-8 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for pos, item in summary["position_from_end_1_to_8"].items():
        lines.append(
            f"| {pos} | {item['positions']} | {pct(item['target_is_top1_rate'])} | {pct(item['target_in_topk_rate'])} |"
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
