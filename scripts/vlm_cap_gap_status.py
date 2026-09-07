#!/usr/bin/env python3
"""Print progress for the CR2 VLM capability-gap eval."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def inspect_predictions(path: Path) -> dict:
    stats = {
        "rows": 0,
        "bad_json": 0,
        "duplicate_keys": 0,
        "parse_error": 0,
        "runtime_error": 0,
        "tasks": {},
        "samples": {},
    }
    if not path.exists():
        return stats
    keys = []
    tasks = Counter()
    samples = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            stats["rows"] += 1
            key = (row.get("clip_id"), row.get("task_id"), int(row.get("sample_idx", -999)))
            keys.append(key)
            tasks[str(row.get("task_id"))] += 1
            samples[str(row.get("sample_idx"))] += 1
            if row.get("parse_error"):
                stats["parse_error"] += 1
            if row.get("error"):
                stats["runtime_error"] += 1
    stats["duplicate_keys"] = len(keys) - len(set(keys))
    stats["tasks"] = dict(sorted(tasks.items()))
    stats["samples"] = dict(sorted(samples.items()))
    return stats


def newest_log_time(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_mtime


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def parse_scheduler_snapshots(path: Path) -> list[dict]:
    if not path.exists():
        return []
    pattern = re.compile(
        r"^\[(?P<ts>\d{8}T\d{6}Z)\]\s+rows:\s+"
        r"2b=(?P<m2>\d+)\s+8b=(?P<m8>\d+)\s+32b=(?P<m32>\d+)"
    )
    snapshots = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line.strip())
            if not match:
                continue
            ts = datetime.strptime(match.group("ts"), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            snapshots.append(
                {
                    "time": ts.timestamp(),
                    "2b": int(match.group("m2")),
                    "8b": int(match.group("m8")),
                    "32b": int(match.group("m32")),
                }
            )
    return snapshots


def rate_from_snapshots(snapshots: list[dict], model: str) -> tuple[float | None, float | None]:
    if len(snapshots) < 2:
        return None, None
    prev, cur = snapshots[-2], snapshots[-1]
    dt = float(cur["time"] - prev["time"])
    if dt <= 0:
        return None, None
    delta = float(cur.get(model, 0) - prev.get(model, 0))
    return delta / dt, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/eval/vlm_cap_gap_human_ood_20260701_full300")
    parser.add_argument("--expected-rows", type=int, default=9000)
    parser.add_argument("--models", default="2b,8b,32b")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    now = time.time()
    snapshots = parse_scheduler_snapshots(out_dir / "logs" / "scheduler.log")
    inspected = {}
    mtimes = {}
    for model in models:
        pred = out_dir / "runs" / model / "predictions.jsonl"
        inspected[model] = inspect_predictions(pred)
        mtimes[model] = newest_log_time(pred)

    payload = {
        "output_dir": str(out_dir),
        "expected_rows_per_model": int(args.expected_rows),
        "models": {},
    }
    for model in models:
        pred_stats = inspected[model]
        n = int(pred_stats["rows"])
        remaining = max(0, int(args.expected_rows) - n)
        progress = n / max(1, int(args.expected_rows))
        age = None if mtimes[model] is None else now - mtimes[model]
        rows_per_sec, recent_delta = rate_from_snapshots(snapshots, model)
        eta = None if not rows_per_sec or rows_per_sec <= 0 else remaining / rows_per_sec
        payload["models"][model] = {
            "rows": n,
            "remaining": remaining,
            "progress_pct": round(progress * 100.0, 3),
            "last_write_age": fmt_seconds(age),
            "recent_delta_rows": None if recent_delta is None else int(recent_delta),
            "recent_rows_per_hour": None if rows_per_sec is None else round(rows_per_sec * 3600.0, 3),
            "eta_at_recent_rate": fmt_seconds(eta),
            "bad_json": pred_stats["bad_json"],
            "duplicate_keys": pred_stats["duplicate_keys"],
            "parse_error": pred_stats["parse_error"],
            "runtime_error": pred_stats["runtime_error"],
            "tasks": pred_stats["tasks"],
            "samples": pred_stats["samples"],
        }
    payload["total_rows"] = sum(int(item["rows"]) for item in inspected.values())
    payload["total_expected_rows"] = int(args.expected_rows) * len(models)
    payload["total_progress_pct"] = round(
        100.0 * payload["total_rows"] / max(1, payload["total_expected_rows"]),
        3,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
