#!/usr/bin/env python3
"""Report live progress for a priority chunk download run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Existing priority plan JSON emitted by 03_download_priority_chunks.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the live progress snapshot. Defaults next to summary JSON.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional downloader log path for last progress line capture.",
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional downloader pid to report liveness.",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=None,
        help="Optional file containing the current downloader pid.",
    )
    parser.add_argument(
        "--pid-cmd-substring",
        default=None,
        help="Optional substring that must appear in the process cmdline for the pid to count as alive.",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="If > 0, keep refreshing every N seconds until pid exits.",
    )
    return parser.parse_args()


def feature_prefix(feature: str) -> str:
    return "camera" if feature.startswith("camera") else "labels"


def local_chunk_path(output_root: Path, feature: str, chunk: int) -> Path:
    return output_root / feature_prefix(feature) / feature / f"{feature}.chunk_{int(chunk):04d}.zip"


def resolve_pid(args: argparse.Namespace) -> int | None:
    if args.pid_file is not None:
        try:
            text = args.pid_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return args.pid


def process_alive(pid: int | None, cmd_substring: str | None = None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if cmd_substring:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            return False
        if cmd_substring not in cmdline:
            return False
    return True


def read_last_progress_line(log_path: Path | None) -> str | None:
    if log_path is None or not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    candidates = [line.strip() for line in text.splitlines() if "Fetching ... files:" in line or "Trying to resume download..." in line]
    return candidates[-1] if candidates else None


def count_active_incomplete(cache_root: Path, feature: str, recent_within_sec: int = 1800) -> dict[str, int]:
    feature_dir = cache_root / feature_prefix(feature) / feature
    if not feature_dir.exists():
        return {"all": 0, "recent": 0}
    now = time.time()
    all_count = 0
    recent_count = 0
    for path in feature_dir.glob("*.incomplete"):
        all_count += 1
        try:
            if now - path.stat().st_mtime <= recent_within_sec:
                recent_count += 1
        except FileNotFoundError:
            continue
    return {"all": all_count, "recent": recent_count}


def build_snapshot(args: argparse.Namespace) -> dict:
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    resolved_pid = resolve_pid(args)
    output_root = Path(summary["output_root"])
    selected_chunks = [int(chunk) for chunk in summary["selected_chunks"]]
    preferred_features = list(summary["preferred_features"])
    anchor_feature = summary["anchor_feature"]

    manifest_path = Path(summary["manifest_path"])
    manifest = pd.read_parquet(manifest_path) if manifest_path.exists() else None

    rows: list[dict] = []
    ready_chunks: list[int] = []
    feature_completed = {feature: 0 for feature in preferred_features}
    feature_missing_chunks = {feature: [] for feature in preferred_features}
    for chunk in selected_chunks:
        row = {"chunk": chunk}
        anchor_path = local_chunk_path(output_root, anchor_feature, chunk)
        row[anchor_feature] = anchor_path.exists() or anchor_path.is_symlink()
        ready = bool(row[anchor_feature])
        for feature in preferred_features:
            path = local_chunk_path(output_root, feature, chunk)
            present = path.exists() or path.is_symlink()
            row[feature] = present
            if present:
                feature_completed[feature] += 1
            else:
                feature_missing_chunks[feature].append(chunk)
            ready = ready and present
        row["ready"] = ready
        rows.append(row)
        if ready:
            ready_chunks.append(chunk)

    ready_sample_count_selected = None
    selected_sample_count = summary.get("selected_sample_count_if_completed")
    ready_sample_count_global = None
    base_ready_sample_count = summary.get("ready_sample_count_now")
    if manifest is not None:
        ready_sample_count_selected = int(manifest[manifest["chunk"].isin(ready_chunks)].shape[0])
        if base_ready_sample_count is not None:
            ready_sample_count_global = int(base_ready_sample_count) + ready_sample_count_selected

    cache_root = output_root / ".cache" / "huggingface" / "download"
    active_incomplete = {
        feature: count_active_incomplete(cache_root, feature)
        for feature in preferred_features
    }

    return {
        "summary_json": str(args.summary_json),
        "pid": resolved_pid,
        "pid_alive": process_alive(resolved_pid, args.pid_cmd_substring),
        "log_path": str(args.log_path) if args.log_path else None,
        "log_last_progress_line": read_last_progress_line(args.log_path),
        "selected_chunk_count": len(selected_chunks),
        "selected_sample_count_if_completed": selected_sample_count,
        "completed_feature_counts": feature_completed,
        "remaining_feature_counts": {
            feature: len(feature_missing_chunks[feature]) for feature in preferred_features
        },
        "remaining_chunks_by_feature": feature_missing_chunks,
        "fully_ready_chunk_count": len(ready_chunks),
        "fully_ready_chunks": ready_chunks,
        "fully_ready_chunk_progress": round(len(ready_chunks) / len(selected_chunks), 4) if selected_chunks else 0.0,
        "selected_ready_sample_count": ready_sample_count_selected,
        "global_ready_sample_delta_from_this_batch": ready_sample_count_selected,
        "global_ready_sample_count_now": ready_sample_count_global,
        "active_incomplete_by_feature": active_incomplete,
        "updated_at_epoch": time.time(),
        "updated_at_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }


def write_snapshot(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.summary_json.with_name(args.summary_json.stem + "_progress.json")

    while True:
        snapshot = build_snapshot(args)
        write_snapshot(output_json, snapshot)
        print(json.dumps(snapshot, indent=2))

        if args.watch_seconds <= 0:
            break
        if snapshot["pid_alive"] is False:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
