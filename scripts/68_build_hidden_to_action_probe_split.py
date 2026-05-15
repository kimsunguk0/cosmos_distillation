#!/usr/bin/env python3
"""Build clip-disjoint splits for hidden-to-action probe experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--split-name", default="hidden_to_action_probe_v1")
    return parser.parse_args()


def clip_id_for(row: dict[str, Any]) -> str:
    value = row.get("clip_id")
    if value not in (None, ""):
        return str(value)
    return str(row.get("sample_id", "")).split("__", 1)[0]


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    with args.corpus_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    train_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_rows: list[dict[str, Any]] = []
    for row in rows:
        split = str(row.get("split", ""))
        clip = clip_id_for(row)
        if split == "val":
            test_rows.append(row)
        elif split == "train":
            train_by_clip[clip].append(row)

    # Use full-size 8-anchor clips so the requested sample counts are exact while
    # preserving clip-level isolation.
    eligible_clips = sorted(clip for clip, clip_rows in train_by_clip.items() if len(clip_rows) == 8)
    rng = random.Random(args.seed)
    rng.shuffle(eligible_clips)

    if args.train_samples % 8 != 0 or args.val_samples % 8 != 0:
        raise ValueError("This builder expects train/val sample counts divisible by 8 for clip-exact splits.")
    train_clip_count = args.train_samples // 8
    val_clip_count = args.val_samples // 8
    needed = train_clip_count + val_clip_count
    if len(eligible_clips) < needed:
        raise RuntimeError(f"Need {needed} eight-anchor clips, found {len(eligible_clips)}")

    train_clips = set(eligible_clips[:train_clip_count])
    val_clips = set(eligible_clips[train_clip_count:needed])
    test_clips = {clip_id_for(row) for row in test_rows}

    if train_clips & val_clips or train_clips & test_clips or val_clips & test_clips:
        raise RuntimeError("Clip leakage detected while constructing splits.")

    split_rows: dict[str, list[dict[str, Any]]] = {"probe_train": [], "probe_val": [], "probe_test": []}
    for clip in sorted(train_clips):
        split_rows["probe_train"].extend(sorted(train_by_clip[clip], key=lambda item: str(item.get("sample_id"))))
    for clip in sorted(val_clips):
        split_rows["probe_val"].extend(sorted(train_by_clip[clip], key=lambda item: str(item.get("sample_id"))))
    split_rows["probe_test"] = sorted(test_rows, key=lambda item: (clip_id_for(item), str(item.get("sample_id"))))

    all_sample_ids_by_split = {
        name: [str(row.get("sample_id")) for row in split_rows[name]]
        for name in ("probe_train", "probe_val", "probe_test")
    }
    all_clip_ids_by_split = {
        name: sorted({clip_id_for(row) for row in split_rows[name]})
        for name in ("probe_train", "probe_val", "probe_test")
    }

    sample_sets = {key: set(value) for key, value in all_sample_ids_by_split.items()}
    clip_sets = {key: set(value) for key, value in all_clip_ids_by_split.items()}
    leakage = {
        "train_val_sample_overlap": len(sample_sets["probe_train"] & sample_sets["probe_val"]),
        "train_test_sample_overlap": len(sample_sets["probe_train"] & sample_sets["probe_test"]),
        "val_test_sample_overlap": len(sample_sets["probe_val"] & sample_sets["probe_test"]),
        "train_val_clip_overlap": len(clip_sets["probe_train"] & clip_sets["probe_val"]),
        "train_test_clip_overlap": len(clip_sets["probe_train"] & clip_sets["probe_test"]),
        "val_test_clip_overlap": len(clip_sets["probe_val"] & clip_sets["probe_test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Leakage detected: {leakage}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.split_name}.jsonl"
    csv_path = args.output_dir / f"{args.split_name}.csv"
    sample_ids_path = args.output_dir / f"{args.split_name}.sample_ids.json"
    clip_ids_path = args.output_dir / f"{args.split_name}.clip_ids.json"
    summary_path = args.output_dir / f"{args.split_name}.summary.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for split_name in ("probe_train", "probe_val", "probe_test"):
            for row in split_rows[split_name]:
                handle.write(
                    json.dumps(
                        {
                            "split_name": split_name,
                            "sample_id": str(row.get("sample_id")),
                            "clip_id": clip_id_for(row),
                            "source_split": str(row.get("split", "")),
                            "chunk_id": str(row.get("chunk_id", "")),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split_name", "sample_id", "clip_id", "source_split", "chunk_id"])
        writer.writeheader()
        for split_name in ("probe_train", "probe_val", "probe_test"):
            for row in split_rows[split_name]:
                writer.writerow(
                    {
                        "split_name": split_name,
                        "sample_id": str(row.get("sample_id")),
                        "clip_id": clip_id_for(row),
                        "source_split": str(row.get("split", "")),
                        "chunk_id": str(row.get("chunk_id", "")),
                    }
                )

    sample_ids_payload = {
        "schema_version": "hidden_to_action_probe_split_sample_ids_v1",
        "split_name": args.split_name,
        "sample_ids": all_sample_ids_by_split,
    }
    clip_ids_payload = {
        "schema_version": "hidden_to_action_probe_split_clip_ids_v1",
        "split_name": args.split_name,
        "clip_ids": all_clip_ids_by_split,
    }
    sample_ids_path.write_text(json.dumps(sample_ids_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    clip_ids_path.write_text(json.dumps(clip_ids_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_counts = {name: len(split_rows[name]) for name in split_rows}
    clip_counts = {name: len(all_clip_ids_by_split[name]) for name in split_rows}
    source_split_counts = {
        name: dict(Counter(str(row.get("split", "")) for row in split_rows[name]))
        for name in split_rows
    }
    chunk_counts_top20 = {
        name: Counter(str(row.get("chunk_id", "")) for row in split_rows[name]).most_common(20)
        for name in split_rows
    }
    summary = {
        "schema_version": "hidden_to_action_probe_split_summary_v1",
        "split_name": args.split_name,
        "seed": args.seed,
        "corpus_jsonl": str(args.corpus_jsonl),
        "total_corpus_rows": len(rows),
        "selected_sample_counts": split_counts,
        "selected_clip_counts": clip_counts,
        "source_split_counts": source_split_counts,
        "leakage": leakage,
        "clip_size_policy": "only 8-anchor source train clips for probe_train/probe_val; full source val for probe_test",
        "test_policy": "full corpus split == val",
        "eligible_train_8_anchor_clip_count": len(eligible_clips),
        "source_train_clip_size_counts": dict(Counter(len(value) for value in train_by_clip.values())),
        "source_test_clip_size_counts": dict(Counter(Counter(clip_id_for(row) for row in test_rows).values())),
        "chunk_counts_top20": chunk_counts_top20,
        "sample_ids_hash": sha256_json(all_sample_ids_by_split),
        "clip_ids_hash": sha256_json(all_clip_ids_by_split),
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "sample_ids_path": str(sample_ids_path),
        "clip_ids_path": str(clip_ids_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
