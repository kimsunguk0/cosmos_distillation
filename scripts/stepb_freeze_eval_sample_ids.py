#!/usr/bin/env python3
"""Freeze the first N sample ids for a corpus split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected: list[str] = []
    with args.corpus_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != args.split:
                continue
            sample_id = str(row.get("sample_id") or "")
            if sample_id:
                selected.append(sample_id)
            if len(selected) >= int(args.num_samples):
                break
    payload = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": str(args.split),
        "num_samples_requested": int(args.num_samples),
        "num_samples": len(selected),
        "sample_ids": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "num_samples": len(selected)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
