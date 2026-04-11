#!/usr/bin/env python3
"""Checkpoint export entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_row = None
    with args.metrics_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            loss = float(row["logs"]["total_loss"])
            if best_row is None or loss < float(best_row["logs"]["total_loss"]):
                best_row = row

    summary = {
        "metrics_jsonl": str(args.metrics_jsonl),
        "best_row": best_row,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
