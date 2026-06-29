#!/usr/bin/env python3
"""Build paired 1A vision-judge manifests from base/FT VQA predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def stable_swap(sample_id: str, seed: int) -> bool:
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def image_name(index: int, sample_id: str, image_format: str) -> str:
    suffix = "jpg" if image_format == "jpg" else "png"
    return f"images/{index:08d}_{sample_id}.{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ft-predictions", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ft = {row["sample_id"]: row for row in read_jsonl(args.ft_predictions)}
    base = {row["sample_id"]: row for row in read_jsonl(args.base_predictions)}
    source: dict[str, dict[str, Any]] = {}
    for path in args.source_jsonl:
        for row in read_jsonl(path):
            sid = str(row.get("sample_id"))
            if sid:
                source[sid] = row

    sample_ids = sorted(set(ft) & set(base) & set(source))
    rng = random.Random(int(args.seed))
    rng.shuffle(sample_ids)
    if args.limit is not None:
        sample_ids = sample_ids[: int(args.limit)]

    selected_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        src = dict(source[sample_id])
        selected_rows.append(src)
        ft_answer = str(ft[sample_id].get("prediction") or "")
        base_answer = str(base[sample_id].get("prediction") or "")
        swap = stable_swap(sample_id, int(args.seed))
        if swap:
            answer_a, answer_b = base_answer, ft_answer
            model_a, model_b = "base2b", "ft_step3488"
        else:
            answer_a, answer_b = ft_answer, base_answer
            model_a, model_b = "ft_step3488", "base2b"
        paired_rows.append(
            {
                "audit_index": index,
                "sample_id": sample_id,
                "split": ft[sample_id].get("split") or base[sample_id].get("split") or src.get("split"),
                "clip_id": src.get("clip_id"),
                "slot": src.get("slot"),
                "question": ft[sample_id].get("question") or base[sample_id].get("question"),
                "image_path": image_name(index, sample_id, str(args.image_format)),
                "answer_a": answer_a,
                "answer_b": answer_b,
                "answer_a_model": model_a,
                "answer_b_model": model_b,
                "ft_prediction": ft_answer,
                "base_prediction": base_answer,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "selected_rows_for_render.jsonl", selected_rows)
    write_jsonl(args.output_dir / "paired_manifest.jsonl", paired_rows)
    summary = {
        "ft_predictions": str(args.ft_predictions),
        "base_predictions": str(args.base_predictions),
        "source_jsonl": [str(path) for path in args.source_jsonl],
        "output_dir": str(args.output_dir),
        "selected": len(paired_rows),
        "seed": int(args.seed),
        "limit": args.limit,
        "image_format": str(args.image_format),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
