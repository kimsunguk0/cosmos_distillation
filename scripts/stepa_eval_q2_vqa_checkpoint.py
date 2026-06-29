#!/usr/bin/env python3
"""Generate and score Step A Q2 VQA outputs from a Cosmos-Reason2 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vqa.q2_stepa import (  # noqa: E402
    ACTION_OVERREACH_RE,
    COORDINATE_RE,
    create_vqa_messages,
    encode_messages,
    load_row_frame_tensors,
    normalize_text,
    read_jsonl,
)


DEFAULT_MODEL = Path("/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b")
DEFAULT_VAL = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "q2_repaired_supported_v1" / "q2_supported_repaired_val.jsonl"
DEFAULT_TEST = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "q2_repaired_supported_v1" / "q2_supported_repaired_test.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "eval" / "stepa_q2_vqa"

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
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "these",
    "this",
    "to",
    "with",
}
VISIBLE_TERMS = {
    "barrier",
    "bike",
    "blocked",
    "bus",
    "car",
    "cone",
    "cones",
    "construction",
    "crosswalk",
    "cyclist",
    "equipment",
    "intersection",
    "lane",
    "lanes",
    "light",
    "marking",
    "obstacle",
    "overpass",
    "parked",
    "pedestrian",
    "road",
    "roadway",
    "sign",
    "sidewalk",
    "traffic",
    "truck",
    "vehicle",
    "vehicles",
    "worker",
    "workers",
}
ACTION_LABELS = {
    "continue",
    "creep",
    "keep lane",
    "keep_lane",
    "proceed",
    "slow down",
    "slow_down",
    "stop",
    "yield",
}
HARD_BAD_FLAGS = {
    "empty",
    "coordinate_like",
    "single_word_or_risk_label",
    "action_label_only",
    "too_short",
    "too_long",
    "no_visible_term",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def word_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_-]*", str(text or "").lower())
        if token not in STOPWORDS
    ]


def token_f1(candidate: str, reference: str) -> float:
    cand = Counter(word_tokens(candidate))
    ref = Counter(word_tokens(reference))
    if not cand or not ref:
        return 0.0
    overlap = sum((cand & ref).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(1, sum(cand.values()))
    recall = overlap / max(1, sum(ref.values()))
    return float(2.0 * precision * recall / max(1e-12, precision + recall))


def clean_generated_text(text: str) -> str:
    cleaned = str(text or "").replace("<|answer_start|>", "")
    for marker in [
        "<|answer_end|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|question_start|>",
        "<|question_end|>",
        "<|vision_end|>",
    ]:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \n\t;")
    return cleaned


def bad_flags(answer: str) -> list[str]:
    normalized = normalize_text(answer).lower()
    words = word_tokens(normalized)
    flags: list[str] = []
    if not normalized:
        flags.append("empty")
    if COORDINATE_RE.search(str(answer or "")):
        flags.append("coordinate_like")
    if normalized in {"yes", "no", "low", "medium", "high", "unknown"}:
        flags.append("single_word_or_risk_label")
    if normalized in ACTION_LABELS:
        flags.append("action_label_only")
    if len(words) < 5:
        flags.append("too_short")
    if len(words) > 90:
        flags.append("too_long")
    if ACTION_OVERREACH_RE.search(normalized):
        flags.append("action_or_future_language")
    if words and not (set(words) & VISIBLE_TERMS):
        flags.append("no_visible_term")
    return flags


def exact_normalized(a: str, b: str) -> bool:
    return normalize_text(a).lower() == normalize_text(b).lower()


def row_metrics(prediction: str, row: dict[str, Any], *, reference_mode: str = "q2") -> dict[str, Any]:
    teacher_short = str(row.get("teacher_answer_short") or "")
    supported = " ".join(str(item) for item in (row.get("teacher", {}).get("vision_judge_supported") or []))
    flags = bad_flags(prediction)
    hard_flags = [flag for flag in flags if flag in HARD_BAD_FLAGS]
    metrics = {
        "word_count": len(word_tokens(prediction)),
        "bad_flags": flags,
        "bad_flag_count": len(flags),
        "hard_bad_flags": hard_flags,
        "hard_bad_flag_count": len(hard_flags),
        "has_action_or_future_language": "action_or_future_language" in flags,
    }
    if reference_mode == "q2":
        metrics.update(
            {
                "teacher_short_token_f1": token_f1(prediction, teacher_short),
                "supported_claim_token_f1": token_f1(prediction, supported) if supported else None,
                "exact_normalized": exact_normalized(prediction, teacher_short),
            }
        )
    else:
        metrics.update(
            {
                "teacher_short_token_f1": None,
                "supported_claim_token_f1": None,
                "exact_normalized": None,
            }
        )
    return metrics


class Q2Rows(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        max_samples: int | None = None,
        override_question: str | None = None,
        eval_family: str | None = None,
        eval_qid: str | None = None,
    ) -> None:
        rows = read_jsonl(path)
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        if override_question:
            rows = [
                dict(
                    row,
                    question=str(override_question),
                    family=str(eval_family or row.get("family") or ""),
                    qid=str(eval_qid or row.get("qid") or ""),
                )
                for row in rows
            ]
        if not rows:
            raise ValueError(f"No rows loaded from {path}")
        self.path = path
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        frames, camera_indices = load_row_frame_tensors(row)
        return {"row": row, "frames": frames, "camera_indices": camera_indices}


@dataclass(slots=True)
class PromptCollator:
    processor: Any
    max_length: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        messages = []
        rows = []
        for item in features:
            row = item["row"]
            rows.append(row)
            messages.append(
                create_vqa_messages(
                    frames=item["frames"],
                    camera_indices=item["camera_indices"],
                    question=str(row["question"]),
                    answer_text=None,
                )
            )
        batch = encode_messages(
            self.processor,
            messages,
            continue_final_message=True,
            max_length=self.max_length,
        )
        batch["rows"] = rows
        batch["sample_ids"] = [str(row.get("sample_id")) for row in rows]
        return batch


def resolve_subdir(path: Path, name: str) -> Path:
    candidate = path / name
    return candidate if candidate.is_dir() else path


def load_model_and_processor(args: argparse.Namespace):
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer or resolve_subdir(model_path, "tokenizer"))
    processor_path = Path(args.processor or resolve_subdir(model_path, "processor"))
    local_only = bool(args.local_files_only or model_path.exists())
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=local_only,
    )
    tokenizer.padding_side = "left"
    processor = AutoProcessor.from_pretrained(
        str(processor_path),
        trust_remote_code=True,
        local_files_only=local_only,
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
    )
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = "left"
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model.to(args.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, processor, tokenizer


def move_tensors(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def generate_batch(
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    *,
    max_new_tokens: int,
) -> list[str]:
    prompt_len = int(batch["input_ids"].shape[1])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(batch["input_ids"].device).startswith("cuda")):
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_tokens = generated[:, prompt_len:]
    decoded = tokenizer.batch_decode(new_tokens.detach().cpu().tolist(), skip_special_tokens=False)
    return [clean_generated_text(text) for text in decoded]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "unknown")].append(row)

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        flag_counter: Counter[str] = Counter()
        hard_flag_counter: Counter[str] = Counter()
        for item in items:
            flag_counter.update(item["metrics"]["bad_flags"])
            hard_flag_counter.update(item["metrics"].get("hard_bad_flags") or [])
        supported_values = [
            float(item["metrics"]["supported_claim_token_f1"])
            for item in items
            if item["metrics"]["supported_claim_token_f1"] is not None
        ]
        teacher_values = [
            float(item["metrics"]["teacher_short_token_f1"])
            for item in items
            if item["metrics"]["teacher_short_token_f1"] is not None
        ]
        exact_values = [
            bool(item["metrics"]["exact_normalized"])
            for item in items
            if item["metrics"]["exact_normalized"] is not None
        ]
        return {
            "n": n,
            "teacher_short_token_f1_mean": sum(teacher_values) / max(1, len(teacher_values)) if teacher_values else None,
            "supported_claim_token_f1_mean": sum(supported_values) / max(1, len(supported_values)) if supported_values else None,
            "exact_normalized_rate": sum(exact_values) / max(1, len(exact_values)) if exact_values else None,
            "hard_bad_output_rate": sum(bool(item["metrics"].get("hard_bad_flags")) for item in items) / max(1, n),
            "action_or_future_language_rate": sum(bool(item["metrics"].get("has_action_or_future_language")) for item in items) / max(1, n),
            "mean_word_count": sum(int(item["metrics"]["word_count"]) for item in items) / max(1, n),
            "bad_flag_counts": dict(flag_counter.most_common()),
            "hard_bad_flag_counts": dict(hard_flag_counter.most_common()),
        }

    return {
        "overall": summarize_group(rows),
        "by_split": {split: summarize_group(items) for split, items in sorted(by_split.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--processor", type=Path)
    parser.add_argument("--input-jsonl", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--override-question", default=None)
    parser.add_argument("--eval-family", default=None)
    parser.add_argument("--eval-qid", default=None)
    parser.add_argument("--reference-mode", choices=["q2", "none"], default="q2")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    input_paths = args.input_jsonl or [DEFAULT_VAL, DEFAULT_TEST]
    run_name = args.run_name or Path(args.model).name
    output_dir = args.output_dir / run_name
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if predictions_path.exists():
        predictions_path.unlink()

    started = time.time()
    model, processor, tokenizer = load_model_and_processor(args)
    all_outputs: list[dict[str, Any]] = []
    total_seen = 0
    for input_path in input_paths:
        split_name = input_path.stem
        dataset = Q2Rows(
            input_path,
            max_samples=args.max_samples,
            override_question=args.override_question,
            eval_family=args.eval_family,
            eval_qid=args.eval_qid,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=PromptCollator(processor=processor, max_length=args.max_length),
        )
        for batch_index, batch in enumerate(loader, start=1):
            rows = batch.pop("rows")
            sample_ids = batch.pop("sample_ids")
            batch = move_tensors(batch, str(args.device))
            predictions = generate_batch(
                model,
                tokenizer,
                batch,
                max_new_tokens=int(args.max_new_tokens),
            )
            out_rows = []
            for sample_id, row, prediction in zip(sample_ids, rows, predictions, strict=True):
                metrics = row_metrics(prediction, row, reference_mode=str(args.reference_mode))
                out = {
                    "sample_id": str(sample_id),
                    "split": str(row.get("split") or split_name),
                    "clip_id": row.get("clip_id"),
                    "slot": row.get("slot"),
                    "family": row.get("family"),
                    "qid": row.get("qid"),
                    "question": row.get("question"),
                    "teacher_answer_short": row.get("teacher_answer_short"),
                    "prediction": prediction,
                    "metrics": metrics,
                    "teacher": {
                        "vision_judge_supported": row.get("teacher", {}).get("vision_judge_supported"),
                        "vision_judge_unsupported": row.get("teacher", {}).get("vision_judge_unsupported"),
                        "repair_label_source": row.get("teacher", {}).get("repair_label_source"),
                    },
                }
                out_rows.append(out)
            append_jsonl(predictions_path, out_rows)
            all_outputs.extend(out_rows)
            total_seen += len(out_rows)
            if int(args.log_every) > 0 and total_seen % int(args.log_every) == 0:
                print(
                    json.dumps(
                        {
                            "event": "eval_progress",
                            "run_name": run_name,
                            "input": str(input_path),
                            "rows": total_seen,
                            "elapsed_sec": round(time.time() - started, 1),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )

    summary = {
        "created_at": utc_now(),
        "model": str(args.model),
        "tokenizer": str(args.tokenizer or resolve_subdir(Path(args.model), "tokenizer")),
        "processor": str(args.processor or resolve_subdir(Path(args.model), "processor")),
        "input_jsonl": [str(path) for path in input_paths],
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_path),
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "override_question": args.override_question,
        "eval_family": args.eval_family,
        "eval_qid": args.eval_qid,
        "reference_mode": str(args.reference_mode),
        "elapsed_sec": round(time.time() - started, 3),
        **summarize(all_outputs),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
