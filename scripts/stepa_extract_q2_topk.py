#!/usr/bin/env python3
"""Extract Alpamayo teacher top-k text logits for Step A Q2 VQA records."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vqa.q2_stepa import (
    active_label_positions,
    create_vqa_messages,
    encode_messages,
    labels_from_prompt_and_full,
    load_row_frame_tensors,
    read_jsonl,
)


DEFAULT_ALPAMAYO_SRC = Path("/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src")
DEFAULT_TEACHER_MODEL = Path("/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B")
DEFAULT_PROCESSOR = DEFAULT_TEACHER_MODEL / "runtime_support" / "Cosmos-Reason2-8B"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "vqa_q2_stepa" / "train_q2_stepa.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vqa_q2_stepa" / "teacher_topk32"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("teacher_topk_ready") and row.get("sample_id"):
                done.add(str(row["sample_id"]))
    return done


def load_teacher(args: argparse.Namespace):
    if str(args.alpamayo_src) not in sys.path:
        sys.path.insert(0, str(args.alpamayo_src))
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    config = Alpamayo1_5Config.from_pretrained(str(args.teacher_model))
    config.vlm_name_or_path = str(args.processor_path)
    model = Alpamayo1_5.from_pretrained(
        str(args.teacher_model),
        config=config,
        dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(args.processor_path),
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        local_files_only=True,
        trust_remote_code=True,
    )
    model.tokenizer.padding_side = "right"
    processor.tokenizer = model.tokenizer
    processor.tokenizer.padding_side = "right"
    return model, processor


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def extract_one(
    *,
    row: dict[str, Any],
    model,
    processor,
    device: str,
    topk: int,
    max_length: int | None,
) -> dict[str, Any]:
    frames, camera_indices = load_row_frame_tensors(row)
    question = str(row["question"])
    target = str(row["teacher_answer_short"]).strip()
    prompt_messages = create_vqa_messages(
        frames=frames,
        camera_indices=camera_indices,
        question=question,
        answer_text=None,
    )
    full_messages = create_vqa_messages(
        frames=frames,
        camera_indices=camera_indices,
        question=question,
        answer_text=target,
    )
    prompt_batch = encode_messages(
        processor,
        [prompt_messages],
        continue_final_message=True,
        max_length=max_length,
    )
    full_batch = encode_messages(
        processor,
        [full_messages],
        continue_final_message=False,
        max_length=max_length,
    )
    labels = labels_from_prompt_and_full(prompt_batch, full_batch)
    active_positions = active_label_positions(labels)[0]
    if not active_positions:
        raise RuntimeError("no active answer-token positions after masking")

    full_device = to_device(full_batch, device)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        outputs = model.vlm(
            input_ids=full_device["input_ids"],
            attention_mask=full_device["attention_mask"],
            pixel_values=full_device.get("pixel_values"),
            image_grid_thw=full_device.get("image_grid_thw"),
            use_cache=False,
            return_dict=True,
        )

    shifted_logits = outputs.logits[:, :-1, :].float().cpu()
    shifted_labels = labels[:, 1:].cpu()
    valid_mask = shifted_labels[0] != -100
    target_logits = shifted_logits[0][valid_mask]
    target_ids = shifted_labels[0][valid_mask].to(torch.int64)
    if int(target_logits.shape[0]) != len(active_positions):
        raise RuntimeError(
            f"active position mismatch: logits={int(target_logits.shape[0])} labels={len(active_positions)}"
        )
    k = min(int(topk), int(target_logits.shape[-1]))
    log_probs = torch.log_softmax(target_logits, dim=-1)
    topk_logprobs, topk_indices = torch.topk(log_probs, k=k, dim=-1)
    target_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    target_ranks = (log_probs > target_logprobs.unsqueeze(-1)).sum(dim=-1) + 1
    target_in_topk = (topk_indices == target_ids.unsqueeze(-1)).any(dim=-1)
    return {
        "topk_indices": topk_indices.numpy().astype(np.int32),
        "topk_logprobs": topk_logprobs.numpy().astype(np.float32),
        "target_token_ids": target_ids.numpy().astype(np.int32),
        "target_logprobs": target_logprobs.numpy().astype(np.float32),
        "target_ranks": target_ranks.numpy().astype(np.int32),
        "target_in_topk": target_in_topk.numpy().astype(bool),
        "target_token_count": int(target_ids.numel()),
        "prompt_token_count": int(prompt_batch["attention_mask"].sum().item()),
        "full_token_count": int(full_batch["attention_mask"].sum().item()),
        "topk": int(k),
        "active_label_positions": np.asarray(active_positions, dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpamayo-src", type=Path, default=DEFAULT_ALPAMAYO_SRC)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--summary-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    started = time.time()
    rows = read_jsonl(args.input_jsonl)
    end = len(rows) if args.limit is None else min(len(rows), int(args.start_index) + int(args.limit))
    selected = rows[int(args.start_index) : end]
    args.output_root.mkdir(parents=True, exist_ok=True)
    topk_dir = args.output_root / "npz"
    topk_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_root / "records_with_topk.jsonl"
    summary_path = args.output_root / "summary.json"
    done_ids = set() if args.overwrite else load_done_ids(output_jsonl)

    model, processor = load_teacher(args)
    print(
        json.dumps(
            {
                "event": "teacher_ready",
                "input_jsonl": str(args.input_jsonl),
                "selected": len(selected),
                "device": args.device,
                "topk": int(args.topk),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    counts: dict[str, int] = {"selected": len(selected), "done_existing": 0, "ok": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    for local_index, row in enumerate(selected, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in done_ids:
            counts["done_existing"] += 1
            continue
        out_path = topk_dir / f"{sample_id}.topk{int(args.topk)}.npz"
        try:
            signals = extract_one(
                row=row,
                model=model,
                processor=processor,
                device=str(args.device),
                topk=int(args.topk),
                max_length=args.max_length,
            )
            np.savez_compressed(
                out_path,
                topk_indices=signals["topk_indices"],
                topk_logprobs=signals["topk_logprobs"],
                target_token_ids=signals["target_token_ids"],
                target_logprobs=signals["target_logprobs"],
                target_ranks=signals["target_ranks"],
                target_in_topk=signals["target_in_topk"],
                active_label_positions=signals["active_label_positions"],
                target_token_count=np.asarray(signals["target_token_count"], dtype=np.int32),
                prompt_token_count=np.asarray(signals["prompt_token_count"], dtype=np.int32),
                full_token_count=np.asarray(signals["full_token_count"], dtype=np.int32),
                topk=np.asarray(signals["topk"], dtype=np.int32),
            )
            out_row = dict(row)
            out_row["teacher_topk_ready"] = True
            out_row["teacher_topk_path"] = str(out_path)
            out_row["teacher_topk_k"] = int(signals["topk"])
            out_row["teacher_target_token_count"] = int(signals["target_token_count"])
            out_row["teacher_target_in_topk_rate"] = float(np.asarray(signals["target_in_topk"]).mean())
            out_row["teacher_topk_created_at"] = utc_now()
            append_jsonl(output_jsonl, out_row)
            counts["ok"] += 1
        except Exception as exc:  # noqa: BLE001
            counts["failed"] += 1
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            print(
                json.dumps(
                    {
                        "event": "topk_failed",
                        "sample_id": sample_id,
                        "local_index": local_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        if local_index % max(1, int(args.summary_every)) == 0:
            print(
                json.dumps(
                    {
                        "event": "topk_progress",
                        "local_index": local_index,
                        "selected": len(selected),
                        "counts": counts,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    summary = {
        "created_at": utc_now(),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(output_jsonl),
        "output_root": str(args.output_root),
        "topk_dir": str(topk_dir),
        "counts": counts,
        "failures": failures[:20],
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()

