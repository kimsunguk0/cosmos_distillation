#!/usr/bin/env python3
"""Run Alpamayo Q2 teacher generation on Step A candidate rows."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
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

from src.data.local_dataset import decode_video_frames
from src.vqa.q2_stepa import create_vqa_messages, pil_to_chw_uint8, read_jsonl, shorten_q2_answer


DEFAULT_ALPAMAYO_SRC = Path("/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src")
DEFAULT_TEACHER_MODEL = Path("/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B")
DEFAULT_PROCESSOR = DEFAULT_TEACHER_MODEL / "runtime_support" / "Cosmos-Reason2-8B"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "q2_candidates_all.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "vqa_q2_stepa_pilot50k" / "teacher_q2_t0p60"

COORDINATE_RE = re.compile(
    r"(\[[^\]]*\d[^\]]*\]|\([^\)]*\d[,\s]+[^\)]*\d[^\)]*\)|"
    r"\b\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\b)"
)
VELOCITY_RE = re.compile(r"\bvelocity|velocities|moving at|speed of|m/s|meters per second\b", re.I)
FUTURE_RE = re.compile(
    r"\b(will|about to|likely|may|might|could|future|collide|sudden|unexpected|hidden)\b",
    re.I,
)
ACTION_RE = re.compile(
    r"\b(slow down|stop|yield|proceed|continue|maintain|prepare|be prepared|"
    r"safe following distance|lane change|avoid|brake|creep)\b",
    re.I,
)
BBOX_RE = re.compile(r"\bbox\b|bounding box|object id|waypoint|\[\s*0?\.\d+", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


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
            sample_id = row.get("sample_id")
            if sample_id:
                done.add(str(sample_id))
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
    model.tokenizer.padding_side = "left"
    processor.tokenizer = model.tokenizer
    processor.tokenizer.padding_side = "left"
    return model, processor


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def batch_extra_text(extra: dict[str, Any], key: str, batch_index: int) -> str:
    value = extra.get(key, [[""]])
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list) and batch_index < len(value):
        value = value[batch_index]
    while isinstance(value, list) and value:
        value = value[0]
    return "" if value is None else str(value).strip()


def answer_metrics(answer: str) -> dict[str, Any]:
    answer = str(answer or "").strip()
    words = re.findall(r"[A-Za-z0-9]+", answer)
    flags: list[str] = []
    if not answer:
        flags.append("empty")
    if len(words) < 5:
        flags.append("too_short")
    if COORDINATE_RE.search(answer) or BBOX_RE.search(answer):
        flags.append("coordinate_or_bbox")
    if VELOCITY_RE.search(answer):
        flags.append("velocity_value")
    if len(words) > 90:
        flags.append("too_long")
    if answer.lower() in {"yes", "no", "low", "medium", "high", "unknown"}:
        flags.append("single_token_answer")
    hard_reject = any(flag in flags for flag in ("empty", "too_short", "coordinate_or_bbox", "velocity_value", "single_token_answer"))
    return {
        "quality_flags": flags,
        "hard_reject": bool(hard_reject),
        "word_count": len(words),
        "has_coordinate": bool(COORDINATE_RE.search(answer) or BBOX_RE.search(answer)),
        "has_velocity_word": bool(VELOCITY_RE.search(answer)),
        "has_future_language": bool(FUTURE_RE.search(answer)),
        "has_action_language": bool(ACTION_RE.search(answer)),
    }


def load_batch_frame_tensors(rows: list[dict[str, Any]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load candidate frames with one video decode per batch clip/camera.

    Candidate rows are written as early/middle/late triples per clip. Loading a
    row independently reopens the same four camera mp4 files three times, which
    dominates teacher generation time. This loader groups all requested frame
    indices by ``(dataset_root, clip_id, chunk, camera feature)`` and then
    reassembles the exact per-row 4cam x 1frame tensor expected downstream.
    """
    requests: dict[tuple[str, str, int, str], set[int]] = {}
    row_specs: list[list[tuple[tuple[str, str, int, str], list[int], int]]] = []

    for row in rows:
        dataset_root = str(row["dataset_root"])
        clip_id = str(row["clip_id"])
        chunk = int(row["chunk"])
        specs: list[tuple[tuple[str, str, int, str], list[int], int]] = []
        for plan in row.get("frame_plan") or []:
            feature = str(plan["feature"])
            frame_indices = [int(v) for v in plan["frame_indices"]]
            key = (dataset_root, clip_id, chunk, feature)
            requests.setdefault(key, set()).update(frame_indices)
            specs.append((key, frame_indices, int(plan.get("camera_index", len(specs)))))
        if not specs:
            raise ValueError(f"row has no frame_plan: {row.get('sample_id')}")
        row_specs.append(specs)

    decoded: dict[tuple[str, str, int, str], dict[int, torch.Tensor]] = {}
    for key, wanted in requests.items():
        dataset_root, clip_id, chunk, feature = key
        sorted_indices = sorted(wanted)
        images = decode_video_frames(Path(dataset_root), clip_id, chunk, feature, sorted_indices)
        decoded[key] = {
            frame_index: pil_to_chw_uint8(image)
            for frame_index, image in zip(sorted_indices, images, strict=True)
        }

    loaded: list[tuple[torch.Tensor, torch.Tensor]] = []
    for specs in row_specs:
        camera_tensors: list[torch.Tensor] = []
        camera_indices: list[int] = []
        for key, frame_indices, camera_index in specs:
            camera_tensors.append(torch.stack([decoded[key][idx] for idx in frame_indices], dim=0))
            camera_indices.append(int(camera_index))
        loaded.append((torch.stack(camera_tensors, dim=0), torch.tensor(camera_indices, dtype=torch.long)))
    return loaded


def run_teacher_batch(
    *,
    model,
    processor,
    rows: list[dict[str, Any]],
    device: str,
    top_p: float,
    temperature: float,
    max_generation_length: int,
) -> list[dict[str, Any]]:
    messages = []
    batch_frames = load_batch_frame_tensors(rows)
    for row, (frames, camera_indices) in zip(rows, batch_frames, strict=True):
        messages.append(
            create_vqa_messages(
                frames=frames,
                camera_indices=camera_indices,
                question=str(row["question"]),
                answer_text=None,
            )
        )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = {"tokenized_data": to_device(inputs, device)}
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode(), autocast_context:
        extra = model.generate_text(
            data=model_inputs,
            top_p=float(top_p),
            temperature=float(temperature),
            num_samples=1,
            max_generation_length=int(max_generation_length),
        )
    outputs: list[dict[str, Any]] = []
    for batch_index, row in enumerate(rows):
        answer = batch_extra_text(extra, "answer", batch_index)
        short, short_flags = shorten_q2_answer(answer)
        metrics = answer_metrics(answer)
        outputs.append(
            {
                "answer": answer,
                "teacher_answer_short": short,
                "target_policy": {
                    "hard_target": "teacher_answer_short",
                    "soft_target": "alpamayo_teacher_forced_topk32_after_answer_start",
                    "shorten_flags": short_flags,
                },
                "teacher": {
                    "model": "Alpamayo-1.5-10B",
                    "temperature": float(temperature),
                    "temperature_label": f"t{float(temperature):.2f}".replace(".", "p"),
                    "top_p": float(top_p),
                    "selected_by": "single-temp-hard-gate",
                    **metrics,
                },
            }
        )
    return outputs


def text_judge_input_from_row(row: dict[str, Any]) -> dict[str, Any]:
    teacher = row["teacher"]
    return {
        "sample_id": row["sample_id"],
        "family": "Q2",
        "candidate_id": teacher["temperature_label"],
        "answer": row["answer"],
        "text_flags": {
            "has_coordinate": bool(teacher.get("has_coordinate")),
            "has_velocity_word": bool(teacher.get("has_velocity_word")),
            "has_future_language": bool(teacher.get("has_future_language")),
            "has_action_language": bool(teacher.get("has_action_language")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alpamayo-src", type=Path, default=DEFAULT_ALPAMAYO_SRC)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-generation-length", type=int, default=192)
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--summary-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = parse_args()
    started = time.time()
    rows = read_jsonl(args.candidate_jsonl)
    end = len(rows) if args.limit is None else min(len(rows), int(args.start_index) + int(args.limit))
    rows = rows[int(args.start_index) : end]
    args.output_root.mkdir(parents=True, exist_ok=True)
    records_path = args.output_root / "teacher_records.jsonl"
    hard_accept_path = args.output_root / "q2_hard_gate_accept.jsonl"
    hard_reject_path = args.output_root / "q2_hard_gate_reject.jsonl"
    text_judge_path = args.output_root / "q2_text_judge_input.jsonl"
    summary_path = args.output_root / "teacher_summary.json"
    done_ids = load_done_ids(records_path) | load_done_ids(hard_reject_path)
    model, processor = load_teacher(args)
    print(
        json.dumps(
            {
                "event": "teacher_ready",
                "candidate_jsonl": str(args.candidate_jsonl),
                "selected": len(rows),
                "batch_size": int(args.batch_size),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    counts: Counter[str] = Counter({"selected": len(rows)})
    pending: list[dict[str, Any]] = []

    def process_batch(batch: list[dict[str, Any]]) -> None:
        try:
            outputs = run_teacher_batch(
                model=model,
                processor=processor,
                rows=batch,
                device=str(args.device),
                top_p=float(args.top_p),
                temperature=float(args.temperature),
                max_generation_length=int(args.max_generation_length),
            )
        except Exception as exc:  # noqa: BLE001
            if len(batch) == 1:
                row = dict(batch[0])
                row["teacher_error"] = f"{type(exc).__name__}: {exc}"
                append_jsonl(hard_reject_path, row)
                counts["failed_generation"] += 1
                return
            mid = len(batch) // 2
            counts["batch_split_after_error"] += 1
            process_batch(batch[:mid])
            process_batch(batch[mid:])
            return
        for row, output in zip(batch, outputs, strict=True):
            out = dict(row)
            out.update(output)
            append_jsonl(records_path, out)
            counts["generated"] += 1
            if out["teacher"].get("hard_reject"):
                append_jsonl(hard_reject_path, out)
                counts["hard_reject"] += 1
            else:
                append_jsonl(hard_accept_path, out)
                append_jsonl(text_judge_path, text_judge_input_from_row(out))
                counts["hard_accept"] += 1

    for row_index, row in enumerate(rows, start=1):
        if str(row["sample_id"]) in done_ids:
            counts["skipped_existing"] += 1
            continue
        pending.append(row)
        if len(pending) >= int(args.batch_size):
            process_batch(pending)
            pending.clear()
        if row_index % max(1, int(args.summary_every)) == 0:
            if pending:
                process_batch(pending)
                pending.clear()
            summary = {
                "updated_at": utc_now(),
                "candidate_jsonl": str(args.candidate_jsonl),
                "output_root": str(args.output_root),
                "counts": dict(counts),
                "elapsed_sec": round(time.time() - started, 3),
            }
            save_json(summary_path, summary)
            print(json.dumps({"event": "teacher_progress", **summary}, ensure_ascii=True), flush=True)
    if pending:
        process_batch(pending)
    summary = {
        "updated_at": utc_now(),
        "candidate_jsonl": str(args.candidate_jsonl),
        "output_root": str(args.output_root),
        "counts": dict(counts),
        "elapsed_sec": round(time.time() - started, 3),
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
