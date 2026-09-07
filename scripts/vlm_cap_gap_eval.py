#!/usr/bin/env python3
"""Evaluate vanilla Cosmos-Reason2 VLM capability gaps on human-OOD clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/home/pm97/workspace/dataset/physical_ai_av_ood_dataset")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "eval" / "vlm_cap_gap_human_ood_20260701"
DEFAULT_MODEL_PATHS = {
    "2b": Path("/home/pm97/workspace/sukim/base_weights/cosmos-reason-2b"),
    "8b": Path("/home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-8B"),
    "32b": Path("/home/pm97/workspace/sukim/base_weights/Cosmos-Reason2-32B"),
}

SYSTEM_PROMPT = """You are analyzing a short driving video recorded from the ego vehicle's front-wide camera.
Think step by step about what is actually visible, then give a structured answer.

Respond in EXACTLY this format and nothing else:
<think>
your step-by-step reasoning about the scene
</think>
<answer>
{a single valid JSON object, no markdown, no code fences}
</answer>

Rules:
- The JSON must match the schema given in the question.
- Every JSON object must include "confidence" (one of "high","medium","low") and
  "abstain" (true ONLY if the video lacks enough evidence to answer reliably).
- Never claim to see an object, person, or event you cannot actually verify in this video.
- If unsure, set "abstain": true and "confidence": "low" rather than guessing."""

TASKS = {
    "T1_pos": {
        "task": "T1",
        "question": (
            'Is there a pedestrian in this scene? If yes, how many, and where are they\n'
            "(left / center / right of the ego lane)?\n"
            'schema: {"present": bool, "count": int,\n'
            '         "positions": ["left"|"center"|"right", ...],\n'
            '         "confidence": str, "abstain": bool}'
        ),
        "key": "present",
    },
    "T1_neg": {
        "task": "T1",
        "question": (
            'Is there a {negative_class} in this scene? If yes, how many, and where are they\n'
            "(left / center / right of the ego lane)?\n"
            'schema: {"present": bool, "count": int,\n'
            '         "positions": ["left"|"center"|"right", ...],\n'
            '         "confidence": str, "abstain": bool}'
        ),
        "key": "present",
    },
    "T2": {
        "task": "T2",
        "question": (
            "Is the most relevant person ahead about to enter the ego vehicle's path?\n"
            " If the evidence is insufficient, abstain.\n"
            'schema: {"decision": "entering"|"not_entering"|"uncertain",\n'
            '         "evidence": str, "confidence": str, "abstain": bool}'
        ),
        "key": "decision",
    },
    "T3": {
        "task": "T3",
        "question": (
            "Describe any human behavior in this scene that is unusual or an edge case for driving\n"
            " (e.g. person in the roadway, unexpected motion, unusual posture). If none, say so.\n"
            'schema: {"ood_present": bool, "description": str,\n'
            '         "severity": "low"|"medium"|"high", "confidence": str, "abstain": bool}'
        ),
        "key": "ood_present",
    },
    "T4": {
        "task": "T4",
        "question": (
            "Given the humans in this scene, should the ego vehicle stop, yield, or proceed?\n"
            " Justify using specific evidence from the video.\n"
            'schema: {"action": "stop"|"yield"|"proceed",\n'
            '         "reason": str, "confidence": str, "abstain": bool}'
        ),
        "key": "action",
    },
}

HUMAN_CLUSTER_RE = re.compile(r"PEDESTRIAN|CYCLISTS|MICROMOBILITY", re.I)
PEDESTRIAN_CLUSTER_RE = re.compile(r"PEDESTRIAN", re.I)
CYCLIST_CLUSTER_RE = re.compile(r"CYCLISTS|MICROMOBILITY", re.I)
ENTERING_RE = re.compile(r"\b(cross|crossing|roadway|in the road|into .*path|enter|yield)\b", re.I)
STOP_RE = re.compile(r"\bstop|red traffic light|red light\b", re.I)
YIELD_RE = re.compile(r"\byield|deceler|slow|maintain a safe distance|wait\b", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_events(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "none":
            return []
        try:
            decoded = json.loads(text)
        except Exception:
            return []
        return parse_events(decoded)
    return []


def event_cot(events: list[dict[str, Any]]) -> str:
    cots = []
    for item in events:
        cot = str(item.get("cot") or "").strip()
        if not cot or cot == "__DISCARDED__":
            continue
        cots.append(cot)
    return " ".join(cots)


def event_center_seconds(events: list[dict[str, Any]]) -> float:
    timestamps = [
        float(item["event_start_timestamp"]) / 1_000_000.0
        for item in events
        if item.get("event_start_timestamp") not in (None, "")
    ]
    if timestamps:
        return float(np.median(timestamps))
    frames = [float(item["event_start_frame"]) for item in events if item.get("event_start_frame") not in (None, "")]
    if frames:
        return float(np.median(frames)) / 10.0
    return 10.0


def infer_split_tag(event_cluster: str, events: list[dict[str, Any]]) -> str:
    cot = event_cot(events)
    if events and re.search(r"pedestrian|cyclist|person|people|cross|road|yield|stop", cot, re.I):
        return "clear"
    return "ambiguous"


def infer_t2_gt(events: list[dict[str, Any]], split_tag: str) -> str | None:
    if split_tag != "clear":
        return None
    cot = event_cot(events)
    if ENTERING_RE.search(cot):
        return "entering"
    if re.search(r"sidewalk|standing|near|parked", cot, re.I):
        return "not_entering"
    return None


def infer_t4_gt(events: list[dict[str, Any]]) -> str | None:
    cot = event_cot(events)
    if not cot:
        return None
    if STOP_RE.search(cot):
        return "stop"
    if YIELD_RE.search(cot):
        return "yield"
    return "proceed"


def camera_zip_path(dataset_root: Path, chunk: int) -> Path:
    return (
        dataset_root
        / "camera"
        / "camera_front_wide_120fov"
        / f"camera_front_wide_120fov.chunk_{int(chunk):04d}.zip"
    )


def camera_member_name(clip_id: str) -> str:
    return f"{clip_id}.camera_front_wide_120fov.mp4"


def build_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    clip_index = pd.read_parquet(dataset_root / "clip_index.parquet")
    reasoning = pd.read_parquet(dataset_root / "reasoning" / "ood_reasoning.parquet")
    reasoning = reasoning[reasoning["feature"].astype(str).eq("camera_front_wide_120fov")]
    reasoning = reasoning[reasoning["event_cluster"].astype(str).str.contains(HUMAN_CLUSTER_RE, na=False)]

    rows: list[dict[str, Any]] = []
    for clip_id, row in reasoning.iterrows():
        if clip_id not in clip_index.index:
            continue
        valid = bool(clip_index.loc[clip_id, "clip_is_valid"])
        if not valid:
            continue
        chunk = int(clip_index.loc[clip_id, "chunk"])
        zip_path = camera_zip_path(dataset_root, chunk)
        if not zip_path.exists():
            continue
        event_cluster = str(row["event_cluster"])
        events = parse_events(row.get("events"))
        cot = event_cot(events)
        split_tag = infer_split_tag(event_cluster, events)
        has_pedestrian = bool(PEDESTRIAN_CLUSTER_RE.search(event_cluster))
        has_cyclist = bool(CYCLIST_CLUSTER_RE.search(event_cluster))
        rows.append(
            {
                "clip_id": str(clip_id),
                "chunk": chunk,
                "dataset_split": str(row.get("split") or clip_index.loc[clip_id, "split"]),
                "event_cluster": event_cluster,
                "events": events,
                "event_cot": cot,
                "center_s": event_center_seconds(events),
                "split_tag": split_tag,
                "negative_class": "animal",
                "gt": {
                    "pedestrian_present": has_pedestrian,
                    "cyclist_present": has_cyclist,
                    "animal_present": False,
                    "t2_decision": infer_t2_gt(events, split_tag),
                    "t4_action_heuristic": infer_t4_gt(events),
                },
            }
        )

    rows = sorted(rows, key=lambda item: (item["split_tag"], item["clip_id"]))
    rng = np.random.default_rng(int(args.seed))
    clear = [row for row in rows if row["split_tag"] == "clear"]
    ambiguous = [row for row in rows if row["split_tag"] == "ambiguous"]
    rng.shuffle(clear)
    rng.shuffle(ambiguous)
    limit = int(args.limit_clips) if args.limit_clips else len(rows)
    half = limit // 2
    selected = clear[: min(len(clear), half)] + ambiguous[: max(0, limit - min(len(clear), half))]
    if len(selected) < limit:
        selected += clear[min(len(clear), half) : limit - len(selected) + min(len(clear), half)]
    selected = selected[:limit]
    rng.shuffle(selected)
    for index, row in enumerate(selected):
        row["eval_index"] = index

    out_dir = Path(args.output_dir)
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists: {manifest}; pass --overwrite")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    write_json(
        out_dir / "manifest_summary.json",
        {
            "created_at": utc_now(),
            "dataset_root": str(dataset_root),
            "candidate_human_ood": len(rows),
            "selected": len(selected),
            "split_tag_counts": dict(Counter(row["split_tag"] for row in selected)),
            "event_cluster_counts": dict(Counter(row["event_cluster"] for row in selected)),
            "manifest": str(manifest),
        },
    )
    print(json.dumps({"manifest": str(manifest), "selected": len(selected)}, sort_keys=True), flush=True)


def clamp_window(center_s: float, duration_s: float, window_s: float) -> float:
    if duration_s <= window_s:
        return 0.0
    return float(max(0.0, min(center_s - window_s / 2.0, duration_s - window_s)))


def extract_trimmed_video(row: dict[str, Any], *, dataset_root: Path, cache_dir: Path, fps: float, seconds: float) -> Path:
    clip_id = str(row["clip_id"])
    chunk = int(row["chunk"])
    out_path = cache_dir / f"{int(row.get('eval_index', 0)):05d}_{clip_id}_frontwide_{seconds:g}s_{fps:g}fps.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    zip_path = camera_zip_path(dataset_root, chunk)
    member = camera_member_name(clip_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vlm_cap_gap_") as tmp_dir:
        tmp_src = Path(tmp_dir) / f"{clip_id}.mp4"
        with zipfile.ZipFile(zip_path) as zf:
            if member not in zf.namelist():
                raise FileNotFoundError(f"{member} not in {zip_path}")
            with zf.open(member) as src, tmp_src.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        cap = cv2.VideoCapture(str(tmp_src))
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"Could not decode {tmp_src}")
        duration_s = total_frames / src_fps
        start_s = clamp_window(float(row.get("center_s") or 10.0), duration_s, seconds)
        num_frames = int(round(seconds * fps))
        indices = (start_s * src_fps + np.arange(num_frames) * (src_fps / fps)).round().astype(int)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(args_width := 640), int(args_height := 360)),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(min(max(idx, 0), total_frames - 1)))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (args_width, args_height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
        cap.release()
        writer.release()
    return out_path


def load_model(model_path: Path, *, device_map: str, attn: str):
    local = model_path.exists()
    processor_path = model_path / "processor" if (model_path / "processor").exists() else model_path
    processor = AutoProcessor.from_pretrained(
        str(processor_path),
        trust_remote_code=True,
        local_files_only=local,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn,
        trust_remote_code=True,
        local_files_only=local,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, processor


def tensor_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_messages(video_path: Path, question: str, fps: float) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path), "fps": float(fps)},
                {"type": "text", "text": question},
            ],
        },
    ]


def generate_one(
    model: Any,
    processor: Any,
    *,
    video_path: Path,
    question: str,
    fps: float,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> str:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.manual_seed(int(seed))
    messages = build_messages(video_path, question, fps)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=float(fps),
    )
    prompt_len = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, tensor_device(model))
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(temperature > 0),
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs.update({"temperature": float(temperature), "top_p": 1.0, "top_k": 0})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        output = model.generate(**inputs, **gen_kwargs)
    decoded = processor.tokenizer.decode(output[0, prompt_len:].detach().cpu().tolist(), skip_special_tokens=False)
    return decoded


def generate_batch(
    model: Any,
    processor: Any,
    *,
    items: list[dict[str, Any]],
    fps: float,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> list[str]:
    if not items:
        return []
    if len(items) == 1:
        item = items[0]
        return [
            generate_one(
                model,
                processor,
                video_path=Path(item["video_path"]),
                question=str(item["question"]),
                fps=fps,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
        ]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.manual_seed(int(seed))
    messages = [
        build_messages(Path(item["video_path"]), str(item["question"]), fps)
        for item in items
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=float(fps),
        padding=True,
    )
    prompt_len = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, tensor_device(model))
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(temperature > 0),
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        gen_kwargs.update({"temperature": float(temperature), "top_p": 1.0, "top_k": 0})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        output = model.generate(**inputs, **gen_kwargs)
    return processor.tokenizer.batch_decode(
        [row[prompt_len:].detach().cpu().tolist() for row in output],
        skip_special_tokens=False,
    )


def parse_response(text: str) -> dict[str, Any]:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    payload = answer_match.group(1).strip() if answer_match else text.strip()
    answer = None
    parse_error = None
    try:
        answer = json.loads(payload)
    except Exception as exc:
        parse_error = str(exc)
        match = re.search(r"\{.*\}", payload, re.S)
        if match:
            try:
                answer = json.loads(match.group(0))
                parse_error = None
            except Exception as exc2:
                parse_error = str(exc2)
    return {
        "think": think_match.group(1).strip() if think_match else "",
        "answer": answer,
        "parse_error": parse_error,
        "raw": text,
    }


def task_question(task_id: str, row: dict[str, Any]) -> str:
    question = TASKS[task_id]["question"]
    return question.replace("{negative_class}", str(row.get("negative_class") or "animal"))


def expected_for_task(task_id: str, row: dict[str, Any]) -> Any:
    gt = row.get("gt") or {}
    if task_id == "T1_pos":
        return bool(gt.get("pedestrian_present"))
    if task_id == "T1_neg":
        return bool(gt.get(f"{row.get('negative_class', 'animal')}_present", False))
    if task_id == "T2":
        return gt.get("t2_decision")
    if task_id == "T4":
        return gt.get("t4_action_heuristic")
    return None


def prediction_row(
    *,
    model_key: str,
    model_path: Path,
    row: dict[str, Any],
    video_path: Path,
    task_id: str,
    question: str,
    sample_idx: int,
    temperature: float,
    raw: str,
    parsed: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "model_key": model_key,
        "model_path": str(model_path),
        "clip_id": row["clip_id"],
        "eval_index": row.get("eval_index"),
        "chunk": row.get("chunk"),
        "event_cluster": row.get("event_cluster"),
        "split_tag": row.get("split_tag"),
        "dataset_split": row.get("dataset_split"),
        "center_s": row.get("center_s"),
        "video_path": str(video_path),
        "task_id": task_id,
        "task": TASKS[task_id]["task"],
        "question": question,
        "expected": expected_for_task(task_id, row),
        "sample_idx": sample_idx,
        "temperature": temperature,
        "raw": raw,
        "think": parsed["think"],
        "answer": parsed["answer"],
        "parse_error": parsed["parse_error"],
        "error": error,
        "gt": row.get("gt"),
    }


def run_model(args: argparse.Namespace) -> None:
    model_key = str(args.model_key)
    model_path = Path(args.model_path or DEFAULT_MODEL_PATHS[model_key])
    manifest = read_jsonl(Path(args.manifest))
    if args.limit_clips:
        manifest = manifest[: int(args.limit_clips)]
    task_ids = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    out_dir = Path(args.output_dir) / "runs" / model_key
    pred_path = out_dir / "predictions.jsonl"
    if pred_path.exists() and args.overwrite:
        pred_path.unlink()
    done = {
        (row["clip_id"], row["task_id"], int(row["sample_idx"]))
        for row in read_jsonl(pred_path)
        if str(row.get("model_key")) == model_key
    }
    model, processor = load_model(model_path, device_map=str(args.device_map), attn=str(args.attn_implementation))
    started = time.time()
    wrote = 0
    batch_size = max(1, int(args.batch_size))
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal wrote, pending
        batch = pending
        if not batch:
            return
        pending = []
        temp = float(batch[0]["temperature"])
        try:
            raws = generate_batch(
                model,
                processor,
                items=batch,
                fps=float(args.fps),
                temperature=temp,
                max_new_tokens=int(args.max_new_tokens),
                seed=int(batch[0]["seed"]),
            )
            errors = [None] * len(batch)
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raws = []
            errors = []
            for item in batch:
                try:
                    raws.append(
                        generate_one(
                            model,
                            processor,
                            video_path=Path(item["video_path"]),
                            question=str(item["question"]),
                            fps=float(args.fps),
                            temperature=float(item["temperature"]),
                            max_new_tokens=int(args.max_new_tokens),
                            seed=int(item["seed"]),
                        )
                    )
                    errors.append(None)
                except Exception as inner_exc:
                    raws.append("")
                    errors.append(repr(inner_exc) if len(batch) == 1 else f"batch_error={exc!r}; fallback_error={inner_exc!r}")
        out_rows: list[dict[str, Any]] = []
        for item, raw, error in zip(batch, raws, errors, strict=True):
            parsed = parse_response(raw) if error is None else {"think": "", "answer": None, "parse_error": str(error), "raw": ""}
            out_rows.append(
                prediction_row(
                    model_key=model_key,
                    model_path=model_path,
                    row=item["row"],
                    video_path=Path(item["video_path"]),
                    task_id=str(item["task_id"]),
                    question=str(item["question"]),
                    sample_idx=int(item["sample_idx"]),
                    temperature=float(item["temperature"]),
                    raw=raw,
                    parsed=parsed,
                    error=error,
                )
            )
        append_jsonl(pred_path, out_rows)
        wrote += len(out_rows)
        if wrote % int(args.log_every) < len(out_rows):
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "model": model_key,
                        "rows_written": wrote,
                        "clip_i": batch[-1]["clip_i"],
                        "elapsed_sec": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    for clip_i, row in enumerate(manifest):
        video_path = extract_trimmed_video(
            row,
            dataset_root=Path(args.dataset_root),
            cache_dir=Path(args.output_dir) / "runs" / model_key / "video_cache",
            fps=float(args.fps),
            seconds=float(args.seconds),
        )
        for task_id in task_ids:
            sample_plan = [(0, 0.0)] + [(i + 1, float(args.sample_temperature)) for i in range(int(args.sample_n))]
            for sample_idx, temp in sample_plan:
                key = (row["clip_id"], task_id, sample_idx)
                if key in done:
                    continue
                question = task_question(task_id, row)
                item = {
                    "row": row,
                    "clip_i": clip_i,
                    "video_path": str(video_path),
                    "task_id": task_id,
                    "question": question,
                    "sample_idx": sample_idx,
                    "temperature": temp,
                    "seed": int(args.seed) + int(row.get("eval_index", clip_i)) * 101 + sample_idx,
                }
                if pending and (float(pending[0]["temperature"]) != float(temp) or len(pending) >= batch_size):
                    flush_pending()
                pending.append(item)
                if len(pending) >= batch_size:
                    flush_pending()
    flush_pending()
    write_json(
        out_dir / "run_summary.json",
        {
            "created_at": utc_now(),
            "model_key": model_key,
            "model_path": str(model_path),
            "manifest": str(args.manifest),
            "tasks": task_ids,
            "clips": len(manifest),
            "sample_n": int(args.sample_n),
            "predictions": str(pred_path),
            "elapsed_sec": round(time.time() - started, 3),
            "new_rows_written": wrote,
        },
    )
    print(json.dumps({"predictions": str(pred_path), "new_rows_written": wrote}, sort_keys=True), flush=True)


def answer_field(row: dict[str, Any], field: str) -> Any:
    answer = row.get("answer")
    if not isinstance(answer, dict):
        return None
    return answer.get(field)


def confidence(row: dict[str, Any]) -> str | None:
    value = answer_field(row, "confidence")
    if isinstance(value, str):
        value = value.strip().lower()
    return value if value in {"high", "medium", "low"} else None


def abstain(row: dict[str, Any]) -> bool | None:
    value = answer_field(row, "abstain")
    return value if isinstance(value, bool) else None


def normalized_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in {"stop", "yield", "proceed"} else None


def task_key_value(row: dict[str, Any]) -> Any:
    if row.get("parse_error") or not isinstance(row.get("answer"), dict):
        return "__invalid__"
    task_id = row["task_id"]
    if task_id in {"T1_pos", "T1_neg"}:
        value = answer_field(row, "present")
        return value if value is not None else "__missing_present__"
    if task_id == "T2":
        value = answer_field(row, "decision")
        return value if value is not None else "__missing_decision__"
    if task_id == "T3":
        value = answer_field(row, "ood_present")
        return value if value is not None else "__missing_ood_present__"
    if task_id == "T4":
        value = normalized_action(answer_field(row, "action"))
        return value if value is not None else "__missing_action__"
    return "__unknown_task__"


def is_valid_task_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.startswith("__"):
        return False
    return True


def bool_match(value: Any, expected: Any) -> bool | None:
    if not isinstance(value, bool) or not isinstance(expected, bool):
        return None
    return value == expected


def row_correctness(row: dict[str, Any], refs_32b: dict[tuple[str, str], dict[str, Any]]) -> bool | None:
    task_id = row["task_id"]
    if row.get("sample_idx") != 0:
        return None
    if task_id in {"T3", "T4"} and row.get("model_key") == "32b":
        return None
    if row.get("parse_error") or not isinstance(row.get("answer"), dict):
        return False
    if task_id in {"T1_pos", "T1_neg"}:
        value = answer_field(row, "present")
        expected = row.get("expected")
        if not isinstance(expected, bool):
            return None
        return value == expected if isinstance(value, bool) else False
    if task_id == "T2" and row.get("expected") in {"entering", "not_entering", "uncertain"}:
        value = answer_field(row, "decision")
        return value == row.get("expected") if isinstance(value, str) else False
    if task_id in {"T3", "T4"}:
        ref = refs_32b.get((row["clip_id"], task_id))
        if not ref:
            return None
        ref_value = task_key_value(ref)
        if not is_valid_task_value(ref_value):
            return None
        row_value = task_key_value(row)
        if not is_valid_task_value(row_value):
            return False
        return row_value == ref_value
    return None


def bootstrap_ci(values: list[float], *, seed: int = 17, n_boot: int = 400) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mean_ci(values: list[float]) -> dict[str, Any]:
    lo, hi = bootstrap_ci(values)
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def balanced_quality_ci(ambiguous_quality: list[float], clear_quality: list[float]) -> dict[str, Any] | None:
    if not ambiguous_quality or not clear_quality:
        return None
    rng = np.random.default_rng(23)
    amb = np.asarray(ambiguous_quality, dtype=np.float64)
    clear_good = np.asarray(clear_quality, dtype=np.float64)
    mean = 0.5 * (float(np.mean(amb)) + float(np.mean(clear_good)))
    if len(amb) == 1 and len(clear_good) == 1:
        lo = hi = mean
    else:
        means = [
            0.5
            * (
                float(np.mean(rng.choice(amb, size=len(amb), replace=True)))
                + float(np.mean(rng.choice(clear_good, size=len(clear_good), replace=True)))
            )
            for _ in range(400)
        ]
        lo = float(np.percentile(means, 2.5))
        hi = float(np.percentile(means, 97.5))
    return {
        "n": len(ambiguous_quality) + len(clear_quality),
        "n_ambiguous": len(ambiguous_quality),
        "n_clear": len(clear_quality),
        "mean": mean,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def summarize_model(rows: list[dict[str, Any]], refs_32b: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    greedy = [row for row in rows if int(row.get("sample_idx", -1)) == 0]
    parsed_rate = mean_ci([0.0 if row.get("parse_error") or not isinstance(row.get("answer"), dict) else 1.0 for row in greedy])
    t1_pos = [row_correctness(row, refs_32b) for row in greedy if row["task_id"] == "T1_pos"]
    t1_neg_present = [answer_field(row, "present") for row in greedy if row["task_id"] == "T1_neg"]
    t1_neg_hallucination = [1.0 if value is True else 0.0 for value in t1_neg_present if isinstance(value, bool)]
    t2_amb_rows = [row for row in greedy if row["task_id"] == "T2" and row.get("split_tag") == "ambiguous"]
    t2_clear_rows = [row for row in greedy if row["task_id"] == "T2" and row.get("split_tag") == "clear"]
    t2_amb_values = [1.0 if abstain(row) is True else 0.0 for row in t2_amb_rows]
    t2_clear_values = [1.0 if abstain(row) is True else 0.0 for row in t2_clear_rows]
    t2_amb_quality = [1.0 if abstain(row) is True else 0.0 for row in t2_amb_rows]
    t2_clear_quality = [1.0 if abstain(row) is False else 0.0 for row in t2_clear_rows]
    t4_ref = [row_correctness(row, refs_32b) for row in greedy if row["task_id"] == "T4"]
    t3_ref = [row_correctness(row, refs_32b) for row in greedy if row["task_id"] == "T3"]

    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("sample_idx", 0)) > 0:
            by_group[(row["clip_id"], row["task_id"])].append(row)
    consistency_values: list[float] = []
    for group_rows in by_group.values():
        values = [task_key_value(row) for row in group_rows]
        if not values:
            continue
        most_common = Counter(values).most_common(1)[0][1]
        consistency_values.append(most_common / len(values))

    ece_rows = []
    bucket_values: dict[str, list[float]] = defaultdict(list)
    for row in greedy:
        correct = row_correctness(row, refs_32b)
        conf = confidence(row)
        if correct is not None and conf is not None:
            bucket_values[conf].append(1.0 if correct else 0.0)
            ece_rows.append((conf, 1.0 if correct else 0.0))
    conf_nominal = {"low": 0.2, "medium": 0.5, "high": 0.85}
    ece = 0.0
    for conf, vals in bucket_values.items():
        ece += (len(vals) / max(1, len(ece_rows))) * abs(float(np.mean(vals)) - conf_nominal[conf])

    return {
        "greedy_n": len(greedy),
        "parsed_json_rate": parsed_rate,
        "T1_accuracy_present": mean_ci([1.0 if v else 0.0 for v in t1_pos if v is not None]),
        "T1_hallucination_rate_negative": mean_ci(t1_neg_hallucination),
        "T2_abstain_ambiguous": mean_ci(t2_amb_values),
        "T2_abstain_clear": mean_ci(t2_clear_values),
        "T2_abstention_quality": balanced_quality_ci(t2_amb_quality, t2_clear_quality),
        "ECE_lite": {"n": len(ece_rows), "ece": ece, "buckets": {k: mean_ci(v) for k, v in bucket_values.items()}},
        "consistency": mean_ci(consistency_values),
        "T4_ref_agreement_32b": mean_ci([1.0 if v else 0.0 for v in t4_ref if v is not None]),
        "T3_ref_agreement_32b": mean_ci([1.0 if v else 0.0 for v in t3_ref if v is not None]),
    }


def score(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for model_key in str(args.models).split(","):
        model_key = model_key.strip()
        if not model_key:
            continue
        rows.extend(read_jsonl(out_dir / "runs" / model_key / "predictions.jsonl"))
    refs_32b = {
        (row["clip_id"], row["task_id"]): row
        for row in rows
        if row.get("model_key") == "32b" and int(row.get("sample_idx", -1)) == 0 and row.get("task_id") in {"T3", "T4"}
    }
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model_key"))].append(row)
    summaries = {model: summarize_model(model_rows, refs_32b) for model, model_rows in sorted(by_model.items())}

    # Pairwise 2B vs 8B proxy winrate against 32B reference for T3/T4.
    proxy = {}
    for task_id in ["T3", "T4"]:
        wins = Counter()
        comparable = 0
        by_model_task = {
            model: {
                row["clip_id"]: row
                for row in model_rows
                if row["task_id"] == task_id and int(row.get("sample_idx", -1)) == 0
            }
            for model, model_rows in by_model.items()
        }
        for clip_id, ref in by_model_task.get("32b", {}).items():
            r2 = by_model_task.get("2b", {}).get(clip_id)
            r8 = by_model_task.get("8b", {}).get(clip_id)
            if not r2 or not r8:
                continue
            ref_value = task_key_value(ref)
            if not is_valid_task_value(ref_value):
                continue
            comparable += 1
            v2 = task_key_value(r2)
            v8 = task_key_value(r8)
            c2 = is_valid_task_value(v2) and v2 == ref_value
            c8 = is_valid_task_value(v8) and v8 == ref_value
            if c2 and not c8:
                wins["2b"] += 1
            elif c8 and not c2:
                wins["8b"] += 1
            else:
                wins["tie"] += 1
        proxy[task_id] = {"n": comparable, "wins": dict(wins)}

    payload = {
        "created_at": utc_now(),
        "output_dir": str(out_dir),
        "models": sorted(by_model),
        "manifest_summary": json.loads((out_dir / "manifest_summary.json").read_text(encoding="utf-8"))
        if (out_dir / "manifest_summary.json").exists()
        else None,
        "summaries": summaries,
        "t3_t4_judge_winrate_proxy_vs_32b": proxy,
    }
    write_json(out_dir / "score_summary.json", payload)
    write_report(out_dir / "report.md", payload)
    print(json.dumps({"score_summary": str(out_dir / "score_summary.json"), "report": str(out_dir / "report.md")}, sort_keys=True), flush=True)


def fmt_metric(item: Any) -> str:
    if not isinstance(item, dict):
        return "-" if item is None else f"{item:.3f}" if isinstance(item, float) else str(item)
    if item.get("mean") is None:
        return "-"
    ci_low = item.get("ci95_low")
    ci_high = item.get("ci95_high")
    ci = ""
    if isinstance(ci_low, (float, int)) and isinstance(ci_high, (float, int)):
        ci = f", 95% CI {ci_low:.3f}-{ci_high:.3f}"
    return f"{item['mean']:.3f} (n={item['n']}{ci})"


def value(summary: dict[str, Any], key: str) -> float | None:
    item = summary.get(key)
    if isinstance(item, dict):
        return item.get("mean")
    if isinstance(item, (float, int)):
        return float(item)
    return None


def ece_value(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    ece = summary.get("ECE_lite")
    if isinstance(ece, dict) and isinstance(ece.get("ece"), (float, int)):
        return float(ece["ece"])
    return None


def recommendation(metric: str, s2: dict[str, Any] | None, s8: dict[str, Any] | None, s32: dict[str, Any] | None) -> str:
    if not s2 or not s8:
        return "insufficient"
    if metric == "T1":
        a2 = value(s2, "T1_accuracy_present")
        a8 = value(s8, "T1_accuracy_present")
        a32 = value(s32 or {}, "T1_accuracy_present")
        h2 = value(s2, "T1_hallucination_rate_negative")
        h8 = value(s8, "T1_hallucination_rate_negative")
        h32 = value(s32 or {}, "T1_hallucination_rate_negative")
        e2 = ece_value(s2)
        if a8 is not None and a32 is not None and a8 < 0.65 and a32 < 0.65:
            return "capacity ceiling/data issue"
        if h8 is not None and h32 is not None and h8 > 0.20 and h32 > 0.20:
            return "capacity ceiling/data issue"
        if a2 is not None and a2 >= 0.75 and e2 is not None and e2 > 0.20:
            return "target pre-stage"
        if a2 is not None and a8 is not None and (a8 - a2) < 0.05 and (h2 is not None and h2 < 0.10):
            return "skip"
        if a2 is not None and a8 is not None and (a8 - a2) >= 0.10:
            return "target pre-stage"
    if metric == "T2":
        q2 = value(s2, "T2_abstention_quality")
        q8 = value(s8, "T2_abstention_quality")
        q32 = value(s32 or {}, "T2_abstention_quality")
        amb2 = value(s2, "T2_abstain_ambiguous")
        clear2 = value(s2, "T2_abstain_clear")
        if q2 is not None and q8 is not None:
            if q32 is not None and q8 < 0.55 and q32 < 0.55:
                return "capacity ceiling/data issue"
            if amb2 is not None and amb2 < 0.70:
                return "target pre-stage"
            if clear2 is not None and clear2 > 0.20:
                return "target pre-stage"
            if q2 >= 0.80 and q8 >= 0.80 and abs(q8 - q2) < 0.05:
                return "skip"
            if (q8 - q2) >= 0.10:
                return "target pre-stage"
    if metric == "T4":
        r2 = value(s2, "T4_ref_agreement_32b")
        r8 = value(s8, "T4_ref_agreement_32b")
        if r2 is not None and r8 is not None:
            if r2 >= 0.80 and (r8 - r2) < 0.05:
                return "skip"
            if (r8 - r2) >= 0.10:
                return "target pre-stage"
            if r8 < 0.55:
                return "capacity ceiling/data issue"
    return "target pre-stage" if s2 else "insufficient"


def recommendation_basis(metric: str, s2: dict[str, Any] | None, s8: dict[str, Any] | None, s32: dict[str, Any] | None) -> str:
    if not s2 or not s8:
        return "missing 2B/8B summaries"
    if metric == "T1":
        return (
            f"2B presence={fmt_metric(s2.get('T1_accuracy_present'))}, "
            f"8B presence={fmt_metric(s8.get('T1_accuracy_present'))}, "
            f"2B neg hallucination={fmt_metric(s2.get('T1_hallucination_rate_negative'))}, "
            f"2B ECE={ece_value(s2)}"
        )
    if metric == "T2":
        return (
            f"2B ambiguous abstain={fmt_metric(s2.get('T2_abstain_ambiguous'))}, "
            f"2B clear abstain={fmt_metric(s2.get('T2_abstain_clear'))}, "
            f"2B quality={fmt_metric(s2.get('T2_abstention_quality'))}, "
            f"8B quality={fmt_metric(s8.get('T2_abstention_quality'))}, "
            f"32B quality={fmt_metric((s32 or {}).get('T2_abstention_quality'))}"
        )
    if metric == "T4":
        return (
            f"2B ref agreement={fmt_metric(s2.get('T4_ref_agreement_32b'))}, "
            f"8B ref agreement={fmt_metric(s8.get('T4_ref_agreement_32b'))}; "
            "blind judge section is appended separately after Codex judge"
        )
    return "no basis"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summaries = payload["summaries"]
    manifest_summary = payload.get("manifest_summary") or {}
    models = ["2b", "8b", "32b"]
    labels = {"2b": "2B", "8b": "8B", "32b": "32B"}
    lines = []
    lines.append("# Vanilla CR2 Human-OOD VLM Capability Gap")
    lines.append("")
    lines.append(f"- Created: `{payload['created_at']}`")
    lines.append(f"- Output root: `{payload['output_dir']}`")
    lines.append("- Protocol: front-wide 8s trim, 4fps, 32 frames, exact English prompts, BF16.")
    lines.append("- Note: local OOD copy has clip-level reasoning metadata but no NCore cuboid GT in this workspace; T1 count/position are logged but only presence/negative hallucination are machine-scored here.")
    lines.append("")
    if manifest_summary:
        lines.append("## Data")
        lines.append("")
        lines.append(f"- Dataset root: `{manifest_summary.get('dataset_root')}`")
        lines.append(f"- Candidate human-OOD clips: `{manifest_summary.get('candidate_human_ood')}`")
        lines.append(f"- Selected clips: `{manifest_summary.get('selected')}`")
        lines.append(f"- Split tags: `{manifest_summary.get('split_tag_counts')}`")
        lines.append(f"- Event clusters: `{manifest_summary.get('event_cluster_counts')}`")
        lines.append("- Gold source: OOD metadata/heuristics for T1/T2, 32B field agreement for T3/T4 proxy, and blind Codex judge for T3/T4 pairwise once appended.")
        lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | 2B | 8B | 32B | delta 8B-2B |")
    lines.append("|---|---:|---:|---:|---:|")
    metric_map = [
        ("parsed_json_rate", "JSON parse rate"),
        ("T1_accuracy_present", "T1 presence accuracy"),
        ("T1_hallucination_rate_negative", "T1 negative hallucination rate"),
        ("T2_abstain_ambiguous", "T2 abstain ambiguous"),
        ("T2_abstain_clear", "T2 abstain clear"),
        ("T2_abstention_quality", "T2 abstention quality"),
        ("consistency", "consistency sampled"),
        ("T4_ref_agreement_32b", "T4 ref agreement to 32B"),
        ("T3_ref_agreement_32b", "T3 ref agreement to 32B"),
    ]
    for key, label in metric_map:
        vals = [value(summaries.get(model, {}), key) for model in models]
        delta = None if vals[0] is None or vals[1] is None else vals[1] - vals[0]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(fmt_metric(summaries.get(model, {}).get(key)) for model in models)
            + f" | {'-' if delta is None else f'{delta:.3f}'} |"
        )
    lines.append("")
    lines.append("## ECE Lite")
    lines.append("")
    lines.append("| model | n | ECE | low acc | medium acc | high acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in models:
        ece = summaries.get(model, {}).get("ECE_lite") or {}
        buckets = ece.get("buckets") or {}
        lines.append(
            f"| {labels[model]} | {ece.get('n', 0)} | {ece.get('ece', None) if ece.get('ece') is not None else '-'} | "
            f"{fmt_metric(buckets.get('low'))} | {fmt_metric(buckets.get('medium'))} | {fmt_metric(buckets.get('high'))} |"
        )
    lines.append("")
    lines.append("## Pairwise Proxy")
    lines.append("")
    lines.append("This is a field-agreement proxy against 32B, not a blind LLM judge.")
    for task_id, result in payload.get("t3_t4_judge_winrate_proxy_vs_32b", {}).items():
        lines.append(f"- {task_id}: n={result.get('n')}, wins={result.get('wins')}")
    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    s2, s8, s32 = summaries.get("2b"), summaries.get("8b"), summaries.get("32b")
    for metric in ["T1", "T2", "T4"]:
        lines.append(
            f"- {metric}: `{recommendation(metric, s2, s8, s32)}` - "
            f"{recommendation_basis(metric, s2, s8, s32)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    common.add_argument("--overwrite", action="store_true")

    p_manifest = sub.add_parser("build-manifest", parents=[common])
    p_manifest.add_argument("--limit-clips", type=int, default=120)
    p_manifest.add_argument("--seed", type=int, default=42)

    p_run = sub.add_parser("run-model", parents=[common])
    p_run.add_argument("--manifest", type=Path, required=True)
    p_run.add_argument("--model-key", choices=["2b", "8b", "32b"], required=True)
    p_run.add_argument("--model-path", type=Path)
    p_run.add_argument("--limit-clips", type=int)
    p_run.add_argument("--tasks", default="T1_pos,T1_neg,T2,T3,T4")
    p_run.add_argument("--sample-n", type=int, default=5)
    p_run.add_argument("--sample-temperature", type=float, default=0.7)
    p_run.add_argument("--max-new-tokens", type=int, default=1024)
    p_run.add_argument("--fps", type=float, default=4.0)
    p_run.add_argument("--seconds", type=float, default=8.0)
    p_run.add_argument("--device-map", default="auto")
    p_run.add_argument("--attn-implementation", default="sdpa")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--log-every", type=int, default=10)
    p_run.add_argument("--batch-size", type=int, default=1)

    p_score = sub.add_parser("score", parents=[common])
    p_score.add_argument("--models", default="2b,8b,32b")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cmd == "build-manifest":
        build_manifest(args)
    elif args.cmd == "run-model":
        run_model(args)
    elif args.cmd == "score":
        score(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
