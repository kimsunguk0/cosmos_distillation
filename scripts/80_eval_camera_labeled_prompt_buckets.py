#!/usr/bin/env python3
"""Compare unlabeled image prompts against Alpamayo-style camera-labeled prompts.

This is a small diagnostic for checkpoints trained with compact image blocks.
It keeps the checkpoint fixed and changes only the inference message format:

1. compact/unlabeled: image, image, ... prompt text
2. camera_labeled: "Front left camera: frame 0 <image> ..."

The output is bucketed by teacher CoT scene phrases so we can quickly see if
camera labels help stop/sign/turn/curve cases.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.decoding import (  # noqa: E402
    StopOnTrajEndCriteria,
    TrajDecodingContract,
    TrajSpanLogitsProcessor,
)
from src.training.collator import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
    load_ego_history_xyz,
    load_sample_images,
    load_traj_future_token_ids,
)


def _load_decode_helpers():
    path = PROJECT_ROOT / "scripts" / "25_decode_checkpoint_overlays.py"
    spec = importlib.util.spec_from_file_location("decode_checkpoint_overlays_25", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import decode helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_decode_helpers()

CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}

SCENE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("traffic_sign", r"traffic sign|stop sign|yield sign|\bsign\b"),
    ("traffic_light", r"traffic light|red light|green light|yellow light"),
    ("stop", r"\bstop\b|stopped|stopping|decelerate|slow down"),
    ("turn_left", r"turn left|left turn|turning left"),
    ("turn_right", r"turn right|right turn|turning right"),
    ("curve", r"\bcurve\b|curving|bend"),
    ("straight", r"\bstraight\b|maintain lane|keep lane|continue"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, default=PROJECT_ROOT / "data/corpus/no_nav_teacher_pair_300chunks.jsonl")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=helpers.resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--buckets", nargs="*", default=[name for name, _ in SCENE_PATTERNS])
    parser.add_argument("--styles", nargs="*", default=["unlabeled", "camera_labeled"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(items: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def teacher_cot(sample: dict[str, Any]) -> str:
    return str(
        (sample.get("teacher_target") or {}).get("cot_text")
        or (sample.get("hard_target") or {}).get("cot_text")
        or ""
    ).strip()


def match_bucket(sample: dict[str, Any]) -> str | None:
    text = teacher_cot(sample).lower()
    for name, pattern in SCENE_PATTERNS:
        if re.search(pattern, text):
            return name
    return None


def select_scene_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    buckets: set[str],
    samples_per_bucket: int,
) -> list[dict[str, Any]]:
    selected_by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in buckets}
    seen_clip_ids: set[str] = set()
    for sample in rows:
        if sample.get("split") != split:
            continue
        bucket = match_bucket(sample)
        if bucket not in buckets:
            continue
        if len(selected_by_bucket[bucket]) >= samples_per_bucket:
            continue
        clip_id = str(sample.get("clip_id") or "")
        if clip_id and clip_id in seen_clip_ids:
            continue
        sample = dict(sample)
        sample["_scene_bucket"] = bucket
        selected_by_bucket[bucket].append(sample)
        if clip_id:
            seen_clip_ids.add(clip_id)
        if all(len(values) >= samples_per_bucket for values in selected_by_bucket.values()):
            break

    # If clip-level de-duplication made rare buckets short, fill the gaps.
    for sample in rows:
        if sample.get("split") != split:
            continue
        bucket = match_bucket(sample)
        if bucket not in buckets:
            continue
        if len(selected_by_bucket[bucket]) >= samples_per_bucket:
            continue
        sample = dict(sample)
        sample["_scene_bucket"] = bucket
        selected_by_bucket[bucket].append(sample)
        if all(len(values) >= samples_per_bucket for values in selected_by_bucket.values()):
            break

    selected: list[dict[str, Any]] = []
    for bucket in [name for name, _ in SCENE_PATTERNS if name in buckets]:
        selected.extend(selected_by_bucket[bucket])
    return selected


def resolve_camera_indices(sample: dict[str, Any], image_count: int) -> list[int]:
    sample_input = sample.get("input") or {}
    raw = sample_input.get("camera_indices")
    if raw:
        return [int(value) for value in raw]
    meta_path = sample_input.get("metadata_path")
    if meta_path:
        path = Path(str(meta_path))
        if path.exists():
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                raw = meta.get("camera_indices")
                if raw:
                    return [int(value) for value in raw]
            except Exception:  # noqa: BLE001
                pass
    if image_count % 4 == 0:
        return [0, 1, 2, 6]
    return list(range(max(image_count, 1)))


def build_camera_labeled_messages(
    prompt_text: str,
    *,
    image_count: int,
    camera_indices: list[int],
    assistant_prefix: str = "<|cot_start|>",
) -> list[dict[str, Any]]:
    if not camera_indices:
        camera_indices = [0, 1, 2, 6] if image_count % 4 == 0 else list(range(max(image_count, 1)))
    frames_per_camera = max(image_count // max(len(camera_indices), 1), 1)
    user_content: list[dict[str, Any]] = []
    emitted = 0
    for camera_index in camera_indices:
        user_content.append(
            {"type": "text", "text": f"{CAMERA_DISPLAY_NAMES.get(camera_index, f'Camera {camera_index}')}: "}
        )
        for frame_index in range(frames_per_camera):
            if emitted >= image_count:
                break
            user_content.append({"type": "text", "text": f"frame {frame_index} "})
            user_content.append({"type": "image"})
            emitted += 1
    while emitted < image_count:
        user_content.append({"type": "text", "text": f"frame {emitted} "})
        user_content.append({"type": "image"})
        emitted += 1
    user_content.append({"type": "text", "text": prompt_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_prefix}]},
    ]


def prepare_messages(
    *,
    style: str,
    prompt_text: str,
    image_count: int,
    camera_indices: list[int],
    assistant_prefix: str = "<|cot_start|>",
) -> list[dict[str, Any]]:
    if style == "unlabeled":
        return build_messages(prompt_text, image_count, assistant_prefix=assistant_prefix)
    if style == "camera_labeled":
        return build_camera_labeled_messages(
            prompt_text,
            image_count=image_count,
            camera_indices=camera_indices,
            assistant_prefix=assistant_prefix,
        )
    raise ValueError(f"Unknown style: {style}")


def move_batch(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def ade_fde(pred: np.ndarray | None, target: np.ndarray | None) -> tuple[float | None, float | None]:
    if pred is None or target is None:
        return None, None
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return None, None
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def max_same_token_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    best = 1
    current = 1
    for left, right in zip(tokens, tokens[1:]):
        if left == right:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return int(best)


def run_style(
    *,
    style: str,
    samples: list[dict[str, Any]],
    model,
    tokenizer,
    processor,
    device: torch.device,
    decoder,
    base_model: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    model_dtype = next(model.backbone.parameters()).dtype
    output_rows: list[dict[str, Any]] = []

    for sample_batch in batched(samples, batch_size):
        texts: list[str] = []
        image_batches: list[list[Any]] = []
        prepared: list[dict[str, Any]] = []
        target_count: int | None = None
        for sample in sample_batch:
            history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
            history_rot = helpers.load_ego_history_rot(sample, PROJECT_ROOT)
            prompt_text = build_user_prompt(sample, PROJECT_ROOT, ego_history_xyz=history_xyz)
            images = load_sample_images(sample, PROJECT_ROOT)
            target_tokens = load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
            target_count = len(target_tokens) if target_count is None else target_count
            if len(target_tokens) != target_count:
                raise ValueError("Mixed target token counts inside one generation batch.")
            camera_indices = resolve_camera_indices(sample, len(images))
            messages = prepare_messages(
                style=style,
                prompt_text=prompt_text,
                image_count=len(images),
                camera_indices=camera_indices,
                assistant_prefix="<|cot_start|>",
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            texts.append(text)
            image_batches.append(images)
            prepared.append(
                {
                    "sample": sample,
                    "history_xyz": history_xyz,
                    "history_rot": history_rot,
                    "target_tokens": target_tokens,
                    "camera_indices": camera_indices,
                    "prompt_char_len": len(text),
                }
            )

        batch = processor(
            text=texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        prompt_len = int(batch["input_ids"].shape[1])
        moved = move_batch(batch, device=device, dtype=model_dtype)
        contract = TrajDecodingContract.from_tokenizer(
            tokenizer,
            prompt_lengths=[prompt_len] * len(prepared),
            traj_token_count=int(target_count or 0),
        )
        with torch.inference_mode():
            generated = model.backbone.generate(
                **moved,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_processor=LogitsProcessorList([TrajSpanLogitsProcessor(contract)]),
                stopping_criteria=StoppingCriteriaList([StopOnTrajEndCriteria(contract)]),
                pad_token_id=tokenizer.pad_token_id,
            )

        for row_index, item in enumerate(prepared):
            sample = item["sample"]
            sample_id = str(sample.get("sample_id"))
            text = helpers._extract_generated_text(tokenizer, batch["input_ids"], generated, row_index=row_index)
            tokens = helpers._extract_generated_traj_tokens(text)
            student_cot = helpers._extract_student_cot(text)
            target_tokens = item["target_tokens"]
            teacher_xyz = decoder.decode(item["history_xyz"], item["history_rot"], target_tokens)
            student_xyz = (
                decoder.decode(item["history_xyz"], item["history_rot"], tokens)
                if len(tokens) == decoder.n_waypoints * 2
                else None
            )
            ade, fde = ade_fde(student_xyz, teacher_xyz)
            exact = sum(1 for left, right in zip(tokens, target_tokens) if int(left) == int(right))
            output_rows.append(
                {
                    "style": style,
                    "scene_bucket": sample.get("_scene_bucket"),
                    "sample_id": sample_id,
                    "clip_id": sample.get("clip_id"),
                    "teacher_cot": teacher_cot(sample),
                    "student_cot": student_cot,
                    "camera_indices": item["camera_indices"],
                    "prompt_char_len": int(item["prompt_char_len"]),
                    "generated_token_count": len(tokens),
                    "target_token_count": len(target_tokens),
                    "token_count_match": bool(len(tokens) == len(target_tokens)),
                    "unique_traj_ids": int(len(set(tokens))),
                    "max_same_token_run": max_same_token_run(tokens),
                    "token_match_rate": float(exact / max(len(target_tokens), 1)),
                    "ade_m_vs_teacher_discrete": ade,
                    "fde_m_vs_teacher_discrete": fde,
                    "generated_tokens_head": tokens[:12],
                    "target_tokens_head": target_tokens[:12],
                }
            )
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "style": style,
                        "bucket": sample.get("_scene_bucket"),
                        "sample_id": sample_id,
                        "ade": ade,
                        "fde": fde,
                        "unique": len(set(tokens)),
                        "token_count": len(tokens),
                    }
                ),
                flush=True,
            )
    return output_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["style"]), str(row["scene_bucket"]))].append(row)

    by_style_bucket: dict[str, dict[str, Any]] = {}
    for (style, bucket), values in grouped.items():
        key = f"{style}/{bucket}"
        by_style_bucket[key] = {
            "n": len(values),
            "ade_m": mean([row["ade_m_vs_teacher_discrete"] for row in values if row["ade_m_vs_teacher_discrete"] is not None]),
            "fde_m": mean([row["fde_m_vs_teacher_discrete"] for row in values if row["fde_m_vs_teacher_discrete"] is not None]),
            "token_count_match_rate": mean([float(row["token_count_match"]) for row in values]),
            "unique_traj_ids": mean([float(row["unique_traj_ids"]) for row in values]),
            "max_same_token_run": mean([float(row["max_same_token_run"]) for row in values]),
            "token_match_rate": mean([float(row["token_match_rate"]) for row in values]),
        }

    by_style: dict[str, Any] = {}
    for style in sorted({str(row["style"]) for row in rows}):
        values = [row for row in rows if row["style"] == style]
        by_style[style] = {
            "n": len(values),
            "ade_m": mean([row["ade_m_vs_teacher_discrete"] for row in values if row["ade_m_vs_teacher_discrete"] is not None]),
            "fde_m": mean([row["fde_m_vs_teacher_discrete"] for row in values if row["fde_m_vs_teacher_discrete"] is not None]),
            "token_count_match_rate": mean([float(row["token_count_match"]) for row in values]),
            "unique_traj_ids": mean([float(row["unique_traj_ids"]) for row in values]),
            "max_same_token_run": mean([float(row["max_same_token_run"]) for row in values]),
            "token_match_rate": mean([float(row["token_match_rate"]) for row in values]),
        }

    paired_delta: dict[str, Any] = {}
    samples = sorted({(str(row["scene_bucket"]), str(row["sample_id"])) for row in rows})
    for bucket, sample_id in samples:
        unlabeled = next((row for row in rows if row["sample_id"] == sample_id and row["style"] == "unlabeled"), None)
        labeled = next((row for row in rows if row["sample_id"] == sample_id and row["style"] == "camera_labeled"), None)
        if unlabeled is None or labeled is None:
            continue
        if unlabeled["ade_m_vs_teacher_discrete"] is None or labeled["ade_m_vs_teacher_discrete"] is None:
            continue
        key = bucket
        paired_delta.setdefault(key, {"n": 0, "ade_delta_labeled_minus_unlabeled": [], "fde_delta_labeled_minus_unlabeled": []})
        paired_delta[key]["n"] += 1
        paired_delta[key]["ade_delta_labeled_minus_unlabeled"].append(
            float(labeled["ade_m_vs_teacher_discrete"] - unlabeled["ade_m_vs_teacher_discrete"])
        )
        paired_delta[key]["fde_delta_labeled_minus_unlabeled"].append(
            float(labeled["fde_m_vs_teacher_discrete"] - unlabeled["fde_m_vs_teacher_discrete"])
        )

    paired_delta_summary = {
        bucket: {
            "n": value["n"],
            "ade_delta_labeled_minus_unlabeled": mean(value["ade_delta_labeled_minus_unlabeled"]),
            "fde_delta_labeled_minus_unlabeled": mean(value["fde_delta_labeled_minus_unlabeled"]),
        }
        for bucket, value in paired_delta.items()
    }

    return {
        "by_style": by_style,
        "by_style_bucket": by_style_bucket,
        "paired_delta_by_bucket": paired_delta_summary,
        "scene_counts": dict(Counter(str(row["scene_bucket"]) for row in rows)),
    }


def main() -> int:
    args = parse_args()
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.samples_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.corpus_jsonl)
    selected = select_scene_rows(
        rows,
        split=args.split,
        buckets=set(args.buckets),
        samples_per_bucket=args.samples_per_bucket,
    )
    if not selected:
        raise SystemExit("No scene samples selected.")

    model, tokenizer, processor, device, base_model = helpers._load_model_and_processors(args)
    decoder_path = helpers.resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not resolve trajectory tokenizer config.")
    decoder = helpers.TrajectoryTokenDecoder(config_path=decoder_path)

    all_rows: list[dict[str, Any]] = []
    for style in args.styles:
        all_rows.extend(
            run_style(
                style=style,
                samples=selected,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=device,
                decoder=decoder,
                base_model=base_model,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
        )

    with args.samples_jsonl.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "samples_per_bucket": int(args.samples_per_bucket),
        "styles": list(args.styles),
        "num_unique_samples": len(selected),
        "num_eval_rows": len(all_rows),
        "traj_tokenizer_config": str(decoder_path),
        **summarize(all_rows),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
