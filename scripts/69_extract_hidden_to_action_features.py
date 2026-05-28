#!/usr/bin/env python3
"""Extract frozen student hidden features for hidden-to-action probes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_readiness_module():
    path = PROJECT_ROOT / "scripts" / "67_eval_backbone_readiness.py"
    spec = importlib.util.spec_from_file_location("backbone_readiness_67", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_readiness_module()

from src.training.collator import fuse_history_tokens_in_input_ids, load_ego_future_xyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split-sample-ids-json", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=os.environ.get("COSMOS_STUDENT_MODEL", str(PROJECT_ROOT / "base_weights/cosmos-reason-2b")))
    parser.add_argument("--prefix-type", choices=("teacher_prefix", "student_free"), default="teacher_prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["probe_train", "probe_val", "probe_test"])
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--empty-cot-token-threshold", type=int, default=3)
    parser.add_argument("--image-mode", choices=("normal", "black", "shuffled"), default="normal")
    parser.add_argument(
        "--image-prompt-style",
        choices=("compact", "camera_labeled"),
        default="compact",
        help="Use compact image blocks or Alpamayo camera/frame labels for prefix construction.",
    )
    parser.add_argument(
        "--prompt-text-style",
        choices=("numeric_history_question", "official_alpamayo"),
        default="numeric_history_question",
        help="Use numeric-history question text or the official Alpamayo history-token prompt.",
    )
    parser.add_argument(
        "--fuse-history-tokens",
        action="store_true",
        help="Replace official <|traj_history|> placeholders with encoded ego-history tokens.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id"))
            rows[sample_id] = row
    return rows


def load_split_sample_ids(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_ids = payload.get("sample_ids") or {}
    return {str(name): [str(value) for value in values] for name, values in sample_ids.items()}


def raw_teacher_pred(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    raw_path = helpers.resolve_path((sample.get("teacher_cache") or {}).get("text_raw_json_path"))
    if raw_path is None:
        return None
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        result = (payload.get("results") or [None])[0]
        if not isinstance(result, dict):
            return None
        xyz = np.asarray(result.get("pred_xyz"), dtype=np.float32).reshape(-1, 64, 3)[0]
        rot = np.asarray(result.get("pred_rot"), dtype=np.float32).reshape(-1, 64, 3, 3)[0]
    except Exception:
        return None
    return xyz, rot


def load_gt_future(sample: dict[str, Any]) -> np.ndarray | None:
    try:
        future = load_ego_future_xyz(sample, PROJECT_ROOT).astype(np.float32)
    except Exception:
        return None
    if future.ndim != 2 or future.shape[0] < 1:
        return None
    if future.shape[0] < 64:
        padded = np.zeros((64, max(future.shape[1], 3)), dtype=np.float32)
        padded[: future.shape[0], : future.shape[1]] = future
        future = padded
    return future[:64, :3] if future.shape[1] >= 3 else np.pad(future[:64, :2], ((0, 0), (0, 1)))


def normalize_history_rot(rot: np.ndarray) -> np.ndarray:
    """Normalize loader variants to [T, 3, 3]."""
    rot = np.asarray(rot, dtype=np.float32)
    while rot.ndim > 3 and rot.shape[0] == 1:
        rot = rot[0]
    while rot.ndim > 3 and rot.shape[0] == 1:
        rot = rot[0]
    if rot.ndim != 3 or rot.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ego history rot shape [T,3,3], got {rot.shape}")
    return rot


def path_length_xy(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=-1).sum())


def heading_delta(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    diffs = np.diff(xyz[:, :2], axis=0)
    good = np.linalg.norm(diffs, axis=-1) > 1e-3
    if not np.any(good):
        return 0.0
    headings = np.unwrap(np.arctan2(diffs[good, 1], diffs[good, 0]))
    return float(headings[-1] - headings[0])


def bucket_for_traj(xyz: np.ndarray) -> str:
    length = path_length_xy(xyz)
    if length <= 5.0:
        return "stop"
    if abs(float(xyz[-1, 1])) >= 2.0 or abs(heading_delta(xyz)) >= 0.15:
        return "curve"
    return "straight"


def batched(items: list[Any], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(items), width):
        yield items[index : index + width]


def token_positions(input_ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(input_ids) if int(value) == int(token_id)]


def apply_image_mode(image_batches: list[list[Any]], image_mode: str) -> list[list[Any]]:
    if image_mode == "normal":
        return image_batches
    if image_mode == "black":
        out: list[list[Any]] = []
        for images in image_batches:
            out.append([Image.new("RGB", image.size, (0, 0, 0)) for image in images])
        return out
    if image_mode == "shuffled":
        if len(image_batches) <= 1:
            return image_batches
        return image_batches[1:] + image_batches[:1]
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def backbone_last_hidden(model, moved: dict[str, Any]) -> torch.Tensor:
    """Run the frozen backbone and return the raw final-layer hidden states."""
    kwargs = {
        "input_ids": moved["input_ids"],
        "attention_mask": moved["attention_mask"],
        "pixel_values": moved.get("pixel_values"),
        "image_grid_thw": moved.get("image_grid_thw"),
        "output_hidden_states": True,
        "return_dict": True,
    }
    try:
        outputs = model.backbone(**kwargs, logits_to_keep=1)
    except TypeError:
        outputs = model.backbone(**kwargs)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and hasattr(outputs, "language_model_outputs"):
        hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("Student backbone did not return hidden states.")
    return hidden_states[-1]


def make_teacher_prefix_batch(
    samples: list[dict[str, Any]],
    *,
    processor,
    tokenizer,
    image_mode: str = "normal",
    image_prompt_style: str = "compact",
    prompt_text_style: str = "numeric_history_question",
    fuse_history_tokens: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    texts: list[str] = []
    image_batches: list[list[Any]] = []
    prepared: list[dict[str, Any]] = []
    for sample in samples:
        history_xyz = helpers.load_ego_history_xyz(sample, PROJECT_ROOT)
        prompt_text = helpers.build_user_prompt(
            sample,
            PROJECT_ROOT,
            ego_history_xyz=history_xyz,
            prompt_text_style=prompt_text_style,
        )
        images = helpers.load_sample_images(sample, PROJECT_ROOT)
        camera_indices = helpers.resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
        teacher_cot = str((sample.get("teacher_target") or {}).get("cot_text") or (sample.get("hard_target") or {}).get("cot_text") or "")
        completion = f"{teacher_cot}<|cot_end|><|traj_future_start|>"
        messages = helpers.build_messages(
            prompt_text,
            len(images),
            completion_text=completion,
            assistant_prefix="<|cot_start|>",
            image_prompt_style=image_prompt_style,
            camera_indices=camera_indices,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        texts.append(text)
        image_batches.append(images)
        prepared.append(sample)
    batch = processor(
        text=texts,
        images=apply_image_mode(image_batches, image_mode),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    if fuse_history_tokens:
        histories = [helpers.load_ego_history_xyz(sample, PROJECT_ROOT) for sample in prepared]
        batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, histories)
    return batch, prepared


def make_student_prompt_batch(
    samples: list[dict[str, Any]],
    *,
    processor,
    tokenizer,
    image_mode: str = "normal",
    image_prompt_style: str = "compact",
    prompt_text_style: str = "numeric_history_question",
    fuse_history_tokens: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    texts: list[str] = []
    image_batches: list[list[Any]] = []
    prepared: list[dict[str, Any]] = []
    for sample in samples:
        history_xyz = helpers.load_ego_history_xyz(sample, PROJECT_ROOT)
        prompt_text = helpers.build_user_prompt(
            sample,
            PROJECT_ROOT,
            ego_history_xyz=history_xyz,
            prompt_text_style=prompt_text_style,
        )
        images = helpers.load_sample_images(sample, PROJECT_ROOT)
        camera_indices = helpers.resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
        messages = helpers.build_messages(
            prompt_text,
            len(images),
            assistant_prefix="<|cot_start|>",
            image_prompt_style=image_prompt_style,
            camera_indices=camera_indices,
        )
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        texts.append(text)
        image_batches.append(images)
        prepared.append(sample)
    old_tokenizer_padding = getattr(tokenizer, "padding_side", None)
    processor_tokenizer = getattr(processor, "tokenizer", None)
    old_processor_padding = getattr(processor_tokenizer, "padding_side", None)
    try:
        tokenizer.padding_side = "left"
        if processor_tokenizer is not None:
            processor_tokenizer.padding_side = "left"
        batch = processor(
            text=texts,
            images=apply_image_mode(image_batches, image_mode),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        if fuse_history_tokens:
            histories = [helpers.load_ego_history_xyz(sample, PROJECT_ROOT) for sample in prepared]
            batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, histories)
    finally:
        if old_tokenizer_padding is not None:
            tokenizer.padding_side = old_tokenizer_padding
        if processor_tokenizer is not None and old_processor_padding is not None:
            processor_tokenizer.padding_side = old_processor_padding
    return batch, prepared


def move_processor_batch(batch: dict[str, Any], *, device: torch.device, model_dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, dtype=model_dtype) if torch.is_floating_point(value) else value.to(device=device)
        else:
            moved[key] = value
    return moved


def flush_shard(
    *,
    split_dir: Path,
    shard_index: int,
    rows: list[dict[str, Any]],
    tensors: dict[str, list[np.ndarray]],
    metadata: dict[str, Any],
) -> Path | None:
    if not rows:
        return None
    split_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": metadata,
        "rows": rows,
    }
    for name, values in tensors.items():
        if values:
            payload[name] = torch.from_numpy(np.stack(values, axis=0))
    path = split_dir / f"features_{shard_index:05d}.pt"
    torch.save(payload, path)
    return path


def load_partial_split_state(split_dir: Path) -> tuple[set[str], list[str], int, Counter, Counter]:
    """Recover already-flushed feature shards so interrupted runs can resume."""
    processed_ids: set[str] = set()
    shard_paths: list[str] = []
    counters = Counter()
    bucket_counts = Counter()
    next_shard_index = 0
    if not split_dir.exists():
        return processed_ids, shard_paths, next_shard_index, counters, bucket_counts

    for shard_path in sorted(split_dir.glob("features_*.pt")):
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        rows = list(payload.get("rows") or [])
        if not rows:
            continue
        shard_paths.append(str(shard_path))
        try:
            next_shard_index = max(next_shard_index, int(shard_path.stem.rsplit("_", 1)[-1]) + 1)
        except ValueError:
            next_shard_index += 1
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if not sample_id or sample_id in processed_ids:
                continue
            processed_ids.add(sample_id)
            counters["ready"] += 1
            if row.get("prefix_type") == "student_free" or "student_generated_new_token_count" in row:
                counters["student_free_generation_attempted"] += 1
                counters["student_free_cot_end_hit"] += int(int(row.get("student_cot_end_count") or 0) > 0)
                counters["student_free_traj_start_hit"] += int(int(row.get("student_traj_start_count") or 0) > 0)
                counters["student_free_multi_start"] += int(int(row.get("student_traj_start_count") or 0) > 1)
                counters["student_free_empty_cot"] += int(not bool(row.get("student_cot_nonempty", True)))
                counters["student_free_valid_boundary"] += 1
            bucket = row.get("bucket")
            if bucket:
                bucket_counts[str(bucket)] += 1
        if "gt_future" in payload:
            counters["gt_available"] += int(payload["gt_future"].shape[0])
    return processed_ids, shard_paths, next_shard_index, counters, bucket_counts


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = load_jsonl_map(args.corpus_jsonl)
    split_ids = load_split_sample_ids(args.split_sample_ids_json)

    model_args = argparse.Namespace(
        checkpoint_dir=args.checkpoint_dir,
        student_model=args.student_model,
        device=args.device,
    )
    model, tokenizer, processor, device, base_model, _train_config = helpers.load_model(model_args)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model_dtype = next(model.backbone.parameters()).dtype
    cot_end_id = helpers.token_id(tokenizer, "<|cot_end|>")
    traj_start_id = helpers.token_id(tokenizer, "<|traj_future_start|>")
    decoder_config = helpers.resolve_traj_tokenizer_config_path(base_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {base_model}")
    decoder = helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    summary: dict[str, Any] = {
        "schema_version": "hidden_to_action_feature_extraction_v1",
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "prefix_type": args.prefix_type,
        "output_dir": str(args.output_dir),
        "batch_size": int(args.batch_size),
        "shard_size": int(args.shard_size),
        "max_new_tokens": int(args.max_new_tokens),
        "empty_cot_token_threshold": int(args.empty_cot_token_threshold),
        "image_mode": str(args.image_mode),
        "image_prompt_style": str(args.image_prompt_style),
        "prompt_text_style": str(args.prompt_text_style),
        "fuse_history_tokens": bool(args.fuse_history_tokens),
        "splits": {},
        "feature_schema": {
            "hidden_feature": "concat(h_cot_end, h_traj_start, h_prefix_mean_last16), float16, [6144]",
            "ego_feature": "flattened ego_history_xyz, float32",
            "target_action": "Alpamayo action_space.traj_to_action(teacher_action_expert_traj), float32, [64,2]",
            "target_traj": "teacher action expert pred_xyz, float32, [64,3]",
            "gt_future": "materialized GT ego future when available, float32, [64,3]",
        },
    }

    for split_name in args.splits:
        ids = [sample_id for sample_id in split_ids.get(split_name, []) if sample_id in rows_by_id]
        if args.max_samples_per_split > 0:
            ids = ids[: int(args.max_samples_per_split)]
        split_dir = args.output_dir / args.checkpoint_name / args.prefix_type / split_name
        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary["splits"][split_name] = existing
            print(json.dumps({"event": "split_skip_existing", "split": split_name, "manifest": str(manifest_path)}), flush=True)
            continue

        resume_ids: set[str] = set()
        shard_paths: list[str] = []
        counters = Counter()
        bucket_counts = Counter()
        shard_index = 0
        if not args.overwrite:
            resume_ids, shard_paths, shard_index, counters, bucket_counts = load_partial_split_state(split_dir)
            if resume_ids:
                ids = [sample_id for sample_id in ids if sample_id not in resume_ids]
                print(
                    json.dumps(
                        {
                            "event": "split_resume_existing_shards",
                            "checkpoint": args.checkpoint_name,
                            "split": split_name,
                            "resumed_ready": len(resume_ids),
                            "remaining": len(ids),
                            "next_shard_index": shard_index,
                        }
                    ),
                    flush=True,
                )

        rows_buffer: list[dict[str, Any]] = []
        tensors: dict[str, list[np.ndarray]] = {
            "hidden_feature": [],
            "h_cot_end": [],
            "h_traj_start": [],
            "h_prefix_mean_last8": [],
            "h_prefix_mean_last16": [],
            "ego_feature": [],
            "ego_history_xyz": [],
            "ego_history_rot": [],
            "target_action": [],
            "target_traj": [],
            "gt_future": [],
        }

        for batch_ids in batched(ids, args.batch_size):
            raw_samples = [rows_by_id[sample_id] for sample_id in batch_ids]
            valid_samples: list[dict[str, Any]] = []
            target_xyz: list[np.ndarray] = []
            target_rot: list[np.ndarray] = []
            history_xyz: list[np.ndarray] = []
            history_rot: list[np.ndarray] = []
            gt_future: list[np.ndarray] = []
            for sample in raw_samples:
                pred = raw_teacher_pred(sample)
                if pred is None:
                    counters["missing_teacher_action"] += 1
                    continue
                try:
                    hxyz = helpers.load_ego_history_xyz(sample, PROJECT_ROOT).astype(np.float32)
                    hrot = normalize_history_rot(helpers.load_ego_history_rot(sample, PROJECT_ROOT))
                except Exception:
                    counters["missing_ego_history"] += 1
                    continue
                gt = load_gt_future(sample)
                if gt is None:
                    gt = np.zeros((64, 3), dtype=np.float32)
                    has_gt = False
                else:
                    has_gt = True
                valid_samples.append(sample)
                target_xyz.append(pred[0])
                target_rot.append(pred[1])
                history_xyz.append(hxyz)
                history_rot.append(hrot)
                gt_future.append(gt)
                counters["gt_available"] += int(has_gt)
            if not valid_samples:
                continue

            prompt_len: int | None = None
            if args.prefix_type == "teacher_prefix":
                batch, prepared = make_teacher_prefix_batch(
                    valid_samples,
                    processor=processor,
                    tokenizer=tokenizer,
                    image_mode=str(args.image_mode),
                    image_prompt_style=str(args.image_prompt_style),
                    prompt_text_style=str(args.prompt_text_style),
                    fuse_history_tokens=bool(args.fuse_history_tokens),
                )
                moved = move_processor_batch(batch, device=device, model_dtype=model_dtype)
            elif args.prefix_type == "student_free":
                batch, prepared = make_student_prompt_batch(
                    valid_samples,
                    processor=processor,
                    tokenizer=tokenizer,
                    image_mode=str(args.image_mode),
                    image_prompt_style=str(args.image_prompt_style),
                    prompt_text_style=str(args.prompt_text_style),
                    fuse_history_tokens=bool(args.fuse_history_tokens),
                )
                prompt_moved = move_processor_batch(batch, device=device, model_dtype=model_dtype)
                prompt_len = int(prompt_moved["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = model.backbone.generate(
                        **prompt_moved,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        use_cache=True,
                        stopping_criteria=helpers.StoppingCriteriaList(
                            [
                                helpers.StopAfterTokenCriteria(
                                    prompt_lengths=[prompt_len] * len(prepared),
                                    stop_token_id=traj_start_id,
                                )
                            ]
                        ),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated_len = int(generated.shape[1] - prompt_len)
                if generated_len < 0:
                    raise RuntimeError(f"Generated sequence shorter than prompt: {generated.shape[1]} < {prompt_len}")
                generated_attention = torch.cat(
                    [
                        prompt_moved["attention_mask"],
                        torch.ones(
                            (generated.shape[0], generated_len),
                            dtype=prompt_moved["attention_mask"].dtype,
                            device=prompt_moved["attention_mask"].device,
                        ),
                    ],
                    dim=1,
                )
                moved = {
                    key: value
                    for key, value in prompt_moved.items()
                    if key not in {"input_ids", "attention_mask"}
                }
                moved["input_ids"] = generated
                moved["attention_mask"] = generated_attention
            else:
                raise ValueError(f"Unsupported prefix type: {args.prefix_type}")
            with torch.inference_mode():
                last_hidden = backbone_last_hidden(model, moved)
            hidden = last_hidden.detach().to(dtype=torch.float32).cpu().numpy()
            input_ids_np = moved["input_ids"].detach().cpu().numpy()

            with torch.inference_mode():
                target_action_t = decoder.action_space.traj_to_action(
                    torch.from_numpy(np.stack(history_xyz, axis=0)),
                    torch.from_numpy(np.stack(history_rot, axis=0)),
                    torch.from_numpy(np.stack(target_xyz, axis=0)),
                    torch.from_numpy(np.stack(target_rot, axis=0)),
                )
            target_action_np = target_action_t.detach().cpu().numpy().astype(np.float32)

            for row_index, sample in enumerate(prepared):
                ids_row = [int(value) for value in input_ids_np[row_index].tolist()]
                row_extra: dict[str, Any] = {"prefix_type": args.prefix_type}
                if prompt_len is None:
                    cot_positions = token_positions(ids_row, cot_end_id)
                    traj_positions = token_positions(ids_row, traj_start_id)
                else:
                    counters["student_free_generation_attempted"] += 1
                    generated_ids = ids_row[prompt_len:]
                    cot_generated = token_positions(generated_ids, cot_end_id)
                    traj_generated = token_positions(generated_ids, traj_start_id)
                    cot_positions = []
                    traj_positions = []
                    traj_start_count = len(traj_generated)
                    cot_end_count = len(cot_generated)
                    counters["student_free_cot_end_hit"] += int(cot_end_count > 0)
                    counters["student_free_traj_start_hit"] += int(traj_start_count > 0)
                    counters["student_free_multi_start"] += int(traj_start_count > 1)
                    counters["student_free_max_new_tokens_without_traj_start"] += int(traj_start_count == 0)
                    if traj_generated:
                        first_traj = int(traj_generated[0])
                        prior_cot = [int(pos) for pos in cot_generated if int(pos) < first_traj]
                        if prior_cot:
                            cot_positions = [prompt_len + prior_cot[-1]]
                        traj_positions = [prompt_len + first_traj]
                        student_cot_len = int(prior_cot[-1]) if prior_cot else int(first_traj)
                    else:
                        student_cot_len = int(cot_generated[-1]) if cot_generated else len(generated_ids)
                    teacher_cot = str((sample.get("teacher_target") or {}).get("cot_text") or (sample.get("hard_target") or {}).get("cot_text") or "")
                    teacher_cot_len = len(tokenizer.encode(teacher_cot, add_special_tokens=False)) if teacher_cot else 0
                    cot_nonempty = student_cot_len >= int(args.empty_cot_token_threshold)
                    counters["student_free_empty_cot"] += int(not cot_nonempty)
                    row_extra.update(
                        {
                            "student_generated_new_token_count": int(len(generated_ids)),
                            "student_cot_token_count": int(student_cot_len),
                            "teacher_cot_token_count": int(teacher_cot_len),
                            "student_teacher_cot_length_ratio": float(student_cot_len / max(teacher_cot_len, 1)),
                            "student_cot_end_count": int(cot_end_count),
                            "student_traj_start_count": int(traj_start_count),
                            "student_cot_nonempty": bool(cot_nonempty),
                        }
                    )
                    if not cot_nonempty:
                        cot_positions = []
                if not cot_positions or not traj_positions:
                    counters["missing_boundary_token"] += 1
                    if args.prefix_type == "student_free":
                        counters["student_free_malformed"] += 1
                    continue
                cot_pos = int(cot_positions[-1])
                traj_pos = int(traj_positions[-1])
                if cot_pos >= traj_pos:
                    counters["bad_boundary_order"] += 1
                    if args.prefix_type == "student_free":
                        counters["student_free_malformed"] += 1
                    continue
                if args.prefix_type == "student_free":
                    counters["student_free_valid_boundary"] += 1
                h_cot = hidden[row_index, cot_pos].astype(np.float32)
                h_traj = hidden[row_index, traj_pos].astype(np.float32)
                start8 = max(0, traj_pos - 7)
                start16 = max(0, traj_pos - 15)
                h_mean8 = hidden[row_index, start8 : traj_pos + 1].mean(axis=0).astype(np.float32)
                h_mean16 = hidden[row_index, start16 : traj_pos + 1].mean(axis=0).astype(np.float32)
                feature = np.concatenate([h_cot, h_traj, h_mean16], axis=0).astype(np.float16)
                ego_feat = history_xyz[row_index].reshape(-1).astype(np.float32)
                bucket = bucket_for_traj(target_xyz[row_index])
                bucket_counts[bucket] += 1

                tensors["hidden_feature"].append(feature)
                tensors["h_cot_end"].append(h_cot.astype(np.float16))
                tensors["h_traj_start"].append(h_traj.astype(np.float16))
                tensors["h_prefix_mean_last8"].append(h_mean8.astype(np.float16))
                tensors["h_prefix_mean_last16"].append(h_mean16.astype(np.float16))
                tensors["ego_feature"].append(ego_feat)
                tensors["ego_history_xyz"].append(history_xyz[row_index].astype(np.float32))
                tensors["ego_history_rot"].append(history_rot[row_index].astype(np.float32))
                tensors["target_action"].append(target_action_np[row_index].astype(np.float32))
                tensors["target_traj"].append(target_xyz[row_index].astype(np.float32))
                tensors["gt_future"].append(gt_future[row_index].astype(np.float32))
                rows_buffer.append(
                    {
                        "sample_id": str(sample.get("sample_id")),
                        "clip_id": str(sample.get("clip_id") or str(sample.get("sample_id", "")).split("__", 1)[0]),
                        "chunk_id": str(sample.get("chunk_id", "")),
                        "bucket": bucket,
                        "target_path_length_m": path_length_xy(target_xyz[row_index]),
                        "target_final_y_m": float(target_xyz[row_index][-1, 1]),
                        "target_heading_delta_rad": heading_delta(target_xyz[row_index]),
                        "cot_end_pos": cot_pos,
                        "traj_start_pos": traj_pos,
                        "prefix_seq_len": int(moved["attention_mask"][row_index].sum().item()),
                        **row_extra,
                    }
                )
                counters["ready"] += 1

                if len(rows_buffer) >= int(args.shard_size):
                    path = flush_shard(
                        split_dir=split_dir,
                        shard_index=shard_index,
                        rows=rows_buffer,
                        tensors=tensors,
                        metadata={
                            "checkpoint_name": args.checkpoint_name,
                            "prefix_type": args.prefix_type,
                            "split_name": split_name,
                            "feature_dim": 6144,
                            "target_type": "teacher_action_space_and_teacher_action_expert_traj",
                            "image_prompt_style": args.image_prompt_style,
                            "prompt_text_style": args.prompt_text_style,
                            "fuse_history_tokens": bool(args.fuse_history_tokens),
                        },
                    )
                    if path is not None:
                        shard_paths.append(str(path))
                    shard_index += 1
                    rows_buffer = []
                    tensors = {name: [] for name in tensors}

            print(json.dumps({"event": "feature_batch_done", "checkpoint": args.checkpoint_name, "split": split_name, "ready": counters["ready"], "requested": len(ids)}), flush=True)

        path = flush_shard(
            split_dir=split_dir,
            shard_index=shard_index,
            rows=rows_buffer,
            tensors=tensors,
            metadata={
                "checkpoint_name": args.checkpoint_name,
                "prefix_type": args.prefix_type,
                "split_name": split_name,
                "feature_dim": 6144,
                "target_type": "teacher_action_space_and_teacher_action_expert_traj",
                "image_prompt_style": args.image_prompt_style,
                "prompt_text_style": args.prompt_text_style,
                "fuse_history_tokens": bool(args.fuse_history_tokens),
            },
        )
        if path is not None:
            shard_paths.append(str(path))

        split_manifest = {
            "schema_version": "hidden_to_action_feature_split_manifest_v1",
            "checkpoint_name": args.checkpoint_name,
            "prefix_type": args.prefix_type,
            "split_name": split_name,
            "requested": len(ids),
            "ready": int(counters["ready"]),
            "counters": dict(counters),
            "bucket_counts": dict(bucket_counts),
            "feature_dim": 6144,
            "image_prompt_style": args.image_prompt_style,
            "prompt_text_style": args.prompt_text_style,
            "fuse_history_tokens": bool(args.fuse_history_tokens),
            "shard_count": len(shard_paths),
            "shards": shard_paths,
        }
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["splits"][split_name] = split_manifest

    summary["elapsed_sec"] = round(time.time() - t0, 3)
    summary_path = args.output_dir / args.checkpoint_name / args.prefix_type / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "feature_extraction_done", "summary": str(summary_path), "elapsed_sec": summary["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
