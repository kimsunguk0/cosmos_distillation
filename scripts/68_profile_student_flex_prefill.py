#!/usr/bin/env python3
"""Profile student prefill/decode latency with optional FLEX compression."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.training.collator import (  # noqa: E402
    build_messages,
    build_user_prompt,
    load_sample_images,
    resolve_camera_indices,
    resolve_image_relative_timestamps,
)
from src.training.flex_batch import compress_batch_for_flex  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick_sample(rows: list[dict[str, Any]], split: str, sample_index: int) -> dict[str, Any]:
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise RuntimeError(f"No samples for split={split!r}")
    if sample_index < 0 or sample_index >= len(selected):
        raise RuntimeError(f"sample-index out of range: {sample_index} (split size={len(selected)})")
    return selected[sample_index]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 6)


def move_batch(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def load_model(args: argparse.Namespace):
    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    tokenizer_dir = args.checkpoint_dir / "tokenizer"
    processor_dir = args.checkpoint_dir / "processor"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.exists() else base_model,
        local_files_only=True,
    )
    ensure_special_tokens(tokenizer)
    processor = AutoProcessor.from_pretrained(
        processor_dir if processor_dir.exists() else base_model,
        local_files_only=True,
    )
    processor.tokenizer = tokenizer

    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
        local_files_only=Path(base_model).expanduser().exists(),
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    if detect_checkpoint_format(args.checkpoint_dir) == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=use_lora)
    model = model.to(device).eval()
    return model, tokenizer, processor, device, train_config


def build_prompt_batch(
    sample: dict[str, Any],
    *,
    processor,
    tokenizer,
    train_config: dict[str, Any],
    model,
) -> dict[str, Any]:
    data_view = train_config.get("data_view") or {}
    prompt_text = build_user_prompt(sample, PROJECT_ROOT)
    images = load_sample_images(sample, PROJECT_ROOT)
    camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
    frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
    messages = build_messages(
        prompt_text,
        len(images),
        target_text=None,
        image_prompt_style=str(data_view.get("image_prompt_style") or "camera_labeled"),
        camera_indices=camera_indices,
        num_frames_per_camera=frames_per_camera,
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    tokenized = processor(
        text=[text],
        images=[images],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    batch = dict(tokenized)
    if hasattr(model, "flex_enabled") and model.flex_enabled():
        relative_times = resolve_image_relative_timestamps(
            sample,
            PROJECT_ROOT,
            camera_count=len(camera_indices),
            frames_per_camera=frames_per_camera,
        )
        batch["camera_indices"] = torch.tensor([camera_indices], dtype=torch.long)
        batch["relative_timestamps"] = torch.tensor([relative_times], dtype=torch.float32)
        batch["camera_counts"] = torch.tensor([len(camera_indices)], dtype=torch.long)
        batch["frames_per_camera"] = torch.tensor([frames_per_camera], dtype=torch.long)
        flex_cfg = getattr(model, "flex_scene_config")
        batch = compress_batch_for_flex(
            batch,
            image_token_id=int(getattr(model, "image_token_id")),
            tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
            pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
        )
    return batch


def _extract_logits_and_past(output: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(output, dict):
        logits = output["logits"]
        backbone_outputs = output.get("backbone_outputs")
        return logits, getattr(backbone_outputs, "past_key_values", None)
    return output.logits, getattr(output, "past_key_values", None)


def _model_call(model, batch: dict[str, Any], *, flex_enabled: bool):
    keys = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "camera_indices",
        "relative_timestamps",
        "camera_counts",
        "frames_per_camera",
        "past_key_values",
        "cache_position",
    )
    kwargs = {key: batch[key] for key in keys if key in batch}
    kwargs["use_cache"] = True
    if flex_enabled:
        kwargs.update(
            {
                "return_hidden_states": False,
                "compute_meta_action": False,
                "compute_traj_aux": False,
            }
        )
        return model(**kwargs)
    kwargs["return_dict"] = True
    return model.backbone(**kwargs)


def _past_seq_len(past_key_values: Any) -> int | None:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:  # noqa: BLE001
        return None


def profile_once(model, batch: dict[str, Any], *, max_new_tokens: int, flex_enabled: bool) -> dict[str, float | int | str | None]:
    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode():
        output = _model_call(model, batch, flex_enabled=flex_enabled)
    sync_cuda()
    prefill_sec = elapsed(started)
    logits, past_key_values = _extract_logits_and_past(output)

    next_token = logits[:, -1:, :].argmax(dim=-1)
    attention_mask = batch["attention_mask"]
    generated = 0
    decode_error = None

    sync_cuda()
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max(max_new_tokens, 0)):
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)], dim=1)
            decode_batch = {
                "input_ids": next_token,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            past_seq_len = _past_seq_len(past_key_values)
            if past_seq_len is not None:
                decode_batch["cache_position"] = torch.arange(
                    past_seq_len,
                    past_seq_len + int(next_token.shape[1]),
                    device=next_token.device,
                    dtype=torch.long,
                )
            try:
                output = _model_call(model, decode_batch, flex_enabled=flex_enabled)
            except RuntimeError as exc:
                decode_error = str(exc)
                break
            logits, past_key_values = _extract_logits_and_past(output)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated += 1
    sync_cuda()
    decode_sec = elapsed(decode_started)
    result: dict[str, float | int | str | None] = {
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
        "decode_ms_per_token": round(decode_sec / max(generated, 1) * 1000.0, 3),
        "generated_tokens": generated,
    }
    if decode_error is not None:
        result["decode_error"] = decode_error
    return result


def summarize(values: list[dict[str, float | int]]) -> dict[str, Any]:
    keys = ("prefill_sec", "decode_sec", "decode_ms_per_token")
    out: dict[str, Any] = {}
    for key in keys:
        arr = [float(item[key]) for item in values]
        out[f"{key}_mean"] = float(np.mean(arr)) if arr else None
        out[f"{key}_min"] = float(np.min(arr)) if arr else None
    out["generated_tokens"] = int(values[0]["generated_tokens"]) if values else 0
    return out


def main() -> None:
    args = parse_args()
    sample = pick_sample(load_jsonl(args.corpus_jsonl), args.split, args.sample_index)

    print(json.dumps({"event": "load_start", "checkpoint_dir": str(args.checkpoint_dir)}), flush=True)
    started = time.perf_counter()
    model, tokenizer, processor, device, train_config = load_model(args)
    sync_cuda()
    load_sec = elapsed(started)
    flex_enabled = bool(hasattr(model, "flex_enabled") and model.flex_enabled())
    print(json.dumps({"event": "load_done", "load_sec": load_sec, "flex_enabled": flex_enabled}), flush=True)

    batch = build_prompt_batch(
        sample,
        processor=processor,
        tokenizer=tokenizer,
        train_config=train_config,
        model=model,
    )
    flex_stats = dict(batch.get("flex_stats") or {})
    model_dtype = next(model.backbone.parameters()).dtype
    batch = move_batch(batch, device=device, dtype=model_dtype)
    prompt_tokens = int(batch["attention_mask"].sum().detach().cpu().item())
    image_tokens = int((batch["input_ids"] == int(getattr(model, "image_token_id"))).sum().detach().cpu().item())
    print(
        json.dumps(
            {
                "event": "batch_ready",
                "prompt_tokens": prompt_tokens,
                "image_tokens": image_tokens,
                **flex_stats,
            }
        ),
        flush=True,
    )

    runs: list[dict[str, float | int]] = []
    for repeat_idx in range(max(args.warmup_runs, 0) + max(args.repeats, 1)):
        is_warmup = repeat_idx < max(args.warmup_runs, 0)
        run = profile_once(
            model,
            batch,
            max_new_tokens=int(args.max_new_tokens),
            flex_enabled=flex_enabled,
        )
        print(json.dumps({"event": "profile_done", "warmup": is_warmup, **run}), flush=True)
        if not is_warmup:
            runs.append(run)

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "sample_id": str(sample.get("sample_id")),
        "split": args.split,
        "sample_index": int(args.sample_index),
        "flex_enabled": flex_enabled,
        "prompt_tokens": prompt_tokens,
        "image_tokens": image_tokens,
        "flex_stats": flex_stats,
        "load_sec": load_sec,
        "warmup_runs": int(args.warmup_runs),
        "repeats": int(args.repeats),
        "max_new_tokens": int(args.max_new_tokens),
        **summarize(runs),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
