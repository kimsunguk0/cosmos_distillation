#!/usr/bin/env python3
"""Profile ViT, prefill, and generation latency for teacher/student models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import StudentWrapperConfig, build_student_model
from src.model.tokenizer_ext import distill_trainable_token_ids
from src.training.collator import build_messages, build_user_prompt, load_sample_images
from src.utils.runtime_paths import remap_external_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2.jsonl",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--teacher-model-path",
        type=Path,
        default=Path("/workspace/base_models_weights/Alpamayo-1.5-10B"),
    )
    parser.add_argument(
        "--alpamayo-src",
        type=Path,
        default=Path("/workspace/alpamayo_repos/alpamayo1.5/src"),
    )
    parser.add_argument(
        "--student-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "stage_b_v3_2" / "final",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "prefill_generation_profile_v3_2.json",
    )
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


def to_device(data: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_prefill_and_decode(
    hf_model: torch.nn.Module,
    tokenized_data: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> tuple[float, float, int]:
    input_ids = tokenized_data["input_ids"]
    attention_mask = tokenized_data["attention_mask"]
    forward_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": tokenized_data.get("pixel_values"),
        "image_grid_thw": tokenized_data.get("image_grid_thw"),
        "use_cache": True,
        "return_dict": True,
    }
    forward_inputs = {k: v for k, v in forward_inputs.items() if v is not None}

    sync_cuda()
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        out = hf_model(**forward_inputs)
    sync_cuda()
    prefill_sec = elapsed(prefill_started)

    next_token = out.logits[:, -1:, :].argmax(dim=-1)
    past_key_values = out.past_key_values
    generated = 0

    sync_cuda()
    decode_started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype),
                ],
                dim=1,
            )
            step_out = hf_model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = step_out.past_key_values
            next_token = step_out.logits[:, -1:, :].argmax(dim=-1)
            generated += 1
    sync_cuda()
    decode_sec = elapsed(decode_started)
    return prefill_sec, decode_sec, generated


def run_text_only_prefill(hf_model: torch.nn.Module, tokenized_data: dict[str, torch.Tensor]) -> float:
    """Measure prefill time without visual tensors for a ViT-inclusive delta estimate."""
    forward_inputs = {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "use_cache": True,
        "return_dict": True,
    }
    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode():
        _ = hf_model(**forward_inputs)
    sync_cuda()
    return elapsed(started)


def canonical_frame_offsets(sample_dir: Path) -> list[float]:
    sample_meta = json.loads((sample_dir / "sample_meta.json").read_text(encoding="utf-8"))
    return [float(value) for value in sample_meta.get("frame_offsets_sec", [-0.3, -0.2, -0.1, 0.0])]


def teacher_image_frames(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    sample_input = sample.get("input") or {}
    image_paths = list(sample_input.get("image_paths") or [])
    if image_paths:
        camera_indices = [int(value) for value in sample_input.get("camera_indices") or [0, 1, 2, 6]]
        num_frames = int(sample_input.get("num_frames_per_camera", 4))
        if len(image_paths) < len(camera_indices) * num_frames:
            raise RuntimeError("image_paths does not contain a complete camera/frame grid")
        frames_by_camera = []
        for camera_offset, _ in enumerate(camera_indices):
            camera_frames = []
            base = camera_offset * num_frames
            for frame_offset in range(num_frames):
                image_path = Path(remap_external_path(image_paths[base + frame_offset]) or image_paths[base + frame_offset])
                image = Image.open(image_path).convert("RGB")
                image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous()
                camera_frames.append(image_tensor)
            frames_by_camera.append(torch.stack(camera_frames, dim=0))
        return torch.stack(frames_by_camera, dim=0), torch.tensor(camera_indices, dtype=torch.int64)

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_front_tele_30fov": 6,
    }
    sample_dir = Path(sample["input"]["canonical_sample_path"])
    if not sample_dir.is_absolute():
        sample_dir = PROJECT_ROOT / sample_dir
    frame_offsets = canonical_frame_offsets(sample_dir)
    camera_names = list(camera_name_to_index.keys())
    frames_by_camera = []
    camera_indices = []
    for camera_name in camera_names:
        camera_frames = []
        for offset in frame_offsets:
            image_path = sample_dir / "frames" / f"{camera_name}_t{offset:+.1f}.jpg"
            image = Image.open(image_path).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous()
            camera_frames.append(image_tensor)
        frames_by_camera.append(torch.stack(camera_frames, dim=0))
        camera_indices.append(camera_name_to_index[camera_name])
    return torch.stack(frames_by_camera, dim=0), torch.tensor(camera_indices, dtype=torch.int64)


def build_teacher_tokenized(sample: dict[str, Any], helper_module, processor) -> dict[str, torch.Tensor]:
    image_frames, camera_indices = teacher_image_frames(sample)
    user_text = (
        "<|traj_history_start|>" + ("<|traj_history|>" * 48) + "<|traj_history_end|>"
        "<|question_start|>Explain the chain of causation for the ego vehicle.<|question_end|>"
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {
            "role": "user",
            "content": helper_module._build_image_content(image_frames.flatten(0, 1), camera_indices, 4)
            + [{"type": "text", "text": user_text}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return tokenized


def profile_teacher(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if str(args.alpamayo_src) not in sys.path:
        sys.path.insert(0, str(args.alpamayo_src))
    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    print(json.dumps({"event": "teacher_load_start"}), flush=True)
    sync_cuda()
    started = time.perf_counter()
    config = Alpamayo1_5Config.from_pretrained(str(args.teacher_model_path / "alpamayo_1.5_config.json"))
    model = Alpamayo1_5.from_pretrained(
        str(args.teacher_model_path),
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    sync_cuda()
    load_sec = elapsed(started)
    print(json.dumps({"event": "teacher_load_done", "load_sec": load_sec}), flush=True)

    print(json.dumps({"event": "teacher_tokenize_start"}), flush=True)
    tokenized = build_teacher_tokenized(sample, helper, processor)
    tokenized = to_device(tokenized, args.device)
    print(json.dumps({"event": "teacher_tokenize_done"}), flush=True)
    hf_model = model.vlm
    print(json.dumps({"event": "teacher_prefill_decode_start"}), flush=True)
    prefill_sec, generation_sec, generated_tokens = run_prefill_and_decode(
        hf_model,
        tokenized,
        max_new_tokens=args.max_new_tokens,
    )
    text_only_prefill_sec = run_text_only_prefill(hf_model, tokenized)
    vit_sec = max(prefill_sec - text_only_prefill_sec, 0.0)
    print(
        json.dumps(
            {
                "event": "teacher_prefill_decode_done",
                "prefill_sec": prefill_sec,
                "text_only_prefill_sec": text_only_prefill_sec,
                "vit_sec_est": vit_sec,
                "generation_sec": generation_sec,
            }
        ),
        flush=True,
    )

    del model
    del processor
    sync_cuda()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": "teacher_alpamayo15_vlm",
        "load_sec": load_sec,
        "vit_sec": vit_sec,
        "text_only_prefill_sec": text_only_prefill_sec,
        "prefill_sec": prefill_sec,
        "generation_sec": generation_sec,
        "generated_tokens": generated_tokens,
        "generation_ms_per_token": round((generation_sec / max(generated_tokens, 1)) * 1000.0, 3),
    }


def profile_student(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    train_config = json.loads((args.student_checkpoint_dir / "train_config.json").read_text(encoding="utf-8"))
    base_model = str(train_config["args"]["student_model"])
    use_lora = not bool(train_config["args"].get("disable_lora", False))

    print(json.dumps({"event": "student_load_start"}), flush=True)
    sync_cuda()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.student_checkpoint_dir / "tokenizer", local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.student_checkpoint_dir / "processor", local_files_only=True)
    processor.tokenizer = tokenizer
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int(train_config["trainer_config"]["max_length"]),
        torch_dtype=(
            torch.bfloat16
            if args.device.startswith("cuda") and bool(train_config["trainer_config"].get("bf16", True))
            else None
        ),
        local_files_only=Path(base_model).expanduser().exists(),
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    checkpoint_format = detect_checkpoint_format(args.student_checkpoint_dir)
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(args.student_checkpoint_dir, model, use_lora=use_lora)
    model = model.to(args.device).eval()
    sync_cuda()
    load_sec = elapsed(started)
    print(json.dumps({"event": "student_load_done", "load_sec": load_sec}), flush=True)

    print(json.dumps({"event": "student_tokenize_start"}), flush=True)
    prompt_text = build_user_prompt(sample, PROJECT_ROOT)
    images = load_sample_images(sample, PROJECT_ROOT)
    messages = build_messages(prompt_text, len(images), target_text=None)
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
    tokenized = to_device(tokenized, args.device)
    print(json.dumps({"event": "student_tokenize_done"}), flush=True)

    hf_model = model.backbone
    print(json.dumps({"event": "student_prefill_decode_start"}), flush=True)
    prefill_sec, generation_sec, generated_tokens = run_prefill_and_decode(
        hf_model,
        tokenized,
        max_new_tokens=args.max_new_tokens,
    )
    text_only_prefill_sec = run_text_only_prefill(hf_model, tokenized)
    vit_sec = max(prefill_sec - text_only_prefill_sec, 0.0)
    print(
        json.dumps(
            {
                "event": "student_prefill_decode_done",
                "prefill_sec": prefill_sec,
                "text_only_prefill_sec": text_only_prefill_sec,
                "vit_sec_est": vit_sec,
                "generation_sec": generation_sec,
            }
        ),
        flush=True,
    )

    del model
    del processor
    del tokenizer
    sync_cuda()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": "student_cosmos_reason2b_lora",
        "load_sec": load_sec,
        "vit_sec": vit_sec,
        "text_only_prefill_sec": text_only_prefill_sec,
        "prefill_sec": prefill_sec,
        "generation_sec": generation_sec,
        "generated_tokens": generated_tokens,
        "generation_ms_per_token": round((generation_sec / max(generated_tokens, 1)) * 1000.0, 3),
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.corpus_jsonl)
    sample = pick_sample(rows, args.split, args.sample_index)

    teacher = profile_teacher(sample, args)
    student = profile_student(sample, args)

    summary = {
        "sample_id": sample["sample_id"],
        "split": sample.get("split"),
        "max_new_tokens": args.max_new_tokens,
        "teacher": teacher,
        "student": student,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
