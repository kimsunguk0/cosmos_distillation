#!/usr/bin/env python3
"""Measure free-run emission of trajectory format tokens for a student checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

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
    fuse_history_tokens_in_input_ids,
    load_ego_history_xyz,
    load_sample_images,
    resolve_camera_indices,
)
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--image-prompt-style", choices=("compact", "camera_labeled"), default="camera_labeled")
    parser.add_argument(
        "--prompt-text-style",
        choices=("numeric_history_question", "official_alpamayo"),
        default="official_alpamayo",
    )
    parser.add_argument("--fuse-history-tokens", action="store_true")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(rows: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(rows), width):
        yield rows[index : index + width]


def token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single-token encoding for {token!r}, got {ids}")
    return int(ids[0])


class StopAfterTokenCriteria(StoppingCriteria):
    def __init__(self, *, prompt_lengths: list[int], stop_token_id: int) -> None:
        self.prompt_lengths = [int(value) for value in prompt_lengths]
        self.stop_token_id = int(stop_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for row_index in range(input_ids.shape[0]):
            prompt_len = self.prompt_lengths[min(row_index, len(self.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_len:].tolist()
            if self.stop_token_id not in generated:
                return False
        return True


def load_model(args: argparse.Namespace):
    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    manifest_path = args.checkpoint_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    data_view = train_config.get("data_view") or {}

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
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
        local_files_only=Path(base_model).expanduser().exists(),
        traj_teacher_hidden_size=(
            int(data_view.get("teacher_traj_hidden_size"))
            if data_view.get("teacher_traj_hidden_size") not in (None, "", 0)
            else None
        ),
        traj_hidden_bridge_size=(
            int(manifest.get("traj_hidden_bridge_size"))
            if manifest.get("traj_hidden_bridge_size") not in (None, "", 0)
            else None
        ),
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    if detect_checkpoint_format(args.checkpoint_dir) == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=use_lora)
    return model.to(device).eval(), tokenizer, processor, device, base_model, train_config


def mean(values: list[float | int]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def ratio(count: int, total: int) -> float:
    return float(count / max(int(total), 1))


def token_positions(values: list[int], needle: int) -> list[int]:
    return [index for index, value in enumerate(values) if int(value) == int(needle)]


def main() -> int:
    args = parse_args()
    rows = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    if args.num_samples > 0:
        rows = rows[: int(args.num_samples)]
    if not rows:
        raise SystemExit(f"No rows selected for split={args.split!r}")

    model, tokenizer, processor, device, base_model, train_config = load_model(args)
    model_dtype = next(model.backbone.parameters()).dtype
    cot_end_id = token_id(tokenizer, "<|cot_end|>")
    traj_start_id = token_id(tokenizer, "<|traj_future_start|>")
    traj_end_id = token_id(tokenizer, "<|traj_future_end|>")

    counters = Counter()
    generated_lengths: list[float] = []
    cot_lengths: list[float] = []
    sample_rows: list[dict[str, Any]] = []

    for batch_rows in batched(rows, args.batch_size):
        texts: list[str] = []
        images_batch: list[list[Any]] = []
        histories = []
        for sample in batch_rows:
            history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
            histories.append(history_xyz)
            prompt_text = build_user_prompt(
                sample,
                PROJECT_ROOT,
                ego_history_xyz=history_xyz,
                prompt_text_style=args.prompt_text_style,
            )
            images = load_sample_images(sample, PROJECT_ROOT)
            camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
            frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
            messages = build_messages(
                prompt_text,
                len(images),
                assistant_prefix="<|cot_start|>",
                image_prompt_style=args.image_prompt_style,
                camera_indices=camera_indices,
                num_frames_per_camera=frames_per_camera,
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            texts.append(text)
            images_batch.append(images)

        batch = processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        if args.fuse_history_tokens:
            batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, histories)
        prompt_len = int(batch["input_ids"].shape[1])
        moved = {
            key: (
                value.to(device=device, dtype=model_dtype)
                if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
                else value.to(device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

        with torch.inference_mode():
            generated = model.backbone.generate(
                **moved,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList(
                    [StopAfterTokenCriteria(prompt_lengths=[prompt_len] * len(batch_rows), stop_token_id=traj_start_id)]
                ),
                pad_token_id=tokenizer.pad_token_id,
            )

        for row_index, sample in enumerate(batch_rows):
            new_ids = [int(value) for value in generated[row_index, prompt_len:].detach().cpu().tolist()]
            cot_end_positions = token_positions(new_ids, cot_end_id)
            traj_start_positions = token_positions(new_ids, traj_start_id)
            traj_end_positions = token_positions(new_ids, traj_end_id)
            cot_end_pos = cot_end_positions[0] if cot_end_positions else None
            traj_start_pos = traj_start_positions[0] if traj_start_positions else None
            traj_end_pos = traj_end_positions[0] if traj_end_positions else None
            valid_order = (
                cot_end_pos is not None
                and traj_start_pos is not None
                and (traj_end_pos is None or traj_start_pos < traj_end_pos)
                and cot_end_pos < traj_start_pos
            )
            cot_stop = cot_end_pos if cot_end_pos is not None else traj_start_pos if traj_start_pos is not None else len(new_ids)
            cot_lengths.append(float(max(int(cot_stop), 0)))
            generated_lengths.append(float(len(new_ids)))

            counters["samples"] += 1
            counters["cot_end_hit"] += int(cot_end_pos is not None)
            counters["traj_start_hit"] += int(traj_start_pos is not None)
            counters["traj_start_after_cot_end"] += int(
                cot_end_pos is not None and traj_start_pos is not None and cot_end_pos < traj_start_pos
            )
            counters["traj_end_hit"] += int(traj_end_pos is not None)
            counters["multi_start"] += int(len(traj_start_positions) > 1)
            counters["invalid_special_order"] += int(traj_start_pos is not None and not valid_order)
            counters["max_new_tokens_without_traj_start"] += int(traj_start_pos is None)

            generated_text = tokenizer.decode(new_ids, skip_special_tokens=False)
            sample_rows.append(
                {
                    "sample_id": str(sample.get("sample_id")),
                    "generated_new_token_count": int(len(new_ids)),
                    "cot_end_hit": cot_end_pos is not None,
                    "traj_start_hit": traj_start_pos is not None,
                    "traj_start_after_cot_end": (
                        cot_end_pos is not None and traj_start_pos is not None and cot_end_pos < traj_start_pos
                    ),
                    "traj_end_hit": traj_end_pos is not None,
                    "traj_start_count": int(len(traj_start_positions)),
                    "cot_end_position": int(cot_end_pos + 1) if cot_end_pos is not None else None,
                    "traj_start_position": int(traj_start_pos + 1) if traj_start_pos is not None else None,
                    "traj_end_position": int(traj_end_pos + 1) if traj_end_pos is not None else None,
                    "invalid_special_order": bool(traj_start_pos is not None and not valid_order),
                    "generated_text_preview": generated_text[:240],
                }
            )

        print(
            json.dumps({"event": "batch_done", "done": len(sample_rows), "total": len(rows)}),
            flush=True,
        )

    n = int(counters["samples"])
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "student_model": str(base_model),
        "split": str(args.split),
        "num_samples": n,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": int(args.max_new_tokens),
            "stop_after_token": "<|traj_future_start|>",
        },
        "prompt": {
            "image_prompt_style": str(args.image_prompt_style),
            "prompt_text_style": str(args.prompt_text_style),
            "fuse_history_tokens": bool(args.fuse_history_tokens),
        },
        "checkpoint_train_data_view": train_config.get("data_view") or {},
        "cot_end_hit_rate": ratio(counters["cot_end_hit"], n),
        "traj_start_hit_rate": ratio(counters["traj_start_hit"], n),
        "traj_start_after_cot_end_rate": ratio(counters["traj_start_after_cot_end"], n),
        "traj_end_hit_rate": ratio(counters["traj_end_hit"], n),
        "multi_start_rate": ratio(counters["multi_start"], n),
        "invalid_special_order_rate": ratio(counters["invalid_special_order"], n),
        "max_new_tokens_without_traj_start_rate": ratio(counters["max_new_tokens_without_traj_start"], n),
        "generated_new_tokens_mean": mean(generated_lengths),
        "cot_tokens_before_boundary_mean": mean(cot_lengths),
        "samples": sample_rows,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.samples_jsonl is not None:
        args.samples_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.samples_jsonl.open("w", encoding="utf-8") as handle:
            for row in sample_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
