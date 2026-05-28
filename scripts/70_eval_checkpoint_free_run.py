#!/usr/bin/env python3
"""Run free-run decode evaluation for a student checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import DecodeEvalConfig, evaluate_decode_subset  # noqa: E402
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-mode", default=None)
    parser.add_argument("--target-mode", default=None)
    parser.add_argument("--image-prompt-style", default=None)
    parser.add_argument("--prompt-text-style", default=None)
    parser.add_argument("--fuse-history-tokens", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--metric-name", default="free_run_geometry_score")
    parser.add_argument("--summary-json", type=Path, required=True)
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


def load_model(args: argparse.Namespace):
    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    checkpoint_manifest_path = args.checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")) if checkpoint_manifest_path.exists() else {}
    )
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
            int(checkpoint_manifest.get("traj_hidden_bridge_size"))
            if checkpoint_manifest.get("traj_hidden_bridge_size") not in (None, "", 0)
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


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.corpus_jsonl)
    model, tokenizer, processor, device, base_model, train_config = load_model(args)
    data_view = train_config.get("data_view") or {}

    config = DecodeEvalConfig(
        enabled=True,
        split=str(args.split),
        num_samples=int(args.num_samples),
        max_new_tokens=int(args.max_new_tokens),
        prompt_mode=str(args.prompt_mode or data_view.get("prompt_mode") or "joint"),
        target_mode=str(args.target_mode or data_view.get("target_mode") or "joint"),
        image_prompt_style=str(args.image_prompt_style or data_view.get("image_prompt_style") or "camera_labeled"),
        prompt_text_style=str(args.prompt_text_style or data_view.get("prompt_text_style") or "official_alpamayo"),
        fuse_history_tokens=(
            bool(args.fuse_history_tokens)
            if args.fuse_history_tokens is not None
            else bool(data_view.get("fuse_history_tokens", False))
        ),
        metric_name=str(args.metric_name),
    )
    print(
        json.dumps(
            {
                "event": "eval_start",
                "checkpoint_dir": str(args.checkpoint_dir),
                "flex_enabled": bool(hasattr(model, "flex_enabled") and model.flex_enabled()),
                "config": {
                    "split": config.split,
                    "num_samples": config.num_samples,
                    "max_new_tokens": config.max_new_tokens,
                    "prompt_mode": config.prompt_mode,
                    "target_mode": config.target_mode,
                    "image_prompt_style": config.image_prompt_style,
                    "prompt_text_style": config.prompt_text_style,
                    "fuse_history_tokens": config.fuse_history_tokens,
                },
            }
        ),
        flush=True,
    )
    result = evaluate_decode_subset(
        model,
        tokenizer=tokenizer,
        processor=processor,
        records=rows,
        device=device,
        project_root=PROJECT_ROOT,
        config=config,
        student_model=base_model,
    )
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
