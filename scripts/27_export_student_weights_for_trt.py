#!/usr/bin/env python3
"""Export a distillation student checkpoint as merged HF weights for TRT tests."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, default=None)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _copy_tree_files(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dst / item.name)


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    train_config = _json_load(checkpoint_dir / "train_config.json")
    checkpoint_manifest = _json_load(checkpoint_dir / "checkpoint_manifest.json")
    data_view = train_config.get("data_view") or {}
    train_args = train_config.get("args") or {}
    student_model = str(args.student_model or train_args.get("student_model") or "")
    if not student_model:
        raise SystemExit("Could not infer student model path; pass --student-model.")
    student_model_path = str(Path(student_model).expanduser())
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    tokenizer_dir = checkpoint_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir if tokenizer_dir.exists() else student_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    wrapper_cfg = StudentWrapperConfig(
        student_model_name=student_model_path,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=dtype,
        local_files_only=True,
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
        traj_aux_num_buckets=int(checkpoint_manifest.get("traj_aux_num_buckets") or 1),
    )
    print(
        json.dumps(
            {
                "event": "load_start",
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_format": detect_checkpoint_format(checkpoint_dir),
                "student_model": student_model_path,
                "tokenizer_len": len(tokenizer),
                "dtype": args.dtype,
                "device": args.device,
            }
        ),
        flush=True,
    )
    model = build_student_model(wrapper_cfg, tokenizer)
    load_info = load_student_checkpoint(checkpoint_dir, model, use_lora=True)
    model = model.to(args.device).eval()
    backbone = model.backbone
    merged_lora = False
    if hasattr(backbone, "merge_and_unload"):
        print(json.dumps({"event": "merge_lora_start"}), flush=True)
        backbone = backbone.merge_and_unload()
        merged_lora = True
    backbone = backbone.to(dtype=dtype)
    if hasattr(backbone, "config"):
        backbone.config.torch_dtype = dtype
    print(json.dumps({"event": "save_start", "output_dir": str(output_dir), "merged_lora": merged_lora}), flush=True)
    with torch.inference_mode():
        backbone.save_pretrained(output_dir, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(output_dir)
    _copy_tree_files(checkpoint_dir / "processor", output_dir)
    export_info = {
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_student_model": student_model_path,
        "checkpoint_load_info": load_info,
        "checkpoint_manifest": checkpoint_manifest,
        "merged_lora": merged_lora,
        "dtype": args.dtype,
        "tokenizer_len": len(tokenizer),
        "note": "Backbone-only HF export for TensorRT Edge-LLM; training-only distill heads are intentionally omitted.",
    }
    (output_dir / "student_export_info.json").write_text(json.dumps(export_info, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "output_dir": str(output_dir), "export_info": export_info}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
