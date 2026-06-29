#!/usr/bin/env python3
"""Create an F0 checkpoint: a frozen no-FLEX checkpoint plus untrained FLEX.

This is a diagnostic artifact.  It preserves the source checkpoint behavior
weights and adds a freshly initialized FLEX scene encoder so free-run decode can
measure the damage from compressed visual tokens before any FLEX training.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint, save_student_checkpoint  # noqa: E402
from src.model.flex_scene_encoder import FlexSceneConfig  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument(
        "--architecture",
        choices=("single_level", "multi_level"),
        default="single_level",
    )
    parser.add_argument("--tokens-per-image", type=int, default=56)
    parser.add_argument("--expected-images-per-sample", type=int, default=16)
    parser.add_argument("--input-hidden-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-camera-time-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-local-slot-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-camera-types", type=int, default=16)
    parser.add_argument("--num-deepstack-levels", type=int, default=3)
    parser.add_argument(
        "--compression-mode",
        choices=("global", "per_image", "anchored_per_image"),
        default="global",
    )
    parser.add_argument("--selection-strategy", choices=("first", "uniform"), default="first")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    args = parse_args()
    base_checkpoint_dir = args.base_checkpoint_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    train_config = _load_json(base_checkpoint_dir / "train_config.json")
    manifest = _load_json(base_checkpoint_dir / "checkpoint_manifest.json")
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    checkpoint_format = detect_checkpoint_format(base_checkpoint_dir)
    if checkpoint_format == "lora_adapter" and not use_lora:
        raise ValueError("Base checkpoint is a LoRA adapter but train_config disables LoRA.")

    tokenizer_dir = base_checkpoint_dir / "tokenizer"
    processor_dir = base_checkpoint_dir / "processor"
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

    data_view = train_config.get("data_view") or {}
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=torch.bfloat16,
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
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(base_checkpoint_dir, model, use_lora=use_lora, adapter_trainable=False)

    torch.manual_seed(int(args.seed))
    compression_mode = str(args.compression_mode)
    if str(args.architecture) == "multi_level" and compression_mode == "global":
        compression_mode = "per_image"
    flex_cfg = FlexSceneConfig(
        enabled=True,
        architecture=str(args.architecture),
        tokens_per_image=int(args.tokens_per_image),
        expected_images_per_sample=int(args.expected_images_per_sample),
        input_hidden_size=int(args.input_hidden_size),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        use_camera_time_embeddings=bool(args.use_camera_time_embeddings),
        use_local_slot_embeddings=bool(args.use_local_slot_embeddings),
        max_camera_types=int(args.max_camera_types),
        compression_mode=compression_mode,
        selection_strategy=str(args.selection_strategy),
        num_deepstack_levels=int(args.num_deepstack_levels),
    )
    model.configure_flex_scene(flex_cfg)

    checkpoint_payload = save_student_checkpoint(
        output_dir,
        model,
        tokenizer,
        processor,
        use_lora=use_lora,
    )
    updated_train_config = dict(train_config)
    updated_train_config["f0_untrained_flex"] = {
        "source_checkpoint_dir": str(base_checkpoint_dir),
        "seed": int(args.seed),
        "note": "B0 weights preserved; FLEX scene encoder freshly initialized.",
    }
    updated_train_config["checkpoint"] = checkpoint_payload
    (output_dir / "train_config.json").write_text(
        json.dumps(updated_train_config, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "f0_checkpoint_created",
                "output_dir": str(output_dir),
                "source_checkpoint_dir": str(base_checkpoint_dir),
                "use_lora": bool(use_lora),
                "checkpoint_format": checkpoint_format,
                "flex_scene_tokens": int(flex_cfg.scene_tokens),
                "checkpoint_payload": checkpoint_payload,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
