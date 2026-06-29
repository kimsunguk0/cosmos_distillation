#!/usr/bin/env python3
"""One-batch ML-FLEX DeepStack contract smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.collator import DistillationCollator  # noqa: E402
from src.training.flex_batch import attach_qwen_mrope_position_ids, compress_batch_for_flex  # noqa: E402
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def _load_eval104():
    path = PROJECT_ROOT / "scripts" / "104_eval_flex_teacher_parity.py"
    spec = importlib.util.spec_from_file_location("flex_teacher_parity_eval104", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval104 = _load_eval104()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    rows = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    if not rows:
        raise SystemExit(f"No rows selected for split={args.split!r} in {args.corpus_jsonl}")
    row = rows[int(args.sample_index) % len(rows)]

    print(json.dumps({"event": "load_model_start", "checkpoint_dir": str(args.checkpoint_dir)}), flush=True)
    model, tokenizer, processor, _base_model, train_config = eval104.load_model(
        args.checkpoint_dir,
        student_model=args.student_model,
        device=device,
    )
    flex_cfg = getattr(model, "flex_scene_config", None)
    if flex_cfg is None or not bool(getattr(flex_cfg, "enabled", False)):
        raise SystemExit("Checkpoint is not FLEX-enabled.")
    if str(getattr(flex_cfg, "architecture", "single_level")) != "multi_level":
        raise SystemExit(f"Expected ML-FLEX architecture, got {getattr(flex_cfg, 'architecture', None)!r}.")

    data_view = train_config.get("data_view") or {}
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=bool(data_view.get("teacher_pair_target", False)),
        enable_teacher_view=False,
        enable_action_aux=False,
        hard_view_uses_teacher_cot=bool(data_view.get("hard_view_uses_teacher_cot", True)),
        prompt_mode=str(data_view.get("prompt_mode") or "joint"),
        target_mode=str(data_view.get("target_mode") or "joint"),
        image_prompt_style=str(data_view.get("image_prompt_style") or "camera_labeled"),
        prompt_text_style=str(data_view.get("prompt_text_style") or "official_alpamayo"),
        fuse_history_tokens=bool(data_view.get("fuse_history_tokens", True)),
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
    )

    batch = collator([row])
    image_token_id = int(getattr(model, "image_token_id"))
    original_image_tokens = int((batch["input_ids"] == image_token_id).sum().item())
    batch = attach_qwen_mrope_position_ids(batch, model)
    batch = compress_batch_for_flex(
        batch,
        image_token_id=image_token_id,
        tokens_per_image=int(getattr(flex_cfg, "tokens_per_image")),
        pad_token_id=int(getattr(model, "pad_token_id", 0) or 0),
        preserve_original_position_ids=True,
        selection_strategy=str(getattr(flex_cfg, "selection_strategy", "first") or "first"),
    )
    batch["flex_scene_deepstack"] = True
    compressed_image_tokens = int((batch["input_ids"] == image_token_id).sum().item())
    dtype = next(model.backbone.parameters()).dtype
    moved = eval104.move_batch(batch, device=device, dtype=dtype)

    language_model = model._conditional_backbone().model.language_model
    original_forward = language_model.forward
    captured: dict[str, Any] = {}

    def wrapped_forward(*forward_args, **forward_kwargs):
        visual_pos_masks = forward_kwargs.get("visual_pos_masks")
        deepstack_visual_embeds = forward_kwargs.get("deepstack_visual_embeds")
        captured["visual_pos_masks_shape"] = _tensor_shape(visual_pos_masks)
        captured["visual_pos_masks_sum"] = int(visual_pos_masks.sum().item()) if isinstance(visual_pos_masks, torch.Tensor) else None
        captured["deepstack_visual_embeds_shapes"] = (
            [_tensor_shape(tensor) for tensor in deepstack_visual_embeds]
            if isinstance(deepstack_visual_embeds, (list, tuple))
            else None
        )
        captured["deepstack_visual_embeds_dtypes"] = (
            [str(tensor.dtype) for tensor in deepstack_visual_embeds]
            if isinstance(deepstack_visual_embeds, (list, tuple))
            else None
        )
        return original_forward(*forward_args, **forward_kwargs)

    language_model.forward = wrapped_forward
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                pixel_values=moved["pixel_values"],
                image_grid_thw=moved["image_grid_thw"],
                camera_indices=moved.get("camera_indices"),
                relative_timestamps=moved.get("relative_timestamps"),
                camera_counts=moved.get("camera_counts"),
                frames_per_camera=moved.get("frames_per_camera"),
                position_ids=moved.get("position_ids"),
                flex_scene_deepstack=True,
                return_hidden_states=True,
                compute_meta_action=False,
                compute_traj_aux=False,
            )
    finally:
        language_model.forward = original_forward

    expected_scene_tokens = int(getattr(flex_cfg, "scene_tokens"))
    deepstack_shapes = captured.get("deepstack_visual_embeds_shapes")
    contract_ok = (
        captured.get("visual_pos_masks_sum") == expected_scene_tokens
        and isinstance(deepstack_shapes, list)
        and len(deepstack_shapes) == int(getattr(flex_cfg, "num_deepstack_levels"))
        and all(shape == [expected_scene_tokens, int(getattr(flex_cfg, "input_hidden_size"))] for shape in deepstack_shapes)
    )
    summary = {
        "event": "mlflex_forward_smoke_done",
        "contract_ok": bool(contract_ok),
        "checkpoint_dir": str(args.checkpoint_dir),
        "corpus_jsonl": str(args.corpus_jsonl),
        "sample_id": row.get("sample_id"),
        "architecture": str(getattr(flex_cfg, "architecture")),
        "tokens_per_image": int(getattr(flex_cfg, "tokens_per_image")),
        "scene_tokens": expected_scene_tokens,
        "num_deepstack_levels": int(getattr(flex_cfg, "num_deepstack_levels")),
        "original_seq_len": int(collator([row])["input_ids"].shape[1]),
        "compressed_seq_len": int(moved["input_ids"].shape[1]),
        "original_image_tokens": original_image_tokens,
        "compressed_image_tokens": compressed_image_tokens,
        "flex_stats": batch.get("flex_stats"),
        "captured": captured,
        "logits_shape": _tensor_shape(outputs.get("logits")),
        "hidden_shape": _tensor_shape(outputs.get("hidden_states")),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if not contract_ok:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
