#!/usr/bin/env python3
"""Extract teacher hidden/logit caches from existing teacher text outputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.teacher_cache import CAMERA_NAME_TO_INDEX, load_jsonl_by_key, write_jsonl


DEFAULT_ALPAMAYO_SRC = Path("/home/pm97/workspace/sukim/alpamayo1.5/src")
DEFAULT_ALPAMAYO_MODEL = Path("/home/pm97/workspace/sukim/weights/alpamayo15_vlm_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-index-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text" / "index.jsonl",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "teacher_cache" / "text",
    )
    parser.add_argument(
        "--alpamayo-src",
        type=Path,
        default=DEFAULT_ALPAMAYO_SRC,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_ALPAMAYO_MODEL,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--logit-topk", type=int, default=32)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_signal_cache_summary.json",
    )
    return parser.parse_args()


def load_teacher_model(args: argparse.Namespace):
    if str(args.alpamayo_src) not in sys.path:
        sys.path.insert(0, str(args.alpamayo_src))

    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    config = Alpamayo1_5Config.from_pretrained(str(args.model_path / "alpamayo_1.5_config.json"))
    model = Alpamayo1_5.from_pretrained(
        str(args.model_path),
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    return helper, model, processor


def canonical_frame_offsets(sample_dir: Path) -> list[float]:
    sample_meta_path = sample_dir / "sample_meta.json"
    sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    return [float(value) for value in sample_meta.get("frame_offsets_sec", [-0.3, -0.2, -0.1, 0.0])]


def load_image_frames(sample_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    camera_names = list(CAMERA_NAME_TO_INDEX)
    frame_offsets = canonical_frame_offsets(sample_dir)
    frames_by_camera = []
    camera_indices = []
    for camera_name in camera_names:
        camera_frames = []
        for offset in frame_offsets:
            image_path = sample_dir / "frames" / f"{camera_name}_t{offset:+.1f}.jpg"
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image, copy=True)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
            camera_frames.append(image_tensor)
        frames_by_camera.append(torch.stack(camera_frames, dim=0))
        camera_indices.append(CAMERA_NAME_TO_INDEX[camera_name])
    return torch.stack(frames_by_camera, dim=0), torch.tensor(camera_indices, dtype=torch.int64)


def build_messages(sample_dir: Path, target_text: str, *, helper_module) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    image_frames, camera_indices = load_image_frames(sample_dir)
    prompt_messages = helper_module.create_message(
        frames=image_frames.flatten(0, 1),
        camera_indices=camera_indices,
    )
    full_messages = copy.deepcopy(prompt_messages)
    full_messages[-1]["content"][0]["text"] = f"<|cot_start|>{target_text.strip()}\n<|cot_end|>"
    return prompt_messages, full_messages


def encode_messages(messages: list[dict[str, Any]], processor) -> dict[str, torch.Tensor]:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )


def extract_teacher_signals(
    *,
    sample_dir: Path,
    target_text: str,
    helper_module,
    processor,
    model,
    device: str,
    logit_topk: int,
) -> dict[str, Any]:
    prompt_messages, full_messages = build_messages(sample_dir, target_text, helper_module=helper_module)
    prompt_batch = helper_module.to_device(encode_messages(prompt_messages, processor), device)
    full_batch = helper_module.to_device(encode_messages(full_messages, processor), device)

    prompt_len = int(prompt_batch["attention_mask"].sum().item())
    full_input_ids = full_batch["input_ids"]
    full_attention_mask = full_batch["attention_mask"]
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        outputs = model.vlm(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=full_batch.get("pixel_values"),
            image_grid_thw=full_batch.get("image_grid_thw"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    logits = outputs.logits[:, :-1, :].float()
    hidden_states = outputs.hidden_states[-1][:, :-1, :].float()
    target_token_ids = full_input_ids[:, 1:]
    valid_mask = torch.zeros_like(target_token_ids, dtype=torch.bool)
    valid_mask[:, prompt_len - 1 :] = True

    target_logits = logits[0][valid_mask[0]]
    target_hidden = hidden_states[0][valid_mask[0]]
    target_ids = target_token_ids[0][valid_mask[0]]
    topk = min(int(logit_topk), int(target_logits.shape[-1]))
    topk_values, topk_indices = torch.topk(target_logits, k=topk, dim=-1)
    pooled_hidden = target_hidden.mean(dim=0)

    return {
        "pooled_hidden": pooled_hidden.cpu().numpy().astype(np.float32),
        "topk_indices": topk_indices.cpu().numpy().astype(np.int32),
        "topk_logits": topk_values.cpu().numpy().astype(np.float32),
        "target_token_ids": target_ids.cpu().numpy().astype(np.int32),
        "target_token_count": int(target_ids.numel()),
        "hidden_dim": int(pooled_hidden.numel()),
        "topk": int(topk),
    }


def select_records(records: list[dict[str, Any]], overwrite: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        output = record.get("output") or {}
        if record.get("status") != "ok":
            continue
        if not output.get("teacher_short_reason"):
            continue
        if (
            not overwrite
            and output.get("teacher_hidden_path")
            and output.get("teacher_logit_cache_path")
            and not output.get("teacher_signal_cache_stale")
        ):
            continue
        selected.append(record)
    return selected


def main() -> None:
    args = parse_args()
    started_at = time.time()
    records = list(load_jsonl_by_key(args.teacher_index_jsonl).values())
    selected = select_records(records, overwrite=args.overwrite)
    if args.max_samples is not None:
        selected = selected[: args.max_samples]

    hidden_dir = args.cache_root / "hidden"
    logits_dir = args.cache_root / "logits"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    logits_dir.mkdir(parents=True, exist_ok=True)

    helper, model, processor = load_teacher_model(args)

    processed = 0
    succeeded = 0
    failed = 0
    failures: list[dict[str, str]] = []
    selected_ids = {record["sample_id"] for record in selected}

    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in selected_ids:
            continue
        processed += 1
        output = record.setdefault("output", {})
        sample_dir = Path(record["canonical_sample_path"])
        hidden_path = hidden_dir / f"{sample_id}.pooled_hidden.npy"
        logits_path = logits_dir / f"{sample_id}.topk_logits.npz"
        try:
            signals = extract_teacher_signals(
                sample_dir=sample_dir,
                target_text=str(output["teacher_short_reason"]),
                helper_module=helper,
                processor=processor,
                model=model,
                device=args.device,
                logit_topk=args.logit_topk,
            )
            np.save(hidden_path, signals["pooled_hidden"])
            np.savez_compressed(
                logits_path,
                topk_indices=signals["topk_indices"],
                topk_logits=signals["topk_logits"],
                target_token_ids=signals["target_token_ids"],
                target_token_count=np.asarray(signals["target_token_count"], dtype=np.int32),
                hidden_dim=np.asarray(signals["hidden_dim"], dtype=np.int32),
                topk=np.asarray(signals["topk"], dtype=np.int32),
            )
            output["teacher_hidden_path"] = str(hidden_path)
            output["teacher_logit_cache_path"] = str(logits_path)
            output["teacher_signal_target_field"] = "teacher_short_reason"
            output["teacher_signal_target_source"] = output.get("teacher_short_reason_source")
            output["teacher_signal_cache_stale"] = False
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

    write_jsonl(args.teacher_index_jsonl, records)
    summary = {
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "processed_records": processed,
        "succeeded": succeeded,
        "failed": failed,
        "logit_topk": args.logit_topk,
        "hidden_dir": str(hidden_dir),
        "logits_dir": str(logits_dir),
        "failures": failures[:20],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
