#!/usr/bin/env python3
"""Run student multimodal inference smoke on a few corpus samples."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

from src.data.parsers import action_record_from_text
from src.inference.decoding import (
    StopOnTrajEndCriteria,
    StopOnTrajOnlyEndCriteria,
    TrajDecodingContract,
    TrajOnlyDecodingContract,
    TrajOnlyLogitsProcessor,
    TrajSpanLogitsProcessor,
)
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import StudentWrapperConfig, build_student_model
from src.model.tokenizer_ext import distill_trainable_token_ids
from src.training.collator import build_messages, build_traj_only_prompt, build_user_prompt, load_sample_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "distill_v3_2.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "stage_b_v3_2" / "final",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--disable-traj-constraint",
        action="store_true",
        help="Disable structured trajectory decoding after `<|cot_end|>`.",
    )
    parser.add_argument(
        "--forced-traj-token-count",
        type=int,
        default=None,
        help="Override the number of `<i...>` trajectory tokens to emit before `<|traj_future_end|>`.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "student_infer_smoke_v3_2.json",
    )
    parser.add_argument(
        "--traj-source",
        choices=("lm", "aux_head", "aux_lm_blend"),
        default="lm",
        help="Use LM logits or the training-time trajectory auxiliary head for the 128-token traj body.",
    )
    parser.add_argument("--aux-blend-sigma", type=float, default=96.0)
    parser.add_argument("--aux-blend-weight", type=float, default=4.0)
    parser.add_argument(
        "--aux-bound-scale",
        type=float,
        default=1.0,
        help="Scale the aux-head tanh bound at inference time to reduce extreme control saturation.",
    )
    parser.add_argument(
        "--aux-edge-margin",
        type=int,
        default=0,
        help="Clamp aux-predicted traj token ids away from the vocab edges by this many bins.",
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


def pick_samples(rows: list[dict[str, Any]], split: str, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split") != split:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def decode_generated_text(tokenizer, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = int(input_ids.shape[-1])
    new_tokens = generated_ids[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    text = text.replace("<|im_start|>assistant", "\n").replace("<|im_end|>", "\n")
    text = re.sub(r"^\s*assistant[:;]?\s*", "", text, flags=re.IGNORECASE)
    lines = [line.strip(" ;") for line in text.splitlines() if line.strip(" ;")]
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n".join(deduped).strip()


def extract_generated_spans(generated_text: str, *, target_mode: str) -> tuple[str, list[int]]:
    if target_mode == "traj_only":
        return "", [int(match) for match in re.findall(r"<i(\d+)>", generated_text)]
    cot_text = generated_text
    if "<|cot_end|>" in cot_text:
        cot_text = cot_text.split("<|cot_end|>", 1)[0]
    if "<|traj_future_start|>" in generated_text and "<|traj_future_end|>" in generated_text:
        traj_text = generated_text.split("<|traj_future_start|>", 1)[1].split("<|traj_future_end|>", 1)[0]
        traj_tokens = [int(match) for match in re.findall(r"<i(\d+)>", traj_text)]
    else:
        traj_tokens = []
    return cot_text.strip(), traj_tokens


def load_model_and_processors(checkpoint_dir: Path, device: str):
    train_config = json.loads((checkpoint_dir / "train_config.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads((checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    base_model = str(train_config["args"]["student_model"])
    use_lora = not bool(train_config["args"].get("disable_lora", False))
    data_view = train_config.get("data_view") or {}

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer", local_files_only=True)
    processor = AutoProcessor.from_pretrained(checkpoint_dir / "processor", local_files_only=True)
    processor.tokenizer = tokenizer

    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int(train_config["trainer_config"]["max_length"]),
        torch_dtype=(
            torch.bfloat16
            if device.startswith("cuda") and bool(train_config["trainer_config"].get("bf16", True))
            else None
        ),
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
    print(json.dumps({"event": "load_base_model_start", "base_model": base_model}), flush=True)
    started_at = time.time()
    model = build_student_model(wrapper_cfg, tokenizer)
    print(json.dumps({"event": "load_base_model_done", "elapsed_sec": round(time.time() - started_at, 3)}), flush=True)
    checkpoint_format = detect_checkpoint_format(checkpoint_dir)
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    print(json.dumps({"event": "checkpoint_load_start", "format": checkpoint_format}), flush=True)
    started_at = time.time()
    load_info = load_student_checkpoint(checkpoint_dir, model, use_lora=use_lora)
    print(
        json.dumps(
            {
                "event": "checkpoint_load_done",
                "format": load_info["format"],
                "elapsed_sec": round(time.time() - started_at, 3),
                "missing_count": len(load_info.get("missing", [])),
                "unexpected_count": len(load_info.get("unexpected", [])),
            }
        ),
        flush=True,
    )
    model = model.to(device).eval()
    print(json.dumps({"event": "model_to_device_done", "device": device}), flush=True)
    return model, tokenizer, processor, {
        **load_info,
        "use_lora": use_lora,
        "data_view": data_view,
        "traj_decode": train_config.get("traj_decode") or {},
    }


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected single-token encoding for {token!r}, got {token_ids}")
    return int(token_ids[0])


def _traj_token_start_id(tokenizer) -> int:
    value = getattr(tokenizer, "traj_token_start_idx", None)
    if isinstance(value, int) and value >= 0:
        return int(value)
    return int(tokenizer.convert_tokens_to_ids("<i0>"))


def _load_traj_decode_runtime(load_info: dict[str, Any]) -> dict[str, Any]:
    traj_decode = dict(load_info.get("traj_decode") or {})
    config_path = traj_decode.get("config_path")
    if not config_path:
        raise ValueError("Checkpoint is missing traj_decode.config_path required for aux_head decoding.")
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    traj_cfg = payload.get("traj_tokenizer_cfg") or {}
    return {
        "num_bins": int(traj_cfg["num_bins"]),
        "dims_min": tuple(float(value) for value in traj_cfg["dims_min"]),
        "dims_max": tuple(float(value) for value in traj_cfg["dims_max"]),
        "n_waypoints": int((traj_cfg.get("action_space_cfg") or {}).get("n_waypoints", traj_decode.get("n_waypoints", 64))),
    }


def _predict_aux_traj_token_id(
    *,
    aux_logits: torch.Tensor,
    token_index: int,
    tokenizer,
    aux_runtime: dict[str, Any],
    traj_runtime: dict[str, Any],
    aux_bound_scale: float,
    aux_edge_margin: int,
) -> int:
    num_buckets = max(int(aux_runtime.get("num_buckets", max(aux_logits.numel() // 2, 1))), 1)
    waypoint_index = int(token_index // 2)
    parity = int(token_index % 2)
    bucket_index = min((waypoint_index * num_buckets) // max(int(traj_runtime["n_waypoints"]), 1), num_buckets - 1)
    channel_index = bucket_index * 2 + parity
    scalar = float(aux_logits[channel_index].item())
    if bool(aux_runtime.get("normalize_targets", False)):
        tanh_bound = float(aux_runtime.get("tanh_bound", 3.0)) * max(float(aux_bound_scale), 0.0)
        scalar = tanh_bound * math.tanh(scalar)
        target_means = aux_runtime.get("target_means") or []
        target_stds = aux_runtime.get("target_stds") or []
        if target_means and target_stds:
            mean = float(target_means[bucket_index][parity])
            std = max(float(target_stds[bucket_index][parity]), 1e-3)
            scalar = mean + std * scalar
    dims_min = traj_runtime["dims_min"][parity]
    dims_max = traj_runtime["dims_max"][parity]
    scalar = max(min(scalar, dims_max), dims_min)
    ratio = 0.0 if dims_max <= dims_min else (scalar - dims_min) / (dims_max - dims_min)
    token_offset = int(round(ratio * max(int(traj_runtime["num_bins"]) - 1, 1)))
    token_offset = max(0, min(token_offset, int(traj_runtime["num_bins"]) - 1))
    edge_margin = max(int(aux_edge_margin), 0)
    if edge_margin > 0 and int(traj_runtime["num_bins"]) > edge_margin * 2:
        token_offset = max(edge_margin, min(token_offset, int(traj_runtime["num_bins"]) - 1 - edge_margin))
    return _traj_token_start_id(tokenizer) + token_offset


def _manual_generate_with_aux_head(
    model,
    batch: dict[str, Any],
    tokenizer,
    *,
    target_traj_token_count: int,
    max_new_tokens: int,
    load_info: dict[str, Any],
    traj_source: str,
    aux_blend_sigma: float,
    aux_blend_weight: float,
    aux_bound_scale: float,
    aux_edge_margin: int,
) -> torch.Tensor:
    device = batch["input_ids"].device
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )
    cot_end_id = _single_token_id(tokenizer, "<|cot_end|>")
    traj_start_id = _single_token_id(tokenizer, "<|traj_future_start|>")
    traj_end_id = _single_token_id(tokenizer, "<|traj_future_end|>")
    aux_runtime = dict(((load_info.get("data_view") or {}).get("traj_aux_interface_runtime")) or {})
    traj_runtime = _load_traj_decode_runtime(load_info)
    traj_token_start_id = _traj_token_start_id(tokenizer)
    traj_token_ids = torch.arange(
        traj_token_start_id,
        traj_token_start_id + int(traj_runtime["num_bins"]),
        device=device,
        dtype=torch.long,
    )

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    generated: list[int] = []

    cot_closed = False
    traj_started = False
    for _ in range(max_new_tokens):
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )
        next_logits = outputs["logits"][:, -1, :]
        if not cot_closed:
            next_token_id = int(torch.argmax(next_logits, dim=-1).item())
            cot_closed = next_token_id == cot_end_id
        elif not traj_started:
            next_token_id = traj_start_id
            traj_started = True
        else:
            break
        next_token = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )
        generated.append(next_token_id)
        if traj_started:
            break

    if not traj_started:
        return input_ids

    for token_index in range(target_traj_token_count):
        with autocast_context:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )
        aux_logits = outputs["traj_aux_values"][0, -1]
        aux_token_id = _predict_aux_traj_token_id(
            aux_logits=aux_logits,
            token_index=token_index,
            tokenizer=tokenizer,
            aux_runtime=aux_runtime,
            traj_runtime=traj_runtime,
            aux_bound_scale=aux_bound_scale,
            aux_edge_margin=aux_edge_margin,
        )
        next_token_id = aux_token_id
        if traj_source == "aux_lm_blend":
            lm_scores = outputs["logits"][0, -1, traj_token_ids].float()
            aux_offset = float(aux_token_id - traj_token_start_id)
            offsets = torch.arange(int(traj_runtime["num_bins"]), device=device, dtype=torch.float32)
            sigma = max(float(aux_blend_sigma), 1.0)
            prior = -torch.square((offsets - aux_offset) / sigma)
            blended_scores = lm_scores + float(aux_blend_weight) * prior
            next_token_id = int(traj_token_ids[int(torch.argmax(blended_scores).item())].item())
        next_token = torch.tensor([[next_token_id]], device=device, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
            dim=1,
        )

    end_token = torch.tensor([[traj_end_id]], device=device, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, end_token], dim=1)
    return input_ids


def infer_one(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    *,
    device: str,
    max_new_tokens: int,
    constrain_traj_decoding: bool,
    forced_traj_token_count: int | None,
    prompt_mode: str,
    target_mode: str,
    traj_source: str,
    load_info: dict[str, Any],
    aux_blend_sigma: float,
    aux_blend_weight: float,
    aux_bound_scale: float,
    aux_edge_margin: int,
) -> dict[str, Any]:
    prompt_text = build_traj_only_prompt(sample, PROJECT_ROOT) if prompt_mode == "traj_only" else build_user_prompt(sample, PROJECT_ROOT)
    images = load_sample_images(sample, PROJECT_ROOT)
    assistant_prefix = "<|traj_future_start|>" if target_mode == "traj_only" else "<|cot_start|>"
    messages = build_messages(prompt_text, len(images), target_text=None, assistant_prefix=assistant_prefix)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    batch = processor(
        text=[text],
        images=[images],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
    target_traj_token_count = int(
        forced_traj_token_count
        if forced_traj_token_count is not None
        else len((sample.get("hard_target") or {}).get("traj_future_token_ids") or [])
    )

    logits_processor = None
    stopping_criteria = None
    if constrain_traj_decoding and target_traj_token_count > 0:
        if target_mode == "traj_only":
            contract = TrajOnlyDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=target_traj_token_count,
            )
            logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])
        else:
            contract = TrajDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=target_traj_token_count,
            )
            logits_processor = LogitsProcessorList([TrajSpanLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajEndCriteria(contract)])

    started_at = time.time()
    with torch.inference_mode():
        if traj_source == "aux_head" and target_mode == "joint" and target_traj_token_count > 0:
            generated = _manual_generate_with_aux_head(
                model,
                batch,
                tokenizer,
                target_traj_token_count=target_traj_token_count,
                max_new_tokens=max_new_tokens,
                load_info=load_info,
                traj_source=traj_source,
                aux_blend_sigma=aux_blend_sigma,
                aux_blend_weight=aux_blend_weight,
                aux_bound_scale=aux_bound_scale,
                aux_edge_margin=aux_edge_margin,
            )
        elif traj_source == "aux_lm_blend" and target_mode == "joint" and target_traj_token_count > 0:
            generated = _manual_generate_with_aux_head(
                model,
                batch,
                tokenizer,
                target_traj_token_count=target_traj_token_count,
                max_new_tokens=max_new_tokens,
                load_info=load_info,
                traj_source=traj_source,
                aux_blend_sigma=aux_blend_sigma,
                aux_blend_weight=aux_blend_weight,
                aux_bound_scale=aux_bound_scale,
                aux_edge_margin=aux_edge_margin,
            )
        else:
            generated = model.backbone.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
    elapsed = time.time() - started_at
    generated_text = decode_generated_text(tokenizer, batch["input_ids"], generated)
    generated_cot, generated_traj_tokens = extract_generated_spans(generated_text, target_mode=target_mode)
    action = action_record_from_text(generated_cot) or {"value": "unknown", "confidence": 0.0, "method": "missing"}
    gt_action = (sample.get("derived") or {}).get("gt_motion_class")
    teacher_action = (sample.get("teacher_target") or {}).get("teacher_motion_class")
    teacher_long_cot = (sample.get("teacher_target") or {}).get("cot_text")
    return {
        "sample_id": sample["sample_id"],
        "split": sample.get("split"),
        "elapsed_sec": round(elapsed, 3),
        "generated_text": generated_text,
        "generated_cot": generated_cot,
        "generated_traj_token_count": len(generated_traj_tokens),
        "generated_action": action,
        "gt_action": gt_action,
        "teacher_action": teacher_action,
        "teacher_long_cot": teacher_long_cot,
        "target_text": (sample.get("hard_target") or {}).get("cot_text"),
        "target_traj_token_count": len((sample.get("hard_target") or {}).get("traj_future_token_ids") or []),
        "forced_traj_token_count": target_traj_token_count,
        "constrain_traj_decoding": constrain_traj_decoding,
        "traj_source": traj_source,
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.corpus_jsonl)
    samples = pick_samples(rows, split=args.split, limit=args.num_samples)
    model, tokenizer, processor, load_info = load_model_and_processors(args.checkpoint_dir, args.device)
    data_view = load_info.get("data_view") or {}
    prompt_mode = str(data_view.get("prompt_mode", "joint"))
    target_mode = str(data_view.get("target_mode", "joint"))

    results = []
    for sample in samples:
        print(json.dumps({"event": "infer_sample_start", "sample_id": sample["sample_id"]}), flush=True)
        result = infer_one(
            model,
            processor,
            tokenizer,
            sample,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            constrain_traj_decoding=not args.disable_traj_constraint,
            forced_traj_token_count=args.forced_traj_token_count,
            prompt_mode=prompt_mode,
            target_mode=target_mode,
            traj_source=args.traj_source,
            load_info=load_info,
            aux_blend_sigma=args.aux_blend_sigma,
            aux_blend_weight=args.aux_blend_weight,
            aux_bound_scale=args.aux_bound_scale,
            aux_edge_margin=args.aux_edge_margin,
        )
        print(
            json.dumps(
                {
                    "event": "infer_sample_done",
                    "sample_id": sample["sample_id"],
                    "elapsed_sec": result["elapsed_sec"],
                }
            ),
            flush=True,
        )
        results.append(result)
    avg_latency = sum(item["elapsed_sec"] for item in results) / len(results) if results else 0.0
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "num_samples": len(results),
        "max_new_tokens": args.max_new_tokens,
        "constrain_traj_decoding": not args.disable_traj_constraint,
        "forced_traj_token_count": args.forced_traj_token_count,
        "traj_source": args.traj_source,
        "aux_blend_sigma": args.aux_blend_sigma,
        "aux_blend_weight": args.aux_blend_weight,
        "aux_bound_scale": args.aux_bound_scale,
        "aux_edge_margin": args.aux_edge_margin,
        "avg_latency_sec": round(avg_latency, 3),
        "load_info": load_info,
        "results": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
