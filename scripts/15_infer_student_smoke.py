#!/usr/bin/env python3
"""Run student multimodal inference smoke on a few corpus samples."""

from __future__ import annotations

import argparse
import json
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
    base_model = str(train_config["args"]["student_model"])
    use_lora = not bool(train_config["args"].get("disable_lora", False))

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
    data_view = train_config.get("data_view") or {}
    return model, tokenizer, processor, {**load_info, "use_lora": use_lora, "data_view": data_view}


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
        "avg_latency_sec": round(avg_latency, 3),
        "load_info": load_info,
        "results": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
