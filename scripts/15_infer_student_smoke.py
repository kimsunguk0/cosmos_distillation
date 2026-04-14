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
from transformers import AutoProcessor, AutoTokenizer

from src.data.parsers import action_record_from_text
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import StudentWrapperConfig, build_student_model
from src.training.collator import build_messages, build_user_prompt, load_sample_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "strict_human_long_cot_262_fixkd.jsonl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "train_stage_b_teacher262_fixkd_full" / "final",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "student_infer_smoke_teacher262_fixkd.json",
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
    if "<|cot_start|>" in text:
        text = text.split("<|cot_start|>", 1)[1]
    if "<|cot_end|>" in text:
        text = text.split("<|cot_end|>", 1)[0]
    text = text.replace("<|im_start|>assistant", "\n").replace("<|im_end|>", "\n")
    text = re.sub(r"^\s*assistant[:;]?\s*", "", text, flags=re.IGNORECASE)
    lines = [line.strip(" ;") for line in text.splitlines() if line.strip(" ;")]
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n".join(deduped).strip()


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
        local_files_only=Path(base_model).expanduser().exists(),
    )
    print(json.dumps({"event": "load_base_model_start", "base_model": base_model}), flush=True)
    started_at = time.time()
    model = build_student_model(wrapper_cfg, tokenizer)
    print(json.dumps({"event": "load_base_model_done", "elapsed_sec": round(time.time() - started_at, 3)}), flush=True)
    model.backbone = maybe_apply_lora(model.backbone, LoraConfigSpec(), enabled=use_lora)
    print(json.dumps({"event": "load_state_dict_start"}), flush=True)
    started_at = time.time()
    load_kwargs = {"map_location": "cpu"}
    try:
        state_dict = torch.load(checkpoint_dir / "student_state.pt", mmap=True, **load_kwargs)
    except TypeError:
        state_dict = torch.load(checkpoint_dir / "student_state.pt", **load_kwargs)
    print(json.dumps({"event": "load_state_dict_done", "elapsed_sec": round(time.time() - started_at, 3)}), flush=True)
    try:
        load_result = model.load_state_dict(state_dict, strict=False, assign=True)
    except TypeError:
        load_result = model.load_state_dict(state_dict, strict=False)
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys
    print(json.dumps({"event": "apply_state_dict_done", "missing_count": len(missing), "unexpected_count": len(unexpected)}), flush=True)
    del state_dict
    model = model.to(device).eval()
    print(json.dumps({"event": "model_to_device_done", "device": device}), flush=True)
    return model, tokenizer, processor, {"missing": missing, "unexpected": unexpected, "use_lora": use_lora}


def infer_one(
    model,
    processor,
    tokenizer,
    sample: dict[str, Any],
    *,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_text = build_user_prompt(sample, PROJECT_ROOT)
    images = load_sample_images(sample, PROJECT_ROOT)
    messages = build_messages(prompt_text, len(images), target_text=None)
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

    started_at = time.time()
    with torch.inference_mode():
        generated = model.backbone.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    elapsed = time.time() - started_at
    generated_text = decode_generated_text(tokenizer, batch["input_ids"], generated)
    action = action_record_from_text(generated_text) or {"value": "unknown", "confidence": 0.0, "method": "missing"}
    gt_action = (((sample.get("derived") or {}).get("action_class_from_gt_path") or {}).get("value"))
    teacher_action = (sample.get("soft_target") or {}).get("teacher_action_class")
    teacher_long_cot = (sample.get("soft_target") or {}).get("teacher_long_cot")
    return {
        "sample_id": sample["source_sample_id"],
        "split": sample.get("split"),
        "elapsed_sec": round(elapsed, 3),
        "generated_text": generated_text,
        "generated_action": action,
        "gt_action": gt_action,
        "teacher_action": teacher_action,
        "teacher_long_cot": teacher_long_cot,
        "target_text": sample["target"]["text"],
    }


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.corpus_jsonl)
    samples = pick_samples(rows, split=args.split, limit=args.num_samples)
    model, tokenizer, processor, load_info = load_model_and_processors(args.checkpoint_dir, args.device)

    results = []
    for sample in samples:
        print(json.dumps({"event": "infer_sample_start", "sample_id": sample["source_sample_id"]}), flush=True)
        result = infer_one(
            model,
            processor,
            tokenizer,
            sample,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        print(
            json.dumps(
                {
                    "event": "infer_sample_done",
                    "sample_id": sample["source_sample_id"],
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
        "avg_latency_sec": round(avg_latency, 3),
        "load_info": load_info,
        "results": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
