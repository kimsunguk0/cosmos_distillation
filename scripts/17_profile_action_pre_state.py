#!/usr/bin/env python3
"""Profile VLM latency until the action-expert-ready trajectory boundary.

This deliberately does not generate the 128 discrete future trajectory tokens.
For Alpamayo-style action expert inference, the useful boundary is the generated
``<|traj_future_start|>`` token plus one extra generated token, because the KV
cache is updated after that next token is produced.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPAMAYO15_SRC = WORKSPACE_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"

for path in (WORKSPACE_ROOT, PROJECT_ROOT, ALPAMAYO15_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor  # noqa: E402
from alpamayo1_5.models.token_utils import StopAfterEOS, to_special_token  # noqa: E402
from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    build_model_inputs_batch,
    load_materialized_samples,
    load_model_and_processor,
)
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids  # noqa: E402
from src.training.collator import (  # noqa: E402
    build_messages,
    build_user_prompt,
    load_sample_images,
    resolve_sample_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks.jsonl",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--teacher-model-path", type=Path, default=WORKSPACE_ROOT / "base_weights" / "Alpamayo-1.5-10B")
    parser.add_argument("--teacher-config-json", default=None)
    parser.add_argument("--teacher-runtime-support", default=None)
    parser.add_argument(
        "--student-checkpoint-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "checkpoints"
            / "no_nav_bp3_h200fast_b4"
            / "no_nav_bp3_h200fast_b4_from_step2288_20260504_053208"
            / "final"
        ),
    )
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    parser.add_argument("--decoding-mode", choices=("greedy", "sampling"), default="greedy")
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--student-input-format",
        choices=("student_real", "teacher_placeholder"),
        default="student_real",
        help=(
            "student_real uses the training collator prompt with numeric ego history. "
            "teacher_placeholder uses the Alpamayo prompt with 48 <|traj_history|> placeholders, "
            "matching the teacher input format for speed-only comparison."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "no_nav_distill" / "action_pre_profile_lora_merged_fa2.json",
    )
    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 6)


def to_device(data: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in data.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def iter_selected_records(path: Path, *, split: str | None, start: int, count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            if skipped < start:
                skipped += 1
                continue
            selected.append(record)
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise RuntimeError(f"Only found {len(selected)} records for split={split!r}, start={start}, count={count}")
    return selected


def make_batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[idx : idx + batch_size] for idx in range(0, len(records), batch_size)]


def mean(values: list[float]) -> float | None:
    return round(float(sum(values) / len(values)), 6) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 6)
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 6)


def collect_attn_info(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    return {
        "config_attn_implementation": getattr(config, "_attn_implementation", getattr(config, "attn_implementation", None)),
        "text_config_attn_implementation": getattr(
            text_config,
            "_attn_implementation",
            getattr(text_config, "attn_implementation", None),
        ),
    }


class GenerateForwardProfiler:
    """CUDA-event profiler for the actual ``generate`` call path."""

    def __init__(self, hf_model: torch.nn.Module) -> None:
        self.hf_model = hf_model
        self.original_forward = None
        self.visual_module = None
        self.original_visual_forward = None
        self.forward_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.visual_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.seen_prefill = False

    def __enter__(self):
        if not torch.cuda.is_available():
            return self
        self.original_forward = self.hf_model.forward

        def wrapped_forward(module_self, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args and isinstance(args[0], torch.Tensor):
                input_ids = args[0]
            seq_len = int(input_ids.shape[1]) if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2 else None
            category = "decode" if self.seen_prefill and seq_len is not None and seq_len <= 1 else "prefill"
            if category == "prefill":
                self.seen_prefill = True
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = self.original_forward(*args, **kwargs)
            end_event.record()
            self.forward_events.append((category, start_event, end_event))
            return output

        self.hf_model.forward = types.MethodType(wrapped_forward, self.hf_model)

        candidate = getattr(self.hf_model, "visual", None)
        if candidate is None:
            candidate = getattr(getattr(self.hf_model, "model", None), "visual", None)
        self.visual_module = candidate
        if self.visual_module is not None:
            self.original_visual_forward = self.visual_module.forward

            def wrapped_visual_forward(module_self, *args, **kwargs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = self.original_visual_forward(*args, **kwargs)
                end_event.record()
                self.visual_events.append((start_event, end_event))
                return output

            self.visual_module.forward = types.MethodType(wrapped_visual_forward, self.visual_module)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.original_forward is not None:
            self.hf_model.forward = self.original_forward
        if self.visual_module is not None and self.original_visual_forward is not None:
            self.visual_module.forward = self.original_visual_forward

    def summary(self) -> dict[str, Any]:
        if not torch.cuda.is_available():
            return {
                "prefill_sec": None,
                "decode_sec": None,
                "vit_sec": None,
                "text_only_prefill_sec": None,
                "forward_call_count": 0,
                "decode_forward_call_count": 0,
            }
        sync_cuda()
        prefill_ms = 0.0
        decode_ms = 0.0
        decode_calls = 0
        for category, start_event, end_event in self.forward_events:
            value = float(start_event.elapsed_time(end_event))
            if category == "decode":
                decode_ms += value
                decode_calls += 1
            else:
                prefill_ms += value
        vit_ms = sum(float(start_event.elapsed_time(end_event)) for start_event, end_event in self.visual_events)
        prefill_sec = round(prefill_ms / 1000.0, 6)
        decode_sec = round(decode_ms / 1000.0, 6)
        vit_sec = round(vit_ms / 1000.0, 6)
        return {
            "prefill_sec": prefill_sec,
            "decode_sec": decode_sec,
            "vit_sec": vit_sec,
            "text_only_prefill_sec": round(max(prefill_sec - vit_sec, 0.0), 6),
            "forward_call_count": len(self.forward_events),
            "decode_forward_call_count": decode_calls,
            "visual_forward_call_count": len(self.visual_events),
        }


def model_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    sync_cuda()
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def build_generation_config(hf_model: torch.nn.Module, tokenizer: Any, args: argparse.Namespace):
    generation_config = copy.deepcopy(hf_model.generation_config)
    generation_config.max_new_tokens = int(args.max_new_tokens)
    generation_config.num_return_sequences = 1
    generation_config.num_beams = 1
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = False
    generation_config.output_scores = False
    generation_config.output_hidden_states = False
    generation_config.pad_token_id = tokenizer.pad_token_id
    if args.decoding_mode == "greedy":
        generation_config.do_sample = False
        generation_config.top_p = 1.0
        generation_config.top_k = None
        generation_config.temperature = 1.0
    else:
        generation_config.do_sample = True
        generation_config.top_p = float(args.top_p)
        generation_config.top_k = None if int(args.top_k) <= 0 else int(args.top_k)
        generation_config.temperature = float(args.temperature)
    return generation_config


def prefill_once(hf_model: torch.nn.Module, tokenized: dict[str, Any], *, with_images: bool = True) -> float | None:
    forward_inputs = dict(tokenized)
    if not with_images:
        for key in list(forward_inputs):
            if "pixel" in key or "image" in key or "grid" in key:
                forward_inputs.pop(key, None)
    forward_inputs["use_cache"] = True
    forward_inputs["return_dict"] = True
    sync_cuda()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            _ = hf_model(**forward_inputs)
    except Exception as exc:
        print(json.dumps({"event": "prefill_failed", "with_images": with_images, "error": str(exc)[:300]}), flush=True)
        return None
    sync_cuda()
    return elapsed(started)


def prefill_with_output(hf_model: torch.nn.Module, tokenized: dict[str, Any]) -> tuple[float | None, Any | None]:
    forward_inputs = dict(tokenized)
    forward_inputs["use_cache"] = True
    forward_inputs["return_dict"] = True
    sync_cuda()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = hf_model(**forward_inputs)
    except Exception as exc:
        print(json.dumps({"event": "prefill_failed", "with_images": True, "error": str(exc)[:300]}), flush=True)
        return None, None
    sync_cuda()
    return elapsed(started), output


def select_next_token(scores: torch.Tensor, generation_config: Any) -> torch.Tensor:
    if bool(getattr(generation_config, "do_sample", False)):
        temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)
        probs = torch.softmax(scores / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return scores.argmax(dim=-1, keepdim=True)


def manual_decode_until_action_pre(
    hf_model: torch.nn.Module,
    tokenized: dict[str, Any],
    prefill_output: Any,
    *,
    traj_start_token_id: int,
    generation_config: Any,
    logits_processor: LogitsProcessorList | None,
) -> tuple[float | None, dict[str, Any]]:
    if prefill_output is None:
        return None, {"manual_decode_status": "missing_prefill"}
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, device=input_ids.device, dtype=torch.long)
    past_key_values = prefill_output.past_key_values
    next_scores = prefill_output.logits[:, -1, :]
    generated: list[torch.Tensor] = []
    seen_start = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    action_pre_ready = torch.zeros_like(seen_start)

    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(generation_config.max_new_tokens)):
            processor_input = torch.cat([input_ids, *generated], dim=1) if generated else input_ids
            scores = next_scores
            if logits_processor is not None:
                scores = logits_processor(processor_input, scores)
            next_token = select_next_token(scores, generation_config)
            generated.append(next_token)
            was_seen_start = seen_start.clone()
            current_is_start = next_token.squeeze(-1) == int(traj_start_token_id)
            action_pre_ready = action_pre_ready | was_seen_start
            seen_start = seen_start | current_is_start
            if bool(action_pre_ready.all().item()):
                break
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
            next_scores = step_out.logits[:, -1, :]
    sync_cuda()
    decode_sec = elapsed(started)
    if generated:
        generated_ids = torch.cat(generated, dim=1)
        fake_output = type("ManualDecodeOutput", (), {"sequences": torch.cat([input_ids, generated_ids], dim=1)})
        summary = analyze_generated_sequences(fake_output, input_ids, traj_start_token_id)
    else:
        summary = {
            "prompt_padded_tokens": int(input_ids.shape[1]),
            "generated_tokens_batch": 0,
            "traj_start_hit_rate": 0.0,
            "traj_start_offsets": [],
            "action_pre_generated_tokens": [],
            "traj_start_offset_mean": None,
            "action_pre_generated_tokens_mean": None,
        }
    summary["manual_decode_status"] = "ok"
    return decode_sec, summary


def generate_to_action_pre(
    hf_model: torch.nn.Module,
    tokenized: dict[str, Any],
    *,
    tokenizer: Any,
    traj_start_token_id: int,
    generation_config: Any,
    logits_processor: LogitsProcessorList | None,
) -> tuple[float, Any]:
    generate_kwargs = {key: value for key, value in tokenized.items() if key != "input_ids"}
    stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=int(traj_start_token_id))])
    sync_cuda()
    started = time.perf_counter()
    with torch.inference_mode():
        output = hf_model.generate(
            input_ids=tokenized["input_ids"],
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **generate_kwargs,
        )
    sync_cuda()
    return elapsed(started), output


def analyze_generated_sequences(output: Any, input_ids: torch.Tensor, traj_start_token_id: int) -> dict[str, Any]:
    sequences = output.sequences.detach()
    prompt_len = int(input_ids.shape[1])
    generated = sequences[:, prompt_len:]
    starts = generated == int(traj_start_token_id)
    offsets: list[int | None] = []
    action_pre_lengths: list[int | None] = []
    for row in starts:
        positions = torch.nonzero(row, as_tuple=False).flatten()
        if positions.numel() == 0:
            offsets.append(None)
            action_pre_lengths.append(None)
            continue
        pos = int(positions[0].item())
        offsets.append(pos)
        action_pre_lengths.append(min(pos + 2, int(generated.shape[1])))
    hit_count = sum(offset is not None for offset in offsets)
    finite_offsets = [float(offset) for offset in offsets if offset is not None]
    finite_lengths = [float(value) for value in action_pre_lengths if value is not None]
    return {
        "prompt_padded_tokens": prompt_len,
        "generated_tokens_batch": int(generated.shape[1]),
        "traj_start_hit_rate": round(hit_count / max(len(offsets), 1), 6),
        "traj_start_offsets": offsets,
        "action_pre_generated_tokens": action_pre_lengths,
        "traj_start_offset_mean": mean(finite_offsets),
        "action_pre_generated_tokens_mean": mean(finite_lengths),
    }


def summarize_runs(runs: list[dict[str, Any]], *, batch_size: int) -> dict[str, Any]:
    numeric_keys = (
        "vit_sec",
        "text_only_prefill_sec",
        "prefill_sec",
        "decode_sec",
        "generate_total_sec",
        "generate_overhead_sec",
        "total_to_action_pre_sec",
    )
    mean_values: dict[str, Any] = {}
    p50_values: dict[str, Any] = {}
    p95_values: dict[str, Any] = {}
    per_sample_mean: dict[str, Any] = {}
    for key in numeric_keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        mean_values[key] = mean(values)
        p50_values[key] = percentile(values, 0.50)
        p95_values[key] = percentile(values, 0.95)
        per_sample_mean[key] = round(float(mean_values[key]) / batch_size, 6) if mean_values[key] is not None else None
    generated_tokens = [float(run["generated_tokens_batch"]) for run in runs if run.get("generated_tokens_batch") is not None]
    hit_rates = [float(run["traj_start_hit_rate"]) for run in runs if run.get("traj_start_hit_rate") is not None]
    return {
        "runs": runs,
        "mean": mean_values,
        "p50": p50_values,
        "p95": p95_values,
        "per_sample_mean": per_sample_mean,
        "generated_tokens_batch_mean": mean(generated_tokens),
        "traj_start_hit_rate_mean": mean(hit_rates),
    }


def profile_batches(
    *,
    model_name: str,
    hf_model: torch.nn.Module,
    tokenizer: Any,
    batches: list[list[dict[str, Any]]],
    prepare_batch,
    args: argparse.Namespace,
    traj_start_token_id: int,
    logits_processor: LogitsProcessorList | None = None,
) -> dict[str, Any]:
    generation_config = build_generation_config(hf_model, tokenizer, args)
    runs: list[dict[str, Any]] = []
    total_iterations = max(args.warmup_runs, 0) + max(args.repeats, 1)
    for iteration in range(total_iterations):
        is_warmup = iteration < max(args.warmup_runs, 0)
        for batch_idx, batch_records in enumerate(batches):
            sample_ids = [str(record.get("sample_id")) for record in batch_records]
            print(
                json.dumps(
                    {
                        "event": "profile_batch_start",
                        "model": model_name,
                        "warmup": is_warmup,
                        "iteration": iteration,
                        "batch_idx": batch_idx,
                        "batch_size": len(batch_records),
                    }
                ),
                flush=True,
            )
            tokenized = prepare_batch(batch_records)
            with GenerateForwardProfiler(hf_model) as forward_profiler:
                generate_total_sec, output = generate_to_action_pre(
                    hf_model,
                    tokenized,
                    tokenizer=tokenizer,
                    traj_start_token_id=traj_start_token_id,
                    generation_config=generation_config,
                    logits_processor=logits_processor,
                )
            forward_summary = forward_profiler.summary()
            prefill_sec = forward_summary["prefill_sec"]
            decode_sec = forward_summary["decode_sec"]
            text_only_prefill_sec = forward_summary["text_only_prefill_sec"]
            vit_sec = forward_summary["vit_sec"]
            generated_summary = analyze_generated_sequences(output, tokenized["input_ids"], traj_start_token_id)
            generate_overhead_sec = (
                round(generate_total_sec - prefill_sec - decode_sec, 6)
                if prefill_sec is not None and decode_sec is not None
                else None
            )
            run = {
                "model": model_name,
                "warmup": is_warmup,
                "iteration": iteration,
                "batch_idx": batch_idx,
                "batch_size": len(batch_records),
                "sample_ids": sample_ids,
                "vit_sec": vit_sec,
                "text_only_prefill_sec": text_only_prefill_sec,
                "prefill_sec": prefill_sec,
                "decode_sec": decode_sec,
                "generate_total_sec": generate_total_sec,
                "generate_overhead_sec": generate_overhead_sec,
                "total_to_action_pre_sec": generate_total_sec,
                **forward_summary,
                **generated_summary,
            }
            print(json.dumps({"event": "profile_batch_done", **{k: v for k, v in run.items() if k != "sample_ids"}}), flush=True)
            if not is_warmup:
                runs.append(run)
            del tokenized
            del output
    return summarize_runs(runs, batch_size=args.batch_size)


def load_teacher(args: argparse.Namespace):
    dtype = torch_dtype_from_name(args.dtype)
    model, processor, config, config_path, runtime_support_path = load_model_and_processor(
        checkpoint_path=args.teacher_model_path,
        dtype=dtype,
        device=args.device,
        config_json=args.teacher_config_json,
        runtime_support=args.teacher_runtime_support,
        attn_implementation=args.attn_implementation,
        min_pixels=None,
        max_pixels=None,
    )
    return model, processor, config, config_path, runtime_support_path


def load_student(args: argparse.Namespace):
    train_config = json.loads((args.student_checkpoint_dir / "train_config.json").read_text(encoding="utf-8"))
    base_model = str(train_config["args"]["student_model"])
    use_lora = not bool(train_config["args"].get("disable_lora", False))
    tokenizer = AutoTokenizer.from_pretrained(args.student_checkpoint_dir / "tokenizer", local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.student_checkpoint_dir / "processor", local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = "left"
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int(train_config["trainer_config"].get("max_length", 4096)),
        torch_dtype=torch_dtype_from_name(args.dtype),
        local_files_only=Path(base_model).expanduser().exists(),
        attn_implementation=args.attn_implementation,
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
    merged_lora = False
    if args.merge_lora and hasattr(model.backbone, "merge_and_unload"):
        model.backbone = model.backbone.merge_and_unload()
        merged_lora = True
    model = model.to(args.device).eval()
    return model, processor, tokenizer, train_config, merged_lora


def make_teacher_prepare(model, processor, args: argparse.Namespace):
    def prepare(batch_records: list[dict[str, Any]]) -> dict[str, Any]:
        sample_dirs = [resolve_sample_path(record, PROJECT_ROOT) for record in batch_records]
        materialized = load_materialized_samples(sample_dirs, args.io_workers)
        model_inputs = build_model_inputs_batch(processor=processor, samples=materialized, device=args.device)
        tokenized = dict(model_inputs["tokenized_data"])
        input_ids = tokenized.pop("input_ids")
        tokenized["input_ids"] = model.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": model_inputs["ego_history_xyz"],
                "ego_history_rot": model_inputs["ego_history_rot"],
            },
        )
        return tokenized

    return prepare


def make_student_prepare(processor, train_config: dict[str, Any], args: argparse.Namespace):
    max_length = int(train_config["trainer_config"].get("max_length", 4096))

    def prepare(batch_records: list[dict[str, Any]]) -> dict[str, Any]:
        if args.student_input_format == "teacher_placeholder":
            sample_dirs = [resolve_sample_path(record, PROJECT_ROOT) for record in batch_records]
            materialized = load_materialized_samples(sample_dirs, args.io_workers)
            model_inputs = build_model_inputs_batch(processor=processor, samples=materialized, device=args.device)
            return dict(model_inputs["tokenized_data"])

        texts: list[str] = []
        image_batch: list[list[Any]] = []
        for record in batch_records:
            images = load_sample_images(record, PROJECT_ROOT)
            prompt_text = build_user_prompt(record, PROJECT_ROOT)
            messages = build_messages(prompt_text, len(images), target_text=None)
            texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            )
            image_batch.append(images)
        tokenized = processor(
            text=texts,
            images=image_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return to_device(tokenized, args.device)

    return prepare


def ratio_summary(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    teacher_mean = teacher["mean"]
    student_mean = student["mean"]
    for key in (
        "vit_sec",
        "text_only_prefill_sec",
        "prefill_sec",
        "decode_sec",
        "total_to_action_pre_sec",
    ):
        t_value = teacher_mean.get(key)
        s_value = student_mean.get(key)
        if t_value in (None, 0) or s_value is None:
            result[key] = None
            continue
        ratio = float(t_value) / float(s_value)
        result[key] = {
            "teacher_sec": t_value,
            "student_sec": s_value,
            "teacher_over_student_speedup": round(ratio, 4),
            "student_time_pct_of_teacher": round(float(s_value) / float(t_value) * 100.0, 2),
            "student_faster_pct": round((1.0 - float(s_value) / float(t_value)) * 100.0, 2),
        }
    return result


def main() -> None:
    args = parse_args()
    total_samples = int(args.batch_size) * int(args.num_batches)
    records = iter_selected_records(
        args.corpus_jsonl,
        split=args.split,
        start=args.sample_start,
        count=total_samples,
    )
    batches = make_batches(records, args.batch_size)
    sample_ids = [str(record.get("sample_id")) for record in records]
    print(
        json.dumps(
            {
                "event": "profile_start",
                "samples": len(records),
                "batch_size": args.batch_size,
                "num_batches": len(batches),
                "split": args.split,
                "attn_implementation": args.attn_implementation,
                "merge_lora": args.merge_lora,
                "student_input_format": args.student_input_format,
                "stop_boundary": "traj_future_start_plus_one_token_for_kv",
            }
        ),
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    teacher_load_start = time.perf_counter()
    teacher_model, teacher_processor, teacher_config, teacher_config_path, runtime_support_path = load_teacher(args)
    sync_cuda()
    teacher_load_sec = elapsed(teacher_load_start)
    teacher_hf_model = teacher_model.vlm
    teacher_traj_start_id = teacher_model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    teacher_logits_processor = LogitsProcessorList(
        [
            ExpertLogitsProcessor(
                traj_token_offset=teacher_config.traj_token_start_idx,
                traj_vocab_size=teacher_config.traj_vocab_size,
            )
        ]
    )
    teacher_profile = profile_batches(
        model_name="teacher_alpamayo15_vlm_action_pre",
        hf_model=teacher_hf_model,
        tokenizer=teacher_model.tokenizer,
        batches=batches,
        prepare_batch=make_teacher_prepare(teacher_model, teacher_processor, args),
        args=args,
        traj_start_token_id=teacher_traj_start_id,
        logits_processor=teacher_logits_processor,
    )
    teacher_memory = model_memory_gb()
    teacher_attn = collect_attn_info(teacher_hf_model)
    del teacher_model, teacher_processor, teacher_hf_model
    sync_cuda()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    student_load_start = time.perf_counter()
    student_model, student_processor, student_tokenizer, train_config, merged_lora = load_student(args)
    sync_cuda()
    student_load_sec = elapsed(student_load_start)
    student_hf_model = student_model.backbone
    student_traj_start_id = student_tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    student_profile = profile_batches(
        model_name="student_cosmos_reason2b_lora_merged_action_pre",
        hf_model=student_hf_model,
        tokenizer=student_tokenizer,
        batches=batches,
        prepare_batch=make_student_prepare(student_processor, train_config, args),
        args=args,
        traj_start_token_id=student_traj_start_id,
        logits_processor=None,
    )
    student_memory = model_memory_gb()
    student_attn = collect_attn_info(student_hf_model)

    summary = {
        "profile_schema_version": "action_pre_v1",
        "corpus_jsonl": str(args.corpus_jsonl),
        "sample_ids": sample_ids,
        "split": args.split,
        "sample_start": args.sample_start,
        "batch_size": args.batch_size,
        "num_batches": len(batches),
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "decoding_mode": args.decoding_mode,
        "attn_implementation_requested": args.attn_implementation,
        "student_input_format": args.student_input_format,
        "stop_boundary": "traj_future_start_plus_one_token_for_kv",
        "teacher": {
            "model_name": "Alpamayo-1.5-10B VLM",
            "model_path": str(args.teacher_model_path),
            "config_path": str(teacher_config_path),
            "runtime_support_path": str(runtime_support_path) if runtime_support_path is not None else None,
            "traj_start_token_id": int(teacher_traj_start_id),
            "load_sec": teacher_load_sec,
            "peak_allocated_gb": teacher_memory,
            "attn": teacher_attn,
            **teacher_profile,
        },
        "student": {
            "model_name": "Cosmos-Reason2-2B student",
            "checkpoint_dir": str(args.student_checkpoint_dir),
            "merged_lora": merged_lora,
            "traj_start_token_id": int(student_traj_start_id),
            "load_sec": student_load_sec,
            "peak_allocated_gb": student_memory,
            "attn": student_attn,
            **student_profile,
        },
    }
    summary["comparison_teacher_over_student"] = ratio_summary(summary["teacher"], summary["student"])
    if torch.cuda.is_available():
        summary["gpu_name"] = torch.cuda.get_device_name()
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
