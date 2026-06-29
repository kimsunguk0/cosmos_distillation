#!/usr/bin/env python3
"""Debug no-FLEX vs FLEX residual generation parity on a small sample set."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.checkpoint_eval import _manual_flex_generate, _model_logits_and_past, _past_seq_len  # noqa: E402
from src.inference.decoding import (  # noqa: E402
    StopOnTrajOnlyEndCriteria,
    TrajOnlyDecodingContract,
    TrajOnlyLogitsProcessor,
)
from src.training.collator import (  # noqa: E402
    build_messages,
    build_user_prompt,
    fuse_history_tokens_in_input_ids,
    load_ego_history_xyz,
    load_sample_images,
    load_traj_future_token_ids,
    resolve_camera_indices,
    resolve_image_relative_timestamps,
)
from src.utils.runtime_paths import resolve_student_model_path  # noqa: E402


def _load_decode_module():
    path = PROJECT_ROOT / "scripts" / "25_decode_checkpoint_overlays.py"
    spec = importlib.util.spec_from_file_location("decode_checkpoint_overlays_25", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import decode helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode_mod = _load_decode_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--b0-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--flex-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _move_batch(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                out[key] = value.to(device=device, dtype=dtype)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def _prepare_one(
    sample: dict[str, Any],
    *,
    processor,
    tokenizer,
) -> tuple[dict[str, Any], int]:
    history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
    target_tokens = load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
    prompt_text = build_user_prompt(
        sample,
        PROJECT_ROOT,
        ego_history_xyz=history_xyz,
        prompt_text_style="official_alpamayo",
    )
    images = load_sample_images(sample, PROJECT_ROOT)
    camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
    frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
    messages = build_messages(
        prompt_text,
        len(images),
        assistant_prefix="<|traj_future_start|>",
        image_prompt_style="camera_labeled",
        camera_indices=camera_indices,
        num_frames_per_camera=frames_per_camera,
    )
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    batch = processor(text=[text], images=[images], return_tensors="pt", padding=True, truncation=True)
    batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, [history_xyz])

    relative_timestamps = resolve_image_relative_timestamps(
        sample,
        PROJECT_ROOT,
        camera_count=len(camera_indices),
        frames_per_camera=frames_per_camera,
    )
    camera_indices_tensor = torch.tensor([camera_indices], dtype=torch.long)
    relative_timestamps_tensor = torch.zeros((1, len(camera_indices), frames_per_camera), dtype=torch.float32)
    for camera_offset, row_times in enumerate(relative_timestamps[: len(camera_indices)]):
        count = min(len(row_times), frames_per_camera)
        if count > 0:
            relative_timestamps_tensor[0, camera_offset, :count] = torch.tensor(row_times[:count], dtype=torch.float32)
    batch["camera_indices"] = camera_indices_tensor
    batch["relative_timestamps"] = relative_timestamps_tensor
    batch["camera_counts"] = torch.tensor([len(camera_indices)], dtype=torch.long)
    batch["frames_per_camera"] = torch.tensor([frames_per_camera], dtype=torch.long)
    batch["flex_residual_image_slots"] = True
    batch["flex_residual_scale"] = 0.0
    return batch, len(target_tokens)


def _top_row(scores: torch.Tensor, topk: int = 5) -> dict[str, Any]:
    probs = torch.softmax(scores.float(), dim=-1)
    values, indices = torch.topk(probs, k=topk, dim=-1)
    return {
        "top_ids": [int(x) for x in indices[0].detach().cpu().tolist()],
        "top_probs": [float(x) for x in values[0].detach().cpu().tolist()],
    }


def _kl(a: torch.Tensor, b: torch.Tensor) -> float:
    b = b.to(device=a.device)
    a_log = F.log_softmax(a.float(), dim=-1)
    b_log = F.log_softmax(b.float(), dim=-1)
    return float((a_log.exp() * (a_log - b_log)).sum(dim=-1).mean().detach().cpu())


def _token_list(generated: torch.Tensor, prompt_len: int, max_tokens: int = 128) -> list[int]:
    return [int(x) for x in generated[0, prompt_len : prompt_len + max_tokens].detach().cpu().tolist()]


def _first_mismatch(a: list[int], b: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(a, b)):
        if int(left) != int(right):
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _prefill_scores(model, batch: dict[str, Any], *, use_wrapper: bool, processor_scores: LogitsProcessorList):
    if use_wrapper:
        kwargs = {
            key: batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "camera_indices",
                "relative_timestamps",
                "camera_counts",
                "frames_per_camera",
                "flex_residual_image_slots",
                "flex_residual_scale",
            )
            if key in batch
        }
        kwargs.update(
            {
                "return_hidden_states": False,
                "compute_meta_action": False,
                "compute_traj_aux": False,
                "use_cache": True,
            }
        )
        out = model(**kwargs)
    else:
        kwargs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            if key in batch
        }
        out = model.backbone(**kwargs, use_cache=True, return_dict=True)
    logits, past = _model_logits_and_past(out)
    scores = processor_scores(batch["input_ids"], logits[:, -1, :])
    return scores, past, logits


def _forced_second_scores(
    model,
    batch: dict[str, Any],
    past_key_values,
    next_token: torch.Tensor,
    *,
    use_wrapper: bool,
    processor_scores: LogitsProcessorList,
) -> torch.Tensor:
    attention_mask = torch.cat([batch["attention_mask"], torch.ones_like(next_token)], dim=1)
    past_len = _past_seq_len(past_key_values)
    if past_len is None:
        past_len = int(batch["input_ids"].shape[1])
    kwargs: dict[str, Any] = {
        "input_ids": next_token,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    kwargs["cache_position"] = torch.arange(
        past_len,
        past_len + int(next_token.shape[1]),
        device=next_token.device,
        dtype=torch.long,
    )
    if use_wrapper:
        kwargs.update(
            {
                "return_hidden_states": False,
                "compute_meta_action": False,
                "compute_traj_aux": False,
            }
        )
        out = model(**kwargs)
    else:
        prepare = getattr(model.backbone, "prepare_inputs_for_generation", None)
        if callable(prepare):
            full_input_ids = torch.cat([batch["input_ids"], next_token], dim=1)
            prepared = prepare(
                full_input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                cache_position=kwargs.get("cache_position"),
            )
            prepared = dict(prepared)
            prepared["return_dict"] = True
            out = model.backbone(**prepared)
        else:
            out = model.backbone(**kwargs, return_dict=True)
    logits, _ = _model_logits_and_past(out)
    generated = torch.cat([batch["input_ids"], next_token], dim=1)
    return processor_scores(generated, logits[:, -1, :])


def main() -> None:
    args = parse_args()
    rows = [row for row in _load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    rows = rows[: max(int(args.num_samples), 1)]
    if not rows:
        raise SystemExit(f"No rows selected from {args.corpus_jsonl} split={args.split!r}")

    b0_args = argparse.Namespace(checkpoint_dir=args.b0_checkpoint_dir, student_model=args.student_model, device=args.device)
    flex_args = argparse.Namespace(checkpoint_dir=args.flex_checkpoint_dir, student_model=args.student_model, device=args.device)
    b0_model, tokenizer, processor, _device, _base_model = decode_mod._load_model_and_processors(b0_args)
    device = _device
    dtype = decode_mod._infer_visual_float_dtype(b0_model)
    prepared_rows: list[dict[str, Any]] = []
    for sample in rows:
        batch, target_count = _prepare_one(sample, processor=processor, tokenizer=tokenizer)
        moved = _move_batch(batch, device=device, dtype=dtype)
        prompt_len = int(moved["input_ids"].shape[1])
        contract = TrajOnlyDecodingContract.from_tokenizer(
            tokenizer,
            prompt_lengths=[prompt_len],
            traj_token_count=int(target_count),
        )
        logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
        stopping = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])

        with torch.inference_mode():
            b0_scores, b0_past, _ = _prefill_scores(
                b0_model,
                moved,
                use_wrapper=False,
                processor_scores=logits_processor,
            )
            f0_scores, f0_past, _ = _prefill_scores(
                b0_model,
                moved,
                use_wrapper=False,
                processor_scores=logits_processor,
            )
            forced_next = b0_scores.argmax(dim=-1, keepdim=True)
            b0_second = _forced_second_scores(
                b0_model,
                moved,
                b0_past,
                forced_next,
                use_wrapper=False,
                processor_scores=logits_processor,
            )
            b0_gen = b0_model.backbone.generate(
                **{key: moved[key] for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")},
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                num_return_sequences=1,
                use_cache=True,
                logits_processor=logits_processor,
                stopping_criteria=stopping,
            )

        b0_tokens = _token_list(b0_gen, prompt_len)
        prepared_rows.append(
            {
                "sample": sample,
                "batch": batch,
                "prompt_len": prompt_len,
                "target_count": int(target_count),
                "b0_scores": b0_scores.detach().cpu(),
                "b0_second": b0_second.detach().cpu(),
                "forced_next": forced_next.detach().cpu(),
                "b0_tokens": b0_tokens,
                "b0_top": _top_row(b0_scores.detach().cpu()),
                "b0_second_top": _top_row(b0_second.detach().cpu()),
            }
        )

    del b0_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    flex_model, _tokenizer2, _processor2, device, _base_model2 = decode_mod._load_model_and_processors(flex_args)
    if not (hasattr(flex_model, "flex_enabled") and flex_model.flex_enabled()):
        raise SystemExit("flex checkpoint does not have FLEX enabled")
    dtype = decode_mod._infer_visual_float_dtype(flex_model)

    records: list[dict[str, Any]] = []
    for prepared in prepared_rows:
        sample = prepared["sample"]
        batch = _move_batch(prepared["batch"], device=device, dtype=dtype)
        prompt_len = int(prepared["prompt_len"])
        target_count = int(prepared["target_count"])
        contract = TrajOnlyDecodingContract.from_tokenizer(
            tokenizer,
            prompt_lengths=[prompt_len],
            traj_token_count=target_count,
        )
        logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
        stopping = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])

        with torch.inference_mode():
            f0_scores, f0_past, _ = _prefill_scores(
                flex_model,
                batch,
                use_wrapper=False,
                processor_scores=logits_processor,
            )
            flex_scores, flex_past, _ = _prefill_scores(
                flex_model,
                batch,
                use_wrapper=True,
                processor_scores=logits_processor,
            )
            forced_next = prepared["forced_next"].to(device=device)
            f0_second = _forced_second_scores(
                flex_model,
                batch,
                f0_past,
                forced_next,
                use_wrapper=False,
                processor_scores=logits_processor,
            )
            flex_second = _forced_second_scores(
                flex_model,
                batch,
                flex_past,
                forced_next,
                use_wrapper=True,
                processor_scores=logits_processor,
            )
            f0_gen = flex_model.backbone.generate(
                **{key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")},
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                num_return_sequences=1,
                use_cache=True,
                logits_processor=logits_processor,
                stopping_criteria=stopping,
            )
            flex_gen = _manual_flex_generate(
                flex_model,
                batch,
                max_new_tokens=int(args.max_new_tokens),
                logits_processor=logits_processor,
                stopping_criteria=stopping,
            )

        b0_tokens = list(prepared["b0_tokens"])
        f0_tokens = _token_list(f0_gen, prompt_len)
        flex_tokens = _token_list(flex_gen, prompt_len)
        b0_scores_cpu = prepared["b0_scores"]
        b0_second_cpu = prepared["b0_second"]
        f0_scores_cpu = f0_scores.detach().cpu()
        flex_scores_cpu = flex_scores.detach().cpu()
        f0_second_cpu = f0_second.detach().cpu()
        flex_second_cpu = flex_second.detach().cpu()
        records.append(
            {
                "sample_id": str(sample.get("sample_id")),
                "prompt_len": prompt_len,
                "target_token_count": int(target_count),
                "prefill_b0_top": prepared["b0_top"],
                "prefill_f0_backbone_top": _top_row(f0_scores_cpu),
                "prefill_flex_residual0_top": _top_row(flex_scores_cpu),
                "prefill_kl_b0_to_f0_backbone": _kl(b0_scores_cpu, f0_scores_cpu),
                "prefill_kl_b0_to_flex_residual0": _kl(b0_scores_cpu, flex_scores_cpu),
                "second_kl_b0_to_f0_backbone": _kl(b0_second_cpu, f0_second_cpu),
                "second_kl_b0_to_flex_residual0": _kl(b0_second_cpu, flex_second_cpu),
                "second_b0_top": prepared["b0_second_top"],
                "second_f0_backbone_top": _top_row(f0_second_cpu),
                "second_flex_residual0_top": _top_row(flex_second_cpu),
                "b0_vs_f0_first_mismatch": _first_mismatch(b0_tokens, f0_tokens),
                "b0_vs_flex_first_mismatch": _first_mismatch(b0_tokens, flex_tokens),
                "b0_first16": b0_tokens[:16],
                "f0_backbone_first16": f0_tokens[:16],
                "flex_residual0_first16": flex_tokens[:16],
                "b0_length": len(b0_tokens),
                "f0_backbone_length": len(f0_tokens),
                "flex_residual0_length": len(flex_tokens),
            }
        )

    summary = {
        "b0_checkpoint_dir": str(args.b0_checkpoint_dir),
        "flex_checkpoint_dir": str(args.flex_checkpoint_dir),
        "num_samples": len(records),
        "records": records,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "debug_done", "summary_json": str(args.summary_json), "num_samples": len(records)}))


if __name__ == "__main__":
    main()
