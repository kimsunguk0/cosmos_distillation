#!/usr/bin/env python3
"""Build token-level DAgger rows by relabeling under student-generated prefixes.

The generated corpus is intentionally hard-label only at first:

1. The current student free-runs from the official Alpamayo prompt.
2. Its CoT and first K trajectory tokens are kept as conditioning context.
3. The teacher VLM continues from that exact prefix under the same structured
   trajectory constraint.
4. The output row masks the student prefix from loss and supervises only the
   teacher continuation.

This is different from scheduled sampling: the label is re-queried from the
teacher under the student's prefix instead of reusing a cached teacher-prefix
distribution.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.decoding import StopOnTrajEndCriteria, TrajDecodingContract, TrajSpanLogitsProcessor
from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora
from src.model.student_wrapper import StudentWrapperConfig, build_student_model
from src.model.tokenizer_ext import distill_trainable_token_ids
from src.training.collator import (
    build_messages,
    build_user_prompt,
    fuse_history_tokens_in_input_ids,
    load_ego_history_xyz,
    load_sample_images,
    resolve_camera_indices,
)
from src.utils.traj_tokens import discrete_traj_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_teacher_pair_300chunks_semantic_balanced_50k.jsonl",
    )
    parser.add_argument(
        "--student-checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / "no_nav_camera_labeled_official_200k"
        / "no_nav_bp3_camera_labeled_official_200k_from_20kbest_20260514_092509"
        / "best_decode",
    )
    parser.add_argument(
        "--teacher-model-path",
        type=Path,
        default=Path("/home/pm97/workspace/sukim/base_weights/Alpamayo-1.5-10B"),
    )
    parser.add_argument(
        "--alpamayo-src",
        type=Path,
        default=Path("/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/src"),
    )
    parser.add_argument("--teacher-runtime-support", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--prefix-tokens", type=int, default=32)
    parser.add_argument("--traj-token-count", type=int, default=128)
    parser.add_argument("--student-max-new-tokens", type=int, default=256)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=192)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data" / "corpus" / "no_nav_token_dagger_smoke64_prefix32.jsonl",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "no_nav_distill" / "token_dagger_smoke64_prefix32.json",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument(
        "--student-only-dry-run",
        action="store_true",
        help="Run student generation and prefix parsing only; do not load the teacher or write corpus rows.",
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


def select_rows(rows: list[dict[str, Any]], *, split: str, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if split and str(row.get("split")) != split:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def select_rows_window(
    rows: list[dict[str, Any]],
    *,
    split: str,
    start_index: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if not split or str(row.get("split")) == split]
    start_index = max(int(start_index), 0)
    if limit <= 0:
        return selected[start_index:]
    return selected[start_index : start_index + int(limit)]


def _single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected single-token encoding for {token!r}, got {token_ids}")
    return int(token_ids[0])


def _traj_start_id(tokenizer) -> int:
    value = getattr(tokenizer, "traj_token_start_idx", None)
    if isinstance(value, int) and value >= 0:
        return int(value)
    value = tokenizer.convert_tokens_to_ids("<i0>")
    if not isinstance(value, int) or value < 0:
        raise ValueError("Tokenizer is missing <i0>")
    return int(value)


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def build_tokenized(
    *,
    sample: dict[str, Any],
    processor,
    tokenizer,
    completion_text: str | None,
    max_length: int,
) -> dict[str, Any]:
    ego_history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
    prompt_text = build_user_prompt(
        sample,
        PROJECT_ROOT,
        ego_history_xyz=ego_history_xyz,
        prompt_text_style="official_alpamayo",
    )
    images = load_sample_images(sample, PROJECT_ROOT)
    camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
    num_frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
    messages = build_messages(
        prompt_text,
        len(images),
        completion_text=completion_text,
        assistant_prefix="<|cot_start|>",
        image_prompt_style="camera_labeled",
        camera_indices=camera_indices,
        num_frames_per_camera=num_frames_per_camera,
    )
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
        max_length=max_length,
    )
    batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, [ego_history_xyz])
    return {
        "batch": batch,
        "text": text,
        "image_count": len(images),
        "camera_indices": camera_indices,
        "num_frames_per_camera": num_frames_per_camera,
    }


def build_tokenized_batch(
    *,
    samples: list[dict[str, Any]],
    processor,
    tokenizer,
    completion_texts: list[str | None],
    max_length: int,
) -> dict[str, Any]:
    texts: list[str] = []
    image_batch: list[list[Any]] = []
    ego_histories = []
    meta: list[dict[str, Any]] = []
    for sample, completion_text in zip(samples, completion_texts, strict=True):
        ego_history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
        prompt_text = build_user_prompt(
            sample,
            PROJECT_ROOT,
            ego_history_xyz=ego_history_xyz,
            prompt_text_style="official_alpamayo",
        )
        images = load_sample_images(sample, PROJECT_ROOT)
        camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
        num_frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
        messages = build_messages(
            prompt_text,
            len(images),
            completion_text=completion_text,
            assistant_prefix="<|cot_start|>",
            image_prompt_style="camera_labeled",
            camera_indices=camera_indices,
            num_frames_per_camera=num_frames_per_camera,
        )
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        )
        image_batch.append(images)
        ego_histories.append(ego_history_xyz)
        meta.append(
            {
                "image_count": len(images),
                "camera_indices": camera_indices,
                "num_frames_per_camera": num_frames_per_camera,
            }
        )
    batch = processor(
        text=texts,
        images=image_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    batch["input_ids"] = fuse_history_tokens_in_input_ids(batch["input_ids"], tokenizer, ego_histories)
    return {"batch": batch, "texts": texts, "meta": meta}


def parse_assistant_span(
    tokenizer,
    token_ids: list[int],
    *,
    traj_token_count: int,
) -> dict[str, Any]:
    cot_end_id = _single_token_id(tokenizer, "<|cot_end|>")
    traj_start_id = _single_token_id(tokenizer, "<|traj_future_start|>")
    traj_end_id = _single_token_id(tokenizer, "<|traj_future_end|>")
    traj_token_start = _traj_start_id(tokenizer)
    traj_token_end = traj_token_start + 2999

    cot_token_ids: list[int] = []
    traj_token_ids: list[int] = []
    cot_end_seen = False
    traj_started = False
    traj_end_seen = False
    for token_id in token_ids:
        token_id = int(token_id)
        if not cot_end_seen:
            if token_id == cot_end_id:
                cot_end_seen = True
            else:
                cot_token_ids.append(token_id)
            continue
        if not traj_started:
            if token_id == traj_start_id:
                traj_started = True
            continue
        if token_id == traj_end_id:
            traj_end_seen = True
            break
        if traj_token_start <= token_id <= traj_token_end:
            traj_token_ids.append(token_id - traj_token_start)
            if len(traj_token_ids) >= traj_token_count:
                continue
    cot_text = tokenizer.decode(cot_token_ids, skip_special_tokens=False).strip()
    return {
        "cot_text": cot_text,
        "cot_token_ids": cot_token_ids,
        "cot_token_count": len(cot_token_ids),
        "traj_token_ids": traj_token_ids,
        "traj_token_count": len(traj_token_ids),
        "cot_end_seen": cot_end_seen,
        "traj_started": traj_started,
        "traj_end_seen": traj_end_seen,
    }


@torch.inference_mode()
def generate_joint(
    *,
    hf_model,
    tokenizer,
    tokenized: dict[str, torch.Tensor],
    prompt_length_for_constraint: int | list[int],
    max_new_tokens: int,
    traj_token_count: int,
    device: str,
) -> torch.Tensor:
    batch = _to_device(tokenized, device)
    if isinstance(prompt_length_for_constraint, int):
        prompt_lengths = [prompt_length_for_constraint]
    else:
        prompt_lengths = [int(value) for value in prompt_length_for_constraint]
    contract = TrajDecodingContract.from_tokenizer(
        tokenizer,
        prompt_lengths=prompt_lengths,
        traj_token_count=traj_token_count,
    )
    generated = hf_model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        logits_processor=LogitsProcessorList([TrajSpanLogitsProcessor(contract)]),
        stopping_criteria=StoppingCriteriaList([StopOnTrajEndCriteria(contract)]),
    )
    return generated.detach().cpu()


def load_student(args: argparse.Namespace):
    train_config = json.loads((args.student_checkpoint_dir / "train_config.json").read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(
        (args.student_checkpoint_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    data_view = train_config.get("data_view") or {}
    base_model = str(train_config["args"]["student_model"])
    use_lora = not bool(train_config["args"].get("disable_lora", False))

    tokenizer = AutoTokenizer.from_pretrained(args.student_checkpoint_dir / "tokenizer", local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.student_checkpoint_dir / "processor", local_files_only=True)
    processor.tokenizer = tokenizer
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int(train_config["trainer_config"].get("max_length", args.max_length)),
        torch_dtype=torch.bfloat16 if str(args.device).startswith("cuda") else None,
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
    checkpoint_format = detect_checkpoint_format(args.student_checkpoint_dir)
    if checkpoint_format == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_info = load_student_checkpoint(args.student_checkpoint_dir, model, use_lora=use_lora)
    model = model.to(args.device).eval()
    return model, tokenizer, processor, {"format": load_info.get("format"), "base_model": base_model}


def load_teacher(args: argparse.Namespace):
    if str(args.alpamayo_src) not in sys.path:
        sys.path.insert(0, str(args.alpamayo_src))
    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    config_path = args.teacher_model_path / "alpamayo_1.5_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    runtime_support = args.teacher_runtime_support
    if runtime_support is None:
        candidate = args.teacher_model_path / "runtime_support" / "Cosmos-Reason2-8B"
        runtime_support = candidate if candidate.exists() else None
    if runtime_support is not None:
        payload["vlm_name_or_path"] = str(runtime_support)
    payload["attn_implementation"] = args.attn_implementation
    config = Alpamayo1_5Config(**payload)
    model = Alpamayo1_5.from_pretrained(
        str(args.teacher_model_path),
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=str(args.device).startswith("cuda"),
    )
    model = model.to(args.device).eval()
    processor = helper.get_processor(model.tokenizer)
    return model, model.tokenizer, processor


def loss_ignore_count_for_prefix(tokenizer, cot_text: str, prefix_count: int) -> int:
    return (
        len(tokenizer.encode(cot_text, add_special_tokens=False))
        + len(tokenizer.encode("<|cot_end|>", add_special_tokens=False))
        + len(tokenizer.encode("<|traj_future_start|>", add_special_tokens=False))
        + int(prefix_count)
    )


def build_dagger_row(
    row: dict[str, Any],
    *,
    student_cot_text: str,
    student_prefix_tokens: list[int],
    dagger_tokens: list[int],
    prefix_tokens: int,
    loss_ignore_count: int,
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    hard_target = copy.deepcopy(out.get("hard_target") or {})
    hard_target.update(
        {
            "cot_text": student_cot_text,
            "traj_future_token_ids": [int(value) for value in dagger_tokens],
            "traj_future_token_ids_path": None,
            "traj_token_count": len(dagger_tokens),
            "source": "token_dagger_teacher_relabel",
            "loss_ignore_completion_token_count": int(loss_ignore_count),
            "dagger_prefix_traj_token_count": int(prefix_tokens),
        }
    )
    out["hard_target"] = hard_target
    weights = dict(out.get("weights") or {})
    weights.update(
        {
            "hard_cot_ce": 0.0,
            "teacher_logit_kd": 0.0,
            "teacher_traj_topk_kd": 0.0,
            "teacher_traj_hidden_align": 0.0,
            "traj_ce": 1.0,
        }
    )
    out["weights"] = weights
    # The original teacher_traj_target/top-k/hidden artifacts were captured
    # under the original teacher prefix. They are conditionally stale for a
    # student-prefix DAgger row, so keep this first DAgger corpus hard-label
    # only unless a later replay pass writes DAgger-specific soft targets.
    out["teacher_traj_target"] = {}
    out["dagger"] = {
        "mode": "token_level_teacher_relabel",
        "prefix_source": "student_free_run",
        "teacher_continuation_source": "alpamayo15_teacher_under_student_prefix",
        "student_prefix_traj_token_count": int(prefix_tokens),
        "student_prefix_traj_token_ids": [int(value) for value in student_prefix_tokens],
        "loss_ignore_completion_token_count": int(loss_ignore_count),
        "supervised_traj_token_count": int(max(len(dagger_tokens) - prefix_tokens, 0)),
    }
    return out


def main() -> None:
    args = parse_args()
    started = time.time()
    rows = select_rows_window(
        load_jsonl(args.corpus_jsonl),
        split=args.split,
        start_index=args.start_index,
        limit=args.max_samples,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "load_student_start", "checkpoint": str(args.student_checkpoint_dir)}), flush=True)
    student, student_tokenizer, student_processor, student_info = load_student(args)
    print(json.dumps({"event": "load_student_done", **student_info}), flush=True)
    teacher = None
    teacher_tokenizer = None
    teacher_processor = None
    if not args.student_only_dry_run:
        print(json.dumps({"event": "load_teacher_start", "model_path": str(args.teacher_model_path)}), flush=True)
        teacher, teacher_tokenizer, teacher_processor = load_teacher(args)
        print(json.dumps({"event": "load_teacher_done"}), flush=True)

    stats = {
        "requested": len(rows),
        "written": 0,
        "student_fail": 0,
        "teacher_fail": 0,
        "dry_run_student_ok": 0,
    }
    failures: list[dict[str, Any]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as out_handle:
        for start_index in range(0, len(rows), max(int(args.batch_size), 1)):
            chunk = rows[start_index : start_index + max(int(args.batch_size), 1)]
            sample_started = time.time()
            try:
                student_base = build_tokenized_batch(
                    samples=chunk,
                    processor=student_processor,
                    tokenizer=student_tokenizer,
                    completion_texts=[None] * len(chunk),
                    max_length=args.max_length,
                )
                student_prompt_lengths = [
                    int(value) for value in student_base["batch"]["attention_mask"].sum(dim=1).tolist()
                ]
                student_input_width = int(student_base["batch"]["input_ids"].shape[1])
                student_sequences = generate_joint(
                    hf_model=student.backbone,
                    tokenizer=student_tokenizer,
                    tokenized=student_base["batch"],
                    prompt_length_for_constraint=student_prompt_lengths,
                    max_new_tokens=args.student_max_new_tokens,
                    traj_token_count=args.traj_token_count,
                    device=args.device,
                ).tolist()
            except Exception as exc:  # noqa: BLE001
                for offset, row in enumerate(chunk):
                    stats["student_fail"] += 1
                    failures.append({"sample_id": str(row.get("sample_id")), "reason": f"student_batch:{str(exc)[:480]}"})
                    print(
                        json.dumps(
                            {
                                "event": "dagger_row_failed",
                                "idx": start_index + offset + 1,
                                "sample_id": str(row.get("sample_id")),
                                "reason": f"student_batch:{exc}",
                            }
                        ),
                        flush=True,
                    )
                continue

            student_spans: list[dict[str, Any] | None] = []
            prefix_texts: list[str | None] = []
            prefix_counts: list[int] = []
            student_prefix_token_items: list[list[int]] = []
            for offset, (row, sequence) in enumerate(zip(chunk, student_sequences, strict=True)):
                sample_id = str(row.get("sample_id"))
                try:
                    student_span = parse_assistant_span(
                        student_tokenizer,
                        sequence[student_input_width:],
                        traj_token_count=args.traj_token_count,
                    )
                    student_tokens = [int(value) for value in student_span["traj_token_ids"]]
                    if (
                        not student_span["cot_text"]
                        or len(student_tokens) < max(int(args.prefix_tokens), 1)
                        or not student_span["cot_end_seen"]
                        or not student_span["traj_started"]
                    ):
                        raise RuntimeError(
                            "student_malformed:"
                            f" cot_len={student_span['cot_token_count']}"
                            f" traj_count={len(student_tokens)}"
                            f" cot_end={student_span['cot_end_seen']}"
                            f" traj_started={student_span['traj_started']}"
                        )
                    stats["dry_run_student_ok"] += 1
                    prefix_count = min(int(args.prefix_tokens), len(student_tokens), int(args.traj_token_count))
                    student_prefix_tokens = student_tokens[:prefix_count]
                    prefix_text = (
                        f"{student_span['cot_text']}<|cot_end|><|traj_future_start|>"
                        + "".join(discrete_traj_token(token_id) for token_id in student_prefix_tokens)
                    )
                    student_spans.append(student_span)
                    prefix_texts.append(prefix_text)
                    prefix_counts.append(prefix_count)
                    student_prefix_token_items.append(student_prefix_tokens)
                    if args.student_only_dry_run:
                        if args.log_every > 0 and (
                            (start_index + offset + 1) % args.log_every == 0
                            or start_index + offset + 1 == len(rows)
                        ):
                            print(
                                json.dumps(
                                    {
                                        "event": "student_prefix_ok",
                                        "idx": start_index + offset + 1,
                                        "sample_id": sample_id,
                                        "cot_tokens": student_span["cot_token_count"],
                                        "student_traj_tokens": len(student_tokens),
                                        "elapsed_sec": round(time.time() - sample_started, 3),
                                    }
                                ),
                                flush=True,
                            )
                    continue
                except Exception as exc:  # noqa: BLE001
                    stats["student_fail"] += 1
                    student_spans.append(None)
                    prefix_texts.append(None)
                    prefix_counts.append(0)
                    student_prefix_token_items.append([])
                    reason = str(exc)
                    failures.append({"sample_id": sample_id, "reason": reason[:500]})
                    print(
                        json.dumps(
                            {
                                "event": "dagger_row_failed",
                                "idx": start_index + offset + 1,
                                "sample_id": sample_id,
                                "reason": reason,
                            }
                        ),
                        flush=True,
                    )

            if args.student_only_dry_run:
                continue

            teacher_rows = [row for row, prefix_text in zip(chunk, prefix_texts, strict=True) if prefix_text is not None]
            teacher_prefix_texts = [prefix_text for prefix_text in prefix_texts if prefix_text is not None]
            teacher_original_offsets = [
                offset for offset, prefix_text in enumerate(prefix_texts) if prefix_text is not None
            ]
            if not teacher_rows:
                continue

            try:
                assert teacher is not None and teacher_tokenizer is not None and teacher_processor is not None
                teacher_base = build_tokenized_batch(
                    samples=teacher_rows,
                    processor=teacher_processor,
                    tokenizer=teacher_tokenizer,
                    completion_texts=[None] * len(teacher_rows),
                    max_length=args.max_length,
                )
                teacher_prefix = build_tokenized_batch(
                    samples=teacher_rows,
                    processor=teacher_processor,
                    tokenizer=teacher_tokenizer,
                    completion_texts=teacher_prefix_texts,
                    max_length=args.max_length,
                )
                teacher_base_lengths = [
                    int(value) for value in teacher_base["batch"]["attention_mask"].sum(dim=1).tolist()
                ]
                teacher_sequences = generate_joint(
                    hf_model=teacher.vlm,
                    tokenizer=teacher_tokenizer,
                    tokenized=teacher_prefix["batch"],
                    prompt_length_for_constraint=teacher_base_lengths,
                    max_new_tokens=args.teacher_max_new_tokens,
                    traj_token_count=args.traj_token_count,
                    device=args.device,
                ).tolist()
            except Exception as exc:  # noqa: BLE001
                for offset, row in zip(teacher_original_offsets, teacher_rows, strict=True):
                    stats["teacher_fail"] += 1
                    failures.append({"sample_id": str(row.get("sample_id")), "reason": f"teacher_batch:{str(exc)[:480]}"})
                    print(
                        json.dumps(
                            {
                                "event": "dagger_row_failed",
                                "idx": start_index + offset + 1,
                                "sample_id": str(row.get("sample_id")),
                                "reason": f"teacher_batch:{exc}",
                            }
                        ),
                        flush=True,
                    )
                continue

            for local_idx, (offset, row, sequence) in enumerate(
                zip(teacher_original_offsets, teacher_rows, teacher_sequences, strict=True)
            ):
                sample_id = str(row.get("sample_id"))
                sample_idx = start_index + offset + 1
                try:
                    teacher_span = parse_assistant_span(
                        teacher_tokenizer,
                        sequence[teacher_base_lengths[local_idx] :],
                        traj_token_count=args.traj_token_count,
                    )
                    dagger_tokens = [int(value) for value in teacher_span["traj_token_ids"]]
                    if len(dagger_tokens) != int(args.traj_token_count) or not teacher_span["traj_end_seen"]:
                        raise RuntimeError(
                            "teacher_malformed:"
                            f" traj_count={len(dagger_tokens)} end={teacher_span['traj_end_seen']}"
                        )
                    student_span = student_spans[offset]
                    assert student_span is not None
                    prefix_count = prefix_counts[offset]
                    loss_ignore = loss_ignore_count_for_prefix(
                        student_tokenizer,
                        student_span["cot_text"],
                        prefix_count,
                    )
                    dagger_row = build_dagger_row(
                        row,
                        student_cot_text=student_span["cot_text"],
                        student_prefix_tokens=student_prefix_token_items[offset],
                        dagger_tokens=dagger_tokens,
                        prefix_tokens=prefix_count,
                        loss_ignore_count=loss_ignore,
                    )
                    out_handle.write(json.dumps(dagger_row, ensure_ascii=True) + "\n")
                    stats["written"] += 1
                    if int(args.flush_every) > 0 and stats["written"] % int(args.flush_every) == 0:
                        out_handle.flush()
                    if args.log_every > 0 and (sample_idx % args.log_every == 0 or sample_idx == len(rows)):
                        print(
                            json.dumps(
                                {
                                    "event": "dagger_row_done",
                                    "idx": sample_idx,
                                    "sample_id": sample_id,
                                    "written": stats["written"],
                                    "student_cot_tokens": student_span["cot_token_count"],
                                    "student_traj_tokens": len(student_span["traj_token_ids"]),
                                    "prefix_tokens": prefix_count,
                                    "dagger_traj_tokens": len(dagger_tokens),
                                    "elapsed_sec": round(time.time() - sample_started, 3),
                                }
                            ),
                            flush=True,
                        )
                except Exception as exc:  # noqa: BLE001
                    reason = str(exc)
                    stats["teacher_fail"] += 1
                    failures.append({"sample_id": sample_id, "reason": reason[:500]})
                    print(
                        json.dumps(
                            {
                                "event": "dagger_row_failed",
                                "idx": sample_idx,
                                "sample_id": sample_id,
                                "reason": reason,
                            }
                        ),
                        flush=True,
                    )

    summary = {
        "corpus_jsonl": str(args.corpus_jsonl),
        "student_checkpoint_dir": str(args.student_checkpoint_dir),
        "teacher_model_path": str(args.teacher_model_path),
        "output_jsonl": str(args.output_jsonl),
        "prefix_tokens": int(args.prefix_tokens),
        "traj_token_count": int(args.traj_token_count),
        "student_only_dry_run": bool(args.student_only_dry_run),
        "elapsed_sec": round(time.time() - started, 3),
        "stats": stats,
        "failures": failures[:50],
    }
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "token_dagger_done", **summary}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
