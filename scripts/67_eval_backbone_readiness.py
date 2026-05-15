#!/usr/bin/env python3
"""Backbone readiness checks for CoT-to-trajectory distillation checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.training.collator import (  # noqa: E402
    DistillationCollator,
    build_messages,
    build_user_prompt,
    load_ego_history_xyz,
    load_sample_images,
    load_traj_future_token_ids,
    resolve_camera_indices,
)
from src.utils.runtime_paths import remap_external_path, resolve_student_model_path  # noqa: E402


def _load_decode_module():
    path = PROJECT_ROOT / "scripts" / "25_decode_checkpoint_overlays.py"
    spec = importlib.util.spec_from_file_location("decode_checkpoint_overlays_25", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import decode helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode_mod = _load_decode_module()
TrajectoryTokenDecoder = decode_mod.TrajectoryTokenDecoder
load_ego_history_rot = decode_mod.load_ego_history_rot
resolve_traj_tokenizer_config_path = decode_mod.resolve_traj_tokenizer_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--tf-batch-size", type=int, default=1)
    parser.add_argument("--gen-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--image-prompt-style",
        choices=("compact", "camera_labeled"),
        default="compact",
        help="Prompt image slots as compact image blocks or Alpamayo camera/frame labels.",
    )
    parser.add_argument("--empty-cot-token-threshold", type=int, default=3)
    parser.add_argument("--stop-path-len-threshold-m", type=float, default=5.0)
    parser.add_argument("--curve-final-y-threshold-m", type=float, default=2.0)
    parser.add_argument("--curve-heading-threshold-rad", type=float, default=0.15)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def batched(rows: list[dict[str, Any]], batch_size: int):
    width = max(int(batch_size), 1)
    for index in range(0, len(rows), width):
        yield rows[index : index + width]


def mean(values: list[float | int]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def percentile(values: list[float | int], q: float) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(clean, q)) if clean else None


def ratio(num: int, den: int) -> float:
    return float(num / max(int(den), 1))


def resolve_path(raw_path: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def load_np_array(raw_path: str | Path | None) -> np.ndarray | None:
    path = resolve_path(raw_path)
    if path is None:
        return None
    return np.load(path)


def load_teacher_traj_topk(sample: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    target = sample.get("teacher_traj_target") or {}
    path = resolve_path(target.get("topk_logits_path") or target.get("topk_ids_path") or target.get("topk_logprobs_path"))
    if path is None:
        return None, None, None
    with np.load(path) as npz:
        ids = np.asarray(npz["topk_indices"], dtype=np.int64) if "topk_indices" in npz else None
        logprobs = np.asarray(npz["topk_logprobs"], dtype=np.float32) if "topk_logprobs" in npz else None
        entropy = np.asarray(npz["entropy"], dtype=np.float32) if "entropy" in npz else None
    return ids, logprobs, entropy


def load_teacher_action_xyz(sample: dict[str, Any]) -> np.ndarray | None:
    raw_path = ((sample.get("teacher_cache") or {}).get("text_raw_json_path"))
    path = resolve_path(raw_path)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = payload.get("results") or []
        pred_xyz = np.asarray((results[0] or {}).get("pred_xyz"), dtype=np.float32)
    except Exception:
        return None
    pred_xyz = np.squeeze(pred_xyz)
    if pred_xyz.ndim != 2 or pred_xyz.shape[-1] < 2:
        return None
    return pred_xyz[:, :3] if pred_xyz.shape[-1] >= 3 else np.pad(pred_xyz[:, :2], ((0, 0), (0, 1)))


def ade_fde(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def path_length(xyz: np.ndarray | None) -> float | None:
    if xyz is None or xyz.shape[0] < 2:
        return None
    return float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=-1).sum())


def heading_change(xyz: np.ndarray | None) -> float | None:
    if xyz is None or xyz.shape[0] < 2:
        return None
    diffs = np.diff(xyz[:, :2], axis=0)
    good = np.linalg.norm(diffs, axis=-1) > 1e-3
    if not np.any(good):
        return 0.0
    headings = np.arctan2(diffs[good, 1], diffs[good, 0])
    return float(headings[-1] - headings[0])


def text_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", value.lower()) if token}


def jaccard(left: str, right: str) -> float | None:
    a = text_tokens(left)
    b = text_tokens(right)
    if not a and not b:
        return None
    return float(len(a & b) / max(len(a | b), 1))


def load_model(args: argparse.Namespace):
    train_config_path = args.checkpoint_dir / "train_config.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.exists() else {}
    checkpoint_manifest_path = args.checkpoint_dir / "checkpoint_manifest.json"
    checkpoint_manifest = (
        json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")) if checkpoint_manifest_path.exists() else {}
    )
    base_model = str((train_config.get("args") or {}).get("student_model") or args.student_model)
    use_lora = not bool((train_config.get("args") or {}).get("disable_lora", False))
    data_view = train_config.get("data_view") or {}

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer_dir = args.checkpoint_dir / "tokenizer"
    processor_dir = args.checkpoint_dir / "processor"
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
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    wrapper_cfg = StudentWrapperConfig(
        student_model_name=base_model,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else None,
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
    if detect_checkpoint_format(args.checkpoint_dir) == "full_state_dict" and use_lora:
        model.backbone = maybe_apply_lora(
            model.backbone,
            LoraConfigSpec(trainable_token_indices=tuple(distill_trainable_token_ids(tokenizer))),
            enabled=True,
        )
    load_student_checkpoint(args.checkpoint_dir, model, use_lora=use_lora)
    return model.to(device).eval(), tokenizer, processor, device, base_model, train_config


def token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected single-token encoding for {token!r}, got {ids}")
    return int(ids[0])


class StopAfterTokenCriteria(StoppingCriteria):
    def __init__(self, *, prompt_lengths: list[int], stop_token_id: int) -> None:
        self.prompt_lengths = [int(value) for value in prompt_lengths]
        self.stop_token_id = int(stop_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for row_index in range(input_ids.shape[0]):
            prompt_len = self.prompt_lengths[min(row_index, len(self.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_len:].tolist()
            if self.stop_token_id not in generated:
                return False
        return True


def summarize_named_stats(values_by_name: dict[str, list[float]]) -> dict[str, dict[str, float | None]]:
    return {
        name: {
            "mean": mean(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
        }
        for name, values in values_by_name.items()
    }


def run_test_a(
    rows: list[dict[str, Any]],
    *,
    model,
    tokenizer,
    processor,
    device: torch.device,
    max_new_tokens: int,
    batch_size: int,
    empty_cot_token_threshold: int,
    image_prompt_style: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cot_end_id = token_id(tokenizer, "<|cot_end|>")
    traj_start_id = token_id(tokenizer, "<|traj_future_start|>")
    traj_end_id = token_id(tokenizer, "<|traj_future_end|>")
    model_dtype = next(model.backbone.parameters()).dtype

    sample_rows: list[dict[str, Any]] = []
    counters = Counter()
    cot_lengths: list[float] = []
    teacher_cot_lengths: list[float] = []
    cot_length_ratios: list[float] = []
    subset_ratios: dict[str, list[float]] = {"all": cot_length_ratios, "teacher_len_ge_32": [], "teacher_len_ge_64": []}
    traj_start_positions: list[int] = []
    cot_jaccards: list[float] = []

    for batch_rows in batched(rows, batch_size):
        texts: list[str] = []
        image_batches: list[list[Any]] = []
        prepared: list[dict[str, Any]] = []
        for sample in batch_rows:
            history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
            prompt_text = build_user_prompt(sample, PROJECT_ROOT, ego_history_xyz=history_xyz)
            images = load_sample_images(sample, PROJECT_ROOT)
            camera_indices = resolve_camera_indices(sample, PROJECT_ROOT, image_count=len(images))
            messages = build_messages(
                prompt_text,
                len(images),
                assistant_prefix="<|cot_start|>",
                image_prompt_style=image_prompt_style,
                camera_indices=camera_indices,
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            texts.append(text)
            image_batches.append(images)
            prepared.append(sample)
        batch = processor(
            text=texts,
            images=image_batches,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        prompt_len = int(batch["input_ids"].shape[1])
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    moved[key] = value.to(device=device, dtype=model_dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        with torch.inference_mode():
            generated = model.backbone.generate(
                **moved,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList(
                    [StopAfterTokenCriteria(prompt_lengths=[prompt_len] * len(prepared), stop_token_id=traj_end_id)]
                ),
                pad_token_id=tokenizer.pad_token_id,
            )
        for row_index, sample in enumerate(prepared):
            new_ids = [int(value) for value in generated[row_index, prompt_len:].detach().cpu().tolist()]
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            cot_end_pos = new_ids.index(cot_end_id) if cot_end_id in new_ids else None
            traj_start_pos = new_ids.index(traj_start_id) if traj_start_id in new_ids else None
            traj_end_pos = new_ids.index(traj_end_id) if traj_end_id in new_ids else None
            traj_start_count = sum(1 for value in new_ids if value == traj_start_id)
            cot_end_count = sum(1 for value in new_ids if value == cot_end_id)
            cot_stop = cot_end_pos if cot_end_pos is not None else traj_start_pos if traj_start_pos is not None else len(new_ids)
            student_cot_ids = new_ids[: max(int(cot_stop), 0)]
            student_cot_text = text.split("<|cot_end|>", 1)[0].split("<|traj_future_start|>", 1)[0]
            student_cot_text = re.sub(r"<\|[^|]+\|>", "", student_cot_text)
            student_cot_text = " ".join(student_cot_text.split())
            teacher_cot = str((sample.get("teacher_target") or {}).get("cot_text") or (sample.get("hard_target") or {}).get("cot_text") or "")
            teacher_cot_ids = tokenizer.encode(teacher_cot, add_special_tokens=False) if teacher_cot else []
            teacher_len = len(teacher_cot_ids)
            student_len = len([value for value in student_cot_ids if value not in (tokenizer.pad_token_id,)])
            length_ratio = float(student_len / max(teacher_len, 1))
            cot_nonempty = bool(student_cot_text.strip()) and student_len >= int(empty_cot_token_threshold)
            invalid_order = not (
                cot_end_pos is not None
                and traj_start_pos is not None
                and traj_end_pos is not None
                and cot_end_pos < traj_start_pos < traj_end_pos
            )
            max_new_tokens_not_ended = traj_end_pos is None
            malformed = (
                not cot_nonempty
                or traj_start_pos is None
                or traj_start_count != 1
                or invalid_order
                or max_new_tokens_not_ended
            )

            counters["samples"] += 1
            counters["cot_nonempty"] += int(cot_nonempty)
            counters["cot_end_hit"] += int(cot_end_pos is not None)
            counters["traj_start_hit"] += int(traj_start_pos is not None)
            counters["traj_end_hit"] += int(traj_end_pos is not None)
            counters["malformed"] += int(malformed)
            counters["empty_cot"] += int(not cot_nonempty)
            counters["multi_start"] += int(traj_start_count >= 2)
            counters["invalid_special_order"] += int(invalid_order)
            counters["max_new_tokens_not_ended"] += int(max_new_tokens_not_ended)
            counters["traj_start_after_cot_end"] += int(
                cot_end_pos is not None and traj_start_pos is not None and traj_start_pos > cot_end_pos
            )
            counters["traj_start_too_early"] += int(length_ratio < 0.7)
            counters["traj_start_too_late"] += int(length_ratio > 1.3)
            cot_lengths.append(float(student_len))
            teacher_cot_lengths.append(float(teacher_len))
            cot_length_ratios.append(length_ratio)
            if teacher_len >= 32:
                subset_ratios["teacher_len_ge_32"].append(length_ratio)
            if teacher_len >= 64:
                subset_ratios["teacher_len_ge_64"].append(length_ratio)
            if traj_start_pos is not None:
                traj_start_positions.append(float(traj_start_pos + 1))
            overlap = jaccard(student_cot_text, teacher_cot)
            if overlap is not None:
                cot_jaccards.append(overlap)

            sample_rows.append(
                {
                    "sample_id": str(sample.get("sample_id")),
                    "teacher_cot_token_count": int(teacher_len),
                    "student_cot_token_count": int(student_len),
                    "cot_length_ratio": length_ratio,
                    "cot_nonempty": cot_nonempty,
                    "cot_end_hit": cot_end_pos is not None,
                    "traj_start_hit": traj_start_pos is not None,
                    "traj_start_after_cot_end": cot_end_pos is not None and traj_start_pos is not None and traj_start_pos > cot_end_pos,
                    "traj_start_position": int(traj_start_pos + 1) if traj_start_pos is not None else None,
                    "traj_end_hit": traj_end_pos is not None,
                    "traj_start_count": int(traj_start_count),
                    "cot_end_count": int(cot_end_count),
                    "invalid_special_order": bool(invalid_order),
                    "max_new_tokens_not_ended": bool(max_new_tokens_not_ended),
                    "malformed": malformed,
                    "cot_jaccard_to_teacher": overlap,
                    "student_cot_preview": student_cot_text[:240],
                    "teacher_cot": teacher_cot,
                }
            )
        print(json.dumps({"event": "test_a_batch_done", "done": len(sample_rows), "total": len(rows)}), flush=True)

    n = int(counters["samples"])
    summary = {
        "num_samples": n,
        "empty_cot_rate": ratio(counters["empty_cot"], n),
        "cot_nonempty_rate": ratio(counters["cot_nonempty"], n),
        "cot_end_hit_rate": ratio(counters["cot_end_hit"], n),
        "traj_start_hit_rate": ratio(counters["traj_start_hit"], n),
        "traj_start_after_cot_end_rate": ratio(counters["traj_start_after_cot_end"], n),
        "traj_end_hit_rate": ratio(counters["traj_end_hit"], n),
        "malformed_output_rate": ratio(counters["malformed"], n),
        "multi_start_rate": ratio(counters["multi_start"], n),
        "invalid_special_order_rate": ratio(counters["invalid_special_order"], n),
        "max_new_tokens_not_ended_rate": ratio(counters["max_new_tokens_not_ended"], n),
        "traj_start_too_early_rate": ratio(counters["traj_start_too_early"], n),
        "traj_start_too_late_rate": ratio(counters["traj_start_too_late"], n),
        "student_cot_tokens_p50": percentile(cot_lengths, 50),
        "student_cot_tokens_p95": percentile(cot_lengths, 95),
        "teacher_cot_tokens_p50": percentile(teacher_cot_lengths, 50),
        "teacher_cot_tokens_p95": percentile(teacher_cot_lengths, 95),
        "cot_length_ratio_p50": percentile(cot_length_ratios, 50),
        "cot_length_ratio_p10": percentile(cot_length_ratios, 10),
        "cot_length_ratio_p90": percentile(cot_length_ratios, 90),
        "cot_length_ratio_p95": percentile(cot_length_ratios, 95),
        "cot_length_ratio_subsets": {
            name: {
                "count": len(values),
                "p10": percentile(values, 10),
                "p50": percentile(values, 50),
                "p90": percentile(values, 90),
            }
            for name, values in subset_ratios.items()
        },
        "traj_start_position_p50": percentile(traj_start_positions, 50),
        "traj_start_position_p95": percentile(traj_start_positions, 95),
        "cot_jaccard_to_teacher_mean": mean(cot_jaccards),
        "cot_jaccard_to_teacher_p50": percentile(cot_jaccards, 50),
    }
    return summary, sample_rows


def run_test_b(
    rows: list[dict[str, Any]],
    *,
    model,
    tokenizer,
    processor,
    device: torch.device,
    train_config: dict[str, Any],
    base_model: str,
    batch_size: int,
    stop_path_len_threshold_m: float,
    curve_final_y_threshold_m: float,
    curve_heading_threshold_rad: float,
    image_prompt_style: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=False,
        enable_teacher_view=False,
        enable_action_aux=False,
        prompt_mode=str((train_config.get("data_view") or {}).get("prompt_mode") or "joint"),
        target_mode=str((train_config.get("data_view") or {}).get("target_mode") or "joint"),
        image_prompt_style=image_prompt_style,
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
    )
    decoder_path = resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not find Alpamayo traj tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=decoder_path)
    traj_start = int(getattr(tokenizer, "traj_token_start_idx", tokenizer.convert_tokens_to_ids("<i0>")))
    num_bins = int(decoder.num_bins)
    model_dtype = next(model.backbone.parameters()).dtype

    bucket_defs = {
        "pos_001_016": (0, 16),
        "pos_017_064": (16, 64),
        "pos_065_128": (64, 128),
    }
    correct_by_bucket = Counter()
    total_by_bucket = Counter()
    rank_by_bucket: dict[str, list[float]] = defaultdict(list)
    kl_by_bucket: dict[str, list[float]] = defaultdict(list)
    entropy_by_bucket: dict[str, list[float]] = defaultdict(list)
    margin_by_bucket: dict[str, list[float]] = defaultdict(list)
    nll_by_bucket: dict[str, list[float]] = defaultdict(list)

    parity_correct = Counter()
    parity_total = Counter()
    parity_ranks: dict[str, list[float]] = defaultdict(list)
    parity_kl: dict[str, list[float]] = defaultdict(list)
    first16_curv_correct = 0
    first16_curv_total = 0
    first16_curv_kl: list[float] = []

    all_ranks: list[float] = []
    all_kl: list[float] = []
    all_entropy: list[float] = []
    all_margin: list[float] = []
    all_nll: list[float] = []

    topk_hits = Counter()
    geom_ade: list[float] = []
    geom_fde: list[float] = []
    geom_action_ade: list[float] = []
    geom_action_fde: list[float] = []
    first16_geom_ade: list[float] = []
    first16_geom_fde: list[float] = []
    stop_geom_ade: list[float] = []
    stop_geom_fde: list[float] = []
    curve_geom_ade: list[float] = []
    curve_geom_fde: list[float] = []
    bucket_counts = Counter()
    token_accs: list[float] = []
    malformed = 0
    samples: list[dict[str, Any]] = []

    for batch_rows in batched(rows, batch_size):
        batch = collator(batch_rows)
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if torch.is_floating_point(value):
                    moved[key] = value.to(device=device, dtype=model_dtype)
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        forward_kwargs = {
            "input_ids": moved["input_ids"],
            "attention_mask": moved["attention_mask"],
            "return_hidden_states": False,
            "compute_meta_action": False,
            "compute_traj_aux": False,
        }
        for optional_key in ("pixel_values", "image_grid_thw"):
            if optional_key in moved:
                forward_kwargs[optional_key] = moved[optional_key]
        with torch.inference_mode():
            outputs = model(**forward_kwargs)
        logits = outputs["logits"].float()
        labels = moved["labels"]
        traj_mask = moved["traj_token_mask"].bool() & (labels != -100)

        for row_index, sample in enumerate(batch_rows):
            label_positions = torch.nonzero(traj_mask[row_index], as_tuple=False).flatten()
            label_positions = label_positions[label_positions > 0]
            logits_pos = logits[row_index, label_positions - 1, traj_start : traj_start + num_bins]
            target_ids = load_traj_future_token_ids(sample.get("hard_target") or {}, PROJECT_ROOT)
            target = torch.as_tensor(target_ids[: logits_pos.shape[0]], device=logits_pos.device, dtype=torch.long)
            usable = min(int(logits_pos.shape[0]), int(target.shape[0]), 128)
            if usable <= 0:
                malformed += 1
                continue
            logits_pos = logits_pos[:usable]
            target = target[:usable]
            log_probs = torch.log_softmax(logits_pos, dim=-1)
            probs = torch.softmax(logits_pos, dim=-1)
            pred = logits_pos.argmax(dim=-1)
            target_logit = logits_pos.gather(1, target[:, None]).squeeze(1)
            ranks = (logits_pos > target_logit[:, None]).sum(dim=-1).to(torch.float32) + 1.0
            top_values, top_indices = torch.topk(logits_pos, k=10, dim=-1)
            del top_values
            target_lp = log_probs.gather(1, target[:, None]).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1)
            top2_log_probs = torch.topk(log_probs, k=2, dim=-1).values
            margins = top2_log_probs[:, 0] - top2_log_probs[:, 1]
            nll = -target_lp
            correct = pred == target

            teacher_topk_ids_np, teacher_topk_lp_np, _teacher_entropy_np = load_teacher_traj_topk(sample)
            teacher_kl = [float("nan")] * usable
            if teacher_topk_ids_np is not None and teacher_topk_lp_np is not None:
                t_ids = torch.as_tensor(teacher_topk_ids_np[:usable], device=logits_pos.device, dtype=torch.long)
                t_lp = torch.as_tensor(teacher_topk_lp_np[:usable], device=logits_pos.device, dtype=torch.float32)
                valid = (t_ids >= 0) & (t_ids < num_bins)
                gathered_student_lp = torch.zeros_like(t_lp)
                safe_ids = t_ids.clamp(0, num_bins - 1)
                gathered_student_lp = log_probs.gather(1, safe_ids)
                gathered_student_lp = torch.where(valid, gathered_student_lp, torch.zeros_like(gathered_student_lp))
                teacher_probs = torch.softmax(torch.where(valid, t_lp, torch.finfo(t_lp.dtype).min), dim=-1)
                teacher_log_probs_norm = torch.log_softmax(
                    torch.where(valid, t_lp, torch.finfo(t_lp.dtype).min), dim=-1
                )
                kl = (teacher_probs * (teacher_log_probs_norm - gathered_student_lp)).sum(dim=-1)
                teacher_kl = [float(value) for value in kl.detach().cpu().tolist()]

            pred_ids = [int(value) for value in pred.detach().cpu().tolist()]
            invalid_count = sum(1 for value in pred_ids if value < 0 or value >= num_bins)
            malformed += int(usable != 128 or invalid_count > 0)
            target_list = [int(value) for value in target.detach().cpu().tolist()]
            acc = float(correct.float().mean().item())
            token_accs.append(acc)
            top1_hit = correct
            top5_hit = (top_indices[:, :5] == target[:, None]).any(dim=-1)
            top10_hit = (top_indices[:, :10] == target[:, None]).any(dim=-1)
            topk_hits["top1"] += int(top1_hit.sum().item())
            topk_hits["top5"] += int(top5_hit.sum().item())
            topk_hits["top10"] += int(top10_hit.sum().item())
            topk_hits["total"] += int(usable)

            rank_list = [float(value) for value in ranks.detach().cpu().tolist()]
            entropy_list = [float(value) for value in entropy.detach().cpu().tolist()]
            margin_list = [float(value) for value in margins.detach().cpu().tolist()]
            nll_list = [float(value) for value in nll.detach().cpu().tolist()]
            correct_list = [bool(value) for value in correct.detach().cpu().tolist()]
            for pos in range(usable):
                all_ranks.append(rank_list[pos])
                all_entropy.append(entropy_list[pos])
                all_margin.append(margin_list[pos])
                all_nll.append(nll_list[pos])
                if math.isfinite(teacher_kl[pos]):
                    all_kl.append(teacher_kl[pos])
                for bucket_name, (start, end) in bucket_defs.items():
                    if start <= pos < end:
                        total_by_bucket[bucket_name] += 1
                        correct_by_bucket[bucket_name] += int(correct_list[pos])
                        rank_by_bucket[bucket_name].append(rank_list[pos])
                        entropy_by_bucket[bucket_name].append(entropy_list[pos])
                        margin_by_bucket[bucket_name].append(margin_list[pos])
                        nll_by_bucket[bucket_name].append(nll_list[pos])
                        if math.isfinite(teacher_kl[pos]):
                            kl_by_bucket[bucket_name].append(teacher_kl[pos])
                        break
                parity_name = "accel_even" if pos % 2 == 0 else "curvature_odd"
                parity_total[parity_name] += 1
                parity_correct[parity_name] += int(correct_list[pos])
                parity_ranks[parity_name].append(rank_list[pos])
                if math.isfinite(teacher_kl[pos]):
                    parity_kl[parity_name].append(teacher_kl[pos])
                if pos < 16 and pos % 2 == 1:
                    first16_curv_total += 1
                    first16_curv_correct += int(correct_list[pos])
                    if math.isfinite(teacher_kl[pos]):
                        first16_curv_kl.append(teacher_kl[pos])

            ade = fde = float("nan")
            action_ade = action_fde = float("nan")
            first16_ade = first16_fde = float("nan")
            if usable == 128 and invalid_count == 0:
                history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
                history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
                pred_xyz = decoder.decode(history_xyz, history_rot, pred_ids[:128])
                teacher_xyz = decoder.decode(history_xyz, history_rot, target_list[:128])
                if pred_xyz is not None and teacher_xyz is not None:
                    ade, fde = ade_fde(pred_xyz, teacher_xyz)
                    geom_ade.append(ade)
                    geom_fde.append(fde)
                    first16_tokens = pred_ids[:16] + target_list[16:128]
                    first16_xyz = decoder.decode(history_xyz, history_rot, first16_tokens)
                    if first16_xyz is not None:
                        first16_ade, first16_fde = ade_fde(first16_xyz, teacher_xyz)
                        first16_geom_ade.append(first16_ade)
                        first16_geom_fde.append(first16_fde)
                    teacher_action_xyz = load_teacher_action_xyz(sample)
                    if teacher_action_xyz is not None:
                        action_ade, action_fde = ade_fde(pred_xyz, teacher_action_xyz)
                        geom_action_ade.append(action_ade)
                        geom_action_fde.append(action_fde)
                    teacher_path_len = path_length(teacher_xyz)
                    teacher_heading_delta = heading_change(teacher_xyz)
                    teacher_final_y = float(teacher_xyz[-1, 1]) if teacher_xyz.shape[0] else 0.0
                    is_stop = teacher_path_len is not None and teacher_path_len <= float(stop_path_len_threshold_m)
                    is_curve = abs(teacher_final_y) >= float(curve_final_y_threshold_m) or (
                        teacher_heading_delta is not None
                        and abs(teacher_heading_delta) >= float(curve_heading_threshold_rad)
                    )
                    bucket_counts["stop"] += int(is_stop)
                    bucket_counts["curve"] += int(is_curve)
                    if is_stop:
                        stop_geom_ade.append(ade)
                        stop_geom_fde.append(fde)
                    if is_curve:
                        curve_geom_ade.append(ade)
                        curve_geom_fde.append(fde)
            samples.append(
                {
                    "sample_id": str(sample.get("sample_id")),
                    "token_acc": acc,
                    "ade_m": ade if math.isfinite(ade) else None,
                    "fde_m": fde if math.isfinite(fde) else None,
                    "action_teacher_ade_m": action_ade if math.isfinite(action_ade) else None,
                    "action_teacher_fde_m": action_fde if math.isfinite(action_fde) else None,
                    "first16_hybrid_ade_m": first16_ade if math.isfinite(first16_ade) else None,
                    "first16_hybrid_fde_m": first16_fde if math.isfinite(first16_fde) else None,
                    "invalid_count": int(invalid_count),
                    "rank_mean": mean(rank_list),
                    "entropy_mean": mean(entropy_list),
                    "margin_mean": mean(margin_list),
                }
            )
        print(json.dumps({"event": "test_b_batch_done", "done": len(samples), "total": len(rows)}), flush=True)

    bucket_summary = {}
    for name in bucket_defs:
        bucket_summary[name] = {
            "accuracy": ratio(correct_by_bucket[name], total_by_bucket[name]),
            "count": int(total_by_bucket[name]),
            "rank_mean": mean(rank_by_bucket[name]),
            "rank_p50": percentile(rank_by_bucket[name], 50),
            "rank_p95": percentile(rank_by_bucket[name], 95),
            "nll_mean": mean(nll_by_bucket[name]),
            "entropy_mean": mean(entropy_by_bucket[name]),
            "top1_margin_mean": mean(margin_by_bucket[name]),
            "teacher_topk_sparse_kl_mean": mean(kl_by_bucket[name]),
        }
    parity_summary = {}
    for name in ("accel_even", "curvature_odd"):
        parity_summary[name] = {
            "accuracy": ratio(parity_correct[name], parity_total[name]),
            "count": int(parity_total[name]),
            "rank_mean": mean(parity_ranks[name]),
            "rank_p95": percentile(parity_ranks[name], 95),
            "teacher_topk_sparse_kl_mean": mean(parity_kl[name]),
        }
    total_positions = int(topk_hits["total"])
    summary = {
        "num_samples": len(samples),
        "malformed_count": int(malformed),
        "malformed_rate": ratio(malformed, len(samples)),
        "decoded_teacher_forced_ade_m": mean(geom_ade),
        "decoded_teacher_forced_fde_m": mean(geom_fde),
        "traj_token_acc": mean(token_accs),
        "target_in_student_top1_rate": ratio(topk_hits["top1"], total_positions),
        "target_in_student_top5_rate": ratio(topk_hits["top5"], total_positions),
        "target_in_student_top10_rate": ratio(topk_hits["top10"], total_positions),
        "target_rank_mean": mean(all_ranks),
        "target_rank_p50": percentile(all_ranks, 50),
        "target_rank_p95": percentile(all_ranks, 95),
        "target_nll_mean": mean(all_nll),
        "student_entropy_mean": mean(all_entropy),
        "student_top1_margin_mean": mean(all_margin),
        "teacher_topk_sparse_kl_mean": mean(all_kl),
        "first16_curvature": {
            "accuracy": ratio(first16_curv_correct, first16_curv_total),
            "count": int(first16_curv_total),
            "teacher_topk_sparse_kl_mean": mean(first16_curv_kl),
        },
        "geometry": {
            "tf_argmax_vs_discrete_teacher_ade_m": mean(geom_ade),
            "tf_argmax_vs_discrete_teacher_fde_m": mean(geom_fde),
            "first16_hybrid_vs_discrete_teacher_ade_m": mean(first16_geom_ade),
            "first16_hybrid_vs_discrete_teacher_fde_m": mean(first16_geom_fde),
            "tf_argmax_vs_action_teacher_ade_m": mean(geom_action_ade),
            "tf_argmax_vs_action_teacher_fde_m": mean(geom_action_fde),
            "stop_bucket_count": int(bucket_counts["stop"]),
            "stop_bucket_ade_m": mean(stop_geom_ade),
            "stop_bucket_fde_m": mean(stop_geom_fde),
            "curve_bucket_count": int(bucket_counts["curve"]),
            "curve_bucket_ade_m": mean(curve_geom_ade),
            "curve_bucket_fde_m": mean(curve_geom_fde),
            "stop_bucket_definition": f"teacher discrete path_len <= {stop_path_len_threshold_m}m",
            "curve_bucket_definition": (
                f"abs(final_y) >= {curve_final_y_threshold_m}m or "
                f"abs(heading_delta) >= {curve_heading_threshold_rad}rad"
            ),
        },
        "position_buckets": bucket_summary,
        "channel_parity": parity_summary,
    }
    return summary, samples


def main() -> int:
    args = parse_args()
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.samples_jsonl is None:
        args.samples_jsonl = args.summary_json.with_suffix(".samples.jsonl")
    rows = [row for row in load_jsonl(args.corpus_jsonl) if row.get("split") == args.split]
    if args.num_samples > 0:
        rows = rows[: args.num_samples]
    if not rows:
        raise SystemExit(f"No rows selected for split={args.split!r}")

    model, tokenizer, processor, device, base_model, train_config = load_model(args)
    test_a, samples_a = run_test_a(
        rows,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size,
        empty_cot_token_threshold=args.empty_cot_token_threshold,
        image_prompt_style=args.image_prompt_style,
    )
    test_b, samples_b = run_test_b(
        rows,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        device=device,
        train_config=train_config,
        base_model=base_model,
        batch_size=args.tf_batch_size,
        stop_path_len_threshold_m=args.stop_path_len_threshold_m,
        curve_final_y_threshold_m=args.curve_final_y_threshold_m,
        curve_heading_threshold_rad=args.curve_heading_threshold_rad,
        image_prompt_style=args.image_prompt_style,
    )

    by_id: dict[str, dict[str, Any]] = {}
    for row in samples_a:
        by_id.setdefault(str(row["sample_id"]), {})["test_a"] = row
    for row in samples_b:
        by_id.setdefault(str(row["sample_id"]), {})["test_b"] = row
    with args.samples_jsonl.open("w", encoding="utf-8") as handle:
        for sample_id, payload in by_id.items():
            handle.write(json.dumps({"sample_id": sample_id, **payload}, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split,
        "num_samples": len(rows),
        "test_a_interface_generation": test_a,
        "test_b_teacher_forced_discrete_action": test_b,
        "decoding_config": {
            "do_sample": False,
            "temperature": 0,
            "top_p": None,
            "top_k": None,
            "max_new_tokens": int(args.max_new_tokens),
            "future_token_vocab_restricted_metrics": True,
            "gen_batch_size": int(args.gen_batch_size),
            "tf_batch_size": int(args.tf_batch_size),
            "image_prompt_style": args.image_prompt_style,
        },
        "samples_jsonl": str(args.samples_jsonl),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
