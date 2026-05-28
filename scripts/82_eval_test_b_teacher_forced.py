#!/usr/bin/env python3
"""Teacher-forced trajectory-token Test B evaluator.

This compares backbone checkpoints on the same corpus/input contract using
future-token-vocab restricted logits at the 128 teacher trajectory-token
positions.  It writes token, horizon, accel/curv, pair, scene, failure-tag, and
decoded geometry metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.checkpoint_io import detect_checkpoint_format, load_student_checkpoint  # noqa: E402
from src.model.peft_setup import LoraConfigSpec, maybe_apply_lora  # noqa: E402
from src.model.student_wrapper import StudentWrapperConfig, build_student_model  # noqa: E402
from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens  # noqa: E402
from src.training.collator import (  # noqa: E402
    DistillationCollator,
    load_ego_history_xyz,
    load_traj_future_token_ids,
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


RANGES: dict[str, tuple[int, int]] = {
    "pos_001_016": (0, 16),
    "pos_017_064": (16, 64),
    "pos_065_128": (64, 128),
    "pos_001_064": (0, 64),
    "pos_001_128": (0, 128),
}

PAIR_RANGES: dict[str, tuple[int, int]] = {
    "pair_all": (0, 64),
    "pair_first8": (0, 8),
    "pair_first16": (0, 16),
    "pair_late": (32, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--student-model", default=resolve_student_model_path())
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-prompt-style", choices=("compact", "camera_labeled"), default="camera_labeled")
    parser.add_argument(
        "--prompt-text-style",
        choices=("numeric_history_question", "official_alpamayo"),
        default="official_alpamayo",
    )
    parser.add_argument("--fuse-history-tokens", action="store_true")
    parser.add_argument("--failure-summary-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--samples-jsonl", type=Path, default=None)
    parser.add_argument(
        "--save-token-sequences",
        action="store_true",
        help="Store teacher-forced argmax and target token ids in the per-sample JSONL.",
    )
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


def mean(values: list[float | int | bool]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def percentile(values: list[float | int], q: float) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(clean, q)) if clean else None


def ratio(num: int, den: int) -> float:
    return float(num / max(int(den), 1))


def resolve_path(raw_path: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def load_teacher_traj_topk(sample: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    target = sample.get("teacher_traj_target") or {}
    path = resolve_path(target.get("topk_logits_path") or target.get("topk_ids_path"))
    if path is None:
        return None, None
    try:
        with np.load(path) as npz:
            ids = np.asarray(npz["topk_indices"], dtype=np.int64)
            logprobs = np.asarray(npz["topk_logprobs"], dtype=np.float32)
        return ids, logprobs
    except Exception:
        return None, None


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


def ade_fde(pred: np.ndarray | None, target: np.ndarray | None) -> tuple[float, float]:
    if pred is None or target is None:
        return float("nan"), float("nan")
    n = min(int(pred.shape[0]), int(target.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    dist = np.linalg.norm(pred[:n, :2] - target[:n, :2], axis=-1)
    return float(dist.mean()), float(dist[-1])


def teacher_text(sample: dict[str, Any]) -> str:
    return str((sample.get("teacher_target") or {}).get("cot_text") or (sample.get("hard_target") or {}).get("cot_text") or "")


def has(text: str, *subs: str) -> bool:
    return any(sub in text for sub in subs)


def scene_bucket(sample: dict[str, Any]) -> str:
    text = teacher_text(sample).lower()
    priority = [
        ("traffic_right_turn", lambda t: has(t, "traffic light", "green light", "red light") and has(t, "turn right", "right turn")),
        ("traffic_left_turn", lambda t: has(t, "traffic light", "green light", "red light") and has(t, "turn left", "left turn")),
        ("right_turn_no_light", lambda t: has(t, "turn right", "right turn")),
        ("left_turn_no_light", lambda t: has(t, "turn left", "left turn")),
        ("red_light_stop", lambda t: has(t, "red light", "light is red", "traffic light is red")),
        ("stop_sign", lambda t: has(t, "stop sign", "all-way stop")),
        ("pedestrian_crosswalk", lambda t: has(t, "pedestrian", "crosswalk")),
        ("cut_in_merge_yield", lambda t: has(t, "cut-in", "cut in", "merge", "merges into our lane")),
        ("lead_vehicle_follow", lambda t: has(t, "lead vehicle", "directly ahead in our lane", "vehicle ahead", "follow the vehicle")),
        ("parked_stopped_obstacle_nudge", lambda t: has(t, "nudge", "parked car", "parked vehicle", "parked cars", "stopped vehicle", "blocking")),
        ("lane_change", lambda t: has(t, "lane change", "change lane", "change lanes")),
        ("curve", lambda t: has(t, "curve", "curvature")),
        ("green_light_go_straight", lambda t: has(t, "green light", "light is green", "traffic light is green")),
        ("intersection_other", lambda t: has(t, "intersection")),
        ("slow_decel_other", lambda t: has(t, "slow down", "decelerate", "deceleration", "slow")),
        ("keep_lane_straight", lambda t: has(t, "keep lane", "lane is clear", "keep speed", "straight")),
    ]
    for name, fn in priority:
        if fn(text):
            return name
    return "other"


def load_failure_tags(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for sample in payload.get("samples") or []:
        sid = str(sample.get("sample_id"))
        out[sid] = [str(tag) for tag in sample.get("failure_tags") or []]
    return out


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


class ScalarStats:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float | int | None) -> None:
        if value is not None and math.isfinite(float(value)):
            self.values.append(float(value))

    def summary(self) -> dict[str, float | None]:
        return {
            "mean": mean(self.values),
            "p50": percentile(self.values, 50),
            "p95": percentile(self.values, 95),
        }


class TokenBucket:
    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.ce = ScalarStats()
        self.kl = ScalarStats()
        self.entropy = ScalarStats()
        self.margin = ScalarStats()
        self.rank = ScalarStats()
        self.top5 = 0
        self.top10 = 0

    def add(
        self,
        *,
        correct: bool,
        ce: float,
        kl: float | None,
        entropy: float,
        margin: float,
        rank: float,
        top5: bool,
        top10: bool,
    ) -> None:
        self.total += 1
        self.correct += int(correct)
        self.top5 += int(top5)
        self.top10 += int(top10)
        self.ce.add(ce)
        self.kl.add(kl)
        self.entropy.add(entropy)
        self.margin.add(margin)
        self.rank.add(rank)

    def summary(self) -> dict[str, Any]:
        return {
            "count": int(self.total),
            "acc": ratio(self.correct, self.total),
            "ce": mean(self.ce.values),
            "kl": mean(self.kl.values),
            "entropy": mean(self.entropy.values),
            "margin": mean(self.margin.values),
            "target_rank_mean": mean(self.rank.values),
            "target_rank_p50": percentile(self.rank.values, 50),
            "target_rank_p95": percentile(self.rank.values, 95),
            "target_rank_median": percentile(self.rank.values, 50),
            "top5": ratio(self.top5, self.total),
            "top10": ratio(self.top10, self.total),
        }


def add_token_to_buckets(
    buckets: dict[str, TokenBucket],
    pos: int,
    payload: dict[str, Any],
    *,
    scene: str,
    failure_tags: list[str],
) -> None:
    names = ["all"]
    action_axis = "accel_even" if pos % 2 == 0 else "curv_odd"
    for name, (start, end) in RANGES.items():
        if start <= pos < end:
            names.append(name)
    names.append(action_axis)
    for range_name, (start, end) in RANGES.items():
        if start <= pos < end:
            names.append(f"{range_name}__{action_axis}")
    names.append(f"scene::{scene}")
    names.append(f"scene::{scene}__{action_axis}")
    for range_name, (start, end) in RANGES.items():
        if start <= pos < end:
            names.append(f"scene::{scene}__{range_name}")
    for tag in failure_tags or ["none"]:
        names.append(f"failure::{tag}")
        names.append(f"failure::{tag}__{action_axis}")
        for range_name, (start, end) in RANGES.items():
            if start <= pos < end:
                names.append(f"failure::{tag}__{range_name}")
    for name in names:
        buckets[name].add(**payload)


def summarize_pair(flags: list[bool], pair_start: int, pair_end: int) -> float | None:
    vals: list[bool] = []
    for pair_idx in range(pair_start, min(pair_end, len(flags) // 2)):
        vals.append(bool(flags[2 * pair_idx]) and bool(flags[2 * pair_idx + 1]))
    return mean(vals)


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
    failure_map = load_failure_tags(args.failure_summary_json)

    model, tokenizer, processor, device, base_model, train_config = load_model(args)
    collator = DistillationCollator(
        tokenizer=tokenizer,
        processor=processor,
        project_root=PROJECT_ROOT,
        teacher_pair_target=False,
        enable_teacher_view=False,
        enable_action_aux=False,
        prompt_mode=str((train_config.get("data_view") or {}).get("prompt_mode") or "joint"),
        target_mode=str((train_config.get("data_view") or {}).get("target_mode") or "joint"),
        image_prompt_style=args.image_prompt_style,
        prompt_text_style=args.prompt_text_style,
        fuse_history_tokens=bool(args.fuse_history_tokens),
        max_length=int((train_config.get("trainer_config") or {}).get("max_length", 4096)),
    )
    decoder_path = resolve_traj_tokenizer_config_path(base_model)
    if decoder_path is None:
        raise SystemExit("Could not find Alpamayo traj tokenizer config.")
    decoder = TrajectoryTokenDecoder(config_path=decoder_path)
    traj_start = int(getattr(tokenizer, "traj_token_start_idx", tokenizer.convert_tokens_to_ids("<i0>")))
    num_bins = int(decoder.num_bins)
    model_dtype = next(model.backbone.parameters()).dtype

    buckets: dict[str, TokenBucket] = defaultdict(TokenBucket)
    pair_values: dict[str, list[float]] = defaultdict(list)
    geometry: dict[str, list[float]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []
    malformed = 0

    for batch_rows in batched(rows, args.batch_size):
        batch = collator(batch_rows)
        moved: dict[str, Any] = {}
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
            if optional_key in moved and moved[optional_key] is not None:
                forward_kwargs[optional_key] = moved[optional_key]
        with torch.inference_mode():
            outputs = model(**forward_kwargs)
        logits = outputs["logits"].float()
        labels = moved["labels"]
        traj_mask = moved["traj_token_mask"].bool() & (labels != -100)

        for row_index, sample in enumerate(batch_rows):
            sid = str(sample.get("sample_id"))
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
            top_indices = torch.topk(logits_pos, k=10, dim=-1).indices
            target_lp = log_probs.gather(1, target[:, None]).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1)
            top2_log_probs = torch.topk(log_probs, k=2, dim=-1).values
            margins = top2_log_probs[:, 0] - top2_log_probs[:, 1]
            nll = -target_lp
            correct = pred == target
            top5_hit = (top_indices[:, :5] == target[:, None]).any(dim=-1)
            top10_hit = (top_indices[:, :10] == target[:, None]).any(dim=-1)

            teacher_topk_ids_np, teacher_topk_lp_np = load_teacher_traj_topk(sample)
            teacher_kl = [float("nan")] * usable
            if teacher_topk_ids_np is not None and teacher_topk_lp_np is not None:
                t_ids = torch.as_tensor(teacher_topk_ids_np[:usable], device=logits_pos.device, dtype=torch.long)
                t_lp = torch.as_tensor(teacher_topk_lp_np[:usable], device=logits_pos.device, dtype=torch.float32)
                valid = (t_ids >= 0) & (t_ids < num_bins)
                safe_ids = t_ids.clamp(0, num_bins - 1)
                gathered_student_lp = log_probs.gather(1, safe_ids)
                gathered_student_lp = torch.where(valid, gathered_student_lp, torch.zeros_like(gathered_student_lp))
                masked_lp = torch.where(valid, t_lp, torch.finfo(t_lp.dtype).min)
                teacher_probs = torch.softmax(masked_lp, dim=-1)
                teacher_log_probs_norm = torch.log_softmax(masked_lp, dim=-1)
                kl = (teacher_probs * (teacher_log_probs_norm - gathered_student_lp)).sum(dim=-1)
                teacher_kl = [float(value) for value in kl.detach().cpu().tolist()]

            pred_ids = [int(value) for value in pred.detach().cpu().tolist()]
            target_list = [int(value) for value in target.detach().cpu().tolist()]
            invalid_count = sum(1 for value in pred_ids if value < 0 or value >= num_bins)
            malformed += int(usable != 128 or invalid_count > 0)

            correct_list = [bool(value) for value in correct.detach().cpu().tolist()]
            rank_list = [float(value) for value in ranks.detach().cpu().tolist()]
            entropy_list = [float(value) for value in entropy.detach().cpu().tolist()]
            margin_list = [float(value) for value in margins.detach().cpu().tolist()]
            nll_list = [float(value) for value in nll.detach().cpu().tolist()]
            top5_list = [bool(value) for value in top5_hit.detach().cpu().tolist()]
            top10_list = [bool(value) for value in top10_hit.detach().cpu().tolist()]
            scene = scene_bucket(sample)
            failure_tags = failure_map.get(sid, [])

            for pos in range(usable):
                add_token_to_buckets(
                    buckets,
                    pos,
                    {
                        "correct": correct_list[pos],
                        "ce": nll_list[pos],
                        "kl": teacher_kl[pos],
                        "entropy": entropy_list[pos],
                        "margin": margin_list[pos],
                        "rank": rank_list[pos],
                        "top5": top5_list[pos],
                        "top10": top10_list[pos],
                    },
                    scene=scene,
                    failure_tags=failure_tags,
                )

            for name, (pair_start, pair_end) in PAIR_RANGES.items():
                pair_values[name].append(summarize_pair(correct_list, pair_start, pair_end) or 0.0)

            ade = fde = action_ade = action_fde = first16_ade = first16_fde = first16_action_ade = first16_action_fde = float("nan")
            if usable == 128 and invalid_count == 0:
                history_xyz = load_ego_history_xyz(sample, PROJECT_ROOT)
                history_rot = load_ego_history_rot(sample, PROJECT_ROOT)
                pred_xyz = decoder.decode(history_xyz, history_rot, pred_ids[:128])
                teacher_xyz = decoder.decode(history_xyz, history_rot, target_list[:128])
                ade, fde = ade_fde(pred_xyz, teacher_xyz)
                geometry["tf_argmax_vs_discrete_ade"].append(ade)
                geometry["tf_argmax_vs_discrete_fde"].append(fde)
                first16_tokens = pred_ids[:16] + target_list[16:128]
                first16_xyz = decoder.decode(history_xyz, history_rot, first16_tokens)
                first16_ade, first16_fde = ade_fde(first16_xyz, teacher_xyz)
                geometry["first16_hybrid_vs_discrete_ade"].append(first16_ade)
                geometry["first16_hybrid_vs_discrete_fde"].append(first16_fde)
                teacher_action_xyz = load_teacher_action_xyz(sample)
                action_ade, action_fde = ade_fde(pred_xyz, teacher_action_xyz)
                first16_action_ade, first16_action_fde = ade_fde(first16_xyz, teacher_action_xyz)
                geometry["tf_argmax_vs_action_ade"].append(action_ade)
                geometry["tf_argmax_vs_action_fde"].append(action_fde)
                geometry["first16_hybrid_vs_action_ade"].append(first16_action_ade)
                geometry["first16_hybrid_vs_action_fde"].append(first16_action_fde)
                for group in (f"scene::{scene}", *(f"failure::{tag}" for tag in failure_tags)):
                    geometry[f"{group}::tf_argmax_vs_discrete_ade"].append(ade)
                    geometry[f"{group}::tf_argmax_vs_discrete_fde"].append(fde)
                    geometry[f"{group}::tf_argmax_vs_action_ade"].append(action_ade)
                    geometry[f"{group}::tf_argmax_vs_action_fde"].append(action_fde)

            sample_row = {
                "sample_id": sid,
                "scene_bucket": scene,
                "failure_tags": failure_tags,
                "usable_token_count": int(usable),
                "invalid_count": int(invalid_count),
                "all_acc": mean(correct_list),
                "accel_even_acc": mean([correct_list[i] for i in range(0, usable, 2)]),
                "curv_odd_acc": mean([correct_list[i] for i in range(1, usable, 2)]),
                "pair_acc_all": pair_values["pair_all"][-1],
                "tf_argmax_vs_discrete_ade_m": ade if math.isfinite(ade) else None,
                "tf_argmax_vs_discrete_fde_m": fde if math.isfinite(fde) else None,
                "tf_argmax_vs_action_ade_m": action_ade if math.isfinite(action_ade) else None,
                "tf_argmax_vs_action_fde_m": action_fde if math.isfinite(action_fde) else None,
                "first16_hybrid_vs_discrete_ade_m": first16_ade if math.isfinite(first16_ade) else None,
                "first16_hybrid_vs_discrete_fde_m": first16_fde if math.isfinite(first16_fde) else None,
                "teacher_cot": teacher_text(sample),
            }
            if args.save_token_sequences:
                sample_row["tf_argmax_token_ids"] = pred_ids[:usable]
                sample_row["target_traj_token_ids"] = target_list[:usable]
            sample_rows.append(sample_row)
        print(json.dumps({"event": "test_b_batch_done", "done": len(sample_rows), "total": len(rows)}), flush=True)

    def bucket_summary(prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, bucket in buckets.items():
            if name.startswith(prefix):
                out[name[len(prefix) :]] = bucket.summary()
        return dict(sorted(out.items(), key=lambda item: item[0]))

    summary = {
        "checkpoint_name": args.checkpoint_name or args.checkpoint_dir.name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "split": args.split,
        "num_samples": len(sample_rows),
        "batch_size": int(args.batch_size),
        "input_contract": {
            "image_prompt_style": args.image_prompt_style,
            "prompt_text_style": args.prompt_text_style,
            "fuse_history_tokens": bool(args.fuse_history_tokens),
        },
        "token_mapping": {
            "index_base": "0-indexed within the 128 future-token body",
            "even_indices": "accel",
            "odd_indices": "curv",
        },
        "malformed_count": int(malformed),
        "malformed_rate": ratio(malformed, len(sample_rows)),
        "overall": buckets["all"].summary(),
        "horizon": {name: buckets[name].summary() for name in RANGES},
        "accel_curv": {
            "accel_even": buckets["accel_even"].summary(),
            "curv_odd": buckets["curv_odd"].summary(),
        },
        "horizon_by_accel_curv": {
            name: {
                "accel_even": buckets[f"{name}__accel_even"].summary(),
                "curv_odd": buckets[f"{name}__curv_odd"].summary(),
            }
            for name in RANGES
        },
        "pair_accuracy": {name: mean(values) for name, values in pair_values.items()},
        "geometry": {name: mean(values) for name, values in geometry.items() if "::" not in name},
        "scene_buckets": bucket_summary("scene::"),
        "failure_buckets": bucket_summary("failure::"),
        "scene_geometry": {
            key: mean(values)
            for key, values in sorted(geometry.items())
            if key.startswith("scene::")
        },
        "failure_geometry": {
            key: mean(values)
            for key, values in sorted(geometry.items())
            if key.startswith("failure::")
        },
        "samples_jsonl": str(args.samples_jsonl),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.samples_jsonl.open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"scene_buckets", "failure_buckets", "scene_geometry", "failure_geometry"}}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
