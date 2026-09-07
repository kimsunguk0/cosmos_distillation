#!/usr/bin/env python3
"""Evaluate Alpamayo-R1 VLM discrete trajectory-token decode against GT.

This intentionally bypasses Alpamayo-R1's continuous action expert/diffusion
path. It lets the VLM naturally emit future trajectory tokens, extracts those
tokens with the model's own tokenizer contract, decodes with the model's own
DiscreteTrajectoryTokenizer, and computes the same horizon metrics as the pilot
continuous-R1 eval.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.models.token_utils import extract_text_tokens, extract_traj_tokens


CORPUS_JSONL = Path(
    "/home/pm97/workspace/sukim/distillation/cosmos_distillation/data/corpus/"
    "val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl"
)
DEFAULT_OUT_JSON = Path(
    "/home/pm97/workspace/sukim/distillation/cosmos_distillation/outputs/reports/"
    "r1_discrete_backbone_20260728/summary.json"
)
MIN_T0_US = 4_800_000
NUM_WAYPOINTS = 64
WAYPOINT_DT_SEC = 0.1
METRIC_NAMES = ("ADE_le2s", "ADE_gt2s", "ADE_full", "FDE_2s", "FDE_6p4s")
MAX_EXAMPLES = 3
MAX_DUMP_TOKENS = 160


@dataclass(frozen=True)
class TrajSpanDiagnostics:
    traj_start_emitted: bool
    traj_end_emitted: bool
    start_position: int | None
    end_position: int | None
    span_token_count: int
    valid_traj_token_count: int
    invalid_span_token_count: int
    config_traj_token_count: int
    decoder_out_of_range_count: int
    non_traj_token_count: int
    first_invalid_position: int | None
    first_invalid_token_id: int | None
    exact_expected_count: bool
    full_valid_block: bool


class StopAfterTrajFutureBlock(StoppingCriteria):
    """Stop after a natural future-trajectory span reaches the configured size.

    The criterion does not constrain logits. It only watches generated tokens and
    stops once each row either emits <|traj_future_end|> after
    <|traj_future_start|> or emits the configured number of post-start tokens.
    """

    def __init__(
        self,
        *,
        prompt_lengths: list[int],
        traj_start_id: int,
        traj_end_id: int,
        tokens_per_future_traj: int,
    ) -> None:
        self.prompt_lengths = [int(v) for v in prompt_lengths]
        self.traj_start_id = int(traj_start_id)
        self.traj_end_id = int(traj_end_id)
        self.tokens_per_future_traj = int(tokens_per_future_traj)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> bool:
        for row_index in range(int(input_ids.shape[0])):
            prompt_length = self.prompt_lengths[min(row_index, len(self.prompt_lengths) - 1)]
            generated = input_ids[row_index, prompt_length:].tolist()
            try:
                start_pos = len(generated) - 1 - generated[::-1].index(self.traj_start_id)
            except ValueError:
                return False
            after_start = generated[start_pos + 1 :]
            if self.traj_end_id in after_start:
                continue
            if len(after_start) < self.tokens_per_future_traj:
                return False
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpamayo-R1 discrete VLM-backbone trajectory-token ADE/FDE eval."
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--corpus-jsonl", type=Path, default=CORPUS_JSONL)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Hard generation cap. The custom stopping criterion normally stops "
            "after <|traj_future_start|> plus config.tokens_per_future_traj tokens."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_sample_id(sample_id: str) -> tuple[str, int]:
    clip_id = sample_id.split("__", 1)[0]
    t0_us = int(sample_id.split("t0_", 1)[1])
    return clip_id, t0_us


def load_selected_samples(corpus_jsonl: Path, num_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with corpus_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            clip_id, t0_us = parse_sample_id(sample_id)
            if t0_us < MIN_T0_US:
                continue
            selected.append(
                {
                    "sample_id": sample_id,
                    "clip_id": clip_id,
                    "t0_us": t0_us,
                }
            )
            if len(selected) >= num_samples:
                break
    return selected


def load_model_and_processor() -> tuple[Any, Any]:
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",
        # load_in_4bit=True removed to test original precision
    )
    processor = helper.get_processor(model.tokenizer)
    return model, processor


def build_model_inputs(processor: Any, data: dict[str, Any], device: str) -> dict[str, Any]:
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    model_inputs = helper.to_device(model_inputs, device)
    return model_inputs


def seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def finite_float(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric value: {value}")
    return value


def get_required_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping:
        raise KeyError(f"missing special token id for {key!r}")
    value = int(mapping[key])
    if value < 0:
        raise ValueError(f"invalid special token id for {key!r}: {value}")
    return value


def model_support_report(model: Any) -> dict[str, Any]:
    traj_tokenizer = getattr(model, "traj_tokenizer", None)
    action_space = getattr(traj_tokenizer, "action_space", None) if traj_tokenizer is not None else None
    action_dims = tuple(int(v) for v in action_space.get_action_space_dims()) if action_space else None
    expected_flat_tokens = int(np.prod(action_dims)) if action_dims else None
    tokens_per_future_traj = int(getattr(model.config, "tokens_per_future_traj", -1))
    decoder_vocab_size = int(getattr(traj_tokenizer, "vocab_size", -1)) if traj_tokenizer else None
    config_traj_vocab_size = int(getattr(model.config, "traj_vocab_size", -1))
    future_token_start_idx = int(getattr(model, "future_token_start_idx", -1))
    traj_token_start_idx = int(getattr(model.config, "traj_token_start_idx", -1))
    target = str((getattr(model.config, "traj_tokenizer_cfg", {}) or {}).get("_target_", ""))
    issues: list[str] = []

    if traj_tokenizer is None:
        issues.append("model.traj_tokenizer is None")
    if not hasattr(traj_tokenizer, "decode"):
        issues.append("model.traj_tokenizer has no decode(...) method")
    if action_dims is None:
        issues.append("traj_tokenizer.action_space.get_action_space_dims() unavailable")
    elif expected_flat_tokens != tokens_per_future_traj:
        issues.append(
            "tokens_per_future_traj does not match flattened action-space dimensions: "
            f"{tokens_per_future_traj} vs {expected_flat_tokens}"
        )
    if decoder_vocab_size is None or decoder_vocab_size <= 0:
        issues.append(f"invalid traj_tokenizer vocab_size: {decoder_vocab_size}")
    if config_traj_vocab_size <= 0:
        issues.append(f"invalid config.traj_vocab_size: {config_traj_vocab_size}")
    if (
        decoder_vocab_size is not None
        and decoder_vocab_size > 0
        and config_traj_vocab_size > 0
        and decoder_vocab_size > config_traj_vocab_size
    ):
        issues.append(
            "traj_tokenizer vocab_size exceeds config.traj_vocab_size: "
            f"{decoder_vocab_size} vs {config_traj_vocab_size}"
        )
    if future_token_start_idx < 0 or traj_token_start_idx < 0:
        issues.append(
            f"invalid future/token start ids: future={future_token_start_idx} "
            f"config={traj_token_start_idx}"
        )
    if future_token_start_idx != traj_token_start_idx:
        issues.append(
            "model.future_token_start_idx differs from config.traj_token_start_idx: "
            f"{future_token_start_idx} vs {traj_token_start_idx}"
        )
    for key in ("traj_future_start", "traj_future_end"):
        try:
            get_required_int(model.special_token_ids, key)
        except Exception as exc:  # noqa: BLE001
            issues.append(str(exc))

    return {
        "supported_for_decode_attempt": not issues,
        "issues": issues,
        "traj_tokenizer_class": type(traj_tokenizer).__name__ if traj_tokenizer is not None else None,
        "traj_tokenizer_config_target": target or None,
        "action_space_class": type(action_space).__name__ if action_space is not None else None,
        "action_space_dims": list(action_dims) if action_dims else None,
        "expected_flat_tokens_from_action_space": expected_flat_tokens,
        "tokens_per_future_traj": tokens_per_future_traj,
        "num_waypoints_expected_by_eval": NUM_WAYPOINTS,
        "decoder_vocab_size": decoder_vocab_size,
        "config_traj_vocab_size": config_traj_vocab_size,
        "future_token_start_idx": future_token_start_idx,
        "traj_token_start_idx": traj_token_start_idx,
        "traj_future_start_id": (
            int(model.special_token_ids["traj_future_start"])
            if "traj_future_start" in model.special_token_ids
            else None
        ),
        "traj_future_end_id": (
            int(model.special_token_ids["traj_future_end"])
            if "traj_future_end" in model.special_token_ids
            else None
        ),
        "notes": [
            "The shipped continuous R1 path masks config.traj_token_start_idx..+traj_vocab_size "
            "with ExpertLogitsProcessor and stops at <|traj_future_start|> before diffusion.",
            "This script deliberately omits that logits processor and only reports metrics for "
            "naturally emitted spans that are exactly decode-valid.",
        ],
    }


def scan_traj_span(
    generated_token_ids: list[int],
    *,
    traj_start_id: int,
    traj_end_id: int,
    future_token_start_idx: int,
    decoder_vocab_size: int,
    config_traj_vocab_size: int,
    tokens_per_future_traj: int,
) -> TrajSpanDiagnostics:
    start_positions = [i for i, token_id in enumerate(generated_token_ids) if token_id == traj_start_id]
    if not start_positions:
        return TrajSpanDiagnostics(
            traj_start_emitted=False,
            traj_end_emitted=False,
            start_position=None,
            end_position=None,
            span_token_count=0,
            valid_traj_token_count=0,
            invalid_span_token_count=0,
            config_traj_token_count=0,
            decoder_out_of_range_count=0,
            non_traj_token_count=0,
            first_invalid_position=None,
            first_invalid_token_id=None,
            exact_expected_count=False,
            full_valid_block=False,
        )

    start_position = start_positions[-1]
    after_start = generated_token_ids[start_position + 1 :]
    end_offset = next((i for i, token_id in enumerate(after_start) if token_id == traj_end_id), None)
    end_position = start_position + 1 + end_offset if end_offset is not None else None
    span = after_start[:end_offset] if end_offset is not None else after_start

    decoder_low = int(future_token_start_idx)
    decoder_high = decoder_low + int(decoder_vocab_size)
    config_high = decoder_low + int(config_traj_vocab_size)

    valid_count = 0
    config_count = 0
    decoder_out_of_range_count = 0
    non_traj_count = 0
    first_invalid_position = None
    first_invalid_token_id = None
    for relative_pos, token_id in enumerate(span):
        token_id = int(token_id)
        in_decoder_range = decoder_low <= token_id < decoder_high
        in_config_range = decoder_low <= token_id < config_high
        if in_decoder_range:
            valid_count += 1
        elif in_config_range:
            config_count += 1
            decoder_out_of_range_count += 1
        else:
            non_traj_count += 1
        if not in_decoder_range and first_invalid_position is None:
            first_invalid_position = relative_pos
            first_invalid_token_id = token_id

    span_token_count = len(span)
    invalid_count = span_token_count - valid_count
    exact_expected_count = span_token_count == int(tokens_per_future_traj)
    full_valid_block = exact_expected_count and valid_count == int(tokens_per_future_traj)
    return TrajSpanDiagnostics(
        traj_start_emitted=True,
        traj_end_emitted=end_position is not None,
        start_position=start_position,
        end_position=end_position,
        span_token_count=span_token_count,
        valid_traj_token_count=valid_count,
        invalid_span_token_count=invalid_count,
        config_traj_token_count=valid_count + config_count,
        decoder_out_of_range_count=decoder_out_of_range_count,
        non_traj_token_count=non_traj_count,
        first_invalid_position=first_invalid_position,
        first_invalid_token_id=first_invalid_token_id,
        exact_expected_count=exact_expected_count,
        full_valid_block=full_valid_block,
    )


def diagnostics_to_dict(diag: TrajSpanDiagnostics) -> dict[str, Any]:
    return {
        "traj_start_emitted": diag.traj_start_emitted,
        "traj_end_emitted": diag.traj_end_emitted,
        "traj_start_position": diag.start_position,
        "traj_end_position": diag.end_position,
        "span_token_count": diag.span_token_count,
        "valid_traj_token_count": diag.valid_traj_token_count,
        "invalid_span_token_count": diag.invalid_span_token_count,
        "config_traj_token_count": diag.config_traj_token_count,
        "decoder_out_of_range_count": diag.decoder_out_of_range_count,
        "non_traj_token_count": diag.non_traj_token_count,
        "first_invalid_position": diag.first_invalid_position,
        "first_invalid_token_id": diag.first_invalid_token_id,
        "exact_expected_count": diag.exact_expected_count,
        "full_valid_block": diag.full_valid_block,
    }


def failure_from_diagnostics(
    diag: TrajSpanDiagnostics, *, expected_tokens: int
) -> tuple[str | None, str | None]:
    if not diag.traj_start_emitted:
        return "no_traj_future_start", "VLM did not emit <|traj_future_start|> before max_new_tokens."
    if diag.span_token_count == 0:
        return "empty_traj_span", "No tokens were emitted after <|traj_future_start|>."
    if diag.traj_end_emitted and diag.span_token_count < expected_tokens:
        return (
            "early_traj_future_end",
            f"<|traj_future_end|> appeared after {diag.span_token_count} tokens; expected {expected_tokens}.",
        )
    if diag.span_token_count != expected_tokens:
        return (
            "wrong_traj_span_length",
            f"Trajectory span has {diag.span_token_count} tokens; expected {expected_tokens}.",
        )
    if diag.invalid_span_token_count:
        return (
            "invalid_traj_tokens",
            (
                f"Trajectory span has {diag.invalid_span_token_count} tokens outside the decoder-valid "
                f"future-token range; first invalid token id={diag.first_invalid_token_id}."
            ),
        )
    if not diag.full_valid_block:
        return (
            "invalid_traj_block",
            f"Only {diag.valid_traj_token_count}/{expected_tokens} tokens are decoder-valid.",
        )
    return None, None


def token_id_dump(tokenizer: Any, token_ids: list[int], limit: int = MAX_DUMP_TOKENS) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for token_id in token_ids[:limit]:
        token_id = int(token_id)
        out.append(
            {
                "id": token_id,
                "text": tokenizer.decode([token_id], skip_special_tokens=False),
            }
        )
    return out


def first_text_value(extra: dict[str, Any], key: str) -> str:
    value = extra.get(key, "")
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value)


def decode_generated_text(tokenizer: Any, token_ids: torch.Tensor) -> str:
    return str(tokenizer.decode(token_ids.tolist(), skip_special_tokens=False))


def run_discrete_inference(
    model: Any,
    processor: Any,
    data: dict[str, Any],
    device: str,
    *,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    model_inputs = build_model_inputs(processor, data, device)
    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    prompt_len = int(input_ids.shape[1])

    traj_start_id = get_required_int(model.special_token_ids, "traj_future_start")
    traj_end_id = get_required_int(model.special_token_ids, "traj_future_end")
    expected_tokens = int(model.config.tokens_per_future_traj)
    future_token_start_idx = int(model.future_token_start_idx)
    decoder_vocab_size = int(model.traj_tokenizer.vocab_size)
    config_traj_vocab_size = int(model.config.traj_vocab_size)

    traj_data_vlm = {
        "ego_history_xyz": model_inputs["ego_history_xyz"],
        "ego_history_rot": model_inputs["ego_history_rot"],
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.do_sample = False
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    stopping_criteria = StoppingCriteriaList(
        [
            StopAfterTrajFutureBlock(
                prompt_lengths=[prompt_len],
                traj_start_id=traj_start_id,
                traj_end_id=traj_end_id,
                tokens_per_future_traj=expected_tokens,
            )
        ]
    )

    seed_everything(seed)
    device_type = torch.device(device).type
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"
    ):
        generated = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            output_logits=False,
            **tokenized_data,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if not hasattr(generated, "sequences"):
        raise RuntimeError("generate(..., return_dict_in_generate=True) did not return .sequences")
    sequences = generated.sequences
    if int(sequences.shape[0]) != 1:
        raise RuntimeError(f"Expected one generated sequence, got shape={tuple(sequences.shape)}")
    generated_new = sequences[:, prompt_len:]
    generated_token_ids = [int(v) for v in generated_new[0].detach().cpu().tolist()]
    diag = scan_traj_span(
        generated_token_ids,
        traj_start_id=traj_start_id,
        traj_end_id=traj_end_id,
        future_token_start_idx=future_token_start_idx,
        decoder_vocab_size=decoder_vocab_size,
        config_traj_vocab_size=config_traj_vocab_size,
        tokens_per_future_traj=expected_tokens,
    )

    text_extra = extract_text_tokens(model.tokenizer, sequences)
    generated_text = decode_generated_text(model.tokenizer, generated_new[0].detach().cpu())
    extracted_traj_tokens = None
    if diag.traj_start_emitted:
        extracted = extract_traj_tokens(
            generated_new,
            model.special_token_ids,
            expected_tokens,
            future_token_start_idx,
            decoder_vocab_size,
        )
        extracted_traj_tokens = extracted.detach().cpu().numpy().astype(np.int64).reshape(-1).tolist()

    failure_type, failure_reason = failure_from_diagnostics(diag, expected_tokens=expected_tokens)
    pred_xyz_64 = None
    pred_rot = None
    if failure_type is None:
        traj_tokens = torch.as_tensor(
            extracted_traj_tokens,
            dtype=torch.long,
            device=model_inputs["ego_history_xyz"].device,
        ).reshape(1, expected_tokens)
        hist_xyz = model_inputs["ego_history_xyz"][:, -1]
        hist_rot = model_inputs["ego_history_rot"][:, -1]
        with torch.inference_mode():
            pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(
                hist_xyz=hist_xyz,
                hist_rot=hist_rot,
                tokens=traj_tokens,
            )
        pred_np = as_numpy(pred_xyz).astype(np.float32)
        pred_xyz_64 = np.squeeze(pred_np)
        if pred_xyz_64.ndim != 2 or pred_xyz_64.shape[0] < NUM_WAYPOINTS or pred_xyz_64.shape[1] < 3:
            raise ValueError(
                f"Decoded pred_xyz must be [>=64, >=3], got shape={tuple(pred_xyz_64.shape)}"
            )
        pred_xyz_64 = pred_xyz_64[:NUM_WAYPOINTS, :3]
        if not np.isfinite(pred_xyz_64).all():
            raise ValueError("Decoded pred_xyz contains non-finite values.")

    return {
        "elapsed_ms": float(elapsed_ms),
        "generated_new_tokens": int(generated_new.shape[1]),
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "cot": first_text_value(text_extra, "cot"),
        "meta_action": first_text_value(text_extra, "meta_action"),
        "trajectory_span": diagnostics_to_dict(diag),
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "extracted_traj_tokens": extracted_traj_tokens,
        "pred_xyz": pred_xyz_64,
        "pred_rot_shape": list(pred_rot.shape) if pred_rot is not None else None,
    }


def compute_horizon_metrics(pred_xyz: Any, ego_future_xyz: Any) -> dict[str, Any]:
    pred_traj = np.asarray(pred_xyz, dtype=np.float32)
    gt_np = as_numpy(ego_future_xyz).astype(np.float32)
    gt_traj = np.squeeze(gt_np)
    if gt_traj.ndim != 2:
        gt_traj = gt_traj.reshape(-1, gt_traj.shape[-1])

    if pred_traj.shape[0] < NUM_WAYPOINTS or gt_traj.shape[0] < NUM_WAYPOINTS:
        raise ValueError(
            f"expected at least {NUM_WAYPOINTS} waypoints, got "
            f"pred={pred_traj.shape[0]} gt={gt_traj.shape[0]}"
        )
    if pred_traj.shape[-1] < 3 or gt_traj.shape[-1] < 3:
        raise ValueError(
            f"3D metrics require xyz dims, got pred={pred_traj.shape[-1]} gt={gt_traj.shape[-1]}"
        )

    pred_traj = pred_traj[:NUM_WAYPOINTS, :3]
    gt_traj = gt_traj[:NUM_WAYPOINTS, :3]
    dist = np.linalg.norm(pred_traj - gt_traj, axis=-1)
    return {
        "metric_dimensionality": "3D",
        "ADE_le2s": finite_float(dist[:20].mean()),
        "ADE_gt2s": finite_float(dist[20:NUM_WAYPOINTS].mean()),
        "ADE_full": finite_float(dist[:NUM_WAYPOINTS].mean()),
        "FDE_2s": finite_float(dist[19]),
        "FDE_6p4s": finite_float(dist[63]),
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    metric_rows = [row for row in rows if row.get("success")]
    if not metric_rows:
        return {name: None for name in METRIC_NAMES}
    return {
        name: float(np.mean([float(row[name]) for row in metric_rows]))
        for name in METRIC_NAMES
    }


def build_emission_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_rows = len(rows)
    if n_rows == 0:
        return {
            "traj_start_hit_rate": None,
            "traj_end_hit_rate": None,
            "full_valid_block_rate": None,
            "mean_valid_traj_token_count": None,
        }
    spans = [row.get("trajectory_span") or {} for row in rows]
    start_hits = sum(1 for span in spans if span.get("traj_start_emitted"))
    end_hits = sum(1 for span in spans if span.get("traj_end_emitted"))
    full_valid = sum(1 for span in spans if span.get("full_valid_block"))
    valid_counts = [int(span.get("valid_traj_token_count") or 0) for span in spans]
    expected_counts = sorted(
        {int(row.get("expected_traj_tokens") or 0) for row in rows if row.get("expected_traj_tokens")}
    )
    return {
        "traj_start_hits": int(start_hits),
        "traj_start_hit_rate": float(start_hits / n_rows),
        "traj_end_hits": int(end_hits),
        "traj_end_hit_rate": float(end_hits / n_rows),
        "full_valid_block_hits": int(full_valid),
        "full_valid_block_rate": float(full_valid / n_rows),
        "mean_valid_traj_token_count": float(np.mean(valid_counts)),
        "expected_traj_token_counts": expected_counts,
    }


def compact_sample_dump(model: Any, row: dict[str, Any]) -> dict[str, Any]:
    token_ids = [int(v) for v in row.get("generated_token_ids") or []]
    extracted = row.get("extracted_traj_tokens") or []
    return {
        "sample_id": row["sample_id"],
        "success": bool(row.get("success")),
        "failure_type": row.get("failure_type"),
        "cot": str(row.get("cot") or "")[:2000],
        "meta_action": str(row.get("meta_action") or "")[:1000],
        "generated_text_preview": str(row.get("generated_text") or "")[:3000],
        "generated_token_dump": token_id_dump(model.tokenizer, token_ids),
        "extracted_traj_tokens_head": [int(v) for v in extracted[:32]],
        "extracted_traj_tokens_tail": [int(v) for v in extracted[-32:]],
        "trajectory_span": row.get("trajectory_span"),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("Alpamayo-R1 discrete backbone eval")
    print(
        f"selected={summary['n_selected']} "
        f"success={summary['n_success']} fail={summary['n_fail']}"
    )
    print(
        f"metric_dimensionality={summary['metric_dimensionality']} "
        f"waypoint_dt_sec={summary['waypoint_dt_sec']} "
        f"tokens_per_future_traj={summary['model_discrete_support'].get('tokens_per_future_traj')}"
    )
    print(
        "emission "
        f"traj_start_hit_rate={summary['emission_stats']['traj_start_hit_rate']} "
        f"full_valid_block_rate={summary['emission_stats']['full_valid_block_rate']}"
    )
    print()
    print(f"{'metric':<12} {'mean_m':>12}")
    print(f"{'-' * 12} {'-' * 12}")
    for name in METRIC_NAMES:
        value = summary["aggregates"][name]
        rendered = "n/a" if value is None else f"{value:.6f}"
        print(f"{name:<12} {rendered:>12}")
    if summary["failure_reason_breakdown"]:
        print()
        print("failure_reason_breakdown")
        for reason, count in summary["failure_reason_breakdown"].items():
            print(f"  {reason}: {count}")
    print()
    print(f"Wrote JSON: {summary['out_json']}")


def make_failure_row(sample: dict[str, Any], failure_type: str, message: str) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "clip_id": sample["clip_id"],
        "t0_us": sample["t0_us"],
        "success": False,
        "failure_type": failure_type,
        "failure_reason": message,
    }


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    samples = load_selected_samples(args.corpus_jsonl, args.num_samples)
    model, processor = load_model_and_processor()
    support = model_support_report(model)

    per_sample_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()

    for index, sample in enumerate(samples):
        seed = int(args.seed) + index
        try:
            if not support["supported_for_decode_attempt"]:
                raise RuntimeError("unsupported_discrete_decode: " + "; ".join(support["issues"]))
            data = load_physical_aiavdataset(sample["clip_id"], t0_us=sample["t0_us"])
            inference = run_discrete_inference(
                model,
                processor,
                data,
                args.device,
                max_new_tokens=int(args.max_new_tokens),
                seed=seed,
            )
            row = {
                "sample_id": sample["sample_id"],
                "clip_id": sample["clip_id"],
                "t0_us": sample["t0_us"],
                "success": False,
                "seed": seed,
                "expected_traj_tokens": int(model.config.tokens_per_future_traj),
                "generated_new_tokens": inference["generated_new_tokens"],
                "elapsed_ms": inference["elapsed_ms"],
                "trajectory_span": inference["trajectory_span"],
                "failure_type": inference["failure_type"],
                "failure_reason": inference["failure_reason"],
                "cot": inference["cot"],
                "meta_action": inference["meta_action"],
                "generated_text": inference["generated_text"],
                "generated_token_ids": inference["generated_token_ids"],
                "extracted_traj_tokens": inference["extracted_traj_tokens"],
            }
            if inference["failure_type"] is not None:
                per_sample_rows.append(row)
                print(
                    f"[FAIL] {sample['sample_id']}: "
                    f"{inference['failure_type']}: {inference['failure_reason']}",
                    flush=True,
                )
                continue

            metrics = compute_horizon_metrics(inference["pred_xyz"], data["ego_future_xyz"])
            pred_xyz = np.asarray(inference["pred_xyz"], dtype=np.float32)
            row.update(
                {
                    "success": True,
                    "failure_type": None,
                    "failure_reason": None,
                    "pred_xyz_shape": list(pred_xyz.shape),
                    "pred_xyz": pred_xyz.tolist(),
                    **metrics,
                }
            )
            per_sample_rows.append(row)
            print(
                f"[OK] {sample['sample_id']}: ADE_full={row['ADE_full']:.6f} "
                f"FDE_6p4s={row['FDE_6p4s']:.6f} "
                f"valid_tokens={row['trajectory_span']['valid_traj_token_count']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {sample['sample_id']}: {message}", flush=True)
            per_sample_rows.append(make_failure_row(sample, type(exc).__name__, message))

    success_rows = [row for row in per_sample_rows if row.get("success")]
    failure_rows = [row for row in per_sample_rows if not row.get("success")]
    dimensionalities = sorted({row["metric_dimensionality"] for row in success_rows})
    metric_dimensionality = (
        dimensionalities[0] if len(dimensionalities) == 1 else "mixed" if dimensionalities else "unavailable"
    )
    failure_reason_breakdown = Counter(str(row.get("failure_type") or "unknown") for row in failure_rows)
    failure_reasons = [
        {
            "sample_id": row["sample_id"],
            "clip_id": row["clip_id"],
            "t0_us": row["t0_us"],
            "failure_type": row.get("failure_type"),
            "reason": row.get("failure_reason"),
        }
        for row in failure_rows
    ]
    examples = [
        compact_sample_dump(model, row)
        for row in per_sample_rows[:MAX_EXAMPLES]
    ]

    summary = {
        "model_key": "nvidia/Alpamayo-R1-10B_discrete_backbone",
        "model_label": "Alpamayo-R1 VLM discrete trajectory-token decode, no ExpertLogitsProcessor, no diffusion expert",
        "corpus_jsonl": str(args.corpus_jsonl),
        "min_t0_us": MIN_T0_US,
        "num_requested": int(args.num_samples),
        "n_selected": len(samples),
        "n_success": len(success_rows),
        "n_fail": len(failure_rows),
        "waypoint_dt_sec": WAYPOINT_DT_SEC,
        "num_waypoints": NUM_WAYPOINTS,
        "metric_dimensionality": metric_dimensionality,
        "metric_dimensionalities": dimensionalities,
        "generation": {
            "do_sample": False,
            "seed": int(args.seed),
            "max_new_tokens": int(args.max_new_tokens),
            "logits_processor": None,
            "stopping_criteria": "StopAfterTrajFutureBlock",
            "stops_after": (
                "natural <|traj_future_start|> plus config.tokens_per_future_traj generated "
                "post-start tokens, or earlier natural <|traj_future_end|>"
            ),
        },
        "model_discrete_support": support,
        "aggregates": aggregate_metrics(per_sample_rows),
        "emission_stats": build_emission_stats(per_sample_rows),
        "failure_reason_breakdown": dict(sorted(failure_reason_breakdown.items())),
        "failure_reasons": failure_reasons,
        "per_sample": per_sample_rows,
        "examples": examples,
        "elapsed_sec": round(time.perf_counter() - started_all, 3),
        "out_json": str(args.out_json),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print_summary(summary)


if __name__ == "__main__":
    main()
