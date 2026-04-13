#!/usr/bin/env python3
"""Run Alpamayo teacher text inference for ready canonical samples."""

from __future__ import annotations

import argparse
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

from src.data.teacher_cache import (
    CAMERA_NAME_TO_INDEX,
    build_hallucination_flags,
    extract_json_object,
    load_jsonl_by_key,
    normalize_teacher_action_class,
    selection_score_from_outputs,
    summarize_json_candidate,
    write_jsonl,
)


DEFAULT_ALPAMAYO_SRC = Path("/home/pm97/workspace/sukim/alpamayo1.5/src")
DEFAULT_ALPAMAYO_MODEL = Path("/home/pm97/workspace/sukim/weights/alpamayo15_vlm_weights")
ALLOWED_META_ACTIONS = {
    "keep_lane",
    "yield",
    "stop",
    "slow_down",
    "creep",
    "left_turn",
    "right_turn",
    "change_lane_left",
    "change_lane_right",
    "follow_lead",
    "overtake",
    "nudge_left",
    "nudge_right",
    "unknown",
}


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
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=("eager", "flash_attention_2"),
    )
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-generation-length", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_text_inference_summary.json",
    )
    return parser.parse_args()


def load_index_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def select_ready_records(records: list[dict[str, Any]], overwrite: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        status = str(record.get("status", ""))
        has_output = bool((record.get("output") or {}).get("teacher_short_reason"))
        if status == "ready_request_bundle":
            selected.append(record)
        elif overwrite and status == "ok" and has_output:
            selected.append(record)
    return selected


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


def history_placeholder_text(*, num_traj_token: int = 48) -> str:
    """Mirror the Alpamayo history placeholder expected by the teacher model."""
    return f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"


def build_chat_messages(
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    *,
    helper_module,
    user_text: str,
    assistant_prefix: str,
) -> list[dict[str, Any]]:
    """Construct a chat payload while keeping the Alpamayo image layout intact."""
    image_content = helper_module._build_image_content(
        image_frames.flatten(0, 1),
        camera_indices,
        4,
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": user_text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_prefix}],
        },
    ]


def build_cot_inputs(
    sample_dir: Path,
    *,
    helper_module,
    processor,
    prompt_text: str,
) -> dict[str, Any]:
    image_frames, camera_indices = load_image_frames(sample_dir)
    ego_history_xyz = torch.from_numpy(np.load(sample_dir / "ego_history_xyz.npy")).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(np.load(sample_dir / "ego_history_rot.npy")).float().unsqueeze(0).unsqueeze(0)

    user_text = f"{history_placeholder_text()}{prompt_text.strip()}"
    messages = build_chat_messages(
        image_frames,
        camera_indices,
        helper_module=helper_module,
        user_text=user_text,
        assistant_prefix="<|cot_start|>",
    )
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        "tokenized_data": tokenized,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def build_vqa_inputs(
    sample_dir: Path,
    *,
    helper_module,
    processor,
    prompt_text: str,
) -> dict[str, Any]:
    image_frames, camera_indices = load_image_frames(sample_dir)
    messages = build_chat_messages(
        image_frames,
        camera_indices,
        helper_module=helper_module,
        user_text=prompt_text.strip(),
        assistant_prefix="<|answer_start|>",
    )
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {"tokenized_data": tokenized}


def build_tagged_triplet_inputs(
    sample_dir: Path,
    *,
    helper_module,
    processor,
    prompt_text: str,
) -> dict[str, Any]:
    """Prompt the model to emit cot/meta_action/answer using explicit special tokens."""
    image_frames, camera_indices = load_image_frames(sample_dir)
    ego_history_xyz = torch.from_numpy(np.load(sample_dir / "ego_history_xyz.npy")).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(np.load(sample_dir / "ego_history_rot.npy")).float().unsqueeze(0).unsqueeze(0)

    user_text = f"{history_placeholder_text()}{prompt_text.strip()}"
    messages = build_chat_messages(
        image_frames,
        camera_indices,
        helper_module=helper_module,
        user_text=user_text,
        assistant_prefix="<|cot_start|>",
    )
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        "tokenized_data": tokenized,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def build_slot_only_inputs(
    sample_dir: Path,
    *,
    helper_module,
    processor,
    prompt_text: str,
    assistant_prefix: str,
) -> dict[str, Any]:
    """Build a single-slot text generation input for answer/meta-action extraction."""
    image_frames, camera_indices = load_image_frames(sample_dir)
    ego_history_xyz = torch.from_numpy(np.load(sample_dir / "ego_history_xyz.npy")).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(np.load(sample_dir / "ego_history_rot.npy")).float().unsqueeze(0).unsqueeze(0)

    user_text = f"{history_placeholder_text()}{prompt_text.strip()}"
    messages = build_chat_messages(
        image_frames,
        camera_indices,
        helper_module=helper_module,
        user_text=user_text,
        assistant_prefix=assistant_prefix,
    )
    tokenized = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        "tokenized_data": tokenized,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


def first_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        return first_text(value[0]) if value else ""
    if value is None:
        return ""
    return str(value).strip()


def is_natural_language(text: str | None) -> bool:
    """Reject numeric or token-only answers that are not useful text targets."""
    if not text:
        return False
    stripped = str(text).strip()
    return any(char.isalpha() for char in stripped)


def cleaned_text(value: Any) -> str | None:
    """Return a stripped string or None when the value is empty."""
    text = first_text(value)
    return text or None


def normalize_meta_action_label(text: str | None) -> str | None:
    """Normalize a short action label when the model emits one directly."""
    if not text:
        return None
    normalized = (
        str(text)
        .strip()
        .lower()
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .strip(".,;:!?")
    )
    aliases = {
        "lane_keep": "keep_lane",
        "keep_lane": "keep_lane",
        "keep": "keep_lane",
        "slow": "slow_down",
        "slow_down": "slow_down",
        "change_left": "change_lane_left",
        "lane_change_left": "change_lane_left",
        "change_lane_left": "change_lane_left",
        "change_right": "change_lane_right",
        "lane_change_right": "change_lane_right",
        "change_lane_right": "change_lane_right",
        "follow": "follow_lead",
        "follow_lead": "follow_lead",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ALLOWED_META_ACTIONS else None


def first_usable_text(
    candidates: list[tuple[Any, str, bool]],
    *,
    require_natural_language: bool = False,
) -> tuple[str | None, str, bool]:
    """Choose the first non-empty text candidate plus provenance metadata."""
    for value, source, direct in candidates:
        text = cleaned_text(value)
        if not text:
            continue
        if require_natural_language and not is_natural_language(text):
            continue
        return text, source, direct
    return None, "missing", False


def first_meta_action_label(
    candidates: list[tuple[Any, str, bool]],
    *,
    fallback_label: str | None,
) -> tuple[str | None, str, bool]:
    """Choose a normalized meta-action label with provenance metadata."""
    for value, source, direct in candidates:
        label = normalize_meta_action_label(cleaned_text(value))
        if label:
            return label, source, direct
    if fallback_label and fallback_label != "unknown":
        return fallback_label, "normalized_label_from_text", False
    return None, "missing", False


def build_slot_channel_behavior(prompt_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Describe whether slot prompts collapsed back into the cot channel."""
    short_reason_prompt = prompt_outputs.get("short_reason_only_v2", {})
    meta_action_prompt = prompt_outputs.get("meta_action_only_v2", {})
    answer_prompt = prompt_outputs.get("answer_only_v2", {})

    short_reason_collapse = bool(cleaned_text(short_reason_prompt.get("cot"))) and not bool(
        cleaned_text(short_reason_prompt.get("answer"))
    )
    meta_action_collapse = bool(cleaned_text(meta_action_prompt.get("cot"))) and not bool(
        cleaned_text(meta_action_prompt.get("meta_action"))
    )
    answer_collapse = bool(cleaned_text(answer_prompt.get("cot"))) and not bool(
        cleaned_text(answer_prompt.get("answer"))
    )

    cot_nonempty = any(
        cleaned_text(prompt.get("cot"))
        for prompt in (short_reason_prompt, meta_action_prompt, answer_prompt)
    )
    meta_action_nonempty = any(
        cleaned_text(prompt.get("meta_action"))
        for prompt in (short_reason_prompt, meta_action_prompt, answer_prompt)
    )
    answer_nonempty = any(
        cleaned_text(prompt.get("answer"))
        for prompt in (short_reason_prompt, meta_action_prompt, answer_prompt)
    )

    all_text_from_same_channel = cot_nonempty and not meta_action_nonempty and not answer_nonempty
    return {
        "cot_nonempty": bool(cot_nonempty),
        "meta_action_nonempty": bool(meta_action_nonempty),
        "answer_nonempty": bool(answer_nonempty),
        "short_reason_prompt_collapsed_to_cot": short_reason_collapse,
        "meta_action_prompt_collapsed_to_cot": meta_action_collapse,
        "answer_prompt_collapsed_to_cot": answer_collapse,
        "channel_collapse_detected": bool(
            short_reason_collapse or meta_action_collapse or answer_collapse or all_text_from_same_channel
        ),
        "all_text_from_same_channel": bool(all_text_from_same_channel),
    }


def build_quality_annotations(
    *,
    long_cot: str | None,
    short_reason: str | None,
    meta_action: str | None,
    answer: str | None,
    long_cot_direct: bool,
    short_reason_direct: bool,
    meta_action_direct: bool,
    answer_direct: bool,
    slot_channel_behavior: dict[str, Any],
) -> dict[str, Any]:
    """Compute teacher-quality signals used for downweighting and analysis."""
    overlap = bool(short_reason and answer and short_reason.strip().lower() == answer.strip().lower())

    if long_cot and short_reason and answer:
        text_quality = "accept"
    elif long_cot or short_reason or answer:
        text_quality = "downweight"
    else:
        text_quality = "reject"

    if meta_action and answer and meta_action_direct and answer_direct and not slot_channel_behavior["channel_collapse_detected"]:
        structured_quality = "accept"
    elif meta_action or answer:
        structured_quality = "downweight"
    else:
        structured_quality = "reject"

    if meta_action_direct and answer_direct and not slot_channel_behavior["channel_collapse_detected"]:
        direct_slot_reliability = "high"
    elif (meta_action_direct or answer_direct) and not slot_channel_behavior["channel_collapse_detected"]:
        direct_slot_reliability = "medium"
    else:
        direct_slot_reliability = "low"

    quality_multiplier = 1.0
    if meta_action and not meta_action_direct:
        quality_multiplier *= 0.85
    if answer and not answer_direct:
        quality_multiplier *= 0.9
    if slot_channel_behavior["channel_collapse_detected"]:
        quality_multiplier *= 0.9
    if overlap:
        quality_multiplier *= 0.95

    return {
        "teacher_text_quality": text_quality,
        "teacher_structured_quality": structured_quality,
        "teacher_direct_slot_reliability": direct_slot_reliability,
        "teacher_answer_short_reason_overlap": overlap,
        "teacher_quality_multiplier": round(quality_multiplier, 4),
    }


def has_tagged_fields(result: dict[str, Any] | None) -> bool:
    """Return True when the prompt result includes usable slot values."""
    if not result:
        return False
    meta_action = first_text(result.get("meta_action"))
    answer = first_text(result.get("answer"))
    cot = first_text(result.get("cot"))
    return bool(cot) and bool(meta_action) and is_natural_language(answer)


def run_generation(model, model_inputs: dict[str, Any], helper_module, args: argparse.Namespace) -> dict[str, Any]:
    model_inputs = helper_module.to_device(model_inputs, args.device)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        extra = model.generate_text(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_samples=1,
            max_generation_length=args.max_generation_length,
        )
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in extra.items()}


def build_prompt_outputs(
    sample_dir: Path,
    prompt_families: list[dict[str, str]],
    *,
    helper_module,
    processor,
    model,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for prompt_family in prompt_families:
        name = str(prompt_family["name"])
        prompt_text = str(prompt_family["prompt"])
        if name == "long_cot_v1":
            model_inputs = build_cot_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
            )
        elif name.startswith("tagged_triplet_"):
            model_inputs = build_tagged_triplet_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
            )
        elif name == "short_reason_only_v2":
            model_inputs = build_slot_only_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                assistant_prefix="<|answer_start|>",
            )
        elif name == "meta_action_only_v2":
            model_inputs = build_slot_only_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                assistant_prefix="<|cot_start|>",
            )
        elif name == "answer_only_v2":
            model_inputs = build_slot_only_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                assistant_prefix="<|answer_start|>",
            )
        else:
            model_inputs = build_vqa_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
            )
        raw_extra = run_generation(model, model_inputs, helper_module, args)
        answer_text = first_text(raw_extra.get("answer"))
        json_payload = extract_json_object(answer_text)
        results[name] = {
            "prompt": prompt_text,
            "raw_extra": raw_extra,
            "cot": first_text(raw_extra.get("cot")),
            "meta_action": first_text(raw_extra.get("meta_action")),
            "answer": answer_text,
            "json_payload": json_payload,
            "json_summary": summarize_json_candidate(json_payload),
        }
    return results


def select_structured_candidate(prompt_outputs: dict[str, dict[str, Any]], long_cot: str | None) -> tuple[str | None, dict[str, Any] | None, float]:
    """Choose the best structured prompt response by parse quality and completeness."""
    best_name = None
    best_payload = None
    best_score = -1.0
    for name, result in prompt_outputs.items():
        if name == "long_cot_v1":
            continue
        payload = result.get("json_payload")
        candidate_score = selection_score_from_outputs(
            long_cot=long_cot,
            json_payload=payload,
            meta_action=(payload or {}).get("meta_action") or result.get("meta_action"),
            answer=(payload or {}).get("answer") or (payload or {}).get("final_answer") or result.get("answer"),
        )
        if candidate_score > best_score:
            best_name = name
            best_payload = payload
            best_score = candidate_score
    return best_name, best_payload, max(best_score, 0.0)


def normalize_teacher_output(prompt_outputs: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the final normalized teacher payload plus diagnostics."""
    long_cot_prompt = prompt_outputs.get("long_cot_v1", {})
    long_cot, long_cot_source, long_cot_direct = first_usable_text(
        [
            (long_cot_prompt.get("cot"), "direct_long_cot_prompt_cot", True),
            (long_cot_prompt.get("answer"), "fallback_from_long_cot_prompt_answer", False),
        ],
        require_natural_language=True,
    )
    selected_prompt, selected_payload, selection_score = select_structured_candidate(prompt_outputs, long_cot)
    selected_result = prompt_outputs.get(selected_prompt or "", {})
    short_reason_slot = prompt_outputs.get("short_reason_only_v2", {})
    meta_action_slot = prompt_outputs.get("meta_action_only_v2", {})
    answer_slot = prompt_outputs.get("answer_only_v2", {})

    short_reason, short_reason_source, short_reason_direct = first_usable_text(
        [
            (short_reason_slot.get("answer"), "direct_short_reason_prompt_answer", True),
            (short_reason_slot.get("cot"), "normalized_from_short_reason_prompt_cot", False),
            ((selected_payload or {}).get("rationale"), "normalized_from_structured_payload_rationale", False),
            ((selected_payload or {}).get("scene_summary"), "normalized_from_structured_payload_scene_summary", False),
            ((selected_payload or {}).get("final_answer"), "normalized_from_structured_payload_final_answer", False),
            (selected_result.get("answer"), "normalized_from_selected_prompt_answer", False),
            (long_cot, "fallback_from_long_cot", False),
        ],
        require_natural_language=True,
    )

    answer, answer_source, answer_direct = first_usable_text(
        [
            (answer_slot.get("answer"), "direct_answer_prompt_answer", True),
            (answer_slot.get("cot"), "normalized_from_answer_prompt_cot", False),
            ((selected_payload or {}).get("answer"), "normalized_from_structured_payload_answer", False),
            ((selected_payload or {}).get("final_answer"), "normalized_from_structured_payload_final_answer", False),
            (long_cot_prompt.get("answer"), "fallback_from_long_cot_prompt_answer", False),
            (short_reason, "fallback_from_short_reason", False),
            (long_cot, "fallback_from_long_cot", False),
        ],
        require_natural_language=True,
    )

    if short_reason and long_cot and short_reason.strip() == long_cot.strip() and answer and answer.strip() != long_cot.strip():
        short_reason = answer
        short_reason_source = "fallback_from_answer"
        short_reason_direct = False

    action_record = normalize_teacher_action_class(
        meta_action=cleaned_text(meta_action_slot.get("meta_action")) or cleaned_text(meta_action_slot.get("cot")),
        answer=answer,
        short_reason=short_reason,
        long_cot=long_cot,
    )
    meta_action, meta_action_source, meta_action_direct = first_meta_action_label(
        [
            (meta_action_slot.get("meta_action"), "direct_meta_action_prompt_meta_action", True),
            (meta_action_slot.get("answer"), "normalized_from_meta_action_prompt_answer", False),
            (meta_action_slot.get("cot"), "normalized_from_meta_action_prompt_cot", False),
            ((selected_payload or {}).get("meta_action"), "normalized_from_structured_payload_meta_action", False),
            (long_cot_prompt.get("meta_action"), "fallback_from_long_cot_prompt_meta_action", False),
        ],
        fallback_label=action_record["value"],
    )

    slot_channel_behavior = build_slot_channel_behavior(prompt_outputs)
    quality = build_quality_annotations(
        long_cot=long_cot,
        short_reason=short_reason,
        meta_action=meta_action,
        answer=answer,
        long_cot_direct=long_cot_direct,
        short_reason_direct=short_reason_direct,
        meta_action_direct=meta_action_direct,
        answer_direct=answer_direct,
        slot_channel_behavior=slot_channel_behavior,
    )

    flags = build_hallucination_flags(
        long_cot=long_cot,
        json_payload=selected_payload,
        meta_action=meta_action,
        answer=answer,
    )
    if selected_payload is not None:
        parse_status = "json_valid"
    elif has_tagged_fields(selected_result):
        parse_status = "tagged_fields_valid"
    else:
        parse_status = "json_invalid"
    normalized = {
        "teacher_long_cot": long_cot,
        "teacher_short_reason": short_reason,
        "teacher_meta_action": meta_action,
        "teacher_answer": answer,
        "teacher_long_cot_source": long_cot_source,
        "teacher_long_cot_direct": long_cot_direct,
        "teacher_short_reason_source": short_reason_source,
        "teacher_short_reason_direct": short_reason_direct,
        "teacher_meta_action_source": meta_action_source,
        "teacher_meta_action_direct": meta_action_direct,
        "teacher_answer_source": answer_source,
        "teacher_answer_direct": answer_direct,
        "slot_channel_behavior": slot_channel_behavior,
        "teacher_action_class": action_record["value"],
        "teacher_action_confidence": action_record["confidence"],
        "teacher_action_source_field": action_record.get("source_field"),
        "teacher_selection_prompt": selected_prompt,
        "teacher_selection_score": selection_score,
        "teacher_parse_status": parse_status,
        "teacher_hallucination_flags": flags,
        **quality,
    }
    diagnostics = {
        "prompt_outputs": prompt_outputs,
        "selected_prompt": selected_prompt,
        "selected_json_payload": selected_payload,
        "selection_score": selection_score,
        "parse_status": parse_status,
        "hallucination_flags": flags,
        "slot_channel_behavior": slot_channel_behavior,
        "source_breakdown": {
            "teacher_long_cot_source": long_cot_source,
            "teacher_short_reason_source": short_reason_source,
            "teacher_meta_action_source": meta_action_source,
            "teacher_answer_source": answer_source,
        },
        "quality": quality,
    }
    return normalized, diagnostics


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
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    return helper, model, processor


def main() -> None:
    args = parse_args()
    started_at = time.time()
    records = load_index_records(args.teacher_index_jsonl)
    selected = select_ready_records(records, overwrite=args.overwrite)
    if args.max_samples is not None:
        selected = selected[: args.max_samples]

    outputs_dir = args.cache_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    helper, model, processor = load_teacher_model(args)

    processed = 0
    succeeded = 0
    failed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    selected_ids = {record["sample_id"] for record in selected}

    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id not in selected_ids:
            skipped += 1
            continue

        processed += 1
        raw_output_path = outputs_dir / f"{sample_id}.teacher_raw.json"
        try:
            request_bundle_path = Path(record["request_bundle_path"])
            request_bundle = json.loads(request_bundle_path.read_text(encoding="utf-8"))
            sample_dir = Path(record["canonical_sample_path"])
            prompt_outputs = build_prompt_outputs(
                sample_dir,
                request_bundle["prompt_families"],
                helper_module=helper,
                processor=processor,
                model=model,
                args=args,
            )
            normalized, diagnostics = normalize_teacher_output(prompt_outputs)
            raw_payload = {
                "sample_id": sample_id,
                "generated_at": time.time(),
                "diagnostics": diagnostics,
                "normalized_output": normalized,
            }
            raw_output_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=True), encoding="utf-8")

            output = record.setdefault("output", {})
            previous_short_reason = output.get("teacher_short_reason")
            previous_signal_source = output.get("teacher_signal_target_source")
            existing_logit_cache_path = output.get("teacher_logit_cache_path")
            existing_hidden_path = output.get("teacher_hidden_path")
            output.update(normalized)
            output["teacher_structured_json_path"] = str(raw_output_path)
            output["teacher_signal_target_field"] = "teacher_short_reason"
            output["teacher_signal_target_source"] = output.get("teacher_short_reason_source")
            signal_cache_stale = (
                previous_short_reason != output.get("teacher_short_reason")
                or previous_signal_source != output.get("teacher_signal_target_source")
            )
            output["teacher_signal_cache_stale"] = signal_cache_stale
            if signal_cache_stale:
                output["teacher_logit_cache_path"] = None
                output["teacher_hidden_path"] = None
            else:
                output["teacher_logit_cache_path"] = existing_logit_cache_path
                output["teacher_hidden_path"] = existing_hidden_path
            record["status"] = "ok" if normalized["teacher_short_reason"] else "generated_empty"
            record["blockers"] = []
            record.setdefault("provenance", {}).update(
                {
                    "soft": "teacher_text",
                    "source": "alpamayo_multi_prompt_generate_text",
                    "teacher_is_gt": False,
                }
            )
            if record["status"] == "ok":
                succeeded += 1
            else:
                failed += 1
                failures.append({"sample_id": sample_id, "error": "teacher_short_reason_empty"})
        except Exception as exc:  # noqa: BLE001
            record["status"] = "generation_failed"
            record["blockers"] = [f"teacher_generation_failed:{type(exc).__name__}"]
            record.setdefault("output", {})
            record["output"]["teacher_structured_json_path"] = None
            failed += 1
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
            if raw_output_path.exists():
                raw_output_path.unlink()

    write_jsonl(args.teacher_index_jsonl, records)

    final_index = load_jsonl_by_key(args.teacher_index_jsonl)
    final_status_counts: dict[str, int] = {}
    for record in final_index.values():
        status = str(record.get("status", "unknown"))
        final_status_counts[status] = final_status_counts.get(status, 0) + 1

    summary = {
        "teacher_index_jsonl": str(args.teacher_index_jsonl),
        "model_path": str(args.model_path),
        "processed_records": processed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "final_status_counts": final_status_counts,
        "failures": failures[:20],
        "elapsed_sec": round(time.time() - started_at, 3),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
