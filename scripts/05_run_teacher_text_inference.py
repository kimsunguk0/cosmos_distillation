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
    compare_teacher_to_reference,
    extract_json_object,
    field_text_hash,
    internal_consistency_score,
    load_jsonl_by_key,
    normalize_teacher_action_class,
    selection_score_from_outputs,
    summarize_json_candidate,
    write_jsonl,
)
from src.data.schema_versions import KD_SCHEMA_VERSION, TEACHER_SIGNAL_CACHE_VERSION, active_versions


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
        output = record.get("output") or {}
        has_output = any(
            output.get(field_name)
            for field_name in ("teacher_long_cot", "teacher_short_reason", "teacher_answer", "teacher_structured_json")
        )
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


def load_sample_inputs(sample_dir: Path) -> dict[str, torch.Tensor]:
    """Load per-sample tensors once so prompt families can reuse them."""
    image_frames, camera_indices = load_image_frames(sample_dir)
    ego_history_xyz = torch.from_numpy(np.load(sample_dir / "ego_history_xyz.npy")).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.from_numpy(np.load(sample_dir / "ego_history_rot.npy")).float().unsqueeze(0).unsqueeze(0)
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }


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
    sample_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    sample_inputs = sample_inputs or load_sample_inputs(sample_dir)
    image_frames = sample_inputs["image_frames"]
    camera_indices = sample_inputs["camera_indices"]
    ego_history_xyz = sample_inputs["ego_history_xyz"]
    ego_history_rot = sample_inputs["ego_history_rot"]

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
    sample_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    sample_inputs = sample_inputs or load_sample_inputs(sample_dir)
    image_frames = sample_inputs["image_frames"]
    camera_indices = sample_inputs["camera_indices"]
    messages = helper_module.create_vqa_message(
        image_frames.flatten(0, 1),
        question=prompt_text.strip(),
        camera_indices=camera_indices,
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
    sample_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Prompt the model to emit cot/meta_action/answer using explicit special tokens."""
    sample_inputs = sample_inputs or load_sample_inputs(sample_dir)
    image_frames = sample_inputs["image_frames"]
    camera_indices = sample_inputs["camera_indices"]
    ego_history_xyz = sample_inputs["ego_history_xyz"]
    ego_history_rot = sample_inputs["ego_history_rot"]

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
    sample_inputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Build a single-slot text generation input for answer/meta-action extraction."""
    sample_inputs = sample_inputs or load_sample_inputs(sample_dir)
    image_frames = sample_inputs["image_frames"]
    camera_indices = sample_inputs["camera_indices"]
    ego_history_xyz = sample_inputs["ego_history_xyz"]
    ego_history_rot = sample_inputs["ego_history_rot"]

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
    short_reason_prompt = prompt_outputs.get("answer_vqa_v1", {})
    meta_action_prompt = prompt_outputs.get("meta_action_vqa_v1", {})
    answer_prompt = prompt_outputs.get("answer_vqa_v1", {})

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
    structured_json_direct: bool,
    parse_status: str,
    slot_channel_behavior: dict[str, Any],
    teacher_vs_weak_gt: dict[str, Any] | None,
    hallucination_flags: list[str] | None,
) -> dict[str, Any]:
    """Compute teacher-quality signals used for downweighting and analysis."""
    overlap = bool(short_reason and answer and short_reason.strip().lower() == answer.strip().lower())
    collapse = bool(slot_channel_behavior["channel_collapse_detected"])
    weak_level = (teacher_vs_weak_gt or {}).get("consistency_level")
    synthesized_fallback = parse_status == "json_synthesized_fallback"
    severe_hallucination = bool(
        {"answer_nonlinguistic", "structured_json_parse_failed"} & set(hallucination_flags or [])
    )

    if meta_action_direct and answer_direct and not collapse:
        direct_slot_reliability = "high"
    elif (meta_action_direct or answer_direct) and not collapse:
        direct_slot_reliability = "medium"
    else:
        direct_slot_reliability = "low"

    if long_cot and not severe_hallucination and weak_level != "hard_fail" and not collapse and (
        short_reason_direct or answer_direct or structured_json_direct
    ):
        text_quality = "accept"
    elif long_cot or short_reason or answer:
        text_quality = "downweight"
    else:
        text_quality = "reject"

    if collapse or synthesized_fallback or weak_level == "hard_fail" or direct_slot_reliability == "low":
        structured_quality = "reject"
    elif meta_action and answer and meta_action_direct and answer_direct and structured_json_direct:
        structured_quality = "accept"
    elif meta_action or answer:
        structured_quality = "downweight"
    else:
        structured_quality = "reject"

    quality_multiplier = 1.0
    if meta_action and not meta_action_direct:
        quality_multiplier *= 0.75
    if answer and not answer_direct:
        quality_multiplier *= 0.8
    if collapse:
        quality_multiplier *= 0.5
    if overlap:
        quality_multiplier *= 0.95
    if severe_hallucination:
        quality_multiplier *= 0.75
    if weak_level == "hard_fail":
        quality_multiplier *= 0.5
    if synthesized_fallback and direct_slot_reliability == "low":
        quality_multiplier = min(quality_multiplier, 0.25)

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


def run_generation(
    model,
    model_inputs: dict[str, Any],
    helper_module,
    *,
    device: str,
    temperature: float,
    top_p: float,
    max_generation_length: int,
) -> dict[str, Any]:
    model_inputs = helper_module.to_device(model_inputs, device)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        extra = model.generate_text(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_samples=1,
            max_generation_length=max_generation_length,
        )
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in extra.items()}


def prompt_candidate_record(raw_extra: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw generation sample into prompt-candidate fields."""
    answer_text = first_text(raw_extra.get("answer"))
    json_payload = extract_json_object(answer_text)
    return {
        "raw_extra": raw_extra,
        "cot": first_text(raw_extra.get("cot")),
        "meta_action": first_text(raw_extra.get("meta_action")),
        "answer": answer_text,
        "json_payload": json_payload,
        "json_summary": summarize_json_candidate(json_payload),
    }


def candidate_structure_score(candidate: dict[str, Any]) -> float:
    """Rank prompt candidates by local structure/coverage before grounded reranking."""
    summary = candidate.get("json_summary") or {}
    score = 0.0
    if candidate.get("cot"):
        score += 0.2
    if summary.get("is_valid"):
        score += 0.45
        score += min(int(summary.get("field_count", 0)), 6) * 0.03
    if normalize_meta_action_label(candidate.get("meta_action")):
        score += 0.15
    if is_natural_language(candidate.get("answer")):
        score += 0.1
    return score


def candidate_is_structurally_usable(candidate: dict[str, Any]) -> bool:
    """Return True when a prompt candidate exposes a usable direct supervision signal."""
    summary = candidate.get("json_summary") or {}
    if summary.get("is_valid"):
        return True
    if normalize_meta_action_label(candidate.get("meta_action")):
        return True
    return is_natural_language(candidate.get("answer"))


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
    sample_inputs = load_sample_inputs(sample_dir)
    for prompt_family in prompt_families:
        name = str(prompt_family["name"])
        prompt_text = str(prompt_family["prompt"])
        generation = dict(prompt_family.get("generation") or {})
        input_builder = str(generation.get("input_builder", "answer"))
        num_candidates = int(generation.get("num_candidates", 1))
        temperature = float(generation.get("temperature", args.temperature))
        top_p = float(generation.get("top_p", args.top_p))
        max_generation_length = int(generation.get("max_generation_length", args.max_generation_length))

        if input_builder == "cot":
            model_inputs = build_cot_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                sample_inputs=sample_inputs,
            )
        elif input_builder == "tagged_triplet":
            model_inputs = build_tagged_triplet_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                sample_inputs=sample_inputs,
            )
        elif input_builder == "answer":
            model_inputs = build_vqa_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                sample_inputs=sample_inputs,
            )
        else:
            model_inputs = build_vqa_inputs(
                sample_dir,
                helper_module=helper_module,
                processor=processor,
                prompt_text=prompt_text,
                sample_inputs=sample_inputs,
            )
        candidates: list[dict[str, Any]] = []
        for _ in range(num_candidates):
            raw_extra = run_generation(
                model,
                model_inputs,
                helper_module,
                device=args.device,
                temperature=temperature,
                top_p=top_p,
                max_generation_length=max_generation_length,
            )
            candidates.append(prompt_candidate_record(raw_extra))
        best = max(candidates, key=candidate_structure_score)
        results[name] = {
            "prompt": prompt_text,
            "generation": generation,
            "candidate_count": len(candidates),
            "alternatives": candidates[1:],
            **best,
        }
    return results


def select_structured_candidate(
    prompt_outputs: dict[str, dict[str, Any]],
    long_cot: str | None,
    *,
    human_action_class: str | None,
    weak_gt_action_class: str | None,
) -> tuple[str | None, dict[str, Any] | None, float, dict[str, Any] | None]:
    """Choose the best structured prompt response by structure then groundedness."""
    best_name = None
    best_payload = None
    best_score = -1.0
    best_result = None
    for name, result in prompt_outputs.items():
        if name == "long_cot_v1":
            continue
        payload = result.get("json_payload")
        if not candidate_is_structurally_usable(result):
            continue
        hallucination_flags = build_hallucination_flags(
            long_cot=long_cot,
            json_payload=payload,
            meta_action=(payload or {}).get("meta_action") or result.get("meta_action"),
            answer=(payload or {}).get("answer") or (payload or {}).get("final_answer") or result.get("answer"),
        )
        internal = internal_consistency_score(
            long_cot=long_cot,
            meta_action=(payload or {}).get("meta_action") or result.get("meta_action"),
            answer=(payload or {}).get("answer") or (payload or {}).get("final_answer") or result.get("answer"),
            json_payload=payload,
        )
        candidate_score = selection_score_from_outputs(
            long_cot=long_cot,
            json_payload=payload,
            meta_action=(payload or {}).get("meta_action") or result.get("meta_action"),
            answer=(payload or {}).get("answer") or (payload or {}).get("final_answer") or result.get("answer"),
            human_action_class=human_action_class,
            weak_gt_action_class=weak_gt_action_class,
            hallucination_flags=hallucination_flags,
            internal_consistency=internal,
        )
        if candidate_score > best_score:
            best_name = name
            best_payload = payload
            best_score = candidate_score
            best_result = result
    return best_name, best_payload, max(best_score, 0.0), best_result


def build_supervision_mode(
    *,
    selected_prompt: str | None,
    selected_payload: dict[str, Any] | None,
    long_cot_direct: bool,
    short_reason_direct: bool,
    meta_action_direct: bool,
    answer_direct: bool,
    parse_status: str,
    direct_slot_reliability: str,
    slot_channel_behavior: dict[str, Any],
    teacher_vs_weak_gt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe how directly usable a teacher sample is for downstream supervision."""
    weak_level = (teacher_vs_weak_gt or {}).get("consistency_level")
    collapse = bool(slot_channel_behavior.get("channel_collapse_detected"))
    synthesized = parse_status == "json_synthesized_fallback"

    if selected_payload is not None:
        selected_direct_prompt_family = selected_prompt
        selected_normalization_mode = "direct_structured_prompt"
    elif long_cot_direct:
        selected_direct_prompt_family = "long_cot_v1"
        selected_normalization_mode = "synthesized_fallback" if synthesized else "long_cot_only"
    else:
        selected_direct_prompt_family = None
        selected_normalization_mode = "missing_direct_prompt"

    direct_slot_count = sum(bool(flag) for flag in (short_reason_direct, meta_action_direct, answer_direct))

    if collapse and synthesized and direct_slot_reliability == "low":
        sample_supervision_mode = "long_cot_only_fallback"
    elif weak_level == "hard_fail" and long_cot_direct:
        sample_supervision_mode = "long_cot_only_fallback"
    elif selected_payload is not None and direct_slot_reliability in {"medium", "high"} and not collapse:
        sample_supervision_mode = "multi_soft_target"
    elif long_cot_direct and direct_slot_count > 0 and not collapse:
        sample_supervision_mode = "partial_soft_target"
    elif long_cot_direct:
        sample_supervision_mode = "long_cot_only_fallback"
    else:
        sample_supervision_mode = "fallback_text_only"

    return {
        "selected_direct_prompt_family": selected_direct_prompt_family,
        "selected_normalization_mode": selected_normalization_mode,
        "sample_supervision_mode": sample_supervision_mode,
    }


def action_from_field_text(field_name: str, text: str | None) -> str | None:
    """Infer a coarse action class from one text field for consistency checks."""
    if not text:
        return None
    if field_name == "teacher_meta_action":
        value = normalize_meta_action_label(text)
        if value not in {None, "unknown"}:
            return value
        value = normalize_teacher_action_class(
            meta_action=text,
            answer=None,
            short_reason=None,
            long_cot=None,
        )["value"]
        return None if value in {None, "unknown"} else value
    kwargs = {
        "meta_action": None,
        "answer": None,
        "short_reason": None,
        "long_cot": None,
    }
    kwargs[field_name.replace("teacher_", "")] = text
    value = normalize_teacher_action_class(**kwargs)["value"]
    return None if value in {None, "unknown"} else value


def prune_conflicting_fallback_slot(
    *,
    field_name: str,
    text: str | None,
    source: str,
    direct: bool,
    anchor_action: str | None,
) -> tuple[str | None, str, bool, dict[str, Any] | None]:
    """Drop collapsed/fallback slot text when it conflicts with the long-cot action."""
    if not text:
        return text, source, direct, None
    if direct:
        return text, source, direct, None
    if anchor_action in {None, "unknown"}:
        return text, source, direct, None
    if not any(
        token in source
        for token in (
            "normalized_from_",
            "fallback_from_",
            "selected_prompt",
            "structured_payload",
            "legacy_recovered_output",
        )
    ):
        return text, source, direct, None

    candidate_action = action_from_field_text(field_name, text)
    if candidate_action in {None, "unknown"}:
        return text, source, direct, None

    comparison = compare_teacher_to_reference(candidate_action, anchor_action)
    if comparison and comparison.get("consistency_level") in {"soft_fail", "hard_fail"}:
        return None, f"dropped_conflicting_{field_name}", False, comparison
    return text, source, direct, comparison


def normalize_teacher_output(
    prompt_outputs: dict[str, dict[str, Any]],
    *,
    human_action_class: str | None,
    weak_gt_action_class: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the final normalized teacher payload plus diagnostics."""
    long_cot_prompt = prompt_outputs.get("long_cot_v1", {})
    long_cot, long_cot_source, long_cot_direct = first_usable_text(
        [
            (long_cot_prompt.get("cot"), "direct_long_cot_prompt_cot", True),
            (long_cot_prompt.get("answer"), "fallback_from_long_cot_prompt_answer", False),
        ],
        require_natural_language=True,
    )
    selected_prompt, selected_payload, selection_score, selected_result = select_structured_candidate(
        prompt_outputs,
        long_cot,
        human_action_class=human_action_class,
        weak_gt_action_class=weak_gt_action_class,
    )
    selected_result = selected_result or prompt_outputs.get(selected_prompt or "", {})
    short_reason_slot = prompt_outputs.get("answer_vqa_v1", {})
    meta_action_slot = prompt_outputs.get("meta_action_vqa_v1", {})
    answer_slot = prompt_outputs.get("answer_vqa_v1", {})
    structured_payload = dict(selected_payload) if selected_payload is not None else None

    short_reason, short_reason_source, short_reason_direct = first_usable_text(
        [
            (short_reason_slot.get("answer"), "direct_answer_vqa_answer", True),
            (short_reason_slot.get("cot"), "normalized_from_answer_vqa_cot", False),
        ],
        require_natural_language=True,
    )

    answer, answer_source, answer_direct = first_usable_text(
        [
            (answer_slot.get("answer"), "direct_answer_vqa_answer", True),
            (answer_slot.get("cot"), "normalized_from_answer_vqa_cot", False),
        ],
        require_natural_language=True,
    )

    anchor_action = normalize_teacher_action_class(
        meta_action=None,
        answer=None,
        short_reason=None,
        long_cot=long_cot,
    )["value"]
    short_reason, short_reason_source, short_reason_direct, short_reason_cmp = prune_conflicting_fallback_slot(
        field_name="teacher_short_reason",
        text=short_reason,
        source=short_reason_source,
        direct=short_reason_direct,
        anchor_action=anchor_action,
    )
    answer, answer_source, answer_direct, answer_cmp = prune_conflicting_fallback_slot(
        field_name="teacher_answer",
        text=answer,
        source=answer_source,
        direct=answer_direct,
        anchor_action=anchor_action,
    )
    meta_action_prompt_text = cleaned_text(meta_action_slot.get("meta_action")) or cleaned_text(meta_action_slot.get("answer"))
    meta_action_prompt_cot, _, _, meta_action_cmp = prune_conflicting_fallback_slot(
        field_name="teacher_meta_action",
        text=cleaned_text(meta_action_slot.get("cot")),
        source="normalized_from_meta_action_prompt_cot",
        direct=False,
        anchor_action=anchor_action,
    )

    action_record = normalize_teacher_action_class(
        meta_action=meta_action_prompt_text or meta_action_prompt_cot,
        answer=answer,
        short_reason=short_reason,
        long_cot=long_cot,
    )
    meta_action, meta_action_source, meta_action_direct = first_meta_action_label(
        [
            (meta_action_slot.get("meta_action"), "direct_meta_action_prompt_meta_action", True),
            (meta_action_slot.get("answer"), "direct_meta_action_vqa_answer", True),
            (meta_action_prompt_cot, "normalized_from_meta_action_prompt_cot", False),
            (long_cot_prompt.get("meta_action"), "fallback_from_long_cot_prompt_meta_action", False),
        ],
        fallback_label=action_record["value"],
    )

    structured_json = None
    structured_json_source = "unsupported_in_official_teacher_contract"
    structured_json_direct = False

    flags = build_hallucination_flags(
        long_cot=long_cot,
        json_payload=structured_payload,
        meta_action=meta_action,
        answer=answer,
    )
    if answer_direct or meta_action_direct:
        parse_status = "answer_vqa_valid"
    elif has_tagged_fields(selected_result):
        parse_status = "tagged_fields_valid"
    else:
        parse_status = "json_invalid"
    slot_channel_behavior = build_slot_channel_behavior(prompt_outputs)
    teacher_vs_human = compare_teacher_to_reference(action_record["value"], human_action_class)
    teacher_vs_weak_gt = compare_teacher_to_reference(action_record["value"], weak_gt_action_class)
    quality = build_quality_annotations(
        long_cot=long_cot,
        short_reason=short_reason,
        meta_action=meta_action,
        answer=answer,
        long_cot_direct=long_cot_direct,
        short_reason_direct=short_reason_direct,
        meta_action_direct=meta_action_direct,
        answer_direct=answer_direct,
        structured_json_direct=structured_json_direct,
        parse_status=parse_status,
        slot_channel_behavior=slot_channel_behavior,
        teacher_vs_weak_gt=teacher_vs_weak_gt,
        hallucination_flags=flags,
    )
    supervision_mode = build_supervision_mode(
        selected_prompt=selected_prompt,
        selected_payload=selected_payload,
        long_cot_direct=long_cot_direct,
        short_reason_direct=short_reason_direct,
        meta_action_direct=meta_action_direct,
        answer_direct=answer_direct,
        parse_status=parse_status,
        direct_slot_reliability=quality["teacher_direct_slot_reliability"],
        slot_channel_behavior=slot_channel_behavior,
        teacher_vs_weak_gt=teacher_vs_weak_gt,
    )
    if supervision_mode["sample_supervision_mode"] == "long_cot_only_fallback":
        quality["teacher_quality_multiplier"] = min(float(quality["teacher_quality_multiplier"]), 0.25)
        short_reason = None
        short_reason_source = "suppressed_due_to_long_cot_only_fallback"
        short_reason_direct = False
        answer = None
        answer_source = "suppressed_due_to_long_cot_only_fallback"
        answer_direct = False
        structured_json = None
        structured_json_source = "suppressed_due_to_long_cot_only_fallback"
        structured_json_direct = False

    signal_fields = [field for field, text in (("teacher_short_reason", short_reason), ("teacher_answer", answer)) if text]
    if supervision_mode["sample_supervision_mode"] == "long_cot_only_fallback":
        signal_fields = ["teacher_long_cot"] if long_cot else []
    elif supervision_mode["sample_supervision_mode"] == "partial_soft_target":
        if not signal_fields and long_cot:
            signal_fields = ["teacher_long_cot"]

    normalized = {
        "teacher_long_cot": long_cot,
        "teacher_structured_json": structured_json,
        "teacher_short_reason": short_reason,
        "teacher_meta_action": meta_action,
        "teacher_answer": answer,
        "teacher_long_cot_source": long_cot_source,
        "teacher_long_cot_direct": long_cot_direct,
        "teacher_structured_json_source": structured_json_source,
        "teacher_structured_json_direct": structured_json_direct,
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
        "teacher_selection_prompt": selected_prompt if selected_payload is not None else None,
        "teacher_selection_score": selection_score,
        "teacher_parse_status": parse_status,
        "teacher_hallucination_flags": flags,
        "teacher_vs_human_consistency": teacher_vs_human,
        "teacher_vs_weak_gt_consistency": teacher_vs_weak_gt,
        **supervision_mode,
        "teacher_signal_schema": {
            "kd_schema_version": KD_SCHEMA_VERSION,
            "teacher_signal_cache_version": TEACHER_SIGNAL_CACHE_VERSION,
            "signal_fields": signal_fields,
            "seq_only_fields": [
                field_name
                for field_name, field_text in (
                    ("teacher_structured_json", structured_json),
                    ("teacher_long_cot", long_cot),
                )
                if field_text and field_name not in signal_fields
            ],
            "label_only_fields": ["teacher_meta_action"],
        },
        **quality,
    }
    diagnostics = {
        "prompt_outputs": prompt_outputs,
        "selected_prompt": selected_prompt if selected_payload is not None else None,
        "selected_direct_prompt_family": supervision_mode["selected_direct_prompt_family"],
        "selected_normalization_mode": supervision_mode["selected_normalization_mode"],
        "sample_supervision_mode": supervision_mode["sample_supervision_mode"],
        "selected_json_payload": structured_payload,
        "selection_score": selection_score,
        "parse_status": parse_status,
        "hallucination_flags": flags,
        "slot_channel_behavior": slot_channel_behavior,
        "source_breakdown": {
            "teacher_long_cot_source": long_cot_source,
            "teacher_structured_json_source": structured_json_source,
            "teacher_short_reason_source": short_reason_source,
            "teacher_meta_action_source": meta_action_source,
            "teacher_answer_source": answer_source,
        },
        "teacher_vs_human_consistency": teacher_vs_human,
        "teacher_vs_weak_gt_consistency": teacher_vs_weak_gt,
        "quality": quality,
        "slot_conflict_checks": {
            "teacher_short_reason": short_reason_cmp,
            "teacher_answer": answer_cmp,
            "teacher_meta_action": meta_action_cmp,
        },
    }
    return normalized, diagnostics


def load_teacher_model(args: argparse.Namespace):
    if str(args.alpamayo_src) not in sys.path:
        sys.path.insert(0, str(args.alpamayo_src))

    from alpamayo1_5 import helper
    from alpamayo1_5.config import Alpamayo1_5Config
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    config_path = args.model_path / "alpamayo_1.5_config.json"
    if config_path.exists():
        config = Alpamayo1_5Config(**json.loads(config_path.read_text()))
    else:
        config = Alpamayo1_5Config.from_pretrained(str(args.model_path))
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
            request_bundle_path = Path(record["runtime_request_path"])
            diagnostics_bundle_path = Path(record["diagnostics_bundle_path"])
            request_bundle = json.loads(request_bundle_path.read_text(encoding="utf-8"))
            diagnostics_bundle = json.loads(diagnostics_bundle_path.read_text(encoding="utf-8"))
            sample_dir = Path(record["canonical_sample_path"])
            prompt_outputs = build_prompt_outputs(
                sample_dir,
                request_bundle["prompt_families"],
                helper_module=helper,
                processor=processor,
                model=model,
                args=args,
            )
            weak = diagnostics_bundle.get("weak_derived") or {}
            human_record = weak.get("meta_action_from_human") or {}
            weak_gt_record = weak.get("action_class_from_gt_path") or {}
            normalized, diagnostics = normalize_teacher_output(
                prompt_outputs,
                human_action_class=human_record.get("value"),
                weak_gt_action_class=weak_gt_record.get("value"),
            )
            raw_payload = {
                "sample_id": sample_id,
                "generated_at": time.time(),
                "versions": active_versions(),
                "diagnostics": diagnostics,
                "normalized_output": normalized,
            }
            raw_output_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=True), encoding="utf-8")

            output = record.setdefault("output", {})
            previous_targets = dict(output.get("teacher_signal_targets") or {})
            output.update(normalized)
            output["teacher_structured_json_path"] = str(raw_output_path)
            output["teacher_signal_targets"] = {}
            for field_name in ("teacher_short_reason", "teacher_answer", "teacher_structured_json", "teacher_long_cot"):
                field_value = output.get(field_name)
                if not field_value:
                    continue
                new_hash = field_text_hash(field_value)
                previous_target = previous_targets.get(field_name) or {}
                output["teacher_signal_targets"][field_name] = {
                    "field_name": field_name,
                    "text_hash": new_hash,
                    "signal_ready": previous_target.get("signal_ready", False) and previous_target.get("text_hash") == new_hash,
                    "signal_cache_stale": previous_target.get("text_hash") != new_hash,
                    "source": output.get(f"{field_name}_source", "missing"),
                    "deterministic_second_pass": field_name in {"teacher_short_reason", "teacher_answer"},
                    "logits_path": previous_target.get("logits_path") if previous_target.get("text_hash") == new_hash else None,
                    "hidden_path": previous_target.get("hidden_path") if previous_target.get("text_hash") == new_hash else None,
                    "cache_version": TEACHER_SIGNAL_CACHE_VERSION,
                }
            primary_field = next(
                (
                    field_name
                    for field_name in output.get("teacher_signal_schema", {}).get("signal_fields", [])
                    if output["teacher_signal_targets"].get(field_name)
                ),
                "teacher_short_reason",
            )
            primary_target = output["teacher_signal_targets"].get(primary_field)
            output["teacher_logit_cache_path"] = None if primary_target is None else primary_target.get("logits_path")
            output["teacher_hidden_path"] = None if primary_target is None else primary_target.get("hidden_path")
            record["status"] = (
                "ok"
                if (
                    normalized.get("teacher_long_cot")
                    or normalized.get("teacher_meta_action")
                    or normalized.get("teacher_short_reason")
                    or normalized.get("teacher_answer")
                    or normalized.get("teacher_structured_json")
                )
                else "generated_empty"
            )
            record["blockers"] = []
            record["versions"] = active_versions()
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
