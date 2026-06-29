"""Shared helpers for Step A Q2 VQA distillation.

The Step A contract is intentionally narrower than the trajectory
distillation path: 4cam x 1frame VQA input, no ego/nav/action tokens, and
answer-only supervision after ``<|answer_start|>``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data.local_dataset import decode_video_frames


IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
Q2_OFFICIAL = (
    "What are the key traffic elements visible in this scene and how should they "
    "influence driving behavior?"
)
CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    6: "Front telephoto camera",
}
DEFAULT_CAMERA_INDICES = (0, 1, 2, 6)
DEFAULT_CAMERA_ALIASES = ("cross_left", "front_wide", "cross_right", "front_tele")

COORDINATE_RE = re.compile(
    r"(\[[^\]]*\d[^\]]*\]|\([^\)]*\d[,\s]+[^\)]*\d[^\)]*\)|"
    r"\b\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\b)"
)
WHITESPACE_RE = re.compile(r"\s+")
ACTION_OVERREACH_RE = re.compile(
    r"\b("
    r"be prepared|prepare|prepared|slow down|reduce speed|stop|yield|proceed|continue|"
    r"maintain|safe following distance|lane change|change lanes|avoid|brake|creep|"
    r"should|must|requires|require|necessitates|necessitate|give way|cautious approach|"
    r"vigilant|will|about to|likely|might|may|could collide|future|sudden"
    r")\b",
    re.IGNORECASE,
)
INFLUENCE_SPLIT_RE = re.compile(
    r"\b(indicating|indicates|suggesting|suggests|suggest|so|therefore|which means|"
    r"meaning that)\b",
    re.IGNORECASE,
)
HARD_STOP_RE = re.compile(
    r"\b(it'?s essential|it is essential|drivers? should|the driver should|should|must|"
    r"need to|needs to|before proceeding|before continuing)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\u2014", "-").replace("\u2013", "-")
    text = COORDINATE_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip(" ;,.")
    return text


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?":
        text = f"{text}."
    return text[0].upper() + text[1:]


def _word_limit(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "."


def _rewrite_action_phrases(text: str) -> str:
    """Turn common Q2 action-advice phrases back into visible evidence."""
    replacements = [
        (r"\bmaintaining a safe distance from\s+([^,.;]+)", r"\1"),
        (r"\bmaintain a safe distance from\s+([^,.;]+)", r"\1"),
        (r"\bstaying alert for\s+([^,.;]+)", r"\1"),
        (r"\bstay alert for\s+([^,.;]+)", r"\1"),
        (r"\bmonitoring for\s+([^,.;]+)", r"\1"),
        (r"\bmonitor for\s+([^,.;]+)", r"\1"),
        (r"\bbeing prepared for\s+([^,.;]+)", r"\1"),
        (r"\bbe prepared for\s+([^,.;]+)", r"\1"),
        (r"\bpotential congestion\b", "traffic congestion"),
        (r"\bsudden stops or changes in traffic flow\b", "nearby traffic flow"),
        (r"\bsudden stops\b", "nearby stopped traffic"),
    ]
    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return out


def _trim_instruction_tail(text: str) -> str:
    text = re.split(
        r"\b(?:all of\s+)?which\s+(?:requires?|necessitates?|suggests?)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.split(r"\s*;\s*", text, maxsplit=1)[0]
    text = re.split(r"\bthese elements\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\bthis requires\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = HARD_STOP_RE.split(text, maxsplit=1)[0]
    text = re.split(r",\s*but\s+(?=with|drivers?|it\b)", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip(" ;,.")


def shorten_q2_answer(raw_answer: str, *, max_words: int = 56) -> tuple[str, dict[str, Any]]:
    """Create a compact hard target from an accepted Alpamayo Q2 answer.

    The LLM judge accepted many useful Q2 answers, but some still contain long
    action advice or future phrasing. This deterministic pass keeps the visible
    evidence and converts the driving implication into a decision-support
    sentence instead of a direct action command.
    """
    raw = normalize_text(raw_answer)
    if not raw:
        return "No visible evidence.", {"shorten_reason": "empty_raw_answer", "word_count": 3}

    raw = re.sub(r"^\s*key traffic elements include\s*:?\s*", "Visible traffic elements include ", raw, flags=re.I)
    raw = re.sub(r"^\s*the key traffic elements (visible )?include\s*:?\s*", "Visible traffic elements include ", raw, flags=re.I)
    raw = re.sub(r"^\s*the key elements are\s*:?\s*", "Visible traffic elements include ", raw, flags=re.I)
    raw = re.sub(r"^\s*the key elements to consider are\s*:?\s*", "Visible traffic elements include ", raw, flags=re.I)
    raw = re.sub(
        r"^\s*in this scenario,?\s*(?:several\s+)?(?:key\s+)?elements(?:\s+that)?\s+"
        r"(?:could|might)\s+affect\s+(?:our|the current)\s+driving behavior\s*(?:include|:)\s*:?\s*",
        "Visible traffic elements include ",
        raw,
        flags=re.I,
    )
    raw = re.sub(r"\bthe ego vehicle\b", "the current driving scene", raw, flags=re.I)
    raw = _rewrite_action_phrases(raw)

    split = INFLUENCE_SPLIT_RE.search(raw)
    evidence = raw[: split.start()].strip(" ;,.") if split else raw.split(".")[0].strip(" ;,.")
    trailing = raw[split.end() :].strip(" ;,.") if split else ""
    if not evidence:
        evidence = raw
    evidence = _trim_instruction_tail(evidence)
    if not evidence:
        evidence = _trim_instruction_tail(raw)
    if not evidence:
        evidence = raw.split(".")[0].strip(" ;,.")

    evidence = _sentence_case(evidence)
    influence = ""
    if trailing:
        # Keep condition-style implications and drop direct action commands.
        first_clause = re.split(r"[.;]", trailing, maxsplit=1)[0].strip(" ;,.")
        if first_clause and not ACTION_OVERREACH_RE.search(first_clause):
            influence = _sentence_case(f"These elements matter because {first_clause}")
    if not influence:
        influence = "These elements are relevant to the current driving judgment."

    evidence = _word_limit(evidence, min(40, max_words))
    remaining_words = max(0, int(max_words) - len(evidence.split()))
    if remaining_words >= 8:
        target = _word_limit(f"{evidence} {influence}", max_words=max_words)
    else:
        target = evidence
    flags = {
        "shorten_reason": "deterministic_visible_evidence_summary",
        "raw_word_count": len(raw.split()),
        "word_count": len(target.split()),
        "had_action_or_future_language": bool(ACTION_OVERREACH_RE.search(raw)),
        "had_coordinate_like_text": bool(COORDINATE_RE.search(str(raw_answer or ""))),
    }
    return target, flags


def pil_to_chw_uint8(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_row_frame_tensors(row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the exact 4cam x 1frame image tensor for a judged Q2 row."""
    dataset_root = Path(row["dataset_root"])
    clip_id = str(row["clip_id"])
    chunk = int(row["chunk"])
    camera_tensors: list[torch.Tensor] = []
    camera_indices: list[int] = []
    for plan in row.get("frame_plan") or []:
        feature = str(plan["feature"])
        frame_indices = [int(v) for v in plan["frame_indices"]]
        images = decode_video_frames(dataset_root, clip_id, chunk, feature, frame_indices)
        camera_tensors.append(torch.stack([pil_to_chw_uint8(image) for image in images], dim=0))
        camera_indices.append(int(plan.get("camera_index", len(camera_indices))))
    if not camera_tensors:
        raise ValueError(f"row has no frame_plan: {row.get('sample_id')}")
    frames = torch.stack(camera_tensors, dim=0)
    return frames, torch.tensor(camera_indices, dtype=torch.long)


def _build_image_content(
    frames_flat: torch.Tensor,
    camera_indices: torch.Tensor | None,
    num_frames_per_camera: int,
) -> list[dict[str, Any]]:
    if camera_indices is None:
        return [{"type": "image", "image": frame} for frame in frames_flat]
    expanded = camera_indices.repeat_interleave(int(num_frames_per_camera))
    content: list[dict[str, Any]] = []
    prev_cam = None
    frame_idx = 0
    for image_index, frame in enumerate(frames_flat):
        cam_id = int(expanded[image_index].item())
        if prev_cam is not None and cam_id != prev_cam:
            frame_idx = 0
        if frame_idx == 0:
            content.append({"type": "text", "text": f"{CAMERA_DISPLAY_NAMES.get(cam_id, f'Camera {cam_id}')}: "})
        content.append({"type": "text", "text": f"frame {frame_idx} "})
        content.append({"type": "image", "image": frame})
        prev_cam = cam_id
        frame_idx += 1
    return content


def create_vqa_messages(
    *,
    frames: torch.Tensor,
    camera_indices: torch.Tensor,
    question: str,
    answer_text: str | None = None,
) -> list[dict[str, Any]]:
    """Build the official VQA message shape used for Q2-only distillation."""
    if frames.ndim != 5:
        raise ValueError(f"frames must be (Cams, Frames, C, H, W), got {tuple(frames.shape)}")
    user_text = f"<|question_start|>{question}<|question_end|>"
    assistant_text = "<|answer_start|>"
    if answer_text is not None:
        assistant_text = f"{assistant_text}{answer_text.strip()}"
    image_content = _build_image_content(
        frames.flatten(0, 1),
        camera_indices,
        num_frames_per_camera=int(frames.shape[1]),
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def encode_messages(
    processor: Any,
    messages_batch: list[list[dict[str, Any]]],
    *,
    continue_final_message: bool,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": False,
        "continue_final_message": bool(continue_final_message),
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
    }
    if max_length is not None:
        kwargs["truncation"] = True
        kwargs["max_length"] = int(max_length)
    return processor.apply_chat_template(messages_batch, **kwargs)


def labels_from_prompt_and_full(prompt_batch: dict[str, torch.Tensor], full_batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = full_batch["input_ids"].clone()
    labels[full_batch["attention_mask"] == 0] = IGNORE_INDEX
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
    full_lengths = full_batch["attention_mask"].sum(dim=1)
    for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
        prompt_length = int(prompt_length)
        full_length = int(full_lengths[row_index].item())
        if prompt_length >= full_length:
            raise ValueError("Prompt length consumed the whole sequence; no answer tokens remain.")
        prompt_prefix = prompt_batch["input_ids"][row_index, :prompt_length]
        full_prefix = full_batch["input_ids"][row_index, :prompt_length]
        if not torch.equal(prompt_prefix, full_prefix):
            raise ValueError("Prompt/full token prefixes diverged; answer mask would be corrupt.")
        labels[row_index, :prompt_length] = IGNORE_INDEX
        if not torch.any(labels[row_index] != IGNORE_INDEX):
            raise ValueError("No labeled answer tokens remain after masking.")
    return labels


def active_label_positions(labels: torch.Tensor) -> list[list[int]]:
    positions: list[list[int]] = []
    for row_index in range(labels.shape[0]):
        positions.append(torch.nonzero(labels[row_index] != IGNORE_INDEX, as_tuple=False).flatten().tolist())
    return positions
