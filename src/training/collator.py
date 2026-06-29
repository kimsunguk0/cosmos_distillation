"""Corpus collation for v3.2 multiview distillation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from src.data.consistency import ACTION_CLASSES
from src.training.losses import resolve_loss_weight_value, resolve_optional_loss_weight_value
from src.utils.runtime_paths import remap_external_path
from src.utils.traj_tokens import discrete_traj_token


IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."
FRAME_OFFSETS_DEFAULT = (-0.3, -0.2, -0.1, 0.0)
BOUNDARY_HIDDEN_NAMES = ("cot_end", "traj_start", "action_pre")
MATERIALIZED_4X4_IMAGE_NAMES = tuple(
    f"cam{camera_idx}_f{frame_idx}.png" for camera_idx in range(4) for frame_idx in range(4)
)
CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


@dataclass(slots=True)
class TargetLayout:
    completion_text: str
    cot_content_len: int
    cot_suffix_len: int
    traj_prefix_len: int
    traj_content_len: int
    traj_suffix_len: int
    loss_ignore_prefix_len: int = 0

    @property
    def cot_span_len(self) -> int:
        return self.cot_content_len + self.cot_suffix_len

    @property
    def traj_span_len(self) -> int:
        return self.traj_prefix_len + self.traj_content_len + self.traj_suffix_len


def build_label_mask() -> str:
    """Return the label-mask policy used by the collator."""
    return "assistant_output_only"


def action_class_to_id(value: str | None) -> int:
    """Convert an action class into a compact integer label."""
    if value is None:
        return ACTION_CLASSES.index("unknown")
    value = str(value).strip().lower()
    return ACTION_CLASSES.index(value) if value in ACTION_CLASSES else ACTION_CLASSES.index("unknown")


def _resolve_path(raw_path: str | Path | None, project_root: Path) -> Path:
    remapped = remap_external_path(raw_path)
    if remapped is None:
        raise FileNotFoundError(f"Missing runtime path for {raw_path!r}")
    path = Path(remapped)
    return path if path.is_absolute() else project_root / path


def resolve_sample_path(sample: dict[str, Any], project_root: Path) -> Path:
    """Resolve the best-available sample root for a corpus record."""
    sample_input = sample.get("input") or {}
    raw_path = sample_input.get("materialized_sample_path") or sample_input.get("canonical_sample_path")
    if raw_path in (None, ""):
        raise KeyError("Record is missing input.materialized_sample_path / input.canonical_sample_path")
    return _resolve_path(raw_path, project_root)


def frame_offsets_from_sample(sample_dir: Path) -> list[float]:
    """Load canonical frame offsets for legacy materialized samples."""
    sample_meta_path = sample_dir / "sample_meta.json"
    if not sample_meta_path.exists():
        return list(FRAME_OFFSETS_DEFAULT)
    sample_meta = json.loads(sample_meta_path.read_text(encoding="utf-8"))
    return [float(value) for value in sample_meta.get("frame_offsets_sec", FRAME_OFFSETS_DEFAULT)]


def load_sample_images(sample: dict[str, Any], project_root: Path) -> list[Image.Image]:
    """Load the canonical multi-camera image set in a stable order."""
    sample_input = sample.get("input") or {}
    image_paths = list(sample_input.get("image_paths") or [])
    if image_paths:
        return [Image.open(_resolve_path(path, project_root)).convert("RGB") for path in image_paths]

    sample_dir = resolve_sample_path(sample, project_root)
    if sample_input.get("image_layout") == "materialized_4x4_png":
        image_names = list(sample_input.get("image_names") or MATERIALIZED_4X4_IMAGE_NAMES)
        return [Image.open(sample_dir / "images" / str(name)).convert("RGB") for name in image_names]

    frame_offsets = frame_offsets_from_sample(sample_dir)
    images: list[Image.Image] = []
    for camera_name in sample_input.get("camera_names") or []:
        for offset in frame_offsets:
            image_path = sample_dir / "frames" / f"{camera_name}_t{offset:+.1f}.jpg"
            images.append(Image.open(image_path).convert("RGB"))
    if not images:
        raise FileNotFoundError(f"No image inputs resolved for sample {sample.get('sample_id')}")
    return images


def apply_image_ablation(images: list[Image.Image], mode: str, *, sample_id: str) -> list[Image.Image]:
    """Apply deterministic visual perturbations used by FLEX sensitivity checks."""
    mode = str(mode or "normal").strip().lower()
    if mode == "normal":
        return images
    if mode in {"black", "gray"}:
        color = (0, 0, 0) if mode == "black" else (127, 127, 127)
        return [Image.new("RGB", image.size, color) for image in images]
    if mode == "noise":
        seed = abs(hash(str(sample_id))) % (2**32)
        out: list[Image.Image] = []
        for image_index, image in enumerate(images):
            local = np.random.default_rng(seed + image_index * 1009)
            arr = local.integers(0, 256, size=(image.size[1], image.size[0], 3), dtype=np.uint8)
            out.append(Image.fromarray(arr, mode="RGB"))
        return out
    if mode == "camera_shuffle":
        frame_count = 4 if len(images) % 4 == 0 else 1
        groups = [images[index : index + frame_count] for index in range(0, len(images), frame_count)]
        return [image for group in reversed(groups) for image in group]
    raise ValueError(f"Unsupported image_ablation={mode!r}")


def load_ego_history_xyz(sample: dict[str, Any], project_root: Path) -> np.ndarray:
    """Load ego history coordinates from the v3.2 or legacy sample layout."""
    sample_input = sample.get("input") or {}
    history_path = sample_input.get("ego_history_path")
    if history_path:
        history = np.load(_resolve_path(history_path, project_root)).astype(np.float32)
    else:
        sample_dir = resolve_sample_path(sample, project_root)
        legacy_path = sample_dir / "ego_history_xyz.npy"
        if not legacy_path.exists():
            raise FileNotFoundError(f"Missing ego history for sample {sample.get('sample_id')}")
        history = np.load(legacy_path).astype(np.float32)
    if history.ndim == 1:
        return history.reshape(-1, 1)
    if history.ndim > 2:
        return history.reshape(-1, history.shape[-1])
    return history


def load_ego_future_xyz(sample: dict[str, Any], project_root: Path) -> np.ndarray:
    """Load ego future coordinates from the v3.2 or legacy sample layout."""
    sample_dir = resolve_sample_path(sample, project_root)
    candidate_paths = (
        sample_dir / "ego" / "ego_future_xyz.npy",
        sample_dir / "ego_future_xyz.npy",
    )
    for path in candidate_paths:
        if path.exists():
            future = np.load(path).astype(np.float32)
            if future.ndim == 1:
                return future.reshape(-1, 1)
            if future.ndim > 2:
                return future.reshape(-1, future.shape[-1])
            return future
    raise FileNotFoundError(f"Missing ego future xyz for sample {sample.get('sample_id')}")


def load_traj_future_token_ids(target: dict[str, Any], project_root: Path) -> list[int]:
    """Load 128 discrete future trajectory token ids from inline values or an artifact path."""
    raw_token_ids = target.get("traj_future_token_ids")
    if raw_token_ids:
        return [int(token_id) for token_id in raw_token_ids]
    token_path = target.get("traj_future_token_ids_path")
    if token_path:
        return [int(token_id) for token_id in np.load(_resolve_path(token_path, project_root)).reshape(-1).tolist()]
    return []


def format_history_text(history_xyz: np.ndarray) -> str:
    """Serialize ego history into a compact token block for the student input."""
    if history_xyz.ndim > 2:
        history_xyz = history_xyz.reshape(-1, history_xyz.shape[-1])
    steps = []
    for index, point in enumerate(history_xyz):
        steps.append(f"{index}:{point[0]:+.2f},{point[1]:+.2f},{point[2]:+.2f}")
    joined = " | ".join(steps)
    return f"<|traj_history_start|>{joined}<|traj_history_end|>"


def format_history_placeholder_text(num_history_tokens: int = 48) -> str:
    """Return Alpamayo's placeholder history span before trajectory-token fusion."""
    return f"<|traj_history_start|>{'<|traj_history|>' * int(num_history_tokens)}<|traj_history_end|>"


def official_alpamayo_prompt_text(sample: dict[str, Any], *, num_history_tokens: int = 48) -> str:
    """Create the public Alpamayo inference prompt text.

    The raw chat template contains placeholder history tokens. Alpamayo replaces
    those placeholders with encoded history delta tokens before the VLM forward.
    """
    sample_input = sample.get("input") or {}
    route_section = ""
    nav_text = str(sample_input.get("nav_text") or "").strip()
    nav_available = bool(sample_input.get("nav_available", False))
    if nav_available and nav_text:
        route_section = f"<|route_start|>{nav_text}<|route_end|>"
    prompt_text = (
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
    return f"{format_history_placeholder_text(num_history_tokens)}{route_section}{prompt_text}"


def build_user_prompt(
    sample: dict[str, Any],
    project_root: Path,
    *,
    ego_history_xyz: np.ndarray | None = None,
    prompt_text_style: str = "numeric_history_question",
) -> str:
    """Create the textual instruction paired with the image stack."""
    style = str(prompt_text_style or "numeric_history_question").strip().lower()
    if style in {"official", "official_alpamayo", "alpamayo"}:
        return official_alpamayo_prompt_text(sample)
    history_xyz = ego_history_xyz if ego_history_xyz is not None else load_ego_history_xyz(sample, project_root)
    question = str((sample.get("input") or {}).get("question") or "").strip()
    history_text = format_history_text(history_xyz)
    return f"{history_text}\n<|question_start|>{question}<|question_end|>"


def _encode_history_delta_token_ids(history_xyz: np.ndarray, tokenizer: Any) -> list[int]:
    """Encode 16 ego-history xyz points into Alpamayo's 48 history token ids."""
    history_xyz = np.asarray(history_xyz, dtype=np.float32)
    if history_xyz.ndim > 2:
        history_xyz = history_xyz.reshape(-1, history_xyz.shape[-1])
    if history_xyz.shape[-1] < 3:
        raise ValueError(f"ego history needs at least xyz dims, got shape={history_xyz.shape}")
    xyz = history_xyz[:, :3]
    padded = np.concatenate([np.zeros((1, 3), dtype=np.float32), xyz], axis=0)
    delta = padded[1:] - padded[:-1]
    mins = np.asarray([-4.0, -4.0, -10.0], dtype=np.float32)
    maxs = np.asarray([4.0, 4.0, 10.0], dtype=np.float32)
    bins = np.rint((delta - mins) / (maxs - mins) * 999.0).astype(np.int64)
    bins = np.clip(bins, 0, 999).reshape(-1)

    token_ids: list[int] = []
    for value in bins.tolist():
        token_id = tokenizer.convert_tokens_to_ids(discrete_traj_token(3000 + int(value)))
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"Tokenizer is missing history token {discrete_traj_token(3000 + int(value))}")
        token_ids.append(int(token_id))
    return token_ids


def fuse_history_tokens_in_input_ids(
    input_ids: torch.Tensor,
    tokenizer: Any,
    ego_history_items: list[np.ndarray],
) -> torch.Tensor:
    """Replace `<|traj_history|>` placeholders with encoded history delta token ids."""
    placeholder_id = tokenizer.convert_tokens_to_ids("<|traj_history|>")
    if not isinstance(placeholder_id, int) or placeholder_id < 0:
        raise ValueError("Tokenizer is missing <|traj_history|> placeholder token")
    fused = input_ids.clone()
    for row_index, history_xyz in enumerate(ego_history_items):
        positions = torch.nonzero(fused[row_index] == int(placeholder_id), as_tuple=False).flatten()
        history_token_ids = _encode_history_delta_token_ids(history_xyz, tokenizer)
        if int(positions.numel()) != len(history_token_ids):
            raise ValueError(
                f"History placeholder count mismatch for row {row_index}: "
                f"expected {len(history_token_ids)}, got {int(positions.numel())}"
            )
        fused[row_index, positions] = torch.tensor(history_token_ids, dtype=fused.dtype, device=fused.device)
    return fused


def resolve_camera_indices(
    sample: dict[str, Any],
    project_root: Path,
    *,
    image_count: int | None = None,
) -> list[int]:
    """Resolve real camera ids for materialized image slots.

    Materialized 4x4 image filenames use slot names (`cam0`..`cam3`), while
    Alpamayo's text contract expects real camera ids (`0,1,2,6` for 4V).
    Prefer explicit corpus/metadata ids and fall back to the public 4V layout.
    """
    sample_input = sample.get("input") or {}
    raw_indices = sample_input.get("camera_indices")
    if raw_indices:
        return [int(value) for value in raw_indices]

    metadata_path = sample_input.get("metadata_path")
    if metadata_path:
        try:
            metadata = json.loads(_resolve_path(metadata_path, project_root).read_text(encoding="utf-8"))
            raw_indices = metadata.get("camera_indices")
            if raw_indices:
                return [int(value) for value in raw_indices]
        except Exception:  # noqa: BLE001
            pass

    camera_count = int(sample_input.get("camera_count") or 0)
    if camera_count == 4 or image_count == 16:
        return [0, 1, 2, 6]
    if camera_count > 0:
        return list(range(camera_count))
    if image_count is not None and image_count > 0:
        return list(range(max(int(image_count) // 4, 1)))
    return [0, 1, 2, 6]


def _metadata_payload(sample: dict[str, Any], project_root: Path) -> dict[str, Any]:
    sample_input = sample.get("input") or {}
    metadata_path = sample_input.get("metadata_path")
    if not metadata_path:
        return {}
    try:
        return json.loads(_resolve_path(metadata_path, project_root).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def resolve_image_relative_timestamps(
    sample: dict[str, Any],
    project_root: Path,
    *,
    camera_count: int,
    frames_per_camera: int,
) -> list[list[float]]:
    """Resolve per-camera frame times relative to the sample timestamp."""
    metadata = _metadata_payload(sample, project_root)
    sample_metadata = sample.get("metadata") or {}
    raw_rel = (
        metadata.get("relative_timestamps_us")
        or metadata.get("relative_timestamps_sec")
        or metadata.get("relative_timestamps")
    )
    if raw_rel is not None:
        arr = np.asarray(raw_rel, dtype=np.float32)
        if arr.size == camera_count * frames_per_camera:
            arr = arr.reshape(camera_count, frames_per_camera)
            if "relative_timestamps_us" in metadata:
                arr = arr / 1_000_000.0
            return arr.astype(np.float32).tolist()

    raw_abs = metadata.get("absolute_timestamps_us")
    if raw_abs is not None:
        arr = np.asarray(raw_abs, dtype=np.float32)
        if arr.size == camera_count * frames_per_camera:
            arr = arr.reshape(camera_count, frames_per_camera)
            t0_us = (
                metadata.get("sample_timestamp_us")
                or sample_metadata.get("sample_timestamp_us")
                or (sample.get("input") or {}).get("sample_timestamp_us")
            )
            if t0_us is not None:
                return ((arr - float(t0_us)) / 1_000_000.0).astype(np.float32).tolist()

    sample_input = sample.get("input") or {}
    raw_offsets = sample_input.get("frame_offsets_sec")
    if raw_offsets is None:
        try:
            raw_offsets = frame_offsets_from_sample(resolve_sample_path(sample, project_root))
        except Exception:  # noqa: BLE001
            raw_offsets = FRAME_OFFSETS_DEFAULT
    offsets = [float(value) for value in raw_offsets]
    if len(offsets) < frames_per_camera:
        offsets = [float(value) for value in FRAME_OFFSETS_DEFAULT[:frames_per_camera]]
    if len(offsets) < frames_per_camera:
        offsets = [float(index - frames_per_camera + 1) * 0.1 for index in range(frames_per_camera)]
    offsets = offsets[:frames_per_camera]
    return [list(offsets) for _ in range(camera_count)]


def _pad_camera_metadata(
    items: list[tuple[list[int], list[list[float]], int]],
) -> dict[str, torch.Tensor]:
    max_cameras = max(len(camera_indices) for camera_indices, _, _ in items)
    max_frames = max(max(max(len(row) for row in relative_times), 1) for _, relative_times, _ in items)
    camera_indices_tensor = torch.zeros((len(items), max_cameras), dtype=torch.long)
    relative_timestamps_tensor = torch.zeros((len(items), max_cameras, max_frames), dtype=torch.float32)
    camera_counts = torch.zeros((len(items),), dtype=torch.long)
    frames_per_camera = torch.zeros((len(items),), dtype=torch.long)
    for row_index, (camera_indices, relative_times, frame_count) in enumerate(items):
        camera_count = len(camera_indices)
        camera_counts[row_index] = camera_count
        frames_per_camera[row_index] = int(frame_count)
        camera_indices_tensor[row_index, :camera_count] = torch.tensor(camera_indices, dtype=torch.long)
        for camera_offset, row_times in enumerate(relative_times[:camera_count]):
            count = min(len(row_times), max_frames)
            if count > 0:
                relative_timestamps_tensor[row_index, camera_offset, :count] = torch.tensor(
                    row_times[:count],
                    dtype=torch.float32,
                )
    return {
        "camera_indices": camera_indices_tensor,
        "relative_timestamps": relative_timestamps_tensor,
        "camera_counts": camera_counts,
        "frames_per_camera": frames_per_camera,
    }


def build_traj_only_prompt(
    sample: dict[str, Any],
    project_root: Path,
    *,
    ego_history_xyz: np.ndarray | None = None,
) -> str:
    """Create the minimal A0 prompt for future-trajectory prediction only."""
    history_xyz = ego_history_xyz if ego_history_xyz is not None else load_ego_history_xyz(sample, project_root)
    history_text = format_history_text(history_xyz)
    return f"{history_text}\n<|question_start|>Predict the future trajectory tokens only.<|question_end|>"


def build_messages(
    prompt_text: str,
    image_count: int,
    completion_text: str | None = None,
    *,
    target_text: str | None = None,
    assistant_prefix: str = "<|cot_start|>",
    image_prompt_style: str = "compact",
    camera_indices: list[int] | tuple[int, ...] | None = None,
    num_frames_per_camera: int = 4,
) -> list[dict[str, Any]]:
    """Construct the multimodal chat message structure for one sample."""
    if completion_text is None and target_text is not None:
        completion_text = target_text
    style = str(image_prompt_style or "compact").strip().lower()
    if style in {"compact", "unlabeled", "plain"}:
        user_content: list[dict[str, Any]] = [{"type": "image"} for _ in range(image_count)]
    elif style in {"camera_labeled", "alpamayo_camera_labeled", "official"}:
        resolved_camera_indices = [int(value) for value in (camera_indices or [])]
        if not resolved_camera_indices:
            resolved_camera_indices = [0, 1, 2, 6] if image_count % 4 == 0 else list(range(max(image_count, 1)))
        frames_per_camera = max(int(num_frames_per_camera or 4), 1)
        if len(resolved_camera_indices) * frames_per_camera != image_count:
            frames_per_camera = max(image_count // max(len(resolved_camera_indices), 1), 1)
        user_content = []
        emitted = 0
        for camera_index in resolved_camera_indices:
            if emitted >= image_count:
                break
            camera_name = CAMERA_DISPLAY_NAMES.get(int(camera_index), f"Camera {int(camera_index)}")
            user_content.append({"type": "text", "text": f"{camera_name}: "})
            for frame_index in range(frames_per_camera):
                if emitted >= image_count:
                    break
                user_content.append({"type": "text", "text": f"frame {frame_index} "})
                user_content.append({"type": "image"})
                emitted += 1
        while emitted < image_count:
            user_content.append({"type": "text", "text": f"frame {emitted} "})
            user_content.append({"type": "image"})
            emitted += 1
    else:
        raise ValueError(f"Unsupported image_prompt_style={image_prompt_style!r}")
    user_content.append({"type": "text", "text": prompt_text})
    assistant_text = assistant_prefix
    if completion_text is not None:
        assistant_text = f"{assistant_text}{completion_text}"
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]


def _encode_messages(
    processor,
    messages_batch: list[list[dict[str, Any]]],
    image_batch: list[list[Image.Image]],
    max_length: int,
    *,
    continue_final_message: bool = False,
):
    texts = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=continue_final_message,
        )
        for messages in messages_batch
    ]
    return processor(
        text=texts,
        images=image_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _labels_from_prompt_and_full(prompt_batch, full_batch) -> torch.Tensor:
    labels = full_batch["input_ids"].clone()
    labels[full_batch["attention_mask"] == 0] = IGNORE_INDEX
    prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
    full_lengths = full_batch["attention_mask"].sum(dim=1)
    for row_index, prompt_length in enumerate(prompt_lengths.tolist()):
        prompt_length = int(prompt_length)
        full_length = int(full_lengths[row_index].item())
        if prompt_length >= full_length:
            raise ValueError(
                "Prompt length consumed the entire completion span; check chat template alignment and max_length."
            )
        prompt_prefix = prompt_batch["input_ids"][row_index, :prompt_length]
        full_prefix = full_batch["input_ids"][row_index, :prompt_length]
        if not torch.equal(prompt_prefix, full_prefix):
            raise ValueError(
                "Prompt/full chat template prefixes diverged; prompt masking would corrupt supervision labels."
            )
        labels[row_index, : int(prompt_length)] = IGNORE_INDEX
        if not torch.any(labels[row_index] != IGNORE_INDEX):
            raise ValueError("No trainable completion tokens remain after prompt masking.")
    return labels


def _build_target_layout(
    tokenizer,
    cot_text: str | None,
    traj_token_ids: list[int],
    *,
    loss_ignore_prefix_len: int = 0,
) -> TargetLayout:
    cot_text = str(cot_text or "").strip()
    cot_content_ids = tokenizer.encode(cot_text, add_special_tokens=False) if cot_text else []
    cot_end_ids = tokenizer.encode("<|cot_end|>", add_special_tokens=False)
    traj_prefix_ids = tokenizer.encode("<|traj_future_start|>", add_special_tokens=False)
    traj_suffix_ids = tokenizer.encode("<|traj_future_end|>", add_special_tokens=False)
    traj_token_text = "".join(discrete_traj_token(int(token_id)) for token_id in traj_token_ids)
    completion_text = f"{cot_text}<|cot_end|><|traj_future_start|>{traj_token_text}<|traj_future_end|>"
    return TargetLayout(
        completion_text=completion_text,
        cot_content_len=len(cot_content_ids),
        cot_suffix_len=len(cot_end_ids),
        traj_prefix_len=len(traj_prefix_ids),
        traj_content_len=len(traj_token_ids),
        traj_suffix_len=len(traj_suffix_ids),
        loss_ignore_prefix_len=max(int(loss_ignore_prefix_len), 0),
    )


def _build_traj_only_target_layout(
    tokenizer,
    traj_token_ids: list[int],
    *,
    loss_ignore_prefix_len: int = 0,
) -> TargetLayout:
    traj_suffix_ids = tokenizer.encode("<|traj_future_end|>", add_special_tokens=False)
    traj_token_text = "".join(discrete_traj_token(int(token_id)) for token_id in traj_token_ids)
    completion_text = f"{traj_token_text}<|traj_future_end|>"
    return TargetLayout(
        completion_text=completion_text,
        cot_content_len=0,
        cot_suffix_len=0,
        traj_prefix_len=0,
        traj_content_len=len(traj_token_ids),
        traj_suffix_len=len(traj_suffix_ids),
        loss_ignore_prefix_len=max(int(loss_ignore_prefix_len), 0),
    )


def _target_layout_masks(labels: torch.Tensor, layouts: list[TargetLayout]) -> dict[str, Any]:
    cot_span_mask = torch.zeros_like(labels, dtype=torch.bool)
    cot_content_mask = torch.zeros_like(labels, dtype=torch.bool)
    traj_span_mask = torch.zeros_like(labels, dtype=torch.bool)
    traj_token_mask = torch.zeros_like(labels, dtype=torch.bool)
    format_token_mask = torch.zeros_like(labels, dtype=torch.bool)
    cot_content_positions: list[list[int]] = []
    text_topk_positions: list[list[int]] = []
    boundary_hidden_positions: list[list[int]] = []

    for row_index, layout in enumerate(layouts):
        valid_positions = torch.nonzero(labels[row_index] != IGNORE_INDEX, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            cot_content_positions.append([])
            text_topk_positions.append([])
            boundary_hidden_positions.append([-1, -1, -1])
            continue

        cursor = int(valid_positions[0].item())
        valid_end = int(valid_positions[-1].item()) + 1
        row_cot_positions: list[int] = []
        row_text_topk_positions: list[int] = []
        row_boundary_positions = [-1, -1, -1]

        cot_content_len = min(layout.cot_content_len, max(valid_end - cursor, 0))
        if cot_content_len > 0:
            cot_content_mask[row_index, cursor : cursor + cot_content_len] = True
            row_cot_positions = list(range(cursor, cursor + cot_content_len))
            row_text_topk_positions.extend(row_cot_positions)

        cot_span_len = min(layout.cot_span_len, max(valid_end - cursor, 0))
        if cot_span_len > 0:
            cot_span_mask[row_index, cursor : cursor + cot_span_len] = True
        cot_suffix_start = cursor + cot_content_len
        cot_suffix_len = min(layout.cot_suffix_len, max(valid_end - cot_suffix_start, 0))
        if cot_suffix_len > 0:
            format_token_mask[row_index, cot_suffix_start : cot_suffix_start + cot_suffix_len] = True
            row_text_topk_positions.extend(range(cot_suffix_start, cot_suffix_start + cot_suffix_len))
            row_boundary_positions[0] = cot_suffix_start + cot_suffix_len - 1
        cot_content_positions.append(row_cot_positions)
        cursor += cot_span_len

        traj_span_len = min(layout.traj_span_len, max(valid_end - cursor, 0))
        if traj_span_len > 0:
            traj_span_mask[row_index, cursor : cursor + traj_span_len] = True
        traj_prefix_len = min(layout.traj_prefix_len, max(valid_end - cursor, 0))
        if traj_prefix_len > 0:
            format_token_mask[row_index, cursor : cursor + traj_prefix_len] = True
            row_text_topk_positions.extend(range(cursor, cursor + traj_prefix_len))
            traj_start_position = cursor + traj_prefix_len - 1
            row_boundary_positions[1] = traj_start_position
            row_boundary_positions[2] = traj_start_position
        text_topk_positions.append(row_text_topk_positions)
        boundary_hidden_positions.append(row_boundary_positions)

        traj_token_start = cursor + min(layout.traj_prefix_len, traj_span_len)
        traj_token_len = min(layout.traj_content_len, max(valid_end - traj_token_start, 0))
        if traj_token_len > 0:
            traj_token_mask[row_index, traj_token_start : traj_token_start + traj_token_len] = True
        traj_suffix_start = traj_token_start + traj_token_len
        traj_suffix_len = min(layout.traj_suffix_len, max(valid_end - traj_suffix_start, 0))
        if traj_suffix_len > 0:
            format_token_mask[row_index, traj_suffix_start : traj_suffix_start + traj_suffix_len] = True

    return {
        "cot_span_mask": cot_span_mask,
        "cot_content_mask": cot_content_mask,
        "traj_span_mask": traj_span_mask,
        "traj_token_mask": traj_token_mask,
        "format_token_mask": format_token_mask,
        "cot_content_positions": cot_content_positions,
        "text_topk_positions": text_topk_positions,
        "boundary_hidden_positions": boundary_hidden_positions,
    }


def _apply_layout_loss_ignore_prefix(
    labels: torch.Tensor,
    masks: dict[str, Any],
    layouts: list[TargetLayout],
) -> None:
    """Mask prefix completion tokens that are context-only, e.g. DAgger prefixes.

    DAgger rows can include a student-generated CoT and a short student-generated
    trajectory prefix in the assistant span. Those tokens must be visible as
    conditioning context, but the supervised loss should start at the teacher
    continuation. The layout is computed before this mutation so token offsets
    still line up with the full assistant completion.
    """
    mask_keys = (
        "cot_span_mask",
        "cot_content_mask",
        "traj_span_mask",
        "traj_token_mask",
        "format_token_mask",
    )
    for row_index, layout in enumerate(layouts):
        prefix_len = max(int(layout.loss_ignore_prefix_len), 0)
        if prefix_len <= 0:
            continue
        valid_positions = torch.nonzero(labels[row_index] != IGNORE_INDEX, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue
        prefix_positions = valid_positions[: min(prefix_len, int(valid_positions.numel()))]
        labels[row_index, prefix_positions] = IGNORE_INDEX
        for key in mask_keys:
            value = masks.get(key)
            if isinstance(value, torch.Tensor):
                value[row_index, prefix_positions] = False


def _build_label_token_weights(
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    token_weight_map: Mapping[int, float] | None,
) -> torch.Tensor | None:
    if not token_weight_map:
        return None
    weights = torch.ones_like(labels, dtype=torch.float32)
    active_positions = torch.nonzero(token_mask & (labels != IGNORE_INDEX), as_tuple=False)
    for row_index, token_index in active_positions.tolist():
        label_id = int(labels[row_index, token_index].item())
        weights[row_index, token_index] = float(token_weight_map.get(label_id, 1.0))
    return weights


def _teacher_traj_token_kd_weights_from_sample(sample_weights: Mapping[str, Any]) -> np.ndarray | None:
    """Read optional per-body-token teacher trajectory KD weights from a corpus record."""
    raw_weights = None
    for key in ("teacher_traj_token_kd_weights", "teacher_traj_kd_token_weights", "teacher_traj_token_weights"):
        if key in sample_weights and sample_weights[key] is not None:
            raw_weights = sample_weights[key]
            break
    if raw_weights is None:
        return None
    if isinstance(raw_weights, (int, float)):
        return np.full((128,), float(raw_weights), dtype=np.float32)
    try:
        weights = np.asarray(raw_weights, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    return weights if weights.size > 0 else None


def _pad_ego_history_batch(history_items: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    max_steps = max(item.shape[0] for item in history_items)
    max_dim = max(item.shape[-1] for item in history_items)
    history = torch.zeros((len(history_items), max_steps, max_dim), dtype=torch.float32)
    history_mask = torch.zeros((len(history_items), max_steps), dtype=torch.bool)
    for row_index, item in enumerate(history_items):
        tensor = torch.from_numpy(item).float()
        history[row_index, : tensor.shape[0], : tensor.shape[1]] = tensor
        history_mask[row_index, : tensor.shape[0]] = True
    return history, history_mask


def _pad_future_xyz_batch(future_items: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    max_steps = max(item.shape[0] for item in future_items)
    max_dim = max(item.shape[-1] for item in future_items)
    future = torch.zeros((len(future_items), max_steps, max_dim), dtype=torch.float32)
    future_mask = torch.zeros((len(future_items), max_steps), dtype=torch.bool)
    for row_index, item in enumerate(future_items):
        tensor = torch.from_numpy(item).float()
        future[row_index, : tensor.shape[0], : tensor.shape[1]] = tensor
        future_mask[row_index, : tensor.shape[0]] = True
    return future, future_mask


def _teacher_signal_from_sample(sample: dict[str, Any], project_root: Path) -> dict[str, np.ndarray | int] | None:
    teacher_target = sample.get("teacher_target") or {}
    topk_path = teacher_target.get("topk_logits_path")
    topk_ids_path = teacher_target.get("topk_ids_path") or teacher_target.get("teacher_text_topk_ids_path")
    topk_logprobs_path = teacher_target.get("topk_logprobs_path") or teacher_target.get(
        "teacher_text_topk_logprobs_path"
    )
    pooled_hidden_path = teacher_target.get("pooled_hidden_path")
    if not topk_path and not (topk_ids_path and topk_logprobs_path) and not pooled_hidden_path:
        return None

    signal: dict[str, np.ndarray | int] = {}
    if topk_path:
        resolved_topk = _resolve_path(topk_path, project_root)
        if resolved_topk.exists():
            topk_npz = np.load(resolved_topk)
            index_key = "topk_indices" if "topk_indices" in topk_npz else "topk_ids"
            logprob_key = "topk_logprobs" if "topk_logprobs" in topk_npz else "topk_logits"
            signal["topk_indices"] = topk_npz[index_key].astype(np.int64)
            signal["topk_logprobs"] = topk_npz[logprob_key].astype(np.float32)
            if "target_token_count" in topk_npz:
                signal["target_token_count"] = int(np.asarray(topk_npz["target_token_count"]).reshape(-1)[0])
            else:
                signal["target_token_count"] = int(np.asarray(signal["topk_indices"]).shape[0])
    if "topk_indices" not in signal and topk_ids_path and topk_logprobs_path:
        resolved_ids = _resolve_path(topk_ids_path, project_root)
        resolved_logprobs = _resolve_path(topk_logprobs_path, project_root)
        if resolved_ids.exists() and resolved_logprobs.exists():
            signal["topk_indices"] = np.load(resolved_ids).astype(np.int64)
            signal["topk_logprobs"] = np.load(resolved_logprobs).astype(np.float32)
            explicit_count = teacher_target.get("target_token_count") or teacher_target.get(
                "teacher_text_topk_num_positions"
            )
            signal["target_token_count"] = (
                int(explicit_count) if explicit_count not in (None, "") else int(signal["topk_indices"].shape[0])
            )
    if pooled_hidden_path:
        resolved_hidden = _resolve_path(pooled_hidden_path, project_root)
        if resolved_hidden.exists():
            signal["pooled_hidden"] = np.load(resolved_hidden).astype(np.float32)
    return signal or None


def _teacher_boundary_hidden_signal_from_sample(
    sample: dict[str, Any],
    project_root: Path,
) -> dict[str, np.ndarray] | None:
    """Load cached teacher boundary hidden states for CoT/action interface alignment."""
    teacher_cache = sample.get("teacher_cache") or {}
    raw_paths = dict(teacher_cache.get("boundary_hidden_paths") or {})
    teacher_target = sample.get("teacher_target") or {}
    for name in BOUNDARY_HIDDEN_NAMES:
        raw_paths.setdefault(name, teacher_target.get(f"{name}_hidden_path"))
        raw_paths.setdefault(name, teacher_cache.get(f"{name}_hidden_path"))

    hidden_items: list[np.ndarray | None] = []
    hidden_dim: int | None = None
    mask = np.zeros((len(BOUNDARY_HIDDEN_NAMES),), dtype=bool)
    for index, name in enumerate(BOUNDARY_HIDDEN_NAMES):
        raw_path = raw_paths.get(name)
        if raw_path in (None, ""):
            hidden_items.append(None)
            continue
        try:
            path = _resolve_path(raw_path, project_root)
            if not path.exists():
                hidden_items.append(None)
                continue
            hidden = np.load(path).astype(np.float32).reshape(-1)
        except Exception:  # noqa: BLE001
            hidden_items.append(None)
            continue
        if hidden_dim is None:
            hidden_dim = int(hidden.shape[-1])
        if int(hidden.shape[-1]) != hidden_dim:
            hidden_items.append(None)
            continue
        hidden_items.append(hidden)
        mask[index] = True

    if hidden_dim is None or not mask.any():
        return None
    stacked = np.zeros((len(BOUNDARY_HIDDEN_NAMES), hidden_dim), dtype=np.float32)
    for index, hidden in enumerate(hidden_items):
        if hidden is not None:
            stacked[index] = hidden
    return {"hidden": stacked, "mask": mask}


def _teacher_action_traj_xyz_from_sample(sample: dict[str, Any], project_root: Path) -> np.ndarray | None:
    """Load teacher action-expert trajectory xyz from the raw Alpamayo output JSON."""
    teacher_cache = sample.get("teacher_cache") or {}
    raw_path = teacher_cache.get("text_raw_json_path")
    if raw_path in (None, ""):
        return None
    try:
        path = _resolve_path(raw_path, project_root)
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = (payload.get("results") or [None])[0]
        if not isinstance(result, dict):
            return None
        xyz = np.asarray(result.get("pred_xyz"), dtype=np.float32).reshape(-1, 64, 3)[0]
    except Exception:  # noqa: BLE001
        return None
    if xyz.ndim != 2 or xyz.shape[0] <= 0 or xyz.shape[-1] < 2:
        return None
    if xyz.shape[-1] < 3:
        padded = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        padded[:, : xyz.shape[-1]] = xyz
        xyz = padded
    return xyz[:64, :3].astype(np.float32)


def _pad_teacher_action_traj_batch(items: list[np.ndarray | None]) -> dict[str, torch.Tensor]:
    steps = 64
    traj = torch.zeros((len(items), steps, 3), dtype=torch.float32)
    mask = torch.zeros((len(items), steps), dtype=torch.bool)
    available = torch.zeros((len(items),), dtype=torch.bool)
    for row_index, item in enumerate(items):
        if item is None:
            continue
        arr = np.asarray(item, dtype=np.float32)
        count = min(int(arr.shape[0]), steps)
        if count <= 0:
            continue
        traj[row_index, :count, : min(int(arr.shape[-1]), 3)] = torch.from_numpy(arr[:count, :3]).float()
        mask[row_index, :count] = True
        available[row_index] = True
    return {
        "teacher_action_traj_xyz": traj,
        "teacher_action_traj_mask": mask,
        "teacher_action_traj_available": available,
    }


def _teacher_traj15_signal_from_sample(
    sample: dict[str, Any],
    *,
    teacher_traj_cache_dir: Path | None,
    teacher_traj_hidden_source: str = "hidden",
    teacher_traj_latent_suffix: str = "lat32",
    project_root: Path | None = None,
) -> dict[str, np.ndarray | float] | None:
    direct_target = sample.get("teacher_traj_target") or {}
    if direct_target:
        status = str(direct_target.get("status") or direct_target.get("inference_status") or "ready").strip().lower()
        valid = bool(direct_target.get("valid", True))
        if not valid or status not in {"", "ready", "ok"}:
            return None

        def _direct_path(raw_path: str | Path | None) -> Path | None:
            if raw_path in (None, ""):
                return None
            remapped = remap_external_path(raw_path)
            if remapped is None:
                return None
            path = Path(remapped)
            if not path.is_absolute() and project_root is not None:
                path = project_root / path
            return path

        signal: dict[str, np.ndarray | float] = {}
        token_ids_path = _direct_path(
            direct_target.get("token_ids_path")
            or direct_target.get("teacher_future_token_ids_path")
            or direct_target.get("tokens_path")
        )
        if token_ids_path is not None and token_ids_path.exists():
            signal["token_ids"] = np.load(token_ids_path).astype(np.int32)
        elif direct_target.get("token_ids") is not None:
            signal["token_ids"] = np.asarray(direct_target.get("token_ids"), dtype=np.int32)

        resolved_source = str(teacher_traj_hidden_source).strip().lower()
        hidden_path_key = "latent_path" if resolved_source == "latent" else "hidden_path"
        selected_hidden_path = _direct_path(
            direct_target.get(hidden_path_key)
            or direct_target.get("teacher_traj_hidden_path")
            or direct_target.get("hidden_path")
        )
        loaded_hidden = None
        if selected_hidden_path is not None and selected_hidden_path.exists():
            loaded_hidden = np.load(selected_hidden_path).astype(np.float32)
            signal["hidden"] = loaded_hidden
            signal["hidden_source"] = resolved_source
        raw_hidden_path = _direct_path(direct_target.get("hidden_raw_path") or direct_target.get("raw_hidden_path"))
        if raw_hidden_path is not None and raw_hidden_path.exists():
            signal["hidden_raw"] = np.load(raw_hidden_path).astype(np.float32)
        elif loaded_hidden is not None and resolved_source != "latent":
            signal["hidden_raw"] = loaded_hidden

        topk_path = _direct_path(
            direct_target.get("topk_logits_path")
            or direct_target.get("topk_path")
            or direct_target.get("teacher_traj_topk_ids_path")
        )
        loaded_topk = False
        if topk_path is not None and topk_path.exists():
            topk_npz = np.load(topk_path)
            if hasattr(topk_npz, "files"):
                index_key = "topk_indices" if "topk_indices" in topk_npz.files else "topk_ids"
                logprob_key = "topk_logprobs" if "topk_logprobs" in topk_npz.files else "topk_logits"
                signal["topk_indices"] = topk_npz[index_key].astype(np.int32)
                signal["topk_logprobs"] = topk_npz[logprob_key].astype(np.float32)
                if "target_token_ids" in topk_npz.files and "token_ids" not in signal:
                    signal["token_ids"] = topk_npz["target_token_ids"].astype(np.int32)
                loaded_topk = True
        if not loaded_topk:
            topk_ids_path = _direct_path(direct_target.get("topk_ids_path") or direct_target.get("topk_indices_path"))
            topk_logprobs_path = _direct_path(
                direct_target.get("topk_logprobs_path") or direct_target.get("topk_logits_array_path")
            )
            if topk_ids_path is not None and topk_logprobs_path is not None:
                if topk_ids_path.exists() and topk_logprobs_path.exists():
                    signal["topk_indices"] = np.load(topk_ids_path).astype(np.int32)
                    signal["topk_logprobs"] = np.load(topk_logprobs_path).astype(np.float32)

        if "teacher_traj_ade_m" in direct_target:
            signal["teacher_traj_ade_m"] = float(direct_target.get("teacher_traj_ade_m") or 0.0)
        if "teacher_traj_fde_m" in direct_target:
            signal["teacher_traj_fde_m"] = float(direct_target.get("teacher_traj_fde_m") or 0.0)
        signal["quality_multiplier"] = float(direct_target.get("quality_multiplier", 1.0) or 1.0)
        return signal or None

    if teacher_traj_cache_dir is None:
        return None
    sample_id = str(sample.get("sample_id") or "").strip()
    if not sample_id:
        return None

    hidden_path = teacher_traj_cache_dir / "hidden" / f"{sample_id}.teacher_traj15.hidden.npy"
    latent_path = (
        teacher_traj_cache_dir / "latent" / f"{sample_id}.teacher_traj15.{teacher_traj_latent_suffix}.npy"
    )
    tokens_path = teacher_traj_cache_dir / "tokens" / f"{sample_id}.teacher_traj15.tokens.npy"
    topk_path = teacher_traj_cache_dir / "topk" / f"{sample_id}.teacher_traj15.topk_logits.npz"
    output_path = teacher_traj_cache_dir / "outputs" / f"{sample_id}.teacher_traj15.json"
    selected_hidden_path = latent_path if str(teacher_traj_hidden_source).strip().lower() == "latent" else hidden_path
    if not selected_hidden_path.exists() and not tokens_path.exists() and not topk_path.exists():
        return None

    signal: dict[str, np.ndarray | float] = {}
    if tokens_path.exists():
        signal["token_ids"] = np.load(tokens_path).astype(np.int32)
    if selected_hidden_path.exists():
        signal["hidden"] = np.load(selected_hidden_path).astype(np.float32)
        signal["hidden_source"] = str(teacher_traj_hidden_source).strip().lower()
    if hidden_path.exists():
        signal["hidden_raw"] = np.load(hidden_path).astype(np.float32)
    if topk_path.exists():
        topk_npz = np.load(topk_path)
        signal["topk_indices"] = topk_npz["topk_indices"].astype(np.int32)
        signal["topk_logprobs"] = topk_npz["topk_logprobs"].astype(np.float32)
        if "target_token_ids" in topk_npz and "token_ids" not in signal:
            signal["token_ids"] = topk_npz["target_token_ids"].astype(np.int32)
    quality_multiplier = 1.0
    if output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
        if str(output.get("status", "")).strip().lower() != "ready":
            return None
        signal["teacher_traj_ade_m"] = float(output.get("best_candidate_ade_m", 0.0) or 0.0)
        signal["teacher_traj_fde_m"] = float(output.get("best_candidate_fde_m", 0.0) or 0.0)
        quality_multiplier = float(output.get("teacher_quality_multiplier", 1.0) or 1.0)
    signal["quality_multiplier"] = quality_multiplier
    return signal or None


def _pad_teacher_signal_batch(
    signal_items: list[dict[str, np.ndarray | int] | None],
    target_positions: list[list[int]],
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    topk_ready = [item for item in signal_items if item and "topk_indices" in item and "topk_logprobs" in item]
    if topk_ready:
        topk = int(topk_ready[0]["topk_indices"].shape[-1])
        max_tokens = max(
            min(
                int(item["target_token_count"]),
                len(target_positions[row_index]),
            )
            for row_index, item in enumerate(signal_items)
            if item and "topk_indices" in item and "topk_logprobs" in item
        )
        topk_indices = torch.zeros((len(signal_items), max_tokens, topk), dtype=torch.long)
        topk_logprobs = torch.zeros((len(signal_items), max_tokens, topk), dtype=torch.float32)
        topk_mask = torch.zeros((len(signal_items), max_tokens), dtype=torch.bool)
        topk_positions = torch.full((len(signal_items), max_tokens), -1, dtype=torch.long)
        for row_index, item in enumerate(signal_items):
            if item is None or "topk_indices" not in item or "topk_logprobs" not in item:
                continue
            token_count = min(int(item["target_token_count"]), len(target_positions[row_index]), max_tokens)
            if token_count <= 0:
                continue
            topk_indices[row_index, :token_count] = torch.from_numpy(item["topk_indices"][:token_count]).long()
            topk_logprobs[row_index, :token_count] = torch.from_numpy(item["topk_logprobs"][:token_count]).float()
            topk_mask[row_index, :token_count] = True
            topk_positions[row_index, :token_count] = torch.tensor(
                target_positions[row_index][:token_count],
                dtype=torch.long,
            )
        outputs.update(
            {
                "teacher_topk_indices": topk_indices,
                "teacher_topk_logprobs": topk_logprobs,
                "teacher_topk_mask": topk_mask,
                "teacher_topk_positions": topk_positions,
            }
        )

    hidden_ready = [item for item in signal_items if item and "pooled_hidden" in item]
    if hidden_ready:
        hidden_dim = int(hidden_ready[0]["pooled_hidden"].shape[-1])
        teacher_pooled_hidden = torch.zeros((len(signal_items), hidden_dim), dtype=torch.float32)
        teacher_hidden_mask = torch.zeros((len(signal_items),), dtype=torch.bool)
        for row_index, item in enumerate(signal_items):
            if item is None or "pooled_hidden" not in item:
                continue
            teacher_pooled_hidden[row_index] = torch.from_numpy(item["pooled_hidden"]).float()
            teacher_hidden_mask[row_index] = True
        outputs["teacher_pooled_hidden"] = teacher_pooled_hidden
        outputs["teacher_pooled_hidden_mask"] = teacher_hidden_mask

    return outputs


def _pad_teacher_boundary_hidden_batch(
    signal_items: list[dict[str, np.ndarray] | None],
    boundary_positions: list[list[int]],
) -> dict[str, torch.Tensor]:
    """Pad teacher boundary hidden states and their student sequence positions."""
    outputs: dict[str, torch.Tensor] = {}
    hidden_ready = [item for item in signal_items if item is not None and "hidden" in item]
    if not hidden_ready:
        return outputs

    hidden_dim = int(np.asarray(hidden_ready[0]["hidden"]).shape[-1])
    teacher_hidden = torch.zeros((len(signal_items), len(BOUNDARY_HIDDEN_NAMES), hidden_dim), dtype=torch.float32)
    teacher_mask = torch.zeros((len(signal_items), len(BOUNDARY_HIDDEN_NAMES)), dtype=torch.bool)
    teacher_positions = torch.full((len(signal_items), len(BOUNDARY_HIDDEN_NAMES)), -1, dtype=torch.long)
    available = torch.zeros((len(signal_items),), dtype=torch.bool)

    for row_index, item in enumerate(signal_items):
        if item is None or "hidden" not in item:
            continue
        hidden = np.asarray(item["hidden"], dtype=np.float32)
        if hidden.ndim != 2 or int(hidden.shape[0]) != len(BOUNDARY_HIDDEN_NAMES) or int(hidden.shape[-1]) != hidden_dim:
            continue
        mask = np.asarray(item.get("mask", np.ones((len(BOUNDARY_HIDDEN_NAMES),), dtype=bool)), dtype=bool).reshape(-1)
        if int(mask.shape[0]) != len(BOUNDARY_HIDDEN_NAMES):
            continue
        positions = list(boundary_positions[row_index]) if row_index < len(boundary_positions) else [-1, -1, -1]
        if len(positions) != len(BOUNDARY_HIDDEN_NAMES):
            positions = [-1, -1, -1]
        teacher_hidden[row_index] = torch.from_numpy(hidden).float()
        teacher_mask[row_index] = torch.from_numpy(mask)
        teacher_positions[row_index] = torch.tensor(positions, dtype=torch.long)
        available[row_index] = bool(mask.any())

    if not bool(available.any()):
        return outputs
    outputs["teacher_text_boundary_hidden"] = teacher_hidden
    outputs["teacher_text_boundary_hidden_mask"] = teacher_mask
    outputs["teacher_text_boundary_hidden_positions"] = teacher_positions
    outputs["teacher_text_boundary_hidden_available"] = available
    return outputs


def _pad_teacher_traj_topk_batch(
    signal_items: list[dict[str, np.ndarray | float] | None],
    traj_token_masks: torch.Tensor,
    tokenizer: Any,
) -> dict[str, torch.Tensor]:
    """Pad teacher trajectory top-k caches for a teacher trajectory view."""
    outputs: dict[str, torch.Tensor] = {}
    topk_ready = [item for item in signal_items if item is not None and "topk_indices" in item and "topk_logprobs" in item]
    if not topk_ready:
        return outputs
    topk = int(np.asarray(topk_ready[0]["topk_indices"]).shape[-1])
    max_tokens = max(
        min(
            int(np.asarray(item["topk_indices"]).shape[0]),
            int(torch.count_nonzero(traj_token_masks[row_index]).item()),
        )
        for row_index, item in enumerate(signal_items)
        if item is not None and "topk_indices" in item and "topk_logprobs" in item
    )
    teacher_traj_topk_indices = torch.zeros((len(signal_items), max_tokens, topk), dtype=torch.long)
    teacher_traj_topk_logprobs = torch.zeros((len(signal_items), max_tokens, topk), dtype=torch.float32)
    teacher_traj_topk_mask = torch.zeros((len(signal_items), max_tokens), dtype=torch.bool)
    traj_token_start_idx = getattr(tokenizer, "traj_token_start_idx", None)
    if not isinstance(traj_token_start_idx, int) or traj_token_start_idx < 0:
        raise ValueError("Tokenizer is missing traj_token_start_idx required for teacher traj cache.")
    for row_index, item in enumerate(signal_items):
        if item is None or "topk_indices" not in item or "topk_logprobs" not in item:
            continue
        raw_indices = np.asarray(item["topk_indices"], dtype=np.int32)
        raw_logprobs = np.asarray(item["topk_logprobs"], dtype=np.float32)
        token_count = min(
            int(raw_indices.shape[0]),
            int(torch.count_nonzero(traj_token_masks[row_index]).item()),
            max_tokens,
        )
        if token_count <= 0:
            continue
        teacher_traj_topk_indices[row_index, :token_count] = torch.from_numpy(
            raw_indices[:token_count] + int(traj_token_start_idx)
        ).long()
        teacher_traj_topk_logprobs[row_index, :token_count] = torch.from_numpy(raw_logprobs[:token_count]).float()
        teacher_traj_topk_mask[row_index, :token_count] = True
    outputs["teacher_traj_topk_indices"] = teacher_traj_topk_indices
    outputs["teacher_traj_topk_logprobs"] = teacher_traj_topk_logprobs
    outputs["teacher_traj_topk_mask"] = teacher_traj_topk_mask
    return outputs


def _action_aux_label(sample: dict[str, Any]) -> str | None:
    teacher_target = sample.get("teacher_target") or {}
    derived = sample.get("derived") or {}
    for value in (
        teacher_target.get("teacher_motion_class"),
        derived.get("teacher_motion_class"),
        derived.get("gt_motion_class"),
        derived.get("human_motion_class"),
    ):
        if value:
            return str(value)
    return None


@dataclass(slots=True)
class DistillationCollator:
    tokenizer: Any
    processor: Any
    project_root: Path
    max_length: int = 4096
    prompt_mode: str = "joint"
    target_mode: str = "joint"
    teacher_pair_target: bool = False
    enable_teacher_view: bool = True
    enable_action_aux: bool = True
    traj_token_weight_map: Mapping[int, float] | None = None
    teacher_traj_cache_dir: Path | None = None
    teacher_traj_hidden_source: str = "hidden"
    teacher_traj_latent_suffix: str = "lat32"
    hard_view_uses_teacher_cot: bool = False
    teacher_view_force_enable: bool = False
    teacher_view_uses_teacher_traj: bool = False
    teacher_view_default_traj_weight: float = 0.0
    teacher_traj_topk_on_teacher_view: bool = False
    image_prompt_style: str = "compact"
    prompt_text_style: str = "numeric_history_question"
    fuse_history_tokens: bool = False
    image_ablation: str = "normal"

    def __post_init__(self) -> None:
        style = str(self.image_prompt_style or "compact").strip().lower()
        aliases = {
            "plain": "compact",
            "unlabeled": "compact",
            "alpamayo_camera_labeled": "camera_labeled",
            "official": "camera_labeled",
        }
        style = aliases.get(style, style)
        if style not in {"compact", "camera_labeled"}:
            raise ValueError(f"Unsupported image_prompt_style={self.image_prompt_style!r}")
        self.image_prompt_style = style
        prompt_style = str(self.prompt_text_style or "numeric_history_question").strip().lower()
        prompt_aliases = {
            "numeric": "numeric_history_question",
            "numeric_history": "numeric_history_question",
            "question": "numeric_history_question",
            "official": "official_alpamayo",
            "alpamayo": "official_alpamayo",
        }
        prompt_style = prompt_aliases.get(prompt_style, prompt_style)
        if prompt_style not in {"numeric_history_question", "official_alpamayo"}:
            raise ValueError(f"Unsupported prompt_text_style={self.prompt_text_style!r}")
        self.prompt_text_style = prompt_style
        ablation = str(self.image_ablation or "normal").strip().lower()
        if ablation not in {"normal", "black", "gray", "noise", "camera_shuffle"}:
            raise ValueError(f"Unsupported image_ablation={self.image_ablation!r}")
        self.image_ablation = ablation

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_messages: list[list[dict[str, Any]]] = []
        full_messages: list[list[dict[str, Any]]] = []
        image_batch: list[list[Image.Image]] = []
        ego_histories: list[np.ndarray] = []
        ego_futures: list[np.ndarray] = []
        hard_layouts: list[TargetLayout] = []
        camera_metadata_items: list[tuple[list[int], list[list[float]], int]] = []

        sample_ids: list[str] = []
        splits: list[str] = []
        image_paths: list[list[str]] = []
        teacher_vs_gt_consistency_level: list[str] = []
        sample_provenance_flags: list[dict[str, Any]] = []
        action_labels: list[int] = []
        hard_cot_weights: list[float] = []
        traj_weights: list[float] = []
        teacher_view_allowed: list[bool] = []
        teacher_view_weight: list[float] = []
        action_aux_allowed: list[bool] = []
        action_aux_weight: list[float] = []
        hard_teacher_signal_items: list[dict[str, np.ndarray | int] | None] = []
        teacher_boundary_hidden_signal_items: list[dict[str, np.ndarray] | None] = []
        teacher_action_traj_items: list[np.ndarray | None] = []
        hard_teacher_signal_weights: list[float] = []
        hard_teacher_quality_multiplier: list[float] = []
        teacher_traj_token_weight_items: list[np.ndarray | None] = []

        teacher_prompt_messages: list[list[dict[str, Any]]] = []
        teacher_full_messages: list[list[dict[str, Any]]] = []
        teacher_image_batch: list[list[Image.Image]] = []
        teacher_layouts: list[TargetLayout] = []
        teacher_camera_metadata_items: list[tuple[list[int], list[list[float]], int]] = []
        teacher_signal_items: list[dict[str, np.ndarray | int] | None] = []
        teacher_view_traj_signal_items: list[dict[str, np.ndarray | float] | None] = []
        teacher_sample_ids: list[str] = []
        teacher_source_indices: list[int] = []
        teacher_seq_ce_weights: list[float] = []
        teacher_logit_kd_weights: list[float] = []
        teacher_traj_weights: list[float] = []
        teacher_quality_multiplier: list[float] = []
        teacher_traj_signal_items: list[dict[str, np.ndarray | float] | None] = []

        for sample_index, sample in enumerate(features):
            hard_target = sample.get("hard_target") or {}
            weights = sample.get("weights") or {}
            gate = sample.get("gate") or {}
            teacher_target = sample.get("teacher_target") or {}
            teacher_signal = _teacher_signal_from_sample(sample, self.project_root)
            teacher_boundary_hidden_signal = _teacher_boundary_hidden_signal_from_sample(sample, self.project_root)
            teacher_action_traj = _teacher_action_traj_xyz_from_sample(sample, self.project_root)
            teacher_traj_signal = _teacher_traj15_signal_from_sample(
                sample,
                teacher_traj_cache_dir=self.teacher_traj_cache_dir,
                teacher_traj_hidden_source=self.teacher_traj_hidden_source,
                teacher_traj_latent_suffix=self.teacher_traj_latent_suffix,
                project_root=self.project_root,
            )
            explicit_teacher_traj_weight = resolve_optional_loss_weight_value(weights, "teacher_traj_ce")
            teacher_traj_token_weight_items.append(_teacher_traj_token_kd_weights_from_sample(weights))

            traj_token_ids = load_traj_future_token_ids(hard_target, self.project_root)
            target_cot_text = hard_target.get("cot_text")
            target_provenance = "hard_target"
            if self.hard_view_uses_teacher_cot and not self.teacher_pair_target:
                teacher_pair_cot = str(teacher_target.get("cot_text") or "").strip()
                if not teacher_pair_cot:
                    raise ValueError(
                        f"Sample {sample.get('sample_id')} is missing teacher_target.cot_text required for hard_view_uses_teacher_cot"
                    )
                target_cot_text = teacher_pair_cot
                target_provenance = "teacher_cot_gt_traj"
            if self.teacher_pair_target:
                teacher_pair_traj_ids = [int(token_id) for token_id in (teacher_traj_signal or {}).get("token_ids", [])]
                if not teacher_pair_traj_ids:
                    raise ValueError(
                        f"Sample {sample.get('sample_id')} is missing teacher traj15 token_ids required for teacher_pair_target"
                    )
                teacher_pair_cot = str(teacher_target.get("cot_text") or "").strip()
                if not teacher_pair_cot:
                    raise ValueError(
                        f"Sample {sample.get('sample_id')} is missing teacher_target.cot_text required for teacher_pair_target"
                    )
                traj_token_ids = teacher_pair_traj_ids
                target_cot_text = teacher_pair_cot
                target_provenance = "teacher_pair"
            if not traj_token_ids:
                raise ValueError(f"Sample {sample.get('sample_id')} is missing target traj_future_token_ids")
            loss_ignore_prefix_len = int(hard_target.get("loss_ignore_completion_token_count") or 0)

            ego_history_xyz = load_ego_history_xyz(sample, self.project_root)
            if self.prompt_mode == "traj_only":
                prompt_text = build_traj_only_prompt(sample, self.project_root, ego_history_xyz=ego_history_xyz)
            else:
                prompt_text = build_user_prompt(
                    sample,
                    self.project_root,
                    ego_history_xyz=ego_history_xyz,
                    prompt_text_style=self.prompt_text_style,
                )
            ego_future_xyz = load_ego_future_xyz(sample, self.project_root)
            image_ablation = str(sample.get("_image_ablation") or self.image_ablation or "normal").strip().lower()
            images = apply_image_ablation(
                load_sample_images(sample, self.project_root),
                image_ablation,
                sample_id=str(sample.get("sample_id")),
            )
            camera_indices = resolve_camera_indices(sample, self.project_root, image_count=len(images))
            num_frames_per_camera = max(len(images) // max(len(camera_indices), 1), 1)
            relative_timestamps = resolve_image_relative_timestamps(
                sample,
                self.project_root,
                camera_count=len(camera_indices),
                frames_per_camera=num_frames_per_camera,
            )
            if self.target_mode == "traj_only":
                hard_layout = _build_traj_only_target_layout(
                    self.tokenizer,
                    traj_token_ids,
                    loss_ignore_prefix_len=loss_ignore_prefix_len,
                )
                assistant_prefix = "<|traj_future_start|>"
            else:
                hard_layout = _build_target_layout(
                    self.tokenizer,
                    target_cot_text,
                    traj_token_ids,
                    loss_ignore_prefix_len=loss_ignore_prefix_len,
                )
                assistant_prefix = "<|cot_start|>"

            prompt_messages.append(
                build_messages(
                    prompt_text,
                    len(images),
                    assistant_prefix=assistant_prefix,
                    image_prompt_style=self.image_prompt_style,
                    camera_indices=camera_indices,
                    num_frames_per_camera=num_frames_per_camera,
                )
            )
            full_messages.append(
                build_messages(
                    prompt_text,
                    len(images),
                    hard_layout.completion_text,
                    assistant_prefix=assistant_prefix,
                    image_prompt_style=self.image_prompt_style,
                    camera_indices=camera_indices,
                    num_frames_per_camera=num_frames_per_camera,
                )
            )
            image_batch.append(images)
            ego_histories.append(ego_history_xyz)
            ego_futures.append(ego_future_xyz)
            hard_layouts.append(hard_layout)
            camera_metadata_items.append((camera_indices, relative_timestamps, num_frames_per_camera))

            sample_ids.append(str(sample.get("sample_id")))
            splits.append(str(sample.get("split", "train")))
            image_paths.append([str(path) for path in (sample.get("input") or {}).get("image_paths") or []])
            teacher_vs_gt_consistency_level.append(str(gate.get("teacher_vs_gt_motion") or "missing"))
            sample_provenance_flags.append(
                {
                    "hard_text": (sample.get("provenance") or {}).get("hard_text"),
                    "soft_text": (sample.get("provenance") or {}).get("soft_text"),
                    "traj_target": (sample.get("provenance") or {}).get("traj_target"),
                    "active_target": target_provenance,
                    "teacher_gt_joint_pair_forbidden": bool(
                        (sample.get("provenance") or {}).get("teacher_gt_joint_pair_forbidden", True)
                    ),
                }
            )
            action_labels.append(action_class_to_id(_action_aux_label(sample)))
            hard_cot_weights.append(resolve_loss_weight_value(weights, "hard_cot_ce", 1.0))
            traj_weights.append(resolve_loss_weight_value(weights, "traj_ce", 1.0))
            teacher_pair_traj_ids_for_view = [
                int(token_id) for token_id in (teacher_traj_signal or {}).get("token_ids", [])
            ]
            teacher_allowed = (
                self.enable_teacher_view
                and self.target_mode != "traj_only"
                and bool(teacher_target.get("cot_text"))
                and (self.teacher_view_force_enable or bool(gate.get("teacher_view_allowed")))
                and (not self.teacher_view_uses_teacher_traj or bool(teacher_pair_traj_ids_for_view))
            )
            teacher_weight = float(gate.get("teacher_view_weight") or teacher_target.get("teacher_view_weight") or 0.0)
            teacher_view_allowed.append(teacher_allowed)
            teacher_view_weight.append(teacher_weight if teacher_allowed else 0.0)
            aux_allowed = self.enable_action_aux and bool(gate.get("action_aux_allowed"))
            aux_weight = float(gate.get("action_aux_weight") or 0.0)
            action_aux_allowed.append(aux_allowed)
            action_aux_weight.append(aux_weight if aux_allowed else 0.0)
            teacher_traj_signal_items.append(teacher_traj_signal)
            hard_teacher_signal_items.append(
                teacher_signal if (self.teacher_pair_target or self.hard_view_uses_teacher_cot) else None
            )
            teacher_boundary_hidden_signal_items.append(
                teacher_boundary_hidden_signal if (self.teacher_pair_target or self.hard_view_uses_teacher_cot) else None
            )
            teacher_action_traj_items.append(
                teacher_action_traj if (self.teacher_pair_target or self.hard_view_uses_teacher_cot) else None
            )
            hard_teacher_signal_weights.append(
                1.0 if (self.teacher_pair_target or self.hard_view_uses_teacher_cot) and teacher_signal is not None else 0.0
            )
            hard_teacher_quality_multiplier.append(float(teacher_target.get("teacher_quality_multiplier", 1.0)))

            if not teacher_allowed:
                continue

            teacher_layout_traj_ids = traj_token_ids
            if not self.teacher_pair_target:
                teacher_layout_traj_ids = [int(token_id) for token_id in hard_target.get("traj_future_token_ids") or []]
            if self.teacher_view_uses_teacher_traj:
                teacher_layout_traj_ids = teacher_pair_traj_ids_for_view
            teacher_layout = _build_target_layout(
                self.tokenizer,
                teacher_target.get("cot_text"),
                teacher_layout_traj_ids,
            )
            teacher_prompt_messages.append(
                build_messages(
                    prompt_text,
                    len(images),
                    image_prompt_style=self.image_prompt_style,
                    camera_indices=camera_indices,
                    num_frames_per_camera=num_frames_per_camera,
                )
            )
            teacher_full_messages.append(
                build_messages(
                    prompt_text,
                    len(images),
                    teacher_layout.completion_text,
                    image_prompt_style=self.image_prompt_style,
                    camera_indices=camera_indices,
                    num_frames_per_camera=num_frames_per_camera,
                )
            )
            teacher_image_batch.append(images)
            teacher_layouts.append(teacher_layout)
            teacher_camera_metadata_items.append((camera_indices, relative_timestamps, num_frames_per_camera))
            teacher_signal_items.append(teacher_signal)
            teacher_view_traj_signal_items.append(teacher_traj_signal)
            teacher_sample_ids.append(str(sample.get("sample_id")))
            teacher_source_indices.append(sample_index)
            teacher_seq_ce_weights.append(resolve_loss_weight_value(weights, "teacher_seq_ce", teacher_weight))
            teacher_logit_kd_weights.append(
                resolve_loss_weight_value(weights, "teacher_logit_kd", teacher_weight)
                if teacher_signal and "topk_indices" in teacher_signal
                else 0.0
            )
            teacher_traj_weights.append(
                explicit_teacher_traj_weight
                if explicit_teacher_traj_weight is not None
                else (float(self.teacher_view_default_traj_weight) if self.teacher_view_uses_teacher_traj else 0.0)
            )
            teacher_quality_multiplier.append(float(teacher_target.get("teacher_quality_multiplier", 1.0)))

        prompt_batch = _encode_messages(
            self.processor,
            prompt_messages,
            image_batch,
            self.max_length,
            continue_final_message=True,
        )
        full_batch = _encode_messages(
            self.processor,
            full_messages,
            image_batch,
            self.max_length,
            continue_final_message=True,
        )
        if self.fuse_history_tokens:
            prompt_batch["input_ids"] = fuse_history_tokens_in_input_ids(
                prompt_batch["input_ids"],
                self.tokenizer,
                ego_histories,
            )
            full_batch["input_ids"] = fuse_history_tokens_in_input_ids(
                full_batch["input_ids"],
                self.tokenizer,
                ego_histories,
            )
        labels = _labels_from_prompt_and_full(prompt_batch, full_batch)
        hard_masks = _target_layout_masks(labels, hard_layouts)
        _apply_layout_loss_ignore_prefix(labels, hard_masks, hard_layouts)
        ego_history_xyz, ego_history_mask = _pad_ego_history_batch(ego_histories)
        ego_future_xyz, ego_future_mask = _pad_future_xyz_batch(ego_futures)

        batch: dict[str, Any] = {
            "input_ids": full_batch["input_ids"],
            "attention_mask": full_batch["attention_mask"],
            "pixel_values": full_batch.get("pixel_values"),
            "image_grid_thw": full_batch.get("image_grid_thw"),
            "labels": labels,
            "cot_span_mask": hard_masks["cot_span_mask"],
            "cot_content_mask": hard_masks["cot_content_mask"],
            "traj_span_mask": hard_masks["traj_span_mask"],
            "traj_token_mask": hard_masks["traj_token_mask"],
            "format_token_mask": hard_masks["format_token_mask"],
            "ego_history_xyz": ego_history_xyz,
            "ego_history_mask": ego_history_mask,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_mask": ego_future_mask,
            "image_paths": image_paths,
            "sample_ids": sample_ids,
            "splits": splits,
            "sample_provenance_flags": sample_provenance_flags,
            "image_prompt_style": self.image_prompt_style,
            "teacher_vs_gt_consistency_level": teacher_vs_gt_consistency_level,
            "teacher_view_allowed": torch.tensor(teacher_view_allowed, dtype=torch.bool),
            "teacher_view_weight": torch.tensor(teacher_view_weight, dtype=torch.float32),
            "action_aux_allowed": torch.tensor(action_aux_allowed, dtype=torch.bool),
            "action_aux_weight": torch.tensor(action_aux_weight, dtype=torch.float32),
            "teacher_quality_multiplier": torch.ones((len(features),), dtype=torch.float32),
            "teacher_pair_weight": torch.tensor(hard_teacher_signal_weights, dtype=torch.float32),
            "teacher_pair_quality_multiplier": torch.tensor(hard_teacher_quality_multiplier, dtype=torch.float32),
            "teacher_traj_topk_on_teacher_view": self.teacher_traj_topk_on_teacher_view,
            "prompt_text_style": self.prompt_text_style,
            "fuse_history_tokens": bool(self.fuse_history_tokens),
            "action_class_labels": torch.tensor(action_labels, dtype=torch.long),
            "hard_cot_weights": torch.tensor(hard_cot_weights, dtype=torch.float32),
            "traj_weights": torch.tensor(traj_weights, dtype=torch.float32),
        }
        batch.update(_pad_camera_metadata(camera_metadata_items))
        if any(item is not None for item in hard_teacher_signal_items):
            batch.update(_pad_teacher_signal_batch(hard_teacher_signal_items, hard_masks["text_topk_positions"]))
        if any(item is not None for item in teacher_boundary_hidden_signal_items):
            batch.update(
                _pad_teacher_boundary_hidden_batch(
                    teacher_boundary_hidden_signal_items,
                    hard_masks["boundary_hidden_positions"],
                )
            )
        if any(item is not None for item in teacher_action_traj_items):
            batch.update(_pad_teacher_action_traj_batch(teacher_action_traj_items))
        teacher_traj_available = torch.tensor(
            [item is not None for item in teacher_traj_signal_items],
            dtype=torch.bool,
        )
        teacher_traj_quality = torch.tensor(
            [float((item or {}).get("quality_multiplier", 1.0)) for item in teacher_traj_signal_items],
            dtype=torch.float32,
        )
        batch["teacher_traj_available"] = teacher_traj_available
        batch["teacher_traj_quality_multiplier"] = teacher_traj_quality
        has_teacher_traj_token_weights = any(item is not None for item in teacher_traj_token_weight_items)
        if any(item is not None and "token_ids" in item for item in teacher_traj_signal_items):
            teacher_traj_labels = torch.full_like(labels, IGNORE_INDEX)
            teacher_traj_label_token_weights = (
                torch.ones_like(labels, dtype=torch.float32) if has_teacher_traj_token_weights else None
            )
            for row_index, item in enumerate(teacher_traj_signal_items):
                if item is None or "token_ids" not in item:
                    continue
                raw_token_ids = np.asarray(item["token_ids"], dtype=np.int32).reshape(-1)
                active_positions = torch.nonzero(
                    hard_masks["traj_token_mask"][row_index] & (labels[row_index] != IGNORE_INDEX),
                    as_tuple=False,
                ).flatten()
                token_count = min(len(raw_token_ids), int(active_positions.numel()))
                if token_count <= 0:
                    continue
                tokenizer_token_ids = [
                    int(getattr(self.tokenizer, "traj_token_start_idx", -1)) + int(token_id)
                    if isinstance(getattr(self.tokenizer, "traj_token_start_idx", None), int)
                    and int(getattr(self.tokenizer, "traj_token_start_idx")) >= 0
                    else int(self.tokenizer.convert_tokens_to_ids(discrete_traj_token(int(token_id))))
                    for token_id in raw_token_ids[:token_count]
                ]
                teacher_traj_labels[row_index, active_positions[:token_count]] = torch.tensor(
                    tokenizer_token_ids,
                    dtype=teacher_traj_labels.dtype,
                )
                if teacher_traj_label_token_weights is not None:
                    raw_weights = teacher_traj_token_weight_items[row_index]
                    if raw_weights is not None:
                        weight_count = min(int(raw_weights.shape[0]), token_count)
                        teacher_traj_label_token_weights[row_index, active_positions[:weight_count]] = torch.from_numpy(
                            raw_weights[:weight_count]
                        ).float()
            batch["teacher_traj_labels"] = teacher_traj_labels
            if teacher_traj_label_token_weights is not None:
                batch["teacher_traj_label_token_weights"] = teacher_traj_label_token_weights
        topk_ready = [item for item in teacher_traj_signal_items if item is not None and "topk_indices" in item]
        if topk_ready:
            topk = int(np.asarray(topk_ready[0]["topk_indices"]).shape[-1])
            max_tokens = max(
                min(
                    int(np.asarray(item["topk_indices"]).shape[0]),
                    int(torch.count_nonzero(hard_masks["traj_token_mask"][row_index]).item()),
                )
                for row_index, item in enumerate(teacher_traj_signal_items)
                if item is not None and "topk_indices" in item
            )
            teacher_traj_topk_indices = torch.zeros((len(features), max_tokens, topk), dtype=torch.long)
            teacher_traj_topk_logprobs = torch.zeros((len(features), max_tokens, topk), dtype=torch.float32)
            teacher_traj_topk_mask = torch.zeros((len(features), max_tokens), dtype=torch.bool)
            teacher_traj_token_weights = (
                torch.ones((len(features), max_tokens), dtype=torch.float32)
                if has_teacher_traj_token_weights
                else None
            )
            traj_token_start_idx = getattr(self.tokenizer, "traj_token_start_idx", None)
            if not isinstance(traj_token_start_idx, int) or traj_token_start_idx < 0:
                raise ValueError("Tokenizer is missing traj_token_start_idx required for teacher traj cache.")
            for row_index, item in enumerate(teacher_traj_signal_items):
                if item is None or "topk_indices" not in item or "topk_logprobs" not in item:
                    continue
                raw_indices = np.asarray(item["topk_indices"], dtype=np.int32)
                raw_logprobs = np.asarray(item["topk_logprobs"], dtype=np.float32)
                token_count = min(
                    int(raw_indices.shape[0]),
                    int(torch.count_nonzero(hard_masks["traj_token_mask"][row_index]).item()),
                    max_tokens,
                )
                if token_count <= 0:
                    continue
                teacher_traj_topk_indices[row_index, :token_count] = torch.from_numpy(
                    raw_indices[:token_count] + int(traj_token_start_idx)
                ).long()
                teacher_traj_topk_logprobs[row_index, :token_count] = torch.from_numpy(raw_logprobs[:token_count]).float()
                teacher_traj_topk_mask[row_index, :token_count] = True
                if teacher_traj_token_weights is not None:
                    raw_weights = teacher_traj_token_weight_items[row_index]
                    if raw_weights is not None:
                        weight_count = min(int(raw_weights.shape[0]), token_count)
                        teacher_traj_token_weights[row_index, :weight_count] = torch.from_numpy(
                            raw_weights[:weight_count]
                        ).float()
            batch["teacher_traj_topk_indices"] = teacher_traj_topk_indices
            batch["teacher_traj_topk_logprobs"] = teacher_traj_topk_logprobs
            batch["teacher_traj_topk_mask"] = teacher_traj_topk_mask
            if teacher_traj_token_weights is not None:
                batch["teacher_traj_token_weights"] = teacher_traj_token_weights
        hidden_ready = [item for item in teacher_traj_signal_items if item is not None and "hidden" in item]
        if hidden_ready:
            hidden_dim = int(np.asarray(hidden_ready[0]["hidden"]).shape[-1])
            max_tokens = max(
                min(
                    int(np.asarray(item["hidden"]).shape[0]),
                    int(torch.count_nonzero(hard_masks["traj_token_mask"][row_index]).item()),
                )
                for row_index, item in enumerate(teacher_traj_signal_items)
                if item is not None and "hidden" in item
            )
            teacher_traj_hidden = torch.zeros((len(features), max_tokens, hidden_dim), dtype=torch.float32)
            teacher_traj_hidden_mask = torch.zeros((len(features), max_tokens), dtype=torch.bool)
            for row_index, item in enumerate(teacher_traj_signal_items):
                if item is None or "hidden" not in item:
                    continue
                hidden = np.asarray(item["hidden"], dtype=np.float32)
                token_count = min(
                    int(hidden.shape[0]),
                    int(torch.count_nonzero(hard_masks["traj_token_mask"][row_index]).item()),
                    max_tokens,
                )
                if token_count <= 0:
                    continue
                teacher_traj_hidden[row_index, :token_count] = torch.from_numpy(hidden[:token_count]).float()
                teacher_traj_hidden_mask[row_index, :token_count] = True
            batch["teacher_traj_hidden"] = teacher_traj_hidden
            batch["teacher_traj_hidden_mask"] = teacher_traj_hidden_mask
        raw_hidden_ready = [item for item in teacher_traj_signal_items if item is not None and "hidden_raw" in item]
        if raw_hidden_ready:
            raw_hidden_dim = int(np.asarray(raw_hidden_ready[0]["hidden_raw"]).shape[-1])
            max_tokens = max(
                min(
                    int(np.asarray(item["hidden_raw"]).shape[0]),
                    int(torch.count_nonzero(hard_masks["traj_token_mask"][row_index]).item()),
                )
                for row_index, item in enumerate(teacher_traj_signal_items)
                if item is not None and "hidden_raw" in item
            )
            teacher_traj_hidden_raw = torch.zeros((len(features), max_tokens, raw_hidden_dim), dtype=torch.float32)
            teacher_traj_hidden_raw_mask = torch.zeros((len(features), max_tokens), dtype=torch.bool)
            for row_index, item in enumerate(teacher_traj_signal_items):
                if item is None or "hidden_raw" not in item:
                    continue
                hidden = np.asarray(item["hidden_raw"], dtype=np.float32)
                token_count = min(
                    int(hidden.shape[0]),
                    int(torch.count_nonzero(hard_masks["traj_token_mask"][row_index]).item()),
                    max_tokens,
                )
                if token_count <= 0:
                    continue
                teacher_traj_hidden_raw[row_index, :token_count] = torch.from_numpy(hidden[:token_count]).float()
                teacher_traj_hidden_raw_mask[row_index, :token_count] = True
            batch["teacher_traj_hidden_raw"] = teacher_traj_hidden_raw
            batch["teacher_traj_hidden_raw_mask"] = teacher_traj_hidden_raw_mask
        traj_token_label_weights = _build_label_token_weights(
            labels,
            hard_masks["traj_token_mask"],
            self.traj_token_weight_map,
        )
        if traj_token_label_weights is not None:
            batch["traj_token_label_weights"] = traj_token_label_weights

        if teacher_full_messages:
            teacher_prompt_batch = _encode_messages(
                self.processor,
                teacher_prompt_messages,
                teacher_image_batch,
                self.max_length,
                continue_final_message=True,
            )
            teacher_full_batch = _encode_messages(
                self.processor,
                teacher_full_messages,
                teacher_image_batch,
                self.max_length,
                continue_final_message=True,
            )
            teacher_labels = _labels_from_prompt_and_full(teacher_prompt_batch, teacher_full_batch)
            teacher_masks = _target_layout_masks(teacher_labels, teacher_layouts)
            teacher_view_batch: dict[str, Any] = {
                "input_ids": teacher_full_batch["input_ids"],
                "attention_mask": teacher_full_batch["attention_mask"],
                "pixel_values": teacher_full_batch.get("pixel_values"),
                "image_grid_thw": teacher_full_batch.get("image_grid_thw"),
                "labels": teacher_labels,
                "cot_span_mask": teacher_masks["cot_span_mask"],
                "cot_content_mask": teacher_masks["cot_content_mask"],
                "traj_span_mask": teacher_masks["traj_span_mask"],
                "traj_token_mask": teacher_masks["traj_token_mask"],
                "format_token_mask": teacher_masks["format_token_mask"],
                "sample_ids": teacher_sample_ids,
                "source_indices": torch.tensor(teacher_source_indices, dtype=torch.long),
                "teacher_view_weight": torch.tensor(teacher_seq_ce_weights, dtype=torch.float32),
                "teacher_logit_kd_weight": torch.tensor(teacher_logit_kd_weights, dtype=torch.float32),
                "traj_weights": torch.tensor(teacher_traj_weights, dtype=torch.float32),
                "teacher_quality_multiplier": torch.tensor(teacher_quality_multiplier, dtype=torch.float32),
            }
            teacher_view_batch.update(_pad_camera_metadata(teacher_camera_metadata_items))
            teacher_view_batch.update(_pad_teacher_signal_batch(teacher_signal_items, teacher_masks["text_topk_positions"]))
            if self.teacher_traj_topk_on_teacher_view:
                teacher_view_batch.update(
                    _pad_teacher_traj_topk_batch(
                        teacher_view_traj_signal_items,
                        teacher_masks["traj_token_mask"],
                        self.tokenizer,
                    )
                )
            teacher_traj_token_label_weights = _build_label_token_weights(
                teacher_labels,
                teacher_masks["traj_token_mask"],
                self.traj_token_weight_map,
            )
            if teacher_traj_token_label_weights is not None:
                teacher_view_batch["traj_token_label_weights"] = teacher_traj_token_label_weights
            batch["teacher_view"] = teacher_view_batch
        else:
            batch["teacher_view"] = None

        return batch
