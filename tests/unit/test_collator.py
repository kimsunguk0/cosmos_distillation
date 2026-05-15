import pytest
import torch

from src.training.collator import (
    IGNORE_INDEX,
    _build_traj_only_target_layout,
    build_messages,
    build_user_prompt,
    fuse_history_tokens_in_input_ids,
    _labels_from_prompt_and_full,
    resolve_camera_indices,
)


def _batch(input_ids: list[list[int]], attention_mask: list[list[int]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def test_labels_from_prompt_and_full_keeps_completion_tokens_trainable() -> None:
    prompt_batch = _batch([[10, 11, 12]], [[1, 1, 1]])
    full_batch = _batch([[10, 11, 12, 13, 14]], [[1, 1, 1, 1, 1]])

    labels = _labels_from_prompt_and_full(prompt_batch, full_batch)

    assert labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 13, 14]]


def test_labels_from_prompt_and_full_rejects_prefix_mismatch() -> None:
    prompt_batch = _batch([[10, 11, 99]], [[1, 1, 1]])
    full_batch = _batch([[10, 11, 12, 13]], [[1, 1, 1, 1]])

    with pytest.raises(ValueError, match="Prompt/full chat template prefixes diverged"):
        _labels_from_prompt_and_full(prompt_batch, full_batch)


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        mapping = {
            "<|traj_future_end|>": [9],
        }
        if text in mapping:
            return mapping[text]
        return [1] * len(text)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|traj_history|>":
            return 777
        if token.startswith("<i") and token.endswith(">"):
            return 1000 + int(token[2:-1])
        return -1


def test_build_messages_supports_traj_only_assistant_prefix() -> None:
    messages = build_messages("prompt", 1, assistant_prefix="<|traj_future_start|>")
    assert messages[-1]["content"][0]["text"] == "<|traj_future_start|>"


def test_build_traj_only_target_layout_tracks_traj_content_without_prefix() -> None:
    layout = _build_traj_only_target_layout(_StubTokenizer(), [1, 2, 3])
    assert layout.cot_span_len == 0
    assert layout.traj_prefix_len == 0
    assert layout.traj_content_len == 3
    assert layout.traj_suffix_len == 1


def test_camera_labeled_messages_use_alpamayo_4v_slot_contract() -> None:
    messages = build_messages(
        "prompt",
        16,
        assistant_prefix="<|cot_start|>",
        image_prompt_style="camera_labeled",
        camera_indices=[0, 1, 2, 6],
        num_frames_per_camera=4,
    )
    user_content = messages[1]["content"]

    image_count = sum(1 for item in user_content if item["type"] == "image")
    text_items = [item["text"] for item in user_content if item["type"] == "text"]
    camera_labels = [text for text in text_items if text.endswith("camera: ")]
    frame_labels = [text for text in text_items if text.startswith("frame ")]

    assert image_count == 16
    assert camera_labels == [
        "Front left camera: ",
        "Front camera: ",
        "Front right camera: ",
        "Front telephoto camera: ",
    ]
    assert frame_labels == ["frame 0 ", "frame 1 ", "frame 2 ", "frame 3 "] * 4
    assert user_content[-1] == {"type": "text", "text": "prompt"}
    assert messages[-1]["content"][0]["text"] == "<|cot_start|>"


def test_resolve_camera_indices_maps_materialized_4v_slot3_to_front_telephoto(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"camera_indices": [0, 1, 2, 6]}', encoding="utf-8")
    sample = {
        "input": {
            "metadata_path": str(metadata_path),
            "camera_count": 4,
            "image_count": 16,
        }
    }

    assert resolve_camera_indices(sample, tmp_path, image_count=16) == [0, 1, 2, 6]
    assert resolve_camera_indices({"input": {"camera_count": 4}}, tmp_path, image_count=16) == [0, 1, 2, 6]


def test_official_alpamayo_prompt_and_history_token_fusion() -> None:
    prompt = build_user_prompt({}, __import__("pathlib").Path("."), prompt_text_style="official_alpamayo")
    assert prompt.count("<|traj_history|>") == 48
    input_ids = torch.tensor([[10] + [777] * 48 + [11]], dtype=torch.long)
    history = torch.zeros((16, 3), dtype=torch.float32).numpy()

    fused = fuse_history_tokens_in_input_ids(input_ids, _StubTokenizer(), [history])

    assert not torch.any(fused == 777)
    assert fused.shape == input_ids.shape
    assert int(fused[0, 1].item()) == 1000 + 3000 + 500
