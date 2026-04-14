#!/usr/bin/env python3
"""Probe Alpamayo 1.5 text routes against a base-style chained meta-action route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_ALPAMAYO_SRC = Path("/home/pm97/workspace/sukim/alpamayo1.5/src")
DEFAULT_ALPAMAYO_MODEL = Path("/home/pm97/workspace/sukim/weights/alpamayo15_vlm_weights")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-request", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_ALPAMAYO_MODEL)
    parser.add_argument("--alpamayo-src", type=Path, default=DEFAULT_ALPAMAYO_SRC)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-generation-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "alpamayo_chain_meta_probe.json",
    )
    return parser.parse_args()


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


def load_runtime_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(bundle: dict[str, Any]) -> torch.Tensor:
    frame_dir = Path(bundle["inputs"]["frame_dir"])
    sample_meta = bundle["sample_meta"]
    frame_offsets = [float(value) for value in sample_meta["frame_offsets_sec"]]
    frames: list[torch.Tensor] = []
    for camera_name in bundle["camera_names"]:
        for offset in frame_offsets:
            image_path = frame_dir / f"{camera_name}_t{offset:+.1f}.jpg"
            image = Image.open(image_path).convert("RGB")
            array = np.asarray(image, dtype=np.uint8).copy()
            frames.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(frames, dim=0)


def summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for message in messages:
        content_summary: list[dict[str, Any]] = []
        for item in message["content"]:
            if item.get("type") == "image":
                content_summary.append({"type": "image"})
            else:
                content_summary.append({"type": item.get("type"), "text": item.get("text")})
        summary.append({"role": message["role"], "content": content_summary})
    return summary


def history_placeholder(num_tokens: int = 48) -> str:
    return f"<|traj_history_start|>{'<|traj_history|>' * num_tokens}<|traj_history_end|>"


def build_cot_only_messages(helper_module, frames: torch.Tensor, camera_indices: list[int]) -> list[dict[str, Any]]:
    image_content = helper_module._build_image_content(
        frames=frames,
        camera_indices=torch.tensor(camera_indices, dtype=torch.long),
        num_frames_per_camera=4,
    )
    user_text = (
        f"{history_placeholder()}"
        "output the chain-of-thought reasoning of the driving process."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]


def build_chained_meta_messages(
    helper_module,
    frames: torch.Tensor,
    camera_indices: list[int],
    cot_text: str,
) -> list[dict[str, Any]]:
    image_content = helper_module._build_image_content(
        frames=frames,
        camera_indices=torch.tensor(camera_indices, dtype=torch.long),
        num_frames_per_camera=4,
    )
    user_text = (
        f"{history_placeholder()}"
        "output the chain-of-thought reasoning of the driving process, then output meta actions."
    )
    cot_text = cot_text.strip()
    assistant_prefix = f"<|cot_start|>{cot_text}<|cot_end|><|meta_action_start|>"
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_prefix}]},
    ]


def tokenize_messages(processor, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )


def run_generate(model, tokenizer, tokenized_data: dict[str, torch.Tensor], *, top_p: float, temperature: float, max_generation_length: int) -> dict[str, Any]:
    inputs = {key: value.to(model.device) for key, value in tokenized_data.items()}
    input_ids = inputs["input_ids"]
    tokenized_extra = {key: value for key, value in inputs.items() if key != "input_ids"}

    generation_config = model.vlm.generation_config
    generation_config.top_p = top_p
    generation_config.temperature = temperature
    generation_config.do_sample = True
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_generation_length
    generation_config.output_logits = True
    generation_config.return_dict_in_generate = True
    generation_config.top_k = None
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.vlm.generate(
            input_ids=input_ids,
            **tokenized_extra,
            generation_config=generation_config,
        )
    generated_tokens = generated.sequences[:, input_ids.shape[1] :]
    from alpamayo1_5.models.token_utils import extract_text_tokens

    extracted = extract_text_tokens(tokenizer, generated_tokens)
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    return {
        "decoded": decoded,
        "extracted": extracted,
    }


def main() -> None:
    args = parse_args()
    bundle = load_runtime_request(args.runtime_request)
    helper, model, processor = load_teacher_model(args)
    frames = load_frames(bundle)
    camera_indices = [int(value) for value in bundle["camera_indices"]]

    official_messages = helper.create_message(
        frames,
        camera_indices=torch.tensor(camera_indices, dtype=torch.long),
        num_frames_per_camera=4,
    )
    official_tokenized = tokenize_messages(processor, official_messages)
    official = run_generate(
        model,
        model.tokenizer,
        official_tokenized,
        top_p=args.top_p,
        temperature=args.temperature,
        max_generation_length=args.max_generation_length,
    )
    official_cot = str(official["extracted"]["cot"][0]).strip()

    chained_messages = build_chained_meta_messages(helper, frames, camera_indices, official_cot)
    chained_tokenized = tokenize_messages(processor, chained_messages)
    chained = run_generate(
        model,
        model.tokenizer,
        chained_tokenized,
        top_p=args.top_p,
        temperature=args.temperature,
        max_generation_length=16,
    )

    result = {
        "sample_id": bundle["sample_id"],
        "runtime_request": str(args.runtime_request),
        "model_path": str(args.model_path),
        "official_cot_route": {
            "messages": summarize_messages(official_messages),
            "decoded": official["decoded"],
            "extracted": official["extracted"],
        },
        "base_style_chained_meta_route": {
            "messages": summarize_messages(chained_messages),
            "decoded": chained["decoded"],
            "extracted": chained["extracted"],
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
