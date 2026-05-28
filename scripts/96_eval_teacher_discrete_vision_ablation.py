#!/usr/bin/env python3
"""Evaluate Alpamayo teacher discrete-token sensitivity to image perturbations."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUKIM_ROOT = PROJECT_ROOT.parents[1]
PROBE_PATH = PROJECT_ROOT.parent / "dataset_prep" / "scripts" / "probe_alpamayo15_discrete_traj.py"

spec = importlib.util.spec_from_file_location("alpamayo15_probe", PROBE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load probe helpers from {PROBE_PATH}")
probe = importlib.util.module_from_spec(spec)
sys.modules["alpamayo15_probe"] = probe
spec.loader.exec_module(probe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=PROJECT_ROOT / "data/corpus/no_nav_teacher_pair_300chunks_semantic_balanced_50k.jsonl",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("/home/pm97/workspace/dataset/distill_dataset/materialized"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--modes",
        default="normal,black,gray,noise,camera_shuffle",
        help="Comma-separated image perturbations.",
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def _load_rows(path: Path, split: str, num_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("split") == split:
                rows.append(rec)
                if num_samples > 0 and len(rows) >= num_samples:
                    break
    return rows


def _max_same_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    best = cur = 1
    for left, right in zip(tokens, tokens[1:]):
        if left == right:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _apply_image_mode(sample: dict[str, Any], mode: str, *, sample_id: str) -> dict[str, Any]:
    if mode == "normal":
        return sample
    out = dict(sample)
    frames = sample["image_frames"].clone()
    if mode == "black":
        frames.zero_()
    elif mode == "gray":
        frames.fill_(127)
    elif mode == "noise":
        seed = abs(hash(sample_id)) % (2**32)
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, size=tuple(frames.shape), dtype=np.uint8)
        frames = torch.from_numpy(arr)
    elif mode == "camera_shuffle":
        frames = torch.flip(frames, dims=(0,))
    else:
        raise ValueError(f"Unsupported image mode: {mode}")
    out["image_frames"] = frames
    return out


def _run_teacher_discrete(
    model: Any,
    model_config: Any,
    processor: Any,
    sample: dict[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = str(next(model.parameters()).device)
    model_inputs, prompt_len = probe.build_model_inputs(model, model_config, sample, device, processor=processor)
    tokenized_data = dict(model_inputs["tokenized_data"])
    input_ids = tokenized_data.pop("input_ids")
    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.do_sample = False
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = max_new_tokens
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = model.tokenizer.pad_token_id

    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=next(model.parameters()).dtype,
        enabled=device.startswith("cuda"),
    ):
        generated = model.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            **tokenized_data,
        )
    generated_tokens = generated.sequences[:, prompt_len:]
    decoded = model.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)[0]
    traj_token_ids = probe.extract_traj_tokens(
        generated_tokens,
        model.special_token_ids,
        model.config.tokens_per_future_traj,
        model.future_token_start_idx,
        model.traj_tokenizer.vocab_size,
    )[0]

    hist_xyz = sample["ego_history_xyz"][:, -1].to(next(model.parameters()).device)
    hist_rot = sample["ego_history_rot"][:, -1].to(next(model.parameters()).device)
    with torch.inference_mode():
        pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(
            hist_xyz=hist_xyz,
            hist_rot=hist_rot,
            tokens=traj_token_ids.unsqueeze(0),
        )
    gt_xyz = sample["ego_future_xyz"][0, 0].to(pred_xyz.device)
    ade, fde = probe.compute_ade_fde(pred_xyz[0], gt_xyz)
    tokens = traj_token_ids.detach().cpu().numpy().astype(np.int32).tolist()
    return {
        "decoded_text": decoded,
        "tokens": tokens,
        "pred_xyz": pred_xyz[0].detach().cpu().numpy().astype(np.float32),
        "ade_gt": float(ade),
        "fde_gt": float(fde),
        "unique": len(set(tokens)),
        "max_same_run": _max_same_run(tokens),
        "invalid_count": sum(1 for token in tokens if token < 0 or token >= 3000),
    }


def _ade_fde_np(left_xyz: np.ndarray, right_xyz: np.ndarray) -> tuple[float, float]:
    diffs = np.linalg.norm(left_xyz[:, :2] - right_xyz[:, :2], axis=-1)
    return float(diffs.mean()), float(diffs[-1])


def _mode_summary(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    subset = [rec for rec in records if rec["mode"] == mode]
    toks: list[int] = []
    for rec in subset:
        toks.extend(rec["tokens"])
    counter = Counter(toks)
    total = sum(counter.values()) or 1
    entropy = -sum((count / total) * math.log(count / total) for count in counter.values())
    return {
        "mode": mode,
        "num_samples": len(subset),
        "ade_gt_mean": float(np.mean([rec["ade_gt"] for rec in subset])) if subset else None,
        "fde_gt_mean": float(np.mean([rec["fde_gt"] for rec in subset])) if subset else None,
        "ade_vs_normal_mean": float(np.mean([rec["ade_vs_normal"] for rec in subset])) if subset else None,
        "fde_vs_normal_mean": float(np.mean([rec["fde_vs_normal"] for rec in subset])) if subset else None,
        "token_same_vs_normal": float(np.mean([rec["token_same_vs_normal"] for rec in subset])) if subset else None,
        "unique_mean": float(np.mean([rec["unique"] for rec in subset])) if subset else None,
        "max_same_run_mean": float(np.mean([rec["max_same_run"] for rec in subset])) if subset else None,
        "invalid_rate": float(sum(rec["invalid_count"] for rec in subset) / max(len(subset) * 128, 1)),
        "global_unique": len(counter),
        "entropy": float(entropy),
        "top_token_histogram": [
            {"token": int(token), "count": int(count), "mass": float(count / total)}
            for token, count in counter.most_common(20)
        ],
    }


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.corpus_jsonl, args.split, args.num_samples)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    if "normal" not in modes:
        modes.insert(0, "normal")

    dtype = probe.torch_dtype_from_name(args.dtype)
    model, model_config = probe.load_model(args.checkpoint_path, dtype=dtype, device=args.device)
    processor_kwargs: dict[str, Any] = {}
    if model_config.min_pixels is not None:
        processor_kwargs["min_pixels"] = model_config.min_pixels
    if model_config.max_pixels is not None:
        processor_kwargs["max_pixels"] = model_config.max_pixels
    processor = AutoProcessor.from_pretrained(model_config.vlm_name_or_path, **processor_kwargs)
    processor.tokenizer = model.tokenizer
    processor.tokenizer.padding_side = "left"
    model.tokenizer.padding_side = "left"

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = str(row["sample_id"])
        raw = probe.load_materialized_sample(args.samples_root / sample_id)
        normal_out: dict[str, Any] | None = None
        for mode in modes:
            sample = _apply_image_mode(raw, mode, sample_id=sample_id)
            out = _run_teacher_discrete(model, model_config, processor, sample, max_new_tokens=args.max_new_tokens)
            if mode == "normal":
                normal_out = out
            assert normal_out is not None
            ade_n, fde_n = _ade_fde_np(out["pred_xyz"], normal_out["pred_xyz"])
            same = sum(int(a == b) for a, b in zip(out["tokens"], normal_out["tokens"])) / max(len(normal_out["tokens"]), 1)
            records.append(
                {
                    "sample_id": sample_id,
                    "mode": mode,
                    "ade_gt": out["ade_gt"],
                    "fde_gt": out["fde_gt"],
                    "ade_vs_normal": ade_n,
                    "fde_vs_normal": fde_n,
                    "token_same_vs_normal": same,
                    "unique": out["unique"],
                    "max_same_run": out["max_same_run"],
                    "invalid_count": out["invalid_count"],
                    "tokens": out["tokens"],
                }
            )
        print(json.dumps({"event": "sample_done", "index": index, "num_samples": len(rows), "sample_id": sample_id}), flush=True)

    summary = {
        "checkpoint_path": str(args.checkpoint_path),
        "corpus_jsonl": str(args.corpus_jsonl),
        "split": args.split,
        "num_samples": len(rows),
        "modes": modes,
        "mode_summaries": [_mode_summary(records, mode) for mode in modes],
        "records": records,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "summary_json": str(args.summary_json)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
