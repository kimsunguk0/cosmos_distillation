#!/usr/bin/env python3
"""Extract downsampled prefix hidden sequences for KV-aware action probes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name: str, rel_path: str):
    path = PROJECT_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feat69 = _load_module("hidden_to_action_features_69", "scripts/69_extract_hidden_to_action_features.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--split-sample-ids-json", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--student-model", default=os.environ.get("COSMOS_STUDENT_MODEL", str(PROJECT_ROOT / "base_weights/cosmos-reason-2b")))
    parser.add_argument("--prefix-type", choices=("teacher_prefix", "student_free"), default="teacher_prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["probe_train", "probe_val", "probe_test"])
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seq-tokens", type=int, default=512)
    parser.add_argument("--empty-cot-token-threshold", type=int, default=3)
    parser.add_argument("--image-mode", choices=("normal", "black", "shuffled"), default="normal")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_prefix_indices(attn_mask: np.ndarray, *, end_pos: int, max_tokens: int) -> np.ndarray:
    valid = np.flatnonzero((attn_mask.astype(bool)) & (np.arange(attn_mask.shape[0]) <= int(end_pos)))
    if valid.size <= int(max_tokens):
        return valid.astype(np.int64)
    picked = np.linspace(0, valid.size - 1, int(max_tokens)).round().astype(np.int64)
    return valid[picked].astype(np.int64)


def flush_shard(*, split_dir: Path, shard_index: int, rows: list[dict[str, Any]], tensors: dict[str, list[np.ndarray]], metadata: dict[str, Any]) -> Path | None:
    if not rows:
        return None
    split_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metadata": metadata, "rows": rows}
    for key, values in tensors.items():
        if values:
            payload[key] = torch.from_numpy(np.stack(values, axis=0))
    path = split_dir / f"features_{shard_index:05d}.pt"
    torch.save(payload, path)
    return path


def load_partial(split_dir: Path) -> tuple[set[str], list[str], int, Counter]:
    processed: set[str] = set()
    shards: list[str] = []
    counters = Counter()
    next_index = 0
    if not split_dir.exists():
        return processed, shards, next_index, counters
    for shard_path in sorted(split_dir.glob("features_*.pt")):
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        rows = list(payload.get("rows") or [])
        if not rows:
            continue
        shards.append(str(shard_path))
        try:
            next_index = max(next_index, int(shard_path.stem.rsplit("_", 1)[-1]) + 1)
        except ValueError:
            next_index += 1
        for row in rows:
            sample_id = str(row.get("sample_id"))
            if sample_id and sample_id not in processed:
                processed.add(sample_id)
                counters["ready"] += 1
    return processed, shards, next_index, counters


def split_limit(args: argparse.Namespace, split_name: str) -> int:
    if split_name == "probe_train" and args.max_train_samples > 0:
        return int(args.max_train_samples)
    if split_name == "probe_val" and args.max_val_samples > 0:
        return int(args.max_val_samples)
    if split_name == "probe_test" and args.max_test_samples > 0:
        return int(args.max_test_samples)
    return int(args.max_samples_per_split)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = feat69.load_jsonl_map(args.corpus_jsonl)
    split_ids = feat69.load_split_sample_ids(args.split_sample_ids_json)

    model_args = argparse.Namespace(checkpoint_dir=args.checkpoint_dir, student_model=args.student_model, device=args.device)
    model, tokenizer, processor, device, base_model, _train_config = feat69.helpers.load_model(model_args)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    model_dtype = next(model.backbone.parameters()).dtype
    cot_end_id = feat69.helpers.token_id(tokenizer, "<|cot_end|>")
    traj_start_id = feat69.helpers.token_id(tokenizer, "<|traj_future_start|>")
    decoder_config = feat69.helpers.resolve_traj_tokenizer_config_path(base_model)
    if decoder_config is None:
        raise RuntimeError(f"Could not resolve trajectory tokenizer config for {base_model}")
    decoder = feat69.helpers.TrajectoryTokenDecoder(config_path=decoder_config)

    summary: dict[str, Any] = {
        "schema_version": "prefix_sequence_feature_extraction_v1",
        "checkpoint_name": args.checkpoint_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "prefix_type": args.prefix_type,
        "image_mode": args.image_mode,
        "max_seq_tokens": int(args.max_seq_tokens),
        "batch_size": int(args.batch_size),
        "shard_size": int(args.shard_size),
        "selection": "uniform_downsample_valid_prefix_tokens_up_to_traj_start",
        "splits": {},
    }

    for split_name in args.splits:
        ids = [sample_id for sample_id in split_ids.get(split_name, []) if sample_id in rows_by_id]
        requested_limit = split_limit(args, split_name)
        if requested_limit > 0:
            ids = ids[:requested_limit]
        split_dir = args.output_dir / args.checkpoint_name / args.prefix_type / split_name
        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary["splits"][split_name] = existing
            print(json.dumps({"event": "seq_split_skip_existing", "split": split_name, "manifest": str(manifest_path)}), flush=True)
            continue

        resume_ids: set[str] = set()
        shard_paths: list[str] = []
        shard_index = 0
        counters = Counter()
        if not args.overwrite:
            resume_ids, shard_paths, shard_index, counters = load_partial(split_dir)
            if resume_ids:
                ids = [sample_id for sample_id in ids if sample_id not in resume_ids]
                print(json.dumps({"event": "seq_split_resume", "split": split_name, "resumed_ready": len(resume_ids), "remaining": len(ids)}), flush=True)

        rows_buffer: list[dict[str, Any]] = []
        tensors: dict[str, list[np.ndarray]] = {
            "prefix_hidden": [],
            "prefix_mask": [],
            "h_cot_end": [],
            "h_traj_start": [],
            "target_action": [],
            "target_traj": [],
            "ego_history_xyz": [],
            "ego_history_rot": [],
            "gt_future": [],
        }

        for batch_ids in feat69.batched(ids, args.batch_size):
            raw_samples = [rows_by_id[sample_id] for sample_id in batch_ids]
            valid_samples: list[dict[str, Any]] = []
            target_xyz: list[np.ndarray] = []
            target_rot: list[np.ndarray] = []
            history_xyz: list[np.ndarray] = []
            history_rot: list[np.ndarray] = []
            gt_future: list[np.ndarray] = []
            for sample in raw_samples:
                pred = feat69.raw_teacher_pred(sample)
                if pred is None:
                    counters["missing_teacher_action"] += 1
                    continue
                try:
                    hxyz = feat69.helpers.load_ego_history_xyz(sample, PROJECT_ROOT).astype(np.float32)
                    hrot = feat69.normalize_history_rot(feat69.helpers.load_ego_history_rot(sample, PROJECT_ROOT))
                except Exception:
                    counters["missing_ego_history"] += 1
                    continue
                gt = feat69.load_gt_future(sample)
                if gt is None:
                    gt = np.zeros((64, 3), dtype=np.float32)
                valid_samples.append(sample)
                target_xyz.append(pred[0])
                target_rot.append(pred[1])
                history_xyz.append(hxyz)
                history_rot.append(hrot)
                gt_future.append(gt)
            if not valid_samples:
                continue

            prompt_len: int | None = None
            if args.prefix_type == "teacher_prefix":
                batch, prepared = feat69.make_teacher_prefix_batch(valid_samples, processor=processor, tokenizer=tokenizer, image_mode=args.image_mode)
                moved = feat69.move_processor_batch(batch, device=device, model_dtype=model_dtype)
            else:
                batch, prepared = feat69.make_student_prompt_batch(valid_samples, processor=processor, tokenizer=tokenizer, image_mode=args.image_mode)
                prompt_moved = feat69.move_processor_batch(batch, device=device, model_dtype=model_dtype)
                prompt_len = int(prompt_moved["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = model.backbone.generate(
                        **prompt_moved,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                        use_cache=True,
                        stopping_criteria=feat69.helpers.StoppingCriteriaList(
                            [feat69.helpers.StopAfterTokenCriteria(prompt_lengths=[prompt_len] * len(prepared), stop_token_id=traj_start_id)]
                        ),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated_len = int(generated.shape[1] - prompt_len)
                generated_attention = torch.cat(
                    [
                        prompt_moved["attention_mask"],
                        torch.ones((generated.shape[0], generated_len), dtype=prompt_moved["attention_mask"].dtype, device=prompt_moved["attention_mask"].device),
                    ],
                    dim=1,
                )
                moved = {key: value for key, value in prompt_moved.items() if key not in {"input_ids", "attention_mask"}}
                moved["input_ids"] = generated
                moved["attention_mask"] = generated_attention

            with torch.inference_mode():
                last_hidden = feat69.backbone_last_hidden(model, moved)
            hidden = last_hidden.detach().to(dtype=torch.float32).cpu().numpy()
            input_ids_np = moved["input_ids"].detach().cpu().numpy()
            attention_np = moved["attention_mask"].detach().cpu().numpy()
            with torch.inference_mode():
                target_action_t = decoder.action_space.traj_to_action(
                    torch.from_numpy(np.stack(history_xyz, axis=0)),
                    torch.from_numpy(np.stack(history_rot, axis=0)),
                    torch.from_numpy(np.stack(target_xyz, axis=0)),
                    torch.from_numpy(np.stack(target_rot, axis=0)),
                )
            target_action_np = target_action_t.detach().cpu().numpy().astype(np.float32)

            for row_index, sample in enumerate(prepared):
                ids_row = [int(value) for value in input_ids_np[row_index].tolist()]
                if prompt_len is None:
                    cot_positions = feat69.token_positions(ids_row, cot_end_id)
                    traj_positions = feat69.token_positions(ids_row, traj_start_id)
                else:
                    generated_ids = ids_row[prompt_len:]
                    cot_generated = feat69.token_positions(generated_ids, cot_end_id)
                    traj_generated = feat69.token_positions(generated_ids, traj_start_id)
                    if not traj_generated:
                        counters["missing_traj_start"] += 1
                        continue
                    first_traj = int(traj_generated[0])
                    prior_cot = [int(pos) for pos in cot_generated if int(pos) < first_traj]
                    if not prior_cot or int(prior_cot[-1]) < int(args.empty_cot_token_threshold):
                        counters["missing_cot_end"] += 1
                        continue
                    cot_positions = [prompt_len + int(prior_cot[-1])]
                    traj_positions = [prompt_len + first_traj]
                if not cot_positions or not traj_positions:
                    counters["missing_boundary_token"] += 1
                    continue
                cot_pos = int(cot_positions[-1])
                traj_pos = int(traj_positions[-1])
                if cot_pos >= traj_pos:
                    counters["bad_boundary_order"] += 1
                    continue

                selected = select_prefix_indices(attention_np[row_index], end_pos=traj_pos, max_tokens=int(args.max_seq_tokens))
                seq = np.zeros((int(args.max_seq_tokens), hidden.shape[-1]), dtype=np.float16)
                mask = np.zeros((int(args.max_seq_tokens),), dtype=np.bool_)
                take = min(selected.shape[0], int(args.max_seq_tokens))
                if take > 0:
                    seq[-take:] = hidden[row_index, selected[-take:]].astype(np.float16)
                    mask[-take:] = True
                tensors["prefix_hidden"].append(seq)
                tensors["prefix_mask"].append(mask)
                tensors["h_cot_end"].append(hidden[row_index, cot_pos].astype(np.float16))
                tensors["h_traj_start"].append(hidden[row_index, traj_pos].astype(np.float16))
                tensors["target_action"].append(target_action_np[row_index].astype(np.float32))
                tensors["target_traj"].append(target_xyz[row_index].astype(np.float32))
                tensors["ego_history_xyz"].append(history_xyz[row_index].astype(np.float32))
                tensors["ego_history_rot"].append(history_rot[row_index].astype(np.float32))
                tensors["gt_future"].append(gt_future[row_index].astype(np.float32))
                rows_buffer.append(
                    {
                        "sample_id": str(sample.get("sample_id")),
                        "clip_id": str(sample.get("clip_id") or str(sample.get("sample_id", "")).split("__", 1)[0]),
                        "bucket": feat69.bucket_for_traj(target_xyz[row_index]),
                        "cot_end_pos": cot_pos,
                        "traj_start_pos": traj_pos,
                        "valid_prefix_token_count": int(attention_np[row_index, : traj_pos + 1].sum()),
                        "stored_prefix_token_count": int(take),
                        "selection": "uniform_downsample_valid_prefix_tokens_up_to_traj_start",
                    }
                )
                counters["ready"] += 1
                if len(rows_buffer) >= int(args.shard_size):
                    path = flush_shard(
                        split_dir=split_dir,
                        shard_index=shard_index,
                        rows=rows_buffer,
                        tensors=tensors,
                        metadata={
                            "checkpoint_name": args.checkpoint_name,
                            "prefix_type": args.prefix_type,
                            "split_name": split_name,
                            "max_seq_tokens": int(args.max_seq_tokens),
                            "hidden_dim": int(hidden.shape[-1]),
                        },
                    )
                    if path is not None:
                        shard_paths.append(str(path))
                    shard_index += 1
                    rows_buffer = []
                    tensors = {name: [] for name in tensors}

            print(json.dumps({"event": "seq_feature_batch_done", "checkpoint": args.checkpoint_name, "split": split_name, "ready": counters["ready"], "requested": len(ids)}), flush=True)

        path = flush_shard(
            split_dir=split_dir,
            shard_index=shard_index,
            rows=rows_buffer,
            tensors=tensors,
            metadata={
                "checkpoint_name": args.checkpoint_name,
                "prefix_type": args.prefix_type,
                "split_name": split_name,
                "max_seq_tokens": int(args.max_seq_tokens),
            },
        )
        if path is not None:
            shard_paths.append(str(path))
        manifest = {
            "schema_version": "prefix_sequence_feature_split_manifest_v1",
            "checkpoint_name": args.checkpoint_name,
            "prefix_type": args.prefix_type,
            "split_name": split_name,
            "requested": len(ids),
            "ready": int(counters["ready"]),
            "counters": dict(counters),
            "shards": shard_paths,
            "shard_count": len(shard_paths),
            "max_seq_tokens": int(args.max_seq_tokens),
        }
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary["splits"][split_name] = manifest

    summary["elapsed_sec"] = round(time.time() - t0, 3)
    summary_path = args.output_dir / args.checkpoint_name / args.prefix_type / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "seq_feature_extraction_done", "summary": str(summary_path), "elapsed_sec": summary["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
