#!/usr/bin/env python3
"""Build a reproducible manifest for patched Alpamayo 1.5 discrete trajectory-teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.consistency import normalize_action_class
from src.data.path_semantics import extract_path_semantics
from src.inference.checkpoint_eval import TrajectoryTokenDecoder, _max_same_token_run, _jaccard, resolve_traj_tokenizer_config_path
from src.utils.runtime_paths import remap_external_path


PROJECT_ROOT = Path("/workspace/cosmos_distillation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/teacher_cache/traj15"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_traj15_manifest.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "teacher_traj15_manifest_summary.json",
    )
    parser.add_argument("--source-model", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--source-weights", default="patched_local_discrete_extraction")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--extraction-mode", default="patched_alpamayo15_direct_discrete_teacher_forced")
    parser.add_argument("--teacher-kind", default="alpamayo1.5_patched_direct_discrete")
    return parser.parse_args()


def _sample_dir_from_output(output: dict[str, Any]) -> Path | None:
    sample_root_raw = output.get("sample_root")
    sample_id = str(output.get("sample_id") or "")
    if sample_root_raw in (None, "") or not sample_id:
        return None
    remapped = remap_external_path(sample_root_raw)
    sample_root = Path(remapped) if remapped not in (None, "") else Path(sample_root_raw)
    if (sample_root / sample_id).exists():
        return sample_root / sample_id
    if sample_root.exists():
        return sample_root
    return None


def _load_history_xyz(sample_dir: Path) -> np.ndarray | None:
    for path in (
        sample_dir / "ego" / "ego_history_xyz.npy",
        sample_dir / "ego_history_xyz.npy",
    ):
        if path.exists():
            return np.load(path).astype(np.float32)
    return None


def _load_history_rot(sample_dir: Path, history_xyz: np.ndarray | None) -> np.ndarray | None:
    for path in (
        sample_dir / "ego" / "ego_history_rot.npy",
        sample_dir / "ego_history_rot.npy",
    ):
        if path.exists():
            return np.load(path).astype(np.float32)
    if history_xyz is None or len(history_xyz) == 0:
        return None
    xy = np.asarray(history_xyz, dtype=np.float32)[:, :2]
    deltas = np.diff(xy, axis=0, prepend=xy[:1])
    headings = np.zeros((len(xy),), dtype=np.float32)
    for index in range(1, len(xy)):
        dx = float(deltas[index, 0])
        dy = float(deltas[index, 1])
        if abs(dx) > 1e-5 or abs(dy) > 1e-5:
            headings[index] = np.arctan2(dy, dx)
        else:
            headings[index] = headings[index - 1]
    rotations = np.zeros((len(xy), 3, 3), dtype=np.float32)
    for index, yaw in enumerate(headings):
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        rotations[index] = np.array(
            [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
    return rotations


def _load_gt_future_xyz(sample_dir: Path) -> np.ndarray | None:
    for path in (
        sample_dir / "ego" / "ego_future_xyz.npy",
        sample_dir / "ego_future_xyz.npy",
    ):
        if path.exists():
            return np.load(path).astype(np.float32)
    return None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    decoder_config_path = resolve_traj_tokenizer_config_path()
    decoder = TrajectoryTokenDecoder(config_path=decoder_config_path) if decoder_config_path is not None else None

    output_dir = args.cache_dir / "outputs"
    token_dir = args.cache_dir / "tokens"
    hidden_dir = args.cache_dir / "hidden"
    topk_dir = args.cache_dir / "topk"

    records: list[dict[str, Any]] = []
    syntax_ok_count = 0
    exact_128_count = 0
    decoded_ok_count = 0

    for output_path in sorted(output_dir.glob("*.json")):
        output = json.loads(output_path.read_text(encoding="utf-8"))
        sample_id = str(output.get("sample_id") or "")
        if not sample_id:
            continue
        token_path = token_dir / f"{sample_id}.teacher_traj15.tokens.npy"
        hidden_path = hidden_dir / f"{sample_id}.teacher_traj15.hidden.npy"
        topk_path = topk_dir / f"{sample_id}.teacher_traj15.topk_logits.npz"
        if not token_path.exists():
            continue

        token_ids = np.load(token_path).astype(np.int32).reshape(-1)
        hidden_shape = list(np.load(hidden_path, mmap_mode="r").shape) if hidden_path.exists() else None
        topk_shape = None
        if topk_path.exists():
            with np.load(topk_path) as topk_npz:
                topk_shape = list(topk_npz["topk_indices"].shape)

        best_candidate_index = int(output.get("best_candidate_index", -1) or -1)
        candidates = list(output.get("rollout_candidates") or [])
        best_candidate = candidates[best_candidate_index] if 0 <= best_candidate_index < len(candidates) else {}
        decoded_text = str(best_candidate.get("decoded_text") or "")
        best_cot_text = str(output.get("best_cot_text") or best_candidate.get("cot_text") or "")
        syntax_ok = bool(output.get("best_has_cot_end")) and bool(output.get("best_has_traj_future_start")) and bool(output.get("best_has_traj_future_end"))
        exact_128 = int(output.get("teacher_traj_token_count", 0) or 0) == 128 and int(token_ids.shape[0]) == 128

        unique_ids = int(len(set(int(value) for value in token_ids.tolist())))
        max_same_token_run = int(_max_same_token_run(token_ids.tolist()))
        sample_dir = _sample_dir_from_output(output)
        history_xyz = _load_history_xyz(sample_dir) if sample_dir is not None else None
        history_rot = _load_history_rot(sample_dir, history_xyz) if sample_dir is not None else None
        gt_future_xyz = _load_gt_future_xyz(sample_dir) if sample_dir is not None else None

        teacher_discrete_xyz = None
        teacher_motion_class = "unknown"
        gt_motion_class = "unknown"
        discrete_ade_m = output.get("best_candidate_ade_m")
        discrete_fde_m = output.get("best_candidate_fde_m")
        gt_set_jaccard = None
        if decoder is not None and history_xyz is not None and history_rot is not None and len(token_ids) == 128:
            try:
                teacher_discrete_xyz = decoder.decode(history_xyz, history_rot, token_ids.tolist())
            except Exception:  # noqa: BLE001
                teacher_discrete_xyz = None
        if teacher_discrete_xyz is not None:
            teacher_motion_class = normalize_action_class(extract_path_semantics(teacher_discrete_xyz).action_class)
            decoded_ok_count += 1
        if gt_future_xyz is not None:
            gt_motion_class = normalize_action_class(extract_path_semantics(gt_future_xyz).action_class)
        if gt_future_xyz is not None and teacher_discrete_xyz is not None:
            gt_token_like = []  # manifest keeps overlap field reserved even without GT discrete cache
            gt_set_jaccard = _jaccard(token_ids.tolist(), gt_token_like) if gt_token_like else None

        record = {
            "manifest_version": "traj15_pair_v1",
            "sample_id": sample_id,
            "status": output.get("status"),
            "source_model": args.source_model,
            "source_weights": args.source_weights,
            "source_commit": args.source_commit,
            "extraction_mode": args.extraction_mode,
            "teacher_kind": args.teacher_kind,
            "teacher_pair_kind": "honest_joint_pair",
            "teacher_cache_dir": str(args.cache_dir),
            "best_candidate_index": best_candidate_index,
            "best_cot_text": best_cot_text,
            "full_completion_hash": _sha256_text(decoded_text) if decoded_text else None,
            "cot_text_hash": _sha256_text(best_cot_text) if best_cot_text else None,
            "teacher_traj_token_ids": token_ids.tolist(),
            "teacher_traj_token_count": int(token_ids.shape[0]),
            "teacher_traj_tokens_path": str(token_path),
            "teacher_traj_hidden_path": str(hidden_path) if hidden_path.exists() else None,
            "teacher_traj_topk_path": str(topk_path) if topk_path.exists() else None,
            "teacher_traj_hidden_shape": hidden_shape,
            "teacher_traj_topk_shape": topk_shape,
            "syntax_ok": syntax_ok,
            "all_teacher_traj_bodies_are_exactly_128_tokens": exact_128,
            "unique_ids": unique_ids,
            "max_same_token_run": max_same_token_run,
            "discrete_ade_m": discrete_ade_m,
            "discrete_fde_m": discrete_fde_m,
            "expert_ade_m": best_candidate.get("expert_ade_m"),
            "expert_fde_m": best_candidate.get("expert_fde_m"),
            "teacher_motion_class": teacher_motion_class,
            "gt_motion_class": gt_motion_class,
            "teacher_forced_elapsed_sec": output.get("teacher_forced_elapsed_sec"),
            "quality_multiplier": 1.0 if syntax_ok and exact_128 else 0.0,
            "quality_multiplier_basis": {
                "syntax_ok": syntax_ok,
                "exact_128": exact_128,
            },
            "sample_root": output.get("sample_root"),
            "decoder_config_path": str(decoder_config_path) if decoder_config_path is not None else None,
            "gt_set_jaccard": gt_set_jaccard,
        }
        records.append(record)
        syntax_ok_count += int(syntax_ok)
        exact_128_count += int(exact_128)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "manifest_version": "traj15_pair_v1",
        "cache_dir": str(args.cache_dir),
        "records": len(records),
        "syntax_ok_records": syntax_ok_count,
        "exact_128_records": exact_128_count,
        "decoded_motion_records": decoded_ok_count,
        "source_model": args.source_model,
        "source_weights": args.source_weights,
        "source_commit": args.source_commit,
        "extraction_mode": args.extraction_mode,
        "teacher_kind": args.teacher_kind,
        "output_jsonl": str(args.output_jsonl),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
