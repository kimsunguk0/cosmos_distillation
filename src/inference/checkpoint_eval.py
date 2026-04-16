"""Decode-based anti-collapse evaluation for trajectory checkpoints."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from src.data.consistency import normalize_action_class
from src.data.path_semantics import extract_path_semantics
from src.inference.decoding import (
    StopOnTrajEndCriteria,
    StopOnTrajOnlyEndCriteria,
    TrajDecodingContract,
    TrajOnlyDecodingContract,
    TrajOnlyLogitsProcessor,
    TrajSpanLogitsProcessor,
)
from src.training.collator import (
    build_messages,
    build_traj_only_prompt,
    build_user_prompt,
    load_ego_history_xyz,
    load_sample_images,
    resolve_sample_path,
)
from src.utils.runtime_paths import remap_external_path


ALPAMAYO_SRC = Path("/workspace/alpamayo_repos/alpamayo1.5/src")
ALPAMAYO_CONFIG_CANDIDATES = (
    Path("/workspace/base_models_weights/Alpamayo-1.5-10B/config.json"),
    Path("/workspace/base_models_weights/Alpamayo-R1-10B/config.json"),
)


def _ensure_alpamayo_imports() -> None:
    if str(ALPAMAYO_SRC) not in sys.path and ALPAMAYO_SRC.exists():
        sys.path.insert(0, str(ALPAMAYO_SRC))


def _decode_generated_text(tokenizer, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> str:
    prompt_len = int(input_ids.shape[-1])
    new_tokens = generated_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def extract_generated_traj_tokens(generated_text: str) -> list[int]:
    return [int(match) for match in re.findall(r"<i(\d+)>", generated_text)]


def _max_same_token_run(token_ids: Sequence[int]) -> int:
    if not token_ids:
        return 0
    best = 1
    current = 1
    for left, right in zip(token_ids, token_ids[1:]):
        if left == right:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def _resolve_existing_path(raw_path: str | Path | None) -> Path | None:
    remapped = remap_external_path(raw_path)
    if remapped in (None, ""):
        return None
    path = Path(remapped)
    return path if path.exists() else None


def load_ego_history_rot(sample: dict[str, Any], project_root: Path) -> np.ndarray:
    """Load history rotations or reconstruct a stable local-frame yaw sequence."""
    sample_input = sample.get("input") or {}
    for raw_path in (
        sample_input.get("ego_history_rot_path"),
        resolve_sample_path(sample, project_root) / "ego" / "ego_history_rot.npy",
        resolve_sample_path(sample, project_root) / "ego_history_rot.npy",
    ):
        if isinstance(raw_path, Path):
            path = raw_path
        else:
            path = _resolve_existing_path(raw_path)
        if path is not None and path.exists():
            return np.load(path).astype(np.float32)

    history_xyz = load_ego_history_xyz(sample, project_root)
    xy = np.asarray(history_xyz, dtype=np.float32)[:, :2]
    if len(xy) == 0:
        return np.zeros((0, 3, 3), dtype=np.float32)
    deltas = np.diff(xy, axis=0, prepend=xy[:1])
    headings = np.zeros((len(xy),), dtype=np.float32)
    for index in range(1, len(xy)):
        dx = float(deltas[index, 0])
        dy = float(deltas[index, 1])
        if abs(dx) > 1e-5 or abs(dy) > 1e-5:
            headings[index] = math.atan2(dy, dx)
        else:
            headings[index] = headings[index - 1]
    rotations = np.zeros((len(xy), 3, 3), dtype=np.float32)
    for index, yaw in enumerate(headings):
        cos_yaw = math.cos(float(yaw))
        sin_yaw = math.sin(float(yaw))
        rotations[index] = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return rotations


class TrajectoryTokenDecoder:
    """Decode Alpamayo discrete trajectory tokens into local-frame future xyz."""

    def __init__(self, *, config_path: Path) -> None:
        _ensure_alpamayo_imports()
        from alpamayo1_5.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

        config = json.loads(config_path.read_text(encoding="utf-8"))
        traj_cfg = config["traj_tokenizer_cfg"]
        action_space_cfg = traj_cfg["action_space_cfg"]
        if "UnicycleAccelCurvatureActionSpace" not in str(action_space_cfg.get("_target_", "")):
            raise ValueError(f"Unsupported action space target: {action_space_cfg.get('_target_')}")
        action_space = UnicycleAccelCurvatureActionSpace(
            accel_mean=float(action_space_cfg["accel_mean"]),
            accel_std=float(action_space_cfg["accel_std"]),
            curvature_mean=float(action_space_cfg["curvature_mean"]),
            curvature_std=float(action_space_cfg["curvature_std"]),
            accel_bounds=tuple(float(value) for value in action_space_cfg["accel_bounds"]),
            curvature_bounds=tuple(float(value) for value in action_space_cfg["curvature_bounds"]),
            dt=float(action_space_cfg["dt"]),
            n_waypoints=int(action_space_cfg["n_waypoints"]),
            theta_lambda=float(action_space_cfg["theta_lambda"]),
            theta_ridge=float(action_space_cfg["theta_ridge"]),
            v_lambda=float(action_space_cfg["v_lambda"]),
            v_ridge=float(action_space_cfg["v_ridge"]),
            a_lambda=float(action_space_cfg["a_lambda"]),
            a_ridge=float(action_space_cfg["a_ridge"]),
            kappa_lambda=float(action_space_cfg["kappa_lambda"]),
            kappa_ridge=float(action_space_cfg["kappa_ridge"]),
        )
        self.action_space = action_space
        self.dims_min = torch.tensor(traj_cfg["dims_min"], dtype=torch.float32)
        self.dims_max = torch.tensor(traj_cfg["dims_max"], dtype=torch.float32)
        self.num_bins = int(traj_cfg["num_bins"])
        self.n_waypoints = int(action_space_cfg["n_waypoints"])

    def decode(self, history_xyz: np.ndarray, history_rot: np.ndarray, token_ids: Sequence[int]) -> np.ndarray | None:
        if len(token_ids) != self.n_waypoints * 2:
            return None
        tokens = torch.tensor(token_ids, dtype=torch.long).reshape(1, self.n_waypoints, 2)
        action = tokens.to(dtype=torch.float32) / float(self.num_bins - 1)
        action = action * (self.dims_max - self.dims_min) + self.dims_min
        hist_xyz = torch.from_numpy(np.asarray(history_xyz, dtype=np.float32)).unsqueeze(0)
        hist_rot = torch.from_numpy(np.asarray(history_rot, dtype=np.float32)).unsqueeze(0)
        future_xyz, _ = self.action_space.action_to_traj(action, hist_xyz, hist_rot)
        return future_xyz.squeeze(0).detach().cpu().numpy()


def resolve_traj_tokenizer_config_path(student_model: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if student_model not in (None, ""):
        model_path = Path(student_model).expanduser()
        if model_path.is_dir():
            candidates.append(model_path / "config.json")
    candidates.extend(ALPAMAYO_CONFIG_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            try:
                config = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if "traj_tokenizer_cfg" in config:
                return candidate
    return None


@dataclass(slots=True)
class DecodeEvalConfig:
    enabled: bool = False
    split: str = "val"
    num_samples: int = 8
    max_new_tokens: int = 160
    prompt_mode: str = "joint"
    target_mode: str = "joint"
    metric_name: str = "anti_collapse_score"


def _select_rows(records: Sequence[dict[str, Any]], *, split: str, num_samples: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in records:
        if row.get("split") != split:
            continue
        selected.append(row)
        if len(selected) >= num_samples:
            break
    return selected


@torch.inference_mode()
def evaluate_decode_subset(
    model,
    *,
    tokenizer,
    processor,
    records: Sequence[dict[str, Any]],
    device: torch.device,
    project_root: Path,
    config: DecodeEvalConfig,
    student_model: str | Path | None = None,
) -> dict[str, Any]:
    """Run greedy constrained generation on a small subset and score anti-collapse quality."""
    if not config.enabled or config.num_samples <= 0:
        return {"enabled": False}

    selected = _select_rows(records, split=config.split, num_samples=config.num_samples)
    if not selected:
        return {"enabled": True, "num_samples": 0}

    decoder: TrajectoryTokenDecoder | None = None
    decoder_config_path = resolve_traj_tokenizer_config_path(student_model)
    if decoder_config_path is not None:
        try:
            decoder = TrajectoryTokenDecoder(config_path=decoder_config_path)
        except Exception:  # noqa: BLE001
            decoder = None

    sample_metrics: list[dict[str, Any]] = []
    motion_matches = 0
    motion_total = 0
    unique_values: list[float] = []
    max_runs: list[float] = []
    jaccards: list[float] = []
    token_count_matches = 0

    for sample in selected:
        history_xyz = load_ego_history_xyz(sample, project_root)
        prompt_text = (
            build_traj_only_prompt(sample, project_root, ego_history_xyz=history_xyz)
            if config.prompt_mode == "traj_only"
            else build_user_prompt(sample, project_root, ego_history_xyz=history_xyz)
        )
        assistant_prefix = "<|traj_future_start|>" if config.target_mode == "traj_only" else "<|cot_start|>"
        images = load_sample_images(sample, project_root)
        messages = build_messages(prompt_text, len(images), assistant_prefix=assistant_prefix)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        batch = processor(
            text=[text],
            images=[images],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        target_traj = [int(token_id) for token_id in (sample.get("hard_target") or {}).get("traj_future_token_ids") or []]
        prompt_lengths = batch["attention_mask"].sum(dim=1).tolist()
        if config.target_mode == "traj_only":
            contract = TrajOnlyDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=len(target_traj),
            )
            logits_processor = LogitsProcessorList([TrajOnlyLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajOnlyEndCriteria(contract)])
        else:
            contract = TrajDecodingContract.from_tokenizer(
                tokenizer,
                prompt_lengths=prompt_lengths,
                traj_token_count=len(target_traj),
            )
            logits_processor = LogitsProcessorList([TrajSpanLogitsProcessor(contract)])
            stopping_criteria = StoppingCriteriaList([StopOnTrajEndCriteria(contract)])

        generated = model.backbone.generate(
            **batch,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            use_cache=True,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
        )
        generated_text = _decode_generated_text(tokenizer, batch["input_ids"], generated)
        generated_traj = extract_generated_traj_tokens(generated_text)
        unique_count = float(len(set(generated_traj)))
        max_same_run = float(_max_same_token_run(generated_traj))
        jaccard = _jaccard(generated_traj, target_traj)
        count_match = int(len(generated_traj) == len(target_traj))
        token_count_matches += count_match

        predicted_motion = "unknown"
        gt_motion = normalize_action_class((sample.get("derived") or {}).get("gt_motion_class"))
        motion_match = None
        if decoder is not None:
            history_rot = load_ego_history_rot(sample, project_root)
            decoded_future_xyz = decoder.decode(history_xyz, history_rot, generated_traj)
            if decoded_future_xyz is not None:
                predicted_motion = normalize_action_class(extract_path_semantics(decoded_future_xyz).action_class)
                if gt_motion != "unknown":
                    motion_total += 1
                    motion_match = int(predicted_motion == gt_motion)
                    motion_matches += motion_match

        unique_values.append(unique_count)
        max_runs.append(max_same_run)
        jaccards.append(jaccard)
        sample_metrics.append(
            {
                "sample_id": sample.get("sample_id"),
                "generated_traj_token_count": len(generated_traj),
                "unique_traj_ids": unique_count,
                "max_same_token_run": max_same_run,
                "target_set_jaccard": jaccard,
                "predicted_motion_class": predicted_motion,
                "gt_motion_class": gt_motion,
                "motion_match": motion_match,
            }
        )

    avg_unique = float(sum(unique_values) / max(len(unique_values), 1))
    avg_max_run = float(sum(max_runs) / max(len(max_runs), 1))
    avg_jaccard = float(sum(jaccards) / max(len(jaccards), 1))
    motion_agreement = float(motion_matches / motion_total) if motion_total > 0 else 0.0
    token_count_match_rate = float(token_count_matches / max(len(sample_metrics), 1))

    normalized_unique = min(avg_unique / 32.0, 1.0)
    repetition_penalty = 1.0 - min(avg_max_run / 128.0, 1.0)
    anti_collapse_score = (
        0.30 * normalized_unique
        + 0.25 * repetition_penalty
        + 0.25 * avg_jaccard
        + 0.20 * motion_agreement
    )

    return {
        "enabled": True,
        "split": config.split,
        "num_samples": len(sample_metrics),
        "avg_unique_traj_ids": avg_unique,
        "avg_max_same_token_run": avg_max_run,
        "avg_target_set_jaccard": avg_jaccard,
        "token_count_match_rate": token_count_match_rate,
        "coarse_motion_agreement_rate": motion_agreement,
        "anti_collapse_score": float(anti_collapse_score),
        "traj_tokenizer_config": str(decoder_config_path) if decoder_config_path is not None else None,
        "samples": sample_metrics,
    }
