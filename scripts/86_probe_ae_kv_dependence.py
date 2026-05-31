#!/usr/bin/env python3
"""Probe whether the AE actually conditions on input KV.

Loads a trained AE28 bundle (e.g. the 3.63m checkpoint), picks the most-static
and most-dynamic samples from the corpus, builds a batch of size 1 for each, and
calls sample_paths() with the SAME seed for every sample. Then reports:

1. Deterministic check: same sample, same seed, two runs → action diff (should be 0).
2. Cross-sample diff: pairwise action diff between static/dynamic samples.
   If the AE conditions on KV, cross-kind (static vs dynamic) diff >> intra-kind diff.
   If the AE ignores KV, all diffs are ~ deterministic noise.

NEVER modifies training/loss/build_batch/sample_paths code. Calls existing functions
in 84_train_student_ae28_official.py via importlib.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


_84_PATH = Path(__file__).resolve().parent / "84_train_student_ae28_official.py"
if not _84_PATH.exists():
    raise FileNotFoundError(f"Cannot locate sibling 84 script at {_84_PATH}")
_spec = importlib.util.spec_from_file_location("script_84", _84_PATH)
script_84 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec.loader is not None
_spec.loader.exec_module(script_84)


def _extract_probe_args(argv: list[str]) -> tuple[Path, int, list[str]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ckpt-path", type=Path, required=True)
    pre.add_argument("--n-extremes", type=int, default=4,
                     help="Number of static and dynamic samples to compare (each).")
    parsed, remaining = pre.parse_known_args(argv)
    return parsed.ckpt_path, parsed.n_extremes, remaining


def _pick_extremes(items: list[dict], n_extremes: int) -> tuple[list, list]:
    """Sort by teacher-pred xyz magnitude (the same signal we already log as target_act);
    return n_extremes smallest and n_extremes largest."""
    mags = []
    for i, item in enumerate(items):
        try:
            xyz, _ = script_84.raw_teacher_pred(Path(item["raw_json"]))
            mag = float(np.abs(xyz).mean())
            mags.append((mag, i, item))
        except Exception as exc:
            print(json.dumps({"event": "pick_extremes_skip", "sample_id": item.get("sample_id"),
                              "error": str(exc)}), flush=True)
    mags.sort(key=lambda x: x[0])
    return mags[:n_extremes], mags[-n_extremes:]


def _run_one(bundle, teacher_model, item, args, student, student_processor,
             student_tokenizer, device, seed: int) -> dict:
    batch = script_84.build_batch(
        args=args,
        student=student,
        student_processor=student_processor,
        student_tokenizer=student_tokenizer,
        teacher_model=teacher_model,
        batch_items=[item],
    )
    pred = script_84.sample_paths(
        bundle=bundle,
        teacher_model=teacher_model,
        batch=batch,
        seed=seed,
        device=device,
    )
    target_xyz_np = batch["target_xyz"].detach().cpu().numpy()
    out = {
        "sample_id": item["sample_id"],
        "action": pred["action"][0],
        "pred_xyz": pred["pred_xyz"][0],
        "target_xyz": target_xyz_np[0],
    }
    del batch, pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ckpt_path, n_extremes, remaining = _extract_probe_args(sys.argv[1:])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + list(remaining)
        args = script_84.parse_args()
    finally:
        sys.argv = saved_argv

    print(json.dumps({
        "event": "probe_start",
        "ckpt_path": str(ckpt_path),
        "n_extremes": int(n_extremes),
        "ae_init_mode": str(args.ae_init_mode),
        "prefix_mode": str(args.prefix_mode),
        "num_samples": int(args.num_samples),
    }), flush=True)

    items = script_84.select_items(args)

    student, student_tokenizer, student_processor, base_model = script_84.load_student(args)
    teacher_model, *_ = script_84.load_model_and_processor(
        checkpoint_path=args.teacher_checkpoint_path,
        dtype=script_84.torch_dtype_from_name(args.ae_dtype),
        device=args.teacher_load_device,
        config_json=None,
        runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=163840,
        max_pixels=196608,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    script_84.force_attention(
        teacher_model.expert,
        "sdpa" if args.attn_implementation != "eager" else "eager",
    )
    bundle, _ = script_84.build_bundle(teacher_model, args, student=student)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bundle.load_state_dict(state["bundle_state_dict"], strict=False)
    bundle = bundle.to(device=args.device, dtype=script_84.torch_dtype_from_name(args.ae_dtype))
    bundle.eval()
    print(json.dumps({"event": "bundle_loaded",
                      "ckpt_payload_step": state.get("payload", {}).get("step")}), flush=True)

    if hasattr(teacher_model, "vlm"):
        delattr(teacher_model, "vlm")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(args.device)
    static_picks, dynamic_picks = _pick_extremes(items, n_extremes)
    print(json.dumps({
        "event": "extremes_picked",
        "static_target_xyz_abs_means": [m for m, _, _ in static_picks],
        "dynamic_target_xyz_abs_means": [m for m, _, _ in dynamic_picks],
    }), flush=True)

    PROBE_SEED = 12345
    results = []
    for mag, _idx, item in static_picks + dynamic_picks:
        kind = "static" if mag <= static_picks[-1][0] else "dynamic"
        r = _run_one(bundle, teacher_model, item, args, student, student_processor,
                     student_tokenizer, device, PROBE_SEED)
        r["kind"] = kind
        r["target_xyz_abs_mean"] = mag
        results.append(r)
        print(json.dumps({
            "event": "sample_done", "kind": kind, "sample_id": r["sample_id"],
            "target_xyz_abs_mean": float(mag),
            "action_abs_mean": float(np.abs(r["action"]).mean()),
            "pred_xyz_endpoint_norm_m": float(np.linalg.norm(r["pred_xyz"][-1])),
        }), flush=True)

    # Deterministic check: re-run first static sample with same seed.
    first_static_item = static_picks[0][2]
    r_first = results[0]
    r_repeat = _run_one(bundle, teacher_model, first_static_item, args, student,
                         student_processor, student_tokenizer, device, PROBE_SEED)
    det_action_diff_max = float(np.abs(r_first["action"] - r_repeat["action"]).max())
    det_action_diff_mean = float(np.abs(r_first["action"] - r_repeat["action"]).mean())
    det_xyz_endpoint_diff_m = float(np.linalg.norm(r_first["pred_xyz"][-1] - r_repeat["pred_xyz"][-1]))
    print(json.dumps({
        "event": "deterministic_check",
        "sample_id": r_first["sample_id"],
        "same_sample_2x_action_diff_max": det_action_diff_max,
        "same_sample_2x_action_diff_mean": det_action_diff_mean,
        "same_sample_2x_xyz_endpoint_diff_m": det_xyz_endpoint_diff_m,
    }), flush=True)

    # Pairwise diffs between all results (static-static, dynamic-dynamic, static-dynamic).
    pairs = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            action_diff_mean = float(np.abs(a["action"] - b["action"]).mean())
            xyz_endpoint_diff_m = float(np.linalg.norm(a["pred_xyz"][-1] - b["pred_xyz"][-1]))
            same_kind = a["kind"] == b["kind"]
            pairs.append({
                "a_id": a["sample_id"], "a_kind": a["kind"], "a_mag": a["target_xyz_abs_mean"],
                "b_id": b["sample_id"], "b_kind": b["kind"], "b_mag": b["target_xyz_abs_mean"],
                "action_diff_mean": action_diff_mean,
                "xyz_endpoint_diff_m": xyz_endpoint_diff_m,
                "same_kind": same_kind,
            })
    print(json.dumps({"event": "pairs", "pairs": pairs}), flush=True)

    # Summary stats.
    same_kind_action = [p["action_diff_mean"] for p in pairs if p["same_kind"]]
    cross_kind_action = [p["action_diff_mean"] for p in pairs if not p["same_kind"]]
    same_kind_xyz = [p["xyz_endpoint_diff_m"] for p in pairs if p["same_kind"]]
    cross_kind_xyz = [p["xyz_endpoint_diff_m"] for p in pairs if not p["same_kind"]]
    summary = {
        "event": "summary",
        "deterministic_same_sample_action_diff_max": det_action_diff_max,
        "deterministic_same_sample_xyz_endpoint_diff_m": det_xyz_endpoint_diff_m,
        "same_kind_pair_action_diff_mean": float(np.mean(same_kind_action)) if same_kind_action else None,
        "cross_kind_pair_action_diff_mean": float(np.mean(cross_kind_action)) if cross_kind_action else None,
        "same_kind_pair_xyz_endpoint_diff_m_mean": float(np.mean(same_kind_xyz)) if same_kind_xyz else None,
        "cross_kind_pair_xyz_endpoint_diff_m_mean": float(np.mean(cross_kind_xyz)) if cross_kind_xyz else None,
        "verdict_hint": (
            "AE_IGNORES_KV"
            if (cross_kind_action and det_action_diff_max > 0
                and float(np.mean(cross_kind_action)) < 5 * det_action_diff_max)
            else "AE_USES_KV"
        ),
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
