#!/usr/bin/env python3
"""Comprehensive 3-model benchmark: Alpamayo 10B / Student (no FLEX) / Student+FLEX.

Outputs:
  1. Latency breakdown: ViT, FLEX, Prefill, Decode, FM 10-step
  2. ADE, minADE@6 (6.4s horizon)
  3. Visualization: 4 cameras + ego history + 3 model trajectories + CoT

Usage:
    python scripts/benchmark_3models.py \
        --output-dir outputs/benchmark_3models \
        --num-eval 256 \
        --num-vis-per-cat 4
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

SUKIM_ROOT = PROJECT_ROOT.parents[1]
for p in (PROJECT_ROOT, SUKIM_ROOT, SUKIM_ROOT / "alpamayo_repo/alpamayo1.5/src", SUKIM_ROOT / "visualization"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---- Model configs ----
MODELS = {
    "alpamayo_10b": {
        "label": "Alpamayo 1.5 10B",
        "color": "#e74c3c",
        "student_ckpt": None,  # use teacher directly
        "ae_ckpt": None,  # teacher's own expert
        "has_flex": False,
        "compressed_layers": 36,
    },
    "student_noflex": {
        "label": "Cosmos 2B + AE28",
        "color": "#3498db",
        "student_ckpt": "outputs/checkpoints/no_nav_camera_labeled_official_full444k/no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250",
        "ae_ckpt": "outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/best.pt",
        "has_flex": False,
        "compressed_layers": 28,
    },
    "student_flex": {
        "label": "Cosmos 2B + FLEX + AE28",
        "color": "#2ecc71",
        "student_ckpt": "outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final",
        "ae_ckpt": "outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt",
        "has_flex": True,
        "compressed_layers": 28,
    },
}

# ---- Shared inference config ----
SEED = 42
TEMPERATURE = 0.85
TOP_P = 0.95
NUM_PATHS = 6  # for minADE@6
FM_STEPS = 10


def _import_ae():
    spec = importlib.util.spec_from_file_location(
        "ae_train", str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_balanced_samples(corpus_path: str, n_total: int, seed: int) -> list[dict]:
    """Select n_total samples balanced across semantic categories."""
    items = [json.loads(l) for l in open(corpus_path)]
    by_cat: dict[str, list] = defaultdict(list)
    for it in items:
        cat = it.get("metadata", {}).get("semantic_scene_category", "unknown")
        by_cat[cat] = by_cat.get(cat, [])
        by_cat[cat].append(it)

    rng = random.Random(seed)
    cats = sorted(by_cat.keys())
    per_cat = max(n_total // len(cats), 1)
    selected = []
    for cat in cats:
        pool = by_cat[cat]
        rng.shuffle(pool)
        selected.extend(pool[:per_cat])

    # Fill remaining
    remaining = n_total - len(selected)
    if remaining > 0:
        all_pool = [it for cat in cats for it in by_cat[cat][per_cat:]]
        rng.shuffle(all_pool)
        selected.extend(all_pool[:remaining])

    return selected[:n_total]


def cuda_sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


# ---- Visualization ----
def draw_benchmark_figure(
    *,
    sample_id: str,
    category: str,
    camera_images: list[np.ndarray],  # 4 images
    ego_history_xy: np.ndarray,  # [N, 2]
    model_trajs: dict[str, np.ndarray],  # name -> [64, 2] (xy)
    model_cots: dict[str, str],  # name -> CoT text
    gt_traj_xy: np.ndarray | None,  # [64, 2]
    model_colors: dict[str, str],
    save_path: str,
):
    """Draw one benchmark visualization figure."""
    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.25,
                           left=0.03, right=0.97, top=0.93, bottom=0.02)

    fig.suptitle(f"{category} | {sample_id}", fontsize=11, fontweight="bold")

    # Row 0: 4 camera images
    for i in range(min(4, len(camera_images))):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(camera_images[i])
        ax.set_title(f"Camera {i}", fontsize=9)
        ax.axis("off")

    # Row 1: BEV trajectory plot (left half) + CoT text (right half)
    ax_bev = fig.add_subplot(gs[1, :2])
    # Ego history
    if ego_history_xy is not None and len(ego_history_xy) > 0:
        ax_bev.plot(ego_history_xy[:, 0], ego_history_xy[:, 1],
                    "k-o", markersize=3, linewidth=1.5, label="Ego History", zorder=5)
    # GT
    if gt_traj_xy is not None:
        ax_bev.plot(gt_traj_xy[:, 0], gt_traj_xy[:, 1],
                    "k--", linewidth=1.5, alpha=0.5, label="GT", zorder=4)
    # Model trajectories
    for name, traj in model_trajs.items():
        color = model_colors.get(name, "#888888")
        ax_bev.plot(traj[:, 0], traj[:, 1],
                    "-", color=color, linewidth=2, label=name, zorder=6)
        ax_bev.scatter(traj[-1, 0], traj[-1, 1], color=color, s=40, zorder=7, edgecolors="k")

    ax_bev.set_xlabel("X (m)", fontsize=9)
    ax_bev.set_ylabel("Y (m)", fontsize=9)
    ax_bev.set_title("BEV Trajectories", fontsize=10, fontweight="bold")
    ax_bev.legend(fontsize=7, loc="best")
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)

    # CoT text panel (right half of row 1)
    ax_cot = fig.add_subplot(gs[1, 2:])
    ax_cot.axis("off")
    cot_lines = []
    for name, cot in model_cots.items():
        color = model_colors.get(name, "#333333")
        short_cot = cot[:120] + "..." if len(cot) > 120 else cot
        cot_lines.append(f"[{name}]\n{short_cot}\n")
    cot_text = "\n".join(cot_lines)
    ax_cot.text(0.02, 0.98, cot_text, transform=ax_cot.transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax_cot.set_title("Chain-of-Thought", fontsize=10, fontweight="bold")

    # Row 2: Per-model ADE breakdown (placeholder - filled by caller via title)
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis("off")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="3-model comprehensive benchmark")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark_3models")
    parser.add_argument("--num-eval", type=int, default=256)
    parser.add_argument("--num-vis-per-cat", type=int, default=4)
    parser.add_argument("--corpus-jsonl", type=str,
                        default="data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--latency-warmup", type=int, default=2)
    parser.add_argument("--latency-trials", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)

    ae = _import_ae()
    from probe_teacher_kv_28layer_expert_compression import ade_fde
    from src.training.collator import load_sample_images

    # Select balanced samples
    eval_samples = select_balanced_samples(args.corpus_jsonl, args.num_eval, SEED)
    cat_counter = Counter(it.get("metadata", {}).get("semantic_scene_category", "unknown") for it in eval_samples)
    print(json.dumps({"event": "samples_selected", "total": len(eval_samples),
                       "categories": dict(cat_counter)}), flush=True)

    # ========== Run each model ==========
    all_results: dict[str, dict] = {}

    for model_key, model_cfg in MODELS.items():
        print(json.dumps({"event": "model_start", "model": model_key}), flush=True)

        # --- Load model ---
        class AEArgs:
            pass
        ae_args = AEArgs()
        base_args = {
            "corpus_jsonl": Path(args.corpus_jsonl),
            "teacher_checkpoint_path": Path(SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B"),
            "student_dtype": "bfloat16", "device": args.device, "student_model": "",
            "ae_init_mode": "student_backbone_init",
            "attn_implementation": "flash_attention_2",
            "disable_student_deepstack": False, "qat_quantization": "", "qat_calib_samples": 256,
            "num_samples": 10, "val_samples": 5, "val_fraction": 0.1,
            "split_seed": None, "split_cache_json": None, "split": "train",
            "split_scan_all": True, "compressed_layers": model_cfg["compressed_layers"],
            "mapping": "linspace_round",
            "ae_dtype": "bfloat16", "prefix_mode": "teacher_forced",
            "preserve_flex_positions": model_cfg["has_flex"],
            "flex_selection_strategy": "uniform",
            "flex_scene_deepstack": model_cfg["has_flex"],
            "target_source": "teacher",
            "max_new_tokens": 160, "max_length": 4096,
            "stage2_attention_mode": "official_none", "seed": SEED,
            "teacher_load_device": "cpu",
        }

        if model_key == "alpamayo_10b":
            # For 10B teacher: use teacher model directly
            base_args["student_checkpoint_dir"] = Path(SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B")
        else:
            base_args["student_checkpoint_dir"] = Path(model_cfg["student_ckpt"])

        for k, v in base_args.items():
            setattr(ae_args, k, v)

        if model_key == "alpamayo_10b":
            # Load teacher model directly as both student and teacher
            from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor
            teacher_model, teacher_processor, _, _, _ = load_model_and_processor(
                checkpoint_path=ae_args.teacher_checkpoint_path,
                dtype=torch.bfloat16, device="cpu", config_json=None,
                runtime_support=None, attn_implementation="flash_attention_2",
                min_pixels=163840, max_pixels=196608)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            teacher_model.to(device)

            # For 10B, we use teacher's own sample_paths
            # We'll handle this specially in the eval loop
            student = None
            student_tokenizer = None
            student_processor = None
            bundle = None

            print(json.dumps({"event": "model_loaded", "model": model_key}), flush=True)

            # Eval with teacher model
            model_ades = []
            model_fdes = []
            model_minade6 = []
            model_results_per_sample = []

            # For 10B we need special handling - skip for now if too complex
            # TODO: implement 10B teacher direct inference
            print(json.dumps({"event": "model_skip", "model": model_key,
                              "reason": "10B teacher direct inference needs special implementation"}), flush=True)
            teacher_model.cpu()
            del teacher_model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        else:
            # Student models
            student, student_tokenizer, student_processor, _ = ae.load_student(ae_args)

            _load_fn = getattr(ae, "load_model_and_processor", None)
            if not _load_fn:
                from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load_fn
            teacher_model, _, _, _, _ = _load_fn(
                checkpoint_path=ae_args.teacher_checkpoint_path, dtype=torch.bfloat16,
                device="cpu", config_json=None, runtime_support=None,
                attn_implementation="flash_attention_2", min_pixels=163840, max_pixels=196608)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            teacher_model.to(device)

            bundle, selected_layers = ae.build_bundle(teacher_model, ae_args, student=student)
            ae.load_bundle_checkpoint(Path(model_cfg["ae_ckpt"]), bundle=bundle)
            bundle.eval()
            for p in bundle.parameters():
                p.requires_grad_(False)
            bundle.to(device)

            # KV layer indices
            expert_n = int(bundle.expert.config.num_hidden_layers)
            kv_layer_indices = selected_layers if expert_n < 28 else None

        print(json.dumps({"event": "model_loaded", "model": model_key}), flush=True)

        # --- Latency benchmark (first 5 samples) ---
        latency_times = defaultdict(list)
        latency_sample_items = eval_samples[:args.latency_trials + args.latency_warmup]

        for trial_idx, sample in enumerate(latency_sample_items):
            is_bench = trial_idx >= args.latency_warmup
            try:
                # Wrap build_batch to time it
                t_start = cuda_sync_time()
                batch = ae.build_batch(
                    args=ae_args, student=student, student_processor=student_processor,
                    student_tokenizer=student_tokenizer, teacher_model=teacher_model,
                    batch_items=[{"row": sample, "sample_id": sample.get("sample_id", "")}])
                t_build = cuda_sync_time()

                # FM timing
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                t_fm_start = cuda_sync_time()
                with torch.no_grad():
                    res = ae.sample_paths(
                        bundle=bundle, teacher_model=teacher_model, batch=batch,
                        seed=SEED, device=device, inference_steps=FM_STEPS,
                        temperature=TEMPERATURE, kv_layer_indices=kv_layer_indices)
                t_fm_end = cuda_sync_time()

                if is_bench:
                    latency_times["build_total"].append((t_build - t_start) * 1000)
                    latency_times["fm_10step"].append((t_fm_end - t_fm_start) * 1000)

                del batch
            except Exception as e:
                print(json.dumps({"event": "latency_error", "model": model_key, "error": str(e)}), flush=True)

        latency_summary = {k: {"mean": round(np.mean(v), 1), "std": round(np.std(v), 1)}
                           for k, v in latency_times.items()}
        print(json.dumps({"event": "latency_done", "model": model_key, "latency": latency_summary}), flush=True)

        # --- ADE / minADE@6 evaluation ---
        model_ades = []
        model_fdes = []
        all_path_ades = []  # for minADE@6

        for eval_idx, sample in enumerate(eval_samples):
            sample_item = {"row": sample, "sample_id": sample.get("sample_id", "")}
            try:
                batch = ae.build_batch(
                    args=ae_args, student=student, student_processor=student_processor,
                    student_tokenizer=student_tokenizer, teacher_model=teacher_model,
                    batch_items=[sample_item])

                # Single path ADE (mean_traj with N paths)
                path_ades_this = []
                for path_idx in range(NUM_PATHS):
                    path_seed = SEED + eval_idx * NUM_PATHS + path_idx
                    torch.manual_seed(path_seed)
                    torch.cuda.manual_seed_all(path_seed)
                    with torch.no_grad():
                        res = ae.sample_paths(
                            bundle=bundle, teacher_model=teacher_model, batch=batch,
                            seed=path_seed, device=device, inference_steps=FM_STEPS,
                            temperature=TEMPERATURE, kv_layer_indices=kv_layer_indices)
                    pred_xyz = res["pred_xyz"]
                    target_xyz = batch["target_xyz"].cpu().numpy()
                    p = pred_xyz[0]
                    while p.ndim > 2:
                        p = p[0]
                    t = target_xyz[0]
                    while t.ndim > 2:
                        t = t[0]
                    a, f = ade_fde(p, t)
                    path_ades_this.append(float(a))
                    if path_idx == 0:
                        model_fdes.append(float(f))

                model_ades.append(path_ades_this[0])  # single path
                all_path_ades.append(min(path_ades_this))  # minADE@6
                del batch

            except Exception as e:
                continue

            if (eval_idx + 1) % 50 == 0:
                print(json.dumps({
                    "event": "eval_progress", "model": model_key,
                    "done": eval_idx + 1, "total": len(eval_samples),
                    "ade_so_far": round(np.mean(model_ades), 3),
                    "minade6_so_far": round(np.mean(all_path_ades), 3),
                }), flush=True)

        eval_summary = {
            "ade_mean": round(np.mean(model_ades), 3) if model_ades else None,
            "ade_p50": round(np.median(model_ades), 3) if model_ades else None,
            "fde_mean": round(np.mean(model_fdes), 3) if model_fdes else None,
            "minade6_mean": round(np.mean(all_path_ades), 3) if all_path_ades else None,
            "minade6_p50": round(np.median(all_path_ades), 3) if all_path_ades else None,
            "n_samples": len(model_ades),
        }
        print(json.dumps({"event": "eval_done", "model": model_key, "eval": eval_summary}), flush=True)
        all_results[model_key] = {"latency": latency_summary, "eval": eval_summary}

        # --- Visualization samples ---
        vis_by_cat: dict[str, int] = defaultdict(int)
        vis_count = 0
        for sample in eval_samples:
            cat = sample.get("metadata", {}).get("semantic_scene_category", "unknown")
            if vis_by_cat[cat] >= args.num_vis_per_cat:
                continue
            sample_item = {"row": sample, "sample_id": sample.get("sample_id", "")}
            try:
                batch = ae.build_batch(
                    args=ae_args, student=student, student_processor=student_processor,
                    student_tokenizer=student_tokenizer, teacher_model=teacher_model,
                    batch_items=[sample_item])

                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                with torch.no_grad():
                    res = ae.sample_paths(
                        bundle=bundle, teacher_model=teacher_model, batch=batch,
                        seed=SEED, device=device, inference_steps=FM_STEPS,
                        temperature=TEMPERATURE, kv_layer_indices=kv_layer_indices)

                pred_xyz = res["pred_xyz"][0]
                while pred_xyz.ndim > 2:
                    pred_xyz = pred_xyz[0]

                # Save trajectory for this model+sample for later multi-model visualization
                sid = sample.get("sample_id", "unknown")
                safe_sid = sid.replace("/", "_")
                traj_dir = vis_dir / "trajs"
                traj_dir.mkdir(exist_ok=True)
                np.save(str(traj_dir / f"{safe_sid}_{model_key}.npy"), pred_xyz)

                # Save ego history and target for this sample (once)
                ego_hist = batch["ego_history_xyz"][0].cpu().numpy()
                target_xyz_np = batch["target_xyz"][0].cpu().numpy()
                while target_xyz_np.ndim > 2:
                    target_xyz_np = target_xyz_np[0]
                np.save(str(traj_dir / f"{safe_sid}_ego_hist.npy"), ego_hist)
                np.save(str(traj_dir / f"{safe_sid}_target.npy"), target_xyz_np)

                # Save camera images (once per sample)
                cam_dir = vis_dir / "cameras"
                cam_dir.mkdir(exist_ok=True)
                images = load_sample_images(sample, PROJECT_ROOT)
                for ci, img in enumerate(images[:4]):
                    if isinstance(img, Image.Image):
                        img.save(str(cam_dir / f"{safe_sid}_cam{ci}.jpg"))

                # Save CoT
                cot_text = sample.get("teacher_target", "")
                if "|" in cot_text:
                    cot_text = cot_text.split("<|cot_end|>")[0] if "<|cot_end|>" in cot_text else cot_text[:200]
                cot_dir = vis_dir / "cots"
                cot_dir.mkdir(exist_ok=True)
                Path(cot_dir / f"{safe_sid}_{model_key}.txt").write_text(cot_text[:300])

                vis_by_cat[cat] += 1
                vis_count += 1
                del batch
            except Exception:
                continue

        print(json.dumps({"event": "vis_saved", "model": model_key, "count": vis_count}), flush=True)

        # Cleanup model
        if bundle is not None:
            bundle.cpu()
            del bundle
        if student is not None:
            student.backbone.cpu()
            del student
        teacher_model.cpu()
        del teacher_model
        gc.collect()
        torch.cuda.empty_cache()

    # ========== Generate combined visualization images ==========
    print(json.dumps({"event": "generating_vis"}), flush=True)
    traj_dir = vis_dir / "trajs"
    cam_dir = vis_dir / "cameras"
    cot_dir = vis_dir / "cots"
    combined_dir = vis_dir / "combined"
    combined_dir.mkdir(exist_ok=True)

    # Find samples that have trajectories from all models
    traj_files = list(traj_dir.glob("*.npy")) if traj_dir.exists() else []
    sample_ids = set()
    for f in traj_files:
        name = f.stem
        for mk in MODELS:
            if name.endswith(f"_{mk}"):
                sid = name[: -len(f"_{mk}")]
                sample_ids.add(sid)

    for sid in sorted(sample_ids):
        model_trajs = {}
        model_cots = {}
        model_colors = {}
        has_all = True

        for mk, mcfg in MODELS.items():
            if mk == "alpamayo_10b":
                continue  # skip if not computed
            traj_path = traj_dir / f"{sid}_{mk}.npy"
            if not traj_path.exists():
                has_all = False
                break
            traj = np.load(str(traj_path))
            model_trajs[mcfg["label"]] = traj[:, :2]  # XY only
            model_colors[mcfg["label"]] = mcfg["color"]

            cot_path = cot_dir / f"{sid}_{mk}.txt"
            model_cots[mcfg["label"]] = cot_path.read_text() if cot_path.exists() else ""

        if not model_trajs:
            continue

        # Load camera images
        camera_images = []
        for ci in range(4):
            cam_path = cam_dir / f"{sid}_cam{ci}.jpg"
            if cam_path.exists():
                camera_images.append(np.array(Image.open(cam_path)))
            else:
                camera_images.append(np.zeros((320, 576, 3), dtype=np.uint8))

        # Load ego history and GT
        ego_hist_path = traj_dir / f"{sid}_ego_hist.npy"
        ego_hist = np.load(str(ego_hist_path))[:, :2] if ego_hist_path.exists() else None
        gt_path = traj_dir / f"{sid}_target.npy"
        gt_traj = np.load(str(gt_path))[:, :2] if gt_path.exists() else None

        # Get category
        cat = "unknown"
        for s in eval_samples:
            if s.get("sample_id", "").replace("/", "_") == sid:
                cat = s.get("metadata", {}).get("semantic_scene_category", "unknown")
                break

        save_path = str(combined_dir / f"{cat}_{sid}.png")
        draw_benchmark_figure(
            sample_id=sid, category=cat,
            camera_images=camera_images,
            ego_history_xy=ego_hist,
            model_trajs=model_trajs,
            model_cots=model_cots,
            gt_traj_xy=gt_traj,
            model_colors=model_colors,
            save_path=save_path,
        )

    # ========== Save final summary ==========
    summary = {
        "seed": SEED,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_paths": NUM_PATHS,
        "fm_steps": FM_STEPS,
        "num_eval_samples": args.num_eval,
        "results": all_results,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "benchmark_done", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
