#!/usr/bin/env python3
"""Generate 3-model trajectory comparison visualizations.

For each category: 4 samples x 1 figure with:
  - Row 0: 4 camera images
  - Row 1 left: BEV trajectory (ego history + GT + 3 model predictions)
  - Row 1 right: CoT text from each model
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

SUKIM_ROOT = PROJECT_ROOT.parents[1]
for p in (PROJECT_ROOT, SUKIM_ROOT, SUKIM_ROOT / "alpamayo_repo/alpamayo1.5/src", SUKIM_ROOT / "visualization"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SEED = 42
TEMP = 0.85
FM_STEPS = 10


def _import_ae():
    spec = importlib.util.spec_from_file_location(
        "ae_train", str(PROJECT_ROOT / "scripts" / "84_train_student_ae28_official.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_figure(
    *,
    sample_id: str,
    category: str,
    camera_images: list,
    ego_hist_xy: np.ndarray | None,
    gt_traj_xy: np.ndarray | None,
    model_trajs: dict[str, np.ndarray],
    model_cots: dict[str, str],
    model_ades: dict[str, float],
    colors: dict[str, str],
    save_path: str,
):
    fig = plt.figure(figsize=(26, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.25,
                           left=0.03, right=0.97, top=0.94, bottom=0.03)
    fig.suptitle(f"[{category}]  {sample_id}", fontsize=11, fontweight="bold", y=0.97)

    # Row 0: 4 cameras
    for i in range(min(4, len(camera_images))):
        ax = fig.add_subplot(gs[0, i])
        if camera_images[i] is not None:
            ax.imshow(camera_images[i])
        ax.set_title(f"Camera {i}", fontsize=9)
        ax.axis("off")

    # Row 1 left: BEV
    ax_bev = fig.add_subplot(gs[1:, :2])
    if ego_hist_xy is not None and len(ego_hist_xy) > 1:
        ax_bev.plot(ego_hist_xy[:, 1], ego_hist_xy[:, 0],
                    "k-o", markersize=3, linewidth=1.5, label="Ego History", zorder=5)
    if gt_traj_xy is not None:
        ax_bev.plot(gt_traj_xy[:, 1], gt_traj_xy[:, 0],
                    "k--", linewidth=2, alpha=0.4, label="GT", zorder=4)
        ax_bev.scatter(gt_traj_xy[-1, 1], gt_traj_xy[-1, 0],
                       color="black", s=50, marker="x", zorder=7, alpha=0.5)

    for name, traj in model_trajs.items():
        c = colors.get(name, "#888")
        ade_str = f" (ADE {model_ades.get(name, 0):.2f}m)" if name in model_ades else ""
        ax_bev.plot(traj[:, 1], traj[:, 0], "-", color=c, linewidth=2.5,
                    label=f"{name}{ade_str}", zorder=6)
        ax_bev.scatter(traj[-1, 1], traj[-1, 0], color=c, s=60, zorder=8,
                       edgecolors="k", linewidths=0.5)

    ax_bev.set_xlabel("Y (m)", fontsize=10)
    ax_bev.set_ylabel("X (m)", fontsize=10)
    ax_bev.set_title("BEV Trajectories (6.4s horizon)", fontsize=11, fontweight="bold")
    ax_bev.legend(fontsize=8, loc="best", framealpha=0.9)
    ax_bev.set_aspect("equal")
    ax_bev.grid(True, alpha=0.3)

    # Row 1 right: CoT
    ax_cot = fig.add_subplot(gs[1, 2:])
    ax_cot.axis("off")
    cot_blocks = []
    for name, cot in model_cots.items():
        c = colors.get(name, "#333")
        short = cot[:150] + "..." if len(cot) > 150 else cot
        cot_blocks.append(f"[{name}]\n{short}")
    ax_cot.text(0.02, 0.98, "\n\n".join(cot_blocks), transform=ax_cot.transAxes,
                fontsize=8, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax_cot.set_title("Chain-of-Thought", fontsize=11, fontweight="bold")

    # Row 2 right: ADE summary
    ax_info = fig.add_subplot(gs[2, 2:])
    ax_info.axis("off")
    info_lines = []
    for name in model_trajs:
        ade = model_ades.get(name, 0)
        c = colors.get(name, "#333")
        info_lines.append(f"{name}: ADE = {ade:.3f}m")
    ax_info.text(0.02, 0.8, "\n".join(info_lines), transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffc107"))

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_student_model(ae, model_cfg, sample_items, device):
    """Load a student model, run inference on samples, return trajectories + CoTs."""
    class A:
        pass
    args = A()
    for k, v in {
        "student_checkpoint_dir": Path(model_cfg["student_ckpt"]),
        "corpus_jsonl": Path("data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl"),
        "teacher_checkpoint_path": Path(SUKIM_ROOT / "base_weights/Alpamayo-1.5-10B"),
        "student_dtype": "bfloat16", "device": str(device), "student_model": "",
        "ae_init_mode": "student_backbone_init", "attn_implementation": "flash_attention_2",
        "disable_student_deepstack": False, "qat_quantization": "", "qat_calib_samples": 256,
        "num_samples": 10, "val_samples": 5, "val_fraction": 0.1,
        "split_seed": None, "split_cache_json": None, "split": "train",
        "split_scan_all": True, "compressed_layers": 28, "mapping": "linspace_round",
        "ae_dtype": "bfloat16", "prefix_mode": "student_free",
        "preserve_flex_positions": model_cfg["has_flex"],
        "flex_selection_strategy": "uniform",
        "flex_scene_deepstack": model_cfg["has_flex"],
        "target_source": "gt", "max_new_tokens": 160, "max_length": 4096,
        "stage2_attention_mode": "official_none", "seed": SEED,
        "teacher_load_device": "cpu",
    }.items():
        setattr(args, k, v)

    student, tok, proc, _ = ae.load_student(args)
    _load = getattr(ae, "load_model_and_processor", None)
    if not _load:
        from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import load_model_and_processor as _load
    teacher, _, _, _, _ = _load(
        checkpoint_path=args.teacher_checkpoint_path, dtype=torch.bfloat16,
        device="cpu", config_json=None, runtime_support=None,
        attn_implementation="flash_attention_2", min_pixels=163840, max_pixels=196608)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.to(device)

    bundle, sel = ae.build_bundle(teacher, args, student=student)
    ae.load_bundle_checkpoint(Path(model_cfg["ae_ckpt"]), bundle=bundle)
    bundle.eval()
    for p in bundle.parameters():
        p.requires_grad_(False)
    bundle.to(device)

    expert_n = int(bundle.expert.config.num_hidden_layers)
    kv_layer_indices = sel if expert_n < 28 else None

    from probe_teacher_kv_28layer_expert_compression import ade_fde
    from src.training.collator import load_sample_images

    results = {}
    for item in sample_items:
        sid = item.get("sample_id", "")
        sample_item = {"row": item, "sample_id": sid}
        try:
            batch = ae.build_batch(
                args=args, student=student, student_processor=proc,
                student_tokenizer=tok, teacher_model=teacher,
                batch_items=[sample_item])

            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            with torch.no_grad():
                res = ae.sample_paths(
                    bundle=bundle, teacher_model=teacher, batch=batch,
                    seed=SEED, device=device, inference_steps=FM_STEPS,
                    temperature=TEMP, kv_layer_indices=kv_layer_indices)

            pred_xyz = res["pred_xyz"][0]
            while pred_xyz.ndim > 2:
                pred_xyz = pred_xyz[0]

            target = batch["target_xyz"].cpu().numpy()[0]
            while target.ndim > 2:
                target = target[0]

            ego_hist = batch["ego_history_xyz"][0].cpu().numpy()
            ade_val, _ = ade_fde(pred_xyz, target)

            # Get CoT from generated text
            cot = batch.get("generated_text", "")
            if not cot and hasattr(batch, "get"):
                cot = ""

            # Get camera images (first time only, shared across models)
            images = load_sample_images(item, PROJECT_ROOT)

            results[sid] = {
                "pred_xyz": pred_xyz,
                "target_xyz": target,
                "ego_hist": ego_hist,
                "ade": float(ade_val),
                "cot": cot,
                "images": [np.array(img) for img in images[:4]],
            }
            del batch
        except Exception as e:
            print(json.dumps({"event": "error", "model": model_cfg["name"], "sid": sid,
                              "error": str(e)[:200]}), flush=True)
    # Cleanup
    bundle.cpu(); del bundle
    student.backbone.cpu(); del student
    teacher.cpu(); del teacher
    gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/benchmark_3models/visualizations")
    parser.add_argument("--num-per-cat", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ae = _import_ae()

    # Select samples: num_per_cat per category
    corpus = [json.loads(l) for l in open(
        "data/corpus/no_nav_teacher_pair_full444k_semantic_balanced_20k_train_val9007_seed42.jsonl")]
    by_cat = defaultdict(list)
    for it in corpus:
        cat = it.get("metadata", {}).get("semantic_scene_category", "unknown")
        by_cat[cat].append(it)

    rng = random.Random(SEED)
    vis_samples = []
    for cat in sorted(by_cat.keys()):
        pool = by_cat[cat]
        rng.shuffle(pool)
        vis_samples.extend(pool[:args.num_per_cat])

    print(json.dumps({"event": "samples_selected", "n": len(vis_samples),
                       "categories": len(by_cat)}), flush=True)

    # Model configs
    models = [
        {
            "name": "Cosmos 2B + AE28",
            "student_ckpt": "outputs/checkpoints/no_nav_camera_labeled_official_full444k/"
                            "no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250",
            "ae_ckpt": "outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603/best.pt",
            "has_flex": False,
            "color": "#3498db",
        },
        {
            "name": "Cosmos 2B + FLEX + AE28",
            "student_ckpt": "outputs/checkpoints/mlflex_k512_bp3_cont3e_lr1e5_b16_fast_20260609_after_k1024/final",
            "ae_ckpt": "outputs/action_expert/flex_k512_6ep_ae_18k_s10k_b16/best.pt",
            "has_flex": True,
            "color": "#2ecc71",
        },
    ]

    colors = {m["name"]: m["color"] for m in models}
    colors["GT"] = "#000000"

    # Run each model
    all_model_results: dict[str, dict] = {}
    for mcfg in models:
        print(json.dumps({"event": "model_start", "model": mcfg["name"]}), flush=True)
        results = run_student_model(ae, mcfg, vis_samples, device)
        all_model_results[mcfg["name"]] = results
        print(json.dumps({"event": "model_done", "model": mcfg["name"],
                          "n_success": len(results)}), flush=True)

    # Generate figures
    print(json.dumps({"event": "generating_figures"}), flush=True)
    fig_count = 0
    for item in vis_samples:
        sid = item.get("sample_id", "")
        cat = item.get("metadata", {}).get("semantic_scene_category", "unknown")
        safe_sid = sid.replace("/", "_")

        # Collect data from all models
        model_trajs = {}
        model_cots = {}
        model_ades = {}
        camera_images = None
        ego_hist = None
        gt_traj = None

        has_all = True
        for mcfg in models:
            mname = mcfg["name"]
            if sid not in all_model_results.get(mname, {}):
                has_all = False
                break
            r = all_model_results[mname][sid]
            model_trajs[mname] = r["pred_xyz"][:, :3]  # XYZ → take XY(Z)
            model_cots[mname] = r["cot"]
            model_ades[mname] = r["ade"]
            if camera_images is None:
                camera_images = r["images"]
                ego_hist = r["ego_hist"]
                gt_traj = r["target_xyz"]

        if not has_all or camera_images is None:
            continue

        cat_dir = output_dir / cat
        cat_dir.mkdir(exist_ok=True)
        save_path = str(cat_dir / f"{safe_sid}.png")

        draw_figure(
            sample_id=sid, category=cat,
            camera_images=camera_images,
            ego_hist_xy=ego_hist[:, :3] if ego_hist is not None else None,
            gt_traj_xy=gt_traj[:, :3] if gt_traj is not None else None,
            model_trajs=model_trajs,
            model_cots=model_cots,
            model_ades=model_ades,
            colors=colors,
            save_path=save_path,
        )
        fig_count += 1

    print(json.dumps({"event": "done", "figures": fig_count,
                       "output_dir": str(output_dir)}), flush=True)


if __name__ == "__main__":
    main()
