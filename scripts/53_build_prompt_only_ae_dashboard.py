#!/usr/bin/env python3
"""Build an HTML dashboard for prompt-only KV action-expert probes."""

from __future__ import annotations

import argparse
import base64
import html
import importlib.util
import json
import sys
import time
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
ALPAMAYO15_SRC = WORKSPACE_ROOT / "alpamayo_repo" / "alpamayo1.5" / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(ALPAMAYO15_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO15_SRC))

from distillation.dataset_prep.scripts.batch_infer_nonhuman_no_nav import (  # noqa: E402
    build_model_inputs,
    enforce_generation_mode,
    load_materialized_sample,
    load_model_and_processor,
    torch_dtype_from_name,
)

DEFAULT_MANIFEST = Path("/home/pm97/workspace/dataset/distill_dataset/manifests/nonhuman_teacher_infer_manifest.parquet")
DEFAULT_OUTPUT_DIR = Path("/home/pm97/workspace/sukim/visualization/prompt_only_ae_dashboard")
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "base_weights" / "Alpamayo-1.5-10B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-parquet", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=163840)
    parser.add_argument("--max-pixels", type=int, default=196608)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--sample-id", action="append", default=[])
    return parser.parse_args()


def load_probe_module() -> Any:
    path = Path(__file__).resolve().with_name("52_probe_prompt_only_action_expert.py")
    spec = importlib.util.spec_from_file_location("prompt_only_probe52", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load probe module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def select_sample_dirs(manifest_parquet: Path, sample_ids: list[str], num_samples: int) -> list[dict[str, Any]]:
    columns = ["sample_id", "clip_id", "chunk_id", "sample_idx_in_clip", "sample_time_sec", "materialized_ref"]
    df = pd.read_parquet(manifest_parquet, columns=columns)
    if sample_ids:
        wanted = set(sample_ids)
        df = df[df["sample_id"].isin(wanted)]
    else:
        df = df[df["materialized_ref"].map(lambda value: Path(str(value)).exists())]
        # Spread the dashboard across clips rather than showing adjacent samples from one clip.
        df = df.drop_duplicates("clip_id").sample(n=min(num_samples, len(df)), random_state=97)
    if len(df) > num_samples:
        df = df.head(num_samples)
    rows = []
    for row in df.to_dict("records"):
        sample_dir = Path(str(row["materialized_ref"]))
        if sample_dir.exists():
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "clip_id": str(row.get("clip_id") or ""),
                    "chunk_id": int(row.get("chunk_id") or 0),
                    "sample_idx_in_clip": int(row.get("sample_idx_in_clip") or 0),
                    "sample_time_sec": float(row.get("sample_time_sec") or 0.0),
                    "sample_dir": sample_dir,
                }
            )
    if not rows:
        raise RuntimeError("No materialized samples selected for dashboard.")
    return rows


def run_full_generate_greedy(
    *,
    model: Any,
    processor: Any,
    sample: dict[str, Any],
    device: str,
    seed: int,
) -> dict[str, Any]:
    data = build_model_inputs(processor=processor, sample=sample, device=device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    started = time.perf_counter()
    with enforce_generation_mode(model, "greedy"):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=data,
            top_p=1.0,
            top_k=None,
            temperature=1.0,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    sync_cuda()
    extra_dict = extra if isinstance(extra, dict) else {}
    cot = str(np.asarray(extra_dict.get("cot", [""])).reshape(-1)[0]) if "cot" in extra_dict else ""
    meta_action = (
        str(np.asarray(extra_dict.get("meta_action", [""])).reshape(-1)[0])
        if "meta_action" in extra_dict
        else ""
    )
    return {
        "status": "ok",
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "pred_xyz": pred_xyz.detach().float().cpu().numpy().reshape(-1, 64, 3)[0],
        "pred_rot_shape": list(pred_rot.shape),
        "cot_preview": cot[:240],
        "meta_action_preview": meta_action[:240],
    }


def camera_grid_image(sample_dir: Path, out_path: Path) -> None:
    images: list[Image.Image] = []
    labels: list[str] = []
    for cam_idx in range(4):
        for frame_idx in range(4):
            path = sample_dir / "images" / f"cam{cam_idx}_f{frame_idx}.png"
            image = Image.open(path).convert("RGB")
            image.thumbnail((288, 160), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (288, 178), (245, 245, 245))
            x = (288 - image.width) // 2
            canvas.paste(image, (x, 18))
            images.append(canvas)
            labels.append(f"cam{cam_idx} f{frame_idx}")
    grid = Image.new("RGB", (288 * 4, 178 * 4), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for idx, image in enumerate(images):
        x = (idx % 4) * 288
        y = (idx // 4) * 178
        grid.paste(image, (x, y))
        draw.rectangle([x, y, x + 287, y + 177], outline=(210, 210, 210), width=1)
        draw.text((x + 8, y + 4), labels[idx], fill=(20, 20, 20))
    grid.save(out_path)


def make_path_plot(
    *,
    paths: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    colors = {
        "full_generate": "#111827",
        "prompt_only_kv": "#2563eb",
        "no_kv": "#dc2626",
    }
    labels = {
        "full_generate": "full generate",
        "prompt_only_kv": "prompt-only KV",
        "no_kv": "no KV",
    }
    fig, ax = plt.subplots(figsize=(5.3, 5.8), dpi=150)
    all_x = []
    all_y = []
    for name, xyz in paths.items():
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        lateral_right = -arr[:, 1]
        forward = arr[:, 0]
        all_x.append(lateral_right)
        all_y.append(forward)
        ax.plot(lateral_right, forward, color=colors[name], linewidth=2.2, label=labels[name])
        ax.scatter(lateral_right[-1], forward[-1], color=colors[name], s=22, zorder=4)
    ax.scatter([0], [0], color="#10b981", marker="o", s=32, zorder=5, label="ego")
    if all_x and all_y:
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        pad_x = max(4.0, float(np.ptp(x)) * 0.2)
        pad_y = max(4.0, float(np.ptp(y)) * 0.08)
        ax.set_xlim(float(x.min() - pad_x), float(x.max() + pad_x))
        ax.set_ylim(min(-2.0, float(y.min() - 1.0)), float(y.max() + pad_y))
    ax.axvline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--")
    ax.set_xlabel("lateral, right positive (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_html(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> str:
    cards = []
    for row in rows:
        prompt_vs = row.get("prompt_only_kv", {}).get("vs_full_generate") or {}
        no_kv_vs = row.get("no_kv", {}).get("vs_full_generate") or {}
        full_summary = row.get("full_generate", {}).get("summary") or {}
        prompt_summary = row.get("prompt_only_kv", {}).get("summary") or {}
        no_kv_summary = row.get("no_kv", {}).get("summary") or {}
        cards.append(
            f"""
            <section class="sample">
              <div class="sample-head">
                <div>
                  <h2>{html.escape(row["sample_id"])}</h2>
                  <p>chunk {row["chunk_id"]} · clip {html.escape(row["clip_id"])} · t={row["sample_time_sec"]:.1f}s</p>
                </div>
                <div class="metric strong">prompt-only vs full<br><b>ADE {prompt_vs.get("ade_xy_m", "n/a")}m / FDE {prompt_vs.get("fde_xy_m", "n/a")}m</b></div>
              </div>
              <div class="visual-row">
                <img class="camera" src="{rel(row["camera_grid"], output_dir)}" alt="4x4 camera grid">
                <img class="plot" src="{rel(row["path_plot"], output_dir)}" alt="trajectory path plot">
              </div>
              <div class="metric-grid">
                <div><span>full final xy</span><b>{full_summary.get("final_xy_m")}</b></div>
                <div><span>prompt-only final xy</span><b>{prompt_summary.get("final_xy_m")}</b></div>
                <div><span>no-KV final xy</span><b>{no_kv_summary.get("final_xy_m")}</b></div>
                <div><span>no-KV vs full</span><b>ADE {no_kv_vs.get("ade_xy_m", "n/a")} / FDE {no_kv_vs.get("fde_xy_m", "n/a")}</b></div>
                <div><span>prompt prefill</span><b>{row.get("prompt_only_kv", {}).get("prefill_elapsed_sec", "n/a")}s</b></div>
                <div><span>prompt-only AE</span><b>{row.get("prompt_only_kv", {}).get("elapsed_sec", "n/a")}s</b></div>
              </div>
              <p class="cot">{html.escape(str(row.get("full_generate", {}).get("cot_preview") or ""))}</p>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt-Only KV Action Expert Probe</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    body {{ margin: 0; background: #f7f8fb; color: #111827; }}
    header {{ position: sticky; top: 0; z-index: 2; background: rgba(255,255,255,.94); border-bottom: 1px solid #e5e7eb; padding: 18px 24px; backdrop-filter: blur(10px); }}
    h1 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 13px; letter-spacing: 0; }}
    p {{ margin: 0; color: #4b5563; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 18px 24px 40px; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
    .summary div, .metric-grid div, .metric {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px; }}
    .summary span, .metric-grid span {{ display: block; color: #6b7280; font-size: 12px; margin-bottom: 4px; }}
    .summary b, .metric-grid b, .metric b {{ color: #111827; font-size: 14px; }}
    .sample {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-top: 16px; }}
    .sample-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; margin-bottom: 12px; }}
    .strong {{ min-width: 230px; font-size: 12px; color: #6b7280; }}
    .visual-row {{ display: grid; grid-template-columns: minmax(360px, 1.1fr) minmax(320px, .9fr); gap: 14px; align-items: stretch; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; background: #fff; }}
    .camera {{ object-fit: contain; }}
    .plot {{ object-fit: contain; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }}
    .cot {{ margin-top: 10px; padding: 10px 12px; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; color: #374151; font-size: 13px; }}
    @media (max-width: 1000px) {{ .summary, .metric-grid, .visual-row {{ grid-template-columns: 1fr; }} .sample-head {{ flex-direction: column; }} .strong {{ min-width: 0; width: calc(100% - 24px); }} }}
  </style>
</head>
<body>
  <header>
    <h1>Prompt-Only KV Action Expert Probe</h1>
    <p>Full generate vs prompt/vision prefill KV only vs no KV. Path plot uses ego frame with right-positive lateral axis.</p>
    <div class="summary">
      <div><span>samples</span><b>{summary["num_samples"]}</b></div>
      <div><span>prompt-only mean ADE/FDE</span><b>{summary["prompt_only_mean_ade"]:.3f} / {summary["prompt_only_mean_fde"]:.3f} m</b></div>
      <div><span>no-KV mean ADE/FDE</span><b>{summary["no_kv_mean_ade"]:.3f} / {summary["no_kv_mean_fde"]:.3f} m</b></div>
      <div><span>mean prefill</span><b>{summary["mean_prefill_sec"]:.3f}s</b></div>
      <div><span>mean prompt-only AE</span><b>{summary["mean_prompt_ae_sec"]:.3f}s</b></div>
    </div>
  </header>
  <main>
    {"".join(cards)}
  </main>
</body>
</html>
"""


def safe_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    probe = load_probe_module()
    sample_rows = select_sample_dirs(Path(args.manifest_parquet), args.sample_id, int(args.num_samples))

    dtype = torch_dtype_from_name(args.dtype)
    model, processor, config, config_path, runtime_support_path = load_model_and_processor(
        Path(args.checkpoint_path),
        dtype=dtype,
        device=args.device,
        config_json=None,
        runtime_support=None,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    autocast_context = (
        torch.autocast("cuda", dtype=dtype)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else nullcontext()
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode(), autocast_context:
        for idx, row in enumerate(sample_rows):
            sample = load_materialized_sample(row["sample_dir"])
            item_seed = int(args.seed) + idx
            full = run_full_generate_greedy(
                model=model,
                processor=processor,
                sample=sample,
                device=args.device,
                seed=item_seed,
            )
            full_xyz = np.asarray(full["pred_xyz"], dtype=np.float32).reshape(-1, 3)
            prefill = probe.build_prompt_prefill(
                model=model,
                processor=processor,
                sample=sample,
                device=args.device,
            )
            prompt_only = probe.sample_action_expert_from_cache(
                model=model,
                prompt_cache=prefill["cache"],
                rope_deltas=prefill["rope_deltas"],
                prefix_mask=prefill["prefix_mask"],
                ego_history_xyz=prefill["ego_history_xyz"],
                ego_history_rot=prefill["ego_history_rot"],
                device=args.device,
                seed=item_seed,
            )
            prompt_xyz = np.asarray(prompt_only["pred_xyz"], dtype=np.float32).reshape(-1, 64, 3)[0]
            no_kv = probe.sample_action_expert_from_cache(
                model=model,
                prompt_cache=None,
                rope_deltas=None,
                prefix_mask=None,
                ego_history_xyz=prefill["ego_history_xyz"],
                ego_history_rot=prefill["ego_history_rot"],
                device=args.device,
                seed=item_seed,
            )
            no_kv_xyz = np.asarray(no_kv["pred_xyz"], dtype=np.float32).reshape(-1, 64, 3)[0]

            sample_asset_prefix = row["sample_id"].replace("/", "_")
            camera_path = assets_dir / f"{sample_asset_prefix}_camera_grid.jpg"
            plot_path = assets_dir / f"{sample_asset_prefix}_paths.png"
            camera_grid_image(row["sample_dir"], camera_path)
            make_path_plot(
                paths={
                    "full_generate": full_xyz,
                    "prompt_only_kv": prompt_xyz,
                    "no_kv": no_kv_xyz,
                },
                out_path=plot_path,
                title=f"{row['sample_id']} trajectories",
            )

            full_summary = probe.summarize_xyz(full_xyz)
            prompt_summary = probe.summarize_xyz(prompt_xyz)
            no_kv_summary = probe.summarize_xyz(no_kv_xyz)
            prompt_only["summary"] = prompt_summary
            prompt_only["prefill_elapsed_sec"] = prefill["elapsed_sec"]
            prompt_only["prompt_token_count"] = int(prefill["input_ids"].shape[1])
            prompt_only["vs_full_generate"] = probe.ade_fde(prompt_xyz, full_xyz)
            no_kv["summary"] = no_kv_summary
            no_kv["vs_full_generate"] = probe.ade_fde(no_kv_xyz, full_xyz)
            full["summary"] = full_summary
            rows.append(
                {
                    **{k: v for k, v in row.items() if k != "sample_dir"},
                    "sample_dir": str(row["sample_dir"]),
                    "camera_grid": camera_path,
                    "path_plot": plot_path,
                    "full_generate": {k: v for k, v in full.items() if k != "pred_xyz"},
                    "prompt_only_kv": {k: v for k, v in prompt_only.items() if k != "pred_xyz"},
                    "no_kv": {k: v for k, v in no_kv.items() if k != "pred_xyz"},
                }
            )
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "index": idx + 1,
                        "total": len(sample_rows),
                        "sample_id": row["sample_id"],
                        "prompt_only_vs_full": prompt_only["vs_full_generate"],
                        "no_kv_vs_full": no_kv["vs_full_generate"],
                    }
                ),
                flush=True,
            )

    prompt_ade = [float(row["prompt_only_kv"]["vs_full_generate"]["ade_xy_m"]) for row in rows]
    prompt_fde = [float(row["prompt_only_kv"]["vs_full_generate"]["fde_xy_m"]) for row in rows]
    no_kv_ade = [float(row["no_kv"]["vs_full_generate"]["ade_xy_m"]) for row in rows]
    no_kv_fde = [float(row["no_kv"]["vs_full_generate"]["fde_xy_m"]) for row in rows]
    prefill_sec = [float(row["prompt_only_kv"]["prefill_elapsed_sec"]) for row in rows]
    prompt_ae_sec = [float(row["prompt_only_kv"]["elapsed_sec"]) for row in rows]
    summary = {
        "num_samples": len(rows),
        "prompt_only_mean_ade": safe_mean(prompt_ade),
        "prompt_only_mean_fde": safe_mean(prompt_fde),
        "no_kv_mean_ade": safe_mean(no_kv_ade),
        "no_kv_mean_fde": safe_mean(no_kv_fde),
        "mean_prefill_sec": safe_mean(prefill_sec),
        "mean_prompt_ae_sec": safe_mean(prompt_ae_sec),
        "checkpoint_path": str(args.checkpoint_path),
        "config_path": str(config_path),
        "runtime_support_path": str(runtime_support_path) if runtime_support_path is not None else None,
        "dtype": str(dtype).replace("torch.", ""),
        "attn_implementation": args.attn_implementation,
    }
    payload = {
        "summary": summary,
        "rows": [
            {
                **row,
                "camera_grid": rel(Path(row["camera_grid"]), output_dir),
                "path_plot": rel(Path(row["path_plot"]), output_dir),
            }
            for row in rows
        ],
    }
    (output_dir / "dashboard_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(rows, summary, output_dir), encoding="utf-8")
    print(json.dumps({"event": "dashboard_done", "output": str(output_dir / "index.html"), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
