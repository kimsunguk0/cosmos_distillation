#!/usr/bin/env python3
"""Render category-wise trajectory visualizations from benchmark_4models outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = PROJECT_ROOT / "outputs/benchmarks/semantic_val806_4models_20260612"
DEFAULT_VIS = PROJECT_ROOT / "data/corpus/benchmark_semantic_vis4_seed42.jsonl"

MODEL_ORDER = (
    "teacher10b",
    "student_noflex_ae28",
    "student_flex_ae28",
    "student_flex_ae14",
)
MODEL_LABELS = {
    "teacher10b": "10B",
    "student_noflex_ae28": "2B+AE28",
    "student_flex_ae28": "FLEX+AE28",
    "student_flex_ae14": "FLEX+AE14",
}
MODEL_COLORS = {
    "teacher10b": "#d62728",
    "student_noflex_ae28": "#1f77b4",
    "student_flex_ae28": "#2ca02c",
    "student_flex_ae14": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--vis-jsonl", type=Path, default=DEFAULT_VIS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-all", action="store_true", default=True)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_")


def category(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("semantic_scene_category") or "unknown")


def squeeze_path(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr[:, :3]


def load_rows(benchmark_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for model_key in MODEL_ORDER:
        path = benchmark_dir / model_key / "rows.jsonl"
        if not path.exists():
            continue
        per_model: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(path):
            per_model[str(row["sample_id"])] = row
        out[model_key] = per_model
    return out


def load_prediction(benchmark_dir: Path, model_key: str, sample_id: str) -> np.ndarray | None:
    path = benchmark_dir / "predictions" / model_key / f"{safe_id(sample_id)}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return squeeze_path(data["selected_path"])


def load_target(benchmark_dir: Path, sample_id: str) -> np.ndarray | None:
    for model_key in MODEL_ORDER:
        path = benchmark_dir / "predictions" / model_key / f"{safe_id(sample_id)}.npz"
        if path.exists():
            with np.load(path) as data:
                return squeeze_path(data["target_gt"])
    return None


def load_camera_images(row: dict[str, Any]) -> list[Image.Image]:
    sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path")))
    images = []
    for cam in range(4):
        path = sample_dir / "images" / f"cam{cam}_f3.png"
        images.append(Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (512, 288), "black"))
    return images


def load_history(row: dict[str, Any]) -> np.ndarray | None:
    sample_dir = Path(str((row.get("input") or {}).get("materialized_sample_path")))
    path = sample_dir / "ego/ego_history_xyz.npy"
    if not path.exists():
        return None
    return squeeze_path(np.load(path))


def cot_preview(metric_row: dict[str, Any] | None) -> str:
    if not metric_row:
        return ""
    text = str(metric_row.get("cot_preview") or "")
    if not text:
        text = str(metric_row.get("generated_text") or "").split("<|cot_end|>", 1)[0]
    if not text and metric_row.get("cot_texts"):
        texts = metric_row.get("cot_texts") or []
        text = str(texts[0]) if texts else ""
    return " ".join(text.replace("<|cot_start|>", "").split())[:180]


def plot_path(ax: plt.Axes, path: np.ndarray, *, label: str, color: str, linestyle: str = "-") -> None:
    path = squeeze_path(path)
    ax.plot(path[:, 1], path[:, 0], linestyle, color=color, lw=2.4, label=label)
    ax.scatter(path[-1, 1], path[-1, 0], s=42, color=color, edgecolor="black", linewidth=0.5, zorder=5)


def render_one(
    *,
    row: dict[str, Any],
    benchmark_dir: Path,
    metrics: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any] | None:
    sample_id = str(row["sample_id"])
    cat = category(row)
    preds = {model: load_prediction(benchmark_dir, model, sample_id) for model in MODEL_ORDER}
    missing = [model for model, pred in preds.items() if pred is None]
    if missing:
        return {"sample_id": sample_id, "category": cat, "status": "missing", "missing": missing}
    target = load_target(benchmark_dir, sample_id)
    history = load_history(row)
    if target is None:
        return {"sample_id": sample_id, "category": cat, "status": "missing_target"}

    images = load_camera_images(row)
    fig = plt.figure(figsize=(24, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.35, 0.85], hspace=0.22, wspace=0.08)
    fig.suptitle(f"{cat} | {sample_id}", fontsize=12, fontweight="bold")

    for i, image in enumerate(images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(image)
        ax.set_title(f"cam{i} latest", fontsize=9)
        ax.axis("off")

    ax_path = fig.add_subplot(gs[1:, :2])
    ax_path.axhline(0, color="#dddddd", lw=1)
    ax_path.axvline(0, color="#dddddd", lw=1)
    if history is not None:
        plot_path(ax_path, history, label="history", color="#6b7280", linestyle="-")
    plot_path(ax_path, target, label="GT", color="#111111", linestyle="--")
    for model_key in MODEL_ORDER:
        metric = metrics.get(model_key, {}).get(sample_id, {})
        label = MODEL_LABELS[model_key]
        if metric:
            label += f" ADE {float(metric.get('ade_gt_m', 0.0)):.2f} / mA6 {float(metric.get('minade6_gt_m', 0.0)):.2f}"
        plot_path(ax_path, preds[model_key], label=label, color=MODEL_COLORS[model_key])
    ax_path.scatter([0], [0], c="#16a34a", s=55, zorder=6, label="ego")
    ax_path.set_xlabel("lateral y (m), +left")
    ax_path.set_ylabel("forward x (m)")
    ax_path.set_title("Ego-frame trajectories, 6.4s horizon")
    ax_path.grid(True, alpha=0.3)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.legend(fontsize=8, loc="best")

    ax_metrics = fig.add_subplot(gs[1, 2:])
    ax_metrics.axis("off")
    lines = ["model                 ADE_GT  minADE6_GT  ADE_10B  minADE6_10B"]
    for model_key in MODEL_ORDER:
        metric = metrics.get(model_key, {}).get(sample_id, {})
        lines.append(
            f"{MODEL_LABELS[model_key]:<20} "
            f"{float(metric.get('ade_gt_m', float('nan'))):>6.3f}  "
            f"{float(metric.get('minade6_gt_m', float('nan'))):>10.3f}  "
            f"{float(metric.get('ade_10b_m', float('nan'))):>7.3f}  "
            f"{float(metric.get('minade6_10b_m', float('nan'))):>11.3f}"
        )
    ax_metrics.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    ax_metrics.set_title("Metrics")

    ax_cot = fig.add_subplot(gs[2, 2:])
    ax_cot.axis("off")
    cot_lines = []
    for model_key in MODEL_ORDER:
        metric = metrics.get(model_key, {}).get(sample_id, {})
        cot_lines.append(f"[{MODEL_LABELS[model_key]}] {cot_preview(metric)}")
    ax_cot.text(0.02, 0.98, "\n\n".join(cot_lines), va="top", ha="left", family="monospace", fontsize=8)
    ax_cot.set_title("Generated CoT Preview")

    out_cat = output_dir / cat
    out_cat.mkdir(parents=True, exist_ok=True)
    out_path = out_cat / f"{safe_id(sample_id)}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"sample_id": sample_id, "category": cat, "status": "ok", "path": str(out_path)}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.benchmark_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_rows(args.benchmark_dir)
    rows = list(iter_jsonl(args.vis_jsonl))
    manifest = []
    for row in rows:
        result = render_one(row=row, benchmark_dir=args.benchmark_dir, metrics=metrics, output_dir=output_dir)
        if result is not None:
            manifest.append(result)
            print(json.dumps({"event": "render", **result}, ensure_ascii=False), flush=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for item in manifest if item.get("status") == "ok")
    print(json.dumps({"event": "done", "ok": ok, "total": len(manifest), "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
