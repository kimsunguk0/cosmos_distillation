#!/usr/bin/env python3
"""
Dashboard generator for step_006250 checkpoint eval.

Shows per-sample: 4-camera image grid + SVG trajectory overlay
(history / GT / teacher / student) + metrics + CoT texts.

Usage:
    .venv/bin/python3 scripts/render_step006250_dashboard.py
"""
from __future__ import annotations

import base64
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

DISTILL_ROOT = Path("/home/pm97/workspace/sukim/distillation/cosmos_distillation")
if str(DISTILL_ROOT) not in sys.path:
    sys.path.insert(0, str(DISTILL_ROOT))

from src.inference.checkpoint_eval import (  # noqa: E402
    TrajectoryTokenDecoder,
    load_ego_history_rot,
    resolve_traj_tokenizer_config_path,
)
from src.training.collator import load_ego_history_xyz  # noqa: E402

CORPUS_JSONL = DISTILL_ROOT / "data/corpus/vis_4per_category_val.jsonl"
SUMMARY_JSON = (
    DISTILL_ROOT
    / "outputs/reports/no_nav_distill"
    / "full_free_run_eval_step006250_20260527_batched"
    / "step_006250_val_full_4760_b16_summary.json"
)
OUTPUT_DIR = Path("/home/pm97/workspace/sukim/visualization/step006250_teacher_dashboard")

# ── trajectory colours ────────────────────────────────────────────────────────
C_HISTORY  = "#111111"
C_GT       = "#1b9e77"   # green
C_TEACHER  = "#f59e0b"   # amber
C_STUDENT  = "#e31a1c"   # red


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_tokens(value: Any) -> list[int]:
    if value is None:
        return []
    return [int(t) for t in value]


def load_gt_tokens(corpus_row: dict[str, Any]) -> list[int]:
    ht = corpus_row.get("hard_target") or {}
    # prefer inline list; fall back to npy
    inline = ht.get("traj_future_token_ids")
    if inline:
        return [int(t) for t in inline]
    npy_path = ht.get("traj_future_token_ids_path")
    if npy_path and Path(npy_path).exists():
        return np.load(npy_path).astype(int).tolist()
    return []


def img_to_b64(path: Path) -> str:
    with path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _polyline(pts: np.ndarray, *, xmin: float, ymin: float, scale: float,
              plot_h: float, margin: float) -> str:
    out: list[str] = []
    for x, y in pts[:, :2]:
        px = margin + (float(x) - xmin) * scale
        py = margin + plot_h - (float(y) - ymin) * scale
        out.append(f"{px:.1f},{py:.1f}")
    return " ".join(out)


def render_svg(
    *,
    title: str,
    history: np.ndarray,
    gt: np.ndarray,
    teacher: np.ndarray,
    student: np.ndarray,
    ade_m: float,
    fde_m: float,
    teacher_ade_m: float,
    teacher_fde_m: float,
    student_cot: str,
    teacher_cot: str,
) -> str:
    def nonempty(arr: np.ndarray) -> np.ndarray:
        return arr if arr.size else np.zeros((0, 2))

    arrays = [a[:, :2] for a in (history, gt, teacher, student) if a.size > 0]
    if not arrays:
        return ""
    all_xy = np.concatenate(arrays, axis=0)
    xmin, ymin = all_xy.min(axis=0) - 5.0
    xmax, ymax = all_xy.max(axis=0) + 5.0

    W, H = 900.0, 560.0
    margin = 28.0
    plot_w, plot_h = 480.0, 480.0
    text_x = 530
    scale = min(plot_w / max(float(xmax - xmin), 1e-3),
                plot_h / max(float(ymax - ymin), 1e-3))

    def line(arr: np.ndarray, color: str, label: str, dash: str = "") -> str:
        if arr.size == 0:
            return ""
        pts = _polyline(arr, xmin=float(xmin), ymin=float(ymin),
                        scale=scale, plot_h=plot_h, margin=margin)
        style = f"stroke='{color}' stroke-width='2.5' fill='none'"
        if dash:
            style += f" stroke-dasharray='{dash}'"
        return f"<polyline {style} points='{pts}'><title>{html.escape(label)}</title></polyline>"

    # text block
    def wrap(text: str, max_chars: int = 44) -> list[str]:
        words = (text or "").split()
        lines, cur = [], ""
        for w in words[:100]:
            cand = f"{cur} {w}".strip()
            if len(cand) > max_chars:
                if cur:
                    lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return lines

    text_parts: list[str] = []
    ty = 36

    def add_label(heading: str, value: str, color: str = "#374151") -> None:
        nonlocal ty
        text_parts.append(
            f"<text x='{text_x}' y='{ty}' font-size='12' font-family='monospace' "
            f"fill='#6b7280'>{html.escape(heading)}</text>"
        )
        ty += 15
        for wl in wrap(value):
            text_parts.append(
                f"<text x='{text_x}' y='{ty}' font-size='11' font-family='monospace' "
                f"fill='{color}'>{html.escape(wl)}</text>"
            )
            ty += 14
        ty += 6

    add_label("student vs teacher:", f"ADE {ade_m:.3f}m  FDE {fde_m:.3f}m", "#e31a1c")
    add_label("teacher vs GT:", f"ADE {teacher_ade_m:.3f}m  FDE {teacher_fde_m:.3f}m", "#f59e0b")
    add_label("student CoT:", student_cot)
    add_label("teacher CoT:", teacher_cot)

    # legend
    legend_items = [
        (C_HISTORY, "history",  ""),
        (C_GT,      "GT",       ""),
        (C_TEACHER, "teacher",  "6,3"),
        (C_STUDENT, "student",  ""),
    ]
    lx, ly = 30, int(plot_h + margin * 2 + 10)
    legend_parts: list[str] = []
    for color, label, dash in legend_items:
        style = f"stroke='{color}' stroke-width='2.5' fill='none'"
        if dash:
            style += f" stroke-dasharray='{dash}'"
        legend_parts.append(f"<line x1='{lx}' y1='{ly}' x2='{lx+30}' y2='{ly}' {style}/>")
        legend_parts.append(
            f"<text x='{lx+36}' y='{ly+4}' font-size='11' font-family='monospace' "
            f"fill='{color}'>{html.escape(label)}</text>"
        )
        lx += 110

    layers = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(W)}' height='{int(H)}'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='white'/>",
        f"<text x='16' y='18' font-size='13' font-family='monospace' fill='#111'>"
        f"{html.escape(title)}</text>",
        line(history,  C_HISTORY, "history"),
        line(gt,       C_GT,      "GT future"),
        line(teacher,  C_TEACHER, "teacher future", "6,3"),
        line(student,  C_STUDENT, "student future"),
        *legend_parts,
        *text_parts,
        "</svg>",
    ]
    return "\n".join(l for l in layers if l)


def make_camera_grid_b64(mat_path: Path) -> str:
    """Return base64 PNG of a 2x2 grid of cam0..cam3 latest frames, or empty string."""
    try:
        from PIL import Image
    except ImportError:
        return ""
    imgs_dir = mat_path / "images"
    tiles: list[Image.Image] = []
    for cam in range(4):
        p = imgs_dir / f"cam{cam}_f3.png"
        if not p.exists():
            p = imgs_dir / f"cam{cam}_f2.png"
        if p.exists():
            with Image.open(p) as im:
                tiles.append(im.convert("RGB").resize((480, 270)))
        else:
            tiles.append(Image.new("RGB", (480, 270), (220, 220, 220)))
    grid = Image.new("RGB", (960, 540))
    for i, tile in enumerate(tiles):
        grid.paste(tile, ((i % 2) * 480, (i // 2) * 270))
    import io
    buf = io.BytesIO()
    grid.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def ade_fde(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    n = min(pred.shape[0], ref.shape[0])
    if n <= 0:
        return float("nan"), float("nan")
    d = np.linalg.norm(pred[:n, :2] - ref[:n, :2], axis=-1)
    return float(d.mean()), float(d[-1])


def t0_from_id(sample_id: str) -> str:
    m = re.search(r"t0_(\d+)", sample_id)
    if m:
        return f"{int(m.group(1)) / 1e6:.1f}s"
    return "?"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] corpus …")
    corpus = {row["sample_id"]: row for row in load_jsonl(CORPUS_JSONL)}

    print("[load] eval summary …")
    summary_data = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    summary_by_id = {str(s["sample_id"]): s for s in summary_data.get("samples", [])}

    traj_cfg = resolve_traj_tokenizer_config_path(None)
    if traj_cfg is None:
        raise SystemExit("Cannot resolve traj tokenizer config")
    decoder = TrajectoryTokenDecoder(config_path=traj_cfg)

    sample_cards: list[str] = []
    stats_ade: list[float] = []
    stats_fde: list[float] = []
    n_missing = 0

    sample_ids = list(corpus.keys())
    print(f"[render] {len(sample_ids)} samples …")

    for idx, sample_id in enumerate(sample_ids, 1):
        corpus_row = corpus[sample_id]
        summary_row = summary_by_id.get(sample_id)
        if summary_row is None:
            print(f"  [skip] {sample_id} not in summary")
            n_missing += 1
            continue

        mat_path = Path(
            str((corpus_row.get("input") or {}).get("materialized_sample_path") or "")
        )

        history = load_ego_history_xyz(corpus_row, DISTILL_ROOT)
        history_rot = load_ego_history_rot(corpus_row, DISTILL_ROOT)

        gt_tokens    = load_gt_tokens(corpus_row)
        teacher_tokens = as_tokens(summary_row.get("target_traj_tokens"))
        student_tokens = as_tokens(summary_row.get("generated_traj_tokens"))

        gt_xyz      = decoder.decode(history, history_rot, gt_tokens)      if gt_tokens      else np.zeros((0, 3))
        teacher_xyz = decoder.decode(history, history_rot, teacher_tokens) if teacher_tokens else np.zeros((0, 3))
        student_xyz = decoder.decode(history, history_rot, student_tokens) if student_tokens else np.zeros((0, 3))

        # ADE/FDE: student vs teacher (geometry_reference=teacher), teacher vs GT
        s_ade, s_fde = ade_fde(student_xyz, teacher_xyz)
        t_ade, t_fde = ade_fde(teacher_xyz, gt_xyz)

        if math.isfinite(s_ade):
            stats_ade.append(s_ade)
            stats_fde.append(s_fde)

        t0_str  = t0_from_id(sample_id)
        title   = f"{idx}/{len(sample_ids)}  {sample_id[:36]}  t={t0_str}"
        svg_str = render_svg(
            title=title,
            history=history,
            gt=gt_xyz,
            teacher=teacher_xyz,
            student=student_xyz,
            ade_m=float(summary_row.get("ade_m") or s_ade),
            fde_m=float(summary_row.get("fde_m") or s_fde),
            teacher_ade_m=t_ade,
            teacher_fde_m=t_fde,
            student_cot=str(summary_row.get("student_cot") or ""),
            teacher_cot=str(summary_row.get("teacher_cot") or ""),
        )
        svg_b64 = base64.b64encode(svg_str.encode()).decode()

        cam_b64 = make_camera_grid_b64(mat_path)

        failure_tags = summary_row.get("failure_tags") or []
        tags_html = " ".join(
            f'<span class="tag">{html.escape(t)}</span>' for t in failure_tags
        ) if failure_tags else '<span class="tag ok">ok</span>'

        ade_val = summary_row.get("ade_m")
        fde_val = summary_row.get("fde_m")
        early_ade = summary_row.get("early_ade_2s_m")
        late_ade  = summary_row.get("late_ade_after_2s_m")
        uniq      = summary_row.get("generated_unique_token_count")
        max_run   = summary_row.get("generated_max_same_token_run")

        def fmt(v: Any, decimals: int = 3) -> str:
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return "—"
            return f"{float(v):.{decimals}f}"

        cam_section = (
            f'<img class="camera" src="data:image/png;base64,{cam_b64}" alt="cameras">'
            if cam_b64
            else '<div class="no-cam">no cameras</div>'
        )

        card = f"""
<section class="card" id="{html.escape(sample_id)}">
  <div class="card-head">
    <div>
      <h2>{html.escape(sample_id)}</h2>
      <p>t = {html.escape(t0_str)} &nbsp;|&nbsp; {tags_html}</p>
    </div>
    <div class="metric-badge">
      <span>ADE</span><b>{fmt(ade_val)}&nbsp;m</b>
      <span>FDE</span><b>{fmt(fde_val)}&nbsp;m</b>
    </div>
  </div>
  <div class="visual-row">
    {cam_section}
    <img class="traj" src="data:image/svg+xml;base64,{svg_b64}" alt="trajectory">
  </div>
  <div class="metric-row">
    <div><span>student vs teacher ADE</span><b>{fmt(ade_val)} m</b></div>
    <div><span>student vs teacher FDE</span><b>{fmt(fde_val)} m</b></div>
    <div><span>teacher vs GT ADE</span><b>{fmt(t_ade)} m</b></div>
    <div><span>teacher vs GT FDE</span><b>{fmt(t_fde)} m</b></div>
    <div><span>early ADE (&lt;2s)</span><b>{fmt(early_ade)} m</b></div>
    <div><span>late ADE (&gt;2s)</span><b>{fmt(late_ade)} m</b></div>
    <div><span>unique tokens</span><b>{uniq or "—"}</b></div>
    <div><span>max token run</span><b>{max_run or "—"}</b></div>
  </div>
</section>"""
        sample_cards.append(card)
        print(f"  [{idx}/{len(sample_ids)}] {sample_id[:48]}  ADE={fmt(ade_val)}")

    mean_ade = sum(stats_ade) / len(stats_ade) if stats_ade else float("nan")
    mean_fde = sum(stats_fde) / len(stats_fde) if stats_fde else float("nan")

    html_page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>step_006250 – Teacher Dashboard</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    body {{ margin: 0; background: #f7f8fb; color: #111827; }}
    header {{ position: sticky; top: 0; z-index: 2; background: rgba(255,255,255,.95);
              border-bottom: 1px solid #e5e7eb; padding: 14px 24px; backdrop-filter: blur(8px); }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    p  {{ margin: 0; color: #4b5563; font-size: 13px; }}
    .summary {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }}
    .summary div {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
                    padding: 8px 14px; min-width: 150px; }}
    .summary span {{ display: block; color: #6b7280; font-size: 11px; margin-bottom: 2px; }}
    .summary b {{ color: #111827; font-size: 14px; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 16px 24px 48px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
             padding: 14px; margin-top: 18px; }}
    .card-head {{ display: flex; justify-content: space-between; align-items: flex-start;
                  gap: 12px; margin-bottom: 10px; }}
    h2 {{ margin: 0 0 4px; font-size: 13px; font-family: monospace; color: #111; }}
    .card-head p {{ font-size: 12px; }}
    .metric-badge {{ text-align: right; white-space: nowrap; }}
    .metric-badge span {{ display: block; font-size: 11px; color: #6b7280; }}
    .metric-badge b {{ font-size: 15px; color: #111; }}
    .visual-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }}
    .camera, .traj {{ width: 100%; height: auto; border: 1px solid #e5e7eb;
                       border-radius: 6px; display: block; }}
    .no-cam {{ width: 100%; height: 200px; background: #f3f4f6; border: 1px solid #e5e7eb;
               border-radius: 6px; display: flex; align-items: center; justify-content: center;
               color: #9ca3af; font-size: 13px; }}
    .metric-row {{ display: grid; grid-template-columns: repeat(8, 1fr); gap: 8px;
                   margin-top: 10px; }}
    .metric-row div {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px;
                       padding: 6px 10px; }}
    .metric-row span {{ display: block; color: #6b7280; font-size: 11px; margin-bottom: 2px; }}
    .metric-row b {{ font-size: 13px; color: #111; }}
    .tag {{ display: inline-block; background: #fef3c7; color: #92400e;
             border-radius: 4px; padding: 1px 6px; font-size: 11px; margin-right: 3px; }}
    .tag.ok {{ background: #d1fae5; color: #065f46; }}
    @media (max-width: 900px) {{
      .visual-row {{ grid-template-columns: 1fr; }}
      .metric-row {{ grid-template-columns: repeat(4, 1fr); }}
      .summary {{ flex-direction: column; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>step_006250 &mdash; Free-Run Evaluation with Teacher Trajectory</h1>
  <p>Checkpoint: no_nav_official_full444k_semantic200k_hidden_gc_b16_w4_final_20260526_051838/step_006250
  &nbsp;|&nbsp; corpus: vis_4per_category_val ({len(sample_cards)} samples)</p>
  <div class="summary">
    <div><span>samples rendered</span><b>{len(sample_cards)}</b></div>
    <div><span>mean student vs teacher ADE</span><b>{mean_ade:.4f} m</b></div>
    <div><span>mean student vs teacher FDE</span><b>{mean_fde:.4f} m</b></div>
    <div><span>colours</span><b style="color:#1b9e77">■</b> GT &nbsp;
         <b style="color:#f59e0b">- -</b> teacher &nbsp;
         <b style="color:#e31a1c">■</b> student</div>
  </div>
</header>
<main>
{"".join(sample_cards)}
</main>
</body>
</html>"""

    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text(html_page, encoding="utf-8")
    print(f"\n[done] {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  rendered: {len(sample_cards)}  missing: {n_missing}")
    print(f"  mean ADE={mean_ade:.4f}m  FDE={mean_fde:.4f}m")


if __name__ == "__main__":
    main()
