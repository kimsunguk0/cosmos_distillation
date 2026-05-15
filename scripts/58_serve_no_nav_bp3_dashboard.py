#!/usr/bin/env python3
"""Serve a lightweight live dashboard for no-nav BP3 training/eval runs."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "no_nav_bp3_200k_epoch_gc_b16_eval64_from_bp3final_20260508_072958"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Checkpoint output directory. If omitted, the server searches outputs/checkpoints for RUN_ID.",
    )
    parser.add_argument("--max-points", type=int, default=600)
    parser.add_argument("--log-lines", type=int, default=120)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def tail_lines(path: Path, count: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            block = 8192
            data = b""
            pos = size
            while pos > 0 and data.count(b"\n") <= count:
                read_size = min(block, pos)
                pos -= read_size
                handle.seek(pos)
                data = handle.read(read_size) + data
        return data.decode("utf-8", errors="replace").splitlines()[-count:]
    except Exception as exc:  # noqa: BLE001
        return [f"<tail failed: {exc!r}>"]


def run_command(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=5)
        return result.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc!r}"


def process_status(run_id: str) -> dict[str, Any]:
    output = run_command(["ps", "-eo", "pid,pgid,stat,etime,cmd"])
    matches = []
    for line in output.splitlines():
        if run_id in line and "58_serve_no_nav_bp3_dashboard.py" not in line:
            matches.append(line.strip())
    return {
        "alive": bool(matches),
        "matches": matches,
    }


def gpu_status() -> list[dict[str, Any]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "memory_used_mib": to_float(parts[1]),
                "memory_total_mib": to_float(parts[2]),
                "utilization_gpu_pct": to_float(parts[3]),
                "power_w": to_float(parts[4]),
                "temperature_c": to_float(parts[5]),
            }
        )
    return rows


def to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(f):
        return None
    return f


def downsample(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    stride = max(1, int(math.ceil(len(rows) / max_points)))
    return rows[::stride]


def bucket_metric_points(points: list[dict[str, Any]], bucket_size: int = 50) -> list[dict[str, Any]]:
    if bucket_size <= 1:
        return points
    numeric_keys = (
        "total_loss",
        "cot_loss",
        "traj_loss",
        "text_kd",
        "traj_kd",
        "cot_acc",
        "traj_acc",
        "format_loss",
        "grad_norm",
        "traj_hidden_align",
        "boundary_hidden_align",
        "hidden_relation",
        "hidden_temporal",
        "hidden_spectrum",
        "scheduled_sampling_rate",
    )
    buckets: dict[int, list[dict[str, Any]]] = {}
    for point in points:
        step = int(point.get("step") or 0)
        if step <= 0:
            continue
        bucket_end = int(math.ceil(step / bucket_size) * bucket_size)
        buckets.setdefault(bucket_end, []).append(point)

    out: list[dict[str, Any]] = []
    for bucket_end in sorted(buckets):
        rows = buckets[bucket_end]
        last = rows[-1]
        item: dict[str, Any] = {
            "step": bucket_end,
            "step_start": rows[0].get("step"),
            "step_end": last.get("step"),
            "count": len(rows),
            "timestamp": last.get("timestamp"),
            "epoch": last.get("epoch"),
            "phase": last.get("phase"),
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
            item[key] = (sum(values) / len(values)) if values else None
        out.append(item)
    return out


def moving_average(values: list[float], window: int = 25) -> list[float | None]:
    out: list[float | None] = []
    buf: list[float] = []
    for value in values:
        if value is not None and math.isfinite(float(value)):
            buf.append(float(value))
        if len(buf) > window:
            buf.pop(0)
        out.append(sum(buf) / len(buf) if buf else None)
    return out


def parse_train_start(log_lines: list[str]) -> dict[str, Any]:
    text = "\n".join(log_lines)
    start_line = next((line for line in log_lines if "[train-start]" in line), "")
    values: dict[str, Any] = {}
    for key in ("train_records", "val_records", "steps_per_epoch", "max_steps", "eval_every_steps", "save_every_steps", "batch_per_gpu", "effective_batch"):
        match = re.search(rf"{key}=([0-9.]+)", start_line)
        if match:
            values[key] = int(float(match.group(1)))
    values["gradient_checkpointing"] = "gradient checkpointing" in text or "use_cache=True is incompatible" in text
    return values


def metric_point(row: dict[str, Any]) -> dict[str, Any]:
    logs = row.get("logs") or {}
    return {
        "step": int(row.get("global_step") or 0),
        "timestamp": float(row.get("timestamp") or 0.0),
        "epoch": to_float(row.get("epoch_progress")),
        "total_loss": to_float(logs.get("total_loss")),
        "cot_loss": to_float(logs.get("gt_cot_loss")),
        "traj_loss": to_float(logs.get("traj_loss") or logs.get("gt_traj_loss")),
        "text_kd": to_float(logs.get("teacher_topk_kd_loss")),
        "traj_kd": to_float(logs.get("teacher_traj_topk_kd_loss")),
        "cot_acc": to_float(logs.get("gt_cot_token_acc")),
        "traj_acc": to_float(logs.get("traj_token_acc")),
        "format_loss": to_float(logs.get("output_format_loss")),
        "grad_norm": to_float(logs.get("grad_norm")),
        "traj_hidden_align": to_float(
            logs.get("teacher_traj_hidden_align_loss", logs.get("teacher_traj_hidden_align"))
        ),
        "boundary_hidden_align": to_float(
            logs.get("teacher_text_boundary_hidden_align_loss", logs.get("teacher_boundary_hidden_align"))
        ),
        "hidden_relation": to_float(logs.get("teacher_traj_hidden_relation")),
        "hidden_temporal": to_float(logs.get("teacher_traj_hidden_temporal")),
        "hidden_spectrum": to_float(logs.get("teacher_traj_hidden_latent_spectrum")),
        "scheduled_sampling_rate": to_float(logs.get("scheduled_sampling_rate")),
        "scheduled_sampling_replaced": to_float(logs.get("scheduled_sampling_replaced")),
        "scheduled_sampling_candidates": to_float(logs.get("scheduled_sampling_candidates")),
        "phase": row.get("phase"),
    }


def val_point(row: dict[str, Any]) -> dict[str, Any]:
    logs = row.get("logs") or {}
    decode = row.get("decode_eval") or {}
    return {
        "step": int(row.get("global_step") or 0),
        "epoch": to_float(row.get("epoch_progress")),
        "total_loss": to_float(logs.get("total_loss")),
        "cot_loss": to_float(logs.get("gt_cot_loss")),
        "traj_loss": to_float(logs.get("traj_loss") or logs.get("gt_traj_loss")),
        "traj_hidden_align": to_float(
            logs.get("teacher_traj_hidden_align_loss", logs.get("teacher_traj_hidden_align"))
        ),
        "boundary_hidden_align": to_float(
            logs.get("teacher_text_boundary_hidden_align_loss", logs.get("teacher_boundary_hidden_align"))
        ),
        "cot_acc": to_float(logs.get("gt_cot_token_acc")),
        "traj_acc": to_float(logs.get("traj_token_acc")),
        "decode_enabled": bool(decode.get("enabled")),
        "decode_samples": decode.get("num_samples"),
        "free_run_ade": to_float(decode.get("avg_ade_m") or decode.get("avg_student_vs_teacher_discrete_ade_m")),
        "free_run_fde": to_float(decode.get("avg_fde_m") or decode.get("avg_student_vs_teacher_discrete_fde_m")),
        "avg_unique": to_float(decode.get("avg_unique_traj_ids")),
        "max_run": to_float(decode.get("avg_max_same_token_run")),
        "bad_geometry_rate": to_float(decode.get("bad_geometry_rate")),
        "token_count_match_rate": to_float(decode.get("token_count_match_rate")),
        "anti_collapse_score": to_float(decode.get("anti_collapse_score")),
        "free_run_geometry_score": to_float(decode.get("free_run_geometry_score")),
    }


def estimate_eta(train_points: list[dict[str, Any]], max_steps: int | None) -> dict[str, Any]:
    if not train_points or not max_steps:
        return {"eta_sec": None, "steps_per_hour": None, "step_sec": None}
    recent = train_points[-min(80, len(train_points)) :]
    first = recent[0]
    last = recent[-1]
    delta_step = int(last["step"]) - int(first["step"])
    delta_time = float(last["timestamp"]) - float(first["timestamp"])
    if delta_step <= 0 or delta_time <= 0:
        return {"eta_sec": None, "steps_per_hour": None, "step_sec": None}
    step_sec = delta_time / delta_step
    remaining = max(int(max_steps) - int(last["step"]), 0)
    return {
        "eta_sec": remaining * step_sec,
        "steps_per_hour": 3600.0 / step_sec,
        "step_sec": step_sec,
    }


def collect_checkpoint_status(output_dir: Path, report_suite_dir: Path) -> list[dict[str, Any]]:
    preferred = ["step_003125", "step_006250", "step_009375", "step_012500", "final"]
    discovered = []
    if output_dir.exists():
        discovered = sorted(
            item.name
            for item in output_dir.iterdir()
            if item.is_dir() and (item.name.startswith("step_") or item.name in {"final", "best_decode"})
        )
    names = preferred if any((output_dir / name).exists() for name in preferred) else discovered
    rows = []
    for name in names:
        checkpoint_dir = output_dir / name
        val_summary = report_suite_dir / f"{name}_val204_decode_summary.json"
        train_summary = report_suite_dir / f"{name}_train64_decode_summary.json"
        prefill_md = ROOT_DIR / "outputs" / "reports" / "no_nav_prefill_qc" / f"{report_suite_dir.name.replace('_checkpoint_suite', '')}_{name}_prefill_qc.md"
        rows.append(
            {
                "name": name,
                "checkpoint_exists": checkpoint_dir.exists(),
                "val_decode_exists": val_summary.exists(),
                "train_decode_exists": train_summary.exists(),
                "prefill_qc_exists": prefill_md.exists(),
                "val_summary": read_json(val_summary),
                "train_summary": read_json(train_summary),
                "prefill_md": str(prefill_md),
            }
        )
    return rows


def nested_get(row: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def collect_hidden_qc_live(root: Path, run_id: str, output_dir: Path) -> dict[str, Any]:
    names = ["step_003125", "step_006250", "step_009375", "step_012500", "final"]
    report_dir = root / "outputs" / "reports" / "no_nav_prefill_qc_live" / run_id
    status_path = report_dir / "live_status.json"
    status = read_json(status_path) or {}
    checkpoints = status.get("checkpoints") if isinstance(status, dict) else {}
    rows: list[dict[str, Any]] = []
    first_report = None
    first_report_source: Path | None = None
    for name in names:
        item = checkpoints.get(name, {}) if isinstance(checkpoints, dict) else {}
        candidate = Path(item.get("report_json") or report_dir / f"{run_id}_{name}_prefill_qc.json")
        first_report = read_json(candidate)
        if isinstance(first_report, dict):
            first_report_source = candidate
            break
    if not isinstance(first_report, dict):
        for candidate in (
            root / "outputs" / "reports" / "no_nav_prefill_qc" / "val128_prefill_base_bp3_trajonly_qc.json",
            root / "outputs" / "reports" / "no_nav_prefill_qc" / "val16_prefill_base_bp3_trajonly_qc.json",
        ):
            first_report = read_json(candidate)
            if isinstance(first_report, dict):
                first_report_source = candidate
                break

    if isinstance(first_report, dict):
        cka_to_teacher = first_report.get("cka_to_teacher") or {}
        students = first_report.get("students") or {}
        base_key = "base_2b" if "base_2b" in cka_to_teacher else None
        bp3_key = (
            "bp3_init"
            if "bp3_init" in cka_to_teacher
            else ("bp5_init" if "bp5_init" in cka_to_teacher else ("bp3_cot_traj_2b" if "bp3_cot_traj_2b" in cka_to_teacher else None))
        )
        base_cka_for_delta = nested_get(cka_to_teacher, (base_key, "prefill_last_linear_cka_to_teacher")) if base_key else None
        bp3_cka_for_delta = nested_get(cka_to_teacher, (bp3_key, "prefill_last_linear_cka_to_teacher")) if bp3_key else None
        baseline_specs = [
            (base_key, "baseline: base_2B"),
            (bp3_key, "baseline: run init"),
        ]
        for baseline_name, label in baseline_specs:
            if baseline_name is None:
                continue
            summary = nested_get(students, (baseline_name, "prefill_last")) or {}
            baseline_cka = nested_get(cka_to_teacher, (baseline_name, "prefill_last_linear_cka_to_teacher"))
            delta_base = None
            delta_bp3 = None
            if baseline_cka is not None and base_cka_for_delta is not None:
                delta_base = float(baseline_cka) - float(base_cka_for_delta)
            if baseline_cka is not None and bp3_cka_for_delta is not None:
                delta_bp3 = float(baseline_cka) - float(bp3_cka_for_delta)
            rows.append(
                {
                    "name": label,
                    "kind": "baseline",
                    "checkpoint_exists": True,
                    "status": "baseline",
                    "reason": f"baseline source: {first_report_source.name if first_report_source else 'qc report'}",
                    "report_exists": True,
                    "report_json": str(first_report_source) if first_report_source else None,
                    "log": None,
                    "updated_at": None,
                    "cka": baseline_cka,
                    "base_cka": base_cka_for_delta,
                    "bp3_init_cka": bp3_cka_for_delta,
                    "delta_vs_base": delta_base,
                    "delta_vs_bp3_init": delta_bp3,
                    "norm_mean": summary.get("norm_mean"),
                    "offdiag_cos_mean": summary.get("offdiag_cos_mean"),
                    "top_pc_var_ratio": summary.get("top_pc_var_ratio"),
                    "effective_rank": summary.get("effective_rank"),
                    "common_ratio": summary.get("common_ratio"),
                    "samples": first_report.get("num_samples"),
                }
            )

    for name in names:
        item = checkpoints.get(name, {}) if isinstance(checkpoints, dict) else {}
        report_json = Path(item.get("report_json") or report_dir / f"{run_id}_{name}_prefill_qc.json")
        log_path = Path(item.get("log") or report_dir / f"{name}.prefill_qc.log")
        report = read_json(report_json) or {}
        cka_to_teacher = report.get("cka_to_teacher") if isinstance(report, dict) else {}
        students = report.get("students") if isinstance(report, dict) else {}
        checkpoint_summary = nested_get(students, (name, "prefill_last")) or {}
        checkpoint_cka = nested_get(cka_to_teacher, (name, "prefill_last_linear_cka_to_teacher"))
        base_cka = nested_get(cka_to_teacher, ("base_2b", "prefill_last_linear_cka_to_teacher"))
        bp3_cka = nested_get(cka_to_teacher, ("bp3_init", "prefill_last_linear_cka_to_teacher"))
        if bp3_cka is None:
            bp3_cka = nested_get(cka_to_teacher, ("bp5_init", "prefill_last_linear_cka_to_teacher"))
        delta_base = None
        delta_bp3 = None
        if checkpoint_cka is not None and base_cka is not None:
            delta_base = float(checkpoint_cka) - float(base_cka)
        if checkpoint_cka is not None and bp3_cka is not None:
            delta_bp3 = float(checkpoint_cka) - float(bp3_cka)
        rows.append(
            {
                "name": name,
                "checkpoint_exists": (output_dir / name).exists(),
                "status": item.get("status") or ("done" if report_json.exists() else "pending"),
                "reason": item.get("reason"),
                "report_exists": report_json.exists(),
                "report_json": str(report_json),
                "log": str(log_path),
                "updated_at": item.get("finished_at") or item.get("started_at") or item.get("updated_at"),
                "cka": checkpoint_cka,
                "base_cka": base_cka,
                "bp3_init_cka": bp3_cka,
                "delta_vs_base": delta_base,
                "delta_vs_bp3_init": delta_bp3,
                "norm_mean": checkpoint_summary.get("norm_mean"),
                "offdiag_cos_mean": checkpoint_summary.get("offdiag_cos_mean"),
                "top_pc_var_ratio": checkpoint_summary.get("top_pc_var_ratio"),
                "effective_rank": checkpoint_summary.get("effective_rank"),
                "common_ratio": checkpoint_summary.get("common_ratio"),
                "samples": report.get("num_samples"),
            }
        )
    return {
        "status": status,
        "report_dir": str(report_dir),
        "status_path": str(status_path),
        "rows": rows,
    }


def build_status(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root_dir
    run_id = args.run_id
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = root / "outputs" / "checkpoints" / "no_nav_bp3_200k_epoch" / run_id
        if not output_dir.exists():
            candidates = sorted((root / "outputs" / "checkpoints").glob(f"*/{run_id}"))
            if candidates:
                output_dir = candidates[0]
    report_dir = root / "outputs" / "reports" / "no_nav_distill"
    train_log = root / "logs" / "no_nav_distill" / f"{run_id}.train.log"
    eval_log = root / "logs" / "no_nav_distill" / f"{run_id}.eval.log"
    launcher_log = root / "logs" / "no_nav_distill" / f"{run_id}.launcher.log"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = report_dir / f"{run_id}_summary.json"
    report_suite_dir = report_dir / f"{run_id}_checkpoint_suite"

    rows = read_jsonl(metrics_path)
    train_rows = [row for row in rows if row.get("phase") == "train"]
    val_rows = [row for row in rows if row.get("phase") == "val"]
    train_points_full = [metric_point(row) for row in train_rows]
    val_points = [val_point(row) for row in val_rows]
    train_tail_lines = tail_lines(train_log, args.log_lines)
    train_start = parse_train_start(tail_lines(train_log, 400))
    max_steps = int(train_start.get("max_steps") or 0) or None
    latest = train_points_full[-1] if train_points_full else None
    eta = estimate_eta(train_points_full, max_steps)

    chart_bucket_size = 50
    chart_points_full = bucket_metric_points(train_points_full, chart_bucket_size)
    loss_values = [point.get("total_loss") for point in chart_points_full]
    traj_acc_values = [point.get("traj_acc") for point in chart_points_full]
    cot_acc_values = [point.get("cot_acc") for point in chart_points_full]
    smoothed = {
        "total_loss_ma": moving_average([v if v is not None else float("nan") for v in loss_values], window=3),
        "traj_acc_ma": moving_average([v if v is not None else float("nan") for v in traj_acc_values], window=3),
        "cot_acc_ma": moving_average([v if v is not None else float("nan") for v in cot_acc_values], window=3),
    }
    compact_points = downsample(chart_points_full, args.max_points)
    if len(compact_points) == len(chart_points_full):
        compact_smoothed = smoothed
    else:
        keep_steps = {point["step"] for point in compact_points}
        compact_smoothed = {key: [value for point, value in zip(chart_points_full, values, strict=False) if point["step"] in keep_steps] for key, values in smoothed.items()}

    return {
        "now": time.time(),
        "run_id": run_id,
        "paths": {
            "output_dir": str(output_dir),
            "metrics": str(metrics_path),
            "train_log": str(train_log),
            "eval_log": str(eval_log),
            "summary": str(summary_path),
            "report_suite_dir": str(report_suite_dir),
        },
        "process": process_status(run_id),
        "gpu": gpu_status(),
        "train_start": train_start,
        "latest_train": latest,
        "progress": {
            "step": latest.get("step") if latest else 0,
            "max_steps": max_steps,
            "percent": (100.0 * float(latest.get("step")) / float(max_steps)) if latest and max_steps else None,
            **eta,
        },
        "train_points": compact_points,
        "train_smoothed": compact_smoothed,
        "chart_bucket_size": chart_bucket_size,
        "val_points": val_points,
        "checkpoints": collect_checkpoint_status(output_dir, report_suite_dir),
        "hidden_qc": collect_hidden_qc_live(root, run_id, output_dir),
        "summary": read_json(summary_path),
        "logs": {
            "launcher": tail_lines(launcher_log, 40),
            "train": train_tail_lines,
            "eval": tail_lines(eval_log, args.log_lines),
        },
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>No-Nav Backbone Dashboard</title>
  <style>
    :root { color-scheme: light; --bg:#f6f7f9; --panel:#ffffff; --ink:#15181d; --muted:#67707c; --line:#dce1e7; --ok:#16784b; --warn:#a65f00; --bad:#ba1a1a; --accent:#2457c5; }
    * { box-sizing: border-box; }
    html { -webkit-text-size-adjust:100%; }
    body { margin:0; min-width:320px; overflow-x:hidden; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--ink); }
    header { padding:18px 24px; background:#10141b; color:white; display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; }
    header > div { min-width:0; }
    h1 { margin:0; font-size:20px; font-weight:650; letter-spacing:0; }
    .sub { color:#b9c2cf; font-size:13px; margin-top:4px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
    main { width:100%; max-width:1800px; margin:0 auto; padding:18px 24px 28px; display:grid; gap:16px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap:12px; }
    .panel { min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
    .metric .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
    .metric .value { margin-top:6px; font-size:clamp(18px, 1.7vw, 24px); font-weight:700; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
    .metric .hint { margin-top:5px; color:var(--muted); font-size:12px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
    .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
    .badge { display:inline-flex; align-items:center; height:24px; border-radius:999px; padding:0 10px; font-size:12px; font-weight:600; background:#eef2ff; color:#173b8f; }
    .badge.ok { background:#e9f7ef; color:var(--ok); }
    .badge.bad { background:#ffecec; color:var(--bad); }
    .bar { height:12px; background:#e9edf3; border-radius:999px; overflow:hidden; margin-top:12px; }
    .bar > div { height:100%; width:0%; background:linear-gradient(90deg,#2b6df3,#16a36f); transition:width .25s ease; }
    h2 { font-size:15px; margin:0 0 10px; }
    canvas { width:100%; height:clamp(180px, 22vh, 230px); display:block; }
    .chart-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 340px), 1fr)); gap:18px 16px; }
    .chart-block { min-width:0; }
    .chart-title { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0 0 6px; }
    .chart-title strong { font-size:13px; }
    .chart-title span { color:var(--muted); font-size:12px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace; white-space:nowrap; }
    .legend { display:flex; flex-wrap:wrap; gap:8px 12px; margin:4px 0 0; color:var(--muted); font-size:12px; }
    .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
    .table-scroll { width:100%; min-width:0; overflow-x:auto; -webkit-overflow-scrolling:touch; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    #latestTable { min-width:300px; }
    #valTable, #ckptTable { min-width:860px; }
    #hiddenQcTable { min-width:1040px; }
    th, td { padding:8px 9px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
    th:first-child, td:first-child { text-align:left; }
    th { color:var(--muted); font-weight:650; background:#fafbfc; position:sticky; top:0; }
    pre { margin:0; padding:12px; overflow:auto; max-height:420px; background:#10141b; color:#d9e2ef; border-radius:8px; font-size:12px; line-height:1.45; }
    .two { display:grid; grid-template-columns:minmax(0, 1.7fr) minmax(280px, .7fr); gap:12px; align-items:start; }
    .paths { font-size:12px; color:var(--muted); font-family:ui-monospace, SFMono-Regular, Menlo, monospace; overflow-wrap:anywhere; }
    @media (max-width: 1350px) { .two { grid-template-columns:1fr; } }
    @media (max-width: 780px) { header, main { padding-left:14px; padding-right:14px; } .chart-grid { grid-template-columns:1fr; } th, td { padding:7px 8px; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>No-Nav Backbone Live Dashboard</h1>
      <div class="sub" id="runId">loading...</div>
    </div>
    <div class="row">
      <span class="badge" id="updated">...</span>
      <span class="badge" id="alive">...</span>
    </div>
  </header>
  <main>
    <section class="panel">
      <div class="row" style="justify-content:space-between">
        <div>
          <h2>Run Progress</h2>
          <div class="paths" id="paths"></div>
        </div>
        <div class="paths" id="eta"></div>
      </div>
      <div class="bar"><div id="progressBar"></div></div>
    </section>

    <section class="grid">
      <div class="panel metric"><div class="label">Step</div><div class="value" id="step">-</div><div class="hint" id="epoch">-</div></div>
      <div class="panel metric"><div class="label">Total Loss</div><div class="value" id="loss">-</div><div class="hint">latest train</div></div>
      <div class="panel metric"><div class="label">TF Traj Acc</div><div class="value" id="trajAcc">-</div><div class="hint">next token, teacher-forced</div></div>
      <div class="panel metric"><div class="label">TF CoT Acc</div><div class="value" id="cotAcc">-</div><div class="hint">next token, teacher-forced</div></div>
      <div class="panel metric"><div class="label">GPU Memory</div><div class="value" id="gpuMem">-</div><div class="hint" id="gpuUtil">-</div></div>
      <div class="panel metric"><div class="label">Speed</div><div class="value" id="speed">-</div><div class="hint">steps/hour</div></div>
      <div class="panel metric"><div class="label">Next Eval</div><div class="value" id="nextEval">-</div><div class="hint">every eval interval</div></div>
      <div class="panel metric"><div class="label">Checkpoints</div><div class="value" id="ckptCount">-</div><div class="hint">expected 4 + final</div></div>
    </section>

    <section class="two">
      <div class="panel">
        <div class="row" style="justify-content:space-between; margin-bottom:4px">
          <h2>Training Curves</h2>
          <div class="paths" id="chartMode">50-step mean, MA(3 bins), recent window = last 20 bins</div>
        </div>
        <div class="chart-grid">
          <div class="chart-block">
            <div class="chart-title"><strong>Loss</strong><span id="lossChartRange">-</span></div>
            <canvas id="lossChart" width="760" height="260"></canvas>
            <div class="legend" id="lossLegend"></div>
          </div>
          <div class="chart-block">
            <div class="chart-title"><strong>Token Accuracy</strong><span id="accChartRange">-</span></div>
            <canvas id="accChart" width="760" height="260"></canvas>
            <div class="legend" id="accLegend"></div>
          </div>
          <div class="chart-block">
            <div class="chart-title"><strong>KD / Aux Loss</strong><span id="kdChartRange">-</span></div>
            <canvas id="kdChart" width="760" height="260"></canvas>
            <div class="legend" id="kdLegend"></div>
          </div>
          <div class="chart-block">
            <div class="chart-title"><strong>Hidden Align</strong><span id="hiddenChartRange">-</span></div>
            <canvas id="hiddenChart" width="760" height="260"></canvas>
            <div class="legend" id="hiddenLegend"></div>
          </div>
          <div class="chart-block">
            <div class="chart-title"><strong>Recent Loss Focus</strong><span id="recentChartRange">-</span></div>
            <canvas id="recentChart" width="760" height="260"></canvas>
            <div class="legend" id="recentLegend"></div>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2>Latest Raw Metrics</h2>
        <div class="table-scroll"><table id="latestTable"></table></div>
      </div>
    </section>

    <section class="panel">
      <h2>Validation / Free-Run Eval</h2>
      <div class="table-scroll"><table id="valTable"></table></div>
    </section>

    <section class="panel">
      <div class="row" style="justify-content:space-between; margin-bottom:8px">
        <h2>Hidden / Interface QC</h2>
        <div class="paths" id="hiddenQcStatus">waiting for watcher...</div>
      </div>
      <div class="table-scroll"><table id="hiddenQcTable"></table></div>
    </section>

    <section class="panel">
      <h2>Checkpoint Suite</h2>
      <div class="table-scroll"><table id="ckptTable"></table></div>
    </section>

    <section class="panel">
      <h2>Recent Train Log</h2>
      <pre id="trainLog">loading...</pre>
    </section>
  </main>

<script>
const fmt = (v, d=4) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "n/a" : Number(v).toFixed(d);
const fmtPct = v => (v === null || v === undefined) ? "n/a" : (Number(v) * 100).toFixed(1) + "%";
const fmtSec = sec => {
  if (sec === null || sec === undefined || !Number.isFinite(Number(sec))) return "n/a";
  sec = Math.max(0, Number(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return `${h}h ${m}m`;
};
function setText(id, text) { document.getElementById(id).textContent = text; }
function table(id, headers, rows) {
  const html = `<thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead><tbody>` +
    rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join("")}</tr>`).join("") + "</tbody>";
  document.getElementById(id).innerHTML = html;
}
function setLegend(id, series) {
  document.getElementById(id).innerHTML = series.map(s=>`<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
}
function drawLineChart(canvasId, points, series, opts={}) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
  const cssWidth = Math.max(260, Math.floor(rect.width || 760));
  const cssHeight = Math.max(170, Math.floor(rect.height || 220));
  const backingWidth = Math.floor(cssWidth * dpr);
  const backingHeight = Math.floor(cssHeight * dpr);
  if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
    canvas.width = backingWidth;
    canvas.height = backingHeight;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0,0,cssWidth,cssHeight);
  const pad = {l:56,r:16,t:12,b:32};
  const w = cssWidth - pad.l - pad.r, h = cssHeight - pad.t - pad.b;
  if (!points.length) {
    ctx.fillStyle="#67707c"; ctx.font="13px ui-monospace, Menlo, monospace";
    ctx.fillText("waiting for data", pad.l, pad.t + 28);
    return;
  }
  const xs = points.map(p=>p.step);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const vals = series.flatMap(s=>s.values).filter(v=>Number.isFinite(Number(v))).map(Number);
  if (!vals.length) return;
  let minY = opts.yMin ?? Math.min(...vals), maxY = opts.yMax ?? Math.max(...vals);
  if (opts.percent) { minY = Math.max(0, minY); maxY = Math.min(1, Math.max(maxY, 0.02)); }
  const span = Math.max(maxY - minY, 1e-6);
  if (opts.yMin === undefined) minY -= span * 0.08;
  if (opts.yMax === undefined) maxY += span * 0.08;
  const xMap = x => pad.l + ((x - minX) / Math.max(maxX - minX, 1)) * w;
  const yMap = v => pad.t + (1 - ((Number(v) - minY) / Math.max(maxY - minY, 1e-6))) * h;
  ctx.strokeStyle="#dce1e7"; ctx.lineWidth=1;
  for (let i=0;i<=4;i++) {
    const y=pad.t+i*h/4;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+w,y); ctx.stroke();
    const tick = maxY - (maxY-minY)*i/4;
    ctx.fillStyle="#67707c"; ctx.font="11px ui-monospace, Menlo, monospace";
    ctx.fillText(opts.percent ? `${(tick*100).toFixed(0)}%` : tick.toFixed(opts.digits ?? 2), 4, y+4);
  }
  ctx.fillStyle="#67707c"; ctx.font="12px ui-monospace, Menlo, monospace";
  ctx.fillText(String(minX), pad.l, cssHeight-10);
  ctx.fillText(String(maxX), Math.max(pad.l, pad.l+w-50), cssHeight-10);
  series.forEach(s => {
    ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath();
    let started=false;
    s.values.forEach((v,i)=>{
      if (!Number.isFinite(Number(v))) return;
      const x=xMap(points[i].step), y=yMap(v);
      if (!started) { ctx.moveTo(x,y); started=true; } else ctx.lineTo(x,y);
    });
    ctx.stroke();
    const lastIdx = [...s.values].map(Number).findLastIndex(Number.isFinite);
    if (lastIdx >= 0) {
      ctx.fillStyle=s.color;
      ctx.beginPath(); ctx.arc(xMap(points[lastIdx].step), yMap(s.values[lastIdx]), 3, 0, Math.PI*2); ctx.fill();
    }
  });
  if (opts.rangeId) setText(opts.rangeId, `${minX} → ${maxX}`);
}
function drawCharts(points, smooth) {
  const ma = (key, fallbackKey) => points.map((p,i)=>smooth[key]?.[i] ?? p[fallbackKey]);
  const recent = points.slice(-20);
  const offset = points.length - recent.length;
  const recentSmooth = {
    total_loss_ma: (smooth.total_loss_ma || []).slice(offset),
    traj_acc_ma: (smooth.traj_acc_ma || []).slice(offset),
    cot_acc_ma: (smooth.cot_acc_ma || []).slice(offset),
  };
  const lossSeries = [
    {name:"total", color:"#2457c5", values:ma("total_loss_ma","total_loss")},
    {name:"traj", color:"#7c3aed", values:points.map(p=>p.traj_loss)},
    {name:"cot", color:"#a65f00", values:points.map(p=>p.cot_loss)},
  ];
  const accSeries = [
    {name:"TF traj acc", color:"#16834a", values:ma("traj_acc_ma","traj_acc")},
    {name:"TF cot acc", color:"#d97706", values:ma("cot_acc_ma","cot_acc")},
  ];
  const kdSeries = [
    {name:"traj KD", color:"#0f766e", values:points.map(p=>p.traj_kd)},
    {name:"text KD", color:"#2563eb", values:points.map(p=>p.text_kd)},
    {name:"format", color:"#be123c", values:points.map(p=>p.format_loss)},
  ];
  const hiddenSeries = [
    {name:"traj hidden", color:"#7c3aed", values:points.map(p=>p.traj_hidden_align)},
    {name:"boundary hidden", color:"#ea580c", values:points.map(p=>p.boundary_hidden_align)},
    {name:"relation", color:"#0f766e", values:points.map(p=>p.hidden_relation)},
    {name:"temporal", color:"#2563eb", values:points.map(p=>p.hidden_temporal)},
  ];
  const recentSeries = [
    {name:"total", color:"#2457c5", values:recent.map((p,i)=>recentSmooth.total_loss_ma?.[i] ?? p.total_loss)},
    {name:"traj", color:"#7c3aed", values:recent.map(p=>p.traj_loss)},
    {name:"cot", color:"#a65f00", values:recent.map(p=>p.cot_loss)},
  ];
  setLegend("lossLegend", lossSeries); setLegend("accLegend", accSeries);
  setLegend("kdLegend", kdSeries); setLegend("hiddenLegend", hiddenSeries); setLegend("recentLegend", recentSeries);
  drawLineChart("lossChart", points, lossSeries, {rangeId:"lossChartRange", digits:2});
  drawLineChart("accChart", points, accSeries, {rangeId:"accChartRange", percent:true, yMin:0, yMax:1});
  drawLineChart("kdChart", points, kdSeries, {rangeId:"kdChartRange", digits:3});
  drawLineChart("hiddenChart", points, hiddenSeries, {rangeId:"hiddenChartRange", digits:3});
  drawLineChart("recentChart", recent, recentSeries, {rangeId:"recentChartRange", digits:2});
}
let lastTrainPoints = [];
let lastTrainSmooth = {};
let resizeTimer = null;
window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => drawCharts(lastTrainPoints, lastTrainSmooth), 120);
});
async function refresh() {
  const res = await fetch("/api/status", {cache:"no-store"});
  const data = await res.json();
  setText("runId", data.run_id);
  setText("updated", "updated " + new Date(data.now*1000).toLocaleTimeString());
  const alive = document.getElementById("alive");
  alive.textContent = data.process.alive ? "training alive" : "not running";
  alive.className = "badge " + (data.process.alive ? "ok" : "bad");
  const p = data.progress || {}, latest = data.latest_train || {};
  const pct = p.percent ?? 0;
  document.getElementById("progressBar").style.width = Math.max(0, Math.min(100, pct)) + "%";
  setText("step", `${p.step || 0}/${p.max_steps || "?"}`);
  setText("epoch", `${fmt((latest.step || 0) / (data.train_start.steps_per_epoch || 1), 4)} epoch · ${fmt(p.percent,2)}%`);
  setText("loss", fmt(latest.total_loss,4));
  setText("trajAcc", fmtPct(latest.traj_acc));
  setText("cotAcc", fmtPct(latest.cot_acc));
  const gpu = data.gpu?.[0] || {};
  setText("gpuMem", gpu.memory_used_mib ? `${(gpu.memory_used_mib/1024).toFixed(1)}GB` : "n/a");
  setText("gpuUtil", `${fmt(gpu.utilization_gpu_pct,0)}% util · ${fmt(gpu.power_w,0)}W · ${fmt(gpu.temperature_c,0)}C`);
  setText("speed", p.steps_per_hour ? fmt(p.steps_per_hour,0) : "n/a");
  const interval = data.train_start.eval_every_steps || 0;
  const nextEval = interval ? Math.ceil((p.step || 0) / interval) * interval : null;
  setText("nextEval", nextEval ? String(nextEval === p.step ? p.step + interval : nextEval) : "n/a");
  setText("ckptCount", String((data.checkpoints || []).filter(c=>c.checkpoint_exists).length));
  setText("eta", `ETA ${fmtSec(p.eta_sec)} · step ${fmt(p.step_sec,2)}s · ${fmt(p.steps_per_hour,0)} steps/h`);
  setText("paths", data.paths.metrics);
  setText("chartMode", `${data.chart_bucket_size || 50}-step mean, MA(3 bins), recent window = last 20 bins`);
  lastTrainPoints = data.train_points || [];
  lastTrainSmooth = data.train_smoothed || {};
  drawCharts(lastTrainPoints, lastTrainSmooth);
  table("latestTable", ["metric","value"], [
    ["traj loss", fmt(latest.traj_loss,4)], ["cot loss", fmt(latest.cot_loss,4)],
    ["text KD", fmt(latest.text_kd,4)], ["traj KD", fmt(latest.traj_kd,4)],
    ["traj hidden", fmt(latest.traj_hidden_align,4)], ["boundary hidden", fmt(latest.boundary_hidden_align,4)],
    ["hidden relation", fmt(latest.hidden_relation,4)], ["hidden temporal", fmt(latest.hidden_temporal,4)],
    ["SS rate", fmtPct(latest.scheduled_sampling_rate)], ["SS replaced", `${fmt(latest.scheduled_sampling_replaced,0)} / ${fmt(latest.scheduled_sampling_candidates,0)}`],
    ["format loss", fmt(latest.format_loss,4)], ["grad norm", fmt(latest.grad_norm,4)]
  ]);
  const vals = data.val_points || [];
  table("valTable", ["step","val loss","TF cot acc","TF traj acc","traj hidden","boundary hidden","FR ADE","FR FDE","unique","max run","bad geom","score"],
    vals.length ? vals.map(v=>[v.step,fmt(v.total_loss,4),fmtPct(v.cot_acc),fmtPct(v.traj_acc),fmt(v.traj_hidden_align,4),fmt(v.boundary_hidden_align,4),fmt(v.free_run_ade,3),fmt(v.free_run_fde,3),fmt(v.avg_unique,2),fmt(v.max_run,2),fmtPct(v.bad_geometry_rate),fmt(v.free_run_geometry_score,3)]) : [["pending","-","-","-","-","-","-","-","-","-","-","-"]]);
  const hq = data.hidden_qc || {};
  const hqStatus = hq.status || {};
  const updated = hqStatus.updated_at ? new Date(hqStatus.updated_at * 1000).toLocaleTimeString() : "not started";
  const freeVram = hqStatus.free_vram_gb === undefined ? "n/a" : fmt(hqStatus.free_vram_gb,1) + "GB";
  setText("hiddenQcStatus", `samples ${hqStatus.num_samples || "?"} · free VRAM ${freeVram} · updated ${updated} · ${hq.report_dir || ""}`);
  table("hiddenQcTable", ["row","ckpt","QC","CKA→teacher","Δbase","Δinit/BP3","eff rank","top PC","offdiag cos","common","reason"],
    (hq.rows || []).map(r=>[
      r.name,
      r.checkpoint_exists ? "yes" : "-",
      r.status || "-",
      fmt(r.cka,4),
      fmt(r.delta_vs_base,4),
      fmt(r.delta_vs_bp3_init,4),
      fmt(r.effective_rank,2),
      fmt(r.top_pc_var_ratio,4),
      fmt(r.offdiag_cos_mean,4),
      fmt(r.common_ratio,4),
      r.reason || (r.report_exists ? "done" : "-")
    ]));
  table("ckptTable", ["checkpoint","ckpt","val decode","train decode","prefill QC","val ADE/FDE"],
    (data.checkpoints || []).map(c=>[c.name,c.checkpoint_exists?"yes":"-",c.val_decode_exists?"yes":"-",c.train_decode_exists?"yes":"-",c.prefill_qc_exists?"yes":"-",c.val_summary ? `${fmt(c.val_summary.avg_ade_m,3)} / ${fmt(c.val_summary.avg_fde_m,3)}` : "-"]));
  setText("trainLog", (data.logs.train || []).join("\n"));
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *values: Any) -> None:
            return

        def send_json(self, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/status":
                self.send_json(build_status(args))
                return
            if path in ("/", "/index.html"):
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving no-nav BP3 dashboard on http://{args.host}:{args.port} run_id={args.run_id}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
