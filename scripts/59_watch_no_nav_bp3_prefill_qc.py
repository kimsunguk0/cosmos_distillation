#!/usr/bin/env python3
"""Watch a BP3 training run and launch true prefill hidden QC per checkpoint.

The training process keeps the H200 busy, so this watcher stays conservative:
it waits until a checkpoint directory is complete, verifies enough free VRAM is
available, then runs the prompt-only teacher/student hidden probe in a separate
process.  Results are written incrementally for the live dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT_DIR.parents[1]
DEFAULT_RUN_ID = "no_nav_bp3_200k_epoch_gc_b16_eval64_from_bp3final_20260508_072958"
DEFAULT_INIT_CHECKPOINT = (
    ROOT_DIR
    / "outputs"
    / "checkpoints"
    / "no_nav_bp3_h200fast_b4"
    / "no_nav_bp3_h200fast_b4_from_step2288_20260504_053208"
    / "final"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training checkpoint directory. Defaults to outputs/checkpoints/no_nav_bp3_200k_epoch/$RUN_ID.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Live QC report directory. Defaults to outputs/reports/no_nav_prefill_qc_live/$RUN_ID.",
    )
    parser.add_argument(
        "--init-checkpoint-dir",
        type=Path,
        default=DEFAULT_INIT_CHECKPOINT,
        help="BP3 warm-start checkpoint included in every QC report as bp3_init.",
    )
    parser.add_argument(
        "--init-label",
        default="bp3_init",
        help="Label for --init-checkpoint-dir inside QC reports.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        action="append",
        default=[],
        help="Additional baseline checkpoint as name=/abs/path. Can be repeated.",
    )
    parser.add_argument(
        "--checkpoint-name",
        action="append",
        default=[],
        help="Checkpoint name to watch. Can be repeated. Defaults to quarter checkpoints plus final.",
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--student-batch-size", type=int, default=4)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--settle-sec", type=float, default=20.0)
    parser.add_argument("--min-free-vram-gb", type=float, default=35.0)
    parser.add_argument("--once", action="store_true", help="Check once and exit after processing available checkpoints.")
    return parser.parse_args()


def now() -> float:
    return time.time()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def checkpoint_ready(path: Path, *, settle_sec: float) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    manifest = path / "checkpoint_manifest.json"
    train_config = path / "train_config.json"
    if not manifest.exists():
        return False, "waiting_for_manifest"
    if not train_config.exists():
        return False, "waiting_for_train_config"
    try:
        json.loads(manifest.read_text(encoding="utf-8"))
        json.loads(train_config.read_text(encoding="utf-8"))
    except Exception:
        return False, "metadata_not_readable_yet"
    newest = max((item.stat().st_mtime for item in path.rglob("*") if item.exists()), default=path.stat().st_mtime)
    age = now() - newest
    if age < settle_sec:
        return False, f"settling_{age:.1f}s"
    return True, "ready"


def free_vram_gb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    best: float | None = None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            used = float(parts[0])
            total = float(parts[1])
        except ValueError:
            continue
        free = (total - used) / 1024.0
        best = free if best is None else max(best, free)
    return best


def initial_status(args: argparse.Namespace, checkpoint_names: list[str], output_dir: Path, report_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "no_nav_bp3_prefill_qc_live_status_v1",
        "run_id": args.run_id,
        "pid": os.getpid(),
        "started_at": now(),
        "updated_at": now(),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "num_samples": int(args.num_samples),
        "split": args.split,
        "student_batch_size": int(args.student_batch_size),
        "min_free_vram_gb": float(args.min_free_vram_gb),
        "checkpoints": {
            name: {
                "status": "pending",
                "checkpoint_dir": str(output_dir / name),
                "report_json": str(report_dir / f"{args.run_id}_{name}_prefill_qc.json"),
                "report_md": str(report_dir / f"{args.run_id}_{name}_prefill_qc.md"),
                "log": str(report_dir / f"{name}.prefill_qc.log"),
            }
            for name in checkpoint_names
        },
    }


def run_probe(args: argparse.Namespace, *, checkpoint_name: str, checkpoint_dir: Path, report_dir: Path) -> int:
    report_name = f"{args.run_id}_{checkpoint_name}_prefill_qc.json"
    markdown_name = f"{args.run_id}_{checkpoint_name}_prefill_qc.md"
    log_path = report_dir / f"{checkpoint_name}.prefill_qc.log"
    cmd = [
        str(args.root_dir / ".venv" / "bin" / "python"),
        str(args.root_dir / "scripts" / "55_probe_no_nav_prefill_hidden_qc.py"),
        "--split",
        args.split,
        "--num-samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--student-batch-size",
        str(args.student_batch_size),
        "--output-dir",
        str(report_dir),
        "--report-name",
        report_name,
        "--markdown-name",
        markdown_name,
    ]
    if str(args.init_label).strip() and args.init_checkpoint_dir is not None:
        cmd.extend(["--student-checkpoint", f"{str(args.init_label).strip()}={args.init_checkpoint_dir}"])
    for spec in args.baseline_checkpoint:
        cmd.extend(["--student-checkpoint", spec])
    cmd.extend(["--student-checkpoint", f"{checkpoint_name}={checkpoint_dir}"])
    env = os.environ.copy()
    env.setdefault("COSMOS_DATA_ROOT", "/home/pm97/workspace/dataset/distill_dataset")
    env.setdefault("PYTHONUNBUFFERED", "1")
    report_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "launch", "cmd": cmd, "time": now()}, default=str) + "\n")
        log.flush()
        process = subprocess.Popen(cmd, cwd=str(args.root_dir), env=env, stdout=log, stderr=subprocess.STDOUT)
        return process.wait()


def main() -> None:
    args = parse_args()
    args.root_dir = args.root_dir.resolve()
    checkpoint_names = args.checkpoint_name or ["step_003125", "step_006250", "step_009375", "step_012500", "final"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.root_dir / "outputs" / "checkpoints" / "no_nav_bp3_200k_epoch" / args.run_id
    )
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else args.root_dir / "outputs" / "reports" / "no_nav_prefill_qc_live" / args.run_id
    )
    status_path = report_dir / "live_status.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    status = read_json(status_path)
    if not isinstance(status, dict) or status.get("run_id") != args.run_id:
        status = initial_status(args, checkpoint_names, output_dir, report_dir)
    else:
        status["pid"] = os.getpid()
        status["updated_at"] = now()
        status.setdefault("checkpoints", {})
        for name in checkpoint_names:
            status["checkpoints"].setdefault(
                name,
                initial_status(args, [name], output_dir, report_dir)["checkpoints"][name],
            )
    write_json(status_path, status)

    print(json.dumps({"event": "watch_start", "run_id": args.run_id, "report_dir": str(report_dir), "pid": os.getpid()}), flush=True)
    while True:
        did_work = False
        status["updated_at"] = now()
        status["free_vram_gb"] = free_vram_gb()

        for name in checkpoint_names:
            item = status["checkpoints"][name]
            report_json = Path(item["report_json"])
            if report_json.exists() and item.get("status") != "running":
                item["status"] = "done"
                item["finished_at"] = item.get("finished_at") or report_json.stat().st_mtime
                continue

            if item.get("status") == "done":
                continue

            checkpoint_dir = output_dir / name
            ready, reason = checkpoint_ready(checkpoint_dir, settle_sec=args.settle_sec)
            if not ready:
                item["status"] = "waiting"
                item["reason"] = reason
                continue

            free_gb = status.get("free_vram_gb")
            if free_gb is not None and float(free_gb) < float(args.min_free_vram_gb):
                item["status"] = "waiting_vram"
                item["reason"] = f"free_vram_gb={float(free_gb):.1f} < {float(args.min_free_vram_gb):.1f}"
                continue

            item["status"] = "running"
            item["started_at"] = now()
            item["reason"] = "running_probe"
            write_json(status_path, status)
            print(json.dumps({"event": "probe_start", "checkpoint": name, "checkpoint_dir": str(checkpoint_dir)}), flush=True)
            rc = run_probe(args, checkpoint_name=name, checkpoint_dir=checkpoint_dir, report_dir=report_dir)
            item["finished_at"] = now()
            item["returncode"] = int(rc)
            if rc == 0 and report_json.exists():
                item["status"] = "done"
                item["reason"] = "done"
                print(json.dumps({"event": "probe_done", "checkpoint": name, "report_json": str(report_json)}), flush=True)
            else:
                item["status"] = "failed"
                item["reason"] = f"returncode={rc}"
                print(json.dumps({"event": "probe_failed", "checkpoint": name, "returncode": rc}), flush=True)
            did_work = True
            write_json(status_path, status)

        write_json(status_path, status)
        if args.once:
            break
        if not did_work:
            time.sleep(max(float(args.poll_sec), 5.0))


if __name__ == "__main__":
    main()
