#!/usr/bin/env python3
"""Select and pin the single B0 GT-SFT body checkpoint from decode summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="Decode summary JSON files from script 25.")
    parser.add_argument(
        "--link-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "B0_GT_SFT_BODY_BEST",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "B0_GT_SFT_BODY_BEST_selection.json",
    )
    parser.add_argument(
        "--selection-md",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "B0_GT_SFT_BODY_BEST_selection.md",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing symlink/file at --link-path.")
    return parser.parse_args()


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_summary_path"] = str(path)
    return payload


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except Exception:  # noqa: BLE001
        return default
    return out


def _candidate_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Lower is better; matches the agreed post-SFT checkpoint priority."""
    ade = _safe_float(summary.get("avg_ade_m"), float("inf"))
    fde = _safe_float(summary.get("avg_fde_m"), float("inf"))
    invalid_rate = _safe_float(summary.get("invalid_future_token_rate_i3000_plus"), float("inf"))
    invalid_avg = _safe_float(summary.get("avg_invalid_future_tokens_i3000_plus"), float("inf"))
    max_run = _safe_float(summary.get("avg_max_same_token_run"), float("inf"))
    unique = _safe_float(summary.get("avg_unique_traj_ids"), 0.0)
    return (ade, fde, invalid_rate, invalid_avg, max_run, -unique)


def _group_by_checkpoint(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        checkpoint = str(summary.get("checkpoint_dir") or "")
        if not checkpoint:
            continue
        item = grouped.setdefault(checkpoint, {"checkpoint_dir": checkpoint, "summaries": []})
        item["summaries"].append(summary)
        split = str(summary.get("split") or "")
        if split:
            item[split] = summary
    return grouped


def _rank_group(item: dict[str, Any]) -> tuple[float, ...]:
    val_summary = item.get("val") or item["summaries"][0]
    key = list(_candidate_key(val_summary))
    train_summary = item.get("train")
    if train_summary is not None:
        val_ade = _safe_float(val_summary.get("avg_ade_m"), float("inf"))
        train_ade = _safe_float(train_summary.get("avg_ade_m"), val_ade)
        key.append(abs(val_ade - train_ade))
    else:
        key.append(float("inf"))
    return tuple(key)


def _write_markdown(path: Path, *, best: dict[str, Any], ranked: list[dict[str, Any]]) -> None:
    lines = [
        "# B0 GT SFT Body Best Selection",
        "",
        f"selected checkpoint: `{best['checkpoint_dir']}`",
        "",
        "| Rank | Checkpoint | Split | ADE | FDE | Invalid Rate | Max Run | Unique | Summary |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(ranked, start=1):
        for summary in sorted(item["summaries"], key=lambda s: str(s.get("split") or "")):
            lines.append(
                "| "
                f"{rank} | "
                f"`{Path(item['checkpoint_dir']).name}` | "
                f"{summary.get('split')} | "
                f"{_safe_float(summary.get('avg_ade_m'), float('nan')):.4f} | "
                f"{_safe_float(summary.get('avg_fde_m'), float('nan')):.4f} | "
                f"{_safe_float(summary.get('invalid_future_token_rate_i3000_plus'), float('nan')):.4f} | "
                f"{_safe_float(summary.get('avg_max_same_token_run'), float('nan')):.4f} | "
                f"{_safe_float(summary.get('avg_unique_traj_ids'), float('nan')):.4f} | "
                f"`{summary.get('_summary_path')}` |"
            )
    lines.extend(
        [
            "",
            "Selection priority:",
            "",
            "1. Full validation ADE/FDE.",
            "2. Invalid future-token rate, max-run, avg-unique, and token histogram sanity.",
            "3. Train/val gap when train decode summaries are supplied.",
            "4. Overlay/failure taxonomy review before using this as the base for KD/hidden/expert runs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _replace_link(link_path: Path, target: Path, *, force: bool) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if link_path.exists() or link_path.is_symlink():
        if not force:
            raise SystemExit(f"{link_path} already exists; pass --force to replace it.")
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            shutil.rmtree(link_path)
    os.symlink(target, link_path, target_is_directory=True)


def main() -> int:
    args = parse_args()
    summaries = [_load_summary(path) for path in args.summaries]
    grouped = _group_by_checkpoint(summaries)
    if not grouped:
        raise SystemExit("No checkpoint_dir entries found in supplied summaries.")
    ranked = sorted(grouped.values(), key=_rank_group)
    best = ranked[0]
    best_checkpoint = Path(best["checkpoint_dir"])
    if not best_checkpoint.exists():
        raise SystemExit(f"Selected checkpoint does not exist: {best_checkpoint}")

    _replace_link(args.link_path, best_checkpoint, force=args.force)
    args.selection_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_name": "B0_GT_SFT_BODY_BEST",
        "selected_checkpoint_dir": str(best_checkpoint),
        "link_path": str(args.link_path),
        "ranking_key": [
            "val_avg_ade_m",
            "val_avg_fde_m",
            "invalid_future_token_rate_i3000_plus",
            "avg_invalid_future_tokens_i3000_plus",
            "avg_max_same_token_run",
            "-avg_unique_traj_ids",
            "train_val_ade_gap_if_available",
        ],
        "ranked": ranked,
    }
    args.selection_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(args.selection_md, best=best, ranked=ranked)
    print(json.dumps({"selected": str(best_checkpoint), "link_path": str(args.link_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
