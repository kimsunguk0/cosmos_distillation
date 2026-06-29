#!/usr/bin/env python3
"""Write a markdown report from benchmark_4models outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH = PROJECT_ROOT / "outputs/benchmarks/semantic_val806_4models_20260612"
DEFAULT_REPORT = PROJECT_ROOT / "reports/138-semantic-val806-4model-benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(model: dict[str, Any], key: str, subkey: str = "mean") -> str:
    value = ((model.get("metrics") or {}).get(key) or {}).get(subkey)
    return "NA" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    summary_path = args.benchmark_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    models = summary.get("models") or []
    lines: list[str] = []
    lines.append("# Report 138: Semantic Val806 4-Model Benchmark")
    lines.append("")
    lines.append("Date: 2026-06-12")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Benchmark public Alpamayo 10B, no-FLEX 2B+AE28, FLEX K512+AE28, and FLEX K512+AE14.")
    lines.append("- All models generate their own CoT/prefix before trajectory inference.")
    lines.append("- Metrics are computed on the same semantic validation benchmark set.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Corpus: `{summary.get('settings', {}).get('corpus_jsonl')}`")
    lines.append(f"- Selected samples: `{summary.get('selected_count')}`")
    lines.append("- Category counts:")
    for cat, count in (summary.get("category_counts") or {}).items():
        lines.append(f"  - `{cat}`: {count}")
    lines.append("")
    lines.append("## Eval Settings")
    lines.append("")
    settings = summary.get("settings") or {}
    for key in (
        "eval_num_paths",
        "eval_temperature",
        "eval_selection_method",
        "default_inference_steps",
        "ae14_inference_steps",
        "batch_size",
        "student_batch_size",
        "attn_implementation",
        "dtype",
        "seed",
    ):
        lines.append(f"- `{key}`: `{settings.get(key)}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Model | N | ADE GT | FDE GT | minADE6 GT | minFDE6 GT | ADE vs 10B | minADE6 vs 10B | latency ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in models:
        lines.append(
            "| "
            f"{model.get('model_label') or model.get('model_key')} | "
            f"{model.get('count')} | "
            f"{metric(model, 'ade_gt_m')} | "
            f"{metric(model, 'fde_gt_m')} | "
            f"{metric(model, 'minade6_gt_m')} | "
            f"{metric(model, 'minfde6_gt_m')} | "
            f"{metric(model, 'ade_10b_m')} | "
            f"{metric(model, 'minade6_10b_m')} | "
            f"{metric(model, 'elapsed_ms')} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Combined summary: `{summary_path}`")
    lines.append(f"- Prediction NPZ root: `{args.benchmark_dir / 'predictions'}`")
    lines.append(f"- Visualizations: `{args.benchmark_dir / 'visualizations'}`")
    for model in models:
        rows = model.get("rows_jsonl")
        if rows:
            lines.append(f"- `{model.get('model_key')}` rows: `{rows}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `ADE GT` is the deployable selected trajectory using `eval_selection_method`.")
    lines.append("- `minADE6 GT` is oracle best-of-6 against GT for diagnostic comparison.")
    lines.append("- Student `vs 10B` metrics compare student trajectories against the 10B selected trajectory on the same sample.")
    lines.append("- AE14 is evaluated with the configured AE14 denoising step count, currently 4 steps for deployment-oriented latency.")
    lines.append("")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "report_written", "path": str(args.report_path)}), flush=True)


if __name__ == "__main__":
    main()
