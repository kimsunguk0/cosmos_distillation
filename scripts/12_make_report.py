#!/usr/bin/env python3
"""WP15 entrypoint: report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "reports" / "pipeline_report.md",
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        type=Path,
        default=[],
        help="Repeat for any JSON summaries to aggregate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = ["# Cosmos Distillation Report", ""]
    for summary_path in args.summary_json:
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        sections.append(f"## {summary_path.stem}")
        sections.append("```json")
        sections.append(json.dumps(payload, indent=2))
        sections.append("```")
        sections.append("")

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")
    print(str(args.report_path))


if __name__ == "__main__":
    main()
