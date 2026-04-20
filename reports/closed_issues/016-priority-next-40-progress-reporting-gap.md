# 016. Priority Next-40 Progress Reporting Gap

## Status

resolved

## Symptom

`priority_next_40.log` showed continued `Fetching ... files` progress, but ad-hoc checks incorrectly concluded that the selected `40` chunks had made no progress.

## Root Cause

The original `priority_next_40_run.json` is a planning snapshot written once before `snapshot_download()` starts and once after it finishes. It does not represent live progress.

This created a visibility gap:

- the log kept advancing
- local chunk files were actually appearing under `data/raw/physical_ai_av`
- but the plan summary stayed static

## Resolution

Added a live progress reporter:

- script: [scripts/03_report_priority_progress.py](/home/pm97/workspace/sukim/cosmos_distillation/scripts/03_report_priority_progress.py)
- output: [outputs/reports/priority_next_40_progress.json](/home/pm97/workspace/sukim/cosmos_distillation/outputs/reports/priority_next_40_progress.json)

The reporter rescans the selected chunk list against the local dataset tree and records:

- completed counts per feature
- remaining chunks by feature
- fully ready chunk count
- selected ready sample count
- active `.incomplete` files by feature
- last downloader log progress line

## Current Snapshot

At the time of the fix:

- fully ready chunks: `22 / 40`
- completed feature counts:
  - `camera_front_wide_120fov: 22`
  - `camera_cross_right_120fov: 40`
  - `camera_front_tele_30fov: 36`
  - `egomotion: 33`
- remaining:
  - `front_wide: 18`
  - `front_tele: 4`
  - `egomotion: 7`

## Follow-up

- Use `priority_next_40_progress.json` instead of the original plan summary for live monitoring.
- If needed later, wire the same reporting directly into `03_download_priority_chunks.py` so every future run emits live progress without a separate reporter step.
