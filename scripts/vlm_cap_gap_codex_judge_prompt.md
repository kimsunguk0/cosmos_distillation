# Codex 5.5 xhigh Blind Judge Task

You are judging blind pairwise answers for the Vanilla Cosmos-Reason2 2B vs 8B human-OOD driving VLM capacity-gap eval.

Input directory:

`outputs/eval/vlm_cap_gap_human_ood_20260701_full300/open_judge_codex55_xhigh_full300`

Files:

- `pairs.jsonl`: blind A/B cases for T3 and T4. It intentionally contains only the 2B/8B answers, not the 32B answer, event cluster, clear/ambiguous tag, or metadata reasoning hint.
- `contact_sheets/`: visual evidence for each case.
- `answer_key.json`: maps A/B to model identity. Do not read this until all blind judgments are written.

Rules:

1. Judge only from visible evidence in the contact sheet and the provided question.
2. Prefer the answer that is more visibly grounded, better calibrated, and less speculative.
3. Penalize claims about people, motion, intent, or future events that are not visible.
4. Penalize overconfident stop/yield/proceed claims when the visible evidence is ambiguous.
5. Penalize answers that ignore visible humans, roadway edge cases, or obvious calibration needs.
6. Use `tie` only when both answers are similarly grounded and similarly calibrated.
7. Use `both_bad` when both answers materially hallucinate, miss the key visible evidence, or answer the wrong task.

Required output:

Write JSONL to:

`outputs/eval/vlm_cap_gap_human_ood_20260701_full300/open_judge_codex55_xhigh_full300/judgments_blind.jsonl`

Each line must be:

```json
{"case_id":"...","task_id":"T3|T4","winner":"A|B|tie|both_bad","rationale":"short visible-evidence rationale","failure_modes":["..."]}
```

Important sequencing:

1. Read `pairs.jsonl` and contact sheets.
2. Write all blind judgments.
3. Validate the blind file without reading `answer_key.json`:

```bash
/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python scripts/validate_vlm_open_judge.py \
  --judge-dir outputs/eval/vlm_cap_gap_human_ood_20260701_full300/open_judge_codex55_xhigh_full300 \
  --out outputs/eval/vlm_cap_gap_human_ood_20260701_full300/open_judge_codex55_xhigh_full300/judgments_validation.json
```

4. Only after validation passes, read `answer_key.json`.
5. Run:

```bash
/home/pm97/workspace/sukim/alpamayo_repo/alpamayo1.5/.venv/bin/python scripts/score_vlm_open_judge.py \
  --judge-dir outputs/eval/vlm_cap_gap_human_ood_20260701_full300/open_judge_codex55_xhigh_full300 \
  --append-report outputs/eval/vlm_cap_gap_human_ood_20260701_full300/report.md
```

Do not change model predictions, score summaries, or judge pack inputs.
