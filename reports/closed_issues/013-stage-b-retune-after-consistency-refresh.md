# 013 Stage B Retune After Consistency Refresh

status: resolved

symptom:
- The first refreshed Stage B run activated `self_cons` and `rank`, but those auxiliary terms occasionally spiked and dominated the total loss on individual steps.
- The pre-refresh corpus used a stale consistency matrix, so one earlier Stage B run had `consistency_score == 0` everywhere and did not actually exercise the intended regularization path.

root cause:
- `08_build_multitask_corpus.py` had first been launched in parallel with consistency generation, so the corpus was built before the newest matrix landed.
- Once the corpus was rebuilt against the latest matrix, `self_cons` and `rank` became active as designed, which exposed that the default Stage B auxiliary weights were a bit aggressive for the currently available `101` teacher-soft train samples.

resolution:
- Rebuilt the corpus after consistency generation completed.
- Verified refreshed train subset statistics:
  - materialized train samples: `195`
  - nonzero consistency samples: `193`
  - nonzero `seq_kd` samples: `101`
- Ran a refreshed Stage B main pass with the default Stage B config.
- Added `configs/train/stage_b_tuned.yaml` with:
  - `self_cons: 0.08`
  - `rank: 0.05`
- Re-ran Stage B on the same sample budget and kept the tuned run as the preferred current baseline.

key files or artifacts:
- `configs/train/stage_b_tuned.yaml`
- `outputs/checkpoints/train_stage_b_specfix_main_refreshed/metrics.jsonl`
- `outputs/checkpoints/train_stage_b_specfix_tuned/metrics.jsonl`
- `outputs/reports/train_stage_b_specfix_main_refreshed.json`
- `outputs/reports/train_stage_b_specfix_tuned.json`

comparison:
- Refreshed default Stage B:
  - `avg_total_loss ~= 7.68`
  - `last_total_loss ~= 5.06`
  - `max_grad_norm ~= 555.21`
- Tuned Stage B:
  - `avg_total_loss ~= 7.03`
  - `last_total_loss ~= 4.92`
  - `max_grad_norm ~= 435.30`

follow-up:
- The checkpointing warning about inputs without gradients still appears in LoRA mode, even though gradients on trainable parameters are present and runs complete successfully.
- If teacher hidden/logit caches are added later, the tuned Stage B config should be revisited because currently only `hard_ce`, `seq_kd`, `aux`, `self_cons`, and `rank` are active.
