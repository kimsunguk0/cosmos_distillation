# 084 Post-Stage2 minADE@6 Prep

## Context

- Active Stage2 run: `outputs/action_expert/stage2_200k_b8_nt16_fa2_speed_20260603`
- Train target: `25000` steps = 1 epoch over 200k samples with batch 8.
- Paper-facing eval must report `ADE` and `minADE@6` for both `temperature=1.0` and `temperature=0.85`.
- Existing training eval logs used `eval_num_paths=16`, so `ade_best_of_n_*` was `minADE@16`, not paper-comparable `minADE@6`.

## Prepared Changes

1. Added explicit metric aliases in `scripts/84_train_student_ae28_official.py`.
   - Keeps existing `ade_best_of_n_*` and `fde_best_of_n_*`.
   - Adds `minade_at_n_*` / `minfde_at_n_*`.
   - Adds `minade_at_6_*` / `minfde_at_6_*` when `eval_num_paths == 6`.

2. Updated `scripts/101_eval_ae28_seed_sweep.py` compact summaries.
   - Carries the new `minade_at_n` and `minade_at_6` aliases into `seed_sweep.jsonl` and `summary.json`.

3. Added post-Stage2 eval launcher.
   - `scripts/launch_stage2_200k_best_minade6_eval.sh`
   - Uses Stage2 `best.pt`.
   - Runs eval-only sweep:
     - `temp1p0_single_n6`
     - `temp0p85_single_n6`
   - Output default: `outputs/action_expert/stage2_200k_best_minade6_eval_<tag>`

4. Added Q3 comparison eval launcher.
   - `scripts/launch_q3_minade6_temp_sweep_seed1042.sh`
   - Uses Q3 `best.pt`.
   - Runs the same `temperature=1.0/0.85`, `N=6`, `single` sweep.
   - Output default: `outputs/action_expert/q3_minade6_temp_sweep_seed42_evalbase1042_<tag>`

5. Added watcher to run eval after Stage2 finishes.
   - `scripts/watch_stage2_done_then_minade6_eval.sh`
   - Waits for active train PID to exit.
   - Waits for `final.pt`.
   - Waits for free VRAM.
   - Runs Stage2 best minADE@6 eval, then Q3 minADE@6 eval.
   - Active tmux session: `stage2_post_minade6`
   - Watch log: `outputs/action_expert/post_stage2_minade6_queue_20260604_145757/watch.log`

6. Added 2-more-epoch launcher.
   - `scripts/launch_stage2_ae28_200k_more2ep_minade6.sh`
   - Default resume checkpoint: Stage2 `final.pt`.
   - Default range: `start_step=25000`, `end_step=75000`.
   - Eval metric: `eval_num_paths=6`, so logs include `minADE@6`.
   - Default eval temperature: `1.0` for paper-facing baseline.

## Runtime Cleanup

- Removed old `q3_e2e_n6_temp1_wait` watcher to avoid duplicate Q3 temp1 eval.
- Removed old `q2q3_seed_sweep_wait` watcher to avoid N16 sweep competing with the paper-facing minADE@6 eval.
- Left the active Stage2 training process untouched.

## Validation

- `bash -n` passed for all new shell launchers.
- `py_compile` passed for:
  - `scripts/84_train_student_ae28_official.py`
  - `scripts/101_eval_ae28_seed_sweep.py`

## Next Expected Outputs

After Stage2 finishes:

1. Stage2 best `ADE` and `minADE@6` at `temperature=1.0`.
2. Stage2 best `ADE` and `minADE@6` at `temperature=0.85`.
3. Q3 best `ADE` and `minADE@6` at `temperature=1.0`.
4. Q3 best `ADE` and `minADE@6` at `temperature=0.85`.

After reviewing those, launch the 2-more-epoch run if the 1-epoch checkpoint is still below the paper baseline and train/val gap remains small.
