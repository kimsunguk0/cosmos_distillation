# 082 Q2/Q3 Diagnostics Launch

Date: 2026-06-02

## Launcher

- Script: `outputs/action_expert/run_q2_q3_plateau_diagnostics_20260602_0220.sh`
- Master log: `outputs/action_expert/q2_q3_plateau_diagnostics_20260602_0220.log`
- tmux session: `q2q3_diag_0220`
- Status at launch check: Q2 entered `train_loop_start`
- Active Q2 PID: 17058

## Queue

| order | diagnostic | output dir | steps | resume | LR schedule |
|---:|---|---|---:|---|---|
| 1 | Q2 continue | `outputs/action_expert/q2_continue_s10000_to_s30000_b8pb8_20260602_0220` | 10000 -> 30000 | `stage1_fast_resume_s5000_b8_fa2_20260601_081126/final.pt` | constant, actual recorded Stage 1 recipe |
| 2 | Q3 control | `outputs/action_expert/q3_constant_b8pb8_s10000_20260602_0220` | 0 -> 10000 | none | constant |
| 3 | Q3 cosine | `outputs/action_expert/q3_cosine_w100_min1e6_b8pb8_s10000_20260602_0220` | 0 -> 10000 | none | warmup 100, cosine, `min_lr=1e-6` |
| 4 | Q3 cosine high floor | `outputs/action_expert/q3_cosine_w100_min1e5_b8pb8_s10000_20260602_0220` | 0 -> 10000 | none | warmup 100, cosine, `min_lr=1e-5` |

## Shared Settings

- Train samples: 20000
- Held-out val samples: 2000
- Eval samples: 512
- Batch size: 8
- Eval batch size: 8
- Eval path batch size: 8
- Eval every: 1000 steps
- `expert_lr=1e-4`
- `proj_lr=1e-4`
- `num_time_samples=16`
- `eval_num_paths=16`
- `eval_selection_method=mean_traj`
- `eval_seed_mode=fixed`
- Attention: `flash_attention_2`

Note: the recorded Stage 1 run has `lr_warmup_steps=0`, so the Q3 queue includes a constant control plus two cosine variants rather than assuming the previous 2.5 m plateau came from an already-enabled cosine schedule.
