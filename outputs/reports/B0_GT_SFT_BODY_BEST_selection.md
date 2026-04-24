# B0 GT SFT Body Best Selection

selected checkpoint: `/workspace/cosmos_distillation/outputs/checkpoints/stage_t1_sft_reweight_off_continue5k/final`

| Rank | Checkpoint | Split | ADE | FDE | Invalid Rate | Max Run | Unique | Summary |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `final` | train | 1.1028 | 3.4929 | 0.0000 | 1.3594 | 66.9688 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/final_train_summary.json` |
| 1 | `final` | val | 4.6685 | 13.3703 | 0.0000 | 1.9069 | 60.7206 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/final_val_summary.json` |
| 2 | `step_003016` | train | 1.4250 | 5.0605 | 0.0000 | 1.3438 | 63.5312 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/step_003016_train_summary.json` |
| 2 | `step_003016` | val | 4.6834 | 13.8255 | 0.0000 | 1.3480 | 53.0392 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/step_003016_val_summary.json` |
| 3 | `step_003770` | train | 1.1078 | 4.0918 | 0.0000 | 1.8125 | 62.8125 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/step_003770_train_summary.json` |
| 3 | `step_003770` | val | 4.7880 | 13.9817 | 0.0000 | 1.4069 | 49.3578 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/step_003770_val_summary.json` |
| 4 | `step_004524` | train | 1.1997 | 4.0700 | 0.0000 | 1.3906 | 69.5625 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/step_004524_train_summary.json` |
| 4 | `step_004524` | val | 5.1725 | 14.7811 | 0.0000 | 1.9069 | 53.8333 | `/workspace/cosmos_distillation/outputs/reports/b0_sft_body_best_sweep/step_004524_val_summary.json` |

Selection priority:

1. Full validation ADE/FDE.
2. Invalid future-token rate, max-run, avg-unique, and token histogram sanity.
3. Train/val gap when train decode summaries are supplied.
4. Overlay/failure taxonomy review before using this as the base for KD/hidden/expert runs.
