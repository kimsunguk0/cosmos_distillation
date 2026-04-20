# Teacher-Pair Upper-Bound SFT From H0d (`step128`)

Base checkpoint:
- [full_h0d_softrel_prefix16_from_h0b_step128/final](/workspace/cosmos_distillation/outputs/checkpoints/full_h0d_softrel_prefix16_from_h0b_step128/final)

Stage config:
- [stage_sft_teacherpair_upperbound_from_h0d.yaml](/workspace/cosmos_distillation/configs/train/stage_sft_teacherpair_upperbound_from_h0d.yaml)

Goal:
- Test a simple upper-bound hypothesis:
  can the student fit pure LM teacher-pair targets if we stop doing hidden/readout probes and just run teacher-pair CE?

Setup:
- corpus: `distill_v3_2_959.jsonl`
- train/val: `754 / 204`
- init: `H0d-A`
- trainable:
  - all-layer LoRA
  - trainable trajectory/special token rows
  - multimodal projector
  - language norm
- losses:
  - `gt_cot_loss = 1.0`
  - `traj_loss = 1.0`
  - `output_format_loss = 0.15`
  - all KD / hidden losses off

Artifacts:
- train summary: [smoke_sft_teacherpair_upperbound_from_h0d_step128.json](/workspace/cosmos_distillation/outputs/reports/smoke_sft_teacherpair_upperbound_from_h0d_step128.json)
- infer val10: [smoke_sft_teacherpair_upperbound_from_h0d_step128_infer_val10_lm.json](/workspace/cosmos_distillation/outputs/reports/smoke_sft_teacherpair_upperbound_from_h0d_step128_infer_val10_lm.json)
- readout eval-only: [evalonly_readout_sft_teacherpair_from_h0d_step128.json](/workspace/cosmos_distillation/outputs/reports/evalonly_readout_sft_teacherpair_from_h0d_step128.json)
- compare: [sft_teacherpair_upperbound_compare.json](/workspace/cosmos_distillation/outputs/reports/sft_teacherpair_upperbound_compare.json)

Training result:
- val total loss improved:
  - `8.342 -> 7.782 -> 7.514`
- so the model does fit teacher-pair CE better than the H0d init checkpoint.

Teacher-forced readout result:
- H0d readout baseline:
  - `traj-vocab target rank = 174.85`
  - `target-vs-1499 margin = -3.50`
  - `traj_top1_1499_ratio = 1.00`
- after teacher-pair SFT:
  - `traj-vocab target rank = 172.39`
  - `target-vs-1499 margin = -2.97`
  - `traj_top1_1499_ratio = 0.00`
  - dominant traj top1 rows become `1501` / `1498`

Free LM decode result:
- `generated_traj_token_count = 128` survives
- but body is still a plateau-like 2-state pattern
- val10 aggregate:
  - `avg_unique_traj_ids = 2.0`
  - `avg_max_same_token_run = 125.6`

Interpretation:
- This is not a clean failure.
- It shows the student can absorb simple teacher-pair CE:
  validation loss drops and the dominant attractor moves.
- But it still does **not** solve trajectory-token discrimination.
- The model shifts from single-token `<i1499>` collapse to a narrow central-band `1498/1501` 2-state collapse.

Verdict:
- `teacher-pair SFT` helps more than rows-only readout probes at changing the free-LM attractor.
- But even this upper-bound-style smoke does **not** make the student produce a healthy trajectory body.
- So the remaining bottleneck is not “just do vanilla SFT”.
- It is still a readout / last-stage coupling problem, now at the level of the whole central trajectory band rather than one token.
