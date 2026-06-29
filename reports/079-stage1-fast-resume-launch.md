# 079 Stage 1 Fast Resume Launch

## Previous Run Stopped

- Stopped PID `11082` after `step_005000.pt` and refreshed `best.pt` were saved.
- Step 5000 eval:
  - val full ADE: `2.7119946169890463`
  - val h1.6 ADE: `0.16828872329945443`
  - val h3.2 ADE: `0.6736939544565975`
  - val best-of-16 ADE: `1.802206107424572`
  - train full ADE: `2.3996994011104107`
  - train h1.6 ADE: `0.15285473412768624`
  - train h3.2 ADE: `0.5894274394813692`

## New Run

- Output: `outputs/action_expert/stage1_fast_resume_s5000_b8_fa2_20260601_081126`
- Parent PID: `29379`
- Python PID: `29383`
- Resume checkpoint: `outputs/action_expert/stage1_heldout20k_val2k_s10000_seed42_full444k_20260531/step_005000.pt`
- Start step: `5000`
- End step: `10000`

## Fast Settings

- `batch_size=8`
- `effective_fm_batch=128` expected from `batch_size * num_time_samples`
- `num_time_samples=16`
- `expert_lr=1e-4`
- `proj_lr=1e-4`
- `attn_implementation=flash_attention_2`
- `eval_samples=512`
- `eval_train_samples=256`
- `eval_num_paths=16`
- `eval_batch_size=8`
- `eval_vectorize_paths=true`
- `eval_path_batch_size=4`
- `eval_log_rows=0`
- `cleanup_every=0`
- `eval_cleanup_every=0`

## Resource Check

- No OOM during startup.
- Observed GPU use after startup:
  - Python process `29383`: about `93-98GB`
  - Other process `31553`: about `27.5GB`
  - Total GPU memory: about `121-125GB / 143.8GB`

