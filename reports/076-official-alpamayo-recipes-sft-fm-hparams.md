# 076 - Official alpamayo-recipes SFT Flow-Matching Hyperparameters

## Source Checked

Repository:

`https://github.com/NVlabs/alpamayo-recipes`

Local clone:

`/home/pm97/workspace/sukim/alpamayo_repo/alpamayo-recipes`

Checked commit:

`f77bc05a3a7e95c8d0cc39e35ad91b8ae3b2d26c` (`2026-05-29`, `Create CONTRIBUTING.md`)

Relevant recipes:

- `recipes/alpamayo1_sft`
- `recipes/alpamayo1_5_sft`

Important dependency pin:

- both SFT recipes depend on `alpamayo_r1`
- both `uv.lock` files pin `alpamayo-r1` to `https://github.com/NVlabs/alpamayo.git#4b96c546ce3416e7b1e4d86e7649662852c3c1ac`
- this is the same official commit that contains the `FlowMatching` implementation checked in report 075

## Alpamayo 1.5 Recipe Findings

### Stage 2 config

Files:

- `/home/pm97/workspace/sukim/alpamayo_repo/alpamayo-recipes/recipes/alpamayo1_5_sft/configs/sft_base.yaml`
- `/home/pm97/workspace/sukim/alpamayo_repo/alpamayo-recipes/recipes/alpamayo1_5_sft/configs/sft_stage2_nav.yaml`
- `/home/pm97/workspace/sukim/alpamayo_repo/alpamayo-recipes/recipes/alpamayo1_5_sft/configs/models/ar1_5_expert.yaml`

Observed config:

- `sft_base.yaml`
  - `per_device_train_batch_size: 1`
  - `per_device_eval_batch_size: 2`
  - `gradient_accumulation_steps: 1`
  - `learning_rate: 1e-4`
  - `warmup_steps: 100`
  - `num_train_epochs: 3`
  - `lr_scheduler_type: cosine_warmup_with_min_lr`
  - `min_lr: 1e-6`
- `sft_stage2_nav.yaml`
  - inherits `sft_base`
  - selects `/models/ar1_5_expert@model`
  - overrides dataset to navigation-conditioned PAI
  - sets `gradient_checkpointing: false`
  - sets `deepspeed: null`
  - sets `ddp_find_unused_parameters: true`
  - does not override `learning_rate`
- `ar1_5_expert.yaml`
  - `pretrained_model_name_or_path: nvidia/Alpamayo-1.5-10B`
  - `cotrain_vlm: false`

Conclusion: official Alpamayo 1.5 Stage 2 action expert recipe uses LR `1e-4`, inherited from `sft_base`.

### Flow-matching training path

File:

`/home/pm97/workspace/sukim/alpamayo_repo/alpamayo-recipes/recipes/alpamayo1_5_sft/models/sft_alpamayo_r1.py`

Observed code path:

- `_process_traj_future_training(...)` converts GT future trajectory to action
- calls `self.diffusion.construct_training_data(action)`
- forward calls `self.action_in_proj(future_traj_data["noisy_x"], future_traj_data["timesteps"])`
- expert runs once
- `action_out_proj` predicts vector field
- loss is `self.diffusion.compute_loss_from_pred(training_data=future_traj_data, pred=pred)`

There is no loop/repeat in the recipe-side model around `construct_training_data`, and no config key corresponding to `num_time_samples`, `n_time_samples`, `timesteps_per_sample`, or `draws_per_step`.

Because `uv.lock` pins `alpamayo_r1` to `NVlabs/alpamayo` commit `4b96c54`, the FM implementation is the one checked in report 075:

- one `t` per action row: `t = beta_dist.sample((batch_size,))`
- one `noise = torch.randn_like(x)`
- `noisy_x = t * x + (1 - t) * noise`
- target is `x - noise`

Conclusion: official Alpamayo 1.5 recipe uses one random `(x0, t)` draw per sample per forward.

## Alpamayo 1 Recipe Cross-Check

The Alpamayo 1 recipe matches the same pattern:

- `recipes/alpamayo1_sft/configs/sft_base.yaml` has `learning_rate: 1e-4`, `warmup_steps: 100`, `gradient_accumulation_steps: 1`, `per_device_train_batch_size: 1`
- `recipes/alpamayo1_sft/configs/sft_stage2.yaml` inherits base and does not override LR
- `recipes/alpamayo1_sft/configs/models/ar1_expert.yaml` uses `pretrained_model_name_or_path: nvidia/Alpamayo-R1-10B`, `cotrain_vlm: false`
- `recipes/alpamayo1_sft/models/sft_alpamayo_r1.py` has the same single `self.diffusion.construct_training_data(action)` training path
- `recipes/alpamayo1_sft/SKILL.md` explicitly says Stage 2 inherits `1e-4` from base

## Comparison Table

| 항목 | official `alpamayo-recipes` value | 우리 H3 / Stage 1 value | 일치? | 비고 |
|---|---:|---:|---|---|
| action expert LR | `1e-4` | `expert_lr=1e-4` | Y | 1.5 Stage2 nav inherits base LR |
| action projection LR | same optimizer/base LR `1e-4` | `proj_lr=1e-4` | Y/부분 | 공식은 projection 전용 LR 분리 없음 |
| timestep/noise draws | `1` | `num_time_samples=16` | N | recipe에도 multi-draw 설정 없음 |
| train batch per GPU | `1` | current run `2` on 1x H200 | N | 공식은 8x H100 가정 |
| gradient accumulation | Stage2 `1` | `1` | Y | Stage1은 task별로 더 큼 |
| official global FM batch | `8 GPUs * 1 batch * 1 draw = 8` | `1 GPU * batch 2 * 16 draws = 32` | N/부분 | 우리 draw 16이 effective FM batch를 크게 키움 |
| timestep sampler | Beta(1.5, 1.0) via pinned `alpamayo_r1` | Beta(1.5, 1.0) | Y | 공식 dependency implementation과 동일 |
| FM target/formula | `noisy_x=t*x+(1-t)*noise`, target `x-noise` | same as `x1-x0` | Y | 방향 정합 |
| warmup/schedule | warmup `100`, cosine to `1e-6` | current run warmup `0`, constant LR | N | 우리 스크립트는 옵션으로 구현되어 있으나 현재 off |
| inference/eval sampling | temp `0.6`, `num_traj_samples` default `6`, metrics include minADE | temp `0.85`, `N=16`, `mean_traj` | N | official metric is closer to multi-sample minADE reporting |
| VLM in Stage2 | frozen (`cotrain_vlm=false`) | frozen/student prefix route | Y/부분 | 공식은 pretrained expert fine-tune; 우리는 student-init/reset AE distillation |

## Interpretation

`alpamayo-recipes` confirms the most important part of H3: `expert_lr=1e-4` is not an arbitrary local hack; it matches the official Stage 2 SFT LR for both Alpamayo 1 and Alpamayo 1.5 recipes.

It also confirms that `num_time_samples=16` is not official SFT default behavior. Official recipes use one FM draw per sample per forward.

The difference is explainable by initialization/task mismatch:

- official Stage 2 fine-tunes a pretrained action expert stack from `nvidia/Alpamayo-R1-10B` or `nvidia/Alpamayo-1.5-10B`
- our distillation path uses a student-KV-compatible AE path with reset/changed projections and student-backbone initialization
- H3 showed that this colder-start 28-layer expert collapses with single-draw FM but recovers with `draws=16`

## One-Line Verdict

우리 H3 recipe가 `alpamayo-recipes` 공식 SFT와 정합하는가 = **부분**. LR `1e-4`는 공식과 정합, `num_time_samples=16`은 공식보다 큼. 공식은 pretrained expert fine-tune, 우리는 student-init/reset AE distillation이라 multi-draw는 우리 쪽 안정화 보강값이다.
