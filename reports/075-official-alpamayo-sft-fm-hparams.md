# 075 - Official Alpamayo SFT Flow-Matching Hyperparameters

## Source Checked

Official repo clone:

`/home/pm97/workspace/sukim/alpamayo_repo/alpamayo_base`

Remote:

`https://github.com/NVlabs/alpamayo.git`

Important repository state:

| ref | date | note |
|---|---|---|
| `4cda35d22bb257f0936ac272397627b9309ca211` | 2026-05-29 | current `origin/main`; `finetune/sft` and `docs/FINETUNE_SFT.md` are removed |
| `4b96c546ce3416e7b1e4d86e7649662852c3c1ac` | 2026-04-14 | public release commit with SFT scripts, `Release SFT and RL post-training scripts (#63)` |
| local `ff067c438dc58cc98e6a677d4e98e6ca583c17b9` | 2026-04-03 | also contains SFT scripts; relevant SFT files match `4b96c54` |

Therefore the SFT hyperparameter check below uses the official release commit `4b96c54` / local checked-out SFT files. Current `origin/main` no longer contains the requested SFT files.

## Direct Code/Config Findings

### Stage 2 Learning Rate

`finetune/sft/configs/sft_stage2.yaml` inherits `sft_base` and does not override `learning_rate`.

Relevant files:

- `finetune/sft/configs/sft_base.yaml`: `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 1`, `learning_rate: 1e-4`, `warmup_steps: 100`, `num_train_epochs: 3`, cosine schedule with `min_lr: 1e-6`
- `finetune/sft/configs/sft_stage2.yaml`: overrides `gradient_checkpointing: false`, `deepspeed: null`, `ddp_find_unused_parameters: true`, but not LR
- `finetune/sft/configs/models/ar1_expert.yaml`: `cotrain_vlm: false`
- `finetune/sft/models/sft_alpamayo_r1.py`: when `cotrain_vlm` is false, VLM parameters are frozen
- `finetune/sft/trainer.py`: if `lr_multiplier` is absent, it falls back to Hugging Face Trainer optimizer behavior; Stage 2 has no `lr_multiplier`

Conclusion: official Stage 2 trains action expert/action projections at base LR `1e-4`. It is not a separate lower expert LR.

### Flow-Matching Draw Count

No config key corresponding to `num_time_samples`, `n_time_samples`, `timesteps_per_sample`, or `draws_per_step` is present in `finetune/sft/configs`.

The training call path is:

- `TrainableAlpamayoR1.forward(...)`
- `_process_traj_future_training(...)`
- `self.diffusion.construct_training_data(action)`
- `action_in_proj(noisy_x, timesteps)`
- `diffusion.compute_loss_from_pred(...)`

`FlowMatching.construct_training_data(x)` samples one `t` tensor with shape `(batch_size,)` and one `noise = torch.randn_like(x)`. There is no loop or repeat over multiple random `(x0, t)` draws per sample in the public SFT training path.

Conclusion: official public Stage 2 uses **1 timestep/noise draw per sample per forward**.

### Timestep Sampler / FM Formula

`FlowMatching` defaults:

- `train_timestep_sampler="beta"`
- `num_inference_steps=10`
- `train_ignore_guidance_rate=0.1`
- `inference_guidance_weight=1.0`
- Beta sampler: `Beta(1.5, 1.0)`
- training `t`: `0.999 - Beta(1.5, 1.0) * 0.999`
- noised action: `noisy_x = t * x + (1 - t) * noise`
- target velocity: `x - noise`
- loss: MSE between predicted vector field and `x - noise`

This matches our FM direction/formula when our `x1 = target_action`, `x0 = noise`, `x_t = (1 - t) * x0 + t * x1`, and `target_v = x1 - x0`.

### Guidance / CFG

The `FlowMatching` constructor exposes `train_ignore_guidance_rate=0.1` and `inference_guidance_weight=1.0`, but in the public SFT training path `construct_training_data` sets `is_drop_guidance: None`, and the SFT forward path does not implement a guidance drop/CFG branch around the expert loss. I found no active CFG/guidance use in the public Stage 2 training loop.

### Official Eval Sampling

`sft_base.yaml` evaluation config uses:

- `ReasoningSampler`
- `top_p: 0.98`
- `temperature: 0.6`
- no explicit `num_traj_samples`, so `ReasoningSampler` default applies: `num_traj_samples=6`

The docs report example Stage-2 validation metrics including `val/metric/ade = 2.0072` and `val/metric/min_ade = 0.6270`, i.e. official reporting includes multi-sample minADE-style metrics, not only a single trajectory.

## Comparison Table

| 항목 | 공식 SFT 값 | 우리 H3 / Stage 1 값 | 일치? | 비고 |
|---|---:|---:|---|---|
| action expert LR | `1e-4` | `expert_lr=1e-4` | Y | 공식 Stage 2는 `sft_base.learning_rate`를 그대로 상속 |
| action projection LR | `1e-4`로 추정 | `proj_lr=1e-4` | Y/부분 | 공식은 별도 projection LR 없이 trainable params에 같은 LR 적용 |
| timestep/noise draws per sample | `1` | `num_time_samples=16` | N | 공식 공개 코드에는 multi-draw 반복 없음 |
| train per-device batch | `1` | `2` on current H200 run | N | 공식 docs는 8x H100 예시 |
| gradient accumulation | `1` for Stage 2 | `1` | Y | Stage 1 공식은 grad accum 4, Stage 2는 base값 1 |
| global/effective FM batch | docs setup 기준 `1 * 8 GPUs * 1 draw = 8` | `2 * 1 GPU * 16 draws = 32` | N/부분 | 우리 multi-draw가 공식보다 effective FM batch를 키움 |
| timestep sampler | Beta(1.5, 1.0), `t=0.999-Beta*0.999` | same | Y | 수식 정합 |
| FM target | `x - noise` | `x1 - x0` | Y | `x=x1`, `noise=x0`로 동일 |
| inference Euler steps | default `10` | `10` in W/Y Stage | Y | X1에서 늘려도 큰 효과 없음 확인 |
| warmup | `100` steps | current `0` | N | 공식은 warmup + cosine; 우리는 constant LR로 돌리는 중 |
| LR schedule | cosine warmup with `min_lr=1e-6` | constant unless `--lr-warmup-steps > 0` | N/부분 | 우리 스크립트에 cosine 구현은 있으나 현재 run은 disabled |
| total training length | `num_train_epochs: 3` | `steps=10000` | 직접 비교 불가 | 공식 exact steps는 dataset size/world size에 의존 |
| VLM training in Stage 2 | frozen (`cotrain_vlm: false`) | frozen / student prefix only | Y/부분 | 공식은 pretrained expert fine-tune, 우리는 student-init/reset AE distillation |
| active CFG/guidance in train | public path에서 명시 사용 못 찾음 | 없음 | Y/불명 | constructor 값은 있으나 train loss 경로에는 미사용 |
| eval sampling | temp `0.6`, top_p `0.98`, default `N=6`, minADE reported | temp `0.85`, `N=16`, mean_traj | N | 우리 deployable selection은 공식 eval과 다름 |

## Interpretation

The strongest alignment is LR: our H3 fix `expert_lr=1e-4` is consistent with the official Stage 2 SFT recipe.

The strongest mismatch is draw count: official public SFT uses one random FM draw per sample per forward, while our H3 recipe needs `num_time_samples=16`.

This mismatch is not necessarily a contradiction. Official Stage 2 starts from `nvidia/Alpamayo-R1-10B` and fine-tunes a pretrained action expert/action projection stack, while our distillation path rebuilds an AE for student KV and resets/changes projection/expert components. In that colder-start setting, H3 showed single draw under-trains the 28-layer expert and collapses; `num_time_samples=16` is acting as a stabilizer/effective-FM-batch increase.

## One-Line Verdict

우리 H3 recipe가 공식 SFT와 정합하는가 = **부분**. `expert_lr=1e-4`는 공식과 정합, `num_time_samples=16`은 공식 공개 SFT보다 큼. 다만 공식은 pretrained expert fine-tune이고 우리는 student-init/reset AE distillation이라 draw 수 차이는 조건 차이로 설명 가능.
