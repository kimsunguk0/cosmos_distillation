# 034. Teacher-Pair T1 Bootstrap

## Summary

`traj15`가 patched Alpamayo 1.5 discrete trajectory extraction 결과라는 전제가 확인된 뒤, `teacher honest joint pair`를 hard branch main target으로 쓰는 `T1 pair-only` 경로를 처음으로 배선했다.

이번 단계의 결론은 두 가지다.

1. `teacher CoT + teacher discrete traj`를 같은 hard branch에서 함께 supervision 하는 경로는 실제로 작동한다.
2. 하지만 `train4 / 20 steps` 기준으로는 아직 `<|cot_end|><|traj_future_start|>...` formatting이 약해서, free decode에서는 traj body가 전혀 나오지 않았다.

즉, teacher pair mainline은 viable하지만, 첫 병목은 이제 `pair provenance`가 아니라 `joint formatting / transition into traj span`이다.

## What Changed

### Collator

- [src/training/collator.py](/workspace/cosmos_distillation/src/training/collator.py)
- 새 `teacher_pair_target` 스위치를 추가했다.
- 이 모드에서는 hard branch target을
  - `teacher_target.cot_text`
  - `/data/teacher_cache/traj15`의 `token_ids`
  로 구성한다.
- 같은 hard branch에 다음 teacher signal을 바로 붙인다.
  - CoT sparse top-k / pooled hidden
  - teacher discrete traj labels
  - teacher traj top-k
  - teacher traj hidden
- hard batch에 `cot_content_mask`도 추가했다.
  - 이게 빠져 있어서 첫 smoke에서 `teacher_topk_kd_loss`가 0으로 죽고 있었다.

### Trainer

- [src/training/trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- `teacher_view`가 없어도 hard batch의 `teacher_topk_*`를 써서 CoT sparse KD를 받을 수 있게 했다.
- 기존 `teacher_view` 경로는 그대로 유지하고, hard-batch teacher signal이 있으면 additive하게 더한다.

### Config

- [configs/train/stage_t1_teacher_pair_only.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_teacher_pair_only.yaml)
- 초기 T1 설정:
  - `prompt_mode: joint`
  - `target_mode: joint`
  - `teacher_pair_target: true`
  - `enable_teacher_view: false`
  - `teacher_traj_cache_dir: /data/teacher_cache/traj15`
- 첫 bootstrap은 raw hidden align 재사용을 피하려고 `teacher_traj_hidden_align_loss: 0.0`으로 뒀다.

## Validation

### Unit / Dry Run

- trainer unit test 통과
- data-only dry run에서 hard batch에 다음이 실제로 붙는 걸 확인했다.
  - `teacher_topk_indices`: `(1, 8, 32)`
  - `teacher_traj_labels`
  - `teacher_traj_topk_indices`: `(1, 128, 32)`
  - `teacher_traj_hidden`: `(1, 128, 4096)`

### 1-step smoke

- summary: [smoke_t1_teacher_pair_only_v2.json](/workspace/cosmos_distillation/outputs/reports/smoke_t1_teacher_pair_only_v2.json)
- metrics: [metrics.jsonl](/workspace/cosmos_distillation/outputs/checkpoints/smoke_t1_teacher_pair_only_v2/metrics.jsonl)

Nonzero 확인:

- `teacher_topk_kd_loss = 2.1866`
- `teacher_traj_topk_kd_loss = 1.6922`

즉, CoT sparse KD와 traj sparse KD가 처음으로 같은 honest pair hard branch에서 동시에 살아났다.

## First Train4 Probe

### Train run

- summary: [subset_t1_teacherpair_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_teacherpair_train4_step20.json)
- checkpoint: [final](/workspace/cosmos_distillation/outputs/checkpoints/subset_t1_teacherpair_train4_step20/final)

초기 학습 추세:

- `gt_cot_loss`(실제로는 teacher CoT CE): `4.50 -> 2.13`
- `teacher_topk_kd_loss`: `2.19 -> 0.63`
- `traj_loss`(실제로는 teacher traj CE): `21.0 -> 12.31`

즉 teacher pair imitation loss는 내려간다.

### Decode

- train decode: [subset_t1_teacherpair_train4_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_teacherpair_train4_step20_infer_train.json)
- val decode: [subset_t1_teacherpair_train4_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_teacherpair_train4_step20_infer_val.json)

핵심 결과:

- train 4/4에서 `generated_traj_token_count = 0`
- val 10/10에서 `generated_traj_token_count = 0`
- 모델은 teacher-like CoT 문장을 더 잘 모사하지만, `<|cot_end|><|traj_future_start|>` 이후 traj span으로 넘어가지 못했다.

## Interpretation

이번 결과는 negative라기보다, 병목 위치를 더 선명하게 보여준다.

- `teacher pair provenance`는 더 이상 blocker가 아니다.
- `teacher CoT KD`와 `teacher traj KD`를 같은 sequence에 정렬하는 것도 된다.
- 그런데 현재 `T1` bootstrap은 아직
  - CoT text imitation
  - joint formatting
  - CoT -> traj transition
  사이에서 formatting pressure가 약하다.

즉, honest pair mainline 자체가 틀린 게 아니라, 첫 runnable version은 `pair imitation before traj emission` 단계에 멈춰 있다.

## Next

다음 우선순위는 이 순서가 맞다.

1. `T1 pair-only`에서 format pressure를 올린 변형
2. 필요하면 `CoT tail -> traj start` 전이만 따로 세게 보는 loss
3. 그 다음 `T1.5 bridge`
4. `T2 student-prefix tail KD`

중요한 점:

- 지금은 다시 GT traj CE mainline으로 돌아갈 때가 아니다.
- 이미 `teacher honest pair` 자체는 살아났고, 지금 문제는 provenance보다 `format / transition` 쪽이다.
