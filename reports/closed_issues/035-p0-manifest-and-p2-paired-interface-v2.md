# 035. P0 Manifest And P2 Paired-Interface V2

## Summary

이번 턴에서는 두 가지를 진행했다.

1. `traj15` teacher cache provenance를 repo 안에서 다시 해석 가능한 manifest로 고정했다.
2. `paired-interface-first`의 첫 구현으로 `horizon-bucketed traj aux head + teacher-paired control target + weak GT prefix/final anchor`를 붙였다.

결론은 분명하다.

- `P0 manifest`는 성공했다.
- 하지만 첫 `P2 paired-interface-v2`는 아직 실패다.

정확히는, 이번 v2는 `LM pair-only`보다 더 나은 planner가 되지 못했고, 오히려 auxiliary interface loss가 크게 폭주하면서 formatting도 살리지 못했다.

## P0 Manifest

### Implementation

- script: [20_build_teacher_traj15_manifest.py](/workspace/cosmos_distillation/scripts/20_build_teacher_traj15_manifest.py)
- outputs:
  - [teacher_traj15_manifest.jsonl](/workspace/cosmos_distillation/outputs/reports/teacher_traj15_manifest.jsonl)
  - [teacher_traj15_manifest_summary.json](/workspace/cosmos_distillation/outputs/reports/teacher_traj15_manifest_summary.json)

manifest는 샘플별로 최소한 아래를 남긴다.

- source model / weights / commit / extraction mode / teacher kind
- sample_id
- best candidate index
- best_cot_text
- full completion hash
- teacher traj token ids
- token / top-k / hidden path
- syntax flags
- exact-128 여부
- unique ids / max same-token run
- discrete ADE/FDE
- decoded teacher motion class
- GT motion class
- quality multiplier와 그 basis

### Result

summary:

- `records = 284`
- `syntax_ok_records = 284`
- `exact_128_records = 284`
- `decoded_motion_records = 284`

즉 `traj15`는 이제 repo 관점에서도 “어떤 teacher에서 어떤 extraction mode로 나온 cache인지”를 다시 고정할 수 있다.

## P2 Paired-Interface V2

### Implementation

주요 변경 파일:

- [student_wrapper.py](/workspace/cosmos_distillation/src/model/student_wrapper.py)
- [checkpoint_io.py](/workspace/cosmos_distillation/src/model/checkpoint_io.py)
- [losses.py](/workspace/cosmos_distillation/src/training/losses.py)
- [trainer.py](/workspace/cosmos_distillation/src/training/trainer.py)
- [09_train_distill.py](/workspace/cosmos_distillation/scripts/09_train_distill.py)
- config: [stage_t1_paired_interface_v2.yaml](/workspace/cosmos_distillation/configs/train/stage_t1_paired_interface_v2.yaml)

핵심 변경:

- training-only `traj_aux_head`를 horizon-bucketed head로 확장
  - `num_buckets = 4`
  - output dim = `8`
- hard branch가 `teacher_pair_target=true`일 때, aux head main target은 hard labels를 통해 teacher discrete traj body를 직접 받음
- GT는 aux head 쪽에서만 low-weight anchor로 사용
  - `traj_aux_xyz_loss`
  - `traj_aux_final_loss`
- LM head는 계속 reasoning + format + low-weight traj CE/KD만 유지

### Validation

- unit tests: `8 passed`
- checkpoint manifest now records `traj_aux_num_buckets`

### Train4 / 20-step

train summary:

- [subset_t1_pairifacev2_train4_step20.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2_train4_step20.json)

결과는 좋지 않았다.

주요 수치:

- `teacher_topk_kd_loss`: `3.42 -> 1.86`
- `teacher_traj_topk_kd_loss`: `1.91 -> 1.97` 근처
- `traj_aux_loss`: `10.58 -> 74.64`
- `traj_aux_xyz_loss`: `16.81 -> 21.21`
- `traj_aux_final_loss`: `90.01 -> 88.03`
- total loss도 후반으로 갈수록 다시 커졌다
- grad norm은 수만~십만대로 폭주했다

즉 bucketed aux head가 collapse를 깨기 전에, **aux interface 자체가 불안정하게 발산**했다.

### Decode

- train decode: [subset_t1_pairifacev2_train4_step20_infer_train.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2_train4_step20_infer_train.json)
- val decode: [subset_t1_pairifacev2_train4_step20_infer_val.json](/workspace/cosmos_distillation/outputs/reports/subset_t1_pairifacev2_train4_step20_infer_val.json)

핵심 결과:

- train 4/4: `generated_traj_token_count = 0`
- val 10/10: `generated_traj_token_count = 0`

즉 이번 v2는

- collapse-free traj body를 복구하지 못했고
- `<|cot_end|><|traj_future_start|>` formatting도 살리지 못했다.

생성문도 teacher-like CoT imitation이라기보다 장황한 free-form 설명문이나 이상한 패턴으로 무너졌다.

## Interpretation

이 결과는 `paired-interface-first` 방향이 틀렸다는 뜻은 아니다.
다만 **첫 구현이 너무 naive했다**는 뜻에 가깝다.

현재 v2의 문제는 이쪽으로 해석된다.

- aux head가 unbounded linear output이라 control target에 대해 너무 쉽게 폭주
- GT prefix/final anchor가 aux main loss를 안정화시키기엔 약함
- formatting transition (`cot_end -> traj_start`)은 별도 강화가 아직 없음

즉 이번 결과는

- `teacher pair provenance`는 이제 문제가 아님
- 하지만 `paired interface`는 그냥 bucketed linear head 하나 붙인다고 바로 되는 건 아님

을 보여준다.

## Next

다음은 이 순서가 맞다.

1. aux interface stabilizer 추가
   - bounded output 또는 normalized target space
   - smaller LR / loss weight
   - 필요하면 gradient clip 더 보수적으로
2. formatting / transition pressure 추가
   - `cot_end -> traj_start` 전이 강화
3. 그 다음 다시 paired-interface rerun

즉 지금 상태는:

- `P0 manifest`: 통과
- `P1 pair-LM sanity`: 통과
- `P2 paired-interface v2`: 첫 구현 실패

이다.
