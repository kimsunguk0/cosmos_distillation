# FLEX Pre-Norm Hidden Fix

Date: 2026-06-06

## 결론

FLEX 실패의 1차 원인은 FLEX encoder capacity가 아니라 `DistillStudentModel._forward_flex()`가 공식 no-FLEX 경로와 다른 hidden state를 반환한 버그였다.

- no-FLEX Qwen3VL output의 `hidden_states[-1]`는 final RMSNorm 입력(pre-norm hidden)에 해당한다.
- 기존 FLEX forward는 `language_model.last_hidden_state` 또는 그 기준 hidden을 반환해 post-norm hidden을 action/boundary parity에 사용했다.
- 이 때문에 logits는 어느 정도 맞아도 AE/boundary가 읽는 hidden norm이 teacher 대비 `~0.017x`로 보였다.
- 패치 후 FLEX forward도 final RMSNorm 입력을 hook으로 잡아 `hidden_states[-1]`로 반환한다.

## 핵심 증거

### 기존 실패

FLEX scene token norm은 원본 visual token보다 작지 않았다.

Artifact: `outputs/reports/flex_scale_probe_k896_20260606.json`

| Model | FLEX / visual token norm | action_pre norm ratio | action_pre cosine |
|---|---:|---:|---:|
| F0 untrained | 1.391 | 0.0169 | 0.696 |
| F1 old FLEX-only | 1.882 | 0.0166 | 0.718 |
| F2 old last4 LoRA | 1.442 | 0.0165 | 0.722 |
| F2 old all-LoRA step1000 | 1.429 | 0.0241 | 0.457 |

즉 input embedding scale 문제가 아니었다.

### no-compression manual probe

원본 2880 visual token을 그대로 scatter해도 manual `language_model(inputs_embeds=...)`의 post-norm hidden은 no-FLEX `hidden_states[-1]`를 재현하지 못했다.

Artifact: `outputs/reports/flex_manual_no_compression_probe_20260606.json`

| Boundary | manual / normal norm ratio | cosine |
|---|---:|---:|
| cot_end | 0.107 | 0.932 |
| traj_start | 0.0126 | 0.706 |
| action_pre | 0.0126 | 0.706 |

이후 hook probe로 no-FLEX `hidden_states[-1]`가 final RMSNorm pre-norm hidden과 동일함을 확인했다.

## 수정

File: `src/model/student_wrapper.py`

- FLEX language model forward 중 `conditional.model.language_model.norm`에 forward hook을 걸어 final RMSNorm 입력을 캡처한다.
- logits는 기존처럼 `language_outputs.last_hidden_state`로 계산한다.
- 외부에 반환하는 `outputs.hidden_states`는 pre-norm hidden tuple로 맞춘다.

부가로 overfit 속도 병목 제거:

- `scripts/105_train_flex_teacher_parity.py`: teacher target cache, collated batch cache, LoRA-open 옵션 추가.
- `src/training/flex_batch.py`: position-preserving diagnostic용 `position_ids` 옵션 추가.

## 패치 후 검증

### F0 untrained, 16-sample parity

Artifact: `outputs/reports/flex_f0_after_prenorm_patch_parity16_20260606.json`

| Metric | Value |
|---|---:|
| action_pre cosine | 0.961 |
| action_pre norm ratio | 0.911 |
| traj KL | 0.100 |
| text KL | 0.279 |
| traj top1 agreement | 0.760 |
| teacher top1 in student top5 | 0.974 |

패치만으로 hidden scale collapse는 사라졌다.

### F1 patched FLEX-only 16-sample overfit

Run:

`outputs/checkpoints/flex_f1_parity_overfit16_prenormfix_s2000_k896_20260606/final`

Train artifact:

`outputs/reports/flex_f1_parity_overfit16_prenormfix_s2000_k896_20260606_train_summary.json`

Independent eval artifact:

`outputs/reports/flex_f1_parity_overfit16_prenormfix_s2000_k896_20260606_eval16.json`

| Metric | F0 patched | F1 patched final |
|---|---:|---:|
| action_pre cosine | 0.961 | 0.9969 |
| action_pre norm ratio | 0.911 | 1.0004 |
| traj KL | 0.100 | 0.00429 |
| text KL | 0.279 | 0.00733 |
| traj top1 agreement | 0.760 | 0.934 |
| teacher top1 in student top5 | 0.974 | 1.000 |
| student TF ADE | 0.0966 | 0.0850 |
| teacher TF ADE | 0.0865 | 0.0865 |

## 판단

FLEX-only 16-sample compression parity는 통과. LoRA-open은 아직 필요하다고 결론내릴 수 없다. 다음 판단은 full-val/free-run과 vision sensitivity에서 해야 한다.

Next:

1. patched F1 checkpoint로 vis68/full-val free-run ADE/FDE 측정.
2. normal vs shuffle/black gap 유지 여부 확인.
3. hard bucket에서 traffic light/sign, cross traffic, intersection 손실 확인.
4. 통과하면 K 축소 또는 LoRA-open으로 넘어간다.

