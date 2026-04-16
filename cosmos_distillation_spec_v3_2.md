# Cosmos Distillation Spec v3.2
## Alpamayo 1.5 backbone distillation into Cosmos-Reason2-2B with trajectory-token auxiliary

이 문서는 현재 `/workspace/sukim/cosmos_distillation` 구현, `alpamayo_teacher_prep`에서 준비한 실제 캐시/게이트 산출물, Alpamayo 1.5 공개 계약을 반영한 **실행용 설계서 v3.2**이다.

v3.2는 v3.1의 큰 방향을 유지하되, 실제 준비된 데이터와 게이트 정책에 맞춰 다음을 명확히 고정한다.

- 최종 목표는 text-only reasoner가 아니라, **나중에 frozen VLM 위에 action expert를 붙일 수 있는 backbone**이다.
- 따라서 backbone distillation 단계에서도 **CoT span과 `traj_future` span 둘 다** 학습한다.
- teacher가 현재 안정적으로 제공하는 핵심 soft signal은 **`teacher_long_cot`와 그 token-position top-k logprobs**이다.
- trajectory 쪽 auxiliary는 **teacher trajectory imitation이 아니라 GT future path를 discrete token으로 바꾼 hard target**이다.
- teacher reasoning과 GT trajectory는 **같은 sample의 다른 provenance supervision**으로 병렬 사용하되, **honest joint pair로 취급하지 않는다.**
- `teacher view`는 반드시 **semantic consistency gate**를 거치며, `unknown`과 복합 caution 케이스는 버리지 않고 **더 보수적으로 사용**한다.

---

## 0. v3.2 핵심 규칙

Codex는 아래를 고정 규칙으로 사용한다.

1. `human_refined_coc`는 backbone distillation의 hard text target이다.
2. `teacher_long_cot`는 backbone distillation의 primary soft text target이다.
3. `teacher_topk_logprobs`는 `teacher_long_cot` token positions에만 붙는 primary KD signal이다.
4. `GT future trajectory`는 discrete `traj_future_token_ids`로 변환해 backbone의 auxiliary hard target으로 사용한다.
5. 현재 trajectory 계약은 **`64` waypoints @ `10Hz` -> `128` discrete traj tokens** 이다.
6. backbone은 최소한 아래 train-time 출력 문법을 배워야 한다.

```text
<|cot_start|> ...reasoning... <|cot_end|>
<|traj_future_start|> t1 t2 ... t128 <|traj_future_end|>
```

7. `human_coc`와 `teacher_long_cot`는 같은 CoT span에 동시에 target으로 들어갈 수 없으므로, **hard view / teacher view 2-view 학습**이 필요하다.
8. `teacher_motion = unknown` 샘플은 버리지 않는다.
   - `teacher_long_cot` CE / top-k KD는 사용 가능
   - `GT traj token` CE도 사용 가능
   - 단, **action/motion auxiliary는 off**다.
9. 남은 **directional hard-fail** 샘플은 teacher view를 과감히 제거한다.
10. `감속 vs 회피`, `보행자 양보 vs 신호 정지` 같은 **복합 caution conflict**는 hard-fail로 버리지 않고 `soft_fail + action_aux off`로 낮출 수 있다.
11. `teacher_answer`, `structured_json`, direct `teacher_meta_action`, feature KD는 baseline off다.
12. pooled hidden은 저장만 하고 `feat_align=0`으로 둔다.

---

## 1. 목표와 비목표

### 1-1. 최종 목표
- **Teacher reasoning backbone**: Alpamayo 1.5 VLM reasoning path
- **Student backbone**: `nvidia/Cosmos-Reason2-2B`
- **데이터**: PhysicalAI AV strict human-CoC subset 우선
- **학습 목표**:
  - human CoC hard supervision
  - teacher long-CoT soft supervision
  - teacher CoT top-k KD
  - GT future trajectory token auxiliary supervision
- **후속 연결 목표**:
  - v3.2에서 학습된 frozen VLM backbone 위에
  - 후속 stage에서 action expert를 붙여
  - full trajectory generation model로 확장

### 1-2. 비목표
- teacher reasoning + GT trajectory를 soft joint pair처럼 취급하는 것
- answer / structured_json / direct meta-action을 baseline soft target으로 적극 사용하는 것
- pooled hidden만으로 feature KD를 켜는 것
- teacher continuous trajectory imitation을 baseline main loss에 포함하는 것

---

## 2. 현재 모델/계약에 대한 고정 사실

### 2-1. Student
- `Cosmos-Reason2-2B`는 이미 physical AI / AV 도메인에 post-train된 VLM이다.
- 따라서 student는 AV-naive model이 아니라, **AV-aware backbone**으로 취급한다.
- 이번 distillation은 AV 자체를 처음 배우는 과정이 아니라, **teacher의 더 정제된 reasoning과 planning interface를 압축**하는 과정이다.

### 2-2. Teacher
- Alpamayo 1.5 public helper path에서 가장 안정적인 teacher text는 **reasoning route의 `teacher_long_cot`** 이다.
- 현재 실측 기준:
  - `teacher_long_cot`: usable
  - `teacher_topk_logprobs`: usable
  - `teacher_answer`: canonical subset에서 collapse가 커서 baseline off
  - direct `teacher_meta_action`: baseline teacher contract에서 제외
  - structured JSON: baseline off

### 2-3. Full model 방향
- 최종 full model은 **frozen VLM + action expert**를 목표로 한다.
- 따라서 backbone distillation 단계에서도 student는:
  - CoT span 생성
  - `traj_future` span 생성
  - trajectory special token grammar
  - planning-aware hidden / representation
  을 익혀야 한다.

---

## 3. strict subset 원칙

### 3-1. 기본 데이터
기본값은 **strict human-CoC subset**이다.

이유:
- backbone distillation의 hard text target이 `human_refined_coc`이기 때문
- teacher soft target이 불안정한 상황에서 human hard supervision 품질이 우선이기 때문

### 3-2. current prep snapshot
현재 준비된 스냅샷 기준:
- manifest: `287`
- materialized ready: `285`
- teacher text/top-k ready: `285`
- GT traj target ready: `284`
- 실제 학습용 whitelist: `284`

실제 학습에서 whitelist 기준은 `transfer_ready_samples.json`을 따른다.

---

## 4. supervision provenance 정의

모든 supervision은 provenance를 분리해서 저장해야 한다.

### A. Hard Human
- `human_refined_coc`
- `clip_uuid`
- `event_cluster`
- `keyframes`
- `ego_history`
- observed `ego_future`

### B. Weak Derived from GT / Human
- `gt_motion_semantics`
- `turn_direction_from_gt_path`
- `stop_profile_from_gt_path`
- `critical_objects_from_human_parser`
- `parsed_meta_action_from_human`

### C. Soft Teacher Reasoning
- `teacher_long_cot`
- `teacher_topk_logprobs`
- `teacher_pooled_hidden`
- prompt family / parse status / quality flags

### D. Derived Teacher Auxiliary
- `parsed_meta_action_from_teacher_cot`
- `teacher_motion_semantics_from_cot`
- `teacher_intent_tags_from_cot`

중요:
- D는 direct teacher slot이 아니라 **CoT 기반 derived signal**이다.
- baseline에서는 D를 main supervision이 아니라 auxiliary / diagnostics로만 사용한다.

---

## 5. backbone output contract

student는 backbone distillation 단계에서 최소한 아래 sequence grammar를 배워야 한다.

```text
<|cot_start|> ...reasoning... <|cot_end|>
<|traj_future_start|> t1 t2 ... t128 <|traj_future_end|>
```

선택적으로 추후 아래를 확장할 수 있다.

```text
<|meta_action_start|> ...coarse action... <|meta_action_end|>
```

하지만 baseline v3.2에서는 meta-action span은 **필수 생성 target이 아니다.**

### 5-1. 왜 traj span이 필요한가
나중에 action expert를 붙일 때 VLM은 frozen된다.
그러므로 frozen backbone은 최소한:
- `traj_future` token span을 이해하고
- 그 span을 생성해본 경험이 있고
- trajectory planning 관련 hidden을 형성할 수 있어야 한다

즉 v3.2는 pure text-only backbone distill이 아니라,
**reasoning-main + trajectory-interface-aware backbone distill**이다.

---

## 6. trajectory tokenization contract

### 6-1. GT future -> discrete token 변환
observed `ego_future`는 discrete `traj_future_token_ids`로 변환해야 한다.

필수 고정 항목:
- coordinate frame: `local@t0`
- horizon: `64` waypoints @ `10Hz`
- token count: **`128`**
- tokenization version
- binning / discretization rule
- special tokens

### 6-2. 저장할 것
- `traj_future_token_ids`
- `traj_tokenizer_version`
- `traj_token_length = 128`
- `traj_waypoint_length = 64`
- `traj_frame = local@t0`
- optional decode helper metadata

### 6-3. 주의
- 이 trajectory token은 teacher가 낸 soft token이 아니라 **GT-derived hard auxiliary target**이다.
- 따라서 teacher reasoning과 같은 provenance로 취급하면 안 된다.

---

## 7. teacher contract v3.2

### 7-1. Primary teacher
- `teacher_long_cot`
- direct CoT route에서 얻은 non-empty reasoning
- trust level: high

### 7-2. Primary KD signal
- `teacher_topk_logprobs`
- `teacher_long_cot` token positions에 대해서만 저장
- default: top-32 sparse logprobs

### 7-3. Stored but not used as main loss
- `teacher_pooled_hidden`
- baseline에서는 cache / probe only

### 7-4. Auxiliary / diagnostics only
- `parsed_meta_action_from_teacher_cot`
- `teacher_motion_semantics_from_cot`
- `teacher_intent_tags_from_cot`

### 7-5. Disabled in baseline
- direct `teacher_answer`
- `teacher_structured_json`
- direct `teacher_meta_action`

---

## 8. semantics extraction 원칙

### 8-1. GT 쪽: motion primitive 중심
GT future path에서 뽑는 메인 class:
- `lane_keep`
- `slow_down`
- `stop`
- `creep`
- `creep_then_go`
- `left_turn`
- `right_turn`
- `change_lane_left`
- `change_lane_right`
- `nudge_left`
- `nudge_right`
- `unknown`

GT만으로 `yield`, `follow_lead`, `blocked_wait` 같은 intent/context를 강하게 확정하려 하지 않는다.

### 8-2. Teacher 쪽: motion + intent/context
Teacher CoT parser 출력:
- `teacher_motion_class`
- `teacher_motion_confidence`
- `teacher_intent_tags`
- `teacher_evidence_spans`

예:
- `blocked_on_right` + `move slightly left` -> `nudge_left`
- `yield before turning left` -> `left_turn` + intent `yield`
- `keep distance ...` -> `keep_distance` intent tag

### 8-3. 왜 이렇게 나누나
- GT는 물리 motion anchor
- Teacher는 reasoning-derived motion + intent
- gate는 `motion hard gate + intent soft gate`

이 분리가 현재 parser failure mode에 가장 잘 맞는다.

---

## 9. teacher CoT vs GT traj consistency gate

이 gate는 v3.2에서 **가장 중요한 안전장치**다.

### 9-1. 목적
`teacher view`에서
- CoT span = teacher
- traj span = GT
를 같이 사용할 수 있는지 결정한다.

### 9-2. 왜 필요한가
`teacher_long_cot`와 `GT future`는 같은 sample에는 속할 수 있지만,
**같은 provenance의 joint pair는 아니다.**

gate 없이 둘을 함께 학습시키면 backbone이
- reasoning A
- trajectory B
를 동시에 배우는 위험이 있다.

### 9-3. 판정 구조
#### Motion gate
- `pass`
- `soft_pass`
- `soft_fail`
- `hard_fail`
- `unknown`

#### Intent gate
- `pass`
- `soft_pass`
- `soft_fail`
- `unknown`

### 9-4. 기본 규칙
- motion `hard_fail` -> teacher view 제거
- motion `soft_fail` -> 강한 downweight
- motion `unknown` -> low weight
- motion `pass/soft_pass` -> teacher view 허용
- intent는 weighting과 diagnostics에만 사용

### 9-5. v3.2 추가 규칙

#### A. unknown-motion teacher 처리
`teacher_motion = unknown`이면:
- `teacher_long_cot` CE / top-k KD: 사용 가능
- `GT traj token` CE: 사용 가능
- `action/motion/meta auxiliary`: **off**

즉 unknown은 teacher text view는 유지하되, action auxiliary는 쓰지 않는다.

#### B. directional hard-fail hard drop
남은 `nudge_left <-> nudge_right` 방향 충돌 hard-fail 버킷은
**teacher view를 과감히 제거**한다.

이 버킷은 sample-level drop list로 관리할 수 있다.

#### C. complementary caution bridge
다음 같은 케이스는 hard-fail로 버리지 않고 `soft_fail + action_aux off`로 낮출 수 있다.

- teacher는 lateral avoidance (`nudge`, `change_lane`)를 말하고
- human은 longitudinal caution (`slow_down`, `stop`)를 말하며
- GT lateral class confidence가 낮은 경우

또는

- teacher는 `stop/slow_down`
- human은 `yield/pedestrian_crossing/cross_traffic/oncoming_traffic`
- GT는 low-confidence `lane_keep`

이런 경우는 이유나 축이 다를 뿐 실제로는 **동시에 맞을 수 있는 caution behavior**다.

### 9-6. absolute prohibition
`teacher_long_cot + GT traj`를
**같은 provenance의 honest joint pair**처럼 취급하지 않는다.

허용되는 것은 같은 sample 안의 **different-span, different-provenance supervision**뿐이다.

---

## 10. 2-view training design

v3.2에서는 최소 **2개 학습 view**가 필요하다.

### 10-1. Hard view
```text
<|cot_start|> human_coc <|cot_end|><|traj_future_start|> gt_traj_tokens <|traj_future_end|>
```

역할:
- 최종 출력 문법 학습
- GT 행동 학습
- backbone이 CoT span과 trajectory span을 동시에 생성하는 법을 학습

Loss:
- CoT span -> hard CE (`human_coc`)
- Traj span -> hard CE (`gt_traj_tokens`)

### 10-2. Teacher view
```text
<|cot_start|> teacher_long_cot <|cot_end|><|traj_future_start|> gt_traj_tokens <|traj_future_end|>
```

역할:
- teacher reasoning compression
- teacher distribution KD
- planning span 문법 유지

Loss:
- CoT span -> seq pseudo CE (`teacher_long_cot`)
- CoT span -> sparse top-k KD (`teacher_topk_logprobs`)
- Traj span -> hard CE (`gt_traj_tokens`)

조건:
- `teacher_long_cot` non-empty
- quality gate 통과
- semantic consistency gate 통과

### 10-3. teacher view 세부 정책
- `pass / soft_pass`: 정상 사용
- `soft_fail`: 강하게 downweight
- `unknown`: low-weight로 사용, action aux off
- `hard_fail`: teacher view drop
- `complementary caution bridge`: `soft_fail + action aux off`

### 10-4. 중요한 점
- `human_coc`와 `teacher_long_cot`는 같은 CoT span에 동시에 target으로 들어갈 수 없다.
- `teacher_topk`는 `teacher_long_cot` positions에만 붙는다.
- 따라서 2-view는 필수에 가깝다.

---

## 11. corpus / loader contract

결과적으로 batch에는 아래가 들어와야 한다.

- `input_ids`
- `attention_mask`
- `labels`
- `cot_span_mask`
- `traj_span_mask`
- `teacher_topk_positions`
- `teacher_topk_ids`
- `teacher_topk_logprobs`
- `sample_provenance_flags`
- `teacher_quality_multiplier`
- `teacher_vs_gt_consistency_level`
- `teacher_view_allowed`
- `teacher_view_weight`
- `action_aux_allowed`
- `action_aux_weight`
- image tensors / image paths
- `ego_history`

구현은 아래 둘 중 하나면 된다.
- pre-packed corpus file
- materialized sample + teacher cache + gate/state files를 loader-time join

중요한 것은 **학습 step에서 span별 supervision이 정렬되어 batch에 존재하느냐**다.

---

## 12. loss 설계 v3.2

### 12-1. total loss
```text
L_total =
  λ1 * L_hard_ce_human_cot
+ λ2 * L_seq_pseudo_ce_teacher_cot
+ λ3 * L_logit_kd_teacher_cot
+ λ4 * L_traj_token_ce_gt
+ λ5 * L_meta_aux
+ λ6 * L_feat_align
```

### 12-2. 기본 권장값
- `L_hard_ce_human_cot` = `1.0`
- `L_seq_pseudo_ce_teacher_cot` = `0.20 ~ 0.30`
- `L_logit_kd_teacher_cot` = `0.30 ~ 0.50`
- `L_traj_token_ce_gt` = `0.10 ~ 0.25`
- `L_meta_aux` = `0.03 ~ 0.08`
- `L_feat_align` = `0.0`

### 12-3. 해석
- reasoning은 teacher-guided
- trajectory token은 GT-guided
- 둘을 같은 sample 안의 다른 span에 병렬 적용
- 단, teacher view는 consistency-gated

### 12-4. action auxiliary 규칙
- `action_aux_allowed = false`이면 action/motion/meta auxiliary loss는 **0**
- 특히:
  - teacher motion `unknown`
  - remaining teacher hard-fail
  - complementary caution bridge
  에서는 action aux를 끈다

### 12-5. baseline off
- `teacher_answer_aux`
- `structured_json_aux`
- pooled hidden feature KD

---

## 13. hidden / top-k 사용 원칙

### 13-1. top-k
- baseline에서 적극 사용
- `teacher_long_cot` token positions에 대해서만 사용
- 실제 값이 `logits`가 아니라 `logprobs`일 수 있으므로 저장 이름을 명확히 한다

### 13-2. hidden
- baseline에서는 저장만 한다
- current pooled hidden `(D,)`는 probe / diagnostics에만 사용
- per-token hidden 또는 selected-layer hidden contract가 생기기 전까지 `feat_align=0`

### 13-3. student hidden export
나중 action expert를 위해 student는 최소한 다음을 export할 수 있어야 한다.
- reasoning span hidden
- traj span hidden
- corresponding masks / indices

이것은 teacher hidden KD와는 다른 요구사항이다.

---

## 14. meta-action 사용 원칙

### 14-1. baseline 정의
baseline에서 meta-action은 direct teacher slot이 아니다.

### 14-2. 사용 방식
- `teacher_long_cot` parser에서 coarse action 추출
- weak GT와 hard-fail이 아니고
- parser confidence가 높을 때만 auxiliary로 사용

### 14-3. naming
- `parsed_meta_action_from_teacher_cot`
- `teacher_motion_semantics_from_cot`

`teacher_meta_action`이라는 이름을 direct teacher slot처럼 과장해서 쓰지 않는다.

---

## 15. prompt families v3.2

### 15-1. Official CoT route
목표:
- `teacher_long_cot`

### 15-2. Optional answer probe
목표:
- diagnostics only
- baseline supervision off

### 15-3. Meta-action probe
목표:
- smoke-only or research-only
- baseline main loss off

### 15-4. Structured JSON
- baseline disabled

---

## 16. implementation priorities

### P0 — must before retrain
1. `traj_future_token_ids`를 loader/train batch에 연결
2. hard view / teacher view 2-view corpus wiring 구현
3. CoT span / traj span mask 구현
4. teacher top-k와 teacher CoT position alignment 구현
5. semantic consistency gate를 teacher view filter에 연결
6. `teacher_view_hard_drop`와 `action_aux_allowed`를 loader에서 반영

### P1 — strongly recommended
7. GT motion semantics extractor 고도화
8. teacher CoT parser에서 motion + intent 추출
9. `nudge_left`, `nudge_right`, `blocked_on_right/left`, `keep_distance`, `yield` 계열 규칙 유지
10. answer / structured / meta probe는 baseline off 상태로 diagnostics만 유지

### P2 — later
11. per-token hidden / selected-layer hidden cache
12. projector / bridge for future action expert
13. action expert interface spec

---

## 17. stage schedule

### Stage A — grammar stabilization
- hard view 중심
- 목적: backbone이 CoT span + traj span 문법을 익힘

권장:
- `human_coc + gt_traj_tokens`
- teacher view 비중 낮게

### Stage B — main distillation
- hard view + consistency-filtered teacher view
- 목적: teacher reasoning distribution 압축 + trajectory span 유지

권장:
- `hard_ce_human_cot`
- `seq_pseudo_ce_teacher_cot`
- `logit_kd_teacher_cot`
- `traj_token_ce_gt`

### Stage C — optional auxiliary polish
- parsed meta-action trusted subset만 약하게 사용
- intent-aware diagnostics / ranking / anti-generic polish

---

## 18. evaluation

### 18-1. Hard-human anchored evaluation
- human CoC overlap / rubric
- hallucination rate
- reasoning consistency

### 18-2. Distillation fidelity
- teacher CoT token NLL
- sparse top-k agreement
- teacher retention score

### 18-3. Planning interface
- GT traj token accuracy / CE
- trajectory span grammar validity

### 18-4. Groundedness
- `teacher CoT vs GT traj` consistency statistics
- `student CoT vs GT traj` consistency statistics
- motion hard-fail rate

### 18-5. Efficiency
- latency
- VRAM
- throughput

---

## 19. Definition of Done

v3.2 최소 성공 조건:

1. strict human-CoC subset 기준 hard/soft provenance 분리가 된다.
2. hard view와 consistency-filtered teacher view가 둘 다 생성된다.
3. `teacher_long_cot` + `teacher_topk_logprobs`가 CoT span에 정렬된다.
4. `GT traj_future_token_ids`가 traj span에 정렬된다.
5. `teacher CoT vs GT traj semantics consistency gate`가 teacher view 사용 여부를 제어한다.
6. student는 CoT span과 traj span을 모두 생성한다.
7. baseline에서 answer / structured / direct meta slot은 꺼져 있다.
8. `logit_kd`는 nonzero, `traj_token_ce`는 nonzero, checkpoint saving은 동작한다.
9. pooled hidden은 저장되지만 `feat_align=0`이다.
10. `unknown`과 complementary caution 샘플에서 action auxiliary가 꺼진다.
11. remaining directional hard-fail teacher view drop policy가 문서와 구현에 모두 반영된다.

---

## 20. 최종 요약

v3.2의 핵심은 다음 한 문장이다.

> **backbone distillation은 text-only reasoning distillation이 아니라, `human CoC + teacher long-CoT/top-k + GT future traj token auxiliary`를 사용하는 reasoning-main / trajectory-interface-aware distillation이다. teacher CoT와 GT trajectory는 같은 sample의 서로 다른 provenance supervision으로 병렬 사용하되, teacher view는 반드시 semantic consistency gate를 통과한 샘플만 허용한다. `unknown`과 복합 caution 샘플은 teacher text는 살리되 action auxiliary는 끄고, 남은 directional hard-fail은 teacher view를 드랍한다.**
