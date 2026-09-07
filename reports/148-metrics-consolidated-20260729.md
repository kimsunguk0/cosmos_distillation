# Step-B 지표 종합 (2026-07-29)

Cosmos-Reason2-2B ← Alpamayo-1.5-10B 증류. 지금까지 **실제로 계산해서 파일로 남아 있는 수치**만 모았습니다.
추정치·기억에 의존한 값은 §9에 따로 표시했습니다.

---

## 0. 읽기 전 — 기준(reference)과 차원(dimensionality)

세 가지가 섞이기 쉬우니 표마다 명시합니다.

| 축 | 값 | 의미 |
|---|---|---|
| **reference** | `teacher_greedy` | student를 **teacher가 낸 경로**와 비교 (증류 충실도) |
| | `gt` | student를 **실제 주행 GT**와 비교 (절대 성능) |
| **차원** | 2D (xy) | `scripts/25_decode_checkpoint_overlays.py`가 리포트하는 `ade_m` |
| | 3D (xyz) | R1 평가 스크립트 및 본 문서의 재계산 |
| **샘플셋** | val512 | `data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl` |
| | 384 | 위 512 중 `t0_us ≥ 4.8s` (R1 로더 제약). 교차모델 비교는 전부 이 셋 |

동일 데이터에서 2D/3D 차이 실측: greedy 3.309(2D) vs 3.360(3D), minADE6 1.663(2D) vs 1.739(3D).

**표준 지표 정의**
- **ADE** = 단일 경로 greedy(n=1). 배포 조건.
- **minADE6@6.4s** = 6개 샘플링 중 최선. 오라클(정답으로 고름) 지표.

---

## 1. Backbone 이산 디코드 — val512, greedy n=1

`outputs/reports/backbone_verify_spr1greedy_20260727/`, 2D, n=512.

### teacher_greedy 기준 (증류 충실도)

| backbone | ADE | FDE | unique | max_run | token match |
|---|---|---|---|---|---|
| 200K CE | 2.0319 | 6.5175 | 22.68 | 3.90 | 0.0774 |
| 200K CE+KD | 2.0677 | 6.6989 | 21.63 | 3.54 | 0.0750 |
| 200K LoRA | 2.0546 | 6.5128 | 20.31 | 1.46 | 0.0777 |
| **444K CE** | **1.8556** | **5.9826** | 23.65 | 3.17 | 0.0818 |

200K 세 방법은 서로 tie (2.03~2.07). 데이터 2.2배(200K→444K)가 유일하게 움직인 레버: −0.18.

### GT 기준

| backbone | ADE | FDE |
|---|---|---|
| 444K CE | 3.0444 | 9.6845 |

teacher↔GT 간격 자체가 커서 두 기준의 절대값은 직접 비교 불가.

---

## 2. Backbone KV 표현 — hidden→action R²

`scripts/112_extract_hidden_kv_repr_compare.py --compute-hidden-action-r2`
(feature = traj_body_128 구간 final hidden 평균 → ridge α=1.0 → teacher pred_xyz [64,3]=192dim, 512행 중 384 train / 128 val)

| backbone | R² | MSE | 산출 파일 |
|---|---|---|---|
| 200K CE | 0.6842 | 55.86 | `kv_probe_444k_ce_20260722/hidden_kv_repr_compare_val512.json` |
| 200K CE+KD | 0.7438 | 45.31 | 〃 |
| 200K LoRA | **0.7931** | 36.59 | 〃 |
| **444K CE** | 0.6868 | 55.39 | 〃 |
| 200K LoRA+KL | 0.7928 | 36.65 | `kv_probe_lorakd_200k_20260724/lorakd_vs_lorace_val512.json` |
| 444K LoRA+hidden-align(HGC) | 0.7294 | 47.86 | `kv_probe_hidden_gc_20260726/hidden_gc_probe.json` |

**KL×method 2×2**

| | CE-only | +KL | ΔKL |
|---|---|---|---|
| FullFT | 0.684 | 0.744 | +0.060 |
| LoRA | 0.793 | 0.793 | **+0.000** |

- LoRA와 KL은 **중복**. LoRA 단독(0.793)이 FullFT+KL(0.744)보다 이미 높음.
- **데이터는 R²를 안 올림**: 200K CE 0.684 → 444K CE 0.687 (2.2배 데이터, flat).
- hidden-align은 **역효과**: 0.729 < LoRA-CE 0.793.

→ **핵심 해리(dissociation): 데이터는 이산 토큰 경로를 개선하고(2.03→1.86), 방법(LoRA/KD)은 KV 기하를 개선한다. 서로 넘어가지 않는다.**

---

## 3. 데이터 증가분은 어디로 갔나 — weight-delta 200K↔444K CE

`outputs/reports/weight_delta_200k_vs_444k_20260725/weight_delta.json` (student-vs-student, 동일 stepa init)

| 버킷 | rel_step ‖ΔW‖/‖W‖ | 전체 변화 에너지 점유 |
|---|---|---|
| embed_tokens | 0.2641 | 33.7% |
| lm_head (embed와 tied, 동일) | 0.2641 | 33.7% |
| transformer_body | 0.0599 | 30.9% |
| mm_projector | 0.0223 | 1.7% |
| final_norm | 0.0114 | 0.0% |

readout(tied embed/lm_head)이 body보다 상대적으로 4.4배 움직임. late layer 24-27은 최소 이동(0.03~0.05) — R²/AE가 읽는 바로 그 층.

**per-row 정밀화** (`perrow_embed_groups.json`, vocab 155685)

| 행 그룹 | rows | mean_row_rel | 에너지 점유 |
|---|---|---|---|
| language BPE | 151,637 | 0.2728 | **99.08%** |
| traj_bin (`<i0>`..`<i3999>`) | 4,000 | **0.5543** | 0.90% |
| marker | 48 | 0.4754 | 0.02% |

→ readout의 벌크 Frobenius 변화는 **99%가 언어 행의 확산적 drift**(CE가 궤적 위치마다 비-traj 로짓을 억제 + weight decay 누적). 기능적으로 ADE에 관여하는 건 traj_bin 행(per-row 최대 변화)과 early-mid layer. 최종 hidden(AE 채널)은 그대로.

---

## 4. 교차 모델 4-way — 384, GT 기준, 3D, 단일 경로

`outputs/reports/r1_discrete_backbone_20260728/threeway_384.json`

| 모델 | ADE ≤2s | ADE >2s | ADE full | FDE@2s | FDE@6.4s |
|---|---|---|---|---|---|
| Teacher 1.5 discrete | 0.205 | 3.057 | 2.166 | 0.591 | 6.309 |
| **R1 (1.0) discrete** | **0.190** | **2.971** | **2.102** | **0.548** | **6.244** |
| R1 (1.0) continuous (diffusion) | 0.262 | 3.718 | 2.638 | 0.755 | 7.470 |
| 444K CE student discrete | 0.252 | 4.772 | 3.360 | 0.751 | 10.529 |

Paired Δ (mean ± SE, n=384)

| 비교 | ADE >2s | ADE full | FDE@6.4s |
|---|---|---|---|
| R1 disc − Teacher 1.5 disc | −0.087 ± 0.136 | **−0.064 ± 0.095 (null)** | −0.065 ± 0.336 |
| Student − Teacher 1.5 disc | +1.715 ± 0.201 | +1.194 ± 0.140 | +4.220 ± 0.465 |
| Student − R1 disc | +1.802 ± 0.198 | +1.258 ± 0.138 | +4.285 ± 0.462 |
| R1 disc − R1 continuous | −0.747 ± 0.132 | **−0.536 ± 0.092** | −1.225 ± 0.288 |

R1 디코드 건전성: `traj_start_hit_rate` 1.0, `full_valid_block_rate` 1.0, 실패 0/384, 소요 9,763초.

**중요한 뒤집힘:** 연속 경로로 비교했을 땐 1.5가 far-horizon에서 0.392 ± 0.153 앞섰지만, 동일 discrete 디코더로 맞추니 **null**. nav/GRPO 없는 1.0 백본의 토큰 예측 능력이 1.5와 동급 → **"no-nav 천장"은 없다.** student의 1.2m 격차를 nav 데이터 부재로 설명할 수 없음.

부수 관찰: R1의 diffusion expert가 자기 discrete head보다 나쁨(−0.536 ± 0.092). 우리 AE의 단일경로 병목이 공개 모델에서도 재현.

---

## 5. 444K CE 격차의 구조 (384, 6.4s 종점, ego 프레임 +x 전방 / +y 좌측)

### 5.1 student = teacher + **독립** 잡음

```
|teacher − GT|        6.16 m
|student − GT|       10.42 m
|student − teacher|   7.94 m
cos(student−teacher, teacher−GT) = +0.034   (median +0.109, P(cos>0)=51.3%)
RMS |student−GT| 실측 14.87 m   vs 독립합 예측 15.56 m
```
teacher의 오차를 증폭하는 게 아니라 직교한 자기 오차를 얹음 → **편향이 아니라 분산 문제.**

### 5.2 속도는 정상, 곡률만 붕괴

| | teacher | student |
|---|---|---|
| 전방 변위 / GT (중앙값, 이동 장면 n=367) | 1.001 | 0.992 |
| 경로 길이 / GT (중앙값) | 0.999 | 0.999 |
| 회전 장면(\|GT 횡변위\|>3m, n=162) 횡변위 중앙값 | 12.39 (GT 13.30의 93%) | 9.79 (**79%**) |
| 회전 "방향 정확" | 97.5% | 87.0% |

방향 오류 21/162의 실체는 **반대로 도는 게 아니라 직진해버리는 것**(횡변위 +0.0~+0.6m). 이 5.5%의 장면이 전체 횡방향 제곱오차의 **39%**.

bias/scatter 분해: bias는 평균제곱오차의 2.2%만 설명 → 대부분 scatter.

### 5.3 병리적 디코드는 주범이 아님

| | n | 격차 점유율 | 평균 격차 |
|---|---|---|---|
| speed collapse (path_len ratio<0.5) | 37 (9.6%) | 12.5% | +1.81 |
| low diversity (uniq<8) | 121 (31.5%) | 26.4% | +1.10 |
| repetition (max_run≥6) | 11 (2.9%) | 2.9% | +1.44 |
| 위 중 하나라도 해당 | 152 (39.6%) | 36.1% | +1.22 |
| **정상 디코드** | **232 (60.4%)** | **63.9%** | **+1.28** |

정상 디코드 장면이 격차의 64%를 지고 평균 격차도 동일 → 소수 붕괴가 아니라 **전 장면에 고르게 깔린 원거리 오차**.

### 5.4 오차 증폭비 (FDE@6.4s / FDE@2s)

teacher 10.7× / R1 11.4× / **student 14.0×** — 초선형. 곡률 편향은 횡변위에 t²로 들어감.

### 5.5 속도 구간별

| GT 전방 변위 | n | \|e_teacher\| | \|e_student\| | 상대 격차 |
|---|---|---|---|---|
| −18.4 ~ 27.3 m | 96 | 6.73 | 11.14 | 1.65× |
| 27.3 ~ 49.5 m | 96 | 7.32 | 12.62 | 1.72× |
| 49.5 ~ 80.2 m | 96 | 5.20 | 8.48 | 1.63× |
| 80.2 ~ 198.3 m | 96 | 5.38 | 9.46 | 1.76× |

속도 무관하게 균일 1.6~1.8배.

---

## 6. 디코드 ablation (2026-07-28~29, 384, GT 기준)

`outputs/reports/decode_probe_20260728/`. 표의 ADE는 3D 재계산.

| 런 | ADE | 중앙값 | ≤2s | >2s | FDE | uniq | 회전 방향 | 회전 크기 | 직진해버림 |
|---|---|---|---|---|---|---|---|---|---|
| baseline (자기 CoC, joint decode) | 3.360 | 2.218 | 0.252 | 4.772 | 10.529 | 24.9 | 87.0% | 0.91 | 17.3% |
| D — teacher CoC 접두 + 궤적만 자유 | 3.299 | 2.208 | 0.249 | 4.685 | 10.343 | 24.9 | 89.5% | 0.91 | 17.9% |
| C0 — CoC 없음 + 궤적만 자유 | 3.297 | 2.243 | 0.257 | 4.678 | 10.317 | 28.1 | 89.5% | 0.95 | 14.2% |
| B1 — student CoC 접두 + greedy (대조군) | 3.367 | 2.208 | 0.252 | 4.782 | 10.541 | 25.5 | 87.7% | 0.89 | 18.5% |
| B8 — student CoC 접두 + **beam 8** | 3.500 | 2.503 | 0.253 | 4.976 | 11.145 | 14.5 | 87.0% | 0.93 | 13.6% |
| **M6 — 6샘플 best-of-6 (오라클)** | **1.739** | — | **0.211** | **2.434** | **5.036** | — | — | — | — |

Paired vs baseline

```
D  (teacher CoC)  −0.061 ± 0.062    better 123/384
C0 (CoC 없음)      −0.063 ± 0.076    better 164/384
B1 (대조군)        +0.007 ± 0.027    better  99/384      ← baseline과 동일, 비교 성립
B8 (beam 8)       +0.140 ± 0.056    better 177/384
B8 − B1 (동일 접두) +0.138 ± 0.055   중앙값 델타 +0.000   악화분의 43%가 최악 5%에서
```

### 6.1 CoC는 궤적에 인과적으로 관여하지 않음

**먼저 무엇을 ablate 했는지 확정.** 이 텍스트는 Alpamayo의 **Chain of Causation (CoC)** 이 맞다. 축약된 CoT나 열화된 캡션이 아니라, 설계상 짧은 `<결정> + <인과절>` 구조다.

```
hard_target.cot_text == teacher_target.cot_text        512/512 완전 동일 (별도 human 주석 없음)
provenance.hard_text = "alpamayo15_no_nav_teacher_cot"
인과 접속사(since / due to / because / to+동사) 포함     472/512 (92.2%)
길이 평균 13.2단어 · 중앙값 12 · p90 16 · target_token_count 11 · 고유 문자열 223/512

[Stop]                             || [due to the stop sign controlling our lane]
[Keep lane]                        || [since the lane is clear ahead]
[Slow down for the red traffic light] || [since it is red]
[Nudge left]                       || [to increase clearance to the stopped vehicle]
```

**결과.** teacher의 올바른 CoC를 줘도, 통째로 없애도 오차범위 내 동일 (−0.061 ± 0.062 / −0.063 ± 0.076).
단 D와 C0는 샘플당 평균 **0.84 m** 다르고 상관 0.205 → 궤적 헤드가 CoC를 **읽긴 하지만 유용한 정보를 못 뽑음** (변화는 신호가 아니라 잡음).

CoC 내용이 가장 크게 갈리는 곳에서도 null:

| 버킷 | n | base ADE | Δ D−base | Δ C0−base |
|---|---|---|---|---|
| CoC 완전 동일 | 155 | 2.833 | −0.013 ± 0.040 | −0.130 ± 0.119 |
| 기동 단어 동일 | 78 | 3.154 | −0.092 ± 0.079 | −0.050 ± 0.120 |
| **기동 단어 다름** | 151 | 3.879 | −0.091 ± 0.147 | +0.007 ± 0.138 |

**가장 날카로운 검증 — 기동 클래스 실패 장면.** CoC가 결정하는 건 미터 단위 경로가 아니라 기동 클래스이므로, 기동을 틀린 곳에서 올바른 CoC를 주면 복구되어야 한다. GT가 3 m 이상 꺾는데 baseline student가 직진해버린 **28개** 장면:

```
teacher CoC 투입 시 기동 복구:  1 / 28
CoC 제거 시:                   2 / 28        ← CoC 없이도 같은 수준
ADE:  base 7.001 → D 6.533 (−0.468 ± 0.448) | C0 6.662 (−0.339 ± 0.367)

예: GT −14.1 m 좌회전 → student −0.02 m (teacher CoC 투입 후에도 직진)
    GT −27.7 m        → −0.15 m
    GT −59.1 m        → −0.02 m
```

**해석.** 우리는 CoC를 제대로 증류했는데도 궤적이 CoC를 인과적으로 사용하지 않는다. 남는 설명은 **SFT가 인과 결합을 만들지 않는다**는 것: teacher forcing으로 (CoC, 궤적) 쌍을 학습하면 궤적 토큰은 항상 정답 CoC를 보면서 학습되고, 이미지만으로 궤적을 맞출 수 있는 한 CoC를 참조하라는 gradient가 발생하지 않는다. 둘은 공유 표현에서 갈라진 **조건부 독립 헤드**로 수렴하고, CoC는 결정의 *산출물*이지 *입력*이 아니게 된다.

→ outcome 기반 RL(GRPO)은 결합 위에 얹는 마감이 아니라 **결합을 만드는 메커니즘**이다. 최종 궤적에 보상을 걸어야 "더 나은 궤적으로 이어지는 CoC"가 강화되며 인과 경로가 생긴다.

§4의 R1(1.0, nav·GRPO 없음) ≈ Teacher 1.5 결과와 일관됨. **미검증 후속:** R1에 동일 CoC ablation (~3 h, 1.0은 직접 구동 가능) → 결합 부재가 우리 student 고유인지 공개 모델 공통인지 판정.

보조 관찰(결정적이지 않음): CoC의 정보량 ≈ log₂(223) ≈ 7.8 비트로 4000-bin × 128토큰 궤적을 미터 단위로 구속할 수는 없다. 다만 위 28개는 기동 *클래스* 실패이므로 7.8비트로도 충분히 고쳐졌어야 하고, 고쳐지지 않았다. 정보량 부족이 아니라 **경로 부재**가 원인.

참고: 코드베이스에 `teacher_long_cot` 경로 존재 (`src/data/teacher_cache.py:384`, `scripts/05_generate_teacher_text_cache.py:71`) — 현재 코퍼스엔 미캐시. 다만 위 결과가 "짧아서 안 된다"를 이미 배제하므로 긴 CoT 확보는 우선순위가 아님.

### 6.2 beam은 구조를 고치지만 ADE를 못 얻음

beam이 개선한 것: 직진 붕괴 18.5%→13.6%, 회전 크기 0.89→0.93, 전방 변위비 0.992→1.003, 경로 길이비 0.998→1.014.
동시에 uniq 25.5→14.5로 붕괴, ADE 중앙값 델타 0.000, 평균 +0.138.

→ **커밋 능력이 병목이 아니라 어느 쪽으로 커밋할지가 병목.** greedy는 헤징해서 어디서나 중간, beam은 커밋해서 일부에서 크게 틀림. 순 효과 0.
→ 부수 결론: **모델 자신의 우도는 나쁜 채점기.** 공짜 selector 없음.

### 6.3 선택 헤드룸 (핵심)

```
단일 greedy 3.360  →  best-of-6 1.739     회수 가능 1.621 m
best-of-6가 Teacher 1.5 단일을 이기는 장면 63.5%
best-of-6가 R1 1.0  단일을 이기는 장면 57.6%
장면 내 6후보 ADE 표준편차: 중앙값 0.783 / 평균 1.069
최악 선택 4.825 vs 최선 1.739 → 3.086 m가 선택에 달림
오라클 정답 인덱스 분포 [71, 56, 66, 71, 68, 52] → 균등, 결정론적 디코드로 회수 불가
≤2s: 0.252 → 0.211 (이미 오라클 수준). 헤드룸 전부 >2s (4.772 → 2.434)
```

**2B 백본은 teacher를 능가하는 궤적을 이미 생성하고 있고 그것을 고르지 못한다.** 용량·데이터가 아니라 선택 문제.

### 6.4 실측 비용

| 설정 | 384 소요 |
|---|---|
| 1 path, CoC 없음 | 14분 |
| 1 path, CoC 접두 | 15~20분 |
| beam 8 | 44분 |
| **6 paths, CoC 없음** | **25분 (1 path의 1.8×)** |

비전 prefill을 6후보가 공유하므로 6×가 아님. CoC 생략이 무손실이므로 **CoC 생략 + 6샘플 + 선택**이 현재 단일 joint 디코드와 비슷한 지연시간대.

---

## 7. Action Expert

### 7.1 3-way (200K 백본 위, teacher_forced, 12,500 step = 0.5 epoch, val 1024)

`outputs/action_expert/ae3way_singlepath_20260722/*/summary.json` — 백본만 다르고 AE 설정 동일.

| 백본 | T=0.10 단일 | T=0.50 | T=0.85 | T=1.00 | **6-path oracle_best** |
|---|---|---|---|---|---|
| CE FullFT (step 7500) | 2.9240 | **2.8473** | 3.2794 | 3.5631 | 1.9216 |
| CE+KD FullFT (step 10000) | **2.6978** | 2.7618 | 3.0710 | 3.3414 | 1.6169 |
| **LoRA CE (step 10000)** | **2.5437** | 2.6771 | 3.0660 | 3.3564 | **1.5081** |

- 순서 LoRA < KD < CE 가 **모든 축에서** 일치 → hidden→action R²(0.793 > 0.744 > 0.684) 순서와 동일. **AE-reads-KV 확인.**
- **저온일수록 단일 경로가 좋음** (T=0.1 최적, T=1.0 최악).
- 단일 ↔ oracle_best 격차 ≈ **1.0 m** — 백본과 동일한 선택 병목.
- teacher_forced mean_traj best_eval: LoRA 2.5919 / KD 2.6116 (step 10000).

### 7.2 현재 학습 중 — format-fix 444K, student_free

`outputs/action_expert/ae_formatfix_444k_studentfree_20260727/main/`

```
step 25,000 / 50,000   경과 46.5 h   traj_start_hit_rate 1.0 (전 구간)
```

| step | ADE mean | ADE p50 | FDE mean | 1.6s | 3.2s | 6.4s |
|---|---|---|---|---|---|---|
| 10,000 | **2.706** | 2.027 | **7.933** | 0.169 | 0.674 | 2.706 |
| 20,000 | 2.748 | 2.021 | 8.109 | 0.160 | 0.664 | 2.748 |

(eval: 6-path, `mean_traj` 선택, T=0.85, n=1024 — **단일 경로 배포 지표 아님**)

`best.pt`는 아직 step 10000. 학습 손실은 계속 하락(0-5k 0.857 → 5-10k 0.277 → 15-20k 0.219 → 20-25k 0.251)하는데 기하 지표는 정체 → **손실과 지표 디커플.** 예전 CE AE가 7500에서 정점 찍고 악화한 패턴과 동일.

### 7.3 format-fix 배경

Step-B 전 백본이 `output_format_loss: 0.0` 때문에 `<|traj_future_start|>`(155680)를 못 냈음 → 공식 AE가 KV를 읽는 위치가 정의되지 않아 student_free 불가. 444K CE에 `output_format_loss:0.20`만 얹어 continue-train(LR 5e-6, ~1088 step, 0.02 epoch): fmt raw loss 12.63→0.015, free-run 유효 궤적 **0% → 98%**(bad_geometry 0.0195), greedy ADE 보존(1.680, val256 teacher_greedy 기준).

---

## 8. 무엇이 닫혔고 무엇이 남았나

| 후보 | 증거 | 판정 |
|---|---|---|
| **CoC 조건화** | −0.061 ± 0.062, 없애도 −0.063 ± 0.076. 기동 클래스 실패 28건 중 복구 1건 | **닫힘** — 증류한 CoC는 Alpamayo 정본이며 궤적에 인과적으로 관여하지 않음. SFT로는 결합이 생기지 않음 (outcome RL 필요) |
| 디코딩 (beam) | 중앙값 델타 0.000, 평균 +0.138 | **닫힘** |
| 커밋/캘리브레이션 | beam이 구조 지표 다 개선하고도 ADE 못 얻음 | **닫힘** |
| 모델 우도 기반 채점 | beam이 더 나쁨 | **닫힘** (학습형 selector 필요) |
| nav/GRPO 부재 | R1 disc − Teacher 1.5 disc = −0.064 ± 0.095 | **닫힘** |
| 데이터 스케일 | 200K→444K = −0.18, R² flat | 수확 체감 |
| KD / KL | 200K에서 CE와 tie, LoRA와 중복 | 토큰 경로엔 무효 |
| hidden-align | R² 0.729 < 0.793 | 역효과 |
| LoRA | R² +0.109, AE 전 축 개선 | **KV 채널엔 유효** |
| **후보 선택 (selector)** | 오라클 헤드룸 **1.621 m**, AE에도 ~1.0 m | **최대 미개척 레버** |
| 용량 (2B vs 8-10B) | best-of-6가 teacher 단일을 63.5% 장면에서 이김 | **병목 아님** |

---

## 9. 출처가 없거나 재확인이 필요한 값

| 값 | 상태 |
|---|---|
| teacher self-R² ≈ 0.804 (teacher 4096-d hidden → 자기 action) | 세션 중 계산, **JSON 미저장**. 스크립트만 scratchpad에 존재 → 재실행 필요 |
| 444K CE minADE6 (val512 전체) | 미측정. 384 서브셋만 존재 |
| `reports/` 하위 `mlflex_*` minADE6 값들 | **다른 계보(flex)** 모델. Step-B 백본과 혼용 금지 |
| 444K CE AE의 단일 경로(T=0.1) 지표 | 미측정. 학습 종료 후 필요 |
| R1(1.0) CoC ablation | 미실행. §6.1의 결합 부재가 우리 student 고유인지 공개 모델 공통인지 판정용 (~3 h) |

---

## 10. Provenance

| 항목 | 경로 |
|---|---|
| Backbone val512 디코드 | `outputs/reports/backbone_verify_spr1greedy_20260727/{CE,KD,LoRA,444K_CE,444K_CE_gt}_summary.json` |
| KV probe R² | `outputs/reports/kv_probe_444k_ce_20260722/`, `kv_probe_lorakd_200k_20260724/`, `kv_probe_hidden_gc_20260726/` |
| Weight delta | `outputs/reports/weight_delta_200k_vs_444k_20260725/{weight_delta,perrow_embed_groups}.json` |
| R1 이산/연속 | `outputs/reports/r1_discrete_backbone_20260728/full_384_summary.json`, `r1_full_horizon_20260728/r1_full_384_summary.json` |
| 4-way paired | `outputs/reports/r1_discrete_backbone_20260728/threeway_384.json` |
| 디코드 ablation | `outputs/reports/decode_probe_20260728/{D_teachercot,C0_nocot,B1_studentcot_greedy,B8_studentcot_beam8,M6_minade6}_summary.json` |
| 실행 스크립트 동결본 | `outputs/reports/decode_probe_20260728/_frozen_25_decode.py` (D 실행 버전, C0와 토큰 8/8 일치 검증) |
| AE 3-way | `outputs/action_expert/ae3way_singlepath_20260722/*/summary.json`, `ae3way_20260720_*/summary.json` |
| AE 현재 학습 | `outputs/action_expert/ae_formatfix_444k_studentfree_20260727/main/train.log` |
| BEV 시각화 (32장) | `/home/pm97/workspace/sukim/visualization/444k_ce_teacher_gt_bev_20260728/` |
| 평가 코퍼스 | `data/corpus/val512_seed42_selected_for_10b_fullft_lora_greedy.jsonl` (split=val 512행) |
| 384 서브셋 id | `outputs/reports/decode_probe_20260728/ids_384.json` |

이전 종합본: `reports/147-stepb-results-consolidated-20260723.md` (2026-07-23까지)
