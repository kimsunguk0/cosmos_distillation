# Step-B 결과 종합 (Backbone + Action Expert, Teacher/GT 기준)

작성 2026-07-23. val512 기준. `ADE` = 단일 path (backbone=greedy / AE=저온 1-sample), `minADE6@6.4s` = 6개 샘플 중 best.

## 참조 객체 (열마다 기준이 다름 — 반드시 유의)

| 기준 | 정의 |
|---|---|
| **Backbone-Teacher** | teacher_greedy **토큰**(결정론 argmax) 디코드. ref = `teacher_greedy_ref_v1` |
| **AE-Teacher** | teacher **연속 action**(`pred_xyz`, teacher action-expert 출력, cache T=0.6). **Backbone-Teacher와 다른 객체** |
| **GT** | 실제 로그 `ego_future_xyz` (`load_ego_future_xyz`) |

> ①Backbone과 ②AE의 "Teacher" 열은 **기준 객체가 달라 직접 비교 불가**. 각 표 내부에서만 비교할 것.

---

## ① Backbone (LM-head 이산 디코드)

| backbone | 데이터 | Teacher ADE | Teacher minADE6 | GT ADE | GT minADE6 |
|---|---|---:|---:|---:|---:|
| 200K CE | 200K | 2.040 | 1.062 | 3.136 | 1.645 |
| 200K CE+KD | 200K | 2.047 | **1.016** | 3.214 | 1.641 |
| 200K LoRA | 200K | 2.051 | 1.052 | 3.257 | 1.710 |
| **444K CE** | 444K | **1.829** | **0.956** | **3.047** | **1.579** |

- Teacher ADE/minADE6 = `teacher_greedy_ref_v1` 기준 (script 70 P-G greedy / script 25 samples6).
- GT ADE = greedy 단일 (script 25 spr=1, geometry_reference gt, 2026-07-23 재수집). 앵커 재현: CE 3.136(기존 3.157), LoRA 3.257(기존 3.221) — ±0.02~0.04, seed/max_new_tokens 차이 수준. GT minADE6 = samples6 T=1.0.
- 참고: teacher greedy 자신의 GT-ADE = 2.115 (student는 teacher 오차 상속+가산이라 3.0~3.3, 데이터↑로 444K가 3.05로 소폭 개선).

## ② Action Expert (200K backbone 위, teacher_forced 학습, 0.5 epoch)

기준 = **AE-Teacher(연속 action)** / **GT**. AE ADE = 저온(T=0.1) 1-sample 단일 path.

| AE on backbone | KV R²(→teacher act) | Teacher ADE | Teacher minADE6 | GT ADE | GT minADE6 |
|---|---:|---:|---:|---:|---:|
| 200K CE | 0.684 | 2.924 | 1.922 | 3.079 | 2.043 |
| 200K CE+KD | 0.744 | 2.698 | 1.607 | 2.834 | 1.746 |
| **200K LoRA** | **0.793** | **2.544** | **1.508** | **2.730** | **1.654** |
| 200K LoRA+KL | 0.793 | 2.645 | 1.597 | — | — |

- ADE = T=0.1 단일-path (저온=greedy/mode). best.pt: CE @7500, KD @12500, LoRA @10000, LoRA+KL @7500 (val minADE6 최소).
- **LoRA+KL AE는 null 실측 확정** (2026-07-25, `outputs/action_expert/ae_lorakd_20260725/`): minADE6 1.597 vs LoRA-CE 1.508, 단일 2.645 vs 2.544 — 두 지표 모두 ~0.09~0.10 근소하게 나쁨(run-noise/조기 overfit @7500). KV R² 0.7928≈0.7931 예측대로 AE 동급. **KL은 LoRA 위에서 완전 중복.** GT열 미측정(teacher 기준 null이 명확 + teacher-default 정책).
- AE Teacher ADE 온도별 (단일): CE 2.92/2.85/3.28/3.56, KD 2.70/2.76/3.07/3.34, LoRA 2.54/2.68/3.07/3.36 (T=0.1/0.5/0.85/1.0). 저온일수록 우수. GT도 동일 경향 (CE는 T=0.5가 2.998로 최저).
- **순서 LoRA<KD<CE가 GT 기준에서도 유지** (minADE6 1.654<1.746<2.043; 단일 2.730<2.834<3.079).
- ⚠️ **eval set 주의**: ①Backbone은 val512, ②AE는 held-out val10k에서 1024 샘플(seed42) — **서로 다른 집합**. 참조 객체(토큰 vs 연속 action)까지 합쳐 ①↔② 직접 비교 불가, 표 내부 비교만 유효.

---

## 부가 지표 — backbone KV 표현 상태 (script 112, hidden→action R²)

teacher 연속 action에 대한 선형 예측력 (25% held-out ridge, val512). **높을수록 AE가 읽기 좋음.**

| backbone | hidden→action R² | vs 200K CE (centered-gram) |
|---|---:|---:|
| 200K CE | 0.684 | 1.000 (기준) |
| 444K CE | **0.687** | 0.930 |
| 200K CE+KD | 0.744 | 0.956 |
| 200K LoRA | 0.793 | 0.518 |
| 200K LoRA+KL | **0.793** (0.7928) | — |

**핵심 dissociation: CE 데이터 2.2배(200K→444K)로도 KV→action R²는 flat(0.684→0.687).** 데이터 scale은 이산 토큰 경로를 개선하지만 AE가 읽는 KV 기하는 못 바꿈. KV 기하는 방법(KD/LoRA)-의존.

### KL×method 2×2 (2026-07-24 완결)

| hidden→action R² | CE-only | +KL (traj top-k, w=0.5) | ΔKL |
|---|---:|---:|---:|
| **FullFT** | 0.684 | 0.744 | **+0.060** |
| **LoRA** | 0.793 | 0.793 (0.7928) | **+0.000** |

- LoRA+KL: `outputs/reports/kv_probe_lorakd_200k_20260724/lorakd_vs_lorace_val512.json`. 같은 프로브(script 112, val512, ridge α1.0, 25% held-out)로 **LoRA-CE 앵커 0.7931 재현**(원값 0.793과 4자리 일치) → harness 검증. KL은 config no-op 아님(train.log ttraj_kd raw 19.7→0.14, weight 0.5 전 구간 활성).
- **KL과 LoRA는 가산이 아니라 중복.** 둘 다 KV를 teacher-정렬 축으로 밀지만 LoRA 단독(0.793)이 이미 FullFT+KL(0.744)보다 높은 천장에 도달 → KL이 더 얹을 게 없음. 저랭크 제약이 지배적 레버, soft-logit KL은 그 안에 흡수. **사전 게이트(≥0.82 진행) 미달 → LoRA+KL AE 학습 스킵**(R²상 LoRA-CE AE minADE6 1.508과 동급 예측).
- backbone 학습: `outputs/checkpoints/stepb_lora_kd_200k/lora_kd_200k_20260723/lprime_lora_lr2e4_200k_e1_r1kd/` (25000 step 완주, best_val 1.428, elapsed ~40h).

### 데이터 개선이 어디로 갔나 — weight-delta 200K↔444K CE (2026-07-25)

같은 stepa init(step_003488)에서 나온 200K/444K CE full-FT 두 체크포인트의 모듈별 가중치 변화 ‖W₄₄₄−W₂₀₀‖ (`scripts/measure_backbone_weight_delta_200k_vs_444k.py`, `outputs/reports/weight_delta_200k_vs_444k_20260725/weight_delta.json`).

| bucket | rel_step ‖Δ‖/‖W‖ | 변화 에너지 share (embed 1회) |
|---|---:|---:|
| **embed/lm_head (tied)** | **0.264** | ~51% |
| transformer_body (28층) | 0.060 | ~47% |
| visual(mm) | 0.022 | ~2.5% |
| final_norm | 0.011 | ~0% |

- embed·lm_head는 dstep 소수점까지 동일 = **weight tying 유지**(한 readout 행렬). bulk Frobenius로는 이 행렬이 자기 크기 대비 **26% 움직여 body(6%)의 ~4.4배**.
- 층별: **후반 층(24~27) rel_step 0.03~0.05로 최소**(= AE·R² 프로브가 읽는 최종 hidden 생성 층), 초·중반 층(12~16) 0.11~0.12로 최대. **이 부분은 견고.**

**per-row 정정 (2026-07-25, `outputs/reports/weight_delta_200k_vs_444k_20260725/perrow_embed_groups.json`):** bucket "readout이 dominant"는 per-row로 까면 **부분 반증**. tied embed 행별 ‖Δ‖를 토큰 그룹으로 나누면:

| 그룹 | 행 수 | 행당 mean_rel | 에너지 share |
|---|---:|---:|---:|
| language BPE | 151,637 (97.4%) | 0.273 | **99.08%** |
| traj_bin `<iN>` (4000) | 4,000 (2.6%) | **0.554**(최대) | 0.90% |
| marker | 48 | 0.475 | 0.02% |

- 변화는 **집중이 아니라 vocab 전체에 거의 균일 확산**(상위 2.6% 행이 에너지 4.2%뿐). traj_bin 행 절대Δ 순위 중앙값 153,179/155,685(바닥). readout bulk의 99%는 trajectory 무관 **언어행이 골고루 ~27% 표류** — CE가 traj 위치마다 비-traj logit 억제 + weight-decay가 444K(2.2배 스텝)에서 더 누적된 부산물로 추정(정확한 원인 CE-억제 vs decay는 미분리).
- **기능적으로 ADE에 관여하는 신호 = traj_bin 행(per-row 최대 0.554)의 refine + 초·중반 층.** 언어행 표류는 argmax(4000 bin 사이 상대 logit)에 영향 거의 없음.
- **해석(정정본): 데이터 추가분 중 ADE에 유효한 부분은 traj_bin readout 행 + 초·중반 층; hidden 만드는 후반 층은 거의 불변.** → greedy 토큰 decode↑(2.04→1.83)이지만 AE가 읽는 KV 기하는 flat(R² 0.684→0.687). 데이터-vs-방법 dissociation은 R² flat·gram 0.93·후반층 최소이동으로 독립 확립(readout 해석과 무관하게 견고).
- caveat: 200K/444K는 별개 run(연속 아님)이라 dstep에 run-noise 섞임; embed·lm_head tied라 입력/출력 역할 분리 불가; per-token 행 집중도 분석은 tokenizer 로드 실패로 미완(트라젝토리-bin 행 집중 여부는 별도 확인 필요).

---

## 핵심 읽기

1. **AE 순서 LoRA < KD < CE 전 지표 일관** — KV R²(0.79/0.74/0.68) 순서와 일치. **AE-reads-KV 확정**: backbone KV의 teacher 정렬도가 AE minADE6·단일-path·FM loss(0.21/0.24 vs 0.76) 전부에 전파.
2. **데이터 vs 방법 dissociation**: 데이터↑ = backbone 토큰 경로 개선(Teacher ADE 2.04→1.83), KD/LoRA = AE용 KV 기하 개선. → **444K CE로 AE 학습해도 200K CE AE와 동급(~1.92) 예상; 더 좋은 AE는 444K에 KD/LoRA 적용이 답.**
3. **단일-path ↔ minADE6 격차 ~1.0m = 먼 미래(6.4s) multimodality tax.** horizon 분해(LoRA): 1.6s gap≈0(단일 이미 oracle급), 3.2s 0.17, 6.4s 1.04. best path는 6슬롯 균등(예측 불가) → 결정론적 디코드로 minADE6 회수 불가. 해법 = selector/planner 또는 goal-conditioning (온도·추가학습 아님). 부분적 환원 불가(teacher도 GT 대비 2.12m).

## Provenance
- Backbone teacher: `outputs/reports/teacher_greedy_minade6_20260721/`, `.../{stepb_r1_signcheck,stepb_ceonly_444k,stepb_r1kd_200k}/.../*_pg_teacher_ref_summary.json`
- Backbone GT: `outputs/benchmarks/val512_*`, `.../stepb_*/.../*_ps6_*summary.json`, `outputs/reports/gt_fill_20260723/backbone_greedy_gt/`
- AE teacher: `outputs/action_expert/ae3way_20260720_*` (minADE6), `ae3way_singlepath_20260722/*` (단일)
- AE GT: `outputs/reports/gt_fill_20260723/ae_gt/`
- KV R²: `outputs/reports/kv_probe_444k_ce_20260722/hidden_kv_repr_compare_val512.json`
