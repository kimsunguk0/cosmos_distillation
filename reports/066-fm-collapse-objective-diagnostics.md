# 066 - FM Collapse Objective Diagnostics

Date: 2026-05-31

## TL;DR

Collapse 원인 = **student hidden/KV 부족만으로는 설명되지 않고, random `x0/t`를 매 step 새로 뽑는 FM velocity objective/dynamics에서 AE가 `target_v = x1 - x0` vector field를 못 배우며 low-amplitude 평균해로 수축하는 문제**.

Claude가 제안한 E1/E2/E3는 해볼 가치가 있었다. 다만 핵심 전제 하나는 틀렸다. 현재 코드에서 `target_v`는 `target_action`이 아니라 `target_action - randn`이다. 그래서 "작은 target_action이 많아서 target_v도 작고 평균으로 도망간다"는 설명은 부족하다.

## What Was Being Done

작업 흐름은 Alpamayo-1.5-10B teacher의 trajectory/action behavior를 Cosmos-Reason-2B student 쪽으로 distill한 뒤, student가 만든 prompt KV 위에서 28-layer action expert(AE28)를 학습시키는 것이다.

최근 report 흐름은 다음과 같았다.

- 060: student AE pairing ceiling 원인을 hidden mismatch로 의심.
- 061: GT target으로 바꿔도 teacher target보다 낫지 않음.
- 062: best-of-N eval이 잠깐 희망적으로 보였지만 이후 해석 보정 필요.
- 063: Stage0 32-sample overfit이 실패.
- 064: attention/conditioning collapse 의심.
- 065: A/B/C isolation 뒤 student KV/hidden incompatibility 쪽으로 결론.
- D1: oracle KV 32-sample overfit도 학습이 진행될수록 `pred_v`가 `target_v`보다 작아지는 collapse 패턴 확인.

이번 진단은 D1과 같은 corpus인 `data/corpus/no_nav_teacher_pair_300chunks.jsonl` 기준으로 실행했다.

## E1 Target Distribution

Output: `outputs/action_expert/fm_collapse_diagnostics/e1_d1_300chunks_seed42/summary.json`

32-sample 기준:

- `target_action_abs_mean`: mean 0.555, p50 0.387, p95 1.712
- `flow_target_v_abs_mean`: mean 1.072, p50 0.942, p95 1.915
- timestep beta sampler: `t < 0.2` 28.1%, `t > 0.8` 9.0%, `t > 0.9` 3.6%

256-sample 기준:

- `target_action_abs_mean`: mean 0.646, p50 0.541
- `flow_target_v_abs_mean`: mean 1.142, p50 1.033
- bucket별 `target_v_abs_mean`: small 0.835, medium 1.039, large 1.554

해석:

- action 자체는 작은 샘플이 꽤 있지만, FM target인 `target_v = x1 - x0`는 Gaussian `x0` 때문에 작게 몰려 있지 않다.
- 따라서 E1만으로 "작은 움직임 다수 -> mode averaging collapse"라고 결론내리면 안 된다.
- large bucket이 더 큰 `target_v`를 갖긴 하지만, collapse가 large-only인지 E2가 필요했다.

## E2 Bucket Probe

Output: `outputs/action_expert/fm_collapse_diagnostics/e2_d1_oracle_bucket_probe_seed42/summary.json`

기존 D1 oracle-KV run의 `initial`, `best.pt`, `final.pt`를 새 학습 없이 같은 16 random probe draw로 비교했다.

| checkpoint | bucket | pred_v_abs | target_v_abs | pred/target | cosine | alpha |
|---|---:|---:|---:|---:|---:|---:|
| initial | overall | 1.182 | 1.071 | 1.216 | 0.094 | 0.088 |
| best | overall | 0.331 | 1.071 | 0.297 | 0.252 | 0.084 |
| final | overall | 0.325 | 1.071 | 0.290 | 0.252 | 0.084 |
| final | small | 0.210 | 0.833 | 0.253 | 0.136 | 0.032 |
| final | medium | 0.250 | 0.936 | 0.267 | 0.252 | 0.075 |
| final | large | 0.510 | 1.433 | 0.349 | 0.366 | 0.142 |

해석:

- 학습 후 collapse는 large bucket에만 국한되지 않는다. small/medium/large 모두 target 대비 25-35% 크기로 줄어든다.
- large bucket은 절대 loss가 가장 크지만, 모든 bucket에서 amplitude shrink가 보인다.
- "큰 velocity sample만 회피한다"보다는 "전체 FM vector field가 낮은 amplitude 평균해로 수축한다"가 더 맞다.

## E3 Unconditional 1-Sample

Output: `outputs/action_expert/fm_collapse_diagnostics/e3_d1_300chunks_1sample_seed42/summary.json`

conditioning/KV를 전부 제거하고 `action_in_proj -> expert -> action_out_proj`만 사용했다.

같은 1개 샘플에서:

- fixed `x0/t`: step 500 loss 0.00059, `pred_v_abs` 0.875 vs `target_v_abs` 0.871, cosine 0.9998, alpha 1.005
- random `x0/t`: step 500 loss 1.1006, `pred_v_abs` 0.109 vs `target_v_abs` 0.820, cosine 0.0205, alpha 0.0029

해석:

- 고정된 single target은 완전히 외운다. 따라서 action projection, expert parameter update, optimizer가 "무조건 고장"난 것은 아니다.
- 같은 1개 trajectory라도 매 step `x0/t`가 바뀌는 FM objective에서는 collapse한다. 이건 multi-sample mode averaging이나 KV/hidden 부족보다 더 하위의 FM objective/parameterization 문제다.

## Conclusion

이번 결과에서 가장 중요한 반전은 E3다. **1-sample random `x0/t` FM도 못 외우므로 collapse 원인은 conditioning/KV 부족이나 multi-sample trajectory 평균화가 아니라, random-noise FM vector field를 현재 AE 입력/시간 parameterization으로 학습하는 경로 자체에 있다.**

다음 ablation 우선순위:

1. `x0=0` 또는 fixed-noise training으로 32-sample overfit을 돌려서 FM noise가 원인인지 확인.
2. timestep weighting/target_v normalization을 넣어 `x0` 지배 구간의 gradient가 평균해로 수축하는지 확인.
3. `action_in_proj(x_t, t)`의 t embedding 및 x_t scale을 직접 로깅해서 random `x0/t` 조건을 실제로 구분하는지 확인.
4. D1 oracle KV는 synthetic target-projection KV라 hidden/KV 가설을 완전히 배제하는 증거는 아니지만, 이번 E3 때문에 우선순위는 FM objective 쪽이 맞다.

## Artifacts

- Diagnostic script: `scripts/94_diagnose_fm_collapse.py`
- E1 summary: `outputs/action_expert/fm_collapse_diagnostics/e1_d1_300chunks_seed42/summary.json`
- E2 summary: `outputs/action_expert/fm_collapse_diagnostics/e2_d1_oracle_bucket_probe_seed42/summary.json`
- E3 summary: `outputs/action_expert/fm_collapse_diagnostics/e3_d1_300chunks_1sample_seed42/summary.json`
