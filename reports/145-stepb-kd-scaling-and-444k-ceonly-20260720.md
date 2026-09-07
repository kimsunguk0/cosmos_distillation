# Report 145: Step-B Trajectory KD Sign-check (20K→200K) and 444K CE-only Scaling

Date: 2026-07-20
Status: active (444K CE-only training ~94% at time of writing)
Scope: Cosmos-Reason2-2B Step-B backbone distillation, no-FLEX discrete VLM path
Baseline reference lineage: FullFT-3e-5, init `stepa_q2_vqa_fullft_repaired_v1_bs8_e1/step_003488`, `--disable-lora`, ViT frozen, LLM+embeddings+lm_head+norms+multimodal projector trainable.

## 0. One-line

Trajectory top-k logit KD helps a little at 20K (on minADE6 / distribution, not greedy) but the benefit **vanishes at 200K** (tie vs CE-only on all geometry axes). Decision: **444K goes CE-only.** "CE-only" here is still distillation (targets are teacher-generated tokens = sequence-level KD). Open thread: AE-reads-KV hidden match.

## 1. Common settings across all runs

- Model: FullFT (no LoRA), LR 3e-5 cosine (warmup 3%), bf16, grad-checkpointing, grad-clip 1.0, batch 8 (effective 8, single H200).
- Loss family: `gt_cot_loss` and `traj_loss` (trajectory CE) are against **teacher-generated** CoT/trajectory tokens (provenance `alpamayo15_no_nav_teacher_*`) → sequence-level KD. The tested KD add-on is `teacher_traj_topk_kd_loss` (soft top-k KL, τ=1, tail_bucket).
- Decode eval reference: `teacher_greedy_ref_v1` (frozen). Greedy P-G = do_sample=false, max_new_tokens 320. minADE6 = samples_per_row 6, τ=1, seed 42, geometry_reference gt.
- val: natural distribution (not balanced). See §6.

## 2. Corpora (important — verified)

| corpus | train rows | unique | dup | note |
|---|---:|---:|---:|---|
| 20K balanced | 20,000 | 18,906 | 5.5% | rare scenes up to 5× |
| 200K balanced | 200,000 | 147,692 | 26.2% | rare scenes up to 9× |
| 444K full | 435,241 | 435,241 | 0% | natural distribution, all unique |

"200K balanced" is really ~148K unique + upsampled duplicates. 444K full = 2.9× more unique data than 200K.

## 3. R1 20K KD sign-check

Config `configs/train/stepb_ladder_r1_tailkl.yaml`: epochs 3.0 (20K→7500 steps), KD weights gt_cot 0.1 / traj_ce 1.0 / teacher_traj_topk_kd 0.5, τ=1, tail_bucket. Trained 13.4h, best_decode ~step 6000. Baseline = R0-F-20K (identical recipe, KD off).

Evaluated on frozen val512, paired bootstrap CI (5000, seed 42), KD − CE:

### Axis 1 — greedy free-run (val512)
| metric | CE (R0-F) | KD (R1) | Δ | 95% CI | sig |
|---|---:|---:|---:|---:|:--:|
| ADE | 3.122 | 3.164 | +0.043 | [−0.297,+0.386] | no |
| FDE | 9.741 | 9.636 | −0.106 | [−1.144,+0.915] | no |
| bad_geom | 0.096 | 0.094 | −0.002 | [−0.027,+0.023] | no |
| unique | 14.20 | 11.63 | −2.57 | [−4.13,−1.06] | **yes (worse)** |
| target jaccard | 0.111 | 0.100 | −0.012 | [−0.019,−0.004] | **yes (worse)** |

### Axis 2 — teacher-forced token distribution (val512)
| metric | CE | KD | Δ |
|---|---:|---:|---:|
| KL(teacher‖student) | 0.311 | 0.287 | −0.025 (KD's own objective ✓) |
| top-1 acc | 0.456 | 0.455 | −0.001 |
| entropy | 1.651 | 1.704 | +0.053 (softer) |
| margin | 0.969 | 0.867 | −0.102 |

### Axis 3 — minADE6 (val512, paired CI)
| metric | CE | KD | Δ | 95% CI | sig |
|---|---:|---:|---:|---:|:--:|
| minADE6 | 2.160 | 2.084 | −0.076 | [−0.200,+0.048] | no |
| unique (6-sample) | 68.8 | 71.4 | +2.58 | — | KD more diverse |

**20K verdict:** KD flattens the student distribution toward teacher's top-k (KL↓, entropy↑, margin↓). Greedy ignores this (argmax unchanged) and free-run diversity drops; but sampled coverage (minADE6) is directionally better + more diverse. Weak positive on the deployment-relevant axis, not sealed.

## 4. 200K + KD

Config `configs/train/stepb_fullft_lr3e5_200k_e1_r1kd.yaml` = CE-only 200K recipe + KD (only 2 knobs changed: `teacher_traj_topk_kd_loss` 0→0.5, `tail_bucket` false→true). epochs 1.0 → 25000 steps. Trained 20.3h. Corpus 200K balanced. Baseline = existing CE-only FullFT-200K (identical init/recipe).

In-training val256 greedy trajectory: s5000 ADE 3.48 → s10000 2.77 → s15000 2.41 → s20000 2.06 → s25000 **1.895** (uniq 23).

Head-to-head vs CE-only-200K, frozen val512:

### Axis 1 — greedy (paired CI)
| metric | CE-only | KD | Δ | 95% CI | sig |
|---|---:|---:|---:|---:|:--:|
| ADE | 2.040 | 2.047 | +0.006 | [−0.152,+0.168] | no |
| FDE | 6.595 | 6.604 | +0.009 | [−0.488,+0.495] | no |
| bad_geom | 0.031 | 0.043 | +0.012 | [−0.004,+0.027] | no |
| unique | 21.64 | 22.20 | +0.56 | [−1.12,+2.28] | no |

### Axis 2 — teacher-forced distribution
| metric | CE-only | KD | Δ |
|---|---:|---:|---:|
| KL(teacher‖student) | 0.234 | 0.190 | **−0.044 (−19%)** |
| top-1 acc | 0.476 | 0.480 | +0.003 |
| entropy | 1.482 | 1.557 | +0.075 |
| margin | 1.080 | 0.924 | −0.156 |

### Axis 3 — minADE6 (paired CI)
| metric | CE-only | KD | Δ | 95% CI | sig |
|---|---:|---:|---:|---:|:--:|
| minADE6 | 1.6453 | 1.6411 | −0.004 | [−0.100,+0.093] | no |
| minFDE6 | 4.879 | 4.825 | −0.054 | [−0.383,+0.287] | no |
| unique (6-sample) | 66.95 | 71.06 | +4.11 | — | KD more diverse |

### Cross-scale (KD − CE)
| KD effect | 20K | 200K | trend |
|---|---:|---:|:--|
| minADE6 gain | −0.076 (directional) | −0.004 (gone) | **vanished** |
| greedy ADE | +0.043 | +0.006 | tie both |
| KL reduction (KD objective) | −0.025 | −0.044 | grew |
| sampling diversity | +2.58 | +4.11 | grew |

**200K verdict:** KD's direct effects (KL↓, diversity↑) grew with scale, but downstream geometry benefit (minADE6) collapsed to zero. 200K hard-target CE already extracts what teacher soft labels gave at 20K. Classic diminishing KD in higher-data regime → **drop KD for 444K.**

## 5. 444K CE-only (full unique set) — in progress

Config `configs/train/stepb_fullft_lr3e5_444k_e1.yaml` (CE-only, KD off, else identical to 200K recipe). Corpus `no_nav_teacher_pair_full444k.jsonl` (435,241 train / 9,007 val, natural distribution). epochs 1.0 → 54,406 steps. Launched 2026-07-18 12:15Z.

In-training val256 greedy (teacher_greedy_ref):
| step | ADE | FDE | bad_g | uniq | frg |
|---|---:|---:|---:|---:|---:|
| 10,881 | 3.582 | 11.18 | 0.121 | 7.4 | −6.98 |
| 21,762 | 2.506 | 8.02 | 0.078 | 15.2 | −4.90 |
| 32,643 | 1.984 | 6.47 | 0.035 | 17.3 | −3.78 |
| 43,524 | 1.837 | 5.93 | 0.035 | 22.1 | −3.50 |

At step 43,524 (80%), val256 ADE **1.837 already beats KD-200K final (1.895)** with strong diversity. Note val is natural distribution, matching 444K's train distribution (favorable); balanced-200K trained on upsampled rare scenes that barely appear in val.

## 6. val distribution note

val9007 / val512 are **natural** distribution: `lead_vehicle_follow` 23–36%, rare cats (`traffic_left_turn`, `left_turn_no_light`) <0.5%, 17 categories, max/min 118–181×. So natural-trained 444K has a train/eval-match advantage on aggregate ADE; balanced models may still win rare per-category. Final 444K-vs-200K comparison must report aggregate AND per-scene-category.

## 7. Remaining work

1. 444K completion (~2h from writing) → val512 P-G + minADE6 + **per-scene-category** vs CE-200K / KD-200K.
2. **AE-reads-KV hidden match** (open): AE (`outputs/action_expert/*`, ae28) reads per-layer KV at traj boundary and was trained on teacher KV. Tier-1 = `scripts/21_probe_hidden_latent.py` CKA/centered-gram (student 2B vs teacher 4096-dim `final_lm_pre_head` cache), KD-200K vs CE-200K to teacher. Tier-2 = frozen-AE swap (CE-KV vs KD-KV → AE trajectory ADE/minADE).

## 8. Artifacts

- Training: `outputs/checkpoints/stepb_r1_signcheck/...` (R1 20K), `outputs/checkpoints/stepb_r1kd_200k/...` (200K+KD), `outputs/checkpoints/stepb_ceonly_444k/...` (444K CE).
- CI/compare: `outputs/reports/stepb_r1_signcheck/.../r1_vs_r0f20k_val512_ci.json`, `outputs/reports/stepb_r1kd_200k/.../compare/`.
- Configs: `stepb_fullft_lr3e5_200k_e1_r1kd.yaml`, `stepb_fullft_lr3e5_444k_e1.yaml`.
