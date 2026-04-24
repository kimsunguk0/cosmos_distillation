# 059 Claude Findings And H200 Transition

- status: open
- scope:
  - capture what Claude actually solved
  - separate useful token-path lessons from failed FM/action-expert detours
  - define the first H200 experiment ladder in this repo

## Executive Summary

Claude's meaningful success was not hidden-first recovery and not Flow Matching.

The successful path was:

- pure token-path training
- all-layer LoRA active
- custom trajectory/control token rows trainable through PEFT `trainable_token_indices`
- enough full-corpus steps
- trajectory token reweighting on
- moderate teacher trajectory/text supervision
- scheduled sampling added only after the token path was already working

Claude's best greedy token checkpoint was roughly:

- `Phase 10 greedy`: ADE `3.877`, FDE not the main headline but clearly better than our current clean SFT line
- `Phase 15 reranker`: ADE `3.740`

Claude's Phase 16 FM/CD line was a failure in absolute quality:

- token greedy: `3.877`
- reranker: `3.740`
- FM 50-step: `22.53`
- FM 4-step consistency student: `20.81`

So the H200 priority should remain:

```text
token-path recipe reproduction and scale-up
-> best checkpoint selection by decode geometry
-> reranker / candidate selection
-> only then revisit FM or heavier hidden alignment
```

## What Claude Actually Proved

Claude established five useful points.

### 1. Plateau root cause was mostly recipe/readout, not first-order hidden failure

The old `<i1499>`-style body plateau was not solved by FM heads, hidden manifold losses, or hybrid decode absorption.
It was solved by making the LM trajectory readout path trainable enough and supervising it cleanly enough.

The most important ingredients were:

- all-layer LoRA active
- custom distill token rows trainable
- trajectory CE trained for enough full-corpus steps
- central-band token prior reduced by reweighting

### 2. Teacher supervision helped after the token path was alive

Claude Phase 10 did not use GT-only SFT.
It used a mixed teacher-heavy token recipe:

- hard target path present
- teacher text top-k KD
- teacher trajectory hard CE
- teacher trajectory top-k KD
- weak hidden align
- scheduled sampling `p=0.25`

The key sequencing insight is:

```text
do not start with complex hidden/FM machinery
start by making token-path LM decode work
then add teacher signals
```

### 3. Scheduled sampling was not universally good

Scheduled sampling only helped inside Claude's already-strong mixed teacher recipe.
When we added SS on top of our B1d GT-anchored baseline, it did not beat B1d.

That means SS is not a standalone fix.
It is a continuation tool whose value depends strongly on the base recipe.

### 4. Reranking was real but small

Claude Phase 15 improved mean ADE modestly and cut the tail more clearly.
That means reranking is worth keeping in the H200 plan, but only after the base token checkpoint is strong.

### 5. FM/action-expert style paths are not the next priority

Claude tried to build an FM decoder and 4-step consistency-distilled variant.
The result was much faster inference but much worse planning quality.

This is important for us because it means:

```text
H200 extra compute should first go to token-path scaling,
not to prematurely moving into FM-only paths.
```

## Important Semantic Correction For This Repo

In our codebase, `teacher_pair_target` changes the meaning of the legacy hard-target losses.

If:

```yaml
data_view:
  teacher_pair_target: true
```

then the collator swaps the hard target to:

- teacher CoT text
- teacher trajectory token ids

That means the legacy names:

- `gt_cot_loss`
- `traj_loss`

still exist, but they no longer supervise the human/GT target.
They supervise the teacher pair hard target.

This matters because Claude Phase 10's stored config had:

```yaml
teacher_pair_target: true
gt_cot_loss: 0.05
traj_loss: 0.70
teacher_traj_loss: 0.50
teacher_traj_topk_kd_loss: 0.40
teacher_topk_kd_loss: 0.30
teacher_traj_hidden_align_loss: 0.05
scheduled_sampling: 0.25
traj_token_reweighting: on
```

So a faithful replay in our repo semantics is more teacher-heavy than its legacy field names suggest.

## What We Likely Got Wrong

The main mistakes were not imaginary diagnoses, but wrong ordering and wrong baselines.

### 1. We escalated too early into hidden-first work

The hidden-manifold experiments were not useless, but they were not the first thing that had to be solved.
We spent too much time in:

- H0 latent recovery
- H1 row probes
- readout/listwise smokes
- hybrid/prefix absorption branches

before locking a strong mixed token baseline.

### 2. We tested scheduled sampling on a different base recipe

Our SS tests were run on:

- B1d
- GT CoT hard target
- GT traj CE main objective
- teacher traj top-k KD only
- no teacher hard traj CE
- no teacher text KD
- no hidden align
- reweight off

Claude's SS success was run on a much richer mixed-teacher recipe.
So our result:

```text
SS did not help B1d
```

is true, but it is not the same experiment as Claude Phase 10.

### 3. We underused long full-corpus continuation on simple token recipes

Many of our negative conclusions came from short or highly entangled branches.
Claude's success came from long continuation of a simple token-path recipe, not from clever one-off probes.

## What To Port Into This Repo

Two new config tracks should exist in the main repo.

### A. Faithful Claude Phase 10 semantic port

Purpose:

- reproduce Claude's teacher-heavy recipe as closely as possible in our repo semantics

Key property:

- `teacher_pair_target: true`

This answers:

```text
If we replay Claude's real recipe on our current main repo,
does it beat the current GT-anchored baselines?
```

### B. Explicit GT-anchored mixed-target port

Purpose:

- keep GT hard target explicit
- add teacher hard CE, teacher top-k KD, weak hidden align, and SS on top

Key property:

- `teacher_pair_target: false`

This answers:

```text
Was Claude's improvement really about teacher-pair hard targets,
or about the teacher auxiliary signals around an otherwise GT-anchored path?
```

## H200 First-Pass Experiment Order

The recommended H200 ladder is:

1. Run the faithful Claude semantic port on the full corpus.
2. Run the explicit GT-anchored mixed port on the full corpus.
3. Select checkpoints by decode geometry, not by loss.
4. Train or reuse a reranker only after the base token checkpoint is fixed.
5. Only then consider bigger adaptation ranges or full fine-tuning.
6. Defer FM/action-expert revival until token-path scaling plateaus.

## Decode-Based Selection Rule

For the H200 runs, checkpoint selection should be based on:

- val204 greedy decode ADE/FDE
- invalid future token rate
- average unique trajectory ids
- max same-token run
- top-token histogram concentration
- failure-tag mix

Not on:

- final train loss
- final validation CE alone

This was already true in our repo and Claude's repo.

## Recommended H200 Priority

### Priority 1

Reproduce and compare:

- `configs/train/stage_h200_claude_phase10_faithful.yaml`
- `configs/train/stage_h200_claude_phase10_gt_anchor.yaml`

### Priority 2

After the better of those two is fixed:

- longer continuation
- periodic decode selection
- reranker

### Priority 3

Only if token quality stalls:

- larger trainable slice
- stronger hidden alignment
- full fine-tuning on H200

### Not Priority 1

- FM-only branch
- consistency-distilled FM branch
- action-expert reconstruction

Claude already showed those were worse than the token path under this project state.

## Current Recommendation

For H200, do not resume the old hidden-first tree as the mainline.

The mainline should now be:

```text
Claude-faithful token recipe
vs
explicit GT-anchored mixed token recipe
-> choose best decode geometry
-> reranker
-> scale-up
```

## New Repo Artifacts

This report is paired with:

- `configs/train/stage_h200_claude_phase10_faithful.yaml`
- `configs/train/stage_h200_claude_phase10_gt_anchor.yaml`

Those configs are meant to be the first H200 comparison pair.
