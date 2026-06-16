# Handoff to Gemini — Consigliere Lane (Adversarial Critique)

> **Paste this entire file as the Gemini prompt.** Self-contained, no follow-ups needed.

## Your role

You are the **Consigliere** in our TriLane (Claude = Compiler, GPT = Architect, Gemini = Consigliere). Your job is **adversarial critique** — find what's bullshit before a hostile public reviewer does. You score highly when you flag real holes; you score zero for politeness.

Conflict resolution rule (operator-set): GPT > Gemini > Claude on architecture; Atom McCree is Final Stop on everything.

## Context

**Operator:** Atom McCree, solo independent operator. AtomEons Systems Laboratory.
**Date:** 2026-06-16.
**Mission:** ARC Prize 2026 — three competitions in flight, all entered.

| Competition | Pool | Teams | Deadline | Atom's entry |
|---|---|---|---|---|
| ARC-AGI-3 (interactive) | $850k | 1235 | 2026-11-02 (Milestone #1: 2026-06-30, $25k/$10k/$2.5k) | atommccree/agi-in-a-video-shop-atom-eons-nostalgia, kernel v5 |
| ARC-AGI-2 (static) | $700k | 849 | 2026-11-02 | sister agent built, not yet pushed |
| Paper Track (writeup) | $75k base + $375k bonus (≥4.5/5 rubric) | 72 | 2026-11-09 | scaffolded, not yet drafted |

**Frontier baseline** (arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis, 2026-05-01, semi-private set):
- GPT-5.5: 0.43%
- Opus 4.7: 0.18%
- StochasticGoose (preview 1st): CNN action-learning

**Foundation's stated pattern** (arcprize.org/blog/hrm-analysis, 2025-08-15):
- Hierarchical architecture contributes ~5pp; **outer refinement loop drives +13pp from 0→1, doubles 1→8**
- HRM ≈ "zero-pretraining test-time training"
- 2025 winners: NVARC 24% (Architects+TTT+TRM), ARChitects 16.5% (2D masked diffusion + recursive self-refinement), MindsAI 12.6% (TTFT + augmentation ensembles)
- TRM: **7M params, 45% on ARC-AGI-1, 8% on ARC-AGI-2** (zero-pretraining track)

## The approach (Misfit substrate)

**Tier framework (mechanically CI-enforced):**
- **Tier-1** — Spelke core priors only. No LLM in inference. No pretrained weights. No score weights tuned on public eval. Honest fluid-intelligence claim per [Chollet 1911.01547](https://arxiv.org/abs/1911.01547).
- **Tier-2** — Tier-1 + small bundled LLM as heuristic. Pretraining-contaminated. Engineering number, not intelligence claim.
- **Tier-3** — Tier-2 + cloud judge. Wildly contaminated. Disclose loudly.

`tests/test_tier1_attestation.py` greps the source tree for `torch.load`, `from transformers`, `from openai`, `from anthropic`, `from llama_cpp`, `huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, `smolagents`, `.gguf`, `.safetensors`, `gpt-?\d+`, etc. CI fails on any hit.

**15 substrate modules** under `src/misfit_agent/`:

| Module | Spelke prior(s) | Risk class |
|---|---|---|
| `perceptor.py` | objectness, geometry, topology | clean |
| `tracker_hungarian.py` | continuity, persistence | clean |
| `fingerprint.py` (50-dim) | numerosity + geometry stats | clean, but bin counts are designer choices |
| `resonance.py` (per-install JSONL, K-NN) | experience (Chollet-allowed) | clean if `source_tagged == "self-solved"` |
| `world_model.py` with `fit_with_refinement(max_iters=4)` | compositionality, sparse causality + HRM outer-loop | clean |
| `rules/translate.py`, `rules/no_op.py` | hand-authored typed grammar (disclosed) | suspect — see below |
| `goal_inducer.py` (3 hypothesis families, ≤3 free params) | goal-directedness, numerosity | designer-authored taxonomy |
| `mcts_puct.py` (PUCT + action deep-copy safety) | budget-aware search | uses ARC methodology loophole "internal ops don't count" |
| `click_quantizer.py` (ACTION6 4096→~20 candidates) | objectness for click candidates | clean |
| `abstain_policy.py` (3-conjunction gate) | quadratic-scoring-derived | min_actions=25 threshold is arbitrary |
| `arc2_solver.py` (Identity / Translate2 / Recolor) | sister for static tasks | low coverage — only global-shift |
| `arc2_runner.py` (wall-clock + per-task governor) | budget discipline | clean |
| `config.py` (frozen thresholds) | each value classified (a)prior/(b)budget/(c)tuned | none are (c) yet |
| `misfit_agent.py` (Misfit class) | orchestration | choose_action gates MCTS at coverage≥0.30 |
| `episode.py` (EpisodeTracker + observe_hungarian) | state log | clean |

**Test count:** 92 passed + 3 xfailed + 1 flaky (`test_mcts_puct.py::test_hard_timeout_is_respected_within_tolerance` fired 780ms vs 720ms grace — pre-existing).

**Public surfaces live:**
- https://github.com/AtomEons/arc-agi-3-misfit-agent (Apache-2.0)
- https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia (kernel v5, Phase A pending)
- https://www.kaggle.com/datasets/atommccree/misfit-agent-substrate (public, currently README only)

**Paper Track rubric (6 criteria, each 0-5, average final, ≥4.5 unlocks $375k bonus split):**
1. Accuracy (linked via Kaggle submission ID)
2. Universality (generalizes beyond ARC)
3. Progress (advances community's ARC chances)
4. Theory (WHY > HOW)
5. Completeness (covers leaderboard submission)
6. Novelty (vs existing public research)

Judges: François Chollet, Greg Kamradt, Mike Knoop, María Cruz.
Word limit: 1,500. Submissions per team: ONE.

Self-target: 28/30 ≈ 4.67/5 — bonus-eligible. Realistic without leaderboard accuracy: 4.5-4.7.

---

## What we want from you, Consigliere

Be relentless. Specifically:

### 1. Find the encoded crystallized knowledge

Where in this stack are we dressing up an ARC-corpus-derived heuristic as "Spelke priors"? Specifically suspect:
- The six rule templates (TRANSLATE / TELEPORT_TO / DESTROY_ON_CONTACT / SPAWN_ON_CONTACT / TOGGLE_AT_CURSOR / NO_OP) — the author has been exposed to ARC-AGI-1/2 puzzles. Honestly: how much of this template list is contamination?
- The 50-dim fingerprint composition (bin counts, feature families)
- Goal Inducer's 3-family taxonomy
- MCTS reward shaping: +10 WIN / -0.01 per action / +0.1 novel fingerprint — defensible or arbitrary?
- The `min_actions = 2 × human_baseline` derivation in abstain_policy

### 2. Find the publication attack surface

You are a hostile reviewer for the Paper Track. The judges include Chollet. Where will the paper get torn apart?

Hint: NVARC won with hybrid TTT+TRM at 24% on ARC-AGI-2. We're claiming a clean Tier-1 honest framework. If we score 3% on Milestone #1, does the framework alone justify a 4.67/5 average across the rubric? Or will the judges weight Accuracy more than the 1/6 the spec implies?

### 3. Find the one architectural gap

If you could add ONE thing to this stack within 7 days that respects Tier-1, what is it? Not five things — ONE. The most-leverage missing piece.

We already have: Goal Inducer, Abstain Policy, Hungarian Tracker, MCTS-PUCT, outer refinement loop, click quantizer, resonance library. What's the highest-EV addition?

### 4. Reply schema

Return your critique in this exact structure:

```
LANE: Consigliere
MODEL: <your model id, e.g. gemini-2.5-pro>
DATE: <ISO 8601>

PRIORS_AUDIT_VERDICT: <HONEST | OVERSTATED | DRESSED_HEURISTIC>
PRIORS_AUDIT_DETAIL:
  - <each contamination found, with file path + how to defuse>

PAPER_TRACK_PUBLICATION_RISK: <LOW | MEDIUM | HIGH>
PAPER_TRACK_ATTACK_SURFACE:
  - <each angle a hostile reviewer would use>

THE_ONE_ARCHITECTURAL_GAP:
  what: <single sentence>
  why_it_matters: <one paragraph>
  implementation_outline: <bullet points fitting in 7-day budget>
  spelke_prior_it_relies_on: <which core knowledge primitive>
  estimated_score_lift_milestone_1: <percentage points, with confidence interval>

WHAT_WE_GOT_RIGHT: <2-4 items — calibrate, do not flatter>

THE_THING_YOU_WOULD_NOT_SAY_OUT_LOUD: <the brutal truth>

PODIUM_CONFIDENCE_MILESTONE_1: <0.0-1.0>
PODIUM_CONFIDENCE_PAPER_TRACK: <0.0-1.0>
BONUS_POOL_CONFIDENCE_PAPER_TRACK: <0.0-1.0>
```

We do not want politeness. We want findings sharp enough to act on this week.
