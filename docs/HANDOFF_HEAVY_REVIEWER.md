# Handoff to Heavy Reviewer — Architecture & Theory

> **Paste this entire file as the prompt to a heavy reviewing model** (GPT-5.5, Opus 4.7, Gemini 2.5 Pro Deep Think, or a deep-research run). Self-contained, no follow-ups needed.

## Your role

You are the **Heavy Reviewer**. Your job is a serious architectural and theoretical critique of a Tier-1 substrate aimed at the ARC Prize 2026 (three competitions). The operator wants the best version possible — be the level of reviewer you would want for your own work.

Reply schema is at the bottom. Stick to it.

## Mission context

**Operator:** Atom McCree, solo independent, AtomEons Systems Laboratory.
**Date:** 2026-06-16.

We are in three ARC Prize 2026 competitions simultaneously, all entered under Kaggle user `atommccree`:

| Slug | Pool | Teams | Deadline | Format |
|---|---|---|---|---|
| `arc-prize-2026-arc-agi-3` | $850k | 1235 | 2026-11-02 (Milestone #1 by 2026-06-30, $25k/$10k/$2.5k) | Interactive episodes, action emission |
| `arc-prize-2026-arc-agi-2` | $700k | 849 | 2026-11-02 | Static input→output pairs, 2 attempts per test grid |
| `arc-prize-2026-paper-track` | $75k base + **$375k rubric-gated bonus** (≥4.5/5 average over 6 criteria → split among qualifying papers) | 72 | 2026-11-09 | 1,500-word writeup, ONE submission per team, CC-BY-4.0 for winners |

Paper Track judges: François Chollet, Greg Kamradt, Mike Knoop, María Cruz.

**Frontier on ARC-AGI-3** (per arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis, 2026-05-01, semi-private set):
- GPT-5.5: **0.43%**
- Opus 4.7: **0.18%**
- Random / brute-force baseline: ~0-1%
- Foundation's own three-failure-mode synthesis: (a) local-vs-global rule misalignment, (b) overfit to training-data analogies, (c) failure to learn cross-level

**Foundation's published pattern** for what scores (per arcprize.org/blog/arc-prize-2025-results-analysis, 2025-12-05 and arcprize.org/blog/hrm-analysis, 2025-08-15):
- Two dominant paradigms: program synthesis with evolutionary search, OR deep learning with zero pretraining
- **Refinement loops** are the unifier — iteratively transform one program/policy into another against a feedback signal
- HRM's hierarchical architecture is worth ~5pp; the outer refinement loop is worth +13pp from one iteration and ~doubles by eight
- 2025 winners on ARC-AGI-2: NVARC 24% (Architects-TTT + TRM hybrid), ARChitects 16.5% (2D masked-diffusion LLM with recursive self-refinement), MindsAI 12.6% (TTFT + augmentation ensembles)
- TRM standalone: 7M params, 45% on ARC-AGI-1 / 8% on ARC-AGI-2

**Chollet 2019 "On the Measure of Intelligence"** (arXiv:1911.01547) is the doctrinal anchor: intelligence = rate at which (priors + experience) become skill on novel tasks under uncertainty. Spelke Core Knowledge priors (objectness, numerosity, goal-directedness, geometry, topology) are the admissible substrate; encoded crystallized knowledge is the contamination.

## Our approach — the Misfit Tier-1 substrate

**The disclosure framework** (mechanically CI-enforced):
- **Tier-1** — Spelke priors only. No LLM in inference path. No pretrained weights. No score weights tuned on the public eval. CI test `tests/test_tier1_attestation.py` greps the source for `torch.load`, `transformers`, `openai`, `anthropic`, `llama_cpp`, `huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, `smolagents`, `.gguf`, `.safetensors`, `gpt-?\d+`, `claude-?\d+`, etc. Build fails on hit.
- **Tier-2** — Tier-1 + small bundled LLM as heuristic. Pretraining-contaminated. Engineering number.
- **Tier-3** — Tier-2 + cloud judge. Wildly contaminated.

**15 modules under `src/misfit_agent/`** (file → role):
- `perceptor.py` — 4-connectivity flood fill over a 0..9-color grid up to 64×64; emits objects with `(color, area, bbox, centroid, touches_edge, v_symmetric, h_symmetric)`. Pure cohesion + geometry + topology.
- `tracker_hungarian.py` — `track(prev, curr)` returns `{prev_idx → curr_idx | None}`. Cost = `α·centroid_dist + β·shape_hamming + γ·color_mismatch` with `α=1.0 β=0.5 γ=2.0` and `GATING_COST=50`. scipy.optimize fallback to greedy.
- `fingerprint.py` — deterministic 50-dim signature over an episode (object stats, palette densities, symmetry rates, action-effect signatures per action enum slot).
- `resonance.py` — per-install JSONL of `(fingerprint, winning_policy, composite_score, source_tag)`. K-NN cosine retrieval. `source_tag` MUST be `"self-solved"` — pre-seeded entries are programmatically rejected.
- `world_model.py` — composes fitted rule templates into deterministic `f(state, action) → (next_state, confidence)`. `fit_with_refinement(observations, max_iters=4)` is the HRM-style outer loop with `_prune_contradicting_rules` as the feedback signal.
- `rules/translate.py`, `rules/no_op.py` — typed rule templates with `.fit(observations) → bool` and `.predict(state, action) → state`. Each rule ≤3 free params.
- `goal_inducer.py` — `GoalInducer` ranks hypotheses across 3 families: `removed_all_of_class(X)`, `agent_reached_class(Y)`, `count_of_class_equals_N(Z, N)`. Posterior is Laplace-smoothed `support/(support+contradictions)`.
- `mcts_puct.py` — PUCT planner. `UCB = Q + c_puct·P(a|s)·√N(s)/(1+N(s,a))`. `P(a|s)=1.0` if `a` in `last_known_progress_path` else `0.5`. Reward: `+10` predicted WIN, `−0.01` per action, `+0.10` novel fingerprint. Depth 6, 200 rollouts, 500ms hard cap. Action enum members deep-copied across branches (critical mutation-safety property; verified by `test_mcts_puct.py::test_DEEP_COPY_safety_*`).
- `click_quantizer.py` — for ACTION6, collapses 4096-cell click space to 5-20 candidates: object centroids + bbox corners + edge midpoints + 9-quadrant fallback.
- `abstain_policy.py` — returns `should_abstain=True` when `(action_counter > min_actions)` AND `(novelty plateau over last K=5 fingerprint deltas)` AND `(world-model variance > 0.20)`. `min_actions` derives from `max(config_floor, 2 × human_baseline)` per quadratic-scoring math.
- `arc2_solver.py` — sister for static ARC-AGI-2. Rule layer: `Identity / Translate2 / Recolor`. Bounded beam (width 4). Returns `(attempt_1, attempt_2)` per the competition's 2-attempt spec.
- `arc2_runner.py` — orchestration. `load_challenges` reads `arc-agi_evaluation_challenges.json`. 8h30m wall-clock self-kill, 30s per-task budget. Identity fallback on crash/timeout keeps submission well-formed.
- `config.py` — `FrozenConfig` with every threshold classified `(a)` derived-from-prior / `(b)` budget-heuristic / `(c)` tuned-on-public-games. Currently zero `(c)` values — tuning gate is closed.
- `misfit_agent.py` — `Misfit(Agent)` class. `is_done`: WIN → abstain_policy → wall-clock. `_maybe_refit_world_model`: calls `fit_with_refinement(max_iters=4)` every 5 steps with `observe_hungarian` correspondences folded in. `choose_action`: gates MCTS when `world_model.coverage() ≥ 0.30`; falls back to `select_action` (priors + 1-step world-model lookahead + click quantizer) below.
- `episode.py` — `EpisodeTracker` (scenes, action history, transition signals). `observe_hungarian(prev, curr)` produces per-class correspondence dicts for `world_model.fit`.

**Compute-moat workaround:** the resonance library is monotonic — every solved task appends `(fingerprint, winning_policy)`. The 101st task either *rhymes* with one of the prior 100 (50-dim cosine) or it doesn't; rhyming gives the search a program seed in microseconds. Per-install isolated, never pre-seeded from public corpora.

**Test state:** 92 passed + 3 xfailed (LakeStrike Goose findings marked xfail until fixed) + 1 flaky (`test_hard_timeout_is_respected_within_tolerance` fired 780ms vs 500ms+220ms grace — pre-existing timing issue, not from this work).

**Public surfaces live:**
- https://github.com/AtomEons/arc-agi-3-misfit-agent (Apache-2.0)
- https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia (kernel v5, Phase A pending)
- https://www.kaggle.com/datasets/atommccree/misfit-agent-substrate (public)

**Paper Track 6-criterion rubric** (each 0-5, avg final, ≥4.5 unlocks $375k bonus pool split):
- Accuracy (leaderboard, via submission ID linkage)
- Universality (generalizes beyond ARC)
- Progress (community contribution)
- Theory (WHY > HOW)
- Completeness
- Novelty

Self-score target: 28/30 ≈ 4.67/5. Word limit 1,500. ONE submission per team. Drafted at `docs/PAPER_DRAFT.md`.

---

## What we want from you, Heavy Reviewer

Return a **serious architectural review** with the following:

### 1. Architectural assessment

Strengths and weaknesses of the substrate as a system. Not module-by-module — system-level. Specifically:
- Is the coverage-gated split between `select_action` (below 0.30 coverage) and MCTS-PUCT (above) the right hinge? Or is the threshold itself a problem?
- The outer refinement loop fires every 5 steps with `max_iters=4`. Is the cadence × iteration count appropriate, or is it under/over-utilizing the HRM lever?
- The resonance library K-NN retrieval is cosine over a 50-dim vector. Is the embedding dimension + retrieval scheme appropriate, or does it have a discrimination problem at scale (>400 entries)?
- The Hungarian tracker uses fixed cost weights (α=1.0, β=0.5, γ=2.0). Should these be learnable per-game from observation, or held fixed?

### 2. Theory-level critique

The framework claims fluid intelligence under Spelke priors. Where does the WHY (Theory rubric criterion = 1/6 of paper score) fall short?
- The hand-authored rule grammar is disclosed as "ARC-exposed author bias." Is that disclosure enough, or is it a 2/5 ceiling on Theory and Novelty?
- The Chollet framework (`priors + experience → skill rate`) is anchored. Is our resonance library a legitimate `experience` contributor, or is it dressed-up memoization?
- The Tier-1/Tier-2/Tier-3 disclosure regime is novel framing — is it sufficiently rigorous to publish, or does it need formalization (e.g. quantified contamination scores)?

### 3. The single highest-leverage intervention

Within 7 days and under Tier-1 constraints, what is the ONE change that most lifts our position across the 3 competitions? Be specific:
- Which file gets modified
- Which test would prove it works
- Which rubric criterion or which leaderboard does it move
- Estimated effort (hours) and estimated lift (percentage points OR rubric point fractions)

You may propose object-relational rule templates, world-model refinements, scoring math, paper-narrative angles, or anything else. ONE proposal.

### 4. Tier-1 contamination audit (do not skip)

Look for encoded crystallized knowledge dressed as Spelke priors:
- The 6 hand-authored rule templates list
- The 50-dim fingerprint bin counts and feature families
- The Goal Inducer 3-family taxonomy
- The MCTS reward shaping constants
- The `min_actions=25` floor (we claim it derives from quadratic scoring; defend or reject)
- The Hungarian cost weights

For each, classify: ADMISSIBLE / SUSPECT-DISCLOSED / BANNED.

### 5. Paper Track positioning

Given the 6-criterion rubric and the $375k bonus pool gate at 4.5/5, calibrate our realistic self-score per criterion if we submit as the current state:
- Accuracy
- Universality
- Progress
- Theory
- Completeness
- Novelty

And what's the **single highest-impact addition to the paper** (not the code) that would lift the lowest-scoring criterion by ≥0.5?

### 6. Reply schema

```
LANE: Heavy Reviewer
MODEL: <your model id>
DATE: <ISO 8601>

ARCHITECTURAL_VERDICT: <ONE OF: ship-as-is / ship-with-one-fix / restructure-needed / architecturally-broken>
ARCHITECTURAL_FINDINGS: <prose, system-level, 250-500 words>

THEORY_CRITIQUE: <250-500 words>

THE_ONE_HIGH_LEVERAGE_INTERVENTION:
  file: <path>
  test: <test file + assertion>
  rubric_or_leaderboard_moved: <which>
  effort_hours: <number>
  estimated_lift: <units + confidence interval>
  why_this_one: <one paragraph>

TIER_1_CONTAMINATION_AUDIT:
  - <each item, with classification ADMISSIBLE | SUSPECT-DISCLOSED | BANNED, and one-sentence justification>

PAPER_TRACK_SELF_SCORE_PER_CRITERION:
  accuracy: <0-5> — <why>
  universality: <0-5> — <why>
  progress: <0-5> — <why>
  theory: <0-5> — <why>
  completeness: <0-5> — <why>
  novelty: <0-5> — <why>
  predicted_average: <number>
  bonus_pool_eligible: <YES | NO | MAYBE>

THE_SINGLE_HIGHEST_IMPACT_PAPER_ADDITION:
  what: <one paragraph>
  which_criterion_it_lifts: <name>
  estimated_lift_in_that_criterion: <0.0-2.0>

WHAT_WE_GOT_RIGHT: <2-4 specific items>

WHAT_NO_ONE_ELSE_HAS_TOLD_US_YET: <the thing experienced reviewers would say only privately>

OVERALL_PODIUM_CONFIDENCE:
  milestone_1_top_3: <0.0-1.0>
  paper_track_top_3: <0.0-1.0>
  paper_track_bonus_pool_eligible: <0.0-1.0>
```

We are looking for the kind of feedback you would give a student you respect. Sharp. Specific. Actionable in 7 days.
