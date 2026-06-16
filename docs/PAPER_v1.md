# Audited Experience Acquisition: A Tier-1 Substrate for ARC Prize 2026 with Provenance-Enforced Memory and an Adversarially-Tested Disclosure Regime

**Atom McCree** · AtomEons Systems Laboratory · 2026-06-16

---

## Abstract

Frontier language models score near the noise floor on ARC-AGI-3: GPT-5.5 at 0.43%, Opus 4.7 at 0.18% on the semi-private set<sup>[6]</sup>. We present **Misfit**, a substrate that uses **no language model in the inference path, no pretrained weights, and no score weights tuned on the public evaluation set**. Three contributions: (1) a **Tier-1 / Tier-2 / Tier-3 disclosure regime** mechanically enforced by a CI grep that fails the build on any pretrained-LLM import; (2) **Provenance-Enforced Memory (PEM)**, an eight-field memory contract that distinguishes audited experience acquisition from retrieval-augmented memoization; (3) honest naming of a **hand-authored typed grammar that is Spelke-priors-adjacent but designer-authored** — not pretending the templates are pure core knowledge. We propose a **resonance-on / resonance-off ablation** as the empirical test that separates memory theater from real compounding skill. Open source, Apache-2.0 (code) / CC-BY-4.0 (paper), 92 tests green at submission. Code: https://github.com/AtomEons/arc-agi-3-misfit-agent.

## 1. Honest naming up front

We claim a Tier-1 substrate. We will not pretend the work is purely Spelke-derived. Chollet (2019)<sup>[1]</sup> and the Spelke Core Knowledge framework<sup>[2]</sup> define admissible priors as cognitive primitives present at birth or acquired pre-instruction: cohesion, continuity, contact-causality, agency, geometry, topology, numerosity. Our **perceptor**, **Hungarian tracker**, **50-dim episode fingerprint**, and **closure-law pruning** map cleanly to these priors. Our **six rule templates** (TRANSLATE, TELEPORT_TO, DESTROY_ON_CONTACT, SPAWN_ON_CONTACT, TOGGLE_AT_CURSOR, NO_OP) do not. They are a **hand-authored typed grammar by an author who has been exposed to ARC-AGI-1 and ARC-AGI-2 puzzles**. The "TELEPORT" and "TOGGLE_AT_CURSOR" templates encode designer intuition about grid-action puzzle mechanics, not the infant physics of Spelke's continuity prior.

We name this so a hostile reviewer does not have to find it. The contribution we defend is not the rule list — it is the framework that lets the community separate admissible substrate from designer-encoded scaffolding without taking the author's word for it.

## 2. Contribution: the Tier-1 / Tier-2 / Tier-3 disclosure regime

- **Tier-1.** Spelke priors plus disclosed designer-authored substrate. No LLM in the inference path. No pretrained weights of any kind. No score weights tuned on the public evaluation set.
- **Tier-2.** Tier-1 + a small bundled LLM as search heuristic. Pretraining-contaminated. Engineering number, not intelligence claim.
- **Tier-3.** Tier-2 + cloud judge lane. Wildly contaminated. Disclose loudly.

The regime is enforced by `tests/test_tier1_attestation.py`, which greps the source tree for `torch.load`, `from transformers`, `from openai`, `from anthropic`, `from llama_cpp`, `huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, `smolagents`, and references to weights like `.gguf`, `.safetensors`, `gpt-?\d+`, `claude-?\d+`. **Any future commit that smuggles a pretrained model breaks the build, without reviewer intervention.** This is the kind of mechanism a community convention can adopt across submissions; we publish it under Apache-2.0 for that reason.

## 3. Contribution: Provenance-Enforced Memory (PEM)

The empirical worry with any memory-augmented agent is memoization: a vector database stuffed with public-corpus answers that retrieves at test time. We answer with a memory contract. **PEM admits a memory entry only if eight fields are present and verifiable:**

1. **Source provenance** — what created this entry (self-solve, ablation, hand-seeded)
2. **Contamination tier** — which Tier (1, 2, 3) the entry was produced under
3. **Creation event** — timestamp + episode signature that produced it
4. **Replay pointer** — exact reproduction path
5. **Mutation history** — any post-creation edits with reasons
6. **Expiry / decay rule** — when this entry stops being trusted
7. **Evidence payload** — the observation that justifies the win
8. **Downstream usage receipt** — every retrieval that consumed this entry

Our `src/misfit_agent/resonance.py` implements a subset of PEM. Source provenance is enforced programmatically: entries with `source_tag != "self-solved"` are rejected at write time. Pre-seeding from public ARC corpora is therefore not just discouraged — it is impossible without disabling a test that ships in the repo. PEM is the formalization that turns a per-install JSONL into an auditable substrate.

The distinction matters under Chollet's framework. A vector DB is retrieval over crystallized knowledge. PEM is **audited experience acquisition**: every entry is provably the agent's own work, and every retrieval leaves a trail. Memoization is a failure to meet PEM. We invite the community to adopt the eight-field contract or argue for a sharper one.

## 4. Substrate architecture (summary)

Fifteen Python modules. Each prior is classified in `docs/TIER_1_DISCLOSURE.md` and each threshold in `config.py` carries a `(a) prior / (b) budget / (c) tuned` tag.

**Perception + tracking.** 4-connectivity flood fill emits objects with bounding box, centroid, symmetry, touches-edge under cohesion / geometry / topology priors. Hungarian matching with cost `α·centroid_dist + β·shape_hamming + γ·color_mismatch` (α=1.0, β=0.5, γ=2.0; classified `(c) designer choice, frozen pre-eval`) implements continuity and persistence.

**World model.** Fitted typed rule templates compose into a deterministic `f(state, action) → (next_state, confidence)` simulator. `fit_with_refinement(observations, max_iters=4)` is the HRM-style<sup>[4]</sup> outer loop with `_prune_contradicting_rules` as the feedback signal. The HRM analysis showed +13pp from a single refinement pass and roughly doubling by eight; we cadence at 5 steps with max 4 iterations.

**Planning.** MCTS-PUCT with UCB = Q + c_puct·P(a|s)·√N(s)/(1+N(s,a)). Per ARC-AGI-3 methodology, internal operations not altering the environment do not count against the action budget<sup>[5]</sup>; we exploit this for 200 rollouts per real action under a 500ms hard cap. Action enum members are deep-copied across branches; a dedicated regression test catches in-place mutation leaks.

**Resonance library (PEM-bound).** Per-install JSONL of `(fingerprint, winning_policy, score, source_tag)`. K-NN cosine retrieval over 50-dim signatures. Source-tag enforcement rejects pre-seeded entries at write time.

**Abstain policy.** Three-conjunction gate: `(action_counter > min_actions)` AND `(novelty plateau over last K=5 fingerprint deltas)` AND `(world-model variance > 0.20)`. `min_actions` derives from quadratic scoring break-even at `2 × human_baseline`; we classify this as `(b) budget heuristic` rather than `(a) prior` to avoid the false claim that the multiplier is innate.

## 5. The empirical test: resonance-on / resonance-off ablation

A reviewer should not have to take our word that the resonance library compounds. We propose the following ablation, to be run on the held-out 7 of the 25 public ARC-AGI-3 games (the other 18 are the train fold; the ablation uses the held-out 7 only):

- **Condition A:** Misfit, resonance disabled (no library reads, no library writes)
- **Condition B:** Misfit, resonance enabled (PEM-bound, source-tag enforced)

Reported metrics on the held-out 7:
- Solved episodes (binary win count)
- Mean actions-to-WIN per solved episode
- Abstain rate
- World-model coverage at episode end
- MCTS rollout efficiency (rollouts per real action)
- False-rhyme failures (resonance seeded a wrong policy)

The single number that distinguishes memory theater from real compounding skill is the **mean actions-to-WIN delta** between A and B. If resonance compounds, B beats A on this metric by a margin that grows with library size. If resonance is theater, the delta is within run-to-run variance.

We will publish the ablation result before the Paper Track final submission (2026-11-09). At time of v1 draft, the ablation has not yet been run; reporting it without running it would itself be theater.

## 6. Static-task sister and shared substrate

`src/misfit_agent/arc2_solver.py` reuses the perceptor, fingerprint, and resonance library under a static-task wrapper for ARC-AGI-2. Identity / Translate2 / Recolor rule templates fit over observed input-output pairs and produce two ranked attempts per the competition's 2-attempt spec. Wall-clock self-kill at 8h30m, per-task 30s budget, identity fallback on failure. Rule additions (reflection, rotation, crop, tile) benefit both ARC-AGI-3 and ARC-AGI-2 submissions through shared substrate.

## 7. Honest limitations

**Universality.** Our perceptor is bound to discrete 2D grids with a small color palette. The click quantizer is bound to ACTION6 coordinate emission. The framework does not extend to continuous physics, 1D audio, or arbitrary action spaces. This is a limitation, not a feature. The Tier-1 / PEM framework, however, is action-space-agnostic and is the part we expect to generalize.

**Accuracy ceiling.** 2025 winners on ARC-AGI-2 used hybrid TTT + TRM (NVARC, 24%) and recursive self-refinement diffusion (ARChitects, 16.5%)<sup>[3]</sup>. Our zero-pretraining Tier-1 substrate will not match those numbers in week 1. A clean Tier-1 substrate that solves even a small fraction of games efficiently under quadratic action-efficiency math is a respectable result; it is not a 24% result.

**Designer-authored grammar.** The six rule templates encode designer intuition. Honest classification: they are Spelke-priors-adjacent, not Spelke-derived. A purer Tier-1 would synthesize rule structure compositionally from atomic primitives during the outer refinement loop. We do not yet do this; we name the gap as future work and accept the Theory-criterion penalty.

**Threshold tuning.** Hungarian cost weights (α=1.0, β=0.5, γ=2.0), MCTS reward shaping (+10 WIN / -0.01 per action / +0.10 novel fingerprint), and fingerprint bin counts (16 / 8 / 4) are designer choices frozen before evaluation. `config.py` classifies each as `(c) designer choice`. They will not be re-tuned during the competition window. We will publish a constants-frozen git tag before the first Phase B submission.

## 8. What we hope survives

The substrate, the result, and the rule templates are likely to be improved upon within months. The pieces that we hope survive longer:

- **The Tier-1 / Tier-2 / Tier-3 disclosure regime** as a community convention.
- **PEM as a memory contract** that distinguishes audited experience acquisition from memoization.
- **The CI attestation pattern** as a mechanical guarantee against contamination drift.
- **The honest naming requirement**: rule templates that encode designer intuition are named as such.

We invite the community to adopt these or argue for sharper alternatives. Critique and findings are tracked at the GitHub issue tracker.

---

## References

[1] Chollet, F. (2019). *On the Measure of Intelligence.* arXiv:1911.01547.
[2] Spelke, E. S., & Kinzler, K. D. (2007). Core knowledge. *Developmental Science*, 10(1), 89-96.
[3] ARC Prize Foundation. (2025-12-05). *ARC Prize 2025 Results and Analysis.* https://arcprize.org/blog/arc-prize-2025-results-analysis
[4] ARC Prize Foundation. (2025-08-15). *The Hidden Drivers of HRM's Performance on ARC-AGI.* https://arcprize.org/blog/hrm-analysis
[5] ARC Prize Foundation. (2026). *ARC-AGI-3 Methodology.* https://docs.arcprize.org/methodology
[6] ARC Prize Foundation. (2026-05-01). *Analyzing GPT-5.5 & Opus 4.7 with ARC-AGI-3.* https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis

---

**Code:** https://github.com/AtomEons/arc-agi-3-misfit-agent
**Live notebook:** https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia
**Dataset:** https://www.kaggle.com/datasets/atommccree/misfit-agent-substrate
**License (paper):** CC-BY-4.0 · **License (code):** Apache-2.0
