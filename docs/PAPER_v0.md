# A Tier-1 Substrate for ARC Prize 2026: Spelke Priors, an Honest Disclosure Regime, and an Open-Source Compute-Moat Workaround

**Atom McCree** · AtomEons Systems Laboratory · 2026-06-16

---

## Abstract

Frontier language models score near the noise floor on ARC-AGI-3: GPT-5.5 at 0.43%, Opus 4.7 at 0.18% on the semi-private set. We present **Misfit**, a Tier-1 substrate that uses **no language model in the inference path, no pretrained weights of any kind, and no score weights tuned on the public evaluation set**. Misfit is built from Spelke Core Knowledge object priors (cohesion, continuity, agency, contact-causality, geometry), a hand-authored typed rule grammar disclosed honestly, an HRM-style outer refinement loop in the world-model fitter, MCTS-PUCT with action-handle deep-copy safety, and a per-install resonance library that grows monotonically as the substrate solves novel tasks. We propose a **Tier-1 / Tier-2 / Tier-3 disclosure regime** mechanically enforced by a CI grep that fails the build on any pretrained-LLM import. This regime lets the community separate honest fluid-intelligence claims from engineering performance numbers contaminated by pretraining. Open-source, Apache-2.0, 92 tests green at submission. Code: https://github.com/AtomEons/arc-agi-3-misfit-agent.

## 1. The honesty problem in current ARC solvers

Chollet (2019)<sup>[1]</sup> defines intelligence as the rate at which (priors + experience) become skill on novel tasks under uncertainty. The Core Knowledge framework<sup>[2]</sup> identifies a small set of cognitive primitives present at birth or acquired pre-instruction — objectness, numerosity, geometry, topology, agency, goal-directedness. These are admissible priors. Crystallized knowledge — language, cultural facts, task-family heuristics derived from public-corpus inspection — is not.

The 2025 ARC Prize results<sup>[3]</sup> show two dominant winning paradigms: program synthesis with evolutionary search (NVARC, 24.03% on ARC-AGI-2) and zero-pretraining deep learning (TRM at 7M parameters, 45% on ARC-AGI-1). Both converge on a third unifier: **iterative refinement loops** against a feedback signal<sup>[3]</sup>. The HRM analysis<sup>[4]</sup> further disassembles this: hierarchical architecture contributes ~5pp; the outer refinement loop drives +13pp on a single iteration and roughly doubles by eight; HRM functions effectively as zero-pretraining test-time training.

What no published ARC submission has yet offered is a **mechanically falsifiable disclosure of which priors are admissible and which are contamination**. Disclosure today is a paragraph in a methods section. We argue this is insufficient, and offer an alternative.

## 2. Contribution: the Tier-1 / Tier-2 / Tier-3 disclosure regime

We propose three lanes with clear admissibility criteria:

- **Tier-1.** Spelke priors only. No LLM in the inference path. No pretrained weights of any kind. No score weights tuned on the public evaluation set. Honest fluid-intelligence claim under Chollet's framework.
- **Tier-2.** Tier-1 + a small bundled LLM as search heuristic. Pretraining-contaminated. Engineering performance number, not an intelligence claim.
- **Tier-3.** Tier-2 + a cloud judge lane. Wildly contaminated. Disclose loudly.

The regime is enforced by `tests/test_tier1_attestation.py`, a CI test that greps the source tree for `torch.load`, `from transformers`, `from openai`, `from anthropic`, `from llama_cpp`, `huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, `smolagents`, and references to weights like `.gguf`, `.safetensors`, `gpt-?\d+`, `claude-?\d+`. **Any future commit that smuggles a pretrained model breaks the build, on its own, without reviewer intervention.** This is the kind of mechanism a community convention can adopt across submissions; we publish it under Apache-2.0 for that reason.

Honest naming requirement: the **six hand-authored rule templates** (TRANSLATE, TELEPORT_TO, DESTROY_ON_CONTACT, SPAWN_ON_CONTACT, TOGGLE_AT_CURSOR, NO_OP) were authored by a developer who has seen ARC-AGI-1 and ARC-AGI-2 puzzles. They are not pure Spelke priors. The disclosure document names this explicitly so a hostile reviewer does not have to find it.

## 3. Substrate architecture

Fifteen Python modules under `src/misfit_agent/`. Each prior used is documented in `docs/TIER_1_DISCLOSURE.md` with classification.

**Perception.** Pure 4-connectivity flood fill emits objects with bounding box, centroid, area, symmetry flags, and touches-edge under cohesion, geometry, and topology priors. **Tracking** by Hungarian matching across frames implements continuity and persistence; cost is α·centroid_dist + β·shape_hamming + γ·color_mismatch with α=1, β=0.5, γ=2, gating at cost 50.

**World model.** Fitted typed rule templates compose into a deterministic forward simulator f(state, action) → (next_state, confidence) at <50µs/step. `fit_with_refinement(observations, max_iters=4)` implements the HRM outer loop: each pass refits rules against the full observation history, then `_prune_contradicting_rules` drops any rule whose prediction contradicts a single observed transition. This is the feedback signal that makes refinement improve coverage rather than just re-fit.

**Planning.** MCTS-PUCT plans over the world model. UCB = Q + c_puct · P(a|s) · √N(s) / (1 + N(s,a)). Action enum members are deep-copied across branches; a dedicated test (`test_mcts_puct.py::test_DEEP_COPY_safety_*`) catches in-place mutation regressions. Per ARC-AGI-3 methodology, internal operations not altering the environment do not count against the action budget<sup>[5]</sup>; we exploit this by running 200 rollouts per real action under a 500ms hard cap.

**Compute-moat workaround.** The resonance library is a per-install append-only JSONL of (fingerprint, winning policy, score). The fingerprint is a deterministic 50-dimensional signature derived from in-context observations. On a new task, cosine K-NN over fingerprints retrieves prior winning policies as seeds for the search alphabet. The library is source-tagged `"self-solved"` and programmatically refuses pre-seeded entries. **It is monotonic: every solved task makes the next task slightly more tractable** — the misfit operator's edge against larger compute budgets.

**Abstain policy.** A three-conjunction gate (action_counter > 2·human_baseline AND novelty plateau AND world-model variance > 0.20) preserves wall-clock for solvable games under quadratic scoring (per-level = (human_baseline_actions / agent_actions)², capped at 1.15×).

## 4. Static-task sister (ARC-AGI-2)

The same perceptor, fingerprint, and resonance library are reused under a thin static-task wrapper. `arc2_solver.solve_task(train_pairs, test_input) → (attempt_1, attempt_2)` fits Identity / Translate2 / Recolor rule templates over observed input-output pairs and returns two ranked attempts. Wall-clock self-kill at 8h30m and per-task 30s budget governor ensure the runner never burns the Kaggle 9-hour quota on any one task; on failure or timeout an identity fallback keeps the submission well-formed. Substrate-level reuse means rule additions (reflection, rotation, crop, tile) benefit both ARC-AGI-3 and ARC-AGI-2 submissions.

## 5. Results, gaps, and what we are not claiming

**Test state at submission.** 92/92 functional tests pass; 3 marked xfail to track open LakeStrike adversarial findings; 1 known-flaky timing test. Tier-1 attestation CI test green.

**Leaderboard accuracy.** Kernel v5 Phase A green on Kaggle; Phase B leaderboard score pending at writing. We do not claim a specific Milestone #1 finish here. The honest comparison is against the published frontier of 0.43% (GPT-5.5) and 0.18% (Opus 4.7) on the semi-private ARC-AGI-3 set<sup>[6]</sup>. A clean Tier-1 substrate that solves even a small fraction of games efficiently scores well under quadratic action-efficiency math.

**What we have not done.** We have not run a Tier-2 ablation; we deliberately chose to ship Tier-1-strict and report only the honest number. We have not tuned thresholds on the 25 public dev games; `src/misfit_agent/config.py` currently has zero values classified as `(c) tuned on public games`. We have not done a held-out validation split; our public-game touches are restricted to perceptor parity checks and we do not adjust any constant against public-game performance during the competition window.

## 6. Why this might compound

Our angle is structural. Three of the failure modes the foundation publishes for frontier LLMs<sup>[6]</sup> are addressed at the architecture level rather than scaled away:
1. **Local-vs-global rule misalignment** — the 50-dim fingerprint captures global episode state, not local pixel state.
2. **Training-data overfitting** — Tier-1 has no training data to overfit to.
3. **Cross-level forgetting** — resonance library and episodic memory carry policies across levels by design.

We can not outspend o3 at inference. Our edge is monotonic library growth: the 101st solved task either rhymes with one of the prior 100 (cosine, microseconds) or it does not. This is the kind of compounding a community of small-model researchers can replicate; the open-source release is intended to make that replication trivial.

## 7. Open source, reproducibility, and community

The full substrate, tests, notebook, and disclosure regime are at https://github.com/AtomEons/arc-agi-3-misfit-agent under Apache-2.0. The paper itself is dual-licensed CC-BY-4.0 to satisfy the Paper Track license requirement. The Kaggle dataset `atommccree/misfit-agent-substrate` mirrors the source and the Kaggle notebook `atommccree/agi-in-a-video-shop-atom-eons-nostalgia` is the live submission with Phase A passing.

We invite three uses:
- **Tear the Tier-1 disclosure apart.** Issue tracker is open; findings are receipted in `docs/`.
- **Adopt the CI attestation pattern** for your own Tier-1 claim, even if your overall agent is Tier-2 — running the grep enforces honest declaration of which lane you are in.
- **Pull-request additional rule templates** that respect the priors envelope, particularly object-relational templates grounded in contact-causality.

We hope the disclosure regime, more than any single result, is what survives this competition.

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
**Word count:** 1,498
