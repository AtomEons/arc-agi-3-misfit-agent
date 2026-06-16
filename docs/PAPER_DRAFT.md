# A Spelke-Priors-Only Substrate for ARC-AGI: Tier-1 Honest Fluid Intelligence Under Frozen Weights

**Atom McCree** · AtomEons Systems Laboratory · 2026

> **Working draft.** This document is the living paper for the ARC Prize 2026 Paper Track. Sections fill in as the work happens. Every claim is anchored to a receipt path under `receipts/`. Every cite is a real paper with a URL — no fabricated references.

---

## Abstract (TODO — draft after week 2 results)

- One paragraph (~150 words)
- Claim, method, result, contribution
- Frame the Tier-1/Tier-2/Tier-3 disclosure regime as the contribution, results as evidence

## 1. Introduction

### 1.1 The fluid-vs-crystallized distinction
- Cite Chollet 2019 "On the Measure of Intelligence" arXiv:1911.01547
- Cite Spelke & Kinzler 2007 "Core knowledge"
- Cite Lake, Ullman, Tenenbaum, Gershman 2017 "Building machines that learn and think like people"
- Frame the contamination problem: pretrained LLMs smuggle crystallized knowledge into agents that claim fluid intelligence

### 1.2 The Tier-1/Tier-2/Tier-3 disclosure regime (our contribution)
- **Tier 1:** Spelke core priors only. No LLM. No pretrained weights. Honest fluid-intelligence claim.
- **Tier 2:** Tier 1 + frozen LLM heuristic. Disclosed pretraining contamination. Engineering performance number.
- **Tier 3:** Tier 2 + cloud judge lane. Wildly contaminated. Disclose loudly.
- Mechanically enforced via CI grep (`test_tier1_attestation.py`)

### 1.3 Contributions
1. Tier-1 substrate for ARC-AGI-3 interactive games
2. Tier-1 substrate for ARC-AGI-2 static tasks (shared modules)
3. Resonance library as a compute-moat workaround (experience-only)
4. Honest disclosure regime that survives adversarial review
5. Open-source Apache-2.0 implementation, reproducible end-to-end

## 2. Related work

### 2.1 ARC-AGI background
- Chollet 2019 (foundational)
- ARC Prize 2024 winners (cite MindsAI, icecuber, Greenblatt)
- Hierarchical Reasoning Model (Sapient Intelligence 2025) — 27M params, ARC-AGI-1 ~30-40%
- (Filled by recon workflow `wndb5xhl9`)

### 2.2 Object-centric perception
- Greff et al. 2020 "On the binding problem in artificial neural networks"
- Spelke core knowledge papers
- (Filled by recon workflow)

### 2.3 Program synthesis from priors
- DreamCoder (Ellis et al. 2021)
- FunSearch (DeepMind 2024)
- (Filled by recon workflow)

### 2.4 World models + MCTS
- Ha & Schmidhuber 2018
- Dreamer lineage (Hafner et al.)
- MuZero / EfficientZero
- (Filled by recon workflow)

## 3. Substrate architecture

### 3.1 Perceptor under Spelke priors
- 4-connectivity flood fill → objects with bbox/centroid/symmetry/touches_edge
- All cohesion, geometry, topology derived from in-context observations
- File: `src/misfit_agent/perceptor.py`
- Test: `tests/test_substrate_smoke.py::test_perceive_*`

### 3.2 Typed rule grammar (hand-authored — disclosed)
- Six templates: TRANSLATE, TELEPORT_TO, DESTROY_ON_CONTACT, SPAWN_ON_CONTACT, TOGGLE_AT_CURSOR, NO_OP
- HONEST DISCLOSURE: hand-authored by an author exposed to ARC-AGI-1/2. Not pure Spelke.
- Each fits from observed (s, a, s') tuples with consistency check
- File: `src/misfit_agent/rules/*.py`

### 3.3 World model composer
- Composes fitted rule library into f(state, action) → (next_state, confidence)
- <50µs/step → enables free MCTS rollouts (per ARC-AGI-3 methodology: "internal operations not counted")
- File: `src/misfit_agent/world_model.py`

### 3.4 50-dim episode fingerprint
- Object stats, palette densities, symmetry rates, persistence proxies
- File: `src/misfit_agent/fingerprint.py`

### 3.5 Resonance library
- Per-install JSONL of (fingerprint, winning_policy)
- Source-tagged: only `self-solved` entries admitted (Tier-1 honesty)
- K-NN retrieval for cross-game compounding
- File: `src/misfit_agent/resonance.py`

### 3.6 ACTION6 click quantizer
- Reduces 4096-cell click space to ~5-20 candidates via objectness prior
- 400× search-efficiency win on click-required ARC-AGI-3 games

### 3.7 MCTS-PUCT planner (Day 8 deliverable)
- UCB with deep-copied GameAction across branches (avoids in-place mutation leak)
- File: `src/misfit_agent/mcts_puct.py` (in flight via workflow wk1vabrot)

### 3.8 AbstainPolicy
- Returns is_done=True when (novelty plateau AND high world-model variance AND past min-actions)
- Derived from quadratic scoring math, not asserted

## 4. Experimental setup

### 4.1 ARC-AGI-3 evaluation
- Kaggle competition `arc-prize-2026-arc-agi-3`, 1234 teams
- 110 private games, no internet during eval
- Scoring: per-level (human_baseline_actions / ai_actions)^2 capped at 1.15×
- Wall clock: 8h55m hard cap
- Submission cap: 5 / day

### 4.2 ARC-AGI-2 evaluation
- Kaggle competition `arc-prize-2026-arc-agi-2`, 849 teams
- Static tasks: 2 predictions per test input, 1 if either matches
- Public eval: 400 tasks (`arc-agi_evaluation_challenges.json`, 985 KB)

### 4.3 Tier-1 attestation methodology
- Mechanical CI grep for forbidden imports (`torch.load`, `transformers`, `openai`, etc.)
- Frozen config with each constant classified (a) prior / (b) budget / (c) tuned
- 18 train / 7 val split on public games — thresholds tuned only on the 18

## 5. Results (TODO — fill from Kaggle leaderboard)

### 5.1 ARC-AGI-3 Milestone #1 (2026-06-30)
- TBD — receipts at `receipts/arc/milestone_1_*.json`

### 5.2 ARC-AGI-2 public eval baseline
- TBD — receipts at `receipts/arc-agi-2/*.json`

### 5.3 Resonance library compounding ablation
- TBD — measure score lift with library on vs off

### 5.4 Tier-1 vs Tier-2 ablation (intentionally not performed)
- We do not run Tier-2 for Milestone #1 to preserve the honest Tier-1 claim
- Future work: Tier-2 with bundled Mamba-2 GGUF, disclosed contamination

## 6. Discussion

### 6.1 Where structured priors beat scale
- Quadratic scoring rewards efficiency × number of games
- LLM agents burn token budget; substrate burns CPU budget
- Per-level math: substrate at 15 actions vs LLM at 60 = 16× advantage

### 6.2 Honest gaps
- Typed rule grammar IS hand-authored by ARC-exposed author (disclosed)
- 50-dim fingerprint bin counts are designer choices
- World model assumes object-physics game class — degrades to AbstainPolicy on non-physics games

### 6.3 The bitter lesson check
- (Filled from research workflow `wndb5xhl9`)

## 7. Limitations

- Single-author, 11-day sprint
- Hardware: Kaggle T4 (CPU-bound substrate)
- No transfer to non-grid environments
- No real-time human-baseline comparison study (use public posted baselines)

## 8. Conclusion

(Drafted after results land.)

---

## Appendix A — Tier-1 attestation receipts

- `docs/TIER_1_DISCLOSURE.md` — full priors audit, ships in cell-0 of submission notebook
- `tests/test_tier1_attestation.py` — CI grep enforcement
- `src/misfit_agent/config.py` — every threshold classified (a)/(b)/(c)

## Appendix B — Receipts manifest

- See `docs/RECEIPTS_MANIFEST.json` for the full reproducibility index

## Appendix C — Decisions ledger

- See `docs/DECISIONS_LEDGER.jsonl` for every architectural decision + rationale + alternative considered

## Appendix D — Bibliography

- See `docs/CITATIONS.bib` for BibTeX
- Populated by research workflow `wndb5xhl9`

---

## Writeup discipline (for the author, not the paper)

- **Every claim has a receipt.** No "results show" without a JSON path that produced the result.
- **Every cite has a URL.** No fabricated references — only papers I actually read or had a sub-agent read.
- **Every fix has a decision ledger entry.** Why we changed it, what we considered, what receipt or judge finding motivated it.
- **Every threshold has a classification.** (a) prior / (b) budget / (c) tuned. Anything (c) gets disclosed.
- **No claim survives without a test.** If the test is missing, the claim is downgraded.
