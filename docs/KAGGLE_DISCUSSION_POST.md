# A Tier-1 Honest Substrate: No LLM, no pretraining, just Spelke priors + experience

> Open-source: https://github.com/AtomEons/arc-agi-3-misfit-agent (Apache-2.0)
> Public notebook: https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia

I want to share an approach and a disclosure regime I've been developing for ARC-AGI-3 (Milestone #1, June 30) that doesn't use any LLM in the inference path. It's open-sourced as a public reference for anyone curious about a substrate-only angle to interactive ARC.

## The disclosure framework

I'm calling it **Tier-1 / Tier-2 / Tier-3** so the honesty claim survives review:

- **Tier-1** — Spelke core priors only. No LLM. No pretrained weights of any kind. No score weights tuned on the public eval. The agent's intelligence claim is fluid-intelligence-under-priors per [Chollet 2019](https://arxiv.org/abs/1911.01547).
- **Tier-2** — Tier-1 + a small bundled LLM as a search heuristic. Pretraining-contaminated. Engineering performance number, not an intelligence claim.
- **Tier-3** — Tier-2 + cloud judge lane. Wildly contaminated. Disclose loudly.

**The Tier-1 attestation is mechanically enforced** by a CI test ([`test_tier1_attestation.py`](https://github.com/AtomEons/arc-agi-3-misfit-agent/blob/main/tests/test_tier1_attestation.py)) that greps the source tree for forbidden imports — `torch.load`, `transformers`, `openai`, `anthropic`, `llama_cpp`, `huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, `smolagents`, and pretrained-model strings. The build fails if any sneak in. Any future commit that smuggles a model breaks the contract on its own.

## What's in the substrate

| Module | What it does | Spelke prior |
|---|---|---|
| `perceptor.py` | 4-connectivity flood fill → objects with bbox/centroid/symmetry/touches_edge | cohesion, geometry, topology |
| `tracker_hungarian.py` | Hungarian matching of object identity across frames | continuity, persistence |
| `fingerprint.py` | 50-dim deterministic episode signature | numerosity + geometry stats |
| `resonance.py` | Per-install JSONL of (fingerprint, winning policy) — source-tagged `self-solved` only | experience (Chollet's allowed input) |
| `world_model.py` | Composes fitted typed rule templates into f(state, action) → next_state, with HRM-style outer refinement loop (1-4 iters) | compositionality, sparse causality |
| `goal_inducer.py` | Three hypothesis families (removed_all_of_class, agent_reached_class, count_of_class_equals_N), max 3 free params per | goal-directedness, numerosity |
| `mcts_puct.py` | PUCT planner with action deep-copy safety + progressive widening for ACTION6 | budget-aware search |
| `abstain_policy.py` | Returns `is_done=True` when (action_counter > 2× human baseline AND novelty plateau AND world-model variance high) | scoring-derived budget gate |

The honest naming: the **six typed rule templates** (TRANSLATE, TELEPORT_TO, DESTROY_ON_CONTACT, SPAWN_ON_CONTACT, TOGGLE_AT_CURSOR, NO_OP) are a **hand-authored grammar by an author who has been exposed to ARC-AGI-1 and ARC-AGI-2 examples**. They're not pure Spelke priors. The disclosure doc names this explicitly so a hostile reviewer doesn't have to find it.

## The compute-moat workaround

I can't outspend o3 at inference. The substrate's edge is **monotonic library growth**:

- Every solved task appends a `(fingerprint, winning_policy)` pair to a per-install JSONL
- On a new task, cosine-K-NN over fingerprints retrieves prior winning policies as **seeds** for the search alphabet
- After 100 solved tasks, the 101st either *rhymes* with one of the prior 100 or it doesn't — and "rhymes" is a 50-dim cosine query, microseconds
- The library is source-tagged `self-solved`; it cannot be pre-seeded from public corpora without failing the Tier-1 honesty test

## Why this might podium even with a modest score

Per the April 2026 [ARC blog on GPT-5.5 / Opus 4.7](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis): GPT-5.5 scored **0.43%** and Opus 4.7 scored **0.18%** on ARC-AGI-3. The current frontier is at the noise floor. A clean Tier-1 substrate that solves even a handful of games efficiently beats that quadratically: `score = (human_baseline / agent_actions)^2`.

Per the August 2025 [HRM analysis](https://arcprize.org/blog/hrm-analysis): the hierarchical architecture only contributes ~5pp; the **outer refinement loop** is HRM's real hidden driver — +13pp from 0→1 refinement, doubled from 1→8. That's now in `world_model.fit_with_refinement(max_iters=4)` in this repo.

## What I want from the community

- **Tear the Tier-1 disclosure apart.** If you find any encoded crystallized knowledge dressed as Spelke priors, please open an issue. Specifically suspect: the six rule templates, the fingerprint bin counts, and any threshold not yet classified `(a)`/`(b)`/`(c)` in `src/misfit_agent/config.py`.
- **Try the CI grep on your own agent.** Even if your stack is fully Tier-2 (LLM-augmented), running the attestation test against your own modules forces an honest declaration of which lane you're in. I'd love to see a community convention emerge here.
- **PRs welcome** on additional rule templates that respect the priors constraint — particularly object-relational rules grounded in contact-causality.

## Receipts

- 79/79 tests green ([test files](https://github.com/AtomEons/arc-agi-3-misfit-agent/tree/main/tests))
- Tier-1 attestation CI test green
- Kaggle notebook v4 currently in Phase A
- All commits, all decisions tracked in [docs/](https://github.com/AtomEons/arc-agi-3-misfit-agent/tree/main/docs)

If you have feedback — especially adversarial findings — I'd value them. The honest path is harder than the LLM-heuristic path, and I'd rather find the holes before the private set does.

— Atom McCree / AtomEons Systems Laboratory

---

*Tagged: ARC Prize 2026, ARC-AGI-3, open source, Tier-1, Spelke priors, no LLM, refinement loops*
