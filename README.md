# misfit-agent — ARC-AGI-3 Tier-1 Substrate

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 79 passing](https://img.shields.io/badge/tests-79%20passing-brightgreen.svg)](#)
[![Tier-1: priors only](https://img.shields.io/badge/Tier--1-priors%20only-orange.svg)](docs/TIER_1_DISCLOSURE.md)
[![Kaggle notebook](https://img.shields.io/badge/Kaggle-notebook-20BEFF.svg)](https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia)

A solo-misfit submission for the [ARC Prize 2026 — ARC-AGI-3 Competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) on Kaggle.

**Live Kaggle notebook (public, v4):** https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia

**Target:** Milestone #1 ($25,000), deadline **2026-06-30**.

## What this is

A pure **Spelke Core Knowledge priors** substrate for ARC-AGI-3 interactive games. No pretrained LLM in the inference path. No internet during evaluation. No cross-episode memorization of test answers. The agent compounds **experience** via a per-install resonance library — every game it solves makes the next game slightly easier.

This is the honest fluid-intelligence claim under [Chollet's intelligence framework](https://arxiv.org/abs/1911.01547). It is Tier-1 by construction.

## Why Tier-1 only

Per ARC-AGI design philosophy, an agent's score should measure fluid intelligence — the rate at which the agent converts priors + experience into skill on novel tasks. Crystallized knowledge (LLM pretraining on ARC papers, hand-crafted task-family heuristics, score weights tuned on the public eval) measures the developer's cleverness, not the agent's intelligence.

We hold ourselves to the harder bar:
- **No LLM heuristic.** No Mamba, no GPT, no Claude proposing actions.
- **No oracle-encoded task families.** Every signal is derived from in-context observations.
- **No pre-seeded resonance library.** The library grows only from the agent's own solved episodes.
- **No score tuning on public eval.** We tune on a held-out fold only.

Tier-2 (LLM-augmented) and Tier-3 (cloud-judge) numbers, if we ever produce them, are reported separately and labeled as engineering performance, not intelligence claims.

## Architecture

```
arcengine.FrameData ──► perceptor (objectness, geometry, numerosity priors)
                            │
                            ▼
                       episode tracker  ◄── frames history
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
            fingerprint           rule induction
            (50-dim sig)          (state, action, next_state) → policy
                  │                   │
                  ▼                   │
            resonance lookup          │
            (per-install JSONL)       │
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    action policy search
                    (oracle-pruned, budget-aware)
                            │
                            ▼
                       GameAction
```

Each module relies only on a documented set of Spelke priors. See [PRIORS.md](docs/PRIORS.md) for the audit.

## Quickstart

```bash
# Install (uses uv)
uv sync

# Local play against a single game
make play-local GAME=locksmith

# Submit to Kaggle
make submit
```

## Repo layout

```
misfit-agent/
├── src/misfit_agent/
│   ├── __init__.py             # registers MisfitAgent with the ARC framework
│   ├── misfit_agent.py         # the Agent subclass — choose_action + is_done
│   ├── perceptor.py            # Spelke priors: objectness, geometry, numerosity
│   ├── fingerprint.py          # 50-dim episode fingerprint
│   ├── resonance.py            # per-install JSONL library + K-NN retrieval
│   ├── episode.py              # episode state tracker (history, transitions)
│   ├── rule_induction.py       # (state, action, next_state) → policy hypotheses
│   └── action_search.py        # oracle-pruned action policy search
├── tests/
├── docs/PRIORS.md              # honest priors audit
├── docs/METHODOLOGY.md         # how scores get reported
├── pyproject.toml
└── Makefile
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

- François Chollet & the ARC Prize Foundation for the benchmark
- Elizabeth Spelke for the Core Knowledge framework

---

> "we need to get past what is expected. we need to get to something unknown. the unknown waters. we want a 100 or past score."
>
> — operator directive, 2026-06-15
