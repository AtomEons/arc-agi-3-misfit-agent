# misfit-agent: Tier-1 ARC substrate (Kaggle dataset)

Source code for the misfit-agent — a Tier-1 Spelke-priors substrate for the ARC Prize 2026 (ARC-AGI-3 + ARC-AGI-2 + Paper Track).

**No LLM in the inference path.** **No pretrained weights.** Mechanically enforced via CI grep (`test_tier1_attestation.py`).

## Companion artifacts

- **GitHub repo (Apache-2.0):** https://github.com/AtomEons/arc-agi-3-misfit-agent
- **Live Kaggle notebook:** https://www.kaggle.com/atommccree/agi-in-a-video-shop-atom-eons-nostalgia
- **Tier-1 disclosure doc:** https://github.com/AtomEons/arc-agi-3-misfit-agent/blob/main/docs/TIER_1_DISCLOSURE.md
- **Competitor intel synthesis:** https://github.com/AtomEons/arc-agi-3-misfit-agent/blob/main/docs/COMPETITOR_INTEL.md

## Substrate modules

| Module | Spelke prior |
|---|---|
| `perceptor.py` — 4-connectivity flood fill, objects with bbox/centroid/symmetry/touches_edge | cohesion, geometry, topology |
| `tracker_hungarian.py` — Hungarian matching of object identity across frames | continuity, persistence |
| `fingerprint.py` — 50-dim deterministic episode signature | numerosity + geometry stats |
| `resonance.py` — per-install JSONL of (fingerprint, winning policy), source-tagged self-solved only | experience (Chollet's allowed input) |
| `world_model.py` — composes typed rule templates with HRM-style outer refinement loop | compositionality, sparse causality |
| `goal_inducer.py` — three hypothesis families, ≤3 free params each | goal-directedness, numerosity |
| `mcts_puct.py` — PUCT planner with action deep-copy safety + progressive widening | budget-aware search |
| `abstain_policy.py` — quadratic-scoring-derived budget gate | scoring math |

## License

Apache-2.0. See [LICENSE](https://github.com/AtomEons/arc-agi-3-misfit-agent/blob/main/LICENSE).
