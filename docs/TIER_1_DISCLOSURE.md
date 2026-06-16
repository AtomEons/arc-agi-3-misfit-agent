# Tier-1 Disclosure

This document accompanies every Kaggle submission as a cell-0 markdown block.

## Claim

This agent uses **Spelke Core Knowledge object priors** combined with a **hand-authored typed rule grammar over those priors**. There are no pretrained model weights of any kind. There is no language model. The codebase is Apache-2.0 licensed.

## Priors used (Tier 1)

| Prior | How it appears in code |
|---|---|
| **Cohesion** | 4-connectivity connected-component labelling in `perceptor.py` — same-color contiguous cells form one object |
| **Continuity / persistence** | Hungarian matching of object identities across consecutive frames in `tracker.py` |
| **Contact-causality** | State changes triggered by spatial contact, encoded in `rules/destroy_on_contact.py` and `rules/spawn_on_contact.py` |
| **Persistence under occlusion** | Hungarian cost prefers hidden-then-reappear over destroyed-then-spawned |
| **Agency vs patient** | The object whose centroid correlates with ACTION1-4 deltas is *the agent* — learned correlation, not hardcoded |
| **Compositionality** | Transition function factorizes per object class |
| **Sparse causality** | Most actions affect at most 1-2 objects |

**Citation:** Spelke, E. S., & Kinzler, K. D. (2007). *Core knowledge.* Developmental Science, 10(1), 89-96.

## Hand-authored contribution (honest naming)

The **six rule templates** in `src/misfit_agent/rules/` (TRANSLATE, TELEPORT_TO, DESTROY_ON_CONTACT, SPAWN_ON_CONTACT, TOGGLE_AT_CURSOR, NO_OP) are a typed grammar **authored by an author who has been exposed to ARC-AGI-1 and ARC-AGI-2 examples**. The grammar is not derived purely from Spelke priors — it encodes a designer's intuition about how grid-action puzzles tend to be structured.

This is disclosed explicitly. A hostile reviewer asking "did the author's exposure to ARC examples inform the template list?" — the answer is yes, and we name it.

## What we do NOT use (Tier 2 contamination — intentionally excluded)

- No language model proposer (no GPT, no Claude, no Gemini, no Mamba, no Qwen, no Llama)
- No pretrained vision encoder
- No ARC-AGI-1 / ARC-AGI-2 task descriptions or solutions in any prompt or training corpus
- No fine-tuned LoRA weights
- No `transformers`, `openai`, `anthropic`, `llama_cpp`, `huggingface_hub`, `sentence_transformers`, `langchain`, `langgraph`, `smolagents` imports

This is mechanically enforced by `tests/test_tier1_attestation.py` which fails the CI build if any forbidden import or pretrained-model string appears in the source tree.

## What we do NOT do (Tier 3 — forbidden)

- No Kaggle private-set leakage (we never see private games)
- No threshold sweeping on the 25 public dev games after Day 10 freeze
- No scraping of past ARC Prize winners' code or solutions
- No human-in-the-loop hint injection at evaluation time
- No out-of-band channel to the gateway

## Frozen-constants policy

All magic numbers live in `src/misfit_agent/config.py` with explicit classification:
- `(a)` derived from a prior (geometric truth, Spelke core)
- `(b)` budget heuristic (math from the scoring rule or wall-clock)
- `(c)` tuned on public games (must be disclosed)

Any threshold change after the first Kaggle submission requires a git tag, a CHANGELOG entry, and a named judge finding that justified it.

## Tuning split

The 25 public games are split **18 train / 7 held-out validation**. Thresholds are tuned only on the 18, with held-out numbers reported separately in `METHODOLOGY.md`. We do not sweep on all 25 and report a single number.

## Source

[https://github.com/AtomEons/arc-agi-3-misfit-agent](https://github.com/AtomEons/arc-agi-3-misfit-agent) (repo URL placeholder — repo not yet pushed)
