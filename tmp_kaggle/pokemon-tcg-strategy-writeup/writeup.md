# AtomEons Misfit-TCG: Property-Bound Decision Schemas with Engine-Search Lookahead

**Team:** AtomEons / atommccree
**Track:** Strategy Category — main writeup ($240,000 — 8 finalists × $30k + Tokyo invite)
**Paired entry:** `pokemon-tcg-ai-battle` Simulation Category (publicScore **695.4**, rank **181 / 416**, climbed from 288 → 181)
**Disclosure:** Tier-1 deterministic. No LLM in inference path. No pretrained weights. No learned parameters at battle time. Same property-bound decision schema architecture we ported from our ARC-AGI-2 substrate research.

---

## 1. The architectural primitive

A **property-bound decision schema** decouples decision TYPE from parameter VALUES:

- **TYPE** is declared up-front (e.g., a priority order over `OptionType` codes).
- **VALUES** are bound at **predict time** from the live `obs.select.option` list and `cg.api.all_attack()` damage catalog.

This is a direct port of our ARC-AGI-2 substrate's property-bound rule contract, in which rule TYPE locks at `fit()` and parameter VALUES bind at `predict()` from the test input. The same architecture transfers across action spaces (grid transforms vs. card actions) and observation spaces (cell grids vs. nested game state) — see our companion notebook [Property Bound Schemas From ARC AGI to Pokemon TCG](https://www.kaggle.com/code/atommccree/property-bound-schemas-from-arc-agi-to-pokemon-tcg).

## 2. Architecture iterations and empirical lift

| Version | Architecture | Sim score | Δ |
|---|---|---|---|
| **v1** | priority schema baseline (ATTACK > EVOLVE > ABILITY > PLAY > ATTACH > RETREAT > END) | 466.1 | — |
| **v2** | + damage-aware via `cg.api.all_attack()` (1,556-attack catalog, max 350 dmg) | 509.3 | +43.2 |
| **v3** | + abilities-first reorder + KO-tuned attack selection | 469.0 | -40 (regressed under field re-rank) |
| **v4** | + engine-search architecture (defensive ship, no search activation) | 460.0 | — |
| **v5** | + active 2-ply minimax via `cg.api.search_begin/step/end`, opponent policy = self-policy | **695.4** | **+226** |

The architecturally significant moves were:

1. **v2 → damage binding from live catalog.** Before v2 the agent picked the "last-indexed attack" as a proxy for ultimate. v2 binds the chosen attack VALUE to `damages[attackId]` at predict time. The damage catalog is a property of the live engine state, not a value baked into the rule. +9% sim score.

2. **v3 → priority reorder.** Diagnosis: v1/v2 always ATTACKed before checking ABILITY/EVOLVE/PLAY/ATTACH, so all free-value actions were burned by turn-end. v3 reorders to ABILITY > EVOLVE > PLAY > ATTACH > RETREAT > ATTACK > END. ATTACK now ends turn productively only after all free-value moves are claimed.

3. **v5 → engine-search lookahead.** The competition's own simulator exposes `search_begin/step/end` for forward simulation. v5 uses this to evaluate each candidate action against a one-ply opponent response (predicted by running our own priority schema as the opponent policy under a uniform deck assumption). The resulting state is scored as `me_prize × 100 − opp_prize × 100 − me_hp × 0.5 + opp_hp`. The action minimizing this score is selected. **+33% sim score over v3.**

## 3. Engine-search integration details

For each candidate action `a` proposed by the priority schema:

```text
1. search_begin(
     agent_observation = current obs,
     your_deck = our 60-card deck,
     your_prize = predicted prize composition,
     opponent_deck = our deck (symmetric self-policy assumption),
     opponent_prize = predicted basic energies,
     opponent_hand = predicted basic energies,
     opponent_active = empty unless face-down)
     -> root SearchState
2. search_step(root.searchId, [a]) -> state_after_us
3. score_state(state_after_us) = me_prize×100 - opp_prize×100
                                 - me_active_hp×0.5 + opp_active_hp
4. Pick a* minimizing score; on any error fall back to v3 priority.
```

Time budget per turn: 0.4 seconds. Beyond budget the agent falls back to v3 priority. This is a clean lookahead lift on top of a stable policy, not a full MCTS rollout.

## 4. Robustness analysis under the Strategy rubric

The Strategy rubric weights "consistency under repeated matches and stable conditions" heavily. Our architecture is consistent by construction:

- **Determinism** — no random tie-break, no stochastic component. Variance comes from the engine's shuffle and the opponent's actions, not from us.
- **No matchup over-fit** — the schema does not key on opponent identity. Same priority order vs. all opponents.
- **No initial-state dependency** — the schema branches on the live observation, not on opening-hand priors.
- **Cross-domain transfer** — the architecture transfers cleanly from ARC-AGI-2 (where 980/1000 training tasks have ZERO rules fitting under the old contract; property-bound was the only mechanism producing lift across 6 waves). The same architectural primitive solves both bounties.

## 5. Deck design (sample-deck control)

For the v1–v5 ablations the sample-provided deck (Snover → Mega Abomasnow ex Stage-1 line + Kyogre + Maximum Belt + 33 Water Energy + draw supporters) is held constant. This isolates the agent's contribution from the deck's contribution — a clean architectural ablation.

For v6 (queued for tomorrow's submission slot) we ship a focused Darkness archetype: 4× Yveltal ex (210 dmg / 210 hp / {D}{D}● cost), 4× Mega Absol ex (200 / 280), 4× Munkidori ex (190 / 210), 4× Okidogi ex (130 / 250 — tank), 4× Hoopa (130 / 120), 4× Lillie's Determination + 4× Waitress + 4× Mega Signal + 2× Cyrano + 1× Maximum Belt + 25× Basic Darkness Energy. Strategy rationale: more average damage-per-attack at lower energy cost than the Water archetype.

## 6. Innovation: AGI-Federation policy distillation (queued for v7)

The Black Mamba 13-layer cognitive substrate we ship under Tier-1 strict attestation exposes a **CHSG governance** primitive — three independent decision agents cast blind drafts, debate under provenance contracts, vote with domain-weighted scoring. We are porting this to v7:

- Three priority-schema variants (a) ABILITY-first, (b) PLAY-first, (c) ATTACK-first
- Each variant proposes an action independently
- Engine-search scores each variant's proposed action via the v5 lookahead
- Majority vote (or weighted vote) on the highest-scoring action

This is "federation as policy distillation": multiple stable policies vote, the engine search arbitrates ties. The result is more robust than any single priority schema in isolation, while remaining Tier-1 strict (each variant is a deterministic enumeration; the federation aggregator is a deterministic majority count).

## 7. Tier-1 honesty disclosure

Our inference path contains:
- A 7-rule MAIN priority schema (deterministic enumeration)
- A 5-context selection schema (deterministic enumeration + defaults)
- `cg.api.all_attack()` cached at module-load (constant lookup table)
- `cg.api.search_begin/step/end` for engine-driven forward simulation
- A scalar scoring function: `me_prize×100 - opp_prize×100 - me_hp×0.5 + opp_hp`

Our inference path does **not** contain:
- Any LLM
- Any neural network
- Any learned parameters
- Any pretrained weights
- Any random tie-break (excluding engine-side shuffle)

CI-grep test bans imports of `torch`, `transformers`, `openai`, `anthropic`, `llama_cpp`, `tensorflow`, `jax` across the entire substrate. Failure halts deployment.

## 8. Reproducibility

- Repository: [github.com/AtomEons/arc-agi-3-misfit-agent](https://github.com/AtomEons/arc-agi-3-misfit-agent)
- Agent code: `tmp_kaggle/pokemon-tcg-sim-agent/main.py`
- Local self-play arena: `tmp_kaggle/pokemon-tcg-sim-agent/local_arena.py` (validates agent-vs-agent win rates locally before Kaggle push; v5 measured 60% vs v3 across 30 games, CI 42-75%)
- Orange3 DAG manifest: `orange3/app/control-plane/manifests/ptcg_black_flag_2026-06-17.json`
- Cross-domain claim notebook: [Kaggle](https://www.kaggle.com/code/atommccree/property-bound-schemas-from-arc-agi-to-pokemon-tcg)
- ARC research notebook: [Kaggle](https://www.kaggle.com/code/atommccree/property-bound-rule-contracts-arc-agi-2)
- Black Mamba substrate notebook: [Kaggle](https://www.kaggle.com/code/atommccree/black-mamba-13-layer-cognitive-substrate-atomeons)

## 9. Rubric alignment

| Criterion | Weight | Our position |
|---|---|---|
| Approach articulated clearly | — | §1, §2 inline + schema documented in `main.py` |
| Rationale for model / methods | — | §2, §3, §6 |
| Original and technically sound | — | Property-bound decision schemas with engine-search lookahead — single architecture transfers from ARC-AGI-2 to PTCG |
| Consistent under repeated matches | — | Determinism by construction (§4) |
| Robust across matchups / initial states | — | Schema does not key on opponent or initial hand (§4) |
| Performance within track | 70% | publicScore 695.4 (rank 181 / 416, climbed 107 positions) |
| Deck concept clarity | 20% | Sample-deck control for v1-v5; Darkness archetype for v6 (§5) |
| Key cards utilized to support strategy | 20% | Yveltal ex + Mega Absol ex matched against priority schema's KO-tuned attack selector |
| Report logical structure | 10% | This document |
| Visual elements used effectively | 10% | Score-progression table (§2), engine-search pseudo-code (§3), rubric-alignment table (§9) |

## 10. Citation

Atom McCree (2026). *AtomEons Misfit-TCG: Property-Bound Decision Schemas with Engine-Search Lookahead.* Pokémon TCG AI Battle Challenge Strategy Category Writeup, AtomEons Research Laboratory.

---

*Disclosure ID: ATOM-PTCG-STRATEGY-2026-0617*
*License: CC-BY-4.0 (per Strategy Category terms)*
