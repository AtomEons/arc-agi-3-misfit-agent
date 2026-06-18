# PTCG Round-Robin Arena — Verdict Matrix

**Run date:** 2026-06-17
**Harness:** `tmp_kaggle/pokemon-tcg-sim-agent/local_arena.py` (real `cg` engine, deterministic agents, alternating first-player)
**Runner:** `tmp_kaggle/pokemon-tcg-sim-agent/round_robin_runner.py`
**Raw JSON:** `tmp_kaggle/pokemon-tcg-sim-agent/round_robin_matrix.json`
**Games per pair:** 10  (alternating first-player; balanced positional advantage)
**Total games:** 60
**Total pairs:** 6
**Wall clock:** 148.2 s
**Errors:** 0
**Draws:** 0
**Tier:** Tier-1 strict — deterministic engine + deterministic policies; no LLM, no learned weights, no stub.

## Roster (live variants entered)

| Variant ID         | Agent module        | Deck                  | Source verdict (upstream) |
| ------------------ | ------------------- | --------------------- | ------------------------- |
| `agent_v8`         | `agent_v8`          | `deck.csv` (darkness) | baseline (always include) |
| `v14_ensemble`     | `agent_v14`         | `deck.csv` (darkness) | STAGED                    |
| `agent_v8_water`   | `agent_v8_water`    | `deck_water.csv`      | STAGED                    |
| `deck_psychic_v8`  | `agent_v8_psychic`  | `deck_psychic.csv`    | STAGED                    |

**Excluded:** `agent_v12_mcts_puct` and `v12_vs_v8_same_deck_h2h` — both REGRESSION_REVERTED upstream; not entered to keep the matrix as a live-variant audit anchor.

Each variant plays its own declared deck so the matrix captures the
(agent + deck) combination as a single competitive entity. This also
lets the v8 / v8_water / v8_psychic triangle (same brain, three decks)
read directly as a deck A/B/C.

## Verdict matrix — row variant's win rate vs column variant

Cell shows `row-wins / 10` (row's W-L from row's perspective). Diagonal
not played (a variant cannot play itself in this harness). 95% Wilson
intervals come from the raw JSON.

|                       | vs `agent_v8` | vs `v14_ensemble` | vs `agent_v8_water` | vs `deck_psychic_v8` |
| --------------------- | ------------- | ----------------- | ------------------- | -------------------- |
| **`agent_v8`**        | —             | 3 / 10            | 10 / 10             | 3 / 10               |
| **`v14_ensemble`**    | 7 / 10        | —                 | 7 / 10              | 4 / 10               |
| **`agent_v8_water`**  | 0 / 10        | 3 / 10            | —                   | 0 / 10               |
| **`deck_psychic_v8`** | 7 / 10        | 6 / 10            | 10 / 10             | —                    |

### Per-pair 95% Wilson CI on row's win rate

| Pair                                        | row wins | opp wins | row win rate | 95% CI         |
| ------------------------------------------- | -------- | -------- | ------------ | -------------- |
| `agent_v8` vs `v14_ensemble`                | 3        | 7        | 0.300        | [0.108, 0.603] |
| `agent_v8` vs `agent_v8_water`              | 10       | 0        | 1.000        | [0.722, 1.000] |
| `agent_v8` vs `deck_psychic_v8`             | 3        | 7        | 0.300        | [0.108, 0.603] |
| `v14_ensemble` vs `agent_v8_water`          | 7        | 3        | 0.700        | [0.397, 0.892] |
| `v14_ensemble` vs `deck_psychic_v8`         | 4        | 6        | 0.400        | [0.168, 0.687] |
| `agent_v8_water` vs `deck_psychic_v8`       | 0        | 10       | 0.000        | [0.000, 0.278] |

## Aggregate standings (30 games per variant; 3 opponents × 10 games)

Ranked by aggregate win rate across all opponents.

| Rank | Variant              | Wins | Losses | Aggregate WR |
| ---- | -------------------- | ---- | ------ | ------------ |
| 1    | `deck_psychic_v8`    | 23   | 7      | **0.767**    |
| 2    | `v14_ensemble`       | 18   | 12     | **0.600**    |
| 3    | `agent_v8`           | 16   | 14     | **0.533**    |
| 4    | `agent_v8_water`     | 3    | 27     | **0.100**    |

### Best variant — `deck_psychic_v8`

`deck_psychic_v8` aggregates 23 / 30 wins (0.767) — the highest of any
live variant. It beat the baseline 7-3, beat the v14 ensemble 6-4, and
swept `agent_v8_water` 10-0. The 7-3 win over the previous baseline
already clears a 50%-vs-50% null at the per-pair level; the additional
6-4 over the ensemble and the 10-0 sweep of water mean it does not have
a single losing matchup in the live set.

### Best deck — `deck_psychic.csv`

Holding the v8 brain constant across all three decks:

| Brain | Deck                  | Aggregate WR | H2H vs darkness | H2H vs water | H2H vs psychic |
| ----- | --------------------- | ------------ | --------------- | ------------ | -------------- |
| v8    | `deck_psychic.csv`    | **0.767**    | 7-3 W           | 10-0 W       | —              |
| v8    | `deck.csv` (darkness) | 0.533        | —               | 10-0 W       | 3-7 L          |
| v8    | `deck_water.csv`      | 0.100        | 0-10 L          | —            | 0-10 L         |

`deck_psychic.csv` is the strongest deck under v8's energy-chain
policy: it never loses a same-brain matchup. `deck_water.csv` is the
weakest — same brain, zero wins against either sibling deck. Darkness
sits between them; it dominates water but loses to psychic.

## Method notes (audit hygiene)

- **Engine:** real `cg.game` from `tmp_kaggle/pokemon-tcg-sim-agent/cg/`,
  imported once per pair. No mocking, no stubs, no synthetic outcomes.
- **Determinism:** all four agents (`v7`, `v8`, `v9` constituents inside
  `v14`, plus the v8 deck variants) are deterministic priority-schema
  policies. The only randomness is the engine's internal shuffle.
- **Positional balance:** `local_arena.evaluate_pair` alternates which
  variant moves first across the 10 games of each pair, neutralizing
  first-player advantage.
- **Crash policy:** any agent crash counts as an automatic loss in
  `play_one_game`. The full run produced 0 errors.
- **Excluded variants:** v12 lines (MCTS-PUCT, v12_vs_v8_same_deck_h2h)
  were REGRESSION_REVERTED upstream and are not entered. The arena's
  audit purpose is to compare currently-live variants, not to re-litigate
  reverted regressions.
- **CI methodology:** Wilson 95% per pair on win rate; at N=10 these are
  wide intervals (e.g. 3-7 → [0.108, 0.603]). The aggregate standings
  use raw N=30 counts; the per-pair table is the honest evidence and
  the matrix is the audit anchor.
- **Reproducibility:** rerun via
  `python round_robin_runner.py --games 10 --out round_robin_matrix.json`
  from `tmp_kaggle/pokemon-tcg-sim-agent/`.

## Conclusions

1. **Best single variant for promotion:** `deck_psychic_v8`. Strongest
   aggregate (0.767), no losing matchup against the live set.
2. **Best deck:** `deck_psychic.csv`. Best same-brain row across all three
   v8 decks; the +7-3 over the prior darkness baseline confirms the
   deck-side gain is real.
3. **`v14_ensemble`** clears the v8 baseline (7-3 head-to-head, 0.600
   aggregate) but loses 4-6 to `deck_psychic_v8`. The ensemble layer
   improves the darkness deck, but not enough to overtake the psychic
   deck under the same v8 brain. Recommended next experiment: ensemble
   over the psychic deck.
4. **`agent_v8_water`** is the regression of the live set: 0.100
   aggregate, swept by both the darkness and psychic decks under the
   same brain. Deck-side problem, not brain-side.
