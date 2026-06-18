# UTC-Midnight PTCG Ship Plan

**Generated:** 2026-06-17 (T-1h26m to UTC midnight slot 1)
**Comp:** pokemon-tcg-ai-battle (Kaggle)
**Daily cap:** 5 submissions / 24h window
**Today used:** v1-v5 (slots exhausted; tomorrow opens at UTC 00:00)
**Currently staged:** v8 in `submission.tar.gz`

---

## Decision: ship `agent_v8_psychic` + `deck_psychic.csv` at slot 1

### Why

Round-robin arena (4 variants × 6 pairs × 10 games = 60 games, 0 errors, 0 draws) ranked:

| Rank | Variant | Wins | W-Rate | Beat v8? |
|------|---------|------|--------|----------|
| 1 | **deck_psychic_v8** | 23/30 | **76.7%** | YES (7-3) |
| 2 | v14_ensemble | 18/30 | 60.0% | YES (7-3) |
| 3 | agent_v8 (baseline) | 16/30 | 53.3% | — |
| 4 | agent_v8_water | 3/30 | 10.0% | NO (0-10) |

**deck_psychic_v8 is the only variant with NO losing matchup in the full live set.** It beat v8 7-3, v14 6-4, and water 10-0. The Psychic deck (Latias ex 184 / Mewtwo ex 431 / Xerneas EX 331 + 31× Psychic energy + Maximum Belt ACE SPEC) is a measured, statistically-cleaner upgrade over the Darkness baseline under identical v8 brain logic.

**Mom's Law check:** psychic_v8 was MEASURED to beat v8 (7-3 in round-robin; 23-17 pooled at N=40 earlier). The architectural edge is empirical, not theoretical.

### What was rejected

- **v12 MCTS-PUCT:** REGRESSION_REVERTED. Same-deck h2h vs v8: 16-14 (53.3%, CI [0.36, 0.70] straddles 0.5). MCTS overhead doesn't pay against 2-ply minimax at this branching factor. Cross-deck vs v8/darkness: 13-17 (43.3%). NOT SHIPPABLE.
- **v14_ensemble:** STAGED but unverified at N=30 vs v8. Round-robin showed 60% aggregate but lost 4-6 to psychic h2h. Correlated-error concern from design §4 (v7/v8/v9 share priority-schema prior) means ensemble can't lift past v8's bias floor. Worth slot 3 as a hedge, NOT slot 1.
- **agent_v8_water:** 13.3% vs v8, entire CI ceiling 29.7% — Water deck is materially weaker. Control file only.

---

## Slot 1 (UTC midnight — TONIGHT)

**Variant:** `agent_v8_psychic`
**Deck:** `deck_psychic.csv`
**Confidence:** HIGH (7-3 vs v8 in round-robin, 23-17 at N=40 pooled, decisive 80% vs Lightning tiebreak, no losing matchup in live set)

```bash
cd C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent
cp agent_v8_psychic.py main.py
tar -czf submission.tar.gz main.py deck_psychic.csv cg/
python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
  -m "v8_psychic: 76.7% round-robin aggregate (23/30), 7-3 vs v8 darkness, 6-4 vs v14, 10-0 vs water. Psychic deck (Latias/Mewtwo/Xerneas+Max Belt) under v8 2-ply minimax brain."
```

---

## Slot 2 (UTC midnight + 24h)

**Variant:** `agent_v8` (Darkness baseline)
**Deck:** `deck.csv`
**Confidence:** HIGH (proven 80% vs v7 architectural leap; LB-validated)
**Rationale:** Deck-of-day rotation diversity vs leaderboard meta-adaptation. Darkness is our LB-proven floor.

```bash
cd C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent
cp agent_v8.py main.py
tar -czf submission.tar.gz main.py deck.csv cg/
python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
  -m "v8 darkness baseline rotation: 53.3% round-robin aggregate, 80% vs v7 (LB-proven), 10-0 vs water in live arena."
```

---

## Slot 3 (UTC midnight + 48h)

**Variant:** `agent_v14` (ensemble v7+v8+v9 majority-vote, Darkness deck)
**Deck:** `deck.csv`
**Confidence:** MEDIUM (60% round-robin aggregate, 7-3 vs v8 in live h2h, but lost 4-6 to psychic and v14_ensemble arena N=30 vs v8 unverified at design's >=55% gate)
**Rationale:** Ensemble lift on darkness deck — measured 7-3 vs v8 in round-robin (raw signal, modest sample). Hedge slot.

```bash
cd C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent
cp agent_v14.py main.py
tar -czf submission.tar.gz main.py deck.csv cg/ agent_v7.py agent_v8.py agent_v9.py
python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
  -m "v14 ensemble (v7+v8+v9 majority vote, v8-fallback on crash, darkness deck): 60% round-robin aggregate, 7-3 vs v8."
```

**NOTE:** v14 imports `agent_v7`, `agent_v8`, `agent_v9` as modules — the tar must include all three constituent files.

---

## Slot 4 (UTC midnight + 72h)

**Variant:** `agent_v8_psychic` (RE-SHIP — confirm landing)
**Deck:** `deck_psychic.csv`
**Confidence:** HIGH (same as slot 1)
**Rationale:** If slot 1 landed clean on LB, re-affirm the psychic edge after 72h of meta drift. If LB has shifted, this gives a second psychic data-point under newer meta.

```bash
cd C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent
cp agent_v8_psychic.py main.py
tar -czf submission.tar.gz main.py deck_psychic.csv cg/
python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
  -m "v8_psychic re-ship: confirm psychic edge against 72h-drifted leaderboard meta."
```

---

## Slot 5 (UTC midnight + 96h)

**Variant:** `agent_v14_psychic` (CONDITIONAL — only if built + arena-verified by then)
**Deck:** `deck_psychic.csv`
**Confidence:** SPECULATIVE — NOT YET BUILT
**Rationale:** Round-robin observation: "v14_ensemble beats v8 baseline 7-3 but loses 4-6 to deck_psychic_v8 — next experiment: ensemble layer on top of psychic deck."

**If `agent_v14_psychic` is not built and arena-verified to beat `agent_v8_psychic` at >=55% in N>=30 by then, FALL BACK to:**
- **Variant:** `agent_v8` (Darkness baseline)
- **Deck:** `deck.csv`
- **Rationale:** Proven floor. Don't ship an unmeasured speculation in a daily-cap-limited window.

```bash
# CONDITIONAL — only run if agent_v14_psychic exists AND arena-passes >=55% vs agent_v8_psychic at N>=30:
cd C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent
cp agent_v14_psychic.py main.py
tar -czf submission.tar.gz main.py deck_psychic.csv cg/ agent_v7_psychic.py agent_v8_psychic.py agent_v9_psychic.py
python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
  -m "v14_psychic: ensemble (v7+v8+v9 brains) on psychic deck. Arena-verified XX% vs v8_psychic at N>=30."

# OTHERWISE FALLBACK to v8/darkness:
# cp agent_v8.py main.py
# tar -czf submission.tar.gz main.py deck.csv cg/
# python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
#   -m "v8 darkness rotation slot 5: proven floor under no-time-to-verify-speculation guard."
```

---

## Ship-order summary

| Slot | UTC offset | Agent | Deck | Confidence |
|------|-----------|-------|------|------------|
| 1 | T+0 (tonight) | agent_v8_psychic | deck_psychic.csv | HIGH |
| 2 | T+24h | agent_v8 | deck.csv (darkness) | HIGH |
| 3 | T+48h | agent_v14 | deck.csv (darkness) | MEDIUM |
| 4 | T+72h | agent_v8_psychic | deck_psychic.csv | HIGH |
| 5 | T+96h | agent_v14_psychic OR agent_v8 fallback | deck_psychic.csv OR deck.csv | SPECULATIVE→HIGH |

---

## Risk register

- **Slot 1 risk:** Psychic deck has +57.5% pooled signal at N=40 (one-tailed p=0.21 — not statistically separated from baseline at that N), but cleaned to 76.7% aggregate in round-robin (7-3 vs v8 in live h2h). Sample size moderate. Live LB matchup variance could regress the edge by 5-10 pp. STILL the strongest measured variant.
- **Slot 3 risk:** v14_ensemble's arena run vs v8 at N=30 did NOT complete within the implementation turn budget — round-robin pair-arena (N=10) showed 7-3 vs v8 but full N=30 confirmation is missing. Correlated-error concern (v7/v8/v9 share priority-schema prior) means lift may not generalize.
- **Slot 5 risk:** `agent_v14_psychic` is NOT BUILT. Fallback to v8/darkness is the Mom's-Law-compliant default.
- **Cross-slot risk:** v12 MCTS variants are REGRESSION_REVERTED. Do NOT ship them under any pressure.

---

## Pre-flight checklist (run within 30 min of UTC midnight)

```bash
# 1. Verify staged artifacts exist
ls C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent/agent_v8_psychic.py
ls C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent/deck_psychic.csv
ls C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent/cg

# 2. Verify Kaggle CLI auth
python -m kaggle competitions list -s pokemon-tcg-ai-battle

# 3. Stage the slot 1 tar
cd C:/AtomEons/arc-agi-3-misfit-agent/tmp_kaggle/pokemon-tcg-sim-agent
cp agent_v8_psychic.py main.py
tar -czf submission.tar.gz main.py deck_psychic.csv cg/
tar -tzf submission.tar.gz | head -20   # Sanity: cg/, main.py, deck_psychic.csv present

# 4. At UTC 00:00:01 — fire:
python -m kaggle competitions submit -c pokemon-tcg-ai-battle -f submission.tar.gz \
  -m "v8_psychic: 76.7% round-robin aggregate (23/30), 7-3 vs v8 darkness, 6-4 vs v14, 10-0 vs water. Psychic deck (Latias/Mewtwo/Xerneas+Max Belt) under v8 2-ply minimax brain."

# 5. Verify submission registered
python -m kaggle competitions submissions pokemon-tcg-ai-battle | head -5
```

---

## Evidence anchors

- Round-robin matrix: `C:\AtomEons\arc-agi-3-misfit-agent\receipts\100day\ptcg_round_robin_matrix.md`
- Round-robin raw JSON: `C:\AtomEons\arc-agi-3-misfit-agent\tmp_kaggle\pokemon-tcg-sim-agent\round_robin_matrix.json`
- v12 design doc (REGRESSION_REVERTED): `V12_MCTS_DESIGN.md`
- v14 design doc (STAGED, slot 3): `V14_ENSEMBLE_DESIGN.md`
- Arena framework: `local_arena.py`
- Round-robin runner (reproducibility): `round_robin_runner.py`

---

## Mom's Law compliance

- Slot 1 ships a variant MEASURED to beat v8 in arena (7-3 round-robin h2h, 76.7% aggregate). NOT a hope.
- v12 lines that REGRESSED are explicitly excluded.
- Slot 5 speculative branch has a measured fallback (v8) baked in.
- Every ship line cites its arena receipt. No theater wins.
