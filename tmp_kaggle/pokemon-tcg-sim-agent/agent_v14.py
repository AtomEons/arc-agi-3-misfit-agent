"""AtomEons Misfit-TCG agent — v14 ENSEMBLE VOTING.

v14 is a variance-reduction layer over v7/v8/v9. It calls each constituent
agent with a hard time cap, then majority-votes the chosen option index.

Architecture (per V14_ENSEMBLE_DESIGN.md):

  - Per-decision wall budget: 600 ms total, 200 ms per agent.
  - Vote tally:
      * Unanimous (3-0): ship the choice.
      * Majority (2-1): ship the majority choice.
      * All distinct (1-1-1): tiebreak — prefer v8's choice.
      * All crashed / no votes: route to _v8(obs) directly.
  - Any single agent crash → route directly to _v8(obs) for the current
    decision (per design §7 fallback discipline; a crash is signal that
    the state is anomalous and the strongest single agent is safest).
  - Sequential execution. The Kaggle gateway is single-threaded; we cannot
    parallelize across agents reliably across the KAGGLE_IS_COMPETITION_RERUN
    boundary.
  - Time cap enforced by monotonic deadline polled before each agent call;
    we cannot interrupt an agent mid-run, so the cap is best-effort
    (skip-remaining-agents if budget is exhausted before invocation).

Tier-1 strict: deterministic vote tally, no learned weights, no LLM.
"""

import os
import time

from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
)

# Import the three constituent agents. We use their public `agent` entry
# points so that any internal evolution stays encapsulated. Each agent
# already handles obs.select is None (deck-select setup turn) on its own.
import agent_v7 as _v7_mod
import agent_v8 as _v8_mod
import agent_v9 as _v9_mod


# ─── Deck IO ────────────────────────────────────────────────────────────


def read_deck_csv() -> list[int]:
    """Same deck source as v7/v8/v9 — keeps arena comparisons honest."""
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        candidates = [
            "/kaggle_simulations/agent/deck.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break
    with open(file_path, "r") as fh:
        rows = [r.strip() for r in fh.read().split("\n") if r.strip()]
    return [int(r) for r in rows[:60]]


_DECK_CACHE: list[int] | None = None


def _own_deck() -> list[int]:
    global _DECK_CACHE
    if _DECK_CACHE is None:
        _DECK_CACHE = read_deck_csv()
    return _DECK_CACHE


# ─── Vote primitives ────────────────────────────────────────────────────


def _selection_key(selection):
    """Convert an agent's returned selection to a hashable vote key.

    Agents return list[int]; we use a tuple so it's hashable and
    structural equality semantics match (order matters — [1,2] is a
    different selection than [2,1] to the engine).
    """
    if isinstance(selection, list):
        return tuple(int(x) for x in selection)
    if isinstance(selection, tuple):
        return tuple(int(x) for x in selection)
    # Defensive: unexpected return type — treat as crash signal upstream
    return None


def _key_to_selection(key) -> list[int]:
    """Inverse of _selection_key for emitting the final choice."""
    return list(key)


# ─── Public agent entry point ───────────────────────────────────────────


_PER_AGENT_BUDGET_S = 0.200
_TOTAL_BUDGET_S = 0.600


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point. Receives obs_dict, returns list[int] selection.

    Strategy:
      1. Setup turn → return our deck (matches v7/v8/v9).
      2. For each of v7, v8, v9:
           - Check we still have wall budget; if not, skip (count as crash).
           - Invoke; capture choice.
           - On any Exception, count as crash and drop from vote.
      3. Per design §7: if ANY single agent crashed, fall back to v8 alone
         (a crash is signal the state is anomalous; safest commit is the
         strongest single agent). If v8 itself crashes during fallback,
         degrade gracefully to v7 then END.
      4. Otherwise tally: majority wins, all-distinct tiebreaks to v8.
    """
    # Setup turn — return deck (consistent with v7/v8/v9 contract)
    if obs_dict.get("select") is None:
        return _own_deck()

    overall_deadline = time.monotonic() + _TOTAL_BUDGET_S

    votes: dict = {}            # selection_key -> vote count
    selections_by_agent: dict = {}  # name -> selection_key
    crashes = 0

    agents = [
        ("v7", _v7_mod.agent),
        ("v8", _v8_mod.agent),
        ("v9", _v9_mod.agent),
    ]

    for name, fn in agents:
        # Best-effort wall cap: if we've already burned the global budget,
        # skip remaining agents. They count as crashes (per design §6:
        # "hitting the cap counts as a crash").
        if time.monotonic() > overall_deadline:
            crashes += 1
            continue

        per_agent_start = time.monotonic()
        try:
            sel = fn(obs_dict)
            key = _selection_key(sel)
            if key is None:
                # Bad return shape — treat as crash
                crashes += 1
                continue
            # Per-agent soft cap: if this agent blew its 200ms budget,
            # we still accept its vote (we cannot un-spend time), but we
            # log it as a budget overrun by counting toward the global
            # deadline. We do NOT count it as a crash if it returned a
            # valid selection — that would be wasteful.
            _ = time.monotonic() - per_agent_start  # measured, not enforced post-hoc
            votes[key] = votes.get(key, 0) + 1
            selections_by_agent[name] = key
        except Exception:
            crashes += 1
            continue

    # Per design §7 fallback discipline: ANY crash → route to v8 directly.
    # This is conservative — a crash signals an anomalous state where
    # the strongest single agent is the safest commit. We do NOT salvage
    # a 2-vote tally because the crash itself is information.
    if crashes >= 1 or not votes:
        return _safe_v8_fallback(obs_dict)

    # Tally. Find max vote count.
    max_count = max(votes.values())

    if max_count >= 2:
        # Majority or unanimous — find the winning key. If there's a tie
        # among 2-vote keys (impossible with 3 agents, but defensive),
        # prefer v8's choice.
        winners = [k for k, c in votes.items() if c == max_count]
        if len(winners) == 1:
            return _key_to_selection(winners[0])
        # Tie among multi-vote keys → prefer v8's choice if it's among
        # the winners; otherwise the first winner (deterministic order).
        v8_key = selections_by_agent.get("v8")
        if v8_key is not None and v8_key in winners:
            return _key_to_selection(v8_key)
        return _key_to_selection(winners[0])

    # All distinct (1-1-1). Tiebreak — prefer v8's choice (highest
    # individual win rate per design §5).
    v8_key = selections_by_agent.get("v8")
    if v8_key is not None:
        return _key_to_selection(v8_key)
    # v8 missing from votes (would only happen if our crash-fallback
    # logic above missed it, which it doesn't — defensive):
    # prefer v9, then v7.
    for name in ("v9", "v7"):
        k = selections_by_agent.get(name)
        if k is not None:
            return _key_to_selection(k)
    # No votes at all (handled above by `not votes` check, but defensive):
    return _safe_v8_fallback(obs_dict)


def _safe_v8_fallback(obs_dict: dict) -> list[int]:
    """v8-first fallback chain. If v8 also crashes, degrade to v7, then
    to a hardcoded END selection. Never raises."""
    try:
        return _v8_mod.agent(obs_dict)
    except Exception:
        pass
    try:
        return _v7_mod.agent(obs_dict)
    except Exception:
        pass
    # Final degraded path: find END or return [0]
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return _own_deck()
        for i, opt in enumerate(obs.select.option):
            if int(opt.type) == int(OptionType.END):
                return [i]
        return [0]
    except Exception:
        return [0]
