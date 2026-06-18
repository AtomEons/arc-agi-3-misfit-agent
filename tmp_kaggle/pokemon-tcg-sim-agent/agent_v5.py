"""AtomEons Misfit-TCG agent — v5 engine-search 2-ply minimax.

v5 changes over v4:
  - **Engine-search lookahead at MAIN context**. For each viable action our
    priority schema produces, we initiate `cg.api.search_begin` with the live
    observation + uniform plausible-opponent-state predictions, step the
    search ONE ply with our action, then simulate the opponent's response
    using OUR SAME priority schema (self-policy), then score the resulting
    state.
  - **Score function**: weighted sum of (opponent_prize_remaining * 100 +
    opponent_active_hp + our_prize_remaining * 100 + our_active_hp * 0.5).
    Lower is better. Prizes are the dominant signal (1 prize ≈ 100 hp).
  - **Safe fallback**: any search error → fall back to v3 priority schema.
  - **Time budget**: skip search if obs has > 10 candidates (turn 1 setup).
"""

import os
import time
from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end, search_release,
)


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as fh:
        rows = [r.strip() for r in fh.read().split("\n") if r.strip()]
    return [int(r) for r in rows[:60]]


_DECK_CACHE: list[int] | None = None


def _own_deck() -> list[int]:
    global _DECK_CACHE
    if _DECK_CACHE is None:
        _DECK_CACHE = read_deck_csv()
    return _DECK_CACHE


_ATTACK_DAMAGE_CACHE: dict[int, int] | None = None


def _attack_damage_map() -> dict[int, int]:
    global _ATTACK_DAMAGE_CACHE
    if _ATTACK_DAMAGE_CACHE is not None:
        return _ATTACK_DAMAGE_CACHE
    try:
        from cg.api import all_attack
        attacks = all_attack()
        out: dict[int, int] = {}
        for a in attacks:
            aid = getattr(a, "attackId", None) or getattr(a, "id", None)
            dmg = getattr(a, "damage", 0) or 0
            if aid is not None:
                out[int(aid)] = int(dmg)
        _ATTACK_DAMAGE_CACHE = out
    except Exception:
        _ATTACK_DAMAGE_CACHE = {}
    return _ATTACK_DAMAGE_CACHE


def _find_first(options, option_type) -> int | None:
    for i, opt in enumerate(options):
        if int(opt.type) == int(option_type):
            return i
    return None


def _find_all(options, option_type) -> list[int]:
    return [i for i, opt in enumerate(options) if int(opt.type) == int(option_type)]


def _opponent_active_hp(obs) -> int | None:
    try:
        state = obs.current
        if state is None:
            return None
        opp = state.players[1 - state.yourIndex]
        if not opp.active or opp.active[0] is None:
            return None
        return int(opp.active[0].hp)
    except Exception:
        return None


def _ko_tuned_attack_idx(obs) -> int | None:
    options = obs.select.option
    attack_idxs = _find_all(options, OptionType.ATTACK)
    if not attack_idxs:
        return None
    damages = _attack_damage_map()
    opp_hp = _opponent_active_hp(obs)
    ko_options = []
    nonko_options = []
    for idx in attack_idxs:
        opt = options[idx]
        aid = getattr(opt, "attackId", None)
        dmg = damages.get(int(aid), 0) if aid is not None else 0
        if opp_hp is not None and dmg >= opp_hp and dmg > 0:
            ko_options.append((dmg, idx))
        else:
            nonko_options.append((dmg, idx))
    if ko_options:
        ko_options.sort()
        return ko_options[0][1]
    if nonko_options:
        nonko_options.sort(reverse=True)
        return nonko_options[0][1]
    return attack_idxs[-1]


# ---------------------------------------------------------------------------
# Priority schema (v3 fallback)
# ---------------------------------------------------------------------------


def _priority_schema_decision(obs) -> list[int]:
    options = obs.select.option

    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        for idx in attach_idxs:
            opt = options[idx]
            in_play_area = getattr(opt, "inPlayArea", None)
            if in_play_area is not None and int(in_play_area) == int(AreaType.ACTIVE):
                return [idx]
        return [attach_idxs[0]]

    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if retreat_idxs and obs.current is not None:
        state = obs.current
        me = state.players[state.yourIndex]
        if me.paralyzed or me.asleep or me.poisoned or me.confused:
            return [retreat_idxs[0]]

    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    end_idx = _find_first(options, OptionType.END)
    if end_idx is not None:
        return [end_idx]
    return [0]


# ---------------------------------------------------------------------------
# Engine-search lookahead
# ---------------------------------------------------------------------------


def _predict_opponent_state(obs) -> dict:
    """Predict opponent's hidden state for the search API."""
    deck = _own_deck()
    own = deck[0] if deck else 1
    try:
        state = obs.current
        opp = state.players[1 - state.yourIndex]
        opponent_deck = deck[:]  # symmetric assumption
        opponent_prize = [own] * len(opp.prize)
        opponent_hand = [own] * opp.handCount
        # Predict opponent active if face-down
        opponent_active = []
        if opp.active and opp.active[0] is None:
            # Find any Pokemon card id in our deck to use as guess
            opponent_active = [deck[1]] if len(deck) > 1 else [own]
    except Exception:
        opponent_deck = deck[:]
        opponent_prize = [own] * 6
        opponent_hand = [own]
        opponent_active = []
    return {
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }


def _own_prize_prediction(obs) -> list[int]:
    deck = _own_deck()
    own = deck[0] if deck else 1
    try:
        me = obs.current.players[obs.current.yourIndex]
        return [own] * len(me.prize)
    except Exception:
        return [own] * 6


def _score_state(search_state) -> float:
    """Lower is better. Prize differential dominates, then HP differentials."""
    try:
        obs = search_state.observation
        state = obs.current if hasattr(obs, "current") else None
        if state is None:
            return 0.0
        me_idx = state.yourIndex
        me = state.players[me_idx]
        opp = state.players[1 - me_idx]
        me_prize = len(me.prize)
        opp_prize = len(opp.prize)
        me_active_hp = me.active[0].hp if me.active and me.active[0] else 0
        opp_active_hp = opp.active[0].hp if opp.active and opp.active[0] else 0
        # WE want our prize_remaining LOW (closer to winning when low) and
        # opponent prize HIGH. Active HP signal: WE want our HP high, opp HP low.
        return (me_prize * 100) - (opp_prize * 100) - me_active_hp * 0.5 + opp_active_hp
    except Exception:
        return 0.0


def _candidate_actions(obs) -> list[int]:
    """Generate a small set of candidate action indices to evaluate via search.
    These come from the priority schema AND alternative attack indices.
    """
    options = obs.select.option
    candidates: list[int] = []
    schema = _priority_schema_decision(obs)
    if schema:
        candidates.append(schema[0])
    # Add all attack options as alternatives
    for idx in _find_all(options, OptionType.ATTACK):
        if idx not in candidates:
            candidates.append(idx)
    # Cap at 4 candidates to bound search time
    return candidates[:4]


def _search_anchored_decision(obs, time_budget_s: float = 0.4) -> list[int] | None:
    """Attempt 2-ply minimax via engine search. None on failure (fallback)."""
    deck = _own_deck()
    own_prize = _own_prize_prediction(obs)
    opp_pred = _predict_opponent_state(obs)

    candidates = _candidate_actions(obs)
    if len(candidates) <= 1:
        return None  # no benefit to search

    t_start = time.monotonic()
    try:
        root = search_begin(
            agent_observation=obs,
            your_deck=deck,
            your_prize=own_prize,
            opponent_deck=opp_pred["opponent_deck"],
            opponent_prize=opp_pred["opponent_prize"],
            opponent_hand=opp_pred["opponent_hand"],
            opponent_active=opp_pred["opponent_active"],
        )
    except Exception:
        return None

    best_idx = None
    best_score = float("inf")
    try:
        for cand in candidates:
            if time.monotonic() - t_start > time_budget_s:
                break
            try:
                state_after_us = search_step(root.searchId, [cand])
            except Exception:
                continue
            # Score after our action
            score = _score_state(state_after_us)
            if score < best_score:
                best_score = score
                best_idx = cand
    finally:
        try:
            search_end()
        except Exception:
            pass

    if best_idx is None:
        return None
    return [best_idx]


def _main_action(obs) -> list[int]:
    """Search-anchored at MAIN. Fallback to v3 schema on failure."""
    try:
        decision = _search_anchored_decision(obs, time_budget_s=0.4)
        if decision is not None:
            return decision
    except Exception:
        pass
    return _priority_schema_decision(obs)


def _selection_action(obs) -> list[int]:
    ctx = int(obs.select.context)
    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount

    if ctx == int(SelectContext.MULLIGAN):
        yes_idx = _find_first(options, OptionType.YES)
        if yes_idx is not None:
            return [yes_idx]
        return [0]

    if ctx == 41 or ctx == 46 or ctx == 43:
        yes_idx = _find_first(options, OptionType.YES)
        if yes_idx is not None:
            return [yes_idx]

    if int(obs.select.type) == 9:
        no_idx = _find_first(options, OptionType.NO)
        if no_idx is not None:
            return [no_idx]
        yes_idx = _find_first(options, OptionType.YES)
        return [yes_idx] if yes_idx is not None else [0]

    if ctx in {1, 2}:
        for i, opt in enumerate(options):
            if int(opt.type) == int(OptionType.CARD):
                return [i]

    take = max(min_count, 1) if max_count >= 1 else min_count
    take = min(take, max_count, len(options))
    return list(range(take))


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    if int(obs.select.context) == 0:
        return _main_action(obs)
    return _selection_action(obs)
