"""AtomEons Misfit-TCG agent — v7 federation voting (CHSG policy distillation).

v7 changes over v5:
  - **Three priority schemas vote** on each MAIN decision:
      a) ABILITY-first (v3 / v5 order)
      b) ATTACK-first (v1 / v2 order)
      c) PLAY-first (supporter-driven setup)
  - Each variant proposes an action independently.
  - Engine-search scores each variant's proposed action (same v5 lookahead).
  - The variant with the BEST search-evaluated state wins; ties broken by
    majority-vote then ABILITY-first preference.

This is "federation as policy distillation": three stable policies vote, the
engine search arbitrates. The aggregator is deterministic — no randomness.

Direct port of Black Mamba CHSG Trilogy mode (Article III §3.1):
  blind draft -> vote with score arbitration -> domain-weighted aggregation.

Tier-1 honest: deterministic priority enumeration × 3 + deterministic engine
search. No LLM, no learned parameters.
"""

import os
import time
from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
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


def _retreat_if_status(obs):
    options = obs.select.option
    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if retreat_idxs and obs.current is not None:
        state = obs.current
        me = state.players[state.yourIndex]
        if me.paralyzed or me.asleep or me.poisoned or me.confused:
            return [retreat_idxs[0]]
    return None


def _end_turn(obs):
    end_idx = _find_first(obs.select.option, OptionType.END)
    return [end_idx] if end_idx is not None else [0]


# ---------------------------------------------------------------------------
# Three priority schemas (federation members)
# ---------------------------------------------------------------------------


def _schema_ability_first(obs) -> list[int]:
    """v3 / v5 order: ABILITY > EVOLVE > PLAY > ATTACH > RETREAT > ATTACK > END"""
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
            in_play_area = getattr(options[idx], "inPlayArea", None)
            if in_play_area is not None and int(in_play_area) == int(AreaType.ACTIVE):
                return [idx]
        return [attach_idxs[0]]

    retreat = _retreat_if_status(obs)
    if retreat:
        return retreat

    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    return _end_turn(obs)


def _schema_attack_first(obs) -> list[int]:
    """v1 order: ATTACK > EVOLVE > ABILITY > PLAY > ATTACH > RETREAT > END"""
    options = obs.select.option

    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        for idx in attach_idxs:
            in_play_area = getattr(options[idx], "inPlayArea", None)
            if in_play_area is not None and int(in_play_area) == int(AreaType.ACTIVE):
                return [idx]
        return [attach_idxs[0]]

    retreat = _retreat_if_status(obs)
    if retreat:
        return retreat

    return _end_turn(obs)


def _schema_play_first(obs) -> list[int]:
    """PLAY > ATTACH > ABILITY > EVOLVE > ATTACK > END
    Supporter-driven setup: play cards (supporter, item, etc.) before anything
    else, then attach and use abilities, then attack."""
    options = obs.select.option

    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        for idx in attach_idxs:
            in_play_area = getattr(options[idx], "inPlayArea", None)
            if in_play_area is not None and int(in_play_area) == int(AreaType.ACTIVE):
                return [idx]
        return [attach_idxs[0]]

    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    retreat = _retreat_if_status(obs)
    if retreat:
        return retreat

    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    return _end_turn(obs)


# ---------------------------------------------------------------------------
# Engine-search scoring (same as v5)
# ---------------------------------------------------------------------------


def _predict_opponent_state(obs):
    deck = _own_deck()
    own = deck[0] if deck else 1
    try:
        state = obs.current
        opp = state.players[1 - state.yourIndex]
        opponent_deck = deck[:]
        opponent_prize = [own] * len(opp.prize)
        opponent_hand = [own] * opp.handCount
        opponent_active = []
        if opp.active and opp.active[0] is None:
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


def _own_prize_prediction(obs):
    deck = _own_deck()
    own = deck[0] if deck else 1
    try:
        me = obs.current.players[obs.current.yourIndex]
        return [own] * len(me.prize)
    except Exception:
        return [own] * 6


def _score_state(search_state):
    try:
        obs = search_state.observation
        state = obs.current if hasattr(obs, "current") else None
        if state is None:
            return 0.0
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        me_prize = len(me.prize)
        opp_prize = len(opp.prize)
        me_hp = me.active[0].hp if me.active and me.active[0] else 0
        opp_hp = opp.active[0].hp if opp.active and opp.active[0] else 0
        return (me_prize * 100) - (opp_prize * 100) - me_hp * 0.5 + opp_hp
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Federation aggregator
# ---------------------------------------------------------------------------


def _federation_decision(obs, time_budget_s: float = 0.6) -> list[int]:
    """Run three priority schemas, score each via engine search, pick best."""
    candidates = []
    for name, fn in [
        ("ability_first", _schema_ability_first),
        ("attack_first",  _schema_attack_first),
        ("play_first",    _schema_play_first),
    ]:
        try:
            decision = fn(obs)
            if decision and isinstance(decision, list):
                candidates.append((name, decision[0]))
        except Exception:
            continue
    if not candidates:
        return [0]

    # Dedup by chosen index
    seen = set()
    uniq = []
    for name, idx in candidates:
        if idx not in seen:
            seen.add(idx)
            uniq.append((name, idx))

    if len(uniq) == 1:
        return [uniq[0][1]]

    # Score each via engine search
    deck = _own_deck()
    own_prize = _own_prize_prediction(obs)
    opp_pred = _predict_opponent_state(obs)

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
        # Fallback: majority vote (only ability_first if no scoring)
        votes = {}
        for name, idx in candidates:
            votes[idx] = votes.get(idx, 0) + 1
        return [max(votes, key=votes.get)]

    best_idx = candidates[0][1]
    best_score = float("inf")
    try:
        for name, idx in uniq:
            if time.monotonic() - t_start > time_budget_s:
                break
            try:
                state_after = search_step(root.searchId, [idx])
            except Exception:
                continue
            score = _score_state(state_after)
            if score < best_score:
                best_score = score
                best_idx = idx
    finally:
        try:
            search_end()
        except Exception:
            pass

    return [best_idx]


def _main_action(obs):
    try:
        return _federation_decision(obs, time_budget_s=0.6)
    except Exception:
        return _schema_ability_first(obs)


def _selection_action(obs):
    ctx = int(obs.select.context)
    options = obs.select.option
    min_count = obs.select.minCount
    max_count = obs.select.maxCount

    if ctx == int(SelectContext.MULLIGAN):
        yes_idx = _find_first(options, OptionType.YES)
        return [yes_idx] if yes_idx is not None else [0]

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


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    if int(obs.select.context) == 0:
        return _main_action(obs)
    return _selection_action(obs)
