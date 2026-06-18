"""AtomEons Misfit-TCG agent — v3 abilities-first + KO-tuned attack.

v3 changes over v2:
  - PRIORITY REORDER: ABILITY > EVOLVE > PLAY > ATTACH > ATTACK > END
    Rationale: ATTACK ends the turn. v1/v2 ATTACKED before checking ABILITY,
    so free ability value (draw, search, board damage) was burned on every
    turn we had a viable attack. v3 uses all free-value actions first.
  - KO-TUNED ATTACK: among legal attacks, picks the LOWEST damage that
    still KOs the opponent's active Pokémon, falling back to highest-damage
    when no KO is available. Preserves energy on attackers for next turn.
  - Opponent HP read from obs.current.players[1 - yourIndex].active[0].hp.

Tier-1 honest: deterministic priority enumeration. No randomness, no LLM,
no learned parameters.
"""

import os
from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class,
)


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as fh:
        rows = [r.strip() for r in fh.read().split("\n") if r.strip()]
    return [int(r) for r in rows[:60]]


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


def _find_first(options: list, option_type: int) -> int | None:
    for i, opt in enumerate(options):
        if int(opt.type) == int(option_type):
            return i
    return None


def _find_all(options: list, option_type: int) -> list[int]:
    return [i for i, opt in enumerate(options) if int(opt.type) == int(option_type)]


def _opponent_active_hp(obs) -> int | None:
    """Read opponent's active Pokémon current HP. None if not visible."""
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
    """Pick the lowest-damage attack that still KOs the opponent's active
    Pokémon. Falls back to highest-damage attack when no KO available.
    """
    options = obs.select.option
    attack_idxs = _find_all(options, OptionType.ATTACK)
    if not attack_idxs:
        return None
    damages = _attack_damage_map()
    opp_hp = _opponent_active_hp(obs)

    # Score (damage, ko_quality) per attack option
    ko_options = []   # (damage, idx) — damage >= opp_hp
    nonko_options = []  # (damage, idx) — damage < opp_hp (or unknown)
    for idx in attack_idxs:
        opt = options[idx]
        aid = getattr(opt, "attackId", None)
        dmg = damages.get(int(aid), 0) if aid is not None else 0
        if opp_hp is not None and dmg >= opp_hp and dmg > 0:
            ko_options.append((dmg, idx))
        else:
            nonko_options.append((dmg, idx))

    if ko_options:
        # Smallest KO — preserves energy for the next attacker
        ko_options.sort()
        return ko_options[0][1]

    if nonko_options:
        # Highest damage available
        nonko_options.sort(reverse=True)
        return nonko_options[0][1]

    return attack_idxs[-1]


def _main_action_priority(obs) -> list[int]:
    options = obs.select.option

    # 1. ABILITY — free value, does not end turn
    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    # 2. EVOLVE — strict upgrade, does not end turn
    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    # 3. PLAY — supporter / item / stadium / tool / basic to bench
    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    # 4. ATTACH energy — prefer ACTIVE Pokemon slot
    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        for idx in attach_idxs:
            opt = options[idx]
            in_play_area = getattr(opt, "inPlayArea", None)
            if in_play_area is not None and int(in_play_area) == int(AreaType.ACTIVE):
                return [idx]
        return [attach_idxs[0]]

    # 5. RETREAT — only under status condition
    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if retreat_idxs and obs.current is not None:
        state = obs.current
        me = state.players[state.yourIndex]
        if me.paralyzed or me.asleep or me.poisoned or me.confused:
            return [retreat_idxs[0]]

    # 6. ATTACK — KO-tuned (smallest KO, else highest damage)
    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    # 7. END turn
    end_idx = _find_first(options, OptionType.END)
    if end_idx is not None:
        return [end_idx]

    return [0]


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

    # IS_FIRST → YES (go first)
    if ctx == 41:
        yes_idx = _find_first(options, OptionType.YES)
        if yes_idx is not None:
            return [yes_idx]

    # COIN_HEAD → YES (consistency)
    if ctx == 46:
        yes_idx = _find_first(options, OptionType.YES)
        if yes_idx is not None:
            return [yes_idx]

    # ACTIVATE → YES (activated effects are usually beneficial to us)
    if ctx == 43:
        yes_idx = _find_first(options, OptionType.YES)
        if yes_idx is not None:
            return [yes_idx]

    # Other YES_NO → NO
    if int(obs.select.type) == 9:
        no_idx = _find_first(options, OptionType.NO)
        if no_idx is not None:
            return [no_idx]
        yes_idx = _find_first(options, OptionType.YES)
        return [yes_idx] if yes_idx is not None else [0]

    # SETUP_ACTIVE_POKEMON / SETUP_BENCH_POKEMON → first CARD option
    if ctx in {1, 2}:
        for i, opt in enumerate(options):
            if int(opt.type) == int(OptionType.CARD):
                return [i]

    # Default — first minCount options
    take = max(min_count, 1) if max_count >= 1 else min_count
    take = min(take, max_count, len(options))
    return list(range(take))


def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    if int(obs.select.context) == 0:
        return _main_action_priority(obs)
    return _selection_action(obs)
