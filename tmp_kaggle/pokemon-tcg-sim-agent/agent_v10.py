"""AtomEons Misfit-TCG agent — v10 BIG-ATTACKER FIX + REAL ENERGY-READINESS.

v10 changes over v8/v9:

  - **CRITICAL BUG FIX.** v6 through v9 had a silent bug: `_card_attack_list`
    treated `card.attacks` (which is `list[int]` of attackIds) as a list of
    Attack objects. `getattr(a, "damage", 0)` returned 0 for every card.
    Big-attacker preemption + energy-readiness scoring were silently dead.
    v10 properly joins cardId -> attackIds -> Attack objects via
    `all_attack()` for the damage and energy-cost lookups.

  - **Energy-readiness scoring now ACTUALLY works.** When our active is
    one attach away from a KO-class attack, the search lookahead now
    sees the +8 weight on energy_ready_next, which steers it toward
    attaching that energy.

  - **Big-attacker preemption now ACTUALLY works.** When opp active is
    a known >=80-damage attacker, retreat to bench big-attacker fires.

  - **Threat-aware ATTACH targeting now ACTUALLY works.** Attach goes to
    bench Pokemon with known >=80-damage attacks first.

This is not an architectural change — it's the architecture WORKING for
the first time. Per AtomEons doctrine: Mom watches what runs, not what
the comment claims.

Tier-1 strict throughout. Project: Double Mamba — AGI Synergy Unit.
"""

import os
import time
from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
    all_attack, all_card_data,
)


def read_deck_csv() -> list[int]:
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        for c in ("/kaggle_simulations/agent/deck.csv",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "deck.csv")):
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


# ---------------------------------------------------------------------------
# Card / attack metadata — FIXED join in v10
# ---------------------------------------------------------------------------


_ATTACK_BY_ID_CACHE: dict[int, dict] | None = None
def _attack_by_id() -> dict[int, dict]:
    """Return {attackId: {damage:int, cost:int, name:str}} dict."""
    global _ATTACK_BY_ID_CACHE
    if _ATTACK_BY_ID_CACHE is not None:
        return _ATTACK_BY_ID_CACHE
    out: dict[int, dict] = {}
    try:
        for a in all_attack():
            aid = getattr(a, "attackId", None)
            if aid is None:
                continue
            dmg = int(getattr(a, "damage", 0) or 0)
            energies = getattr(a, "energies", None) or []
            cost = len(energies) if isinstance(energies, list) else 0
            out[int(aid)] = {"damage": dmg, "cost": cost,
                              "name": getattr(a, "name", "?")}
    except Exception:
        pass
    _ATTACK_BY_ID_CACHE = out
    return out


_CARD_BY_ID_CACHE: dict[int, dict] | None = None
def _card_by_id() -> dict[int, dict]:
    """Return {cardId: {attacks: list[int], basic: bool, hp: int, name: str}}."""
    global _CARD_BY_ID_CACHE
    if _CARD_BY_ID_CACHE is not None:
        return _CARD_BY_ID_CACHE
    out: dict[int, dict] = {}
    try:
        for c in all_card_data():
            cid = getattr(c, "cardId", None)
            if cid is None:
                continue
            atks = getattr(c, "attacks", None) or []
            # PROPERLY: card.attacks is list[int] of attackIds, not Attack objects
            attack_ids = [int(x) for x in atks if x is not None]
            out[int(cid)] = {
                "attacks": attack_ids,
                "basic": bool(getattr(c, "basic", False)),
                "hp": int(getattr(c, "hp", 0) or 0),
                "name": getattr(c, "name", "?"),
                "energyType": int(getattr(c, "energyType", 0) or 0),
                "retreatCost": int(getattr(c, "retreatCost", 0) or 0),
            }
    except Exception:
        pass
    _CARD_BY_ID_CACHE = out
    return out


def _card_attack_damages_and_costs(card_id: int) -> list[tuple[int, int]]:
    """Return list of (damage, cost) for each attack of this card.
    v10 FIX: properly joins through attackId.
    """
    cb = _card_by_id().get(card_id)
    if not cb:
        return []
    ab = _attack_by_id()
    out = []
    for aid in cb["attacks"]:
        atk = ab.get(aid)
        if atk:
            out.append((atk["damage"], atk["cost"]))
    return out


_BIG_ATTACKER_THRESHOLD = 80
def _is_big_attacker_card(card_id: int) -> bool:
    return any(d >= _BIG_ATTACKER_THRESHOLD
               for d, _ in _card_attack_damages_and_costs(card_id))


# ---------------------------------------------------------------------------
# Option helpers
# ---------------------------------------------------------------------------


def _find_first(options, option_type) -> int | None:
    for i, opt in enumerate(options):
        if int(opt.type) == int(option_type):
            return i
    return None


def _find_all(options, option_type) -> list[int]:
    return [i for i, opt in enumerate(options) if int(opt.type) == int(option_type)]


def _opp_active_hp(obs) -> int | None:
    try:
        s = obs.current
        if s is None: return None
        opp = s.players[1 - s.yourIndex]
        if not opp.active or opp.active[0] is None: return None
        return int(opp.active[0].hp)
    except Exception:
        return None


def _own_active_hp(obs) -> int | None:
    try:
        s = obs.current
        if s is None: return None
        me = s.players[s.yourIndex]
        if not me.active or me.active[0] is None: return None
        return int(me.active[0].hp)
    except Exception:
        return None


def _own_active_max_hp(obs) -> int | None:
    try:
        s = obs.current
        if s is None: return None
        me = s.players[s.yourIndex]
        if not me.active or me.active[0] is None: return None
        return int(getattr(me.active[0], "maxHp", me.active[0].hp))
    except Exception:
        return None


def _own_active_energy_count(obs) -> int:
    try:
        s = obs.current
        if s is None: return 0
        me = s.players[s.yourIndex]
        if not me.active or me.active[0] is None: return 0
        return len(getattr(me.active[0], "energies", None) or [])
    except Exception:
        return 0


def _own_active_attack_costs(obs) -> list[int]:
    try:
        s = obs.current
        if s is None: return []
        me = s.players[s.yourIndex]
        if not me.active or me.active[0] is None: return []
        cid = int(getattr(me.active[0], "cardId", -1))
        return [c for _, c in _card_attack_damages_and_costs(cid)]
    except Exception:
        return []


def _bench_has_big_attacker_waiting(obs) -> bool:
    try:
        s = obs.current
        if s is None: return False
        me = s.players[s.yourIndex]
        bench = getattr(me, "bench", None) or []
        for slot in bench:
            if slot is None: continue
            cid = int(getattr(slot, "cardId", -1))
            if _is_big_attacker_card(cid):
                return True
    except Exception:
        return False
    return False


def _opp_active_is_threat(obs) -> bool:
    try:
        s = obs.current
        if s is None: return False
        opp = s.players[1 - s.yourIndex]
        if not opp.active or opp.active[0] is None: return False
        cid = int(getattr(opp.active[0], "cardId", -1))
        return _is_big_attacker_card(cid)
    except Exception:
        return False


def _hand_big_attacker_count(obs) -> int:
    try:
        s = obs.current
        if s is None: return 0
        me = s.players[s.yourIndex]
        hand = getattr(me, "hand", None) or []
        n = 0
        for c in hand:
            if c is None: continue
            cid = int(getattr(c, "cardId", -1))
            if _is_big_attacker_card(cid):
                n += 1
        return n
    except Exception:
        return 0


def _ko_tuned_attack_idx(obs) -> int | None:
    options = obs.select.option
    attack_idxs = _find_all(options, OptionType.ATTACK)
    if not attack_idxs:
        return None
    ab = _attack_by_id()
    opp_hp = _opp_active_hp(obs)
    ko_options = []
    nonko_options = []
    for idx in attack_idxs:
        opt = options[idx]
        aid = getattr(opt, "attackId", None)
        dmg = ab.get(int(aid), {}).get("damage", 0) if aid is not None else 0
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


def _endgame_mode(obs) -> bool:
    try:
        s = obs.current
        if s is None: return False
        me = s.players[s.yourIndex]
        opp = s.players[1 - s.yourIndex]
        own_prize = int(getattr(me, "prizeCount", 6))
        opp_prize = int(getattr(opp, "prizeCount", 6))
        return opp_prize <= 1 or (own_prize - opp_prize) >= 5
    except Exception:
        return False


def _priority_schema_decision(obs) -> list[int]:
    options = obs.select.option
    endgame = _endgame_mode(obs)

    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs: return [ability_idxs[0]]

    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs: return [evolve_idxs[0]]

    if endgame:
        atk = _ko_tuned_attack_idx(obs)
        if atk is not None: return [atk]

    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs: return [play_idxs[0]]

    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        own_energy = _own_active_energy_count(obs)
        costs = _own_active_attack_costs(obs)
        # v10 FIX: this now actually fires because costs is non-empty
        active_ready_next_attach = any(0 < c <= own_energy + 1 for c in costs)
        if active_ready_next_attach:
            for idx in attach_idxs:
                opt = options[idx]
                ipa = getattr(opt, "inPlayArea", None)
                if ipa is not None and int(ipa) == int(AreaType.ACTIVE):
                    return [idx]
        # Otherwise attach to bench big-attacker (v10 FIX: this now actually fires)
        for idx in attach_idxs:
            opt = options[idx]
            ipa = getattr(opt, "inPlayArea", None)
            if ipa is not None and int(ipa) == int(AreaType.BENCH):
                # Check the bench slot for big-attacker card
                s = obs.current
                if s is not None:
                    me = s.players[s.yourIndex]
                    bench = getattr(me, "bench", None) or []
                    slot_idx = getattr(opt, "inPlayIndex", None)
                    if slot_idx is not None and 0 <= int(slot_idx) < len(bench):
                        slot = bench[int(slot_idx)]
                        if slot is not None:
                            cid = int(getattr(slot, "cardId", -1))
                            if _is_big_attacker_card(cid):
                                return [idx]
        return [attach_idxs[0]]

    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if retreat_idxs and obs.current is not None:
        s = obs.current
        me = s.players[s.yourIndex]
        own_hp = _own_active_hp(obs)
        own_max_hp = _own_active_max_hp(obs)
        opp_threat = _opp_active_is_threat(obs)  # v10 FIX: now actually fires
        bench_has_big = _bench_has_big_attacker_waiting(obs)  # v10 FIX
        status_bad = (me.paralyzed or me.asleep or me.poisoned or me.confused)
        low_hp = (own_hp is not None and own_max_hp is not None
                  and own_hp < 0.30 * own_max_hp)
        threat_swap = opp_threat and bench_has_big and not endgame
        if (status_bad or low_hp or threat_swap) and bench_has_big:
            return [retreat_idxs[0]]

    atk = _ko_tuned_attack_idx(obs)
    if atk is not None: return [atk]

    end_idx = _find_first(options, OptionType.END)
    if end_idx is not None: return [end_idx]
    return [0]


def _safe_hand_count(opp_player_state, raw_obs_dict: dict | None = None) -> int | None:
    for name in ("handCount", "hand_count", "handSize", "hand_size"):
        try:
            v = getattr(opp_player_state, name, None)
            if v is not None and int(v) >= 0:
                return int(v)
        except (TypeError, ValueError):
            continue
    try:
        h = getattr(opp_player_state, "hand", None)
        if h is not None and isinstance(h, list):
            return len(h)
    except Exception:
        pass
    return None


def _score_state(obs) -> float:
    """v10 eval — same as v9 BUT all the big-attacker signals now fire."""
    try:
        s = obs.current
        if s is None: return 0.0
        me = s.players[s.yourIndex]
        opp = s.players[1 - s.yourIndex]
        own_prize = int(getattr(me, "prizeCount", 6))
        opp_prize = int(getattr(opp, "prizeCount", 6))
        own_active_hp = _own_active_hp(obs) or 0
        opp_active_hp = _opp_active_hp(obs) or 0
        own_bench_hp = sum(int(getattr(b, "hp", 0) or 0)
                            for b in (getattr(me, "bench", None) or [])
                            if b is not None)
        opp_bench_hp = sum(int(getattr(b, "hp", 0) or 0)
                            for b in (getattr(opp, "bench", None) or [])
                            if b is not None)
        own_hand = _safe_hand_count(me, None) or 0
        opp_hand = _safe_hand_count(opp, None) or 0
        own_energy = _own_active_energy_count(obs)
        own_costs = _own_active_attack_costs(obs)
        # v10 FIX: energy-ready signal now actually computed correctly
        energy_ready_next = 0
        if own_costs and any(c > 0 for c in own_costs):
            energy_gap = min(max(0, c - own_energy) for c in own_costs if c > 0)
            energy_ready_next = max(0, 3 - energy_gap)
        # Hand big-attacker bonus (replaces vague hand_setup_potential)
        hand_big = _hand_big_attacker_count(obs)
        # NEW: bench_big_attacker_count
        bench_big = sum(1 for b in (getattr(me, "bench", None) or [])
                          if b is not None
                          and _is_big_attacker_card(int(getattr(b, "cardId", -1))))
        score = ((own_prize - opp_prize) * 250
                 + (own_active_hp - opp_active_hp)
                 + (own_bench_hp - opp_bench_hp) * 0.3
                 + (own_hand - opp_hand) * 5
                 + energy_ready_next * 8
                 + hand_big * 5
                 + bench_big * 12)
        return float(score)
    except Exception:
        return 0.0


def _search_anchored_decision(obs, time_budget_ms: int = 220) -> list[int] | None:
    deadline = time.monotonic() + (time_budget_ms / 1000.0)

    if obs.select is None: return None
    n_opts = len(obs.select.option)
    if n_opts < 2 or n_opts > 12: return None
    if obs.current is None: return None
    try:
        ctx = int(obs.select.context)
        if ctx != int(SelectContext.MAIN): return None
    except Exception:
        return None

    priority_choice = _priority_schema_decision(obs)
    if not priority_choice: return None

    candidates = [priority_choice[0]]
    for opt_type in (OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                      OptionType.PLAY, OptionType.ATTACH, OptionType.RETREAT):
        for idx in _find_all(obs.select.option, opt_type):
            if idx not in candidates:
                candidates.append(idx)
            if len(candidates) >= 4: break
        if len(candidates) >= 4: break
    candidates = candidates[:4]

    state = obs.current
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]
    deck = _own_deck()

    own_prize = list(range(min(int(getattr(me, "prizeCount", 0)), 6)))
    opp_prize = list(range(min(int(getattr(opp, "prizeCount", 0)), 6)))
    opp_hand_count = _safe_hand_count(opp)
    if opp_hand_count is None: return None
    opp_hand = [0] * opp_hand_count
    opp_active = []
    if opp.active and opp.active[0] is not None:
        opp_active = [int(getattr(opp.active[0], "cardId", 0))]

    best_score = float("-inf")
    best_choice = priority_choice

    for choice in candidates:
        if time.monotonic() > deadline: break
        try:
            ss = search_begin(obs,
                              your_deck=deck, your_prize=own_prize,
                              opponent_deck=deck, opponent_prize=opp_prize,
                              opponent_hand=opp_hand,
                              opponent_active=opp_active)
            ply1 = search_step(ss, [choice])
            ply1_obs = to_observation_class(ply1)
            opp_scores = []
            if ply1_obs.select is not None and ply1_obs.current is not None:
                opp_priority = _priority_schema_decision(ply1_obs)
                opp_options = ply1_obs.select.option or []
                opp_candidates = [opp_priority[0] if opp_priority else 0]
                for ot in (OptionType.ATTACK, OptionType.ABILITY,
                           OptionType.PLAY):
                    for j in _find_all(opp_options, ot):
                        if j not in opp_candidates:
                            opp_candidates.append(j)
                        if len(opp_candidates) >= 3: break
                    if len(opp_candidates) >= 3: break
                for opp_choice in opp_candidates[:3]:
                    try:
                        ss2 = search_begin(obs,
                                            your_deck=deck, your_prize=own_prize,
                                            opponent_deck=deck, opponent_prize=opp_prize,
                                            opponent_hand=opp_hand,
                                            opponent_active=opp_active)
                        _ = search_step(ss2, [choice])
                        ply2 = search_step(ss2, [opp_choice])
                        score = _score_state(to_observation_class(ply2))
                        opp_scores.append(score)
                        search_end(ss2)
                    except Exception:
                        try: search_end(ss2)
                        except Exception: pass
                        continue
            search_end(ss)
            if opp_scores:
                avg = sum(opp_scores) / len(opp_scores)
            else:
                avg = _score_state(ply1_obs)
            if avg > best_score:
                best_score = avg
                best_choice = [choice]
        except Exception:
            try: search_end(ss)
            except Exception: pass
            continue

    return best_choice


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _own_deck()
    try:
        choice = _search_anchored_decision(obs, time_budget_ms=220)
        if choice is not None:
            return choice
    except Exception:
        pass
    return _priority_schema_decision(obs)
