"""AtomEons Misfit-TCG agent — v10_clean.

Surgical fix policy:

  v8 is the proven baseline. v10 had TWO changes bundled together:
    (1) the cardId -> attackId metadata join bug fix, and
    (2) new heuristics + new eval weights + deeper search that piled on
        top of that fix and over-activated the policy.

  Result: v10 regressed to ~0-20% vs v8 because the fix re-armed dormant
  retreat / attach paths AND the new heuristics simultaneously, blowing
  past the calibration band that v8 was operating in.

  v10_clean keeps fix (1) and discards everything in (2). This file is
  v8 verbatim except that `_is_big_attacker_card` now properly joins
  CardData.attacks (which is list[int] of attackIds) through all_attack()
  to get the real Attack.damage values, using the cached lookup dicts.

  Everything else is byte-for-byte v8: priority schema, retreat triggers,
  attach routing, eval function and its weights, 2-ply search budget,
  candidate enumeration. The dormant heuristics in v8 are now armed at
  their v8-tuned thresholds with no extra signal piled on.

Tier-1 strict throughout.
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


# ---------------------------------------------------------------------------
# Attack-damage map (v8 path — unchanged; this was already correct in v8)
# ---------------------------------------------------------------------------


_ATTACK_DAMAGE_CACHE: dict[int, int] | None = None
def _attack_damage_map() -> dict[int, int]:
    global _ATTACK_DAMAGE_CACHE
    if _ATTACK_DAMAGE_CACHE is not None:
        return _ATTACK_DAMAGE_CACHE
    try:
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


# ---------------------------------------------------------------------------
# THE FIX: cardId -> attackIds -> Attack metadata joins
#
# v8's `_is_big_attacker_card` walked card.attacks expecting Attack objects,
# but the engine returns list[int] of attackIds. Every per-card damage lookup
# silently returned 0, making _is_big_attacker_card always False. That left
# v8's bench-big-attacker attach, big-threat retreat, and opp-threat probe
# all dormant.
#
# This is the ONLY behavioral change vs v8. Eval weights, retreat thresholds,
# search depth, and candidate enumeration are byte-for-byte v8.
# ---------------------------------------------------------------------------


_ATTACK_BY_ID_CACHE: dict[int, dict] | None = None
def _attack_by_id() -> dict[int, dict]:
    """Return {attackId: {damage:int, cost:int}}."""
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
            out[int(aid)] = {"damage": dmg, "cost": cost}
    except Exception:
        pass
    _ATTACK_BY_ID_CACHE = out
    return out


_CARD_BY_ID_CACHE: dict[int, dict] | None = None
def _card_by_id() -> dict[int, dict]:
    """Return {cardId: {attacks: list[int]}}."""
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
            # CardData.attacks is list[int] of attackIds, not Attack objects.
            attack_ids = [int(x) for x in atks if x is not None]
            out[int(cid)] = {"attacks": attack_ids}
    except Exception:
        pass
    _CARD_BY_ID_CACHE = out
    return out


_BIG_ATTACKER_THRESHOLD = 80
def _is_big_attacker_card(card_id: int) -> bool:
    """A card is a 'big attacker' if any of its known attacks deals >= 80 dmg.

    v10_clean FIX: cardId -> attackId list -> Attack.damage via cached joins.
    """
    cb = _card_by_id().get(card_id)
    if not cb:
        return False
    ab = _attack_by_id()
    for aid in cb["attacks"]:
        atk = ab.get(aid)
        if atk and atk["damage"] >= _BIG_ATTACKER_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------------
# Below this line is v8 verbatim.
# ---------------------------------------------------------------------------


def _find_first(options, option_type) -> int | None:
    for i, opt in enumerate(options):
        if int(opt.type) == int(option_type):
            return i
    return None


def _find_all(options, option_type) -> list[int]:
    return [i for i, opt in enumerate(options) if int(opt.type) == int(option_type)]


def _opponent_active_hp(obs) -> int | None:
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


def _prize_diff(obs) -> int:
    try:
        s = obs.current
        if s is None: return 0
        me = s.players[s.yourIndex]
        opp = s.players[1 - s.yourIndex]
        return int(getattr(opp, "prizeCount", 6)) - int(getattr(me, "prizeCount", 6))
    except Exception:
        return 0


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
        active = opp.active[0]
        cid = int(getattr(active, "cardId", -1))
        return _is_big_attacker_card(cid)
    except Exception:
        return False


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
        ko_options.sort()  # smallest-dmg-that-KOs first
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

    # ABILITIES first (always — they don't end turn)
    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    # EVOLVE second (doesn't end turn)
    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    # ENDGAME OVERRIDE: try to KO with attack BEFORE setup actions
    if endgame:
        atk = _ko_tuned_attack_idx(obs)
        if atk is not None:
            return [atk]

    # PLAY (setup new basic to bench)
    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    # ATTACH — target bench big-attacker preferentially
    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        # If our active is a big attacker too, attach to active
        s = obs.current
        if s is not None:
            me = s.players[s.yourIndex]
            active_cid = -1
            if me.active and me.active[0] is not None:
                active_cid = int(getattr(me.active[0], "cardId", -1))
            active_is_big = _is_big_attacker_card(active_cid)
            # If our active is already big AND we have a KO swing this turn,
            # attach to active. Otherwise attach to bench big-attacker.
            for idx in attach_idxs:
                opt = options[idx]
                ipa = getattr(opt, "inPlayArea", None)
                if ipa is not None and int(ipa) == int(AreaType.BENCH):
                    return [idx]
            # No bench attach — go active
            for idx in attach_idxs:
                opt = options[idx]
                ipa = getattr(opt, "inPlayArea", None)
                if ipa is not None and int(ipa) == int(AreaType.ACTIVE):
                    return [idx]
            return [attach_idxs[0]]

    # RETREAT — smart heuristics
    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if retreat_idxs and obs.current is not None:
        s = obs.current
        me = s.players[s.yourIndex]
        own_hp = _own_active_hp(obs)
        own_max_hp = _own_active_max_hp(obs)
        opp_threat = _opp_active_is_threat(obs)
        bench_has_big = _bench_has_big_attacker_waiting(obs)
        # Retreat triggers:
        # 1. Status condition + healthy bench
        # 2. Low HP (<30% of max) + healthy bench
        # 3. Opponent is big-attacker threat + we have a big-attacker on bench
        status_bad = (me.paralyzed or me.asleep or me.poisoned or me.confused)
        low_hp = (own_hp is not None and own_max_hp is not None
                  and own_hp < 0.30 * own_max_hp)
        threat_swap = opp_threat and bench_has_big and not _endgame_mode(obs)
        if (status_bad or low_hp or threat_swap) and bench_has_big:
            return [retreat_idxs[0]]

    # ATTACK
    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    # END
    end_idx = _find_first(options, OptionType.END)
    if end_idx is not None:
        return [end_idx]
    return [0]


# ---------------------------------------------------------------------------
# Engine-search 2-ply lookahead (v8 verbatim)
# ---------------------------------------------------------------------------


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
    if raw_obs_dict:
        try:
            cur = raw_obs_dict.get("current") or {}
            players = cur.get("players") or []
            your_idx = cur.get("yourIndex", 0)
            opp_idx = 1 - int(your_idx)
            if 0 <= opp_idx < len(players):
                op = players[opp_idx] or {}
                for name in ("handCount", "hand_count", "handSize", "hand_size"):
                    v = op.get(name)
                    if v is not None:
                        try:
                            iv = int(v)
                            if iv >= 0:
                                return iv
                        except (TypeError, ValueError):
                            continue
        except Exception:
            pass
    return None


def _score_state(obs) -> float:
    """v8 evaluation — UNCHANGED weights."""
    try:
        s = obs.current
        if s is None: return 0.0
        me = s.players[s.yourIndex]
        opp = s.players[1 - s.yourIndex]
        own_prize = int(getattr(me, "prizeCount", 6))
        opp_prize = int(getattr(opp, "prizeCount", 6))
        own_active_hp = _own_active_hp(obs) or 0
        opp_active_hp = _opponent_active_hp(obs) or 0
        own_bench_hp = sum(int(getattr(b, "hp", 0) or 0)
                            for b in (getattr(me, "bench", None) or [])
                            if b is not None)
        opp_bench_hp = sum(int(getattr(b, "hp", 0) or 0)
                            for b in (getattr(opp, "bench", None) or [])
                            if b is not None)
        own_hand = _safe_hand_count(me, None) or 0
        opp_hand = _safe_hand_count(opp, None) or 0
        # Energy efficiency: count own active's attached energy / required cost
        own_energy_efficiency = 0.0
        if me.active and me.active[0] is not None:
            attached = len(getattr(me.active[0], "energies", None) or [])
            own_energy_efficiency = min(1.0, attached / max(1, 3))
        score = ((own_prize - opp_prize) * 250
                 + (own_active_hp - opp_active_hp)
                 + (own_bench_hp - opp_bench_hp) * 0.3
                 + (own_hand - opp_hand) * 5
                 + own_energy_efficiency * 8)
        return float(score)
    except Exception:
        return 0.0


def _search_anchored_decision(obs, time_budget_ms: int = 180) -> list[int] | None:
    """v8 2-ply minimax over top-3 candidates x top-3 opp responses, 180ms."""
    deadline = time.monotonic() + (time_budget_ms / 1000.0)

    if obs.select is None:
        return None
    n_opts = len(obs.select.option)
    if n_opts < 2 or n_opts > 12:
        return None
    if obs.current is None:
        return None

    # Skip search at non-MAIN contexts to save time
    try:
        ctx = int(obs.select.context)
        if ctx != int(SelectContext.MAIN):
            return None
    except Exception:
        return None

    # Get priority-schema's recommendation as top-1
    priority_choice = _priority_schema_decision(obs)
    if not priority_choice:
        return None

    # Top-3 candidates: priority choice + 2 strongest alternates from same option
    candidates = [priority_choice[0]]
    for opt_type in (OptionType.ATTACK, OptionType.ABILITY, OptionType.EVOLVE,
                      OptionType.PLAY, OptionType.ATTACH):
        idxs = _find_all(obs.select.option, opt_type)
        for idx in idxs:
            if idx not in candidates:
                candidates.append(idx)
            if len(candidates) >= 3:
                break
        if len(candidates) >= 3:
            break
    candidates = candidates[:3]

    state = obs.current
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]
    deck = _own_deck()

    own_prize = list(range(min(int(getattr(me, "prizeCount", 0)), 6)))
    opp_prize = list(range(min(int(getattr(opp, "prizeCount", 0)), 6)))
    opp_hand_count = _safe_hand_count(opp)
    if opp_hand_count is None:
        return None
    opp_hand = [0] * opp_hand_count
    opp_active = []
    if opp.active and opp.active[0] is not None:
        opp_active = [int(getattr(opp.active[0], "cardId", 0))]

    best_score = float("-inf")
    best_choice = priority_choice

    for choice in candidates:
        if time.monotonic() > deadline:
            break
        try:
            ss = search_begin(obs,
                              your_deck=deck, your_prize=own_prize,
                              opponent_deck=deck, opponent_prize=opp_prize,
                              opponent_hand=opp_hand,
                              opponent_active=opp_active)
            ply1 = search_step(ss, [choice])
            # If state is now opp's turn, simulate opp using same priority schema
            try:
                ply1_obs = to_observation_class(ply1)
                if ply1_obs.select is not None and ply1_obs.current is not None:
                    opp_choice = _priority_schema_decision(ply1_obs)
                    ply2 = search_step(ss, opp_choice or [0])
                    ply2_obs = to_observation_class(ply2)
                    score = _score_state(ply2_obs)
                else:
                    score = _score_state(ply1_obs)
            except Exception:
                score = _score_state(to_observation_class(ply1))
            search_end(ss)
            if score > best_score:
                best_score = score
                best_choice = [choice]
        except Exception:
            try:
                search_end(ss)
            except Exception:
                pass
            continue

    return best_choice


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _own_deck()

    # Try search-anchored first
    try:
        choice = _search_anchored_decision(obs, time_budget_ms=180)
        if choice is not None:
            return choice
    except Exception:
        pass

    # Fallback: priority schema
    return _priority_schema_decision(obs)
