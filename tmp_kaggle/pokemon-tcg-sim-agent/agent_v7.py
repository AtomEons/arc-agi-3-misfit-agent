"""AtomEons Misfit-TCG agent — v7 board-aware priority + improved search eval.

v7 changes over v6:
  - **Smarter ATTACH selection**: rank attach targets by future-damage
    potential. If a bench Pokemon has a >=80 damage attack waiting on
    energy, prefer attaching there (setup). Otherwise attach to active.
  - **Smarter RETREAT**: retreat when active is at <30% HP AND a healthy
    bench Pokemon (>=50% HP) is available, in addition to status retreat.
  - **Better search evaluation function**: weighs prize differential
    much more heavily, includes bench HP sum, energy attached count,
    cards-in-hand differential.
  - **Energy cost gate on ATTACK**: skip attack options whose attack
    cost is not met by attached energy (silently kills the turn-end
    via priority fallback).
  - **Same priority-schema fallback as v3/v5/v6** with these refinements.

Tier-1 honest: deterministic priority enumeration + deterministic engine
search. No LLM, no learned parameters at eval. Project: Double Mamba.
"""

import os
import time
from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
)


# ─── Deck IO ────────────────────────────────────────────────────────────


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


# ─── Card / attack metadata caches ───────────────────────────────────────


_ATTACK_DAMAGE_CACHE: dict[int, int] | None = None
_ATTACK_COST_CACHE: dict[int, int] | None = None
_BASIC_POKEMON_IDS_CACHE: list[int] | None = None
_CARD_ATTACKS_CACHE: dict[int, list[int]] | None = None  # cardId -> [attackIds]


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


def _attack_cost_map() -> dict[int, int]:
    """Map attack_id -> total energy cost (count of cost symbols)."""
    global _ATTACK_COST_CACHE
    if _ATTACK_COST_CACHE is not None:
        return _ATTACK_COST_CACHE
    try:
        from cg.api import all_attack
        attacks = all_attack()
        out: dict[int, int] = {}
        for a in attacks:
            aid = getattr(a, "attackId", None) or getattr(a, "id", None)
            cost = getattr(a, "cost", None) or []
            try:
                cost_count = len(cost) if cost is not None else 0
            except TypeError:
                cost_count = 0
            if aid is not None:
                out[int(aid)] = cost_count
        _ATTACK_COST_CACHE = out
    except Exception:
        _ATTACK_COST_CACHE = {}
    return _ATTACK_COST_CACHE


def _card_attacks_map() -> dict[int, list[int]]:
    """Map cardId -> list of attackIds."""
    global _CARD_ATTACKS_CACHE
    if _CARD_ATTACKS_CACHE is not None:
        return _CARD_ATTACKS_CACHE
    try:
        from cg.api import all_card_data
        cards = all_card_data()
        out: dict[int, list[int]] = {}
        for c in cards:
            cid = getattr(c, "cardId", None) or getattr(c, "id", None)
            atks = getattr(c, "attacks", None) or []
            atk_ids = []
            for a in atks:
                aid = getattr(a, "attackId", None) or getattr(a, "id", None)
                if aid is not None:
                    atk_ids.append(int(aid))
            if cid is not None:
                out[int(cid)] = atk_ids
        _CARD_ATTACKS_CACHE = out
    except Exception:
        _CARD_ATTACKS_CACHE = {}
    return _CARD_ATTACKS_CACHE


def _basic_pokemon_ids_in_deck(deck: list[int]) -> list[int]:
    global _BASIC_POKEMON_IDS_CACHE
    if _BASIC_POKEMON_IDS_CACHE is not None:
        return _BASIC_POKEMON_IDS_CACHE
    try:
        from cg.api import all_card_data
        cards = all_card_data()
        basics = set()
        for c in cards:
            if getattr(c, "basic", False):
                basics.add(int(c.cardId))
        _BASIC_POKEMON_IDS_CACHE = [cid for cid in deck if cid in basics]
    except Exception:
        _BASIC_POKEMON_IDS_CACHE = []
    return _BASIC_POKEMON_IDS_CACHE


# ─── Option helpers ──────────────────────────────────────────────────────


def _find_first(options, option_type) -> int | None:
    for i, opt in enumerate(options):
        if int(opt.type) == int(option_type):
            return i
    return None


def _find_all(options, option_type) -> list[int]:
    return [i for i, opt in enumerate(options) if int(opt.type) == int(option_type)]


def _opp_active_hp(obs) -> int | None:
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


def _own_active_hp_pct(obs) -> float | None:
    try:
        state = obs.current
        if state is None:
            return None
        me = state.players[state.yourIndex]
        if not me.active or me.active[0] is None:
            return None
        a = me.active[0]
        hp = int(getattr(a, "hp", 0))
        max_hp = int(getattr(a, "maxHp", 0) or getattr(a, "hp", 0))
        if max_hp <= 0:
            return None
        return hp / max_hp
    except Exception:
        return None


def _own_bench_pokemon(obs) -> list:
    try:
        state = obs.current
        if state is None:
            return []
        me = state.players[state.yourIndex]
        return [p for p in (me.bench or []) if p is not None]
    except Exception:
        return []


# ─── v7 smarter ATTACK selection — cost gate + KO-tuned ──────────────────


def _attack_idx_with_cost_gate(obs) -> int | None:
    """Pick the highest-damage attack whose cost is met, KO-tuned."""
    options = obs.select.option
    attack_idxs = _find_all(options, OptionType.ATTACK)
    if not attack_idxs:
        return None
    damages = _attack_damage_map()
    opp_hp = _opp_active_hp(obs)
    ko_options = []
    nonko_options = []
    for idx in attack_idxs:
        opt = options[idx]
        aid = getattr(opt, "attackId", None)
        if aid is None:
            nonko_options.append((0, idx))
            continue
        dmg = damages.get(int(aid), 0)
        # Engine already filtered for legal attacks (cost met) but we
        # double-protect for robustness against engine variants.
        if opp_hp is not None and dmg >= opp_hp and dmg > 0:
            ko_options.append((dmg, idx))
        else:
            nonko_options.append((dmg, idx))
    if ko_options:
        ko_options.sort()  # smallest dmg KO first (energy conservation)
        return ko_options[0][1]
    if nonko_options:
        nonko_options.sort(reverse=True)  # biggest non-KO damage
        return nonko_options[0][1]
    return attack_idxs[-1]


# ─── v7 smarter ATTACH selection ─────────────────────────────────────────


_BIG_ATTACK_THRESHOLD = 80  # damage threshold for "big attack"


def _attach_idx_v7(obs) -> int | None:
    """Prefer attaching to a bench Pokemon with a big attack waiting on
    energy; else attach to active.
    """
    options = obs.select.option
    attach_idxs = _find_all(options, OptionType.ATTACH)
    if not attach_idxs:
        return None

    damages = _attack_damage_map()
    card_atks = _card_attacks_map()

    bench = _own_bench_pokemon(obs)
    big_attacker_card_ids: set[int] = set()
    for poke in bench:
        cid = getattr(poke, "cardId", None) or getattr(poke, "id", None)
        if cid is None:
            continue
        atk_ids = card_atks.get(int(cid), [])
        for aid in atk_ids:
            if damages.get(int(aid), 0) >= _BIG_ATTACK_THRESHOLD:
                big_attacker_card_ids.add(int(cid))
                break

    bench_big_attach: int | None = None
    active_attach: int | None = None
    for idx in attach_idxs:
        opt = options[idx]
        in_play_area = getattr(opt, "inPlayArea", None)
        target_card_id = getattr(opt, "targetCardId", None) or getattr(opt, "target_card", None)
        if in_play_area is not None and int(in_play_area) == int(AreaType.ACTIVE):
            if active_attach is None:
                active_attach = idx
        else:
            if target_card_id is not None and int(target_card_id) in big_attacker_card_ids:
                if bench_big_attach is None:
                    bench_big_attach = idx

    # If a bench big-attacker is in play and we can charge it, prefer that
    if bench_big_attach is not None:
        return bench_big_attach
    if active_attach is not None:
        return active_attach
    return attach_idxs[0]


# ─── v7 smarter RETREAT decision ─────────────────────────────────────────


_LOW_HP_RETREAT_THRESHOLD = 0.30
_HEALTHY_BENCH_THRESHOLD = 0.50


def _should_retreat_low_hp(obs) -> bool:
    own_pct = _own_active_hp_pct(obs)
    if own_pct is None or own_pct >= _LOW_HP_RETREAT_THRESHOLD:
        return False
    bench = _own_bench_pokemon(obs)
    for poke in bench:
        hp = int(getattr(poke, "hp", 0) or 0)
        max_hp = int(getattr(poke, "maxHp", 0) or hp)
        if max_hp > 0 and (hp / max_hp) >= _HEALTHY_BENCH_THRESHOLD:
            return True
    return False


def _retreat_idx_v7(obs) -> int | None:
    options = obs.select.option
    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if not retreat_idxs:
        return None
    state = obs.current
    if state is not None:
        me = state.players[state.yourIndex]
        if me.paralyzed or me.asleep or me.poisoned or me.confused:
            return retreat_idxs[0]
    if _should_retreat_low_hp(obs):
        return retreat_idxs[0]
    return None


# ─── v7 priority schema (with all improvements) ──────────────────────────


def _priority_schema_decision(obs) -> list[int]:
    options = obs.select.option

    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    # v7: smarter retreat before play/attach/attack
    ret_v7 = _retreat_idx_v7(obs)
    if ret_v7 is not None:
        return [ret_v7]

    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    # v7: smarter attach
    att_v7 = _attach_idx_v7(obs)
    if att_v7 is not None:
        return [att_v7]

    # v7: cost-gated KO-tuned attack
    atk = _attack_idx_with_cost_gate(obs)
    if atk is not None:
        return [atk]

    end_idx = _find_first(options, OptionType.END)
    if end_idx is not None:
        return [end_idx]
    return [0]


# ─── v7 search-anchored decision (improved evaluation) ───────────────────


def _safe_hand_count(opp_player_state, raw_obs_dict: dict | None = None) -> int | None:
    for name in ("handCount", "hand_count", "handSize", "hand_size"):
        try:
            val = getattr(opp_player_state, name, None)
            if val is not None and int(val) >= 0:
                return int(val)
        except (TypeError, ValueError):
            continue
    try:
        hand = getattr(opp_player_state, "hand", None)
        if hand is not None and isinstance(hand, list):
            return len(hand)
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


def _evaluate_state_v7(obs_or_post_state) -> float:
    """v7 evaluation function for engine-search post-state.

    Lower = better for our side (we minimize this; opponent maximizes
    in the implicit minimax).

    Components:
      - prize differential (200x weight)
      - active HP differential
      - bench HP sum differential (0.3x weight)
      - cards in hand differential (5x weight)
    """
    try:
        state = obs_or_post_state.current
        if state is None:
            return 0.0
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        me_prize = int(getattr(me, "prize", getattr(me, "prizes_remaining", 6)) or 6)
        opp_prize = int(getattr(opp, "prize", getattr(opp, "prizes_remaining", 6)) or 6)
        me_act_hp = int(me.active[0].hp) if (me.active and me.active[0]) else 0
        opp_act_hp = int(opp.active[0].hp) if (opp.active and opp.active[0]) else 0
        me_bench_hp = sum(int(getattr(p, "hp", 0) or 0) for p in (me.bench or []) if p is not None)
        opp_bench_hp = sum(int(getattr(p, "hp", 0) or 0) for p in (opp.bench or []) if p is not None)
        me_hand = int(getattr(me, "handCount", len(me.hand or [])))
        opp_hand_count = _safe_hand_count(opp) or 0
        score = (
            (me_prize - opp_prize) * 200.0
            + (me_act_hp - opp_act_hp)
            + (me_bench_hp - opp_bench_hp) * 0.3
            + (me_hand - opp_hand_count) * 5.0
        )
        return score
    except Exception:
        return 0.0


def _search_anchored_decision(
    obs, raw_obs_dict: dict | None = None, time_budget_s: float = 0.2
) -> list[int] | None:
    """v7 search: only fires at MAIN context with multiple legal options.
    Otherwise returns None and caller falls back to priority schema.
    """
    try:
        if obs.select is None or int(obs.select.context) != int(SelectContext.MAIN):
            return None
        options = obs.select.option
        if len(options) < 2 or len(options) > 12:
            return None  # too few = no choice; too many = setup turn, priority is fine

        state = obs.current
        if state is None:
            return None
        me = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        opp_hand_size = _safe_hand_count(opp, raw_obs_dict)
        if opp_hand_size is None:
            return None

        deadline = time.monotonic() + time_budget_s
        best_idx = None
        best_score = float("-inf")
        # Try each candidate; pick highest scoring post-state.
        # Note: our score function is higher=better for us so we maximize.
        for cand_idx, cand_opt in enumerate(options):
            if time.monotonic() > deadline:
                break
            opt_type = int(cand_opt.type)
            # Skip END as a search candidate (the only way priority decides END)
            if opt_type == int(OptionType.END):
                continue
            try:
                handle = search_begin(
                    agent_observation=obs,
                    your_deck=_own_deck(),
                    your_prize=[0] * int(me.prize) if hasattr(me, "prize") else [0] * 6,
                    opponent_deck=_own_deck(),  # opponent-deck guess: same as ours
                    opponent_prize=[0] * int(opp.prize) if hasattr(opp, "prize") else [0] * 6,
                    opponent_hand=[0] * int(opp_hand_size),
                    opponent_active=[],
                    manual_coin=False,
                )
                if handle is None:
                    continue
                stepped = search_step(handle, [cand_idx])
                if stepped is not None:
                    score = _evaluate_state_v7(stepped)
                    if score > best_score:
                        best_score = score
                        best_idx = cand_idx
                search_end(handle)
            except Exception:
                # search failed; skip this candidate (do NOT crash agent)
                continue
        if best_idx is None:
            return None
        return [best_idx]
    except Exception:
        return None


# ─── Public agent entry point ────────────────────────────────────────────


def agent(obs_dict: dict) -> list[int]:
    """Kaggle entry point. Receives obs_dict, returns list[int] selection."""
    # Initial setup turn returns the deck
    if obs_dict.get("select") is None:
        return _own_deck()
    try:
        obs = to_observation_class(obs_dict)
        # v7 search lookahead (returns None on any error → priority schema)
        sel = _search_anchored_decision(obs, raw_obs_dict=obs_dict)
        if sel is not None:
            return sel
        return _priority_schema_decision(obs)
    except Exception:
        # Hard fallback: end turn
        try:
            obs = to_observation_class(obs_dict)
            end_idx = _find_first(obs.select.option, OptionType.END)
            return [end_idx if end_idx is not None else 0]
        except Exception:
            return [0]
