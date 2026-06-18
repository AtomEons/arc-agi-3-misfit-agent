"""AtomEons Misfit-TCG agent — v15 RECURSIVE SELF-REPRESENTATION.

v15 is the first PTCG agent with the Self-Model Module (SMM) integrated
into its decision loop. It is the closure of the higher-order self-
representation loop for a game-playing substrate:

  - Per-game InMemorySpine + PTCGSelfModelModule (reset on the setup turn
    — detected when obs.select is None).
  - On every agent() call:
      a. PUSH the current obs context into the spine as one event row.
      b. Every 5 events the spine collected, EMIT one self-receipt.
      c. READ the most recent 3 self-receipts.
      d. PASS those self-receipts into _score_state as an EXTRA TERM:
         - If the most recent self-receipt's prediction was "expect KO
           next turn" (PRIZE_DRAW after ATTACK with ko=True) AND we
           can KO opp active this turn: BOOST score by +50.
         - If self-receipt predicted "expect prize draw" AND opp's
           prizeCount differs from the spine snapshot: ALIGNMENT bonus
           +30.
         - Otherwise: ZERO contribution. The v8_psychic policy is the
           floor; the SMM never degrades the baseline.

Doctrine reference: C:/AtomEons/orangebox/docs/PHENOMENON_APPROACH_v1.md
Disclosure: ATOM-PHENOMENON-v1-2026-0617 (PTCG closed-loop integration).

Architectural property (regardless of arena win rate):
  v15's representation of the world INCLUDES v15's representation of
  itself representing the world. The substrate reads its own first-
  person self-receipts as a higher-order term in the same evaluation
  function that picks its next move. That is loop closure.

Tier-1 strict:
  - No LLM, no learned weights, no pretrained parameters.
  - Deterministic spine event log, deterministic template fill,
    deterministic transition rule matching, deterministic scoring.
  - v8_psychic priority schema and 2-ply minimax preserved as the floor.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
)

from ptcg_self_receipt_layer import (
    InMemorySpine,
    PTCGSelfModelModule,
)


# ---------------------------------------------------------------------------
# Deck IO — read deck.csv (psychic content per task)
# ---------------------------------------------------------------------------


def read_deck_csv() -> list[int]:
    """Read deck.csv (psychic content). Same surface as v8_psychic."""
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
# Attack/card metadata caches (unchanged from v8_psychic)
# ---------------------------------------------------------------------------


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


_BIG_ATTACKER_THRESHOLD = 80
def _is_big_attacker_card(card_id: int) -> bool:
    """A card is a 'big attacker' if any of its known attacks deals >= 80 dmg."""
    try:
        from cg.api import all_card_data
        for c in all_card_data():
            if int(getattr(c, "cardId", -1)) != card_id:
                continue
            atks = getattr(c, "attacks", None) or []
            for a in atks:
                dmg = int(getattr(a, "damage", 0) or 0)
                if dmg >= _BIG_ATTACKER_THRESHOLD:
                    return True
            return False
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# Obs helpers (unchanged from v8_psychic)
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


def _own_prize_count(obs) -> int | None:
    try:
        s = obs.current
        if s is None: return None
        me = s.players[s.yourIndex]
        return int(getattr(me, "prizeCount", 6))
    except Exception:
        return None


def _opp_prize_count(obs) -> int | None:
    try:
        s = obs.current
        if s is None: return None
        opp = s.players[1 - s.yourIndex]
        return int(getattr(opp, "prizeCount", 6))
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


def _own_active_card_id(obs) -> int:
    try:
        s = obs.current
        if s is None: return -1
        me = s.players[s.yourIndex]
        if not me.active or me.active[0] is None: return -1
        return int(getattr(me.active[0], "cardId", -1))
    except Exception:
        return -1


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


def _can_ko_this_turn(obs) -> bool:
    """True if we have an ATTACK option whose damage >= opp active HP."""
    if obs.select is None:
        return False
    options = obs.select.option
    attack_idxs = _find_all(options, OptionType.ATTACK)
    if not attack_idxs:
        return False
    damages = _attack_damage_map()
    opp_hp = _opponent_active_hp(obs)
    if opp_hp is None:
        return False
    for idx in attack_idxs:
        opt = options[idx]
        aid = getattr(opt, "attackId", None)
        dmg = damages.get(int(aid), 0) if aid is not None else 0
        if dmg >= opp_hp and dmg > 0:
            return True
    return False


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


# ---------------------------------------------------------------------------
# v15: Per-game Self-Model Module wiring
# ---------------------------------------------------------------------------


# Per-process SMM state. Reset on setup-turn (obs.select is None) so each
# game gets a fresh spine. In a single-process arena run we use module-
# level globals; this is safe because the arena drives games sequentially.
_SPINE: Optional[InMemorySpine] = None
_SMM: Optional[PTCGSelfModelModule] = None
_EMIT_INTERVAL: int = 5  # emit a self-receipt every N spine events
# Track last opp prize count we observed at receipt-emission time, so we
# can detect "prize drew" since the receipt was written.
_LAST_RECEIPT_OPP_PRIZE: Optional[int] = None


def _reset_smm() -> None:
    """Fresh SMM per game. Called at setup-turn detection."""
    global _SPINE, _SMM, _LAST_RECEIPT_OPP_PRIZE
    _SPINE = InMemorySpine()
    _SMM = PTCGSelfModelModule(spine=_SPINE, actor="Misfit-PTCG-v15")
    _LAST_RECEIPT_OPP_PRIZE = None


def _ensure_smm() -> None:
    """Make sure the SMM exists. Lazy init on first non-setup obs."""
    global _SPINE, _SMM
    if _SPINE is None or _SMM is None:
        _reset_smm()


def _infer_move_from_obs(obs) -> Optional[str]:
    """Guess the action move-name we are ABOUT to choose, from the
    available SelectContext + option types. This is the substrate's
    representation of what it is doing 'now' (before the engine sees it).

    For the spine, we want a stable, low-cardinality vocabulary aligned
    with PTCG_TRANSITION_RULES:
      ATTACK, EVOLVE, ATTACH, PLAY, RETREAT, END_TURN, ABILITY, SELECT.
    """
    try:
        ctx = int(obs.select.context) if obs.select is not None else -1
    except Exception:
        ctx = -1
    options = obs.select.option if obs.select is not None else []

    # If MAIN context, look at the priority-schema's likely top pick to
    # decide which family of moves dominates this turn.
    type_counts: dict[int, int] = {}
    for opt in options:
        t = int(opt.type)
        type_counts[t] = type_counts.get(t, 0) + 1

    # Priority order matches v8_psychic's _priority_schema_decision so the
    # spine event tags what the agent's policy is actually pursuing.
    for ot, name in (
        (OptionType.ABILITY, "ABILITY"),
        (OptionType.EVOLVE, "EVOLVE"),
        (OptionType.PLAY, "PLAY"),
        (OptionType.ATTACH, "ATTACH"),
        (OptionType.RETREAT, "RETREAT"),
        (OptionType.ATTACK, "ATTACK"),
        (OptionType.END, "END_TURN"),
    ):
        if int(ot) in type_counts:
            return name
    return "SELECT"


def _push_obs_event(obs) -> None:
    """Push one spine event for the current obs/context."""
    global _SPINE
    if _SPINE is None:
        return
    move = _infer_move_from_obs(obs) or "SELECT"
    opp_hp = _opponent_active_hp(obs)
    own_hp = _own_active_hp(obs)
    own_prize = _own_prize_count(obs)
    opp_prize = _opp_prize_count(obs)
    own_active = _own_active_card_id(obs)
    can_ko = _can_ko_this_turn(obs)
    action_dict = {
        "move": move,
        "active": str(own_active),
        "opp_hp": opp_hp,
        "own_hp": own_hp,
        "own_prize": own_prize,
        "opp_prize": opp_prize,
        "ko": bool(can_ko),
    }
    _SPINE.push_event(
        hemisphere="cortex",
        action_dict=action_dict,
        wake_reason=None,
    )


def _maybe_emit_receipt(obs) -> None:
    """Emit a self-receipt every _EMIT_INTERVAL events on the spine."""
    global _SPINE, _SMM, _LAST_RECEIPT_OPP_PRIZE
    if _SPINE is None or _SMM is None:
        return
    if _SPINE.event_count == 0:
        return
    if _SPINE.event_count % _EMIT_INTERVAL == 0:
        _SMM.emit_self_receipt()
        # Snapshot the opp prize at receipt time so the next decision can
        # detect "prize drew since this receipt".
        _LAST_RECEIPT_OPP_PRIZE = _opp_prize_count(obs)


# ---------------------------------------------------------------------------
# Self-receipt influenced scoring (v15's distinguishing feature)
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


def _self_receipt_bonus(obs) -> float:
    """v15's extra term: read recent self-receipts and contribute a
    bounded BONUS to the score. ZERO contribution by default so the
    v8_psychic baseline is preserved as the floor.

    Rules (per task spec):
      - If the most recent self-receipt predicted PRIZE_DRAW (i.e. it
        followed an ATTACK with ko=True, meaning the substrate's prior
        receipt said 'expect KO next turn') AND the current state shows
        we can KO opp active this turn: +50.
      - If the most recent self-receipt predicted PRIZE_DRAW AND opp's
        prizeCount has changed since the receipt was emitted: +30.
      - Otherwise: 0.0.
    """
    global _SMM, _LAST_RECEIPT_OPP_PRIZE
    if _SMM is None:
        return 0.0
    recent = _SMM.read_recent_self_receipts(k=3)
    if not recent:
        return 0.0
    latest = recent[-1]
    predicted = latest.get("predicted_next")
    if predicted != "PRIZE_DRAW":
        return 0.0

    bonus = 0.0
    # Boost when the prior self-narrative said "expect KO next turn" and
    # the current options actually include a KO swing.
    if _can_ko_this_turn(obs):
        bonus += 50.0

    # Alignment bonus when the predicted prize draw actually materialized
    # (opp prize counter moved since the receipt was written).
    cur_opp_prize = _opp_prize_count(obs)
    if (_LAST_RECEIPT_OPP_PRIZE is not None
            and cur_opp_prize is not None
            and cur_opp_prize != _LAST_RECEIPT_OPP_PRIZE):
        bonus += 30.0
    return bonus


def _score_state(obs) -> float:
    """v15 eval. Identical to v8 except for the EXTRA SMM term, which
    only contributes non-zero when self-receipt expectations align with
    current state. Otherwise the v8 score is the floor.
    """
    try:
        s = obs.current
        if s is None:
            return 0.0
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
        own_energy_efficiency = 0.0
        if me.active and me.active[0] is not None:
            attached = len(getattr(me.active[0], "energies", None) or [])
            own_energy_efficiency = min(1.0, attached / max(1, 3))
        score = ((own_prize - opp_prize) * 250
                 + (own_active_hp - opp_active_hp)
                 + (own_bench_hp - opp_bench_hp) * 0.3
                 + (own_hand - opp_hand) * 5
                 + own_energy_efficiency * 8)
        # v15 EXTRA TERM — bounded, never degrades baseline.
        score += _self_receipt_bonus(obs)
        return float(score)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Priority schema (unchanged from v8_psychic) — the floor policy
# ---------------------------------------------------------------------------


def _priority_schema_decision(obs) -> list[int]:
    options = obs.select.option
    endgame = _endgame_mode(obs)

    # ABILITIES first
    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    # EVOLVE second
    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    # ENDGAME OVERRIDE
    if endgame:
        atk = _ko_tuned_attack_idx(obs)
        if atk is not None:
            return [atk]

    # PLAY (setup new basic to bench)
    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    # ATTACH
    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
        s = obs.current
        if s is not None:
            me = s.players[s.yourIndex]
            active_cid = -1
            if me.active and me.active[0] is not None:
                active_cid = int(getattr(me.active[0], "cardId", -1))
            active_is_big = _is_big_attacker_card(active_cid)
            for idx in attach_idxs:
                opt = options[idx]
                ipa = getattr(opt, "inPlayArea", None)
                if ipa is not None and int(ipa) == int(AreaType.BENCH):
                    return [idx]
            for idx in attach_idxs:
                opt = options[idx]
                ipa = getattr(opt, "inPlayArea", None)
                if ipa is not None and int(ipa) == int(AreaType.ACTIVE):
                    return [idx]
            return [attach_idxs[0]]

    # RETREAT
    retreat_idxs = _find_all(options, OptionType.RETREAT)
    if retreat_idxs and obs.current is not None:
        s = obs.current
        me = s.players[s.yourIndex]
        own_hp = _own_active_hp(obs)
        own_max_hp = _own_active_max_hp(obs)
        opp_threat = _opp_active_is_threat(obs)
        bench_has_big = _bench_has_big_attacker_waiting(obs)
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
# Engine-search 2-ply lookahead (uses v15 _score_state with SMM bonus)
# ---------------------------------------------------------------------------


def _search_anchored_decision(obs, time_budget_ms: int = 180) -> list[int] | None:
    """Same 2-ply minimax as v8, but the leaf evaluator is v15's
    SMM-aware _score_state. The SMM bonus only fires when its
    expectation matches the current state, so the v8 baseline is
    preserved on every non-aligned step.
    """
    deadline = time.monotonic() + (time_budget_ms / 1000.0)

    if obs.select is None:
        return None
    n_opts = len(obs.select.option)
    if n_opts < 2 or n_opts > 12:
        return None
    if obs.current is None:
        return None

    try:
        ctx = int(obs.select.context)
        if ctx != int(SelectContext.MAIN):
            return None
    except Exception:
        return None

    priority_choice = _priority_schema_decision(obs)
    if not priority_choice:
        return None

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


# ---------------------------------------------------------------------------
# Public entry point — closes the loop
# ---------------------------------------------------------------------------


def agent(obs_dict):
    obs = to_observation_class(obs_dict)

    # SETUP TURN: obs.select is None → return the deck.
    # Reset the per-game SMM so each game starts with a fresh self-narrative.
    if obs.select is None:
        _reset_smm()
        return _own_deck()

    # 1) Lazy-ensure the SMM exists (defensive — should always exist after
    #    setup turn, but games may interleave or be replayed in tests).
    _ensure_smm()

    # 2) PUSH the current obs context onto the spine as one event.
    try:
        _push_obs_event(obs)
    except Exception:
        # Spine push failures must never break the policy. Fall through
        # to v8_psychic floor behavior.
        pass

    # 3) Every _EMIT_INTERVAL events, emit one self-receipt.
    try:
        _maybe_emit_receipt(obs)
    except Exception:
        pass

    # 4) Decide. _search_anchored_decision uses v15's _score_state which
    #    already includes the self-receipt bonus term, so the substrate's
    #    higher-order self-representation flows directly into the choice.
    try:
        choice = _search_anchored_decision(obs, time_budget_ms=180)
        if choice is not None:
            return choice
    except Exception:
        pass

    # Fallback: priority schema (floor).
    return _priority_schema_decision(obs)
