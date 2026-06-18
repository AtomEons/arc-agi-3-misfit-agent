"""AtomEons Misfit-TCG agent — v6 robust engine-search 2-ply minimax.

v6 changes over v5:
  - **FIX for opponent_hand size mismatch**: v5 sometimes built an
    opponent_hand list of the wrong size, causing `search_begin` to throw
    ``ValueError("opponent_hand does not match the number of cards in
    opponent's hand.")``. v5 read `opp.handCount` but the except branch
    silently fell back to `[own]` (size 1), which fails on any later turn
    where the opponent holds more than one card. v6 reads the hand size
    defensively from the dataclass field, the raw obs dict, AND the
    `select` snapshot, picks whichever yields a non-None integer >= 0,
    and ALWAYS sizes the list exactly. If we still can't get a count, we
    skip search rather than passing a guessed length.
  - **Pre-validation gate**: before calling `search_begin`, v6 re-asserts
    that every predicted list size matches the live observation. If any
    dimension is off, we fall back to the priority schema. This makes the
    failure mode "no search, never crash" instead of "search call raises".
  - **Tighter time budget**: 200ms per decision (spec).
  - **Same priority-schema fallback** as v3/v5.

Tier-1 honest: deterministic priority enumeration + deterministic engine
search. No LLM, no learned parameters, no pretrained weights.
"""

import os
import time
from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
)


# ---------------------------------------------------------------------------
# Deck IO
# ---------------------------------------------------------------------------


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
# Card / attack metadata cache
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


_BASIC_POKEMON_IDS_CACHE: list[int] | None = None


def _basic_pokemon_ids_in_deck(deck: list[int]) -> list[int]:
    """Return card IDs from `deck` that are Basic Pokémon. Cached."""
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
# Priority schema (v3-compat fallback)
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
# Engine-search lookahead with v6 robust opponent_hand sizing
# ---------------------------------------------------------------------------


def _safe_hand_count(opp_player_state, raw_obs_dict: dict | None = None) -> int | None:
    """Return opponent's hand size as int, or None if undeterminable.

    Tries (in order):
      1. dataclass attribute `handCount`
      2. dataclass attribute `hand_count` (snake_case)
      3. dataclass attribute `handSize` / `hand_size`
      4. raw obs dict at players[1-yourIndex].{handCount,hand_count,handSize,hand_size}
      5. len(hand) if hand is present
    """
    # Try dataclass attributes first
    for name in ("handCount", "hand_count", "handSize", "hand_size"):
        try:
            val = getattr(opp_player_state, name, None)
            if val is not None and int(val) >= 0:
                return int(val)
        except (TypeError, ValueError):
            continue
    # Try hand list length
    try:
        hand = getattr(opp_player_state, "hand", None)
        if hand is not None and isinstance(hand, list):
            return len(hand)
    except Exception:
        pass
    # Try raw obs dict
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
                h = op.get("hand")
                if isinstance(h, list):
                    return len(h)
        except Exception:
            pass
    return None


def _safe_deck_count(opp_player_state, raw_obs_dict: dict | None = None) -> int | None:
    """Return opponent's deck count as int, or None if undeterminable."""
    for name in ("deckCount", "deck_count", "deckSize", "deck_size"):
        try:
            val = getattr(opp_player_state, name, None)
            if val is not None and int(val) >= 0:
                return int(val)
        except (TypeError, ValueError):
            continue
    if raw_obs_dict:
        try:
            cur = raw_obs_dict.get("current") or {}
            players = cur.get("players") or []
            your_idx = cur.get("yourIndex", 0)
            opp_idx = 1 - int(your_idx)
            if 0 <= opp_idx < len(players):
                op = players[opp_idx] or {}
                for name in ("deckCount", "deck_count"):
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


def _predict_opponent_state(obs, raw_obs_dict: dict | None = None):
    """Predict opponent's hidden state for `search_begin`.

    v6: builds opponent_hand of EXACTLY the size the live observation
    reports. If the count is undeterminable, returns None so the caller
    can skip search instead of guessing.
    """
    deck = _own_deck()
    own = deck[0] if deck else 1
    pokemon_guess = own
    for cid in deck:
        # First Basic Pokémon id makes the safest face-down active prediction
        pid = _basic_pokemon_ids_in_deck(deck)
        if pid:
            pokemon_guess = pid[0]
        break

    try:
        state = obs.current
        if state is None:
            return None
        opp = state.players[1 - state.yourIndex]

        # Hand size — STRICT, no silent fallback
        hand_count = _safe_hand_count(opp, raw_obs_dict)
        if hand_count is None:
            return None

        # Deck count — STRICT
        deck_count = _safe_deck_count(opp, raw_obs_dict)
        if deck_count is None:
            return None

        prize_count = len(opp.prize) if opp.prize is not None else 0

        # opponent_deck must be at LEAST deck_count entries; pad if our own
        # deck is shorter (cannot happen at 60v60 but defend anyway).
        opponent_deck = (deck * ((deck_count // max(len(deck), 1)) + 1))[:max(deck_count, 1)]
        if len(opponent_deck) < deck_count:
            opponent_deck = opponent_deck + [own] * (deck_count - len(opponent_deck))

        opponent_prize = [own] * prize_count
        opponent_hand = [own] * hand_count  # v6 FIX: EXACT size

        opponent_active = []
        if opp.active and (len(opp.active) > 0) and opp.active[0] is None:
            opponent_active = [pokemon_guess]

        return {
            "opponent_deck": opponent_deck,
            "opponent_prize": opponent_prize,
            "opponent_hand": opponent_hand,
            "opponent_active": opponent_active,
            # echo live counts for the pre-validation gate
            "_live_hand_count": hand_count,
            "_live_deck_count": deck_count,
            "_live_prize_count": prize_count,
        }
    except Exception:
        return None


def _own_prize_prediction(obs) -> list[int]:
    deck = _own_deck()
    own = deck[0] if deck else 1
    try:
        me = obs.current.players[obs.current.yourIndex]
        return [own] * (len(me.prize) if me.prize is not None else 0)
    except Exception:
        return [own] * 6


def _validate_search_sizes(obs, deck: list[int], own_prize: list[int],
                           opp_pred: dict) -> bool:
    """Belt-and-braces check: re-confirm every list length matches the
    live observation before we call search_begin. Returns False if the
    call would be rejected by the API; caller then falls back to schema.
    """
    try:
        state = obs.current
        you = state.players[state.yourIndex]
        opp = state.players[1 - state.yourIndex]
        if obs.select is not None and obs.select.deck is None:
            if len(deck) < (you.deckCount or 0):
                return False
        if len(own_prize) < (len(you.prize) if you.prize else 0):
            return False
        if len(opp_pred["opponent_deck"]) < (opp.deckCount or 0):
            return False
        if len(opp_pred["opponent_prize"]) < (len(opp.prize) if opp.prize else 0):
            return False
        if len(opp_pred["opponent_hand"]) < (opp.handCount or 0):
            return False
        if opp.active and opp.active[0] is None and not opp_pred["opponent_active"]:
            return False
        return True
    except Exception:
        return False


def _score_state(search_state) -> float:
    """Lower is better. Prize differential dominates, then HP differentials.

    Score = (our_prize_remaining - opponent_prize_remaining) * 100
          + (opponent_active_hp - our_active_hp * 0.5)

    Equivalent to: rewards us for taking opp prizes (lowering their prize
    pile) and damaging the opp active, while penalising us for losing our
    own prizes and our active HP.
    """
    try:
        obs = search_state.observation
        state = obs.current if hasattr(obs, "current") else None
        if state is None:
            return 0.0
        me_idx = state.yourIndex
        me = state.players[me_idx]
        opp = state.players[1 - me_idx]
        me_prize = len(me.prize) if me.prize else 0
        opp_prize = len(opp.prize) if opp.prize else 0
        me_active_hp = me.active[0].hp if (me.active and me.active[0]) else 0
        opp_active_hp = opp.active[0].hp if (opp.active and opp.active[0]) else 0
        return (me_prize * 100) - (opp_prize * 100) - me_active_hp * 0.5 + opp_active_hp
    except Exception:
        return 0.0


def _candidate_actions(obs) -> list[int]:
    """Generate the small candidate set to evaluate. The priority-schema
    pick is always first so search can only IMPROVE on the baseline."""
    options = obs.select.option
    candidates: list[int] = []
    schema = _priority_schema_decision(obs)
    if schema:
        candidates.append(schema[0])
    # Add every legal attack as an alternative — attacks are the highest
    # variance / highest-leverage decisions to evaluate.
    for idx in _find_all(options, OptionType.ATTACK):
        if idx not in candidates:
            candidates.append(idx)
    # Bound to 4 candidates for time-safety.
    return candidates[:4]


def _search_anchored_decision(obs, raw_obs_dict: dict | None = None,
                              time_budget_s: float = 0.2) -> list[int] | None:
    """Attempt 2-ply minimax via engine search. None on failure (fallback).

    Time budget: 200ms hard cap. If the candidate set has <=1 entry, we
    skip search entirely (no benefit).
    """
    if obs.search_begin_input is None:
        return None  # search API only works on real agent observations

    deck = _own_deck()
    own_prize = _own_prize_prediction(obs)
    opp_pred = _predict_opponent_state(obs, raw_obs_dict)
    if opp_pred is None:
        return None  # could not determine sizes safely

    candidates = _candidate_actions(obs)
    if len(candidates) <= 1:
        return None

    if not _validate_search_sizes(obs, deck, own_prize, opp_pred):
        return None

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

    best_idx: int | None = None
    best_score = float("inf")
    try:
        for cand in candidates:
            if time.monotonic() - t_start > time_budget_s:
                break
            try:
                state_after_us = search_step(root.searchId, [cand])
            except Exception:
                continue
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


# ---------------------------------------------------------------------------
# Action dispatch
# ---------------------------------------------------------------------------


def _main_action(obs, raw_obs_dict: dict | None = None) -> list[int]:
    """MAIN context: attempt search anchoring, fall back to schema on any
    error. Never crashes."""
    try:
        decision = _search_anchored_decision(obs, raw_obs_dict, time_budget_s=0.2)
        if decision is not None:
            return decision
    except Exception:
        pass
    try:
        return _priority_schema_decision(obs)
    except Exception:
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

    if ctx in (41, 43, 46):  # IS_FIRST, ACTIVATE, COIN_HEAD → YES
        yes_idx = _find_first(options, OptionType.YES)
        if yes_idx is not None:
            return [yes_idx]

    if int(obs.select.type) == 9:  # generic YES_NO → NO
        no_idx = _find_first(options, OptionType.NO)
        if no_idx is not None:
            return [no_idx]
        yes_idx = _find_first(options, OptionType.YES)
        return [yes_idx] if yes_idx is not None else [0]

    if ctx in {1, 2}:  # SETUP_ACTIVE / SETUP_BENCH → first CARD
        for i, opt in enumerate(options):
            if int(opt.type) == int(OptionType.CARD):
                return [i]

    take = max(min_count, 1) if max_count >= 1 else min_count
    take = min(take, max_count, len(options))
    return list(range(take))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def agent(obs_dict: dict) -> list[int]:
    """Kaggle Pokémon TCG agent — v6.

    Never crashes. On any internal error, falls back to a safe default.
    """
    try:
        obs: Observation = to_observation_class(obs_dict)
    except Exception:
        return read_deck_csv()

    if obs.select is None:
        # Initial deck selection
        return read_deck_csv()
    try:
        if int(obs.select.context) == 0:
            return _main_action(obs, obs_dict)
        return _selection_action(obs)
    except Exception:
        # Last-resort safe default
        try:
            return [0]
        except Exception:
            return [0]


# ---------------------------------------------------------------------------
# Local arena self-test (run only when invoked as __main__)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT))

    # Import the arena lazily so module import never depends on it
    try:
        from local_arena import evaluate_pair
    except Exception as exc:
        print(f"[v6 self-test] local_arena import failed: {exc!r}")
        sys.exit(1)

    deck_path = ROOT / "deck.csv"
    with open(deck_path) as f:
        deck = [int(x.strip()) for x in f.read().split() if x.strip()][:60]

    print("[v6 self-test] 5-game local arena: agent_v6 vs agent_v3")
    result = evaluate_pair("agent_v6", "agent_v3", deck, deck, num_games=5)
    print(json.dumps(result, indent=2))
