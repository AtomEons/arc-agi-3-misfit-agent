"""AtomEons Misfit-TCG agent — v12 MCTS-PUCT.

v12 changes over v8:

  - **MCTS-PUCT at MAIN context.** Best-first search guided by v8's
    priority schema as policy prior. PUCT formula
        a* = argmax_a Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
    with c_puct = 1.25 (PTCG branching is small — lean exploit).

  - **Value-bootstrap leaf eval.** No random rollouts. At a leaf we
    take ONE priority-schema action, step the engine once, and score
    the resulting state with v8's `_score_state`. Sign-flips when
    yourIndex no longer matches the agent at root.

  - **Time budget = 250 ms.** Hard deadline via time.monotonic().
    Early stop if best N >= 2x second-best after >= 20 iterations.

  - **Engine search integration.** Each iteration opens its own
    SearchState via `search_begin(...)`, replays the selection path,
    expands the leaf, scores, then `search_end()`. One begin/end pair
    per iteration guarantees hygiene.

  - **Triple-layer fallback to v8 priority schema:**
      1. Any MCTS exception -> schema.
      2. Non-MAIN context at root -> schema.
      3. Option count outside [2, 12] -> schema.
      4. Game-over (obs.select is None) -> own deck.

Tier-1 strict: deterministic enumeration + deterministic engine
search + deterministic priority policy. No LLM, no learned
parameters, no pretrained weights.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from cg.api import (
    Observation, Option, SelectContext, OptionType, AreaType,
    to_observation_class, search_begin, search_step, search_end,
)


# ---------------------------------------------------------------------------
# Deck + attack/card data caches (shared with v8)
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
_BIG_ATTACKER_CACHE: dict[int, bool] = {}
def _is_big_attacker_card(card_id: int) -> bool:
    if card_id in _BIG_ATTACKER_CACHE:
        return _BIG_ATTACKER_CACHE[card_id]
    try:
        from cg.api import all_card_data
        for c in all_card_data():
            if int(getattr(c, "cardId", -1)) != card_id:
                continue
            atks = getattr(c, "attacks", None) or []
            for a in atks:
                dmg = int(getattr(a, "damage", 0) or 0)
                if dmg >= _BIG_ATTACKER_THRESHOLD:
                    _BIG_ATTACKER_CACHE[card_id] = True
                    return True
            _BIG_ATTACKER_CACHE[card_id] = False
            return False
    except Exception:
        _BIG_ATTACKER_CACHE[card_id] = False
        return False
    _BIG_ATTACKER_CACHE[card_id] = False
    return False


# ---------------------------------------------------------------------------
# Inspectors (shared with v8)
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


# ---------------------------------------------------------------------------
# v8 priority schema (re-used as policy prior + fallback)
# ---------------------------------------------------------------------------

def _priority_schema_decision(obs) -> list[int]:
    options = obs.select.option
    endgame = _endgame_mode(obs)

    ability_idxs = _find_all(options, OptionType.ABILITY)
    if ability_idxs:
        return [ability_idxs[0]]

    evolve_idxs = _find_all(options, OptionType.EVOLVE)
    if evolve_idxs:
        return [evolve_idxs[0]]

    if endgame:
        atk = _ko_tuned_attack_idx(obs)
        if atk is not None:
            return [atk]

    play_idxs = _find_all(options, OptionType.PLAY)
    if play_idxs:
        return [play_idxs[0]]

    attach_idxs = _find_all(options, OptionType.ATTACH)
    if attach_idxs:
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

    atk = _ko_tuned_attack_idx(obs)
    if atk is not None:
        return [atk]

    end_idx = _find_first(options, OptionType.END)
    if end_idx is not None:
        return [end_idx]
    return [0]


# ---------------------------------------------------------------------------
# Evaluation function (v8's _score_state)
# ---------------------------------------------------------------------------

def _score_state(obs) -> float:
    """v8 evaluation. Always evaluated from `obs.current.yourIndex`'s perspective.
    Caller is responsible for sign-flipping when yourIndex != root agent's index.
    """
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


def _score_from_root_perspective(obs, root_agent_idx: int) -> float:
    """Return _score_state as seen by root_agent_idx (always our perspective).
    If current observation's yourIndex == root_agent_idx, score is already ours.
    Otherwise sign-flip (because v8 score is symmetric: own - opp).
    """
    try:
        if obs.current is None:
            return 0.0
        raw = _score_state(obs)
        if int(obs.current.yourIndex) == int(root_agent_idx):
            return raw
        return -raw
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# MCTS-PUCT
# ---------------------------------------------------------------------------

C_PUCT = 1.25
TIME_BUDGET_MS = 250
MIN_OPTIONS = 2
MAX_OPTIONS = 12
ROLLOUT_TARGET = 50
EARLY_STOP_MIN_ITERS = 20


@dataclass
class _Node:
    """One MCTS node per (parent, action) edge. Children keyed by option index."""
    parent: Optional["_Node"] = None
    incoming: Optional[int] = None       # option index that produced this node
    children: dict = field(default_factory=dict)  # option_idx -> _Node
    child_priors: dict = field(default_factory=dict)   # option_idx -> P(a|s)
    n_sa: dict = field(default_factory=dict)           # option_idx -> visits
    w_sa: dict = field(default_factory=dict)           # option_idx -> cumulative value
    n_s: int = 0                          # total visits at this node
    is_expanded: bool = False
    is_terminal: bool = False
    cached_score: float = 0.0


def _compute_priors(obs) -> dict[int, float]:
    """Priority schema's top pick gets prior 1.0; siblings 0.5; normalized."""
    options = obs.select.option
    n = len(options)
    if n == 0:
        return {}
    try:
        top = _priority_schema_decision(obs)
        top_idx = int(top[0]) if top else 0
    except Exception:
        top_idx = 0
    raw: dict[int, float] = {}
    for i in range(n):
        raw[i] = 1.0 if i == top_idx else 0.5
    total = sum(raw.values()) or 1.0
    return {i: w / total for i, w in raw.items()}


def _puct_select(node: _Node) -> int:
    """Return option index with highest PUCT score among node.children's priors keys."""
    sqrt_ns = math.sqrt(max(1, node.n_s))
    best_idx = -1
    best_score = float("-inf")
    # iterate sorted to keep tie-break deterministic (lowest option index)
    for a in sorted(node.child_priors.keys()):
        p = node.child_priors.get(a, 0.0)
        n_sa = node.n_sa.get(a, 0)
        w_sa = node.w_sa.get(a, 0.0)
        q = (w_sa / n_sa) if n_sa > 0 else 0.0
        u = C_PUCT * p * sqrt_ns / (1.0 + n_sa)
        score = q + u
        if score > best_score:
            best_score = score
            best_idx = a
    return best_idx


def _backpropagate(path: list[tuple[_Node, int]], value: float) -> None:
    """Walk path in reverse; increment edge visits, accumulate edge value, bump parent n_s."""
    for (parent, edge_idx) in path:
        parent.n_s += 1
        parent.n_sa[edge_idx] = parent.n_sa.get(edge_idx, 0) + 1
        parent.w_sa[edge_idx] = parent.w_sa.get(edge_idx, 0.0) + value


def _build_search_args(obs):
    """Construct the search_begin keyword arguments from an observation.
    Returns dict or None if hand-count is missing (can't search).
    """
    state = obs.current
    if state is None:
        return None
    me = state.players[state.yourIndex]
    opp = state.players[1 - state.yourIndex]
    deck = _own_deck()
    own_prize = list(range(min(int(getattr(me, "prizeCount", 0)), 6)))
    opp_prize = list(range(min(int(getattr(opp, "prizeCount", 0)), 6)))
    opp_hand_count = _safe_hand_count(opp)
    if opp_hand_count is None:
        return None
    opp_hand = [0] * opp_hand_count
    opp_active: list[int] = []
    if opp.active and opp.active[0] is not None:
        opp_active = [int(getattr(opp.active[0], "cardId", 0))]
    return dict(your_deck=deck, your_prize=own_prize,
                opponent_deck=deck, opponent_prize=opp_prize,
                opponent_hand=opp_hand, opponent_active=opp_active)


def _expand_and_evaluate(ss_id: int, leaf_obs, root_agent_idx: int) -> tuple[dict[int, float], float, bool]:
    """At a leaf node: compute priors, then take ONE priority-schema action to
    bootstrap value via _score_state.
    Returns (priors_dict, value_from_root_perspective, terminal_flag).
    """
    # Terminal: game over already
    if leaf_obs.select is None or leaf_obs.current is None:
        return ({}, _score_from_root_perspective(leaf_obs, root_agent_idx), True)
    # Compute priors over option set
    priors = _compute_priors(leaf_obs)
    # Bootstrap value: take schema action and step once
    try:
        sched = _priority_schema_decision(leaf_obs)
        if not sched:
            return (priors, _score_from_root_perspective(leaf_obs, root_agent_idx), False)
        next_raw = search_step(ss_id, sched)
        next_obs = next_raw.observation
        val = _score_from_root_perspective(next_obs, root_agent_idx)
        return (priors, val, False)
    except Exception:
        # Fall back to scoring leaf without bootstrap step
        return (priors, _score_from_root_perspective(leaf_obs, root_agent_idx), False)


def _mcts_decision(obs, time_budget_ms: int = TIME_BUDGET_MS) -> list[int] | None:
    """Run MCTS-PUCT and return [chosen_option_idx] from root. None if not applicable."""
    if obs.select is None or obs.current is None:
        return None
    n_opts = len(obs.select.option)
    if n_opts < MIN_OPTIONS or n_opts > MAX_OPTIONS:
        return None
    try:
        ctx = int(obs.select.context)
        if ctx != int(SelectContext.MAIN):
            return None
    except Exception:
        return None

    sa = _build_search_args(obs)
    if sa is None:
        return None

    root_agent_idx = int(obs.current.yourIndex)

    # Build root node with priors over root option set
    root = _Node()
    root.child_priors = _compute_priors(obs)
    if not root.child_priors:
        return None
    root.is_expanded = True

    deadline = time.monotonic() + (time_budget_ms / 1000.0)
    iters = 0
    while iters < ROLLOUT_TARGET and time.monotonic() < deadline:
        iters += 1

        # ---- Selection phase: walk from root via PUCT until unexpanded ----
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            if not node.child_priors:
                # No options - terminal at this node
                break
            edge = _puct_select(node)
            if edge < 0:
                break
            path.append((node, edge))
            child = node.children.get(edge)
            if child is None:
                # Will expand below
                break
            if not child.is_expanded or child.is_terminal:
                node = child
                break
            node = child

        if not path:
            break  # nothing to explore (degenerate)

        # ---- Engine playout: replay path of edges from root ----
        ss = None
        try:
            ss = search_begin(obs, **sa)
            ss_id = ss.searchId
            cur_raw = None
            for (parent, edge_idx) in path:
                cur_raw = search_step(ss_id, [edge_idx])
            if cur_raw is None:
                # No edges - shouldn't happen given path non-empty check
                search_end()
                continue

            leaf_obs = cur_raw.observation

            # ---- Expansion + bootstrap value ----
            last_parent, last_edge = path[-1]
            child = last_parent.children.get(last_edge)
            if child is None:
                child = _Node(parent=last_parent, incoming=last_edge)
                last_parent.children[last_edge] = child
            priors, value, terminal = _expand_and_evaluate(ss_id, leaf_obs, root_agent_idx)
            child.child_priors = priors
            child.is_expanded = True
            child.is_terminal = terminal or not priors
            child.cached_score = value

            # ---- Backprop ----
            _backpropagate(path, value)
        except Exception:
            # Skip this iteration on engine error
            pass
        finally:
            try:
                if ss is not None:
                    search_end()
            except Exception:
                pass

        # Early stop: if best edge dominates after enough iters
        if iters >= EARLY_STOP_MIN_ITERS and root.n_sa:
            counts = sorted(root.n_sa.values(), reverse=True)
            if len(counts) >= 2 and counts[0] >= 2 * counts[1]:
                break
            if len(counts) == 1 and root.n_s >= EARLY_STOP_MIN_ITERS:
                break

    if not root.n_sa:
        return None  # never got any rollouts in - caller falls back

    # Final pick: highest visits; tie-break highest Q; then lowest index
    def _key(a):
        n = root.n_sa.get(a, 0)
        w = root.w_sa.get(a, 0.0)
        q = (w / n) if n > 0 else float("-inf")
        return (n, q, -a)
    best = max(root.n_sa.keys(), key=_key)
    return [int(best)]


# ---------------------------------------------------------------------------
# Agent entrypoint
# ---------------------------------------------------------------------------

def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _own_deck()
    # Triple-layer guard:
    #   (a) MCTS exception   -> schema (try/except below)
    #   (b) non-MAIN context -> schema (handled inside _mcts_decision)
    #   (c) option count out of [2, 12] -> schema (handled inside _mcts_decision)
    try:
        choice = _mcts_decision(obs, time_budget_ms=TIME_BUDGET_MS)
        if choice is not None:
            return choice
    except Exception:
        pass
    return _priority_schema_decision(obs)
