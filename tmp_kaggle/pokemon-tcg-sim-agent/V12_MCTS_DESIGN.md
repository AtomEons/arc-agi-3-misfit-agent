# PTCG Agent v12 — MCTS-PUCT Design Document

**Status:** DESIGNED (not implemented). Successor to v8's 2-ply minimax.
Reference: `C:\AtomEons\arc-agi-3-misfit-agent\src\misfit_agent\mcts_puct.py`.
Tier-1 strict — deterministic, no learned weights, no LLM.

## 1. Motivation

v8 explores top-3 × top-3 = 9 leaves at fixed depth 2 in ~180 ms. MCTS-PUCT
reallocates the same budget to a best-first search that goes deep on
promising lines and prunes mediocre ones. Engine search is free of
real-environment cost; we burn it on lookahead.

## 2. Node Structure

One `_Node` per (parent, action) edge:

```
@dataclass
class _Node:
    state_key:     bytes              # hash of obs.current for de-dup
    depth:         int
    parent:        Optional[_Node]
    incoming:      Optional[int]      # option index that produced this node
    children:      dict[int, _Node]   # keyed by option index
    child_priors:  dict[int, float]   # P(a|s) from priority schema
    n_sa:          dict[int, int]     # visit count per edge
    w_sa:          dict[int, float]   # cumulative reward per edge
    n_s:           int = 0            # total visits at this node
    is_terminal:   bool = False
    is_expanded:   bool = False
    cached_score:  float = 0.0        # _score_state at expansion
```

Edge key is the integer option index. No enum-mutation hazard like ARC's
`GameAction.set_data`, so no `ActionHandle` wrapper needed.

## 3. PUCT Formula

```
UCB(a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
```

- `c_puct = 1.25` (lower than substrate's 1.41; PTCG branching is small —
  typically 2-8 options — so lean exploit).
- `Q(s,a) = W(s,a) / N(s,a)`, defaults to 0 for unvisited edges.
- `P(a|s)`: priority schema's top choice gets prior 1.0; siblings 0.5;
  normalize over the option set.
- Tie-break: highest N, then highest Q, then lowest option index.

## 4. Four-Phase Loop

**Selection.** From root, descend by argmax UCB until reaching an unexpanded
node or terminal. Track `[(node, edge_idx), ...]`.

**Expansion.** At the leaf, call `to_observation_class(raw_obs)` and
enumerate `obs.select.option`. Compute priors via priority schema (reuse
`_priority_schema_decision` as the policy oracle). Mark `is_expanded`.

**Rollout.** Random rollouts are too expensive (~3-5 ms per `search_step`).
Take one priority-schema action from the leaf and evaluate the result with
v8's `_score_state` — a cheap value estimate respecting the policy prior.
Depth cap = 4 ply.

**Backpropagation.** Walk path in reverse, increment `n_sa`, add value to
`w_sa`, increment `n_s`. Sign-flip on opponent plies so Q is always
from-our-perspective.

## 5. Engine Search Integration

```
ss = search_begin(obs, your_deck=deck, your_prize=own_prize,
                  opponent_deck=deck, opponent_prize=opp_prize,
                  opponent_hand=opp_hand, opponent_active=opp_active)
try:
    cur_raw = None
    for (parent, edge_idx) in selection_path:
        cur_raw = search_step(ss, [edge_idx])
    leaf_obs = to_observation_class(cur_raw)
    # expand + bootstrap value + backprop on leaf_obs
finally:
    search_end(ss)
```

Each rollout opens its own `SearchState` and replays the path from root.
One begin/end pair per rollout guarantees hygiene at the cost of redundant
replay (acceptable at depth 4).

## 6. Time Budget

- Hard deadline: **250 ms** per decision (`time.monotonic()` per rollout).
- Per-rollout target: ~5 ms (1 begin + ~4 steps + score).
- Expected throughput: **~50 rollouts** within budget.
- Soft yield: stop early if best `N >= 2x` second-best after >= 20 rollouts.

## 7. Termination Criteria

A node is terminal when any of:
- `obs.select is None` (game over).
- Either player's `prizeCount == 0`.
- `depth >= 4` (depth cap; backpropagate leaf score).
- `obs.select.context != MAIN` and option count `== 1` (forced move).

## 8. Integration with v8 Priority Schema

The schema is not replaced — it is repurposed:
1. **Prior generator** for PUCT (`P(a|s)`).
2. **Default policy** for value-bootstrap rollouts.
3. **Hard fallback** if MCTS fails (begin raises, option count > 12 / < 2).
4. **Cold-start oracle** at non-MAIN contexts: skip MCTS, defer to schema.

Agent entrypoint:

```
def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _own_deck()
    try:
        choice = _mcts_decision(obs, time_budget_ms=250)
        if choice is not None:
            return choice
    except Exception:
        pass
    return _priority_schema_decision(obs)  # v8 fallback
```

## 9. Expected Wins vs v8

- **Depth.** v8 caps at 2 ply; v12 reaches 4 ply on promising lines.
- **Adaptive breadth.** v8 fixed top-3; v12 widens with visit count.
- **Q-anchored.** v8 ranks by single leaf score; v12 by Monte-Carlo average.

## 10. Non-Goals

- No progressive widening (PTCG options are pre-enumerated).
- No learned value head (Tier-1 strict).
- No opponent-modeling beyond self-policy.
