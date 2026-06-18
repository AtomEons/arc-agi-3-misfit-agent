"""MCTS-PUCT planner — (human/agent)^2 leverage activation.

This is the lookahead engine that converts the world-model into search depth
WITHOUT spending real-environment actions. ARC-AGI-3 explicitly states that
"internal operations that do not alter the environment are NOT counted as
actions" — so every rollout here is free under the scoring rule.

Score math: per-level score = min(1.15, (human_baseline / agent_actions)^2).
A 2x reduction in real actions yields a 4x score multiplier. MCTS is the only
mechanism in the stack with the structural ability to *quadratically* lift
score, which is why this module exists.

------------------------------------------------------------------------------
ALGORITHM — PUCT (AlphaZero style)
------------------------------------------------------------------------------
At each node, child action `a` is selected by:

    UCB(a) = Q(a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s, a))

  - Q(a)   : mean simulated reward through child a (running average)
  - P(a|s) : prior over actions (resonance-seed warm-start)
             = 1.0 if a in last_known_progress_path[step_index]
             = 0.5 otherwise
             (normalized over expanded children at this node)
  - N(s)   : total visit count at this node
  - N(s,a) : visit count through child a
  - c_puct : exploration constant (CONFIG.mcts.c_puct, default 1.41)

------------------------------------------------------------------------------
REWARD MODEL
------------------------------------------------------------------------------
  +10.0   per predicted WIN in the simulated rollout
  -0.01   per simulated action (action-budget pressure inside the tree)
  +0.10   per *novel* predicted-grid fingerprint visited in this rollout

WIN prediction proxy: the world model does not directly emit WIN. We treat a
rollout as a predicted WIN iff the last action in the rollout has a historical
`level_advance_rate >= 0.5` AND the predicted grid materially differs from the
node's grid. Conservative on purpose — false WIN credit poisons Q.

Novelty: we hash the predicted-grid bytes; a fingerprint is "novel" the first
time it is visited within the current rollout.

------------------------------------------------------------------------------
CRITICAL CORRECTNESS — DEEP-COPY SAFETY (Lane A risk, architect-flagged)
------------------------------------------------------------------------------
`arcengine.GameAction` is a Python Enum. `GameAction.ACTION6` is a singleton.
If MCTS calls `GameAction.ACTION6.set_data({"x": 5, "y": 10})` on branch A
and `set_data({"x": 47, "y": 2})` on branch B, branch B *silently overwrites*
the click data branch A is still searching with. This produces correlated,
incoherent search across branches.

The fix here is structural: search never touches the enum. Every action in
the tree is wrapped in an immutable `ActionHandle(action_id, action_name,
is_complex, data)` where `data` is a deep-copied dict. Only at the end of
`plan()`, when returning the chosen ActionHandle to the caller, do we apply
`set_data` to the canonical enum member exactly once.

`test_mcts_puct.py` includes a deep-copy mutation-safety test that asserts
sibling branches in the tree do not see each other's click coordinates.

------------------------------------------------------------------------------
FRONTIER LIFT (Misfits lane, disclosed)
------------------------------------------------------------------------------
Progressive widening on ACTION6: instead of expanding every click candidate
(up to 20) at root, we expand ceil(N(s) ** alpha) candidates with alpha = 0.5.
Branching factor grows with visit count. This keeps early rollouts cheap and
focuses depth on the most-visited children before broadening. For simple
actions (ACTION1-5, ACTION7, RESET) we always expand all available — they
are <=7, branching is not the problem there.

Disclosed in TIER_1_DISCLOSURE.md scope: progressive widening is a generic
search heuristic from continuous-action MCTS literature (Couetoux 2011), not
a game-specific prior. Admissible as a (b) budget heuristic.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .config import CONFIG


# ---------------------------------------------------------------------------
# Action handle — the only object the search ever mutates.
# ---------------------------------------------------------------------------

@dataclass
class ActionHandle:
    """Immutable-ish wrapper around a candidate action for the MCTS tree.

    Each handle owns a deep-copied `data` dict so sibling tree branches
    cannot leak click coordinates into each other. The `enum_ref` is kept
    only so the caller can re-bind data onto the canonical enum at the
    very end of `plan()`.
    """
    action_id: int               # the GameAction enum value
    action_name: str             # "ACTION1", "ACTION6", "RESET", ...
    is_complex: bool             # True for ACTION6 (needs x, y data)
    data: dict                   # always a fresh dict, deep-copied
    enum_ref: Any = None         # the original enum member (do not mutate)

    def key(self) -> tuple[str, tuple]:
        """Hashable identity for tree-edge bookkeeping."""
        if self.is_complex:
            xy = (self.data.get("x", -1), self.data.get("y", -1))
        else:
            xy = ()
        return (self.action_name, xy)


def make_handle_from_enum(enum_action: Any, data: Optional[dict] = None) -> ActionHandle:
    """Build a deep-copy-safe handle from a GameAction enum member.

    `data` is deep-copied; if not provided and the action is complex,
    we default to {}.
    """
    name = getattr(enum_action, "name", str(enum_action))
    aid = int(getattr(enum_action, "value", 0))
    is_complex = bool(getattr(enum_action, "is_complex", lambda: False)())
    return ActionHandle(
        action_id=aid,
        action_name=name,
        is_complex=is_complex,
        data=copy.deepcopy(data) if data else {},
        enum_ref=enum_action,
    )


# ---------------------------------------------------------------------------
# Tree node.
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    """A node in the search tree. One node per (parent, action) edge."""
    grid_fingerprint: bytes      # hash of the grid this node represents
    depth: int
    parent: Optional["_Node"] = None
    incoming: Optional[ActionHandle] = None    # action that produced this node

    # Children, keyed by ActionHandle.key()
    children: dict[tuple, "_Node"] = field(default_factory=dict)
    child_handles: dict[tuple, ActionHandle] = field(default_factory=dict)
    child_priors: dict[tuple, float] = field(default_factory=dict)

    # Per-edge statistics — N(s,a), W(s,a), Q(s,a)
    n_sa: dict[tuple, int] = field(default_factory=dict)
    w_sa: dict[tuple, float] = field(default_factory=dict)

    # Node statistics
    n_s: int = 0                 # total visit count at this node
    is_terminal: bool = False    # set when WIN proxy fires or depth cap hit
    is_expanded: bool = False    # children populated?

    def q(self, edge_key: tuple) -> float:
        n = self.n_sa.get(edge_key, 0)
        if n == 0:
            return 0.0
        return self.w_sa.get(edge_key, 0.0) / n


# ---------------------------------------------------------------------------
# Planner.
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """Output of `plan()`."""
    chosen: ActionHandle
    root_stats: dict             # {edge_key_str: {"N":..,"Q":..,"P":..,"name":..}}
    rollouts_run: int
    wallclock_ms: float
    timed_out: bool


class MCTSPUCT:
    """PUCT planner with progressive widening and deep-copy-safe expansion.

    Construction is cheap — the costly work is in `plan(...)`.

    Constructor parameters:
      world_model_predict:
          callable(grid: np.ndarray, action_name: str) -> (next_grid, conf).
          Use `WorldModel.predict` in production. Tests pass synthetic stubs.

      click_candidates_fn:
          callable(scene_like) -> list of objects with .x, .y, .source attrs.
          Use `click_quantizer.click_candidates` in production. Tests stub it.

      last_known_progress_path:
          list[str] of action names from a winning resonance seed at this
          step_index, used to bias the prior. May be empty.

      historical_advance_rate:
          callable(action_name: str) -> float in [0, 1]. The episode-level
          empirical rate at which this action advanced a level. Drives the
          WIN proxy. In production, derived from EpisodeTracker; in tests,
          stubbed to a dict-lookup.

      rng:
          numpy.random.Generator for reproducible tie-breaking.

    Configuration (all from `CONFIG.mcts`):
      c_puct, max_depth, rollouts_per_action, hard_timeout_ms.
    """

    PROGRESSIVE_WIDENING_ALPHA = 0.5
    PROGRESSIVE_WIDENING_MIN = 3      # always expand at least this many click candidates

    # Reward constants (frozen per the spec)
    REWARD_WIN = 10.0
    REWARD_PER_ACTION = -0.01
    REWARD_NOVEL_FINGERPRINT = 0.10
    WIN_PROXY_ADVANCE_RATE_THRESHOLD = 0.5

    def __init__(
        self,
        world_model_predict: Callable[[np.ndarray, str], tuple[np.ndarray, float]],
        click_candidates_fn: Callable[[Any], Sequence[Any]],
        last_known_progress_path: Sequence[str] = (),
        historical_advance_rate: Optional[Callable[[str], float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.world_model_predict = world_model_predict
        self.click_candidates_fn = click_candidates_fn
        self.progress_path = list(last_known_progress_path)
        self.advance_rate = historical_advance_rate or (lambda _name: 0.0)
        self.rng = rng or np.random.default_rng(0xA70AEE05)

        self.c_puct = CONFIG.mcts.c_puct
        self.max_depth = CONFIG.mcts.max_depth
        self.rollouts_target = CONFIG.mcts.rollouts_per_action
        self.timeout_ms = CONFIG.mcts.hard_timeout_ms

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def plan(
        self,
        scene: Any,                      # SceneObservation-like, has .grid
        available_actions: Sequence[Any] # iterable of GameAction enum members
    ) -> PlanResult:
        """Run MCTS-PUCT and return the chosen action handle plus stats.

        The chosen handle is safe to mutate — its `data` dict is its own
        deep copy. Caller may apply `chosen.enum_ref.set_data(chosen.data)`
        exactly once to attach the click coordinates to the canonical enum
        member just before submitting to the engine.
        """
        t0 = time.perf_counter()
        deadline_s = t0 + (self.timeout_ms / 1000.0)

        root_grid = np.asarray(scene.grid).copy()
        root = _Node(
            grid_fingerprint=self._fp(root_grid),
            depth=0,
        )
        self._expand(root, root_grid, scene, available_actions)

        # Edge case: no actions to expand. Return a synthesized RESET handle
        # if it's in the available set, else first available, else None.
        if not root.children:
            return self._empty_plan(available_actions, t0)

        rollouts = 0
        timed_out = False
        for _ in range(self.rollouts_target):
            if time.perf_counter() >= deadline_s:
                timed_out = True
                break
            self._simulate(root, root_grid, scene, available_actions)
            rollouts += 1

        # Pick the most-visited child at root (standard AlphaZero choice).
        best_key = self._best_root_edge(root)
        chosen = root.child_handles[best_key]

        # Defensively re-deep-copy on the way out so the caller cannot
        # accidentally back-mutate any internal tree state.
        chosen_out = ActionHandle(
            action_id=chosen.action_id,
            action_name=chosen.action_name,
            is_complex=chosen.is_complex,
            data=copy.deepcopy(chosen.data),
            enum_ref=chosen.enum_ref,
        )

        wallclock_ms = (time.perf_counter() - t0) * 1000.0
        return PlanResult(
            chosen=chosen_out,
            root_stats=self._root_stats(root),
            rollouts_run=rollouts,
            wallclock_ms=wallclock_ms,
            timed_out=timed_out,
        )

    # ------------------------------------------------------------------
    # Tree expansion.
    # ------------------------------------------------------------------

    def _expand(
        self,
        node: _Node,
        grid: np.ndarray,
        scene: Any,
        available_actions: Sequence[Any],
    ) -> None:
        """Populate this node's children + priors. Idempotent."""
        if node.is_expanded:
            return
        node.is_expanded = True

        handles = self._enumerate_handles(node, scene, available_actions)
        if not handles:
            return

        # Compute unnormalized priors per handle, then normalize.
        raw: list[tuple[ActionHandle, float]] = []
        for h in handles:
            p = 1.0 if h.action_name in self.progress_path else 0.5
            raw.append((h, p))
        total = sum(p for _, p in raw) or 1.0

        for h, p in raw:
            k = h.key()
            if k in node.children:
                continue  # progressive widening may try to re-add; skip
            node.child_handles[k] = h
            node.child_priors[k] = p / total
            # Placeholder child node — instantiated on first descent.
            node.children[k] = _Node(
                grid_fingerprint=b"",  # filled on first descent
                depth=node.depth + 1,
                parent=node,
                incoming=h,
            )
            node.n_sa[k] = 0
            node.w_sa[k] = 0.0

    def _enumerate_handles(
        self,
        node: _Node,
        scene: Any,
        available_actions: Sequence[Any],
    ) -> list[ActionHandle]:
        """Build the candidate ActionHandle list for `node`, applying
        progressive widening to ACTION6 click candidates.

        Each handle gets a freshly deep-copied `data` dict — this is the
        Lane-A safety guarantee.
        """
        handles: list[ActionHandle] = []

        # How many click candidates to expand for this node?
        # Progressive widening: ceil(N(s) ** alpha), floored at the min.
        widen_k = max(
            self.PROGRESSIVE_WIDENING_MIN,
            int(math.ceil((node.n_s + 1) ** self.PROGRESSIVE_WIDENING_ALPHA)),
        )

        for ea in available_actions:
            is_complex = bool(getattr(ea, "is_complex", lambda: False)())
            if not is_complex:
                handles.append(make_handle_from_enum(ea, data={}))
                continue

            # Complex action → ClickQuantizer candidates, widened.
            cands = list(self.click_candidates_fn(scene))[:widen_k]
            if not cands:
                # No quantized candidates — fall back to grid centre.
                rows = int(getattr(scene, "rows", 0)) or int(scene.grid.shape[0])
                cols = int(getattr(scene, "cols", 0)) or int(scene.grid.shape[1])
                handles.append(make_handle_from_enum(
                    ea, data={"x": cols // 2, "y": rows // 2}
                ))
                continue
            for c in cands:
                handles.append(make_handle_from_enum(
                    ea, data={"x": int(c.x), "y": int(c.y)}
                ))
        return handles

    # ------------------------------------------------------------------
    # Simulation.
    # ------------------------------------------------------------------

    def _simulate(
        self,
        root: _Node,
        root_grid: np.ndarray,
        root_scene: Any,
        available_actions: Sequence[Any],
    ) -> None:
        """Run a single PUCT rollout from the root to a leaf/terminal."""
        node = root
        grid = root_grid.copy()
        scene = root_scene
        path: list[tuple[_Node, tuple]] = []       # (parent_node, edge_key)
        visited_fps: set[bytes] = {self._fp(grid)}
        cumulative_reward = 0.0

        for _depth in range(self.max_depth):
            if not node.children:
                break
            edge_key = self._select_edge(node)
            handle = node.child_handles[edge_key]
            child = node.children[edge_key]
            path.append((node, edge_key))

            # Forward sim — uses world_model.predict; THIS DOES NOT COUNT
            # against the real-environment action budget.
            try:
                next_grid, conf = self.world_model_predict(grid, handle.action_name)
            except Exception:
                conf = 0.0
                next_grid = grid

            # Reward shaping per the spec.
            cumulative_reward += self.REWARD_PER_ACTION
            fp = self._fp(next_grid)
            if fp not in visited_fps:
                cumulative_reward += self.REWARD_NOVEL_FINGERPRINT
                visited_fps.add(fp)

            # WIN proxy: high historical advance-rate AND material grid change.
            grid_changed = not np.array_equal(next_grid, grid)
            if (self.advance_rate(handle.action_name) >= self.WIN_PROXY_ADVANCE_RATE_THRESHOLD
                    and grid_changed
                    and conf >= 0.5):
                cumulative_reward += self.REWARD_WIN
                child.is_terminal = True

            # Advance into child.
            grid = next_grid
            if not child.grid_fingerprint:
                child.grid_fingerprint = fp
            node = child

            if node.is_terminal:
                break

            # Expand if the child has not been visited before.
            if not node.is_expanded:
                # Build a lightweight scene-like view over the simulated grid
                # so click_candidates_fn can still operate. We do NOT re-run
                # the perceptor here — that'd be a fat dependency for the
                # planner. Instead reuse root_scene; ClickQuantizer's job is
                # to give plausible candidates given some scene; using the
                # root scene's objects is a conservative approximation
                # consistent with "no Tier-2 contamination" — we never invent
                # game-specific positions, only reuse the prior frame's
                # objectness derived from the real observation.
                self._expand(node, grid, root_scene, available_actions)
                # Leaf rollout: one-step "value bootstrap" — already folded
                # into cumulative_reward above via the per-action terms.
                break

        # Backpropagate the cumulative reward up the visited path.
        self._backprop(root, path, cumulative_reward)

    def _select_edge(self, node: _Node) -> tuple:
        """PUCT edge selection at `node`. Returns the child edge key."""
        sqrt_ns = math.sqrt(max(node.n_s, 1))
        best_key = None
        best_score = -math.inf
        # Iterate in a deterministic order (sorted by key) for reproducibility.
        for k in sorted(node.child_handles.keys()):
            n_sa = node.n_sa.get(k, 0)
            q = node.q(k)
            p = node.child_priors.get(k, 0.0)
            u = self.c_puct * p * sqrt_ns / (1 + n_sa)
            score = q + u
            # Tiny noise on ties for deterministic-but-unbiased breaking.
            if score > best_score:
                best_score = score
                best_key = k
        if best_key is None:
            best_key = next(iter(node.child_handles))
        return best_key

    def _backprop(
        self,
        root: _Node,
        path: list[tuple[_Node, tuple]],
        reward: float,
    ) -> None:
        # Root visit first.
        root.n_s += 1
        for parent, edge_key in path:
            parent.n_sa[edge_key] = parent.n_sa.get(edge_key, 0) + 1
            parent.w_sa[edge_key] = parent.w_sa.get(edge_key, 0.0) + reward
            child = parent.children[edge_key]
            child.n_s += 1

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _fp(grid: np.ndarray) -> bytes:
        """Cheap content fingerprint for novelty tracking."""
        a = np.ascontiguousarray(grid).astype(np.int32, copy=False)
        return a.tobytes()

    def _best_root_edge(self, root: _Node) -> tuple:
        """Most-visited child at root, tie-broken by Q then by sorted key."""
        items = list(root.child_handles.keys())
        if not items:
            raise RuntimeError("MCTS root has no children")
        items.sort()  # deterministic key order
        best = items[0]
        best_n = root.n_sa.get(best, 0)
        best_q = root.q(best)
        for k in items[1:]:
            n = root.n_sa.get(k, 0)
            q = root.q(k)
            if (n, q) > (best_n, best_q):
                best, best_n, best_q = k, n, q
        return best

    def _root_stats(self, root: _Node) -> dict:
        out: dict = {}
        for k, h in root.child_handles.items():
            out[self._stat_key(k)] = {
                "name": h.action_name,
                "data": dict(h.data),
                "N": root.n_sa.get(k, 0),
                "Q": root.q(k),
                "P": root.child_priors.get(k, 0.0),
            }
        out["__node__"] = {"N_total": root.n_s, "n_children": len(root.children)}
        return out

    @staticmethod
    def _stat_key(edge_key: tuple) -> str:
        name, xy = edge_key
        if xy:
            return f"{name}@{xy[0]},{xy[1]}"
        return name

    def _empty_plan(
        self,
        available_actions: Sequence[Any],
        t0: float,
    ) -> PlanResult:
        """Degenerate case — return a safe handle without crashing."""
        if available_actions:
            fallback = make_handle_from_enum(available_actions[0], data={})
        else:
            fallback = ActionHandle(
                action_id=0,
                action_name="RESET",
                is_complex=False,
                data={},
                enum_ref=None,
            )
        return PlanResult(
            chosen=fallback,
            root_stats={"__node__": {"N_total": 0, "n_children": 0}},
            rollouts_run=0,
            wallclock_ms=(time.perf_counter() - t0) * 1000.0,
            timed_out=False,
        )
